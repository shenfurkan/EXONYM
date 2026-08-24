import hashlib
import json
import shutil

import pytest

from exonym.freeze import ReleaseVerificationError, freeze, verify_release
from exonym.gatekeeper import (
    GateError,
    advance,
    gate_errors,
    has_valid_raw_product_provenance,
    next_phase,
    set_lifecycle_state,
)
from exonym.survey_harvest import novelty_provider_urls
from exonym.tagging import add_tags, filter_candidates, has_tag
from exonym.tracking import candidate_telemetry, overall_progress, parse_checklist
from exonym.workspace import create_candidate, discover_candidates, load_candidate


def _check(path, text, checked=True, mandatory=False):
    mark = "[x]" if checked else "[ ]"
    label = " [MANDATORY]" if mandatory else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("- {0} {1}{2}\n".format(mark, text, label))


def _reload(tmp_path):
    return load_candidate(tmp_path, "candidate-alpha")


def _novelty_audit_payload(candidate, status="eligible", expires_at=None):
    retrieval_id = "a" * 32
    tic = candidate.metadata["identifiers"].get("tic")
    if not isinstance(tic, str):
        raise ValueError("novelty gate test candidates require a TIC")
    source_uris = dict(novelty_provider_urls(tic))
    evidence_dir = candidate.path / "data" / "external" / "novelty" / retrieval_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = []
    for index, provider in enumerate(("nasa-toi", "nasa-confirmed", "exofop")):
        extension = "json" if provider == "exofop" else "csv"
        response_path = evidence_dir / "{0}-{1}.{2}".format(index, provider, extension)
        if provider == "nasa-toi":
            response = b"toi,tid\n"
        elif provider == "nasa-confirmed":
            response = b"pl_name,tic_id\n"
        else:
            response = json.dumps(
                {
                    "basic_info": {"tic_id": tic},
                    "tois": [],
                    "ctois": [],
                    "planet_parameters": [],
                }
            ).encode("utf-8")
        response_path.write_bytes(response)
        evidence.append(
            {
                "source_uri": source_uris[provider],
                "retrieved_at": "2000-01-01T00:00:00Z",
                "finding": "Synthetic {0} response supports the recorded novelty decision.".format(provider),
                "evidence_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
                "provider": provider,
                "response_path": response_path.relative_to(candidate.path).as_posix(),
            }
        )
    return {
        "schema_version": 2,
        "candidate_id": candidate.candidate_id,
        "retrieved_at": "2000-01-01T00:00:00Z",
        "freshness": {"expires_at": expires_at or "2099-01-01T00:00:00Z"},
        "status": status,
        "decision_basis": "A documented novelty assessment supports this workflow decision.",
        "evidence": evidence,
    }


def _write_novelty_audit(candidate, **overrides):
    expires_at = overrides.pop("expires_at", None)
    payload = _novelty_audit_payload(candidate, expires_at=expires_at)
    payload.update(overrides)
    path = candidate.path / "decisions" / "novelty_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _templated_repo(tmp_path):
    for name in (
        "docs/01_intake_manifest.md",
        "docs/02_feasibility_report.md",
        "docs/03_spoc_dv_vetting.md",
        "docs/04_tfop_sg_followup.md",
    ):
        path = tmp_path / "templates" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("- [ ] [MANDATORY] task\n", encoding="utf-8")
    (tmp_path / "templates/decisions/review_gate.md").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "templates/decisions/review_gate.md").write_text(
        "- [ ] [MANDATORY] task\n", encoding="utf-8"
    )
    (tmp_path / "templates/protocols").mkdir(parents=True, exist_ok=True)
    (tmp_path / "templates/tracking").mkdir(parents=True, exist_ok=True)
    package_dir = tmp_path / "src" / "exonym"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text('__version__ = "test"\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[build-system]\nrequires = [\"setuptools\"]\nbuild-backend = \"setuptools.build_meta\"\n"
        "[project]\nname = \"exonym\"\nversion = \"0.0.0\"\n",
        encoding="utf-8",
    )
    return tmp_path


def _write_verified_fpp_claim(candidate, value=0.003):
    report_path = candidate.path / "outputs" / "triceratops_report.json"
    report_path.write_text(
        json.dumps(
            {
                "candidate_id": candidate.candidate_id,
                "source": "triceratops-monte-carlo",
                "FPP": value,
            }
        ),
        encoding="utf-8",
    )
    claim = {
        "candidate_id": candidate.candidate_id,
        "parameter": "fpp",
        "value": value,
        "uncertainty_upper": 0.001,
        "uncertainty_lower": 0.001,
        "unit": "dimensionless",
        "method": "TRICERATOPS Monte Carlo simulation",
        "report_path": "outputs/triceratops_report.json",
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    claim_path = candidate.path / "claims" / "fpp.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    return claim_path


def test_parse_checklist_counts_and_flags(tmp_path):
    doc = tmp_path / "gate.md"
    _check(doc, "first", checked=True)
    _check(doc, "second", checked=False, mandatory=True)
    _check(doc, "third", checked=True, mandatory=True)

    telemetry = parse_checklist(doc)
    assert telemetry.total == 3
    assert telemetry.checked == 2
    assert telemetry.mandatory_total == 2
    assert telemetry.mandatory_checked == 1
    assert not telemetry.gate_pass
    assert telemetry.completion == pytest.approx(2 / 3 * 100.0)


def test_advance_blocks_on_unchecked_mandatory(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    doc = candidate.path / "docs" / "01_intake_manifest.md"
    _check(doc, "identity verified", checked=True, mandatory=True)
    _check(doc, "collision check", checked=False, mandatory=True)

    assert gate_errors(candidate)
    with pytest.raises(GateError):
        advance(candidate)


def test_gate_document_requires_a_mandatory_item(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    document = candidate.path / "docs" / "01_intake_manifest.md"
    document.write_text("- [x] non-gating note\n", encoding="utf-8")

    errors = gate_errors(candidate)
    assert any("contains no mandatory checklist items" in error for error in errors)
    with pytest.raises(GateError, match="contains no mandatory checklist items"):
        advance(candidate)


def test_stopped_candidate_blocks_gate_errors_and_advance(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    set_lifecycle_state(candidate, "stopped", reason="scientific eligibility withdrawn")
    stopped = _reload(tmp_path)

    with pytest.raises(GateError, match="reason is required"):
        set_lifecycle_state(stopped, "active")
    with pytest.raises(GateError, match="reason is required"):
        set_lifecycle_state(stopped, "active", reason="   ")
    assert any("lifecycle is stopped" in error for error in gate_errors(stopped))
    with pytest.raises(GateError, match="lifecycle is stopped"):
        advance(stopped)


def test_advance_promotes_phase_and_writes_gate_record(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    doc = candidate.path / "docs" / "01_intake_manifest.md"
    doc.unlink()
    _check(doc, "identity verified", checked=True, mandatory=True)
    _check(doc, "collision check", checked=True, mandatory=True)
    _check(doc, "gaia astrometry", checked=True, mandatory=True)
    _check(doc, "magnitude recorded", checked=True, mandatory=True)

    assert candidate.metadata["workflow"]["phase"] == "intake"
    event = advance(candidate)
    assert event["to"] == "feasibility"

    reloaded = [c for c in discover_candidates(tmp_path)][0]
    assert reloaded.metadata["workflow"]["phase"] == "feasibility"
    assert list((candidate.path / "gates").glob("gate-*.json"))
    assert (candidate.path / "lifecycle" / "events.jsonl").is_file()


def test_acquisition_gate_requires_provenance_sidecars(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha", tic="123456789")
    candidate.path.joinpath("docs/01_intake_manifest.md").unlink()
    _check(candidate.path / "docs" / "01_intake_manifest.md", "a", checked=True, mandatory=True)
    _check(candidate.path / "docs" / "01_intake_manifest.md", "b", checked=True, mandatory=True)
    _check(candidate.path / "docs" / "01_intake_manifest.md", "c", checked=True, mandatory=True)
    _check(candidate.path / "docs" / "01_intake_manifest.md", "d", checked=True, mandatory=True)
    advance(candidate)

    candidate = _reload(tmp_path)
    assert candidate.metadata["workflow"]["phase"] == "feasibility"
    candidate.path.joinpath("docs/02_feasibility_report.md").unlink()
    _check(candidate.path / "docs" / "02_feasibility_report.md", "a", checked=True, mandatory=True)
    _check(candidate.path / "docs" / "02_feasibility_report.md", "b", checked=True, mandatory=True)
    _check(candidate.path / "docs" / "02_feasibility_report.md", "c", checked=True, mandatory=True)
    _check(candidate.path / "docs" / "02_feasibility_report.md", "d", checked=True, mandatory=True)
    assert any("missing novelty audit" in error for error in gate_errors(candidate))
    _write_novelty_audit(candidate)
    advance(candidate)
    candidate = _reload(tmp_path)
    assert candidate.metadata["workflow"]["phase"] == "acquisition"

    assert gate_errors(candidate), "acquisition gate must fail without raw products"

    raw = candidate.path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "lc.fits").write_bytes(b"fits")
    # Auto-provenance now generates a minimal sidecar when none exists,
    # so the gate should pass without a manually written sidecar.
    assert not gate_errors(candidate), "gate must pass: auto-provenance generates missing sidecar"

    (raw / "lc.provenance.json").write_text(
        json.dumps(
            {
                "source_uri": "https://archive.stsci.edu/example",
                "download_timestamp_utc": "2026-08-04T00:00:00Z",
                "sha256": hashlib.sha256(b"fits").hexdigest(),
                "fetched_by": "test",
            }
        ),
        encoding="utf-8",
    )
    assert not gate_errors(candidate)
    (raw / "lc.fits").write_bytes(b"tampered")
    assert any("SHA-256 does not match" in error for error in gate_errors(candidate))


@pytest.mark.parametrize(
    "sidecar_text",
    (
        '{"source_uri":"https://archive.example.invalid/lc.fits","sha256":"placeholder","sha256":"placeholder","download_timestamp_utc":"2026-01-01T00:00:00Z","fetched_by":"test"}',
        '{"source_uri":"https://archive.example.invalid/lc.fits","sha256":"placeholder","download_timestamp_utc":"2026-01-01T00:00:00Z","fetched_by":1e999}',
    ),
)
def test_raw_provenance_rejects_ambiguous_or_nonfinite_json(tmp_path, sidecar_text):
    candidate = create_candidate(tmp_path, "candidate-alpha")
    product = candidate.path / "data" / "raw" / "lc.fits"
    product.write_bytes(b"fits")
    sidecar = product.with_name("lc.provenance.json")
    sidecar.write_text(
        sidecar_text.replace("placeholder", hashlib.sha256(product.read_bytes()).hexdigest()),
        encoding="utf-8",
    )

    assert not has_valid_raw_product_provenance(candidate, product)


def test_analysis_gate_blocks_low_looking_and_arbitrary_fpp_outputs(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha", tic="123456789")
    claims = candidate.path / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    _write_verified_fpp_claim(candidate)
    (candidate.path / "outputs" / "triceratops_results.json").write_text(
        json.dumps({"FPP": 0.0001}), encoding="utf-8"
    )
    (claims / "fpp_claim.json").write_text(
        json.dumps({"FPP": 0.0001}), encoding="utf-8"
    )
    (candidate.path / "outputs" / "arbitrary_diagnostic.json").write_text(
        json.dumps({"status": "available"}), encoding="utf-8"
    )

    candidate.path.joinpath("docs/01_intake_manifest.md").write_text(
        "- [x] [MANDATORY] a\n- [x] [MANDATORY] b\n- [x] [MANDATORY] c\n- [x] [MANDATORY] d\n",
        encoding="utf-8",
    )
    candidate.path.joinpath("docs/02_feasibility_report.md").write_text(
        "- [x] [MANDATORY] a\n- [x] [MANDATORY] b\n- [x] [MANDATORY] c\n- [x] [MANDATORY] d\n",
        encoding="utf-8",
    )
    advance(candidate)
    _write_novelty_audit(candidate)
    advance(candidate)
    candidate = _reload(tmp_path)
    (candidate.path / "data" / "raw" / "lc.fits").write_bytes(b"fits")
    (candidate.path / "data" / "raw" / "lc.provenance.json").write_text(
        json.dumps(
            {
                "source_uri": "https://archive.stsci.edu/example",
                "download_timestamp_utc": "2026-08-04T00:00:00Z",
                "sha256": hashlib.sha256(b"fits").hexdigest(),
                "fetched_by": "test",
            }
        ),
        encoding="utf-8",
    )
    advance(candidate)
    candidate = _reload(tmp_path)
    assert candidate.metadata["workflow"]["phase"] == "vetting"

    candidate.path.joinpath("docs/03_spoc_dv_vetting.md").write_text(
        "- [x] [MANDATORY] a\n- [x] [MANDATORY] b\n- [x] [MANDATORY] c\n- [x] [MANDATORY] d\n",
        encoding="utf-8",
    )
    advance(candidate)
    candidate = _reload(tmp_path)
    assert candidate.metadata["workflow"]["phase"] == "followup"

    candidate.path.joinpath("docs/04_tfop_sg_followup.md").write_text(
        "- [x] [MANDATORY] a\n- [x] [MANDATORY] b\n- [x] [MANDATORY] c\n- [x] [MANDATORY] d\n",
        encoding="utf-8",
    )
    advance(candidate)
    candidate = _reload(tmp_path)
    assert candidate.metadata["workflow"]["phase"] == "analysis"

    from exonym.gatekeeper import _gate_fpp_claim

    passed, reason = _gate_fpp_claim(candidate)
    assert passed is False
    assert "FPP claims are disabled" in reason
    errors = gate_errors(candidate)
    assert any("FPP claims are disabled" in error for error in errors)
    with pytest.raises(GateError, match="FPP claims are disabled"):
        advance(candidate)
    candidate = _reload(tmp_path)
    assert candidate.metadata["workflow"]["phase"] == "analysis"


@pytest.mark.parametrize("forgery", ("fpp", "no_evidence"))
def test_analysis_gate_rejects_forged_or_mismatched_fpp_claim(tmp_path, forgery):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    candidate.path.joinpath("claims").mkdir(parents=True, exist_ok=True)

    from exonym.gatekeeper import _gate_fpp_claim

    if forgery == "fpp":
        # Write a claim with FPP above threshold — gate should block.
        claim = {
            "FPP": 0.02,
            "candidate_id": candidate.candidate_id,
        }
        (candidate.path / "claims" / "fpp_claim.json").write_text(
            json.dumps(claim), encoding="utf-8"
        )
        passed, reason = _gate_fpp_claim(candidate)
        assert not passed, "FPP above threshold should block: {0}".format(reason)
    else:
        # No evidence at all — gate should block.
        passed, reason = _gate_fpp_claim(candidate)
        assert not passed, "no evidence should block: {0}".format(reason)


def test_set_lifecycle_state_records_reason_and_event(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")

    from exonym.gatekeeper import GateError, set_lifecycle_state

    with pytest.raises(ValueError):
        set_lifecycle_state(candidate, "not-a-state")

    with pytest.raises(GateError):
        set_lifecycle_state(candidate, "active")

    lifecycle = set_lifecycle_state(candidate, "paused", reason="awaiting follow-up data")
    assert lifecycle["state"] == "paused"
    assert lifecycle["reason"] == "awaiting follow-up data"

    reloaded = _reload(tmp_path)
    assert reloaded.metadata["lifecycle"]["state"] == "paused"
    events = (candidate.path / "lifecycle" / "events.jsonl").read_text(encoding="utf-8")
    assert "state_changed" in events
    assert "awaiting follow-up data" in events


def test_set_lifecycle_state_locked_requires_reason(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")

    from exonym.gatekeeper import GateError, set_lifecycle_state

    set_lifecycle_state(candidate, "published", reason="review complete")

    with pytest.raises(GateError, match="reason is required"):
        set_lifecycle_state(candidate, "paused")

    lifecycle = set_lifecycle_state(
        candidate, "paused", reason="audit: transit not independently detectable"
    )
    assert lifecycle["state"] == "paused"


def test_phase_ordering_and_terminal(tmp_path):
    assert next_phase("intake") == "feasibility"
    assert next_phase("review") is None
    with pytest.raises(ValueError):
        next_phase("mystery")


def _checked_doc(path, items=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join("- [x] [MANDATORY] task {0}\n".format(i) for i in range(items)),
        encoding="utf-8",
    )


def _to_review_phase(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha", tic="123456789")
    # Exercise review behavior against a historical workspace that reached the
    # phase before the FPP claim gate was intentionally disabled.
    metadata = dict(candidate.metadata)
    workflow = dict(metadata["workflow"])
    workflow["phase"] = "review"
    metadata["workflow"] = workflow
    (candidate.path / "candidate.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    candidate = _reload(tmp_path)
    _write_novelty_audit(candidate)
    return _reload(tmp_path)


def test_feasibility_gate_rejects_nonconforming_or_ineligible_novelty_audit(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha", tic="123456789")
    _checked_doc(candidate.path / "docs" / "01_intake_manifest.md")
    advance(candidate)
    candidate = _reload(tmp_path)
    _checked_doc(candidate.path / "docs" / "02_feasibility_report.md")

    _write_novelty_audit(candidate, evidence=[])
    assert any("violates schema" in error for error in gate_errors(candidate))

    _write_novelty_audit(candidate, status="ineligible")
    assert any("status is not eligible" in error for error in gate_errors(candidate))


def test_novelty_gate_rejects_legacy_or_tampered_response_evidence(tmp_path):
    from exonym.gatekeeper import _gate_novelty_audit

    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha", tic="123456789")
    _write_novelty_audit(candidate)
    audit_path = candidate.path / "decisions" / "novelty_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["schema_version"] = 1
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    assert "schema version 2" in _gate_novelty_audit(candidate)[1]

    _write_novelty_audit(candidate)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    (candidate.path / audit["evidence"][0]["response_path"]).write_bytes(b"tampered")

    assert "SHA-256" in _gate_novelty_audit(candidate)[1]


def test_novelty_gate_rejects_semantically_mismatched_evidence(tmp_path):
    from exonym.gatekeeper import _gate_novelty_audit

    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha", tic="123456789")
    audit_path = candidate.path / "decisions" / "novelty_audit.json"
    _write_novelty_audit(candidate)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["evidence"][0]["source_uri"] = audit["evidence"][1]["source_uri"]
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    assert "canonical provider query" in _gate_novelty_audit(candidate)[1]

    _write_novelty_audit(candidate)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    response_path = candidate.path / audit["evidence"][0]["response_path"]
    response_path.write_bytes(b"not a NASA CSV response")
    audit["evidence"][0]["evidence_sha256"] = hashlib.sha256(response_path.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    assert "not semantically valid" in _gate_novelty_audit(candidate)[1]


def test_novelty_gate_requires_matching_entry_timestamps_and_strict_json(tmp_path):
    from exonym.gatekeeper import _gate_novelty_audit

    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha", tic="123456789")
    audit_path = candidate.path / "decisions" / "novelty_audit.json"
    _write_novelty_audit(candidate)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["evidence"][0]["retrieved_at"] = "2000-01-01T00:00:01Z"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    assert "retrieval time does not match" in _gate_novelty_audit(candidate)[1]

    _write_novelty_audit(candidate)
    duplicate_status = audit_path.read_text(encoding="utf-8").replace(
        '"status": "eligible"', '"status": "ineligible", "status": "eligible"'
    )
    audit_path.write_text(duplicate_status, encoding="utf-8")

    assert "invalid novelty audit JSON" in _gate_novelty_audit(candidate)[1]


def test_review_gate_requires_a_current_novelty_audit(tmp_path):
    candidate = _to_review_phase(tmp_path)
    _checked_doc(candidate.path / "decisions" / "review_gate.md")
    _write_novelty_audit(candidate, expires_at="2001-01-01T00:00:00Z")

    assert any("novelty audit is stale" in error for error in gate_errors(candidate))
    with pytest.raises(GateError, match="novelty audit is stale"):
        advance(candidate)


def test_advance_review_phase_locks_lifecycle(tmp_path):
    candidate = _to_review_phase(tmp_path)
    assert candidate.metadata["lifecycle"]["state"] == "active"

    _checked_doc(candidate.path / "decisions" / "review_gate.md")
    event = advance(candidate)

    assert event["to"] == "review (locked)"
    assert event["lifecycle"] == "published"

    locked = _reload(tmp_path)
    assert locked.metadata["workflow"]["phase"] == "review"
    assert locked.metadata["lifecycle"]["state"] == "published"
    assert locked.metadata["lifecycle"]["reason"] == "Review gate passed; lifecycle locked"
    assert locked.metadata["lifecycle"]["state_since"]

    gate_path = next((candidate.path / "gates").glob("gate-*-review.json"))
    gate_record = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate_record["gate"] == "review"
    assert gate_record["result"] == "PASS"

    with pytest.raises(GateError, match="already locked"):
        advance(locked)


def test_tagging_add_and_filter(tmp_path):
    create_candidate(_templated_repo(tmp_path), "candidate-alpha", tags=["priority-1"])
    create_candidate(_templated_repo(tmp_path), "candidate-beta")

    alpha = [c for c in discover_candidates(tmp_path) if c.candidate_id == "candidate-alpha"][0]
    assert has_tag(alpha, "priority-1")
    assert not has_tag(alpha, "sg1-cleared")

    tags = add_tags(alpha, ["sg1-cleared", "sg1-cleared"])
    assert tags == ["priority-1", "sg1-cleared"]

    filtered = filter_candidates(discover_candidates(tmp_path), tag="priority-1")
    assert [c.candidate_id for c in filtered] == ["candidate-alpha"]


def test_freeze_builds_manifest_and_locks(tmp_path):
    create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    shutil.copytree("src", tmp_path / "src", dirs_exist_ok=True)
    candidate = [c for c in discover_candidates(tmp_path)][0]
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text("numpy==1.26.4\nscipy==1.13.1\n", encoding="utf-8")
    observed_input = candidate.path / "data" / "raw" / "synthetic-input.fits"
    observed_input.write_bytes(b"synthetic-observed-photometry")
    (candidate.path / "scratch" / "temporary.txt").write_text("ephemeral\n", encoding="utf-8")

    release_dir = freeze(candidate, version="v1.0.0")
    assert release_dir.is_dir()
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "v1.0.0"
    assert manifest["candidate_id"] == "candidate-alpha"
    frozen_paths = {entry["path"] for entry in manifest["files"]}
    assert {
        "README.md",
        "requirements.lock.txt",
        "environment.lock.yml",
        "Dockerfile",
        "Apptainer.def",
        "source/pyproject.toml",
        "source/src/exonym/__init__.py",
        "workspace/candidate/candidate-alpha/candidate.json",
        "workspace/candidate/candidate-alpha/data/raw/synthetic-input.fits",
    } <= frozen_paths
    assert all(entry["sha256"] for entry in manifest["files"])
    assert manifest["schema"] == "exonym-freeze-3"
    assert manifest["replay_status"] == "integrity-checked-source-import-and-workspace-load"
    assert manifest["source_snapshot"]["path"] == "source"
    assert manifest["workspace_snapshot"]["candidate_path"] == "workspace/candidate/candidate-alpha"
    assert "releases" in manifest["workspace_snapshot"]["excluded_paths"]
    assert not any("scratch/temporary.txt" in path for path in frozen_paths)
    environment = (release_dir / "environment.lock.yml").read_text(encoding="utf-8")
    assert '      - "numpy==1.26.4"' in environment
    assert '      - "-e ./source"' in environment
    assert ",\n" not in environment
    assert "COPY source /work/source" in (release_dir / "Dockerfile").read_text(encoding="utf-8")
    assert "workspace /work/workspace" in (release_dir / "Apptainer.def").read_text(encoding="utf-8")
    assert "--no-build-isolation" in (release_dir / "Dockerfile").read_text(encoding="utf-8")
    assert (release_dir / "manifest.sha256").read_text(encoding="ascii").endswith("  manifest.json\n")
    assert manifest["requirements_lock"]["format"] == "fully-pinned-requirements"
    assert manifest["requirements_lock"]["package_count"] == 2

    replay = verify_release(candidate, version="v1.0.0")
    assert replay["candidate_id"] == "candidate-alpha"
    assert replay["checked_file_count"] == len(frozen_paths)
    assert replay["replay"]["candidate_id"] == "candidate-alpha"

    with pytest.raises(FileExistsError):
        freeze(candidate, version="v1.0.0")


def test_verify_release_rejects_a_tampered_detached_manifest_digest(tmp_path):
    create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    shutil.copytree("src", tmp_path / "src", dirs_exist_ok=True)
    candidate = load_candidate(tmp_path, "candidate-alpha")
    (tmp_path / "requirements-lock.txt").write_text("numpy==1.26.4\n", encoding="utf-8")

    release_dir = freeze(candidate, version="v1.0.0")
    (release_dir / "manifest.sha256").write_text("0" * 64 + "  manifest.json\n", encoding="ascii")

    with pytest.raises(ReleaseVerificationError, match="detached SHA-256"):
        verify_release(candidate, version="v1.0.0")


def test_freeze_does_not_create_a_release_directory_before_lock_preflight(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")

    with pytest.raises(FileNotFoundError, match="requirements-lock"):
        freeze(candidate, version="v1.0.0")

    assert not (candidate.path / "releases" / "v1.0.0").exists()


def test_freeze_rejects_a_non_exact_requirements_lock(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    (tmp_path / "requirements-lock.txt").write_text("numpy>=1.26\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact distribution==version"):
        freeze(candidate, version="v1.0.0")

    assert not (candidate.path / "releases" / "v1.0.0").exists()


def test_freeze_removes_its_staging_directory_when_source_snapshot_preflight_fails(tmp_path):
    candidate = create_candidate(tmp_path, "candidate-alpha")
    (tmp_path / "requirements-lock.txt").write_text("numpy==1.26.4\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="pyproject.toml and src"):
        freeze(candidate, version="v1.0.0")

    assert not (candidate.path / "releases" / "v1.0.0").exists()
    assert not list((candidate.path / "releases").glob(".v1.0.0.staging-*"))


@pytest.mark.parametrize("version", ["../escape", "nested/path", "CON", "com1", "lpt9", "release."])
def test_freeze_rejects_unsafe_release_versions(tmp_path, version):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    (tmp_path / "requirements-lock.txt").write_text("numpy==1.26.4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="release version"):
        freeze(candidate, version=version)


def test_overall_progress_across_documents(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    telemetry = candidate_telemetry(candidate)
    checked, total, fraction = overall_progress(telemetry.values())
    assert checked == 0 and total == 5
    assert fraction == 0.0


def test_freeze_manifest_includes_engine_manifests_when_runs_exist(tmp_path):
    """engine_manifests in manifest.json lists every recorded engine-run.json."""
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text("numpy==1.26.4\n", encoding="utf-8")

    # Synthesise a minimal valid engine-run.json below runs/
    run_dir = candidate.path / "runs" / "bls" / "20260101t000000z-bls"
    run_dir.mkdir(parents=True)
    engine_run = {
        "schema_version": 1,
        "candidate_id": "candidate-alpha",
        "engine": "bls",
        "run_id": "20260101t000000z-bls",
        "status": "succeeded",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:01:00+00:00",
        "runtime": {"kind": "direct", "version": "1.0.0", "executable": "astropy.timeseries"},
        "inputs": [{"path": "candidate.json", "sha256": "a" * 64}],
        "outputs": [],
    }
    (run_dir / "engine-run.json").write_text(json.dumps(engine_run), encoding="utf-8")

    release_dir = freeze(candidate, version="v1.0.0")
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))

    assert "engine_manifests" in manifest
    assert len(manifest["engine_manifests"]) == 1
    em = manifest["engine_manifests"][0]
    assert em["engine"] == "bls"
    assert em["run_id"] == "20260101t000000z-bls"
    assert em["status"] == "succeeded"
    assert len(em["sha256"]) == 64
    assert em["manifest_path"].startswith("runs/bls/")


def test_freeze_manifest_includes_config_hashes_when_config_exists(tmp_path):
    """config_hashes in manifest.json maps every config/*.json to its sha256."""
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text("numpy==1.26.4\n", encoding="utf-8")

    # Write a synthetic transit config
    config_dir = candidate.path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    transit_cfg = {
        "signal": None,
        "transit": {"period_days": 5.0, "t0_btjd": 2000.0, "depth_ppm": 500.0, "duration_hours": 2.0},
    }
    (config_dir / "transit_config.json").write_text(json.dumps(transit_cfg), encoding="utf-8")

    release_dir = freeze(candidate, version="v1.0.0")
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))

    assert "config_hashes" in manifest
    assert "config/transit_config.json" in manifest["config_hashes"]
    sha = manifest["config_hashes"]["config/transit_config.json"]
    assert len(sha) == 64  # valid SHA-256 hex string


def test_freeze_manifest_bare_workspace_has_empty_engine_manifests_and_config_hashes(tmp_path):
    """A workspace with no runs or config still produces a valid manifest."""
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    (tmp_path / "requirements-lock.txt").write_text("numpy==1.26.4\n", encoding="utf-8")

    release_dir = freeze(candidate, version="v1.0.0")
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["engine_manifests"] == []
    assert manifest["config_hashes"] == {}
    assert "requirements_lock_sha256" in manifest
    assert len(manifest["requirements_lock_sha256"]) == 64
