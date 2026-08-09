import json

import pytest

from exonym.freeze import freeze
from exonym.gatekeeper import GateError, advance, gate_errors, next_phase, set_lifecycle_state
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


def _novelty_audit_payload(candidate_id="candidate-alpha", status="eligible", expires_at=None):
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "retrieved_at": "2000-01-01T00:00:00Z",
        "freshness": {"expires_at": expires_at or "2099-01-01T00:00:00Z"},
        "status": status,
        "decision_basis": "A documented novelty assessment supports this workflow decision.",
        "evidence": [
            {
                "source_uri": "https://example.invalid/novelty-evidence",
                "retrieved_at": "2000-01-01T00:00:00Z",
                "finding": "The source was assessed under the recorded novelty protocol.",
                "evidence_sha256": "a" * 64,
            }
        ],
    }


def _write_novelty_audit(candidate, **overrides):
    expires_at = overrides.pop("expires_at", None)
    payload = _novelty_audit_payload(candidate.candidate_id, expires_at=expires_at)
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
    return tmp_path


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
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
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
    assert gate_errors(candidate), "gate must fail without provenance sidecar"

    (raw / "lc.provenance.json").write_text(
        json.dumps(
            {
                "source_uri": "https://archive.stsci.edu/example",
                "download_timestamp_utc": "2026-08-04T00:00:00Z",
                "sha256": "a" * 64,
                "fetched_by": "test",
            }
        ),
        encoding="utf-8",
    )
    assert not gate_errors(candidate)


def test_analysis_gate_requires_fpp_claim(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    claims = candidate.path / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    claims.joinpath("fpp.json").write_text(
        json.dumps(
            {
                "parameter": "fpp",
                "value": 0.003,
                "uncertainty_upper": 0.001,
                "uncertainty_lower": 0.001,
                "unit": "dimensionless",
                "method": "triceratops",
            }
        ),
        encoding="utf-8",
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
                "sha256": "a" * 64,
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
    assert not gate_errors(candidate)


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
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    _checked_doc(candidate.path / "docs" / "01_intake_manifest.md")
    advance(candidate)
    _checked_doc(candidate.path / "docs" / "02_feasibility_report.md")
    _write_novelty_audit(candidate)
    advance(candidate)
    raw = candidate.path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "lc.fits").write_bytes(b"fits")
    (raw / "lc.provenance.json").write_text(
        json.dumps(
            {
                "source_uri": "https://archive.stsci.edu/example",
                "download_timestamp_utc": "2026-08-04T00:00:00Z",
                "sha256": "a" * 64,
                "fetched_by": "test",
            }
        ),
        encoding="utf-8",
    )
    advance(candidate)
    _checked_doc(candidate.path / "docs" / "03_spoc_dv_vetting.md")
    advance(candidate)
    _checked_doc(candidate.path / "docs" / "04_tfop_sg_followup.md")
    advance(candidate)
    claims = candidate.path / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    claims.joinpath("fpp.json").write_text(
        json.dumps(
            {
                "parameter": "fpp",
                "value": 0.003,
                "uncertainty_upper": 0.001,
                "uncertainty_lower": 0.001,
                "unit": "dimensionless",
                "method": "triceratops",
            }
        ),
        encoding="utf-8",
    )
    advance(candidate)
    reloaded = _reload(tmp_path)
    assert reloaded.metadata["workflow"]["phase"] == "review"
    return reloaded


def test_feasibility_gate_rejects_nonconforming_or_ineligible_novelty_audit(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    _checked_doc(candidate.path / "docs" / "01_intake_manifest.md")
    advance(candidate)
    candidate = _reload(tmp_path)
    _checked_doc(candidate.path / "docs" / "02_feasibility_report.md")

    _write_novelty_audit(candidate, evidence=[])
    assert any("violates schema" in error for error in gate_errors(candidate))

    _write_novelty_audit(candidate, status="ineligible")
    assert any("status is not eligible" in error for error in gate_errors(candidate))


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

    gate_record = json.loads(
        (candidate.path / "gates" / "gate-006-review.json").read_text(encoding="utf-8")
    )
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
    candidate = [c for c in discover_candidates(tmp_path)][0]
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text("numpy==1.26.4\nscipy==1.13.1\n", encoding="utf-8")

    release_dir = freeze(candidate, version="v1.0.0")
    assert release_dir.is_dir()
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "v1.0.0"
    assert manifest["candidate_id"] == "candidate-alpha"
    assert {entry["path"] for entry in manifest["files"]} == {
        "requirements.lock.txt",
        "environment.lock.yml",
        "Dockerfile",
        "Apptainer.def",
    }
    assert all(entry["sha256"] for entry in manifest["files"])

    with pytest.raises(FileExistsError):
        freeze(candidate, version="v1.0.0")


def test_overall_progress_across_documents(tmp_path):
    candidate = create_candidate(_templated_repo(tmp_path), "candidate-alpha")
    telemetry = candidate_telemetry(candidate)
    checked, total, fraction = overall_progress(telemetry.values())
    assert checked == 0 and total == 5
    assert fraction == 0.0
