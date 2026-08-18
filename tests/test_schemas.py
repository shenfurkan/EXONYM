import hashlib
import json
import shutil

import pytest

from exonym.isolation import IsolationReport
from exonym.schemas import validate_schemas
from exonym.survey import _run_survey_robustness, create_survey, register_survey_target
from exonym.workspace import create_candidate, load_candidate


def _make_repo(tmp_path, with_templates=True):
    create_candidate(tmp_path, "candidate-alpha", toi="1234.01", tic="123456789")
    (tmp_path / "schemas").mkdir(parents=True, exist_ok=True)
    for name in (
        "candidate.schema.json",
        "provenance.schema.json",
        "claim.schema.json",
        "novelty-audit.schema.json",
        "survey.schema.json",
        "survey-target.schema.json",
        "survey-robustness.schema.json",
        "engine-run.schema.json",
        "automated-triage.schema.json",
    ):
        shutil.copy2(
            "schemas/{0}".format(name), tmp_path / "schemas" / name
        )
    return tmp_path


def _audit(tmp_path):
    report = IsolationReport()
    validate_schemas(tmp_path, report)
    return report


def test_clean_repository_passes_schema_validation(tmp_path):
    report = _audit(_make_repo(tmp_path))
    assert report.ok


def test_invalid_candidate_record_is_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    path = repo / "candidate" / "candidate-alpha" / "candidate.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["lifecycle"]["state"] = "mystery"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    report = _audit(repo)
    assert not report.ok
    assert any(v.rule == "schema-violation" for v in report.violations)


def test_invalid_provenance_sidecar_is_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    raw = repo / "candidate" / "candidate-alpha" / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "lc.fits").write_bytes(b"fits")
    (raw / "lc.provenance.json").write_text(
        json.dumps({"sha256": "not-a-hash"}), encoding="utf-8"
    )

    report = _audit(repo)
    assert not report.ok
    assert any(v.rule == "schema-violation" for v in report.violations)


def test_valid_provenance_sidecar_passes(tmp_path):
    repo = _make_repo(tmp_path)
    raw = repo / "candidate" / "candidate-alpha" / "data" / "raw"
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

    assert _audit(repo).ok


def test_invalid_claim_is_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    claims = repo / "candidate" / "candidate-alpha" / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    (claims / "bad.json").write_text(
        json.dumps({"parameter": "period_days"}), encoding="utf-8"
    )

    report = _audit(repo)
    assert not report.ok
    assert any(v.rule == "schema-violation" for v in report.violations)


def _valid_novelty_audit():
    return {
        "schema_version": 1,
        "candidate_id": "candidate-alpha",
        "retrieved_at": "2000-01-01T00:00:00Z",
        "freshness": {"expires_at": "2099-01-01T00:00:00Z"},
        "status": "eligible",
        "decision_basis": "The recorded evidence supports the documented eligibility decision.",
        "evidence": [
            {
                "source_uri": "https://example.invalid/novelty-evidence",
                "retrieved_at": "2000-01-01T00:00:00Z",
                "finding": "A source was reviewed using the recorded method.",
                "evidence_sha256": "a" * 64,
            }
        ],
    }


def _valid_engine_run():
    return {
        "schema_version": 1,
        "candidate_id": "candidate-alpha",
        "engine": "test-engine",
        "run_id": "run-001",
        "status": "succeeded",
        "started_at": "2000-01-01T00:00:00Z",
        "completed_at": "2000-01-01T00:01:00Z",
        "runtime": {"kind": "direct", "version": "test-version"},
        "inputs": [{"path": "data/raw/input.fits", "sha256": "a" * 64}],
        "outputs": [{"path": "outputs/test-result.json", "sha256": "b" * 64}],
    }


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_engine_run(repo, payload):
    path = (
        repo
        / "candidate"
        / "candidate-alpha"
        / "runs"
        / "test-engine"
        / "run-001"
        / "engine-run.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    input_path = repo / "candidate" / "candidate-alpha" / "data" / "raw" / "input.fits"
    output_path = repo / "candidate" / "candidate-alpha" / "outputs" / "test-result.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"synthetic input")
    output_path.write_text('{"synthetic": true}\n', encoding="utf-8")
    payload["inputs"][0]["sha256"] = _sha256(input_path)
    payload["outputs"][0]["sha256"] = _sha256(output_path)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_automated_triage():
    return {
        "schema_version": 1,
        "candidate_id": "candidate-alpha",
        "generated_at": "2000-01-01T00:01:00Z",
        "policy_id": "test-policy",
        "policy_version": "test-version",
        "status": "review-required",
        "records": [
            {
                "engine": "test-engine",
                "run_manifest_path": "runs/test-engine/run-001/engine-run.json",
                "run_manifest_sha256": "a" * 64,
                "status": "review-required",
                "reason": "Synthetic review condition.",
            }
        ],
    }


def _valid_survey_robustness():
    def bls_result(snr=8.0):
        return {
            "best_period": 3.0,
            "best_epoch": 1.0,
            "best_depth_ppm": 1000.0,
            "best_duration_hours": 2.0,
            "snr": snr,
        }

    phase_offsets = [0.25, 0.5, 0.75]
    recovery = []
    for index, phase_offset in enumerate(phase_offsets):
        recovered = index < 2
        recovery.append(
            {
                "injection": {
                    "period_days": 3.0,
                    "epoch_btjd": 1.0 + phase_offset * 3.0,
                    "duration_hours": 2.0,
                    "depth_ppm": 1000.0,
                    "phase_offset": phase_offset,
                },
                "period_match": recovered,
                "epoch_match": recovered,
                "snr_pass": recovered,
                "epoch_tolerance_hours": 2.0,
                "recovered": recovered,
                "best": bls_result(snr=8.0 if recovered else 5.0),
            }
        )
    return {
        "schema_version": 1,
        "source": "candidate-data",
        "survey_id": "test-survey",
        "candidate_id": "candidate-alpha",
        "sectors": [17],
        "configuration": {
            "duration_grid_hours": [1.0, 2.0, 4.0],
            "period_min_days": 0.5,
            "period_max_days": 20.0,
            "n_periods": 200,
            "detrend_window_days": 1.0,
            "scramble_seeds": [5, 7, 11],
            "period_agreement_fraction": 0.01,
            "injection_recovery": {
                "phase_offsets": phase_offsets,
                "minimum_recovered_trials": 2,
                "minimum_recovery_fraction": 2.0 / 3.0,
                "epoch_tolerance_duration_fraction": 1.0,
                "minimum_snr": 6.0,
            },
        },
        "input_files": ["data/raw/synthetic.fits"],
        "reference_signal": {
            **bls_result(),
            "usable_for_injection_recovery": True,
        },
        "diagnostics": {
            "variants": {
                "normalized": {"best": bls_result(), "trials": [bls_result()]},
                "running-median": {"best": bls_result(snr=7.0), "trials": [bls_result()]},
            },
            "controls": {
                "inverted": bls_result(snr=2.0),
                "scrambles": [{"seed": 5, "best": bls_result(snr=3.0)}],
                "max_snr": 3.0,
            },
        },
        "injection_recovery": recovery,
        "injection_recovery_summary": {
            "reference_signal_usable": True,
            "masked_cadences": 3,
            "trial_count": 3,
            "expected_trial_count": 3,
            "recovered_count": 2,
            "recovery_fraction": 2.0 / 3.0,
            "minimum_recovered_trials": 2,
            "minimum_recovery_fraction": 2.0 / 3.0,
            "passed": True,
        },
    }


def _register_artifact_survey_target(repo):
    survey_path = repo / "candidate" / "_surveys" / "test-survey" / "survey.json"
    if not survey_path.is_file():
        create_survey(repo, "test-survey", "tess", [17])
    target_path = (
        repo
        / "candidate"
        / "_surveys"
        / "test-survey"
        / "targets"
        / "candidate-alpha"
        / "target.json"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "survey_id": "test-survey",
                "candidate_id": "candidate-alpha",
                "registered_at": "2000-01-01T00:00:00Z",
                "status": "pending-eligibility",
                "search_result_path": None,
                "reason": "Synthetic test registration.",
                "search_reused": False,
            }
        ),
        encoding="utf-8",
    )


def _write_survey_robustness(repo, payload):
    _register_artifact_survey_target(repo)
    path = (
        repo
        / "candidate"
        / "candidate-alpha"
        / "outputs"
        / "survey_robustness.survey-test-survey.json"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_novelty_audit_passes_schema_validation(tmp_path):
    repo = _make_repo(tmp_path)
    path = repo / "candidate" / "candidate-alpha" / "decisions" / "novelty_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_valid_novelty_audit()), encoding="utf-8")

    assert _audit(repo).ok


def test_invalid_novelty_audit_is_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    audit = _valid_novelty_audit()
    audit["evidence"] = []
    path = repo / "candidate" / "candidate-alpha" / "decisions" / "novelty_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit), encoding="utf-8")

    report = _audit(repo)
    assert not report.ok
    assert any(v.rule == "schema-violation" for v in report.violations)


def test_valid_engine_run_passes_schema_validation(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    _write_engine_run(repo, _valid_engine_run())

    # Act and assert
    assert _audit(repo).ok


def test_engine_run_requires_matching_directory_identity(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    artifact = _valid_engine_run()
    artifact["engine"] = "other-engine"
    path = _write_engine_run(repo, artifact)

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path == path.as_posix() for violation in report.violations)


def test_engine_run_requires_matching_artifact_hashes(tmp_path):
    repo = _make_repo(tmp_path)
    artifact = _valid_engine_run()
    path = _write_engine_run(repo, artifact)
    artifact["outputs"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(artifact), encoding="utf-8")

    report = _audit(repo)

    assert any(violation.rule == "artifact-hash-mismatch" for violation in report.violations)


def test_engine_run_outside_candidate_is_rejected(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    path = repo / "engine-run.json"
    path.write_text(json.dumps(_valid_engine_run()), encoding="utf-8")

    # Act
    report = _audit(repo)

    # Assert
    assert any(
        violation.path == path.as_posix() and violation.rule == "engine-run-outside-candidate"
        for violation in report.violations
    )


def test_valid_automated_triage_passes_schema_validation(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    manifest_path = _write_engine_run(repo, _valid_engine_run())
    triage = _valid_automated_triage()
    triage["records"][0]["run_manifest_sha256"] = _sha256(manifest_path)
    triage["records"][0]["artifact_path"] = "outputs/test-result.json"
    triage["records"][0]["artifact_sha256"] = _sha256(
        repo / "candidate" / "candidate-alpha" / "outputs" / "test-result.json"
    )
    path = repo / "candidate" / "candidate-alpha" / "decisions" / "automated_triage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(triage), encoding="utf-8")

    # Act and assert
    assert _audit(repo).ok


def test_automated_triage_requires_matching_candidate_identity(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    manifest_path = _write_engine_run(repo, _valid_engine_run())
    artifact = _valid_automated_triage()
    artifact["candidate_id"] = "another-candidate"
    artifact["records"][0]["run_manifest_sha256"] = _sha256(manifest_path)
    artifact["records"][0]["artifact_path"] = "outputs/test-result.json"
    artifact["records"][0]["artifact_sha256"] = _sha256(
        repo / "candidate" / "candidate-alpha" / "outputs" / "test-result.json"
    )
    path = repo / "candidate" / "candidate-alpha" / "decisions" / "automated_triage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact), encoding="utf-8")

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path == path.as_posix() for violation in report.violations)


def test_automated_triage_requires_matching_manifest_hash(tmp_path):
    repo = _make_repo(tmp_path)
    _write_engine_run(repo, _valid_engine_run())
    triage = _valid_automated_triage()
    triage["records"][0]["run_manifest_sha256"] = "0" * 64
    triage["records"][0]["artifact_path"] = "outputs/test-result.json"
    triage["records"][0]["artifact_sha256"] = _sha256(
        repo / "candidate" / "candidate-alpha" / "outputs" / "test-result.json"
    )
    path = repo / "candidate" / "candidate-alpha" / "decisions" / "automated_triage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(triage), encoding="utf-8")

    report = _audit(repo)

    assert any(violation.rule == "triage-provenance-invalid" for violation in report.violations)
def test_legacy_subtree_sidecars_are_skipped(tmp_path):
    repo = _make_repo(tmp_path)
    legacy = repo / "candidate" / "candidate-alpha" / "legacy-project" / "data"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "old.provenance.json").write_text(
        json.dumps({"legacy": "format"}), encoding="utf-8"
    )

    assert _audit(repo).ok


def test_missing_schema_file_is_reported(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "schemas" / "provenance.schema.json").unlink()

    report = _audit(repo)
    assert not report.ok
    assert any(v.rule == "schema-file-missing" for v in report.violations)


def test_valid_survey_and_target_records_pass_schema_validation(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    survey = create_survey(repo, "test-survey", "tess", [17])
    target = create_candidate(repo, "survey-target", tic="123456789", mission="tess")
    register_survey_target(survey, target)

    # Act and assert
    assert _audit(repo).ok


@pytest.mark.parametrize(
    "path_parts, update",
    [
        (("candidate", "_surveys", "test-survey", "survey.json"), {"sectors": []}),
        (("candidate", "_surveys", "test-survey", "survey.json"), {"review_snr": -1}),
        (("candidate", "_surveys", "test-survey", "survey.json"), {"scientific_status": "claimed"}),
        (("candidate", "_surveys", "test-survey", "targets", "survey-target", "target.json"), {"status": "unknown-status"}),
        (("candidate", "_surveys", "test-survey", "targets", "survey-target", "target.json"), {"review_snr": -1}),
    ],
)
def test_invalid_survey_records_are_flagged(tmp_path, path_parts, update):
    # Arrange
    repo = _make_repo(tmp_path)
    survey = create_survey(repo, "test-survey", "tess", [17])
    target = create_candidate(repo, "survey-target", tic="123456789", mission="tess")
    register_survey_target(survey, target)
    path = repo.joinpath(*path_parts)
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(update)
    path.write_text(json.dumps(record), encoding="utf-8")

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path.endswith(path.as_posix()) for violation in report.violations)


def test_valid_survey_robustness_artifact_passes_schema_validation(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    _write_survey_robustness(repo, _valid_survey_robustness())

    # Act and assert
    assert _audit(repo).ok


def test_generated_survey_robustness_artifact_passes_schema_validation(tmp_path, monkeypatch):
    # Arrange
    repo = _make_repo(tmp_path)
    survey = create_survey(repo, "test-survey", "tess", [17])
    candidate = load_candidate(repo, "candidate-alpha")
    _register_artifact_survey_target(repo)
    raw_input = candidate.path / "data" / "raw" / "synthetic.fits"
    raw_input.write_bytes(b"fits")

    def bls_result(snr=8.0):
        return {
            "best_period": 3.0,
            "best_epoch": 1.0,
            "best_depth_ppm": 1000.0,
            "best_duration_hours": 2.0,
            "snr": snr,
        }

    def fake_table(*args, **kwargs):
        return {
            "time": [index * 0.1 for index in range(50)],
            "flux": [1.0] * 50,
            "sector": [17] * 50,
            "input_files": [raw_input],
        }

    def fake_diagnostics(*args, **kwargs):
        return {
            "variants": {
                "normalized": {"best": bls_result(), "trials": [bls_result()]},
                "running-median": {"best": bls_result(snr=7.0), "trials": [bls_result()]},
            },
            "controls": {
                "inverted": bls_result(snr=2.0),
                "scrambles": [{"seed": 5, "best": bls_result(snr=3.0)}],
                "max_snr": 3.0,
            },
        }

    def fake_recovery(
        time,
        flux,
        injections,
        duration_grid_hours,
        period_min_days,
        period_max_days,
        n_periods,
        tolerance,
        minimum_snr=None,
        epoch_tolerance_duration_fraction=1.0,
    ):
        results = []
        for injection in injections:
            best = bls_result(snr=minimum_snr)
            best["best_epoch"] = injection["epoch_btjd"]
            results.append(
                {
                    "injection": dict(injection),
                    "period_match": True,
                    "epoch_match": True,
                    "snr_pass": True,
                    "epoch_tolerance_hours": injection["duration_hours"]
                    * epoch_tolerance_duration_fraction,
                    "recovered": True,
                    "best": best,
                }
            )
        return results

    monkeypatch.setattr("exonym.survey.load_light_curve_table", fake_table)
    monkeypatch.setattr("exonym.survey.robustness_diagnostics", fake_diagnostics)
    monkeypatch.setattr("exonym.survey.injection_recovery_diagnostics", fake_recovery)

    # Act
    path, artifact = _run_survey_robustness(
        survey, candidate, {"source": "candidate-data", **bls_result()}, review_snr=6.0
    )

    # Assert
    assert path.is_file()
    assert artifact["injection_recovery_summary"]["passed"] is True
    assert _audit(repo).ok


def test_malformed_survey_robustness_artifact_is_flagged(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    path = _write_survey_robustness(repo, _valid_survey_robustness())
    path.write_text("not json", encoding="utf-8")

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path == path.as_posix() for violation in report.violations)


def test_incomplete_survey_robustness_artifact_is_flagged(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    artifact = _valid_survey_robustness()
    del artifact["injection_recovery_summary"]
    path = _write_survey_robustness(repo, artifact)

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path == path.as_posix() for violation in report.violations)


def test_wrong_type_survey_robustness_artifact_is_flagged(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    artifact = _valid_survey_robustness()
    artifact["sectors"] = "not-an-array"
    path = _write_survey_robustness(repo, artifact)

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path == path.as_posix() for violation in report.violations)


def test_usable_survey_robustness_artifact_requires_all_recovery_trials(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    artifact = _valid_survey_robustness()
    artifact["injection_recovery"] = []
    artifact["injection_recovery_summary"]["trial_count"] = 0
    artifact["injection_recovery_summary"]["recovered_count"] = 0
    artifact["injection_recovery_summary"]["recovery_fraction"] = 0.0
    path = _write_survey_robustness(repo, artifact)

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path == path.as_posix() for violation in report.violations)


def test_nonfinite_survey_robustness_artifact_number_is_flagged(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    artifact = _valid_survey_robustness()
    artifact["reference_signal"]["snr"] = float("nan")
    path = _write_survey_robustness(repo, artifact)

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path == path.as_posix() for violation in report.violations)


def test_relaxed_survey_robustness_recovery_policy_is_flagged(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    artifact = _valid_survey_robustness()
    policy = artifact["configuration"]["injection_recovery"]
    policy["minimum_recovered_trials"] = 1
    policy["minimum_recovery_fraction"] = 1.0 / 3.0
    path = _write_survey_robustness(repo, artifact)

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path == path.as_posix() for violation in report.violations)


def test_non_utf8_survey_robustness_artifact_is_flagged(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    path = _write_survey_robustness(repo, _valid_survey_robustness())
    path.write_bytes(b"\xff")

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path == path.as_posix() for violation in report.violations)


def test_survey_robustness_artifact_requires_its_own_candidate_workspace(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    artifact = _valid_survey_robustness()
    artifact["candidate_id"] = "another-candidate"
    path = _write_survey_robustness(repo, artifact)

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path == path.as_posix() for violation in report.violations)


def test_survey_robustness_artifact_requires_candidate_metadata(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    artifact = _valid_survey_robustness()
    artifact["candidate_id"] = "unowned-candidate"
    path = (
        repo
        / "candidate"
        / "unowned-candidate"
        / "outputs"
        / "survey_robustness.survey-test-survey.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(artifact), encoding="utf-8")

    # Act
    report = _audit(repo)

    # Assert
    assert any(
        violation.path == path.as_posix()
        and violation.rule == "survey-robustness-outside-candidate"
        for violation in report.violations
    )


def test_survey_robustness_artifact_requires_a_registered_survey_target(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    create_survey(repo, "test-survey", "tess", [17])
    path = (
        repo
        / "candidate"
        / "candidate-alpha"
        / "outputs"
        / "survey_robustness.survey-test-survey.json"
    )
    path.write_text(json.dumps(_valid_survey_robustness()), encoding="utf-8")

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path == path.as_posix() for violation in report.violations)


def test_survey_robustness_artifact_requires_matching_survey_target_identity(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    path = _write_survey_robustness(repo, _valid_survey_robustness())
    target_path = (
        repo
        / "candidate"
        / "_surveys"
        / "test-survey"
        / "targets"
        / "candidate-alpha"
        / "target.json"
    )
    target = json.loads(target_path.read_text(encoding="utf-8"))
    target["candidate_id"] = "other-candidate"
    target_path.write_text(json.dumps(target), encoding="utf-8")

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path == path.as_posix() for violation in report.violations)


def test_non_utf8_survey_metadata_is_flagged_without_aborting_validation(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    survey = create_survey(repo, "test-survey", "tess", [17])
    path = survey.path / "survey.json"
    path.write_bytes(b"\xff")

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.path == path.as_posix() for violation in report.violations)


def test_survey_robustness_artifact_outside_candidate_is_rejected(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    path = repo / "survey_robustness.survey-test-survey.json"
    path.write_text(json.dumps(_valid_survey_robustness()), encoding="utf-8")

    # Act
    report = _audit(repo)

    # Assert
    assert any(
        violation.path == path.as_posix()
        and violation.rule == "survey-robustness-outside-candidate"
        for violation in report.violations
    )


def test_legacy_survey_robustness_artifact_is_excluded_from_validation(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    path = (
        repo
        / "candidate"
        / "candidate-alpha"
        / "legacy-project"
        / "outputs"
        / "survey_robustness.survey-test-survey.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")

    # Act and assert
    assert _audit(repo).ok
