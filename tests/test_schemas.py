import hashlib
import json
import shutil

import numpy as np
import pytest

from exonym.isolation import IsolationReport
from exonym.freeze import freeze
from exonym.schemas import validate_schema_definitions, validate_schemas
from exonym.survey import (
    _run_survey_robustness,
    create_survey,
    register_survey_target,
    run_survey_sensitivity,
)
from exonym.survey_harvest import novelty_provider_urls
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
        "survey-sensitivity.schema.json",
        "engine-run.schema.json",
        "automated-triage.schema.json",
        "radial-velocity-observations.schema.json",
        "rv-keplerian-fit.schema.json",
        "planetsynth-characterization.schema.json",
        "anomalous-transit-hypothesis.schema.json",
        "planetsynth-interpretation.schema.json",
        "pyppluss-hypothesis-test.schema.json",
        "statistical-vetting-evidence.schema.json",
        "decisive-rejection.schema.json",
        "catalog-query-manifest.schema.json",
        "catalog-raw-response-metadata.schema.json",
        "catalog-snapshot.schema.json",
        "catalog-stellar-parameters.schema.json",
        "catalog-stellar-photometry.schema.json",
        "catalog-archive-discovery.schema.json",
        "catalog-contrast-curves.schema.json",
        "catalog-context.schema.json",
        "catalog-cross-match.schema.json",
        "known-signal-ephemeris-match.schema.json",
        "known-signal-ephemeris-evidence.schema.json",
        "stellar-activity.schema.json",
        "phase-curve.schema.json",
        "detrending-manifest.schema.json",
        "ldtk-quadratic-limb-darkening-prior.schema.json",
        "exofop-prior-retrieval.schema.json",
    ):
        shutil.copy2(
            "schemas/{0}".format(name), tmp_path / "schemas" / name
        )
    return tmp_path


def _audit(tmp_path):
    report = IsolationReport()
    validate_schemas(tmp_path, report)
    return report


def _prepare_freeze_source(repo):
    package_dir = repo / "src" / "exonym"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text('__version__ = "test"\n', encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        "[build-system]\nrequires = [\"setuptools\"]\nbuild-backend = \"setuptools.build_meta\"\n"
        "[project]\nname = \"exonym\"\nversion = \"0.0.0\"\n",
        encoding="utf-8",
    )
    (repo / "requirements-lock.txt").write_text("numpy==1.26.4\n", encoding="utf-8")


def test_clean_repository_passes_schema_validation(tmp_path):
    report = _audit(_make_repo(tmp_path))
    assert report.ok


def test_schema_definition_validation_rejects_invalid_shared_schema(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "schemas" / "candidate.schema.json").write_text('{"type": 42}\n', encoding="utf-8")
    report = IsolationReport()

    validate_schema_definitions(repo, report)

    assert any(violation.rule == "schema-definition-invalid" for violation in report.violations)


def test_release_snapshot_is_checked_for_inventory_and_hash_integrity(tmp_path):
    repo = _make_repo(tmp_path)
    _prepare_freeze_source(repo)
    candidate = load_candidate(repo, "candidate-alpha")
    release = freeze(candidate, version="v1.0.0")

    assert _audit(repo).ok

    (release / "workspace" / "candidate" / "candidate-alpha" / "candidate.json").write_text(
        "{}\n", encoding="utf-8"
    )
    report = _audit(repo)

    assert any(violation.rule == "release-file-hash-mismatch" for violation in report.violations)
    assert any(violation.rule == "release-snapshot-hash-mismatch" for violation in report.violations)


def test_release_snapshot_rejects_tampered_detached_manifest_digest(tmp_path):
    repo = _make_repo(tmp_path)
    _prepare_freeze_source(repo)
    candidate = load_candidate(repo, "candidate-alpha")
    release = freeze(candidate, version="v1.0.0")

    (release / "manifest.sha256").write_text("0" * 64 + "  manifest.json\n", encoding="ascii")
    report = _audit(repo)

    assert any(violation.rule == "release-manifest-digest-mismatch" for violation in report.violations)


def test_release_staging_directory_is_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    staging = repo / "candidate" / "candidate-alpha" / "releases" / ".v1.0.0.staging-leftover"
    staging.mkdir(parents=True)

    report = _audit(repo)

    assert any(violation.rule == "release-staging-leftover" for violation in report.violations)


def test_invalid_candidate_record_is_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    path = repo / "candidate" / "candidate-alpha" / "candidate.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["lifecycle"]["state"] = "mystery"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    report = _audit(repo)
    assert not report.ok
    assert any(v.rule == "schema-violation" for v in report.violations)


def test_schema_audit_rejects_duplicate_json_keys(tmp_path):
    repo = _make_repo(tmp_path)
    path = repo / "candidate" / "candidate-alpha" / "candidate.json"
    path.write_text(
        '{"schema_version": 2, "schema_version": 2}\n', encoding="utf-8"
    )

    report = _audit(repo)

    assert not report.ok
    assert any(
        violation.rule == "schema-violation" and "duplicate JSON key" in violation.detail
        for violation in report.violations
    )


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


def test_statistical_vetting_evidence_requires_candidate_owned_artifacts(tmp_path):
    repo = _make_repo(tmp_path)
    path = repo / "candidate" / "candidate-alpha" / "outputs" / "statistical_vetting_evidence.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id": "another-candidate",
                "generated_at": "2000-01-01T00:00:00Z",
                "signal": None,
                "status": "pass",
                "diagnostics": [],
            }
        ),
        encoding="utf-8",
    )

    report = _audit(repo)

    assert any(violation.path == path.as_posix() for violation in report.violations)


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
                "sha256": hashlib.sha256(b"fits").hexdigest(),
                "fetched_by": "test",
            }
        ),
        encoding="utf-8",
    )

    assert _audit(repo).ok


def test_provenance_sidecar_hash_mismatch_is_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    raw = repo / "candidate" / "candidate-alpha" / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    product = raw / "lc.fits"
    product.write_bytes(b"original")
    sidecar = raw / "lc.provenance.json"
    sidecar.write_text(
        json.dumps(
            {
                "source_uri": "https://archive.stsci.edu/example",
                "download_timestamp_utc": "2026-08-04T00:00:00Z",
                "sha256": hashlib.sha256(b"original").hexdigest(),
                "fetched_by": "test",
            }
        ),
        encoding="utf-8",
    )
    product.write_bytes(b"tampered")

    report = _audit(repo)

    assert any(violation.rule == "provenance-hash-mismatch" for violation in report.violations)


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


def _write_observed_triceratops_report(repo, claim_eligible=False):
    workspace = repo / "candidate" / "candidate-alpha"
    photometry_path = workspace / "data" / "raw" / "observed.fits"
    photometry_path.parent.mkdir(parents=True, exist_ok=True)
    photometry_path.write_bytes(b"observed photometry")
    report_path = workspace / "outputs" / "triceratops_report.json"
    report_path.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-alpha",
                "random_seed": 17,
                "claim_eligible": claim_eligible,
                "input_provenance": {
                    "representation": "phase-folded observed candidate photometry",
                    "flux_error_source": "reported per-cadence uncertainties",
                    "raw_cadence_count": 50,
                    "phase_bin_count": 10,
                    "flux_error_scalar": 0.001,
                    "exposure_days": 0.001,
                    "observed_depth_ppm": 100.0,
                    "input_files": [
                        {
                            "path": "data/raw/observed.fits",
                            "sha256": _sha256(photometry_path),
                        }
                    ],
                    "ephemeris_field_sources": {
                        "period_days": "candidate-config",
                        "epoch_btjd": "candidate-config",
                        "duration_days": "candidate-config",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return report_path


def test_scientific_verification_requires_observed_triceratops_input_and_disables_claims(tmp_path):
    repo = _make_repo(tmp_path)
    report_path = _write_observed_triceratops_report(repo, claim_eligible=True)
    report = _audit(repo)

    assert any(
        violation.path == report_path.as_posix()
        and violation.rule == "scientific-fpp-claim-disabled"
        for violation in report.violations
    )


def test_scientific_verification_accepts_observed_unclaimable_triceratops_report(tmp_path):
    repo = _make_repo(tmp_path)
    _write_observed_triceratops_report(repo)

    assert _audit(repo).ok


def test_scientific_verification_rejects_any_active_fpp_claim(tmp_path):
    repo = _make_repo(tmp_path)
    claim_path = repo / "candidate" / "candidate-alpha" / "claims" / "fpp.json"
    claim_path.write_text(
        json.dumps(
            {
                "parameter": "fpp",
                "value": 0.001,
                "uncertainty_upper": 0.001,
                "uncertainty_lower": 0.001,
                "unit": "dimensionless",
                "method": "test",
                "candidate_id": "candidate-alpha",
                "report_path": "outputs/triceratops_report.json",
                "report_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    report = _audit(repo)
    assert any(violation.rule == "scientific-fpp-claim-disabled" for violation in report.violations)


def test_scientific_verification_rejects_overclaimed_or_unowned_localization(tmp_path):
    repo = _make_repo(tmp_path)
    path = repo / "candidate" / "candidate-alpha" / "outputs" / "prf_localization_results.json"
    path.write_text(
        json.dumps(
            {
                "candidate_id": "wrong-candidate",
                "calibration_status": "uncalibrated",
                "summary": {"conclusion": "target_dominant_among_modeled_sources"},
            }
        ),
        encoding="utf-8",
    )

    report = _audit(repo)
    assert any(violation.rule == "scientific-localization-ownership-invalid" for violation in report.violations)
    assert any(violation.rule == "scientific-localization-overclaim" for violation in report.violations)


def test_scientific_verification_accepts_inconclusive_uncalibrated_localization(tmp_path):
    repo = _make_repo(tmp_path)
    path = repo / "candidate" / "candidate-alpha" / "outputs" / "prf_localization_results.json"
    path.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-alpha",
                "calibration_status": "uncalibrated",
                "summary": {"conclusion": "inconclusive_uncalibrated_prf"},
            }
        ),
        encoding="utf-8",
    )

    assert _audit(repo).ok


def test_scientific_verification_rejects_self_declared_localization_calibration(tmp_path):
    repo = _make_repo(tmp_path)
    path = repo / "candidate" / "candidate-alpha" / "outputs" / "prf_localization_results.json"
    path.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-alpha",
                "calibration_status": "calibrated",
                "calibration_evidence": {"path": "outputs/benchmark.json", "sha256": "a" * 64},
                "summary": {"conclusion": "target_dominant_among_modeled_sources"},
            }
        ),
        encoding="utf-8",
    )

    report = _audit(repo)

    assert any(
        violation.rule == "scientific-localization-calibration-unsupported"
        for violation in report.violations
    )


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


def _write_valid_v2_novelty_audit(repo):
    workspace = repo / "candidate" / "candidate-alpha"
    source_uris = dict(novelty_provider_urls("123456789"))
    retrieval_id = "a" * 32
    evidence_dir = workspace / "data" / "external" / "novelty" / retrieval_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = []
    for index, provider in enumerate(("nasa-toi", "nasa-confirmed", "exofop")):
        extension = "json" if provider == "exofop" else "csv"
        response_path = evidence_dir / "{0}-{1}.{2}".format(index, provider, extension)
        response_path.write_bytes(
            (
                b'{"basic_info":{"tic_id":"123456789"},"tois":[],"ctois":[],"planet_parameters":[]}'
                if provider == "exofop"
                else (b"toi,tid\n" if provider == "nasa-toi" else b"pl_name,tic_id\n")
            )
        )
        evidence.append(
            {
                "source_uri": source_uris[provider],
                "retrieved_at": "2000-01-01T00:00:00Z",
                "finding": "Synthetic {0} response.".format(provider),
                "evidence_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
                "provider": provider,
                "response_path": response_path.relative_to(workspace).as_posix(),
            }
        )
    audit = _valid_novelty_audit()
    audit["schema_version"] = 2
    audit["evidence"] = evidence
    path = workspace / "decisions" / "novelty_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit), encoding="utf-8")
    return path, evidence_dir


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
            "n_distinct_transit_events": 3,
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
            "period_grid": "astropy-autopower-baseline-duration-resolved",
            "injection_model": "finite-exposure-box-overlap",
            "exposure_model": "median-positive-cadence-inferred",
            "detrend_window_days": 1.0,
            "detrend_gap_break_window_fraction": 0.5,
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
                "by_variant": {
                    "normalized": {
                        "inverted": bls_result(snr=2.0),
                        "scrambles": [{"seed": 5, "best": bls_result(snr=3.0)}],
                        "max_snr": 3.0,
                    },
                    "running-median": {
                        "inverted": bls_result(snr=2.0),
                        "scrambles": [{"seed": 5, "best": bls_result(snr=3.0)}],
                        "max_snr": 3.0,
                    },
                },
                "scramble_method": "independent-sector-circular-shift",
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


def test_valid_v2_novelty_audit_is_hash_bound(tmp_path):
    repo = _make_repo(tmp_path)
    _write_valid_v2_novelty_audit(repo)

    assert _audit(repo).ok


def test_v2_novelty_audit_rejects_a_tampered_retained_response(tmp_path):
    repo = _make_repo(tmp_path)
    audit_path, evidence_dir = _write_valid_v2_novelty_audit(repo)
    (evidence_dir / "0-nasa-toi.csv").write_bytes(b"tampered")

    report = _audit(repo)

    assert any(
        violation.path == audit_path.as_posix()
        and violation.rule == "novelty-evidence-provenance-invalid"
        for violation in report.violations
    )


def test_v2_novelty_audit_rejects_a_semantically_invalid_retained_response(tmp_path):
    repo = _make_repo(tmp_path)
    audit_path, evidence_dir = _write_valid_v2_novelty_audit(repo)
    response_path = evidence_dir / "0-nasa-toi.csv"
    response_path.write_bytes(b"not a NASA response")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["evidence"][0]["evidence_sha256"] = hashlib.sha256(response_path.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    report = _audit(repo)

    assert any(
        violation.path == audit_path.as_posix()
        and violation.rule == "novelty-evidence-provenance-invalid"
        and "semantically valid" in violation.detail
        for violation in report.violations
    )


def test_historic_engine_run_without_version_known_passes_schema_validation(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    _write_engine_run(repo, _valid_engine_run())

    # Act and assert
    assert _audit(repo).ok


def test_engine_run_accepts_effective_auto_vet_sector_scope(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    artifact = _valid_engine_run()
    artifact["automation"] = {
        "steps": [{"name": "ingest", "status": "succeeded"}],
        "sectors_used": [2, 5],
        "claim_eligible": False,
        "disposition_changed": False,
        "workflow_advanced": False,
    }
    _write_engine_run(repo, artifact)

    # Act and assert
    assert _audit(repo).ok


def test_engine_run_allows_explicit_unknown_runtime_version(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    artifact = _valid_engine_run()
    artifact["runtime"] = {
        "kind": "direct",
        "version": None,
        "version_known": False,
    }
    _write_engine_run(repo, artifact)

    # Act and assert
    assert _audit(repo).ok


@pytest.mark.parametrize(
    ("version", "version_known"),
    (
        (None, True),
        ("test-version", False),
    ),
)
def test_engine_run_rejects_inconsistent_runtime_version_metadata(
    tmp_path, version, version_known
):
    # Arrange
    repo = _make_repo(tmp_path)
    artifact = _valid_engine_run()
    artifact["runtime"] = {
        "kind": "direct",
        "version": version,
        "version_known": version_known,
    }
    _write_engine_run(repo, artifact)

    # Act
    report = _audit(repo)

    # Assert
    assert any(violation.rule == "schema-violation" for violation in report.violations)


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


def test_detrending_manifest_schema_and_artifact_hash_are_verified(tmp_path):
    from exonym.detrending import detrend_candidate

    repo = _make_repo(tmp_path)
    candidate = load_candidate(repo, "candidate-alpha")
    raw_path = candidate.path / "data" / "raw" / "source.fits"
    raw_path.write_bytes(b"synthetic source")
    raw_path.with_name("source.provenance.json").write_text(
        json.dumps(
            {
                "source_uri": "https://example.invalid/source",
                "download_timestamp_utc": "2026-01-01T00:00:00Z",
                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "fetched_by": "synthetic-test",
            }
        ),
        encoding="utf-8",
    )
    result = detrend_candidate(
        candidate,
        time_btjd=[0.0, 1.0, 2.0, 3.0, 4.0],
        flux=[1.0, 1.001, 0.999, 1.0, 1.0],
        method="running-median",
        window_days=0.5,
        sector=[1, 1, 1, 1, 1],
        input_products=[
            {
                "path": "data/raw/source.fits",
                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            }
        ],
    )

    assert _audit(repo).ok
    legacy_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    legacy_manifest["schema_version"] = 1
    legacy_manifest["configuration"] = {"window_days": 0.5}
    result.manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
    assert _audit(repo).ok
    result.artifact_path.write_bytes(b"tampered")
    report = _audit(repo)
    assert any(v.rule == "artifact-hash-mismatch" for v in report.violations)


def test_ldtk_prior_schema_and_stellar_input_hash_are_verified(tmp_path):
    repo = _make_repo(tmp_path)
    candidate = repo / "candidate" / "candidate-alpha"
    parameters_path = candidate / "data" / "external" / "stellar_params.json"
    parameters_path.parent.mkdir(parents=True, exist_ok=True)
    parameters_path.write_text("{}\n", encoding="utf-8")
    prior_path = candidate / "outputs" / "ldtk_quadratic_limb_darkening_prior.json"
    prior_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "work_package": "LDTK_QUADRATIC_LIMB_DARKENING_PRIOR",
                "generated_utc": "2026-01-01T00:00:00Z",
                "candidate_id": "candidate-alpha",
                "method": "test",
                "ldtk": {"version": "test", "coefficient_method": "coeffs_qd", "monte_carlo": True},
                "input_provenance": {
                    "stellar_parameters_path": "data/external/stellar_params.json",
                    "stellar_parameters_sha256": _sha256(parameters_path),
                },
                "stellar_parameters": {
                    "teff_k": {"value": 5700.0, "uncertainty": 75.0, "unit": "K"}
                },
                "quadratic_coefficients": [
                    {"filter": "TESS", "u1": 0.2, "u1_err": 0.01, "u2": 0.3, "u2_err": 0.02, "unit": "dimensionless"}
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _audit(repo).ok
    parameters_path.write_text('{"tampered": true}\n', encoding="utf-8")
    report = _audit(repo)
    assert any(v.rule == "artifact-hash-mismatch" for v in report.violations)


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
            "n_distinct_transit_events": 3,
        }

    def fake_table(*args, **kwargs):
        return {
            "time": [index * 0.1 for index in range(50)],
            "flux": [1.0] * 50,
            "sector": [17] * 50,
            "input_files": [raw_input],
        }

    def fake_diagnostics(*args, **kwargs):
        normalized_controls = {
            "inverted": bls_result(snr=2.0),
            "scrambles": [{"seed": 5, "best": bls_result(snr=3.0)}],
            "max_snr": 3.0,
        }
        running_median_controls = {
            "inverted": bls_result(snr=2.0),
            "scrambles": [{"seed": 5, "best": bls_result(snr=3.0)}],
            "max_snr": 3.0,
        }
        return {
            "variants": {
                "normalized": {"best": bls_result(), "trials": [bls_result()]},
                "running-median": {"best": bls_result(snr=7.0), "trials": [bls_result()]},
            },
            "controls": {
                "inverted": normalized_controls["inverted"],
                "scrambles": normalized_controls["scrambles"],
                "by_variant": {
                    "normalized": normalized_controls,
                    "running-median": running_median_controls,
                },
                "scramble_method": "independent-sector-circular-shift",
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
            sectors=None,
            detrend_window_days=None,
            flux_err=None,
            gap_break_window_fraction=0.5,
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
                    "branches": {
                        "normalized": {
                            "period_match": True,
                            "epoch_match": True,
                            "snr_pass": True,
                            "recovered": True,
                            "best": dict(best),
                        },
                        "running-median": {
                            "period_match": True,
                            "epoch_match": True,
                            "snr_pass": True,
                            "recovered": True,
                            "best": dict(best),
                        },
                    },
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


def test_generated_survey_sensitivity_artifact_passes_schema_validation(tmp_path, monkeypatch):
    # Arrange
    repo = _make_repo(tmp_path)
    survey = create_survey(repo, "test-survey", "tess", [17])
    candidate = load_candidate(repo, "candidate-alpha")
    _register_artifact_survey_target(repo)
    raw_input = candidate.path / "data" / "raw" / "s0017_lc.fits"
    raw_input.write_bytes(b"fits")

    def fake_table(*args, **kwargs):
        return {
            "time": [index * 0.1 for index in range(100)],
            "flux": [1.0] * 100,
            "sector": [17] * 100,
            "input_files": [raw_input],
        }

    def fake_recovery(time, flux, injections, *args, **kwargs):
        results = []
        for injection in injections:
            best = {
                "best_period": injection["period_days"],
                "best_epoch": injection["epoch_btjd"],
                "best_depth_ppm": injection["depth_ppm"],
                "best_duration_hours": injection["duration_hours"],
                "snr": 7.0,
                "n_distinct_transit_events": 2,
            }
            branch = {
                "period_match": True,
                "epoch_match": True,
                "snr_pass": True,
                "recovered": True,
                "best": best,
            }
            results.append(
                {
                    "injection": dict(injection),
                    "period_match": True,
                    "epoch_match": True,
                    "snr_pass": True,
                    "epoch_tolerance_hours": injection["duration_hours"],
                    "recovered": True,
                    "best": best,
                    "branches": {
                        "normalized": dict(branch),
                        "running-median": dict(branch),
                    },
                }
            )
        return results

    monkeypatch.setattr("exonym.survey._current_eligible_audit", lambda candidate: True)
    monkeypatch.setattr("exonym.survey.load_light_curve_table", fake_table)
    monkeypatch.setattr("exonym.survey.injection_recovery_diagnostics", fake_recovery)
    monkeypatch.setattr(
        "exonym.survey._input_manifest_records",
        lambda *args, **kwargs: [
            {
                "path": "data/raw/s0017_lc.fits",
                "sha256": _sha256(raw_input),
                "provenance_path": None,
                "provenance": None,
            }
        ],
    )

    # Act
    output = run_survey_sensitivity(survey, candidate)

    # Assert
    assert output.is_file()
    assert _audit(repo).ok


def test_survey_sensitivity_retains_literal_none_trials_as_schema_valid_non_detections(
    tmp_path, monkeypatch
):
    """A missing BLS trial result must produce a valid zero-recovery artifact."""
    repo = _make_repo(tmp_path)
    survey = create_survey(repo, "test-survey", "tess", [17])
    candidate = load_candidate(repo, "candidate-alpha")
    _register_artifact_survey_target(repo)
    raw_input = candidate.path / "data" / "raw" / "s0017_lc.fits"
    raw_input.write_bytes(b"fits")

    def fake_table(*args, **kwargs):
        return {
            "time": [index * 0.1 for index in range(100)],
            "flux": [1.0] * 100,
            "sector": [17] * 100,
            "input_files": [raw_input],
        }

    monkeypatch.setattr("exonym.survey._current_eligible_audit", lambda candidate: True)
    monkeypatch.setattr("exonym.survey.load_light_curve_table", fake_table)
    monkeypatch.setattr(
        "exonym.discovery.search_duration_grid", lambda *args, **kwargs: (None, [])
    )
    monkeypatch.setattr(
        "exonym.survey._input_manifest_records",
        lambda *args, **kwargs: [
            {
                "path": "data/raw/s0017_lc.fits",
                "sha256": _sha256(raw_input),
                "provenance_path": None,
                "provenance": None,
            }
        ],
    )

    output = run_survey_sensitivity(survey, candidate)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["summary"]["recovered_count"] == 0
    assert artifact["summary"]["trial_count"] == 108
    assert all(entry["recovered"] is False for entry in artifact["injection_recovery"])
    assert all(
        entry["best"]["detection_status"] == "no-detection"
        and entry["best"]["best_period"] is None
        and entry["branches"]["normalized"]["best"]["detection_status"] == "no-detection"
        and entry["branches"]["running-median"]["best"]["detection_status"] == "no-detection"
        for entry in artifact["injection_recovery"]
    )
    assert _audit(repo).ok


def test_exofop_prior_retrieval_is_schema_and_hash_bound(tmp_path, monkeypatch):
    # Arrange
    from exonym.priors import fetch_exofop_priors

    repo = _make_repo(tmp_path)
    candidate = load_candidate(repo, "candidate-alpha")
    csv_body = (
        "TOI,TIC ID,Period (days),Epoch (BJD),Depth (ppm),Duration (hours)\n"
        "100.01,123456789,4.0,2458123.0,500.0,2.0\n"
    )

    class FakeResponse:
        status = 200

        def read(self):
            return csv_body.encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr("exonym.priors.urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())

    # Act
    written = fetch_exofop_priors(candidate)

    # Assert
    assert len(written) == 1
    assert _audit(repo).ok
    config = json.loads(written[0].read_text(encoding="utf-8"))
    raw_path = candidate.path / config["provenance"]["raw_response_path"]
    raw_path.write_text("tampered", encoding="utf-8")
    report = _audit(repo)
    assert any("raw_response hash does not match" in violation.detail for violation in report.violations)


def test_generated_stellar_activity_artifact_is_schema_valid(tmp_path, monkeypatch):
    from exonym import activity

    repo = _make_repo(tmp_path)
    candidate = load_candidate(repo, "candidate-alpha")
    time = np.linspace(0.0, 27.0, 800)
    table = {
        "time": time,
        "flux": 1.0 + 300e-6 * np.sin(2.0 * np.pi * time / 5.0),
        "flux_err": np.full(time.size, 100e-6),
        "sector": np.ones(time.size, dtype=int),
    }
    ephemeris = {
        "period_days": 3.5,
        "epoch_btjd": 2.0,
        "duration_days": 0.12,
        "source": "candidate-data",
        "field_sources": {
            "period_days": "candidate-data",
            "epoch_btjd": "candidate-data",
            "duration_days": "candidate-data",
        },
    }
    monkeypatch.setattr(activity, "load_light_curve_table", lambda *_args, **_kwargs: table)
    monkeypatch.setattr(activity, "load_transit_ephemeris", lambda _workspace: ephemeris)

    output = activity.run_stellar_activity(candidate)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scientific_status"] == "exploratory-activity-diagnostic"
    assert payload["validation_eligible"] is False
    assert _audit(repo).ok


def test_generated_phase_curve_artifact_is_schema_valid(tmp_path, monkeypatch):
    from exonym import phasecurve

    repo = _make_repo(tmp_path)
    candidate = load_candidate(repo, "candidate-alpha")
    outputs = candidate.path / "outputs"
    ephemeris = {
        "period_days": 3.0,
        "epoch_btjd": 100.0,
        "duration_days": 0.1,
        "source": "candidate-data",
        "field_sources": {},
    }
    (outputs / "mcmc_transit_fit.json").write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "model": "batman quadratic limb darkening, stellar-density locked, eccentric orbit",
                "ephemeris": {"period_days": 3.0, "epoch_btjd": 100.0},
            }
        ),
        encoding="utf-8",
    )
    chain = np.tile(
        np.array([0.1, 0.0, 0.2, 1.0, -8.0, 0.3, 0.3, 0.2**0.5, 0.0]),
        (32, 1),
    )
    np.save(str(outputs / "mcmc_transit_fit_chain.npy"), chain)
    table = phasecurve._synthetic_phase_curve_table()
    table.pop("_duration_days")
    table.pop("_epoch_btjd")
    table.pop("_period_days")
    monkeypatch.setattr(phasecurve, "load_light_curve_table", lambda *_args, **_kwargs: table)
    monkeypatch.setattr(phasecurve, "load_transit_ephemeris", lambda _workspace: ephemeris)

    output = phasecurve.run_phase_curve_search(candidate)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.1"
    assert payload["secondary_eclipse_control"]["mode"] == "eccentric-posterior-marginalized-box-control"
    assert _audit(repo).ok


def test_exofop_prior_manifest_outside_candidate_is_rejected(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    outside_manifest = repo / "exofop-prior-manifest.json"
    outside_manifest.write_text("{}\n", encoding="utf-8")

    # Act
    report = _audit(repo)

    # Assert
    assert any(
        violation.path == outside_manifest.as_posix()
        and violation.rule == "exofop-prior-manifest-outside-candidate"
        for violation in report.violations
    )


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


def test_survey_robustness_artifact_requires_distinct_event_evidence(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    artifact = _valid_survey_robustness()
    del artifact["reference_signal"]["n_distinct_transit_events"]
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
