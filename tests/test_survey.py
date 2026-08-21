"""Tests for candidate-local blind-discovery survey controls."""

import json
import hashlib

import pytest

from exonym.survey import (
    _robustness_passes,
    _survey_alias_controls,
    create_survey,
    exclude_survey_target,
    load_survey,
    register_survey_target,
    run_survey_sensitivity,
    run_survey_search,
    survey_summary,
    validate_survey_id,
)
from exonym.survey_harvest import novelty_provider_urls
from exonym.workspace import create_candidate


def _eligible_audit(workspace):
    retrieval_id = "a" * 32
    tic = workspace.metadata["identifiers"]["tic"]
    source_uris = dict(novelty_provider_urls(tic))
    evidence_dir = workspace.path / "data" / "external" / "novelty" / retrieval_id
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
                "retrieved_at": "2026-01-01T00:00:00Z",
                "finding": "Synthetic {0} eligibility response.".format(provider),
                "evidence_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
                "provider": provider,
                "response_path": response_path.relative_to(workspace.path).as_posix(),
            }
        )
    audit = {
        "schema_version": 2,
        "candidate_id": workspace.candidate_id,
        "retrieved_at": "2026-01-01T00:00:00Z",
        "freshness": {"expires_at": "2099-01-01T00:00:00Z"},
        "status": "eligible",
        "decision_basis": "Synthetic test audit.",
        "evidence": evidence,
    }
    path = workspace.path / "decisions" / "novelty_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit), encoding="utf-8")


def _bls_result(snr=8.0):
    return {
        "detection_status": "detected" if snr >= 7.1 else "no-detection",
        "time_system": "BTJD_TDB",
        "detection_threshold_snr": 7.1,
        "best_period": 3.0,
        "best_epoch": 1.0,
        "best_depth_ppm": 1000.0,
        "best_duration_hours": 2.0,
        "snr": snr,
        "n_distinct_transit_events": 3,
    }


def _raw_input_table(workspace):
    product = workspace.path / "data" / "raw" / "s0017_lc.fits"
    product.parent.mkdir(parents=True, exist_ok=True)
    product.write_bytes(b"synthetic-raw-survey-photometry")
    product_hash = hashlib.sha256(product.read_bytes()).hexdigest()
    sidecar = product.with_name(product.stem + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "source_uri": "https://archive.example.invalid/" + product.name,
                "download_timestamp_utc": "2026-01-01T00:00:00Z",
                "sha256": product_hash,
                "fetched_by": "synthetic-test",
            }
        ),
        encoding="utf-8",
    )
    return {
        "input_files": [product],
        "input_sha256s": [product_hash],
    }


def _robustness_artifact(
    survey, candidate, search_result, review_snr, recovered_trials=(True, True, False)
):
    signal = {
        "best_period": float(search_result["best_period"]),
        "best_epoch": float(search_result["best_epoch"]),
        "best_depth_ppm": float(search_result["best_depth_ppm"]),
        "best_duration_hours": float(search_result["best_duration_hours"]),
        "snr": float(search_result["snr"]),
        "n_distinct_transit_events": int(search_result["n_distinct_transit_events"]),
        "usable_for_injection_recovery": True,
    }
    phase_offsets = [0.25, 0.5, 0.75]
    recovery = []
    for phase_offset, recovered in zip(phase_offsets, recovered_trials):
        injection = {
            "period_days": signal["best_period"],
            "epoch_btjd": signal["best_epoch"] + phase_offset * signal["best_period"],
            "duration_hours": signal["best_duration_hours"],
            "depth_ppm": signal["best_depth_ppm"],
            "phase_offset": phase_offset,
        }
        best = _bls_result(snr=review_snr if recovered else review_snr - 1.0)
        best["best_epoch"] = injection["epoch_btjd"] if recovered else signal["best_epoch"]
        branch = {
            "period_match": True,
            "epoch_match": recovered,
            "snr_pass": recovered,
            "recovered": recovered,
            "best": best,
        }
        recovery.append(
            {
                "injection": injection,
                "period_match": True,
                "epoch_match": recovered,
                "snr_pass": recovered,
                "epoch_tolerance_hours": injection["duration_hours"],
                "recovered": recovered,
                "best": best,
                "branches": {
                    "normalized": dict(branch),
                    "running-median": dict(branch),
                },
            }
        )
    recovered_count = sum(recovered_trials)
    normalized_controls = {
        "inverted": _bls_result(snr=2.0),
        "scrambles": [
            {"seed": seed, "best": _bls_result(snr=2.0)} for seed in (5, 7, 11)
        ],
        "max_snr": 2.0,
    }
    running_median_controls = {
        "inverted": _bls_result(snr=2.0),
        "scrambles": [
            {"seed": seed, "best": _bls_result(snr=2.0)} for seed in (5, 7, 11)
        ],
        "max_snr": 2.0,
    }
    return (
        candidate.path / "outputs" / "survey_robustness.survey-test-survey.json",
        {
            "candidate_id": candidate.candidate_id,
            "configuration": {
                "duration_grid_hours": [1.0, 2.0, 4.0],
                "period_min_days": 0.5,
                "period_max_days": 20.0,
                "n_periods": 200,
                "period_grid": "astropy-autopower-baseline-duration-resolved",
                "detrend_window_days": 1.0,
                "scramble_seeds": [5, 7, 11],
                "period_agreement_fraction": 0.01,
                "injection_recovery": {
                    "phase_offsets": phase_offsets,
                    "minimum_recovered_trials": 2,
                    "minimum_recovery_fraction": 2.0 / 3.0,
                    "epoch_tolerance_duration_fraction": 1.0,
                    "minimum_snr": review_snr,
                },
            },
            "diagnostics": {
                "variants": {
                    "normalized": {"best": _bls_result(snr=8.0)},
                    "running-median": {"best": _bls_result(snr=7.0)},
                },
                "controls": {
                    "inverted": normalized_controls["inverted"],
                    "scrambles": normalized_controls["scrambles"],
                    "by_variant": {
                        "normalized": normalized_controls,
                        "running-median": running_median_controls,
                    },
                    "scramble_method": "independent-sector-circular-shift",
                    "max_snr": 2.0,
                },
            },
            "reference_signal": signal,
            "injection_recovery": recovery,
            "injection_recovery_summary": {
                "reference_signal_usable": True,
                "masked_cadences": 3,
                "trial_count": len(recovered_trials),
                "expected_trial_count": len(phase_offsets),
                "recovered_count": recovered_count,
                "recovery_fraction": float(recovered_count) / len(recovered_trials),
                "minimum_recovered_trials": 2,
                "minimum_recovery_fraction": 2.0 / 3.0,
                "passed": recovered_count >= 2,
            },
        },
    )


def test_survey_registers_toi_free_target_and_retains_denominator(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(
        tmp_path, "survey-target", tic="123456789", mission="tess"
    )

    # Act
    record_path = register_survey_target(survey, workspace)
    summary = survey_summary(survey)

    # Assert
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "pending-eligibility"
    assert summary["outcome_counts"] == {"pending-eligibility": 1}
    assert summary["targets"][0]["candidate_id"] == workspace.candidate_id


def test_survey_rejects_known_toi_workspace(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(
        tmp_path,
        "known-workspace",
        toi="1234.01",
        tic="123456789",
        mission="tess",
    )

    # Act and assert
    with pytest.raises(ValueError, match="known TOI"):
        register_survey_target(survey, workspace)


def test_survey_search_blocks_without_a_current_eligible_audit(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(
        tmp_path, "survey-target", tic="123456789", mission="tess"
    )
    register_survey_target(survey, workspace)

    # Act
    record_path = run_survey_search(survey, workspace, review_snr=6.0)

    # Assert
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "blocked-novelty-audit"
    assert record["search_result_path"] is None
    assert "updated_at" in record


def test_survey_sensitivity_is_two_branch_diagnostic_without_routing_change(tmp_path, monkeypatch):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    target_path = register_survey_target(survey, workspace)
    _eligible_audit(workspace)
    raw_input = workspace.path / "data" / "raw" / "s0017_lc.fits"
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

    monkeypatch.setattr("exonym.survey.load_light_curve_table", fake_table)
    monkeypatch.setattr("exonym.survey.injection_recovery_diagnostics", fake_recovery)
    monkeypatch.setattr(
        "exonym.survey._input_manifest_records",
        lambda *args, **kwargs: [
            {
                "path": "data/raw/s0017_lc.fits",
                "sha256": "a" * 64,
                "provenance_path": None,
                "provenance": None,
            }
        ],
    )

    # Act
    output = run_survey_sensitivity(survey, workspace)

    # Assert
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["scientific_status"] == "candidate-level-injection-recovery-diagnostic"
    assert artifact["validation_eligible"] is False
    assert artifact["completeness_eligible"] is False
    assert artifact["summary"]["grid_cell_count"] == 27
    assert artifact["summary"]["trial_count"] == 108
    assert artifact["summary"]["recovery_interval_95"] == {
        "method": "wilson-score",
        "confidence_level": 0.95,
        "lower": pytest.approx(0.9657, rel=1e-3),
        "upper": 1.0,
    }
    assert artifact["summary"]["cells"][0]["trial_count"] == 4
    assert artifact["summary"]["cells"][0]["recovery_interval_95"]["lower"] < 1.0
    assert artifact["calibration_limits"]["population_false_alarm_probability"] is None
    assert artifact["calibration_limits"]["population_detection_reliability"] is None
    assert set(artifact["injection_recovery"][0]["branches"]) == {
        "normalized",
        "running-median",
    }
    assert json.loads(target_path.read_text(encoding="utf-8"))["status"] == "pending-eligibility"


def test_survey_alias_controls_preserve_harmonic_diagnostics_without_disposition():
    table = {
        "time": [index * 0.1 for index in range(160)],
        "flux": [1.0] * 160,
    }
    reference_signal = {
        "best_period": 3.0,
        "best_epoch": 1.0,
        "best_depth_ppm": 1000.0,
        "best_duration_hours": 2.0,
        "snr": 8.0,
        "n_distinct_transit_events": 3,
        "usable_for_injection_recovery": True,
    }

    controls = _survey_alias_controls(table, reference_signal)

    assert controls["status"] == "computed"
    assert controls["method"] == "fixed-ephemeris-odd-even-half-phase-double-period-v1"
    assert controls["half_phase_control"]["status"] in {
        "insufficient_coverage",
        "uncertainty_unavailable",
        "measured",
    }
    assert controls["double_period_hypothesis"]["period_days"] == 6.0
    assert "do not identify an eclipsing binary" in controls["interpretation"]


def test_survey_search_replaces_an_invalidated_record_after_a_fresh_run(tmp_path, monkeypatch):
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    record_path = register_survey_target(survey, workspace)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["status"] = "invalidated-needs-rerun"
    record["reason"] = "Prior artifact does not satisfy the current protocol."
    record_path.write_text(json.dumps(record), encoding="utf-8")
    _eligible_audit(workspace)

    def fake_search(
        candidate,
        period_min=0.5,
        period_max=15.0,
        n_periods=2000,
        sectors=None,
        result_suffix=None,
        duration_grid_hours=None,
    ):
        output = candidate.path / "outputs" / ("bls_search_results" + result_suffix + ".json")
        output.write_text(
            json.dumps({"source": "candidate-data", **_bls_result(snr=4.0)}) + "\n",
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("exonym.survey.run_bls_on_candidate", fake_search)
    monkeypatch.setattr("exonym.survey._run_survey_robustness", _robustness_artifact)

    updated = json.loads(run_survey_search(survey, workspace).read_text(encoding="utf-8"))
    assert updated["status"] == "searched-no-alert"
    assert updated["search_reused"] is False


def test_survey_search_routes_only_to_human_review(tmp_path, monkeypatch):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(
        tmp_path, "survey-target", tic="123456789", mission="tess"
    )
    register_survey_target(survey, workspace)
    _eligible_audit(workspace)
    calls = []

    def fake_search(
        candidate,
        period_min=0.5,
        period_max=15.0,
        n_periods=2000,
        sectors=None,
        result_suffix=None,
        duration_grid_hours=None,
    ):
        calls.append(
            {
                "candidate": candidate.candidate_id,
                "period_min": period_min,
                "period_max": period_max,
                "n_periods": n_periods,
                "sectors": sectors,
                "result_suffix": result_suffix,
                "duration_grid_hours": duration_grid_hours,
            }
        )
        output = candidate.path / "outputs" / ("bls_search_results" + result_suffix + ".json")
        output.write_text(
            json.dumps({"source": "candidate-data", **_bls_result(snr=6.0)}) + "\n",
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("exonym.survey.run_bls_on_candidate", fake_search)
    monkeypatch.setattr("exonym.survey._run_survey_robustness", _robustness_artifact)

    # Act
    record_path = run_survey_search(survey, workspace, review_snr=6.0)

    # Assert
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert calls == [
        {
            "candidate": workspace.candidate_id,
            "period_min": 0.5,
            "period_max": 20.0,
            "n_periods": 200,
            "sectors": [17],
            "result_suffix": ".survey-test-survey",
            "duration_grid_hours": [1.0, 2.0, 4.0],
        }
    ]
    assert record["status"] == "alert-for-human-review"
    assert record["search_result_path"] == "outputs/bls_search_results.survey-test-survey.json"
    assert workspace.metadata["scientific_disposition"] == "unknown"


def test_survey_search_reuses_matching_result_after_interrupted_record_write(tmp_path, monkeypatch):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    register_survey_target(survey, workspace)
    _eligible_audit(workspace)
    outputs = workspace.path / "outputs"
    output = outputs / "bls_search_results.survey-test-survey.json"
    output.write_text(
        json.dumps({"source": "candidate-data", **_bls_result(snr=6.0)}) + "\n",
        encoding="utf-8",
    )
    table = _raw_input_table(workspace)
    product = table["input_files"][0]
    sidecar = product.with_name(product.stem + ".provenance.json")
    (outputs / "bls_search_manifest.survey-test-survey.json").write_text(
        json.dumps(
            {
                "schema": "exonym-bls-search-manifest-1",
                "candidate_id": workspace.candidate_id,
                "source": "candidate-data",
                "result_path": "outputs/bls_search_results.survey-test-survey.json",
                "result_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "detection_status": "no-detection",
                "inputs": [
                    {
                        "path": "data/raw/s0017_lc.fits",
                        "sha256": table["input_sha256s"][0],
                        "provenance_path": "data/raw/s0017_lc.provenance.json",
                        "provenance_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                    }
                ],
                "configuration": {
                    "period_min_days": 0.5,
                    "period_max_days": 20.0,
                    "duration_hours": None,
                    "duration_grid_hours": [1.0, 2.0, 4.0],
                    "n_periods": 200,
                    "n_periods_role": "minimum requested trial density; Astropy baseline-duration grid may use more",
                    "period_grid": "astropy-autopower-baseline-duration-resolved",
                    "max_points": 4000,
                    "quality_filter": "quality == 0 when available",
                    "normalization": "lightkurve.remove_nans().normalize()",
                    "binning": "per-product median binning; no global rebinning",
                    "signal": None,
                    "engine": "bls",
                    "cadence": "median-binned",
                    "use_threads": None,
                    "uncertainty_source": ["reported"],
                    "time_system": "BTJD_TDB",
                    "detection_threshold_snr": 7.1,
                    "sectors": [17],
                },
            }
        ),
        encoding="utf-8",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("matching BLS result should be reused")

    monkeypatch.setattr("exonym.survey.run_bls_on_candidate", fail_if_called)
    monkeypatch.setattr("exonym.survey.load_light_curve_table", lambda *_args, **_kwargs: table)
    monkeypatch.setattr("exonym.survey._run_survey_robustness", _robustness_artifact)

    # Act
    record_path = run_survey_search(survey, workspace, review_snr=6.0)

    # Assert
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "alert-for-human-review"
    assert record["search_reused"] is True


def test_survey_search_requires_robustness_before_human_review(tmp_path, monkeypatch):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    register_survey_target(survey, workspace)
    _eligible_audit(workspace)

    def fake_search(
        candidate,
        period_min=0.5,
        period_max=15.0,
        n_periods=2000,
        sectors=None,
        result_suffix=None,
        duration_grid_hours=None,
    ):
        output = candidate.path / "outputs" / ("bls_search_results" + result_suffix + ".json")
        output.write_text(
            json.dumps({"source": "candidate-data", **_bls_result(snr=9.0)}) + "\n",
            encoding="utf-8",
        )
        return output

    def failed_controls(survey, candidate, search_result, review_snr):
        path, artifact = _robustness_artifact(survey, candidate, search_result, review_snr)
        artifact["diagnostics"]["controls"]["max_snr"] = 9.0
        return path, artifact

    monkeypatch.setattr("exonym.survey.run_bls_on_candidate", fake_search)
    monkeypatch.setattr("exonym.survey._run_survey_robustness", failed_controls)

    # Act
    record_path = run_survey_search(survey, workspace)

    # Assert
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "searched-no-alert"
    assert record["robustness_passed"] is False


def test_survey_search_blocks_weak_candidate_scale_recovery(tmp_path, monkeypatch):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    register_survey_target(survey, workspace)
    _eligible_audit(workspace)

    def fake_search(
        candidate,
        period_min=0.5,
        period_max=15.0,
        n_periods=2000,
        sectors=None,
        result_suffix=None,
        duration_grid_hours=None,
    ):
        output = candidate.path / "outputs" / ("bls_search_results" + result_suffix + ".json")
        output.write_text(
            json.dumps({"source": "candidate-data", **_bls_result(snr=9.0)}) + "\n",
            encoding="utf-8",
        )
        return output

    def weak_recovery(survey, candidate, search_result, review_snr):
        return _robustness_artifact(
            survey, candidate, search_result, review_snr, recovered_trials=(True, False, False)
        )

    monkeypatch.setattr("exonym.survey.run_bls_on_candidate", fake_search)
    monkeypatch.setattr("exonym.survey._run_survey_robustness", weak_recovery)

    # Act
    record_path = run_survey_search(survey, workspace)

    # Assert
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "searched-no-alert"
    assert record["robustness_passed"] is False


def test_robustness_requires_injections_at_the_bls_signal_scale(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    _, artifact = _robustness_artifact(
        survey, workspace, _bls_result(snr=8.0), review_snr=6.0
    )
    artifact["injection_recovery"][0]["injection"]["depth_ppm"] = 2000.0

    # Act and assert
    assert not _robustness_passes(artifact, 6.0)


def test_robustness_requires_detrending_results_at_the_bls_period(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    _, artifact = _robustness_artifact(
        survey, workspace, _bls_result(snr=8.0), review_snr=6.0
    )
    artifact["diagnostics"]["variants"]["normalized"]["best"]["best_period"] = 4.0
    artifact["diagnostics"]["variants"]["running-median"]["best"]["best_period"] = 4.0

    # Act and assert
    assert not _robustness_passes(artifact, 6.0)


def test_robustness_requires_multiple_distinct_events_in_every_signal_search(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    _, artifact = _robustness_artifact(
        survey, workspace, _bls_result(snr=8.0), review_snr=6.0
    )
    artifact["diagnostics"]["variants"]["normalized"]["best"]["n_distinct_transit_events"] = 1

    # Act and assert
    assert not _robustness_passes(artifact, 6.0)


def test_robustness_rejects_a_relaxed_recovery_policy(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    _, artifact = _robustness_artifact(
        survey, workspace, _bls_result(snr=8.0), review_snr=6.0
    )
    policy = artifact["configuration"]["injection_recovery"]
    policy["minimum_recovered_trials"] = 1
    policy["minimum_recovery_fraction"] = 1.0 / 3.0
    summary = artifact["injection_recovery_summary"]
    summary["minimum_recovered_trials"] = 1
    summary["minimum_recovery_fraction"] = 1.0 / 3.0

    # Act and assert
    assert not _robustness_passes(artifact, 6.0)


def test_survey_search_rejects_audit_that_the_workflow_gate_rejects(tmp_path, monkeypatch):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    register_survey_target(survey, workspace)
    _eligible_audit(workspace)
    audit_path = workspace.path / "decisions" / "novelty_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["evidence"] = []
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("invalid novelty audit must block BLS")

    monkeypatch.setattr("exonym.survey.run_bls_on_candidate", fail_if_called)

    # Act
    record_path = run_survey_search(survey, workspace, review_snr=6.0)

    # Assert
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "blocked-novelty-audit"
    assert record["search_reused"] is False


def test_survey_search_rejects_a_changed_review_threshold(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17], review_snr=6.0)
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    register_survey_target(survey, workspace)

    # Act and assert
    with pytest.raises(ValueError, match="preregistered"):
        run_survey_search(survey, workspace, review_snr=7.0)


def test_survey_exclusion_preserves_the_registered_target(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(
        tmp_path, "survey-target", tic="123456789", mission="tess"
    )
    register_survey_target(survey, workspace)

    # Act
    record_path = exclude_survey_target(survey, workspace.candidate_id, "Synthetic exclusion.")

    # Assert
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "excluded-before-search"
    assert record["reason"] == "Synthetic exclusion."
    assert survey_summary(survey)["outcome_counts"] == {"excluded-before-search": 1}


@pytest.mark.parametrize("reason", ["", "   "])
def test_survey_exclusion_requires_a_reason(tmp_path, reason):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    register_survey_target(survey, workspace)

    # Act and assert
    with pytest.raises(ValueError, match="reason"):
        exclude_survey_target(survey, workspace.candidate_id, reason)


def test_survey_exclusion_rejects_unregistered_and_unsafe_target_ids(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])

    # Act and assert
    with pytest.raises(FileNotFoundError):
        exclude_survey_target(survey, "survey-target", "Synthetic exclusion.")
    with pytest.raises(ValueError):
        exclude_survey_target(survey, "../outside", "Synthetic exclusion.")


def test_empty_survey_summary_is_explicit(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])

    # Act and assert
    summary = survey_summary(survey)
    assert summary["survey"] == survey.metadata
    assert summary["outcome_counts"] == {}
    assert summary["targets"] == []


def test_survey_summary_retains_an_invalid_target_record(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    invalid = survey.path / "targets" / "survey-target" / "target.json"
    invalid.parent.mkdir(parents=True)
    invalid.write_text("not json", encoding="utf-8")

    # Act
    summary = survey_summary(survey)

    # Assert
    assert summary["outcome_counts"] == {"invalid-record": 1}
    assert summary["targets"][0]["candidate_id"] == "survey-target"


def test_survey_summary_marks_a_missing_candidate_workspace(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    register_survey_target(survey, workspace)
    (workspace.path / "candidate.json").unlink()

    # Act
    summary = survey_summary(survey)

    # Assert
    assert summary["outcome_counts"] == {"orphaned-candidate-record": 1}


def test_survey_rejects_target_record_with_mismatched_identity(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    workspace = create_candidate(tmp_path, "survey-target", tic="123456789", mission="tess")
    record_path = register_survey_target(survey, workspace)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["survey_id"] = "other-survey"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    # Act and assert
    with pytest.raises(ValueError, match="does not match"):
        exclude_survey_target(survey, workspace.candidate_id, "Synthetic exclusion.")


@pytest.mark.parametrize("value", ["", "bad_value", "bad.value"])
def test_validate_survey_id_rejects_unsafe_or_ambiguous_values(value):
    with pytest.raises(ValueError):
        validate_survey_id(value)


def test_validate_survey_id_normalizes_case():
    assert validate_survey_id("UPPER") == "upper"


@pytest.mark.parametrize("field, value", [("schema_version", 2), ("mission", "kepler")])
def test_load_survey_rejects_invalid_metadata(tmp_path, field, value):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    path = survey.path / "survey.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[field] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")

    # Act and assert
    with pytest.raises(ValueError):
        load_survey(tmp_path, survey.survey_id)


def test_load_survey_rejects_invalid_json(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "test-survey", "tess", [17])
    (survey.path / "survey.json").write_text("not json", encoding="utf-8")

    # Act and assert
    with pytest.raises(ValueError, match="invalid survey metadata"):
        load_survey(tmp_path, survey.survey_id)
