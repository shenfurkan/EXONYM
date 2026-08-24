"""Synthetic unit tests for engine execution, run manifests, and automated triage."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import lightkurve as lk

from exonym import __version__
import exonym.engines as engines_module
from exonym.__main__ import main
from exonym.engines import report_candidate_engines, run_automated_triage, run_engine
from exonym.isolation import IsolationReport
from exonym.schemas import validate_schemas
from exonym.workspace import create_candidate, load_candidate


def _write_synthetic_lc_fits(path: Path, sector: int = 1, period: float = 2.5, depth_ppm: float = 1200.0) -> None:
    """Write a synthetic light curve FITS file with a clean transit box."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    time = np.arange(100.0, 125.0, 2.0 / 1440.0)
    flux = 1.0 + rng.normal(0.0, 100e-6, time.size)
    phase = ((time - 100.5) % period) / period
    in_tr = (phase < 0.02) | (phase > 0.98)
    flux[in_tr] -= depth_ppm * 1e-6
    err = np.full_like(time, 100e-6)

    meta = {
        "MISSION": "TESS",
        "TELESCOP": "TESS",
        "SECTOR": sector,
        "TIMEDEL": 120.0 / 86400.0,
        "TIMEUNIT": "BJD",
        "BJDREFI": 2457000,
        "BJDREFF": 0.0,
    }
    lk.LightCurve(time=time, flux=flux, flux_err=err, meta=meta).to_fits(
        path=path, overwrite=True
    )
    sidecar = path.with_name(path.stem + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "source_uri": "https://archive.example.invalid/" + path.name,
                "download_timestamp_utc": "2026-01-01T00:00:00Z",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fetched_by": "synthetic-test",
            }
        ),
        encoding="utf-8",
    )


def _setup_synthetic_workspace(tmp_path: Path, candidate_id: str = "synth-triage-target") -> Path:
    """Create a fully populated synthetic candidate workspace with light curve and ephemeris."""
    workspace = create_candidate(
        repository_root=tmp_path,
        candidate_id=candidate_id,
        tic="987654321",
        mission="tess",
    )
    cand_path = workspace.path

    # 1. Ephemeris config
    ephem_payload = {
        "period_days": 2.5,
        "epoch_btjd": 100.5,
        "duration_days": 0.1,
        "depth_ppm": 1200.0,
    }
    (cand_path / "config" / "transit_config.json").write_text(
        json.dumps(ephem_payload, indent=2), encoding="utf-8"
    )

    # 2. Synthetic light curve FITS
    _write_synthetic_lc_fits(cand_path / "data" / "raw" / "s0001_lc.fits", sector=1)

    return cand_path


def _write_passing_pre_vetting_artifacts(
    candidate_path: Path, candidate_id: str, include_search: bool = True
) -> None:
    """Write minimal candidate-data diagnostic reports for routing tests."""
    outputs = candidate_path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "fixed_ephemeris_screen.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "source": "candidate-data",
                "ephemeris": {
                    "period_days": 2.5,
                    "epoch_btjd": 100.5,
                    "epoch_time_system": "BTJD_TDB",
                    "duration_hours": 2.4,
                    "source": "candidate-config",
                },
                "screen": {
                    "primary": {
                        "status": "measured",
                        "depth_ppm": 1200.0,
                        "depth_significance_sigma": 8.0,
                    },
                    "odd_even": {
                        "status": "measured",
                        "z": 0.2,
                        "consistent_at_threshold": True,
                        "consistency_threshold_sigma": 3.0,
                    },
                    "half_phase_control": {
                        "status": "measured",
                        "depth_significance_sigma": 0.1,
                    },
                    "double_period_hypothesis": {
                        "primary": {"status": "measured"},
                        "alternating_event": {
                            "status": "measured",
                            "depth_significance_sigma": 0.1,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs / "archival_vetting_report.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "gaia_astrometry": {
                    "validated": True,
                    "query_status": "ok",
                    "ruwe": 1.0,
                    "suspected_binary": False,
                    "search_radius_arcsec": 60.0,
                    "nearby_sources_count": 1,
                },
                "scientific_assessment": {
                    "1_is_hidden_binary": {"answer": False, "ruwe": 1.0},
                    "2_has_nearby_contaminants": {
                        "answer": False,
                        "search_radius_arcsec": 60.0,
                        "search_radius_sufficient_for_crowding": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs / "prf_localization_results.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "source": "candidate-data",
                "calibration_status": "uncalibrated",
                "summary": {
                    "conclusion": "inconclusive_uncalibrated_prf",
                    "median_target_to_other_difference_ratio": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs / "stellar_activity_results.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "work_package": "STELLAR_ACTIVITY",
                "generated_utc": "2026-08-01T00:00:00Z",
                "source": "candidate-data",
                "scientific_status": "exploratory-activity-diagnostic",
                "validation_eligible": False,
                "transit_mask_status": "applied-candidate-ephemeris",
                "method": "synthetic test diagnostic",
                "period_search_range_days": [0.5, 20.0],
                "rotation_period_days": 7.0,
                "rotation_period_std_days": 0.2,
                "modulation_amplitude_ppm": 100.0,
                "best_analytic_white_noise_false_alarm_probability": 0.1,
                "n_segments": 1,
                "segments": [
                    {
                        "sector": 1,
                        "n_points": 100,
                        "baseline_days": 20.0,
                        "best_period_days": 7.0,
                        "max_power": 0.2,
                        "analytic_white_noise_false_alarm_probability": 0.1,
                        "sampling_window": {
                            "method": "normalized-spectral-window-v1",
                            "baseline_days": 20.0,
                            "frequency_resolution_days_inverse": 0.05,
                            "top_window_peaks": [
                                {"frequency_days_inverse": 0.1, "period_days": 10.0, "window_power": 0.2}
                            ],
                            "nearest_window_peak": None,
                            "direct_frequency_proximity_within_resolution": False,
                            "interpretation": "Synthetic test-only window diagnostic."
                        }
                    }
                ],
                "harmonic_persistence": {
                    "status": "unresolved-insufficient-segments",
                    "reference_frequency_days_inverse": None,
                    "reference_period_days": None,
                    "segments": [],
                    "consistent_segment_count": 0,
                    "interpretation": "One segment cannot establish persistence."
                },
                "caveat": "Synthetic test-only activity diagnostic."
            }
        ),
        encoding="utf-8",
    )
    (outputs / "dilution_sensitivity_results.json").write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "depth_stability": {"interpretation": "stable", "max_variation_relative_to_median": 0.1},
                "contamination": {"availability": "available", "contamination_factor": 0.01},
            }
        ),
        encoding="utf-8",
    )
    if include_search:
        _write_remaining_real_data_prerequisites(
            candidate_path, include_analysis_outputs=False
        )


def _write_remaining_real_data_prerequisites(
    candidate_path: Path, include_analysis_outputs: bool = True
) -> None:
    """Complete the provenance-only prerequisites for readiness-order tests."""
    outputs = candidate_path / "outputs"
    input_path = candidate_path / "data" / "raw" / "s0001_lc.fits"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        input_path.write_bytes(b"synthetic-raw-photometry")
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    provenance_path = input_path.with_name(input_path.stem + ".provenance.json")
    if not provenance_path.exists():
        provenance_path.write_text(
            json.dumps(
                {
                    "source_uri": "https://archive.example.invalid/s0001_lc.fits",
                    "download_timestamp_utc": "2026-01-01T00:00:00Z",
                    "sha256": input_sha256,
                    "fetched_by": "synthetic-test",
                }
            ),
            encoding="utf-8",
        )
    result_path = outputs / "bls_search_results.json"
    result_path.write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "detection_status": "detected",
                "time_system": "BTJD_TDB",
                "best_period": 2.5,
                "best_epoch": 100.5,
                "best_duration_hours": 2.4,
                "best_depth_ppm": 1200.0,
                "snr": 12.0,
                "n_distinct_transit_events": 3,
                "detection_threshold_snr": 7.1,
            }
        ),
        encoding="utf-8",
    )
    (outputs / "bls_search_manifest.json").write_text(
        json.dumps(
            {
                "schema": "exonym-bls-search-manifest-1",
                "candidate_id": candidate_path.name,
                "result_path": "outputs/bls_search_results.json",
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "source": "candidate-data",
                "detection_status": "detected",
                "inputs": [
                    {
                        "path": "data/raw/s0001_lc.fits",
                        "sha256": input_sha256,
                        "provenance_path": "data/raw/s0001_lc.provenance.json",
                        "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
                    }
                ],
                "configuration": {
                    "engine": "bls",
                    "signal": None,
                    "time_system": "BTJD_TDB",
                    "detection_threshold_snr": 7.1,
                },
            }
        ),
        encoding="utf-8",
    )
    if include_analysis_outputs:
        for filename in (
            "asteroseismic_results.json",
            "sed_fit_results.json",
            "mcmc_transit_fit.json",
            "ttv_analysis_results.json",
            "phase_curve_results.json",
        ):
            (outputs / filename).write_text(
                json.dumps({"source": "candidate-data"}), encoding="utf-8"
            )


def test_engine_run_generates_valid_schema_manifest(tmp_path: Path):
    _setup_synthetic_workspace(tmp_path, "synth-engine-manifest")
    cand = load_candidate(tmp_path, "synth-engine-manifest")

    manifest_path = run_engine(cand, "screen")
    assert manifest_path.is_file()

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["schema_version"] == 1
    assert manifest_data["candidate_id"] == "synth-engine-manifest"
    assert manifest_data["engine"] == "screen"
    assert manifest_data["status"] == "succeeded"
    assert len(manifest_data["inputs"]) >= 1
    assert len(manifest_data["outputs"]) >= 1
    assert isinstance(manifest_data["runtime"]["version"], str)
    assert manifest_data["runtime"]["version_known"] is True

    # Verify that schemas.py validates the workspace with 0 violations
    report = IsolationReport()
    validate_schemas(tmp_path, report)
    assert report.ok, f"Schema violations: {[v.detail for v in report.violations]}"


def test_engine_run_records_unknown_dependency_version_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Arrange
    candidate = create_candidate(tmp_path, "synth-unversioned-engine")
    output = candidate.path / "outputs" / "screen.json"
    output.write_text("{}\n", encoding="utf-8")
    unversioned = engines_module.EngineStatus(
        name="screen",
        capability="screening",
        optional_group="core",
        module_name="synthetic_unversioned",
        description="Synthetic unversioned engine.",
        installed=True,
        version=None,
    )
    monkeypatch.setattr(engines_module, "get_engine", lambda _name: unversioned)
    monkeypatch.setattr(
        "exonym.screening.run_fixed_ephemeris_screen",
        lambda workspace, signal=None: output,
    )

    # Act
    manifest_path = engines_module.run_engine(candidate, "screen")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Assert
    assert manifest["runtime"] == {
        "kind": "direct",
        "version": None,
        "version_known": False,
        "executable": "synthetic_unversioned",
    }
    report = IsolationReport()
    validate_schemas(tmp_path, report)
    assert report.ok, f"Schema violations: {[v.detail for v in report.violations]}"


def test_engine_report_lists_completed_runs(tmp_path: Path):
    _setup_synthetic_workspace(tmp_path, "synth-report-candidate")
    cand = load_candidate(tmp_path, "synth-report-candidate")

    run_engine(cand, "screen")
    runs = report_candidate_engines(cand)
    assert len(runs) == 1
    assert runs[0]["engine"] == "screen"
    assert runs[0]["status"] == "succeeded"


def test_automated_triage_requires_review_for_uncalibrated_activity(tmp_path: Path):
    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-triage-pass")
    cand = load_candidate(tmp_path, "synth-triage-pass")

    _write_passing_pre_vetting_artifacts(candidate_path, cand.candidate_id)
    triage_path = run_automated_triage(cand)
    assert triage_path.is_file()

    triage_data = json.loads(triage_path.read_text(encoding="utf-8"))
    assert triage_data["schema_version"] == 1
    assert triage_data["candidate_id"] == "synth-triage-pass"
    assert triage_data["status"] == "review-required"
    assert len(triage_data["records"]) == 5
    evidence = json.loads(
        (candidate_path / "outputs" / "statistical_vetting_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["status"] == "review-required"
    assert {item["name"] for item in evidence["diagnostics"]} == {
        "screening", "archive", "localization", "activity", "dilution"
    }
    statistical_manifest = next(
        (candidate_path / "runs" / "statistical-vetting").glob("*/engine-run.json")
    )
    statistical_runtime = json.loads(
        statistical_manifest.read_text(encoding="utf-8")
    )["runtime"]
    assert statistical_runtime == {
        "kind": "direct",
        "version": __version__,
        "version_known": True,
        "executable": "exonym.statistical_vetting",
    }

    report = IsolationReport()
    validate_schemas(tmp_path, report)
    assert report.ok, f"Schema violations: {[v.detail for v in report.violations]}"


def test_statistical_vetting_routes_present_rv_data_through_triage(tmp_path: Path):
    """A real candidate-local RV report becomes an explicit triage diagnostic."""
    from exonym.radial_velocity import (
        fit_radial_velocity,
        ingest_radial_velocity_observations,
        keplerian_velocity_m_per_s,
    )
    from exonym.statistical_vetting import build_statistical_vetting_evidence

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-rv-triage")
    candidate = load_candidate(tmp_path, "synth-rv-triage")
    _write_passing_pre_vetting_artifacts(candidate_path, candidate.candidate_id)

    period_days = 2.5
    reference_time = 100.5
    times = np.linspace(reference_time - 10.0, reference_time + 10.0, 48)
    velocities = keplerian_velocity_m_per_s(
        times,
        semi_amplitude_m_per_s=10.0,
        mean_anomaly_reference_rad=0.0,
        eccentricity=0.0,
        argument_periastron_rad=0.0,
        reference_time_bjd_tdb=reference_time,
        period_days=period_days,
    )
    source = tmp_path / "synthetic-rv-triage.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id": candidate.candidate_id,
                "observations": [
                    {
                        "observation_time": {"value": float(time), "unit": "BJD_TDB"},
                        "velocity": {"value": float(velocity), "unit": "m/s"},
                        "uncertainty": {"value": 0.5, "unit": "m/s"},
                        "instrument": "synthetic-spectrograph",
                        "provenance": {
                            "source_uri": "https://example.invalid/synthetic-rv",
                            "retrieved_at_utc": "2000-01-01T00:00:00Z",
                            "record_label": "synthetic-rv-{0}".format(index),
                        },
                    }
                    for index, (time, velocity) in enumerate(zip(times, velocities))
                ],
            }
        ),
        encoding="utf-8",
    )
    ingest_radial_velocity_observations(candidate, source)
    report_path = fit_radial_velocity(candidate, period_days)
    assert report_path.is_file()

    evidence = json.loads(build_statistical_vetting_evidence(candidate).read_text(encoding="utf-8"))
    rv = next(record for record in evidence["diagnostics"] if record["name"] == "radial-velocity")
    assert rv["status"] == "review-required"
    assert rv["score"]["value"] >= 10.0
    assert evidence["status"] == "review-required"

    triage = json.loads(run_automated_triage(candidate).read_text(encoding="utf-8"))
    assert len(triage["records"]) == 6
    assert any("radial-velocity:" in record["reason"] for record in triage["records"])
    audit = IsolationReport()
    validate_schemas(tmp_path, audit)
    assert audit.ok, f"Schema violations: {[violation.detail for violation in audit.violations]}"


def test_automated_triage_review_required_on_odd_even_anomaly(tmp_path: Path):
    cand_path = _setup_synthetic_workspace(tmp_path, "synth-triage-anomaly")
    cand = load_candidate(tmp_path, "synth-triage-anomaly")
    manifest_path = run_engine(cand, "screen")
    _write_passing_pre_vetting_artifacts(cand_path, cand.candidate_id)

    # Manually plant an odd-even failure in screening outputs
    outputs_dir = cand_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    screen_path = cand_path / "outputs" / "fixed_ephemeris_screen.json"
    screen_artifact = json.loads(screen_path.read_text(encoding="utf-8"))
    screen_artifact["screen"]["odd_even"].update(
        {
            "status": "measured",
            "z": 4.5,
            "consistent_at_threshold": False,
            "consistency_threshold_sigma": 3.0,
        }
    )
    screen_path.write_text(
        json.dumps(screen_artifact, indent=2), encoding="utf-8"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][0]["sha256"] = hashlib.sha256(screen_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    triage_path = run_automated_triage(cand)
    triage_data = json.loads(triage_path.read_text(encoding="utf-8"))
    assert triage_data["status"] == "review-required"
    assert any(r["status"] == "review-required" for r in triage_data["records"])

    report = IsolationReport()
    validate_schemas(tmp_path, report)
    assert report.ok, f"Schema violations: {[v.detail for v in report.violations]}"


def test_statistical_vetting_requires_review_for_alternating_events(tmp_path: Path):
    from exonym.statistical_vetting import build_statistical_vetting_evidence

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-alternating-events")
    candidate = load_candidate(tmp_path, "synth-alternating-events")
    _write_passing_pre_vetting_artifacts(candidate_path, candidate.candidate_id)
    screen_path = candidate_path / "outputs" / "fixed_ephemeris_screen.json"
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    screen["screen"]["double_period_hypothesis"]["alternating_event"]["depth_significance_sigma"] = 3.1
    screen_path.write_text(json.dumps(screen), encoding="utf-8")

    evidence = json.loads(build_statistical_vetting_evidence(candidate).read_text(encoding="utf-8"))

    screening = next(record for record in evidence["diagnostics"] if record["name"] == "screening")
    assert screening["status"] == "review-required"

    screen["screen"]["half_phase_control"]["depth_significance_sigma"] = 0.1
    screen["screen"]["double_period_hypothesis"]["alternating_event"][
        "depth_significance_sigma"
    ] = -3.1
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    evidence = json.loads(build_statistical_vetting_evidence(candidate).read_text(encoding="utf-8"))
    screening = next(record for record in evidence["diagnostics"] if record["name"] == "screening")
    assert screening["status"] == "review-required"
    assert evidence["status"] == "review-required"


def test_statistical_vetting_requires_review_for_signed_secondary_controls(tmp_path: Path):
    from exonym.statistical_vetting import build_statistical_vetting_evidence

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-signed-secondary-controls")
    candidate = load_candidate(tmp_path, "synth-signed-secondary-controls")
    _write_passing_pre_vetting_artifacts(candidate_path, candidate.candidate_id)
    screen_path = candidate_path / "outputs" / "fixed_ephemeris_screen.json"
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    screen["screen"]["half_phase_control"]["depth_significance_sigma"] = -3.1
    screen_path.write_text(json.dumps(screen), encoding="utf-8")

    evidence = json.loads(build_statistical_vetting_evidence(candidate).read_text(encoding="utf-8"))

    screening = next(record for record in evidence["diagnostics"] if record["name"] == "screening")
    assert screening["status"] == "review-required"


def test_statistical_vetting_requires_review_for_inconclusive_localization(tmp_path: Path):
    from exonym.statistical_vetting import build_statistical_vetting_evidence

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-inconclusive-localization")
    candidate = load_candidate(tmp_path, "synth-inconclusive-localization")
    _write_passing_pre_vetting_artifacts(candidate_path, candidate.candidate_id)
    localization_path = candidate_path / "outputs" / "prf_localization_results.json"
    localization = json.loads(localization_path.read_text(encoding="utf-8"))
    localization["summary"].update(
        {
            "conclusion": "inconclusive_no_competing_sources_modeled",
            "median_target_to_other_difference_ratio": None,
            "sectors_with_competing_sources_modeled": 0,
        }
    )
    localization_path.write_text(json.dumps(localization), encoding="utf-8")

    evidence = json.loads(
        build_statistical_vetting_evidence(candidate).read_text(encoding="utf-8")
    )

    record = next(
        item for item in evidence["diagnostics"] if item["name"] == "localization"
    )
    assert record["status"] == "review-required"
    assert "modeled no competing sources" in record["reason"]
    assert evidence["status"] == "review-required"


@pytest.mark.parametrize(
    ("ratio", "reason_fragment"),
    [
        (2.0, "target-favored"),
        (1.0, "competitor-favored"),
        (0.5, "competitor-favored"),
    ],
)
def test_statistical_vetting_distinguishes_uncalibrated_localization_ratio_direction(
    tmp_path: Path, ratio: float, reason_fragment: str
):
    from exonym.statistical_vetting import build_statistical_vetting_evidence

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-localization-ratio")
    candidate = load_candidate(tmp_path, "synth-localization-ratio")
    _write_passing_pre_vetting_artifacts(candidate_path, candidate.candidate_id)
    localization_path = candidate_path / "outputs" / "prf_localization_results.json"
    localization = json.loads(localization_path.read_text(encoding="utf-8"))
    localization["summary"].update(
        {
            "median_target_to_other_difference_ratio": ratio,
            "sectors_with_competing_sources_modeled": 1,
        }
    )
    localization_path.write_text(json.dumps(localization), encoding="utf-8")

    evidence = json.loads(
        build_statistical_vetting_evidence(candidate).read_text(encoding="utf-8")
    )

    record = next(
        item for item in evidence["diagnostics"] if item["name"] == "localization"
    )
    assert record["status"] == "review-required"
    assert reason_fragment in record["reason"]
    assert evidence["status"] == "review-required"


def test_statistical_vetting_blocks_missing_localization_ratio_with_modeled_competitor(
    tmp_path: Path,
):
    from exonym.statistical_vetting import build_statistical_vetting_evidence

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-localization-missing-ratio")
    candidate = load_candidate(tmp_path, "synth-localization-missing-ratio")
    _write_passing_pre_vetting_artifacts(candidate_path, candidate.candidate_id)
    localization_path = candidate_path / "outputs" / "prf_localization_results.json"
    localization = json.loads(localization_path.read_text(encoding="utf-8"))
    localization["summary"].update(
        {
            "median_target_to_other_difference_ratio": None,
            "sectors_with_competing_sources_modeled": 1,
        }
    )
    localization_path.write_text(json.dumps(localization), encoding="utf-8")

    evidence = json.loads(
        build_statistical_vetting_evidence(candidate).read_text(encoding="utf-8")
    )

    record = next(
        item for item in evidence["diagnostics"] if item["name"] == "localization"
    )
    assert record["status"] == "blocked"
    assert "modeled competing sources" in record["reason"]
    assert evidence["status"] == "blocked"


def test_statistical_vetting_blocks_self_declared_localization_calibration(tmp_path: Path):
    from exonym.statistical_vetting import build_statistical_vetting_evidence

    # Arrange
    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-forged-localization-calibration")
    candidate = load_candidate(tmp_path, "synth-forged-localization-calibration")
    _write_passing_pre_vetting_artifacts(candidate_path, candidate.candidate_id)
    localization_path = candidate_path / "outputs" / "prf_localization_results.json"
    localization = json.loads(localization_path.read_text(encoding="utf-8"))
    localization["calibration_status"] = "calibrated"
    localization["summary"]["conclusion"] = "target_dominant_among_modeled_sources"
    localization_path.write_text(json.dumps(localization), encoding="utf-8")

    # Act
    evidence = json.loads(build_statistical_vetting_evidence(candidate).read_text(encoding="utf-8"))

    # Assert
    record = next(item for item in evidence["diagnostics"] if item["name"] == "localization")
    assert record["status"] == "blocked"
    assert evidence["status"] == "blocked"


def test_statistical_vetting_requires_review_for_aperture_sensitive_depths(tmp_path: Path):
    from exonym.statistical_vetting import build_statistical_vetting_evidence

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-aperture-sensitive")
    candidate = load_candidate(tmp_path, "synth-aperture-sensitive")
    _write_passing_pre_vetting_artifacts(candidate_path, candidate.candidate_id)
    dilution_path = candidate_path / "outputs" / "dilution_sensitivity_results.json"
    dilution = json.loads(dilution_path.read_text(encoding="utf-8"))
    dilution["depth_stability"]["interpretation"] = "aperture-sensitive"
    dilution["contamination"]["contamination_factor"] = 0.0
    dilution_path.write_text(json.dumps(dilution), encoding="utf-8")

    evidence = json.loads(build_statistical_vetting_evidence(candidate).read_text(encoding="utf-8"))

    record = next(item for item in evidence["diagnostics"] if item["name"] == "dilution")
    assert record["status"] == "review-required"
    assert evidence["status"] == "review-required"


def test_cli_engine_run_and_triage(tmp_path: Path, capsys: pytest.CaptureFixture):
    _setup_synthetic_workspace(tmp_path, "synth-cli-candidate")

    # Test 'exonym engine run'
    rc_run = main(["--root", str(tmp_path), "engine", "run", "screen", "synth-cli-candidate"])
    assert rc_run == 0
    captured_run = capsys.readouterr().out
    assert "engine-run.json" in captured_run

    # Test 'exonym engine report'
    rc_rep = main(["--root", str(tmp_path), "engine", "report", "synth-cli-candidate"])
    assert rc_rep == 0
    captured_rep = capsys.readouterr().out
    assert "screen" in captured_rep

    # Test 'exonym triage'
    rc_tri = main(["--root", str(tmp_path), "triage", "synth-cli-candidate"])
    assert rc_tri == 0
    captured_tri = capsys.readouterr().out
    assert "automated_triage.json" in captured_tri


def test_engine_run_failure_is_recorded_and_returns_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture):
    create_candidate(tmp_path, "synth-engine-failure", tic="123456789", mission="tess")

    rc = main(["--root", str(tmp_path), "engine", "run", "screen", "synth-engine-failure"])

    assert rc == 1
    output = capsys.readouterr().out.strip()
    manifest_path = tmp_path / output
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"]["code"]


def test_engine_run_with_no_output_is_blocked_without_scientific_artifacts(tmp_path: Path, monkeypatch):
    _setup_synthetic_workspace(tmp_path, "synth-engine-no-output")
    cand = load_candidate(tmp_path, "synth-engine-no-output")
    monkeypatch.setattr("exonym.screening.run_fixed_ephemeris_screen", lambda *args, **kwargs: None)

    manifest_path = run_engine(cand, "screen")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["outputs"] == []
    assert manifest["failure"]["code"] == "no-output-artifacts"
    assert not (cand.path / "outputs" / "fixed_ephemeris_screen.json").exists()
    assert not list((cand.path / "claims").glob("*.json"))


def test_automated_triage_blocks_without_manifest_backed_evidence(tmp_path: Path):
    _setup_synthetic_workspace(tmp_path, "synth-triage-blocked")
    cand = load_candidate(tmp_path, "synth-triage-blocked")

    triage_path = run_automated_triage(cand)

    triage_data = json.loads(triage_path.read_text(encoding="utf-8"))
    assert triage_data["status"] == "blocked"
    assert triage_data["records"][0]["status"] == "blocked"


def test_activity_harmonic_requires_human_review(tmp_path: Path):
    from exonym.statistical_vetting import build_statistical_vetting_evidence

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-activity-harmonic")
    cand = load_candidate(tmp_path, "synth-activity-harmonic")
    _write_passing_pre_vetting_artifacts(candidate_path, cand.candidate_id)
    (candidate_path / "outputs" / "stellar_activity_results.json").write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "rotation_period_days": 5.0,
                "rotation_period_std_days": 0.2,
            }
        ),
        encoding="utf-8",
    )

    evidence = json.loads(build_statistical_vetting_evidence(cand).read_text(encoding="utf-8"))

    activity = next(record for record in evidence["diagnostics"] if record["name"] == "activity")
    assert activity["status"] == "review-required"
    assert evidence["status"] == "review-required"


def test_statistical_vetting_blocks_activity_alias_triage_without_screening_ephemeris(
    tmp_path: Path,
):
    from exonym.statistical_vetting import build_statistical_vetting_evidence

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-activity-no-ephemeris")
    candidate = load_candidate(tmp_path, "synth-activity-no-ephemeris")
    _write_passing_pre_vetting_artifacts(candidate_path, candidate.candidate_id)
    screen_path = candidate_path / "outputs" / "fixed_ephemeris_screen.json"
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    screen.pop("ephemeris")
    screen_path.write_text(json.dumps(screen), encoding="utf-8")

    evidence = json.loads(
        build_statistical_vetting_evidence(candidate).read_text(encoding="utf-8")
    )

    activity = next(
        record for record in evidence["diagnostics"] if record["name"] == "activity"
    )
    assert activity["status"] == "blocked"
    assert "requires a finite screening ephemeris transit period" in activity["reason"]
    assert evidence["status"] == "blocked"


def test_vetting_readiness_refuses_review_required_evidence_without_claims(tmp_path: Path):
    from exonym.statistical_vetting import require_vetting_readiness

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-vet-review")
    cand = load_candidate(tmp_path, "synth-vet-review")
    run_engine(cand, "screen")
    _write_passing_pre_vetting_artifacts(candidate_path, cand.candidate_id)
    _write_remaining_real_data_prerequisites(candidate_path)
    localization = candidate_path / "outputs" / "prf_localization_results.json"
    localization.write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "calibration_status": "uncalibrated",
                "summary": {"conclusion": "inconclusive_g_band_flux_bound_exceeded"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="current routing is review-required"):
        require_vetting_readiness(cand)

    triage = json.loads((candidate_path / "decisions" / "automated_triage.json").read_text(encoding="utf-8"))
    assert triage["status"] == "review-required"
    assert not list((candidate_path / "claims").glob("*.json"))


def test_vetting_readiness_requires_all_real_data_prerequisites(tmp_path: Path):
    from exonym.statistical_vetting import require_vetting_readiness

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-vet-prerequisites")
    cand = load_candidate(tmp_path, "synth-vet-prerequisites")
    _write_passing_pre_vetting_artifacts(
        candidate_path, cand.candidate_id, include_search=False
    )

    with pytest.raises(RuntimeError, match="real candidate-data prerequisite outputs: search"):
        require_vetting_readiness(cand)


def test_vetting_readiness_rejects_a_non_detection_bls_artifact(tmp_path: Path):
    from exonym.statistical_vetting import require_vetting_readiness

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-vet-no-detection")
    cand = load_candidate(tmp_path, "synth-vet-no-detection")
    _write_passing_pre_vetting_artifacts(candidate_path, cand.candidate_id)
    _write_remaining_real_data_prerequisites(candidate_path)
    result_path = candidate_path / "outputs" / "bls_search_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["detection_status"] = "no-detection"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(RuntimeError, match="detected hash-bound BLS search"):
        require_vetting_readiness(cand)


def test_vetting_readiness_rejects_a_subthreshold_bls_artifact(tmp_path: Path):
    from exonym.statistical_vetting import require_vetting_readiness

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-vet-low-snr")
    cand = load_candidate(tmp_path, "synth-vet-low-snr")
    _write_passing_pre_vetting_artifacts(candidate_path, cand.candidate_id)
    _write_remaining_real_data_prerequisites(candidate_path)
    result_path = candidate_path / "outputs" / "bls_search_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["snr"] = 6.0
    result_path.write_text(json.dumps(result), encoding="utf-8")
    manifest_path = candidate_path / "outputs" / "bls_search_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="detected hash-bound BLS search"):
        require_vetting_readiness(cand)


def test_decisive_rejection_prohibits_triceratops(tmp_path: Path):
    from exonym.statistical_vetting import record_decisive_rejection, require_vetting_readiness

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-vet-rejection")
    cand = load_candidate(tmp_path, "synth-vet-rejection")
    evidence = candidate_path / "outputs" / "decisive.json"
    evidence.write_text('{"result": "synthetic rejection"}\n', encoding="utf-8")
    record_decisive_rejection(cand, "Synthetic decisive alias.", "outputs/decisive.json")

    with pytest.raises(RuntimeError, match="decisive rejection"):
        require_vetting_readiness(cand)


def test_decisive_rejection_rejects_evidence_outside_the_candidate_workspace(tmp_path: Path):
    from exonym.statistical_vetting import record_decisive_rejection

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-rejection-boundary")
    cand = load_candidate(tmp_path, "synth-rejection-boundary")
    outside_evidence = tmp_path / "outside.json"
    outside_evidence.write_text('{"synthetic": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="inside the candidate workspace"):
        record_decisive_rejection(cand, "Synthetic boundary check.", "../outside.json")

    assert not (candidate_path / "decisions" / "decisive_rejection.json").exists()
