import json

import numpy as np
import pytest

from exonym.__main__ import main
from exonym.isolation import IsolationReport
from exonym.radial_velocity import (
    fit_radial_velocity,
    ingest_radial_velocity_observations,
    keplerian_velocity_m_per_s,
)
from exonym.schemas import validate_schemas
from exonym.workspace import create_candidate


def _observation_record(
    candidate_id,
    time_values,
    velocity_values,
    uncertainty=0.5,
    instruments=None,
    activity_values=None,
    activity_units=None,
):
    observations = []
    for index, (time_value, velocity_value) in enumerate(zip(time_values, velocity_values)):
        observation = {
            "observation_time": {"value": float(time_value), "unit": "BJD_TDB"},
            "velocity": {"value": float(velocity_value), "unit": "m/s"},
            "uncertainty": {"value": uncertainty, "unit": "m/s"},
            "instrument": instruments[index] if instruments is not None else "synthetic-spectrograph",
            "provenance": {
                "source_uri": "https://example.invalid/synthetic-rv",
                "retrieved_at_utc": "2000-01-01T00:00:00Z",
                "record_label": "synthetic-{0}".format(index),
            },
        }
        if activity_values is not None:
            observation["activity_indicator"] = {
                "value": float(activity_values[index]),
                "unit": activity_units[index] if activity_units is not None else "relative-index",
            }
        observations.append(observation)
    return {"schema_version": 1, "candidate_id": candidate_id, "observations": observations}


def test_rv_ingestion_rejects_duplicate_json_keys_and_wrong_units(tmp_path):
    workspace = create_candidate(tmp_path, "rv-ingest-safety")
    source = tmp_path / "source.json"
    source.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        ingest_radial_velocity_observations(workspace, source)

    record = _observation_record(workspace.candidate_id, [1.0, 2.0, 3.0], [0.0, 1.0, 0.0])
    record["observations"][0]["velocity"]["unit"] = "km/s"
    source.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="schema violation"):
        ingest_radial_velocity_observations(workspace, source)


def test_rv_ingestion_rejects_nonfinite_json_without_candidate_output(tmp_path):
    workspace = create_candidate(tmp_path, "rv-ingest-nonfinite")
    source = tmp_path / "source.json"
    record = _observation_record(workspace.candidate_id, [1.0, 2.0, 3.0], [0.0, 1.0, 0.0])
    record["observations"][0]["velocity"]["value"] = float("nan")
    source.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON number"):
        ingest_radial_velocity_observations(workspace, source)

    assert not (workspace.path / "data" / "external" / "radial_velocity_observations.json").exists()


def test_rv_fit_with_insufficient_observations_writes_no_report_or_manifest(tmp_path):
    workspace = create_candidate(tmp_path, "rv-fit-insufficient")
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(_observation_record(workspace.candidate_id, [1.0, 2.0, 3.0], [0.0, 1.0, 0.0])),
        encoding="utf-8",
    )
    ingest_radial_velocity_observations(workspace, source)

    with pytest.raises(ValueError, match="more observations"):
        fit_radial_velocity(workspace, 2.0)

    assert not (workspace.path / "outputs" / "rv_keplerian_fit.json").exists()
    assert not (workspace.path / "runs" / "rv-keplerian").exists()


def test_rv_keplerian_fit_recovers_synthetic_amplitude_and_records_provenance(tmp_path):
    workspace = create_candidate(tmp_path, "rv-keplerian-fit")
    period_days = 4.0
    reference_time_bjd_tdb = 1000.0
    time = np.linspace(reference_time_bjd_tdb - 10.0, reference_time_bjd_tdb + 10.0, 48)
    injected_amplitude = 9.0
    velocity = 3.0 + keplerian_velocity_m_per_s(
        time,
        injected_amplitude,
        0.4,
        0.0,
        0.0,
        reference_time_bjd_tdb,
        period_days,
    )
    source = tmp_path / "synthetic-rv.json"
    source.write_text(
        json.dumps(_observation_record(workspace.candidate_id, time, velocity)), encoding="utf-8"
    )

    ingested = ingest_radial_velocity_observations(workspace, source)
    output = fit_radial_velocity(workspace, period_days, period_uncertainty_days=0.02)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert output == workspace.path / "outputs" / "rv_keplerian_fit.json"
    assert report["models"]["keplerian"]["parameters"]["semi_amplitude"]["value"] == pytest.approx(
        injected_amplitude, abs=0.2
    )
    assert report["models"]["keplerian"]["parameters"]["semi_amplitude"]["unit"] == "m/s"
    assert report["models"]["keplerian"]["parameters"]["semi_amplitude"]["uncertainty"] is not None
    assert report["model_comparison"]["delta_bic_constant_minus_keplerian"] > 0
    assert report["input_artifacts"][0]["path"] == "data/external/radial_velocity_observations.json"
    assert report["input_artifacts"][0]["sha256"]
    assert report["diagnostics"]["kepler_equation_solver"] == {
        "method": "Danby starter + Halley third-order iteration with residual convergence check",
        "tolerance_rad": 1e-12,
        "max_iterations": 64,
    }
    eccentricity_parameterization = report["diagnostics"]["eccentricity_parameterization"]
    assert eccentricity_parameterization["mapping"] == (
        "e = 0.95 * (x^2 + y^2) / (1 + x^2 + y^2)"
    )
    assert eccentricity_parameterization["maximum_eccentricity_exclusive"] == 0.95
    assert "numerical support restriction" in eccentricity_parameterization["scientific_limitation"]
    assert report["diagnostics"]["activity_regression"]["status"] == "not-provided"
    assert "Hessian" not in report["diagnostics"]["uncertainty_estimation"]
    sampling = report["diagnostics"]["posterior_sampling"]
    assert sampling["keplerian_model"]["sampler"] == "emcee-affine-invariant-ensemble"
    assert sampling["keplerian_model"]["retained_draws"] > 0
    assert len(report["models"]["keplerian"]["parameters"]["instrument_jitters"]) == 1

    manifests = list((workspace.path / "runs" / "rv-keplerian").glob("*/engine-run.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["inputs"][0]["sha256"] == report["input_artifacts"][0]["sha256"]
    assert manifest["outputs"][0]["path"] == "outputs/rv_keplerian_fit.json"
    assert ingested.is_file()

    audit = IsolationReport()
    validate_schemas(tmp_path, audit)
    assert audit.violations == []


def test_rv_fit_jointly_models_instrument_jitter_linear_trend_and_activity(tmp_path):
    workspace = create_candidate(tmp_path, "rv-nuisance-model")
    period_days = 5.0
    reference_time_bjd_tdb = 1200.0
    time = np.linspace(reference_time_bjd_tdb - 30.0, reference_time_bjd_tdb + 30.0, 96)
    activity = np.sin(0.61 * time) + 0.31 * np.cos(0.17 * time)
    instruments = ["spectrograph-a" if index % 2 == 0 else "spectrograph-b" for index in range(time.size)]
    offsets = np.asarray([2.0 if instrument == "spectrograph-a" else -1.5 for instrument in instruments])
    velocity = (
        offsets
        + 0.12 * (time - reference_time_bjd_tdb)
        + 2.5 * activity
        + keplerian_velocity_m_per_s(
            time,
            7.0,
            0.8,
            0.0,
            0.0,
            reference_time_bjd_tdb,
            period_days,
        )
    )
    source = tmp_path / "rv-nuisance.json"
    source.write_text(
        json.dumps(
            _observation_record(
                workspace.candidate_id,
                time,
                velocity,
                uncertainty=0.4,
                instruments=instruments,
                activity_values=activity,
            )
        ),
        encoding="utf-8",
    )
    ingest_radial_velocity_observations(workspace, source)

    report = json.loads(fit_radial_velocity(workspace, period_days).read_text(encoding="utf-8"))
    parameters = report["models"]["keplerian"]["parameters"]

    assert parameters["semi_amplitude"]["value"] == pytest.approx(7.0, abs=0.3)
    assert parameters["linear_trend"]["value"] == pytest.approx(0.12, abs=0.03)
    assert parameters["activity_coefficient"]["value"] == pytest.approx(2.5, abs=0.3)
    assert parameters["activity_coefficient"]["unit"] == "m/s per relative-index"
    assert {item["instrument"] for item in parameters["instrument_jitters"]} == {
        "spectrograph-a",
        "spectrograph-b",
    }
    assert report["diagnostics"]["activity_regression"]["status"] == "jointly-fitted"
    assert report["diagnostics"]["noise_model"].startswith("quoted per-observation")


def test_rv_ingestion_rejects_partial_activity_indicator_series(tmp_path):
    workspace = create_candidate(tmp_path, "rv-partial-activity")
    source = tmp_path / "partial-activity.json"
    record = _observation_record(
        workspace.candidate_id,
        np.linspace(1.0, 20.0, 16),
        np.zeros(16),
        activity_values=np.linspace(-1.0, 1.0, 16),
    )
    del record["observations"][0]["activity_indicator"]
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="every observation or none"):
        ingest_radial_velocity_observations(workspace, source)
    assert not (workspace.path / "data" / "external" / "radial_velocity_observations.json").exists()


def test_rv_ingestion_rejects_mixed_activity_indicator_units(tmp_path):
    workspace = create_candidate(tmp_path, "rv-mixed-activity-units")
    source = tmp_path / "mixed-activity-units.json"
    count = 16
    record = _observation_record(
        workspace.candidate_id,
        np.linspace(1.0, 20.0, count),
        np.zeros(count),
        activity_values=np.linspace(-1.0, 1.0, count),
        activity_units=["relative-index"] * (count - 1) + ["m/s"],
    )
    source.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="one common unit"):
        ingest_radial_velocity_observations(workspace, source)
    assert not (workspace.path / "data" / "external" / "radial_velocity_observations.json").exists()


def test_kepler_solver_reaches_declared_residual_at_high_eccentricity():
    from exonym.radial_velocity import _solve_kepler_equation

    mean_anomaly = np.linspace(-8.0, 8.0, 101)
    eccentricity = 0.95
    eccentric_anomaly = _solve_kepler_equation(mean_anomaly, eccentricity)
    residual = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - np.mod(
        mean_anomaly, 2.0 * np.pi
    )

    assert np.max(np.abs(residual)) <= 1e-12


def test_schema_audit_rejects_rv_record_outside_its_candidate_workspace(tmp_path):
    workspace = create_candidate(tmp_path, "rv-schema-ownership")
    source = tmp_path / "observations.json"
    source.write_text(
        json.dumps(_observation_record(workspace.candidate_id, [1.0, 2.0, 3.0], [0.0, 1.0, 0.0])),
        encoding="utf-8",
    )
    ingested = ingest_radial_velocity_observations(workspace, source)
    record = json.loads(ingested.read_text(encoding="utf-8"))
    record["candidate_id"] = "different-workspace"
    ingested.write_text(json.dumps(record), encoding="utf-8")

    audit = IsolationReport()
    validate_schemas(tmp_path, audit)

    assert any("RV observation candidate_id" in violation.detail for violation in audit.violations)


def test_cli_rv_ingests_and_fits_candidate_local_artifacts(tmp_path, capsys):
    workspace = create_candidate(tmp_path, "rv-cli")
    time = np.linspace(20.0, 28.0, 24)
    velocity = keplerian_velocity_m_per_s(time, 4.0, 0.2, 0.0, 0.0, 24.0, 2.0)
    source = tmp_path / "rv-cli-source.json"
    source.write_text(
        json.dumps(_observation_record(workspace.candidate_id, time, velocity)), encoding="utf-8"
    )
    root = ["--root", str(tmp_path)]

    assert main(root + ["rv", "ingest", workspace.candidate_id, str(source)]) == 0
    assert main(root + ["rv", "fit", workspace.candidate_id, "--period-days", "2"]) == 0

    output = capsys.readouterr().out
    assert "radial_velocity_observations.json" in output
    assert "rv_keplerian_fit.json" in output
