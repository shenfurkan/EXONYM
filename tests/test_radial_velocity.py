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


def _observation_record(candidate_id, time_values, velocity_values, uncertainty=0.5):
    observations = []
    for index, (time_value, velocity_value) in enumerate(zip(time_values, velocity_values)):
        observations.append(
            {
                "observation_time": {"value": float(time_value), "unit": "BJD_TDB"},
                "velocity": {"value": float(velocity_value), "unit": "m/s"},
                "uncertainty": {"value": uncertainty, "unit": "m/s"},
                "instrument": "synthetic-spectrograph",
                "provenance": {
                    "source_uri": "https://example.invalid/synthetic-rv",
                    "retrieved_at_utc": "2000-01-01T00:00:00Z",
                    "record_label": "synthetic-{0}".format(index),
                },
            }
        )
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

    manifests = list((workspace.path / "runs" / "rv-keplerian").glob("*/engine-run.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["inputs"][0]["sha256"] == report["input_artifacts"][0]["sha256"]
    assert manifest["outputs"][0]["path"] == "outputs/rv_keplerian_fit.json"
    assert ingested.is_file()

    audit = IsolationReport()
    validate_schemas(tmp_path, audit)
    assert audit.violations == []


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
