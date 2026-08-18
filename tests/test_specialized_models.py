"""Deterministic tests for opt-in specialized physical-model adapters."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from exonym.__main__ import main
from exonym.isolation import IsolationReport
from exonym.schemas import validate_schemas
from exonym.specialized_models import run_planetsynth, run_pyppluss
from exonym.workspace import create_candidate


def _planetsynth_input(candidate_id, mass_mjup=1.0):
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "purpose": "giant-planet-cooling-evolution",
        "characterization": {
            "mass_mjup": {"value": mass_mjup, "unit": "M_jup"},
            "radius_rjup": {"value": 1.0, "unit": "R_jup"},
            "age_gyr": {"value": 2.0, "unit": "Gyr"},
            "equilibrium_temperature_k": {"value": 900.0, "unit": "K"},
        },
        "provenance": {
            "source_description": "Synthetic candidate-local characterization for adapter testing.",
            "recorded_at": "2000-01-01T00:00:00Z",
        },
    }


def _pyppluss_input(candidate_id):
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "purpose": "anomalous-transit-hypothesis-test",
        "observation": {
            "time_days": {"values": [-0.2, -0.1, 0.0, 0.1, 0.2], "unit": "days"},
            "normalized_flux": {"values": [1.0, 0.99, 0.98, 0.99, 1.0], "unit": "relative_flux"},
        },
        "hypothesis": {
            "model": "ringed-planet",
            "planet_radius_ratio": 0.1,
            "impact_parameter": 0.2,
            "ring_inner_radius_ratio": 0.15,
            "ring_outer_radius_ratio": 0.25,
            "ring_obliquity_deg": 15.0,
        },
        "provenance": {
            "source_description": "Synthetic candidate-local anomalous-transit hypothesis for adapter testing.",
            "recorded_at": "2000-01-01T00:00:00Z",
        },
    }


def _write_input(workspace, filename, payload):
    path = workspace.path / "data" / "external" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _runtime(module_name):
    return (
        {"kind": "direct", "version": "1.2.3", "executable": module_name},
        {"package": module_name, "version": "1.2.3", "python_requires": ">=3.9"},
    )


def test_planetsynth_success_records_candidate_local_hashes_units_and_manifest(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "planetsynth-success")
    original_disposition = workspace.metadata["scientific_disposition"]
    _write_input(workspace, "planetsynth_characterization.json", _planetsynth_input(workspace.candidate_id))
    package = types.ModuleType("planetsynth")
    package.evolve_giant_planet = lambda **kwargs: {"radius_rjup": 1.1, "luminosity_lsun": 0.0002}
    monkeypatch.setitem(sys.modules, "planetsynth", package)
    monkeypatch.setattr("exonym.specialized_models._resolve_runtime", lambda *args: _runtime("planetsynth"))

    result = run_planetsynth(workspace)

    assert result.status == "succeeded"
    assert result.report_path.parent == workspace.path / "outputs"
    assert result.report_path.name.startswith("planetsynth_interpretation.")
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert report["interpretation"]["radius"]["unit"] == "R_jup"
    assert report["runtime"]["version"] == "1.2.3"
    assert report["input_artifact"]["sha256"]
    assert manifest["outputs"][1]["path"] == result.report_path.relative_to(workspace.path).as_posix()
    assert workspace.metadata["scientific_disposition"] == original_disposition
    assert not list((workspace.path / "claims").glob("*.json"))
    audit = IsolationReport()
    validate_schemas(tmp_path, audit)
    assert audit.ok


def test_pyppluss_success_uses_declared_hypothesis_and_records_relative_flux(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "pyppluss-success")
    _write_input(workspace, "anomalous_transit_hypothesis.json", _pyppluss_input(workspace.candidate_id))
    package = types.ModuleType("pyppluss")
    package.model_anomalous_transit = lambda **kwargs: {"model_flux": [1.0, 0.99, 0.98, 0.99, 1.0]}
    monkeypatch.setitem(sys.modules, "pyppluss", package)
    monkeypatch.setattr("exonym.specialized_models._resolve_runtime", lambda *args: _runtime("pyppluss"))

    result = run_pyppluss(workspace)

    assert result.status == "succeeded"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["model"] == "ringed-planet"
    assert report["fit_diagnostics"]["rms_residual"]["unit"] == "relative_flux"
    assert report["fit_diagnostics"]["rms_residual"]["value"] == 0.0


def test_specialized_reports_are_run_specific_and_keep_prior_manifest_hashes_valid(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "specialized-immutable-outputs")
    _write_input(workspace, "planetsynth_characterization.json", _planetsynth_input(workspace.candidate_id))
    package = types.ModuleType("planetsynth")
    package.evolve_giant_planet = lambda **kwargs: {"radius_rjup": 1.1, "luminosity_lsun": 0.0002}
    monkeypatch.setitem(sys.modules, "planetsynth", package)
    monkeypatch.setattr("exonym.specialized_models._resolve_runtime", lambda *args: _runtime("planetsynth"))

    first = run_planetsynth(workspace)
    second = run_planetsynth(workspace)

    assert first.report_path != second.report_path
    assert first.report_path.is_file()
    assert second.report_path.is_file()
    audit = IsolationReport()
    validate_schemas(tmp_path, audit)
    assert audit.ok


def test_missing_specialized_input_creates_no_run_or_astrophysical_output(tmp_path):
    workspace = create_candidate(tmp_path, "specialized-missing-input")

    with pytest.raises(FileNotFoundError, match="candidate-owned"):
        run_planetsynth(workspace)

    assert not (workspace.path / "runs" / "planetsynth").exists()
    assert not list((workspace.path / "outputs").glob("planetsynth_interpretation.*.json"))
    assert not list((workspace.path / "claims").glob("*.json"))


def test_invalid_applicability_stops_before_package_resolution_and_writes_nothing(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "specialized-invalid-limits")
    _write_input(workspace, "planetsynth_characterization.json", _planetsynth_input(workspace.candidate_id, mass_mjup=30.0))
    monkeypatch.setattr(
        "exonym.specialized_models._resolve_runtime",
        lambda *args: pytest.fail("package resolution ran before applicability validation"),
    )

    with pytest.raises(ValueError, match="applicability"):
        run_planetsynth(workspace)

    assert not (workspace.path / "runs" / "planetsynth").exists()
    assert not list((workspace.path / "outputs").glob("planetsynth_interpretation.*.json"))


def test_unavailable_package_writes_only_status_manifest(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "specialized-unavailable")
    _write_input(workspace, "anomalous_transit_hypothesis.json", _pyppluss_input(workspace.candidate_id))
    monkeypatch.setattr(
        "exonym.specialized_models._resolve_runtime",
        lambda *args: (_ for _ in ()).throw(LookupError("module-unavailable: synthetic missing package")),
    )

    result = run_pyppluss(workspace)

    assert result.status == "unavailable"
    assert result.report_path is None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "unavailable"
    assert manifest["outputs"] == []
    assert not list((workspace.path / "outputs").glob("pyppluss_hypothesis_test.*.json"))
    assert not list((workspace.path / "claims").glob("*.json"))


def test_planetsynth_runtime_failure_cleans_package_outputs_and_writes_no_report(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "planetsynth-runtime-failure")
    _write_input(workspace, "planetsynth_characterization.json", _planetsynth_input(workspace.candidate_id))
    package = types.ModuleType("planetsynth")

    def fail_after_writing_scratch(**kwargs):
        del kwargs
        (Path.cwd() / "scratch.txt").write_text("synthetic", encoding="utf-8")
        raise RuntimeError("synthetic failure")

    package.evolve_giant_planet = fail_after_writing_scratch
    monkeypatch.setitem(sys.modules, "planetsynth", package)
    monkeypatch.setattr("exonym.specialized_models._resolve_runtime", lambda *args: _runtime("planetsynth"))

    result = run_planetsynth(workspace)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert result.report_path is None
    assert manifest["failure"]["code"] == "adapter-execution-failed"
    assert manifest["outputs"] == []
    assert not list(result.manifest_path.parent.glob("scratch.txt"))
    assert not list((workspace.path / "outputs").glob("planetsynth_interpretation.*.json"))
    assert not list((workspace.path / "claims").glob("*.json"))


def test_unsupported_pyppluss_interface_writes_only_status_manifest(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "specialized-unsupported-interface")
    _write_input(workspace, "anomalous_transit_hypothesis.json", _pyppluss_input(workspace.candidate_id))
    monkeypatch.setitem(sys.modules, "pyppluss", types.ModuleType("pyppluss"))
    monkeypatch.setattr("exonym.specialized_models._resolve_runtime", lambda *args: _runtime("pyppluss"))

    result = run_pyppluss(workspace)

    assert result.status == "unavailable"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["failure"]["code"] == "unsupported-interface"
    assert manifest["outputs"] == []
    assert not list((workspace.path / "outputs").glob("pyppluss_hypothesis_test.*.json"))


def test_cli_specialized_commands_report_success_and_runtime_failure_without_placeholder(tmp_path, monkeypatch, capsys):
    workspace = create_candidate(tmp_path, "specialized-cli")
    root = ["--root", str(tmp_path)]
    _write_input(workspace, "planetsynth_characterization.json", _planetsynth_input(workspace.candidate_id))
    package = types.ModuleType("planetsynth")
    package.evolve_giant_planet = lambda **kwargs: {"radius_rjup": 1.0, "luminosity_lsun": 0.0001}
    monkeypatch.setitem(sys.modules, "planetsynth", package)
    monkeypatch.setattr("exonym.specialized_models._resolve_runtime", lambda *args: _runtime("planetsynth"))

    assert main(root + ["planetsynth", workspace.candidate_id]) == 0
    assert "planetsynth_interpretation." in capsys.readouterr().out

    _write_input(workspace, "anomalous_transit_hypothesis.json", _pyppluss_input(workspace.candidate_id))
    broken_package = types.ModuleType("pyppluss")
    broken_package.model_anomalous_transit = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure"))
    monkeypatch.setitem(sys.modules, "pyppluss", broken_package)
    monkeypatch.setattr("exonym.specialized_models._resolve_runtime", lambda *args: _runtime("pyppluss"))

    assert main(root + ["pyppluss", workspace.candidate_id]) == 1
    assert not list((workspace.path / "outputs").glob("pyppluss_hypothesis_test.*.json"))


def test_cli_invalid_specialized_input_exits_without_a_run(tmp_path):
    workspace = create_candidate(tmp_path, "specialized-cli-invalid")
    _write_input(workspace, "planetsynth_characterization.json", _planetsynth_input(workspace.candidate_id, mass_mjup=30.0))

    with pytest.raises(SystemExit) as exc_info:
        main(["--root", str(tmp_path), "planetsynth", workspace.candidate_id])

    assert exc_info.value.code == 2
    assert not (workspace.path / "runs" / "planetsynth").exists()
