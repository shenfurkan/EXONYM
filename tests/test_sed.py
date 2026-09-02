"""Synthetic coverage for manifest-bound response-integrated SED calibration."""

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from exonym.sed import run_sed_fit
from exonym.isolation import IsolationReport
from exonym.schemas import validate_schemas
from exonym.workspace import create_candidate


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(workspace, path, role):
    return {"path": path.relative_to(workspace.path).as_posix(), "sha256": _sha256(path), "role": role}


def _write_response_integrated_assets(workspace, bands=("J", "H", "Ks", "W1", "W2")):
    external = workspace.path / "data" / "external"
    atmosphere = external / "atmosphere"
    filters = external / "filters"
    atmosphere.mkdir()
    filters.mkdir()
    photometry = external / "stellar_photometry.json"
    measurements = {
        "J": ("2MASS", 5.0),
        "H": ("2MASS", 5.1),
        "Ks": ("2MASS", 5.2),
        "W1": ("AllWISE", 5.3),
        "W2": ("AllWISE", 5.4),
    }
    photometry_payload = {"2MASS": {}, "AllWISE": {}}
    for band in bands:
        catalog, magnitude = measurements[band]
        photometry_payload[catalog][band] = {"mag": magnitude, "error": 0.03}
    photometry.write_text(json.dumps(photometry_payload), encoding="utf-8")
    stellar = external / "stellar_params.json"
    stellar.write_text(json.dumps({"teff_k": 5000.0, "logg_cgs": 4.5, "feh": 0.0, "mass_solar": 1.0, "radius_solar": 1.0, "parallax_mas": 10.0}), encoding="utf-8")
    spectra = []
    fixture_root = Path(__file__).parent / "fixtures"
    for index, (teff, logg, feh) in enumerate(((4900.0, 4.4, -0.1), (5100.0, 4.4, 0.1), (4900.0, 4.6, 0.1), (5100.0, 4.6, -0.1))):
        path = atmosphere / "synthetic-{}.csv".format(index)
        shutil.copy2(fixture_root / "synthetic_atmosphere_spectrum.csv", path)
        spectra.append({**_artifact(workspace, path, "atmosphere-spectrum"), "teff_k": teff, "logg_cgs": logg, "feh": feh})
    responses = []
    zero_points = {"J": 1594.0, "H": 1024.0, "Ks": 666.7, "W1": 309.540, "W2": 171.787}
    for band in bands:
        path = filters / (band + ".csv")
        shutil.copy2(fixture_root / "sed_response_curve.csv", path)
        responses.append({**_artifact(workspace, path, "filter-response"), "band": band, "zero_point_flux_jy": zero_points[band]})
    manifest = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "purpose": "response-integrated-sed-calibration",
        "photometry_artifact": _artifact(workspace, photometry, "stellar-photometry"),
        "stellar_parameters_artifact": _artifact(workspace, stellar, "stellar-parameters"),
        "atmosphere_spectra": spectra,
        "filter_responses": responses,
    }
    (external / "sed_input_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return filters / "J.csv"


def _fake_sampler(log_probability, start, n_walkers, burn_in, production, seed):
    return np.tile(start, (8, 1)), SimpleNamespace(acceptance_fraction=np.full(n_walkers, 0.5))


def test_sed_writes_calibrated_result_only_with_full_structural_contract(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "synthetic-sed")
    _write_response_integrated_assets(workspace)
    monkeypatch.setattr("exonym.sed._run_emcee", _fake_sampler)

    result_path = run_sed_fit(workspace)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["calibrated"] is True
    assert result["calibration_status"] == "verified-response-integrated-fitzpatrick99-rv31"
    assert set(result["posterior"]) == {"teff_k", "logg_cgs", "feh", "rstar_over_distance", "av_mag"}
    assert result["calibration_assets"]["extinction_law"] == "Fitzpatrick 1999 R_V=3.1"
    assert result["calibration_assets"]["independent_filter_count"] == 5
    report = IsolationReport()
    validate_schemas(tmp_path, report)
    assert report.ok


def test_sed_writes_uncalibrated_result_when_filters_are_fewer_than_free_parameters(tmp_path):
    workspace = create_candidate(tmp_path, "synthetic-sed-two-bands")
    _write_response_integrated_assets(workspace, bands=("J", "H"))

    result = json.loads(run_sed_fit(workspace).read_text(encoding="utf-8"))

    assert result["calibrated"] is False
    assert result["calibration_status"] == "uncalibrated"
    assert "filter_responses" in result["calibration_assets"]["reason"]
    assert "posterior" not in result


def test_sed_writes_uncalibrated_result_when_manifest_asset_is_stale(tmp_path):
    workspace = create_candidate(tmp_path, "synthetic-sed-stale")
    stale_response = _write_response_integrated_assets(workspace)
    stale_response.write_text("wavelength_angstrom,response\n11000,0.5\n14000,1.0\n17000,0.5\n", encoding="utf-8")

    result = json.loads(run_sed_fit(workspace).read_text(encoding="utf-8"))

    assert result["calibrated"] is False
    assert result["calibration_status"] == "uncalibrated"
    assert "SHA-256" in result["calibration_assets"]["reason"]
    assert "posterior" not in result


def test_sed_writes_uncalibrated_result_outside_the_finite_atmosphere_hull(tmp_path):
    workspace = create_candidate(tmp_path, "synthetic-sed-outside-hull")
    _write_response_integrated_assets(workspace)
    external = workspace.path / "data" / "external"
    stellar_path = external / "stellar_params.json"
    stellar = json.loads(stellar_path.read_text(encoding="utf-8"))
    stellar["teff_k"] = 5300.0
    stellar_path.write_text(json.dumps(stellar), encoding="utf-8")
    manifest_path = external / "sed_input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stellar_parameters_artifact"] = _artifact(workspace, stellar_path, "stellar-parameters")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = json.loads(run_sed_fit(workspace).read_text(encoding="utf-8"))

    assert result["calibrated"] is False
    assert "outside the finite atmosphere-grid interpolation hull" in result["calibration_assets"]["reason"]
