"""Synthetic contract coverage for the candidate-local frozen MIST grid path."""

from __future__ import annotations

import hashlib
import json
import warnings
from itertools import product

import pytest


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_synthetic_mist_contract(workspace):
    external = workspace.path / "data" / "external"
    source_path = external / "synthetic_mist_grid_source.txt"
    source_path.write_text("synthetic normalized-grid source\n", encoding="utf-8")

    base_magnitudes = {
        "gaia_g": 5.0,
        "gaia_bp": 5.2,
        "gaia_rp": 4.8,
        "twomass_j": 4.4,
        "twomass_h": 4.2,
        "twomass_ks": 4.1,
    }
    magnitude_columns = {
        "gaia_g": "gaia_g_abs_mag",
        "gaia_bp": "gaia_bp_abs_mag",
        "gaia_rp": "gaia_rp_abs_mag",
        "twomass_j": "twomass_j_abs_mag",
        "twomass_h": "twomass_h_abs_mag",
        "twomass_ks": "twomass_ks_abs_mag",
    }
    headers = ["evolutionary_stage", "teff_k", "logg_cgs", "feh", *magnitude_columns.values()]
    rows = [",".join(headers)]
    for teff_k, logg_cgs, feh_dex in product((5700.0, 5900.0), (4.3, 4.5), (-0.1, 0.1)):
        perturbation = 0.001 * (teff_k - 5800.0) + 0.25 * (logg_cgs - 4.4) + 0.15 * feh_dex
        row = ["main_sequence", str(teff_k), str(logg_cgs), str(feh_dex)]
        row.extend(str(base_magnitudes[band] + perturbation) for band in magnitude_columns)
        rows.append(",".join(row))
    grid_path = external / "mist_isochrone_grid.csv"
    grid_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    stellar = {
        "source": "candidate-data",
        "teff_k": 5800.0,
        "logg_cgs": 4.4,
        "feh": 0.0,
        "parallax_mas": 10.0,
        "parallax_mas_err": 0.1,
    }
    stellar_path = external / "stellar_params.json"
    stellar_path.write_text(json.dumps(stellar), encoding="utf-8")

    def measurement(absolute_magnitude):
        return {"value": absolute_magnitude + 5.1, "uncertainty": 0.01, "unit": "mag"}

    manifest = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "purpose": "mist-main-sequence-check",
        "photometry": {
            "gaia_dr3": {
                "g": measurement(base_magnitudes["gaia_g"]),
                "bp": measurement(base_magnitudes["gaia_bp"]),
                "rp": measurement(base_magnitudes["gaia_rp"]),
            },
            "twomass": {
                "j": measurement(base_magnitudes["twomass_j"]),
                "h": measurement(base_magnitudes["twomass_h"]),
                "ks": measurement(base_magnitudes["twomass_ks"]),
            },
        },
        "extinction": {"band_extinction_mag": {band: 0.1 for band in base_magnitudes}},
        "grid_artifact": {
            "path": "data/external/mist_isochrone_grid.csv",
            "sha256": _sha256(grid_path),
            "role": "mist-isochrone-grid",
        },
        "provenance": {
            "source_description": "synthetic normalized-grid regression contract",
            "recorded_at": "2024-01-01T00:00:00Z",
            "mist_release": "synthetic-normalized-grid-contract",
            "filters": list(magnitude_columns),
            "input_artifacts": [
                {
                    "path": "data/external/synthetic_mist_grid_source.txt",
                    "sha256": _sha256(source_path),
                    "role": "synthetic-grid-source",
                }
            ],
        },
    }
    manifest_path = external / "mist_main_sequence_input.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "base_magnitudes": base_magnitudes,
        "grid_path": grid_path,
        "manifest_path": manifest_path,
        "source_path": source_path,
        "stellar_path": stellar_path,
        "stellar": stellar,
    }


def _update_manifest_grid_digest(manifest_path, grid_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest["grid_artifact"]["sha256"] = _sha256(grid_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_synthetic_mist_contract_evaluates_hash_bound_interpolated_grid(tmp_path):
    from exonym.sed import _mist_main_sequence_check
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "mist-grid-contract")
    inputs = _write_synthetic_mist_contract(workspace)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = _mist_main_sequence_check(workspace, inputs["stellar"])

    assert result["status"] == "evaluated"
    assert result["method"] == "frozen-mist-main-sequence-linear-interpolation"
    assert result["validation_eligible"] is False
    assert result["claim_eligible"] is False
    assert result["parallax_snr"] == pytest.approx(100.0)
    assert result["interpolated_main_sequence_absolute_magnitudes"] == pytest.approx(
        inputs["base_magnitudes"], abs=1e-12
    )
    assert result["observed_absolute_magnitudes"] == pytest.approx(
        inputs["base_magnitudes"], abs=1e-12
    )
    assert result["residuals_mag"] == pytest.approx(
        {band: 0.0 for band in inputs["base_magnitudes"]}, abs=1e-12
    )
    assert result["chi_square_fixed_stellar_parameters"] == pytest.approx(0.0, abs=1e-20)
    assert result["grid_artifact"]["sha256"] == _sha256(inputs["grid_path"])
    assert result["input_artifact"]["sha256"] == _sha256(inputs["manifest_path"])
    assert result["stellar_parameters_artifact"]["sha256"] == _sha256(inputs["stellar_path"])
    assert result["source_artifacts"] == [
        {
            "path": "data/external/synthetic_mist_grid_source.txt",
            "sha256": _sha256(inputs["source_path"]),
            "role": "synthetic-grid-source",
        }
    ]


def test_mist_contract_accepts_hash_bound_utf8_bom_inputs(tmp_path):
    from exonym.sed import _mist_main_sequence_check
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "mist-grid-utf8-bom")
    inputs = _write_synthetic_mist_contract(workspace)
    manifest_path = inputs["manifest_path"]
    grid_path = inputs["grid_path"]
    grid_path.write_bytes(b"\xef\xbb\xbf" + grid_path.read_bytes())
    _update_manifest_grid_digest(manifest_path, grid_path)
    manifest_path.write_bytes(b"\xef\xbb\xbf" + manifest_path.read_bytes())

    result = _mist_main_sequence_check(workspace, inputs["stellar"])

    assert result["status"] == "evaluated"


def test_mist_contract_rejects_grid_hash_tampering(tmp_path):
    from exonym.sed import _mist_main_sequence_check
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "mist-grid-hash-tamper")
    inputs = _write_synthetic_mist_contract(workspace)
    inputs["grid_path"].write_text(
        inputs["grid_path"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="MIST grid artifact SHA-256"):
        _mist_main_sequence_check(workspace, inputs["stellar"])


def test_mist_contract_rejects_nonfinite_main_sequence_rows_and_negative_extinction(tmp_path):
    from exonym.sed import _mist_main_sequence_check
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "mist-grid-domain-guards")
    inputs = _write_synthetic_mist_contract(workspace)
    rows = inputs["grid_path"].read_text(encoding="utf-8").splitlines()
    fields = rows[1].split(",")
    fields[1] = "nan"
    rows[1] = ",".join(fields)
    inputs["grid_path"].write_text("\n".join(rows) + "\n", encoding="utf-8")
    _update_manifest_grid_digest(inputs["manifest_path"], inputs["grid_path"])

    with pytest.raises(RuntimeError, match="non-finite main-sequence row"):
        _mist_main_sequence_check(workspace, inputs["stellar"])

    negative_extinction_workspace = create_candidate(tmp_path, "mist-grid-negative-extinction")
    inputs = _write_synthetic_mist_contract(negative_extinction_workspace)
    manifest = json.loads(inputs["manifest_path"].read_text(encoding="utf-8"))
    manifest["extinction"]["band_extinction_mag"]["gaia_g"] = -0.1
    inputs["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="finite and non-negative"):
        _mist_main_sequence_check(negative_extinction_workspace, inputs["stellar"])
