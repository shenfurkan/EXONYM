"""Regression coverage for evidence-backed asteroseismic Δν corrections."""

import json

import numpy as np
import pytest


def test_dnu_correction_provenance_is_identity_or_unavailable_without_eligible_input():
    from exonym.asteroseismology import _resolve_dnu_correction

    identity = _resolve_dnu_correction({}, 60.0, None)
    unavailable = _resolve_dnu_correction({}, None, None)
    invalid = _resolve_dnu_correction(
        {"dnu_correction": {"factor": 0.98, "evidence": {"reference": "source only"}}},
        60.0,
        None,
    )

    assert identity["status"] == "identity-no-evidence-backed-input"
    assert identity["factor"] == pytest.approx(1.0)
    assert identity["applied"] is False
    assert unavailable["status"] == "unavailable-no-measured-dnu"
    assert unavailable["scaling_dnu_uhz"] is None
    assert invalid["status"] == "identity-invalid-evidence-record"
    assert invalid["factor"] == pytest.approx(1.0)


def test_runner_applies_evidence_backed_dnu_correction_and_records_input_provenance(
    tmp_path, monkeypatch
):
    import exonym.asteroseismology as asteroseismology
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "asteroseismic-correction-test")
    stellar_path = workspace.path / "data" / "external" / "stellar_params.json"
    stellar_path.parent.mkdir(parents=True, exist_ok=True)
    stellar_path.write_text(
        json.dumps(
            {
                "teff_k": 5700.0,
                "teff_k_err": 75.0,
                "logg_cgs": 4.4,
                "feh": 0.0,
                "mass_solar": 1.0,
                "radius_solar": 1.0,
                "dnu_correction": {
                    "factor": 0.98,
                    "evidence": {
                        "reference": "synthetic calibrated-grid record",
                        "applicability": "synthetic regression fixture",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    time = np.linspace(0.0, 10.0, 200)
    table = {
        "time": time,
        "flux": np.ones_like(time),
        "flux_err": np.full_like(time, 1e-4),
        "sector": np.ones(time.size, dtype=int),
    }
    envelope = {
        "numax_candidate_uhz": 1500.0,
        "dnu_candidate_uhz": 120.0,
        "dnu_correlation": 0.9,
        "envelope_peak_ratio": 2.0,
        "rayleigh_uhz": 1.0,
        "baseline_days": 10.0,
        "numax_min_requested_uhz": 100.0,
        "numax_max_requested_uhz": 1600.0,
        "numax_min_used": 100.0,
        "numax_max_used": 1600.0,
        "numax_min_clipped": False,
        "numax_max_clipped": False,
        "frequency_support": {
            "baseline_days": 10.0,
            "median_cadence_seconds": 120.0,
            "rayleigh_uhz": 1.0,
            "nyquist_uhz": 2000.0,
            "duty_cycle": 0.5,
        },
    }
    unavailable_pysyd = {
        "status": "unavailable",
        "manifest_path": workspace.path / "runs" / "pysyd" / "manifest.json",
        "crosscheck": None,
    }
    unavailable_tess_atl = {
        "status": "unavailable",
        "manifest_path": workspace.path / "runs" / "tess-atl" / "manifest.json",
    }

    monkeypatch.setattr(asteroseismology, "load_light_curve_table", lambda *_args, **_kwargs: table)
    monkeypatch.setattr(
        asteroseismology,
        "load_transit_ephemeris",
        lambda *_args, **_kwargs: {"source": "synthetic-demo", "field_sources": {}},
    )
    monkeypatch.setattr(
        asteroseismology,
        "_highpass_segments",
        lambda source_time, source_flux, *_args, **_kwargs: (source_time, source_flux),
    )
    monkeypatch.setattr(asteroseismology, "estimate_oscillation_envelope", lambda *_args: envelope)
    monkeypatch.setattr(
        asteroseismology,
        "_run_pysyd_adapter",
        lambda *_args, **_kwargs: unavailable_pysyd,
    )
    monkeypatch.setattr(
        asteroseismology,
        "_record_tess_atl_adapter",
        lambda *_args, **_kwargs: unavailable_tess_atl,
    )

    output = asteroseismology.run_asteroseismology(workspace)
    payload = json.loads(output.read_text(encoding="utf-8"))

    correction = payload["dnu_correction"]
    assert correction["status"] == "corrected-evidence-backed-input"
    assert correction["factor"] == pytest.approx(0.98)
    assert correction["raw_dnu_uhz"] == pytest.approx(120.0)
    assert correction["scaling_dnu_uhz"] == pytest.approx(117.6)
    assert correction["evidence"] == {
        "reference": "synthetic calibrated-grid record",
        "applicability": "synthetic regression fixture",
    }
    assert correction["input_artifact"]["path"] == "data/external/stellar_params.json"
    assert len(correction["input_artifact"]["sha256"]) == 64
    assert payload["uncertainty"]["dnu_correction_factor"] == pytest.approx(0.98)
    assert payload["uncertainty"]["dnu_corrected_uhz"]["median"] == pytest.approx(
        0.98 * payload["uncertainty"]["dnu_uhz"]["median"]
    )

    expected = asteroseismology.seismic_mass_radius(1500.0, 120.0, 5700.0, dnu_correction_factor=0.98)
    assert payload["stellar_parameters"]["mass_solar"] == pytest.approx(expected["mass_solar"])
    assert payload["stellar_parameters"]["radius_solar"] == pytest.approx(expected["radius_solar"])
