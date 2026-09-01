"""Regression coverage for candidate-local transit-fit TTV templates."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from exonym.ttv import _timing_values_agree, fit_weighted_linear_ephemeris, run_ttv_analysis
from exonym.workspace import create_candidate


def _ttv_inputs():
    return (
        {
            "time": np.array([0.0, 1.0]),
            "flux": np.ones(2),
            "flux_err": np.full(2, 1e-4),
            "sector": np.ones(2, dtype=int),
        },
        {
            "period_days": 2.0,
            "epoch_btjd": 0.5,
            "duration_days": 0.1,
            "depth_ppm": 1600.0,
            "source": "candidate-config",
            "field_sources": {
                "period_days": "candidate-config",
                "epoch_btjd": "candidate-config",
                "duration_days": "candidate-config",
                "depth_ppm": "candidate-config",
            },
        },
        {"mass_solar": 1.0, "radius_solar": 1.0, "source": "candidate-data"},
    )


def _posterior_artifact(signal=None, impact_parameter=0.47, q1=0.64, q2=0.2):
    return {
        "work_package": "MCMC_TRANSIT_FIT",
        "source": "candidate-data",
        "signal": signal,
        "scientific_status": "exploratory-native-cadence-inference",
        "validation_eligible": False,
        "parameter_names": [
            "rp_rs",
            "log_rho_star",
            "impact_parameter",
            "baseline",
            "log_jitter",
            "q1",
            "q2",
        ],
        "ephemeris": {
            "period_days": 2.0,
            "epoch_btjd": 0.5,
            "source": "candidate-config",
        },
        "posterior": {
            "impact_parameter": {"median": impact_parameter},
            "q1": {"median": q1},
            "q2": {"median": q2},
        },
    }


def _write_transit_fit(workspace, payload, signal=None):
    suffix = f".{signal.lstrip('.')}" if signal else ""
    path = workspace.path / "outputs" / f"mcmc_transit_fit{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _patch_ttv_inputs(monkeypatch, table, ephemeris, stellar):
    monkeypatch.setattr("exonym.ttv.load_light_curve_table", lambda *_args, **_kwargs: table)
    monkeypatch.setattr("exonym.ttv.load_transit_ephemeris", lambda *_args, **_kwargs: ephemeris)
    monkeypatch.setattr("exonym.ttv.load_stellar_parameters", lambda *_args, **_kwargs: stellar)


def _empty_timing_analysis():
    return {
        "n_transits_fit": 0,
        "n_excluded_no_detection": 0,
        "n_rejected_epochs": 0,
        "n_template_failures": 0,
        "oc_rms_minutes": None,
        "mean_uncertainty_minutes": None,
        "epochs": [],
        "t_observed_btjd": [],
        "t_calculated_btjd": [],
        "oc_minutes": [],
        "input_ephemeris_oc_minutes": [],
        "oc_error_minutes": [],
        "rejected_epochs": [],
        "uncertainty_clipped_epochs": [],
        "search_boundary_epochs": [],
        "per_epoch": [],
        "epoch_acceptance": {},
        "linear_ephemeris": fit_weighted_linear_ephemeris(
            np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)
        ),
    }


def _single_epoch_timing_analysis():
    timing_error_days = 0.001
    linear_ephemeris = fit_weighted_linear_ephemeris(
        np.array([0]), np.array([0.5]), np.array([timing_error_days])
    )
    return {
        "n_transits_fit": 1,
        "n_excluded_no_detection": 0,
        "n_rejected_epochs": 0,
        "n_template_failures": 0,
        "oc_rms_minutes": 0.0,
        "mean_uncertainty_minutes": timing_error_days * 1440.0,
        "epochs": [0],
        "t_observed_btjd": [0.5],
        "t_calculated_btjd": [0.5],
        "oc_minutes": [0.0],
        "input_ephemeris_oc_minutes": [0.0],
        "oc_error_minutes": [timing_error_days * 1440.0],
        "rejected_epochs": [],
        "uncertainty_clipped_epochs": [],
        "search_boundary_epochs": [],
        "per_epoch": [
            {
                "epoch": 0,
                "t0_fit_btjd": 0.5,
                "t_expected_btjd": 0.5,
                "sigma_t0_days": timing_error_days,
                "sigma_t0_raw_days": timing_error_days,
                "local_depth_ppm": 1000.0,
                "local_depth_uncertainty_ppm": 100.0,
                "local_depth_snr": 10.0,
                "at_search_boundary": False,
                "sigma_t0_clipped": False,
                "excluded_no_detection": False,
                "rejection_reason": None,
            }
        ],
        "epoch_acceptance": {},
        "linear_ephemeris": linear_ephemeris,
    }


def test_ttv_threads_candidate_posterior_medians_and_records_template_provenance(
    tmp_path, monkeypatch
):
    workspace = create_candidate(tmp_path, "ttv-template-threading")
    artifact_path = _write_transit_fit(workspace, _posterior_artifact(signal=".02"), signal=".02")
    table, ephemeris, stellar = _ttv_inputs()
    captured = {}

    def fake_timing_analysis(_time, _flux, _flux_err, _ephemeris, template, **_kwargs):
        captured["template"] = template
        return _empty_timing_analysis()

    _patch_ttv_inputs(monkeypatch, table, ephemeris, stellar)
    monkeypatch.setattr("exonym.ttv.transit_timing_analysis", fake_timing_analysis)

    output_path = run_ttv_analysis(workspace, signal=".02")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    template = payload["timing"]["template"]

    assert captured["template"]["impact_parameter"] == pytest.approx(0.47)
    assert captured["template"]["q1"] == pytest.approx(0.64)
    assert captured["template"]["q2"] == pytest.approx(0.2)
    assert captured["template"]["u1"] == pytest.approx(0.32)
    assert captured["template"]["u2"] == pytest.approx(0.48)
    assert captured["template"]["duration_days"] == pytest.approx(0.1)
    assert template["kind"] == "candidate-local-transit-fit-posterior-median"
    assert template["artifact"]["path"] == "outputs/mcmc_transit_fit.02.json"
    assert template["artifact"]["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert template["artifact"]["signal"] == ".02"
    assert template["parameters"]["limb_darkening"]["q1"] == pytest.approx(0.64)
    assert template["parameters"]["limb_darkening"]["u2"] == pytest.approx(0.48)
    assert template["limitations"]
    assert payload["input_provenance"]["transit_fit_artifact"] == template["artifact"]
    json.dumps(payload, allow_nan=False)


def test_ttv_fails_closed_when_transit_fit_artifact_is_missing(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "ttv-template-missing-fit")
    table, ephemeris, stellar = _ttv_inputs()
    _patch_ttv_inputs(monkeypatch, table, ephemeris, stellar)

    with pytest.raises(RuntimeError, match="candidate-local transit-fit artifact"):
        run_ttv_analysis(workspace)

    assert not (workspace.path / "outputs" / "ttv_analysis_results.json").exists()


def test_ttv_rejects_invalid_or_ambiguous_transit_fit_posterior(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "ttv-template-invalid-fit")
    table, ephemeris, stellar = _ttv_inputs()
    _patch_ttv_inputs(monkeypatch, table, ephemeris, stellar)
    incomplete = _posterior_artifact()
    incomplete["posterior"].pop("q2")
    path = _write_transit_fit(workspace, incomplete)

    with pytest.raises(RuntimeError, match=r"missing posterior\.q2\.median"):
        run_ttv_analysis(workspace)
    assert not (workspace.path / "outputs" / "ttv_analysis_results.json").exists()

    path.write_text(json.dumps(_posterior_artifact(q1=1.1)), encoding="utf-8")

    with pytest.raises(RuntimeError, match="q1 and q2"):
        run_ttv_analysis(workspace)
    assert not (workspace.path / "outputs" / "ttv_analysis_results.json").exists()

    path.write_text(
        '{"work_package":"MCMC_TRANSIT_FIT","source":"candidate-data","signal":null,'
        '"posterior":{},"posterior":{}}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        run_ttv_analysis(workspace)
    assert not (workspace.path / "outputs" / "ttv_analysis_results.json").exists()


def test_ttv_recomputes_summary_fields_and_rejects_tampered_timing_data(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "ttv-summary-recompute")
    _write_transit_fit(workspace, _posterior_artifact())
    table, ephemeris, stellar = _ttv_inputs()
    _patch_ttv_inputs(monkeypatch, table, ephemeris, stellar)
    analysis = _single_epoch_timing_analysis()
    monkeypatch.setattr(
        "exonym.ttv.transit_timing_analysis", lambda *_args, **_kwargs: analysis
    )

    analysis["oc_rms_minutes"] = 99.0
    with pytest.raises(RuntimeError, match="oc_rms_minutes does not match its recomputed value"):
        run_ttv_analysis(workspace)
    assert not (workspace.path / "outputs" / "ttv_analysis_results.json").exists()

    analysis = _single_epoch_timing_analysis()
    analysis["oc_minutes"] = [float("nan")]
    with pytest.raises(RuntimeError, match="oc_minutes\\[0\\] must be finite"):
        run_ttv_analysis(workspace)
    assert not (workspace.path / "outputs" / "ttv_analysis_results.json").exists()


def test_ttv_nonlinear_replay_uses_explicit_relative_tolerance():
    reported = {"model": {"decay_timescale_days": 2.729994502e9}}
    expected = {"model": {"decay_timescale_days": 2.729994545e9}}

    assert not _timing_values_agree(reported, expected, absolute_tolerance=1e-6)
    assert _timing_values_agree(
        reported,
        expected,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-7,
    )


def test_ttv_rejects_stale_fitted_ephemeris(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "ttv-template-stale-ephemeris")
    stale = _posterior_artifact()
    stale["ephemeris"]["period_days"] = 2.5
    _write_transit_fit(workspace, stale)
    table, ephemeris, stellar = _ttv_inputs()
    _patch_ttv_inputs(monkeypatch, table, ephemeris, stellar)

    with pytest.raises(RuntimeError, match="stale fitted period_days"):
        run_ttv_analysis(workspace)
    assert not (workspace.path / "outputs" / "ttv_analysis_results.json").exists()


def test_ttv_rejects_eccentric_posterior_template(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "ttv-template-eccentric")
    eccentric = _posterior_artifact()
    eccentric["parameter_names"] = eccentric["parameter_names"] + ["sqe_cosw", "sqe_sinw"]
    _write_transit_fit(workspace, eccentric)
    table, ephemeris, stellar = _ttv_inputs()
    _patch_ttv_inputs(monkeypatch, table, ephemeris, stellar)

    with pytest.raises(RuntimeError, match="eccentric fit"):
        run_ttv_analysis(workspace)
    assert not (workspace.path / "outputs" / "ttv_analysis_results.json").exists()


def test_ttv_rejects_missing_parameter_names_contract(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "ttv-template-missing-names")
    incomplete = _posterior_artifact()
    incomplete.pop("parameter_names")
    _write_transit_fit(workspace, incomplete)
    table, ephemeris, stellar = _ttv_inputs()
    _patch_ttv_inputs(monkeypatch, table, ephemeris, stellar)

    with pytest.raises(RuntimeError, match="parameter_names contract"):
        run_ttv_analysis(workspace)
    assert not (workspace.path / "outputs" / "ttv_analysis_results.json").exists()


def test_ttv_accepts_nested_sampling_template_with_provenance(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "ttv-template-nested")
    artifact_path = _write_transit_fit(
        workspace, {**_posterior_artifact(), "work_package": "NESTED_TRANSIT_FIT"}
    )
    table, ephemeris, stellar = _ttv_inputs()
    _patch_ttv_inputs(monkeypatch, table, ephemeris, stellar)
    monkeypatch.setattr(
        "exonym.ttv.transit_timing_analysis", lambda *_args, **_kwargs: _empty_timing_analysis()
    )

    output_path = run_ttv_analysis(workspace)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["input_provenance"]["transit_fit_artifact"]["path"] == (
        "outputs/mcmc_transit_fit.json"
    )
    assert payload["timing"]["template"]["artifact"]["sha256"] == hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
