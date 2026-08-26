"""Domain and contract regressions for phase-curve diagnostics."""

import json
import math

import numpy as np
import pytest


def _valid_phasecurve_inputs():
    period_days = 1.0
    epoch_btjd = 100.0
    time = np.linspace(100.001, 107.999, 512)
    phase_days = ((time - epoch_btjd + 0.5 * period_days) % period_days) - 0.5 * period_days
    flux = 1.0 + 40e-6 * (-np.cos(2.0 * np.pi * phase_days / period_days))
    return (
        time,
        flux,
        np.full(time.size, 100e-6),
        np.ones(time.size, dtype=int),
        {"period_days": period_days, "epoch_btjd": epoch_btjd, "duration_days": 0.04},
    )


@pytest.mark.parametrize("block_days", [0.0, -0.5, np.nan, np.inf])
def test_phasecurve_rejects_invalid_covariance_block_width(block_days):
    from exonym.phasecurve import fit_phase_curve_components

    with pytest.raises(ValueError, match="block_days must be finite and positive"):
        fit_phase_curve_components(*_valid_phasecurve_inputs(), block_days=block_days)


def test_phasecurve_rejects_secondary_box_that_is_constant_over_an_orbit():
    from exonym.phasecurve import fit_phase_curve_components

    with pytest.raises(ValueError, match="shorter than the orbital period"):
        fit_phase_curve_components(
            *_valid_phasecurve_inputs(),
            secondary_eclipse_duration_days=1.0,
        )


def test_phasecurve_single_cluster_has_undefined_component_significance():
    from exonym.phasecurve import fit_phase_curve_components

    result = fit_phase_curve_components(*_valid_phasecurve_inputs(), block_days=100.0)

    assert result["n_covariance_clusters"] == 1
    assert result["status"] == "undefined_component_significance"
    assert all(
        component["block_robust_error_ppm"] == 0.0
        and component["significance_sigma"] is None
        and component["three_sigma_absolute_upper_bound_ppm"] is None
        for component in result["components"].values()
    )


def test_secondary_template_requires_an_integral_represented_sample_count():
    from exonym.phasecurve import _posterior_secondary_eclipse_template

    phase_days = np.array([0.4, 0.5, 0.6])
    phase_samples = np.array([0.45, 0.55])
    duration_samples = np.array([0.1, 0.1])

    with pytest.raises(ValueError, match="sample count must be an integer"):
        _posterior_secondary_eclipse_template(
            phase_days, 1.0, phase_samples, duration_samples, total_samples=2.0
        )
    with pytest.raises(ValueError, match="invalid dimensions"):
        _posterior_secondary_eclipse_template(
            phase_days, 1.0, phase_samples, duration_samples, total_samples=1
        )


def test_circular_phase_summary_never_serializes_one_as_a_phase_fraction():
    from exonym.phasecurve import _circular_phase_summary

    summary = _circular_phase_summary(np.array([0.999999999]))

    assert all(0.0 <= summary[key] < 1.0 for key in ("median", "p16", "p84"))
    assert summary["median"] == 0.0


def test_eccentric_secondary_control_uses_named_reordered_chain_coordinates(tmp_path):
    from exonym.phasecurve import resolve_secondary_eclipse_control
    from exonym.transit_fit import PARAMETER_NAMES_ECCENTRIC
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "phasecurve-reordered-contract")
    ephemeris = {"period_days": 3.0, "epoch_btjd": 100.0, "duration_days": 0.1}
    parameter_names = list(reversed(PARAMETER_NAMES_ECCENTRIC))
    values = {
        "rp_rs": 0.1,
        "log_rho_star": 0.0,
        "impact_parameter": 0.2,
        "baseline": 1.0,
        "log_jitter": -8.0,
        "q1": 0.3,
        "q2": 0.3,
        "sqe_cosw": math.sqrt(0.2),
        "sqe_sinw": 0.0,
    }
    chain = np.tile(
        np.asarray([values[name] for name in parameter_names], dtype=float),
        (16, 1),
    )
    outputs = workspace.path / "outputs"
    (outputs / "mcmc_transit_fit.json").write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "model": "batman quadratic limb darkening, stellar-density locked, eccentric orbit",
                "ephemeris": {"period_days": 3.0, "epoch_btjd": 100.0},
                "parameter_names": parameter_names,
            }
        ),
        encoding="utf-8",
    )
    np.save(str(outputs / "mcmc_transit_fit_chain.npy"), chain)

    arguments, report = resolve_secondary_eclipse_control(workspace, ephemeris)

    assert report["mode"] == "eccentric-posterior-marginalized-box-control"
    assert arguments["secondary_eclipse_phase"] == pytest.approx(0.5 + 0.4 / math.pi, abs=3e-3)
