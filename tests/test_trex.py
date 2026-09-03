"""Unit tests for TREX native statistical validation engine.

Target-neutral synthetic tests exercising all TREX modules.
"""

from __future__ import annotations

import math
import inspect

import numpy as np
import pytest
from scipy.special import ndtr

from exonym.vetting.trex.constants import Msun, Rsun, Rearth, G, au, pi
from exonym.vetting.trex._numerics import _log_mean_exp, _normalize_probabilities
from exonym.vetting.trex.funcs import (
    stellar_relations, J_Ks_to_Tmag, companion_flux_ratio, dilute_flux,
    separation_at_contrast, delta_mag_to_flux_ratio, semi_major_axis_cgs,
    a_over_Rs, impact_parameter, secondary_eclipse_phase,
    tess_surface_brightness_ratio,
)
from exonym.vetting.trex.priors import (
    sample_rp, sample_inc, sample_ecc, sample_w, sample_q,
    lnprior_bound, lnprior_background,
)
from exonym.vetting.trex.target import TargetScene
from exonym.vetting.trex.diagnostics import TrexResult, generate_diagnostics
from exonym.vetting.trex.marginal_likelihoods import _eval_scenario, calc_target_evidences
from exonym.vetting.trex.likelihoods import simulate_EB


# ============================================================================
# Constants
# ============================================================================

def test_physical_constants_are_finite():
    assert math.isfinite(Msun) and Msun > 0
    assert math.isfinite(Rsun) and Rsun > 0
    assert math.isfinite(Rearth) and Rearth > 0
    assert math.isfinite(G) and G > 0
    from exonym.vetting.trex.constants import SOLAR_MEAN_DENSITY_G_CM3
    assert 1.0 < SOLAR_MEAN_DENSITY_G_CM3 < 2.0


# ============================================================================
# Numerics
# ============================================================================

def test_log_mean_exp_all_finite():
    lnw = np.full(10, -5.0)
    result = _log_mean_exp(lnw, N_total=10)
    assert result == pytest.approx(-5.0, rel=1e-12)


def test_log_mean_exp_with_neginf():
    lnw = np.array([-1.0, -2.0, -np.inf, -3.0])
    expected = math.log((math.exp(-1) + math.exp(-2) + math.exp(-3)) / 4.0)
    result = _log_mean_exp(lnw, N_total=4)
    assert result == pytest.approx(expected, rel=1e-12)


def test_log_mean_exp_all_neginf():
    result = _log_mean_exp(np.full(5, -np.inf), N_total=5)
    assert result == -np.inf


def test_log_mean_exp_posinf():
    result = _log_mean_exp(np.array([-1.0, np.inf, -2.0]), N_total=3)
    assert result == np.inf


def test_log_mean_exp_size_mismatch():
    with pytest.raises(ValueError, match="N_total"):
        _log_mean_exp(np.array([1.0, 2.0]), N_total=5)


def test_normalize_probabilities_ok():
    lnZ = np.array([0.0, -2.0, -10.0])
    probs, status = _normalize_probabilities(lnZ)
    assert status == "ok"
    assert np.sum(probs) == pytest.approx(1.0)


def test_normalize_probabilities_anomaly():
    probs, status = _normalize_probabilities(np.array([-1.0, np.nan]))
    assert status == "anomaly"


# ============================================================================
# Funcs
# ============================================================================

def test_stellar_relations_solar():
    r, t = stellar_relations(np.array([1.0]))
    assert 1.0 < r[0] < 1.2
    assert 5500 < t[0] < 6000


def test_stellar_relations_vectorized():
    masses = np.array([0.3, 1.0, 2.0])
    r, _ = stellar_relations(masses)
    assert len(r) == 3
    assert r[0] < r[1] < r[2]


def test_stellar_relations_clamp():
    r, _ = stellar_relations(np.array([1.0]), max_radii=np.array([0.5]))
    assert r[0] == pytest.approx(0.5)


def test_tmag_conversion():
    tmag = J_Ks_to_Tmag(np.array([4.5]), np.array([4.0]))
    expected = 4.5 + 1.22163 * 0.125 - 1.74299 * 0.25 + 1.89115 * 0.5 + 0.0563
    assert tmag[0] == pytest.approx(expected)


@pytest.mark.parametrize("colour", [-0.1, 0.7, 1.0])
def test_tmag_relation_is_continuous_at_piecewise_edges(colour):
    epsilon = 1e-9
    lower = J_Ks_to_Tmag(np.array([colour - epsilon]), np.array([0.0]))[0]
    upper = J_Ks_to_Tmag(np.array([colour + epsilon]), np.array([0.0]))[0]
    assert lower == pytest.approx(upper, abs=1e-7)


def test_tess_surface_brightness_uses_temperature():
    ratio = tess_surface_brightness_ratio(np.array([4000.0]), np.array([6000.0]))
    assert 0.0 < ratio[0] < 1.0


def test_flux_helpers():
    assert companion_flux_ratio(np.array([0.5]))[0] == pytest.approx(1.0)
    diluted = dilute_flux(np.array([0.99, 0.98, 1.0]), 0.0)
    np.testing.assert_array_equal(diluted, np.array([0.99, 0.98, 1.0]))


def test_contrast_roundtrip():
    dm = np.array([1.0, 2.5, 5.0])
    fr = delta_mag_to_flux_ratio(dm)
    from exonym.vetting.trex.funcs import flux_ratio_to_delta_mag
    dm_back = flux_ratio_to_delta_mag(fr)
    np.testing.assert_allclose(dm, dm_back, rtol=1e-12)


def test_keplerian():
    a = semi_major_axis_cgs(365.25, 1.989e33)
    assert a / au == pytest.approx(1.0, rel=0.01)
    a_rs = a_over_Rs(365.25, 1.989e33, 6.957e10)
    assert a_rs == pytest.approx(215.0, rel=0.01)
    b = impact_parameter(90.0, 10.0)
    assert b == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Priors
# ============================================================================

def test_sample_rp_range():
    rng = np.random.default_rng(42)
    rp = sample_rp(rng.random(2000), np.ones(2000))
    assert np.all(rp >= 0.5)
    assert np.all(rp <= 20.0)
    assert np.median(rp) < 5.0  # Small planets favored


def test_sample_inc_isotropic():
    rng = np.random.default_rng(42)
    inc = sample_inc(rng.random(2000))
    assert 55.0 < np.median(inc) < 65.0


def test_sample_ecc_planet():
    rng = np.random.default_rng(42)
    ecc = sample_ecc(rng.random(500), planet=True, P_orb=5.0)
    assert np.all((ecc >= 0) & (ecc < 1))
    assert 0.1 < np.median(ecc) < 0.4


@pytest.mark.parametrize("planet, period_days", [(True, 5.0), (False, 5.0), (False, 20.0)])
def test_sample_ecc_is_deterministic_for_supplied_draws(planet, period_days):
    draws = np.array([0.01, 0.2, 0.5, 0.9])

    first = sample_ecc(draws, planet=planet, P_orb=period_days)
    second = sample_ecc(draws, planet=planet, P_orb=period_days)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("draws", [np.array([-0.1]), np.array([1.0]), np.array([np.nan])])
def test_sample_ecc_rejects_invalid_draws(draws):
    with pytest.raises(ValueError, match="finite values"):
        sample_ecc(draws, planet=True, P_orb=5.0)


def test_sample_q_solar():
    rng = np.random.default_rng(42)
    q = sample_q(rng.random(500), M_s=1.0)
    assert np.all((q >= 0.1) & (q <= 1.0))


def test_eb_scenarios_use_empirical_companion_radii(monkeypatch):
    import exonym.vetting.trex.marginal_likelihoods as marginal_likelihoods

    captured = []
    monkeypatch.setattr(
        marginal_likelihoods,
        "lnL_EB",
        lambda *_args, **kwargs: captured.append((_args[3], _args[4])) or 0.0,
    )
    monkeypatch.setattr(marginal_likelihoods, "sample_q", lambda draws, _mass: np.full(draws.size, 0.2))
    monkeypatch.setattr(marginal_likelihoods, "sample_inc", lambda draws: np.full(draws.size, 90.0))
    monkeypatch.setattr(marginal_likelihoods, "sample_ecc", lambda draws, **_kwargs: np.zeros(draws.size))
    monkeypatch.setattr(marginal_likelihoods, "sample_w", lambda draws: np.full(draws.size, 90.0))

    _eval_scenario(
        np.array([0.0]), np.array([1.0]), 0.01, 5.0, 1.0, 1.0, 0.4, 0.2,
        2, is_planet=False, is_EB=True, use_2xP=False, exptime_days=0.01,
        rng=np.random.default_rng(1),
    )

    expected_radius, _ = stellar_relations(np.array([0.2]))
    assert [item[0] for item in captured] == pytest.approx([float(expected_radius[0])] * 2)
    assert captured[0][0] != pytest.approx(0.2)
    assert 0.0 < captured[0][1] < 1.0


@pytest.mark.parametrize("exptime_days", [0.0, -0.01, np.nan, np.inf])
def test_evidence_requires_finite_positive_exptime(exptime_days):
    with pytest.raises(ValueError, match="exptime_days"):
        calc_target_evidences(
            np.array([0.0]), np.array([1.0]), 0.01, 5.0, 1000.0, 1.0, 1.0,
            0.4, 0.2, exptime_days=exptime_days,
        )


def test_evidence_forwards_exptime_to_every_scenario(monkeypatch):
    import exonym.vetting.trex.marginal_likelihoods as marginal_likelihoods

    captured = []
    monkeypatch.setattr(
        marginal_likelihoods,
        "_eval_scenario",
        lambda *_args, **kwargs: captured.append(_args[12]) or -1.0,
    )
    calc_target_evidences(
        np.array([0.0]), np.array([1.0]), 0.01, 5.0, 1000.0, 1.0, 1.0,
        0.4, 0.2, exptime_days=0.01,
    )
    assert captured
    assert all(value == pytest.approx(0.01) for value in captured)


def test_exptime_is_required_at_each_trex_layer():
    from exonym.vetting.trex import run_trex_vetting
    from exonym.vetting.trex.likelihoods import _batman_transit, lnL_EB, lnL_TP, simulate_TP

    for function in (
        run_trex_vetting,
        calc_target_evidences,
        _eval_scenario,
        _batman_transit,
        simulate_TP,
        simulate_EB,
        lnL_TP,
        lnL_EB,
    ):
        assert inspect.signature(function).parameters["exptime_days"].default is inspect.Parameter.empty


def test_secondary_eclipse_phase_uses_keplerian_mean_anomalies():
    assert secondary_eclipse_phase(0.0, 0.0) == pytest.approx(0.5)
    assert secondary_eclipse_phase(0.3, 0.0) != pytest.approx(0.5)


def test_eb_model_evaluates_secondary_at_observed_times(monkeypatch):
    import exonym.vetting.trex.likelihoods as likelihoods

    calls = []
    observed_time = np.array([-0.2, 0.0, 0.2])
    monkeypatch.setattr(
        likelihoods,
        "_batman_transit",
        lambda time, *_args, **kwargs: calls.append((time, kwargs.get("t0_days", 0.0))) or np.full_like(time, 0.8),
    )

    _, phase = simulate_EB(
        observed_time, 0.6, 0.25, 5.0, 89.0, 1.0e12, 1.0, 0.4, 0.2,
        0.01, ecc=0.3, argp_deg=0.0,
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1][0], observed_time)
    assert calls[1][1] == pytest.approx(phase * 5.0)


def test_eb_model_preserves_an_exact_equal_radius_ratio():
    """Batman must receive the physical equal-radius EB geometry unchanged."""
    pytest.importorskip("batman")

    flux, _ = simulate_EB(
        np.array([-0.01, 0.0, 0.01]), 1.0, 0.25, 5.0, 89.0, 1.0e12,
        1.0, 0.4, 0.2, 0.01,
    )

    assert np.all(np.isfinite(flux))


def test_lnprior_bound_finite():
    lnp = lnprior_bound(1.0, np.array([1.0]), np.array([0.1, 2.0]), np.array([0.0, 3.0]), 5.0)
    assert np.all(np.isfinite(lnp))


@pytest.mark.parametrize("primary_mass_solar", [0.075, 0.15, 0.30, 0.60])
def test_lnprior_bound_uses_winters_m_dwarf_separation_cdf(primary_mass_solar):
    """Winters et al. (2019) fixes the M-dwarf normalization and log-a CDF."""
    parallax_mas = 10.0
    separation_arcsec = np.array([2.0, 2.0])
    maximum_separation_au = separation_arcsec[0] * (1000.0 / parallax_mas)
    expected_probability = 0.268 * ndtr(
        (np.log10(maximum_separation_au) - np.log10(20.0)) / 1.16
    )

    log_prior = lnprior_bound(
        primary_mass_solar, np.array([1.0]), separation_arcsec,
        np.array([1.0, 2.0]), parallax_mas,
    )

    np.testing.assert_allclose(np.exp(log_prior), expected_probability)


def test_lnprior_bound_m_dwarf_prior_increases_with_searchable_separation():
    """The retained Winters Gaussian CDF must be monotone in physical separation."""
    log_prior = lnprior_bound(
        0.30, np.array([1.0, 2.0, 3.0]), np.array([0.1, 1.0, 10.0]),
        np.array([1.0, 2.0, 3.0]), 10.0,
    )

    assert np.all(np.diff(np.exp(log_prior)) > 0.0)


def test_lnprior_bound_rejects_substellar_primary_below_winters_domain():
    with pytest.raises(ValueError, match=r"Winters et al. \(2019\)"):
        lnprior_bound(0.07, np.array([1.0]), np.array([0.1, 2.0]), np.array([0.0, 3.0]), 5.0)


@pytest.mark.parametrize("parallax_mas", [0.0, -1.0, np.nan, np.inf])
def test_lnprior_bound_rejects_unusable_parallax(parallax_mas):
    with pytest.raises(ValueError, match="finite positive parallax"):
        lnprior_bound(
            1.0, np.array([1.0]), np.array([0.1, 2.0]), np.array([0.0, 3.0]),
            parallax_mas,
        )


def test_lnprior_background_finite():
    lnp = lnprior_background(100, np.array([1.0]), np.array([0.1, 2.0]), np.array([0.0, 3.0]))
    assert np.all(np.isfinite(lnp))


# ============================================================================
# TargetScene
# ============================================================================

def _target_scene_kwargs(tmp_path):
    background = tmp_path / "synthetic_trilegal.csv"
    background.write_text("synthetic background population\n", encoding="utf-8")
    import hashlib

    return {
        "tic_id": 123,
        "ra_deg": 90.0,
        "dec_deg": -60.0,
        "M_s_Msun": 1.0,
        "R_s_Rsun": 1.0,
        "Teff_K": 5700.0,
        "Tmag": 10.0,
        "plx_mas": 2.0,
        "sectors": [1],
        "contrast_separations": np.array([0.1, 1.0]),
        "contrast_values": np.array([2.0, 5.0]),
        "resolved_neighbors": [],
        "N_background": 1,
        "trilegal_cache": background,
        "background_sha256": hashlib.sha256(background.read_bytes()).hexdigest(),
    }


def test_target_scene_basic(tmp_path):
    scene = TargetScene(**_target_scene_kwargs(tmp_path))
    assert scene.tic_id == 123
    assert scene.n_neighbors == 0


def test_target_scene_neighbors(tmp_path):
    kwargs = _target_scene_kwargs(tmp_path)
    kwargs["resolved_neighbors"] = [
        {
            "source_id": "synthetic-neighbor",
            "M_s": 0.5,
            "R_s": 0.45,
            "delta_mag": 2.0,
            "separation_arcsec": 1.0,
        }
    ]
    scene = TargetScene(**kwargs)
    assert scene.n_neighbors == 1
    m, r, _ = scene.neighbor_masses_radii()
    assert m[0] == 0.5
    assert r[0] == 0.45


def test_target_scene_rejects_changed_background(tmp_path):
    scene = TargetScene(**_target_scene_kwargs(tmp_path))
    scene.trilegal_cache.write_text("changed background population\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after scene construction"):
        scene.verify_background()


# ============================================================================
# Diagnostics
# ============================================================================

def test_generate_diagnostics_full():
    lnZ = np.full(15, -100.0)
    lnZ[0] = -10.0
    result = generate_diagnostics(lnZ)
    assert result.fpp is not None
    assert 0 <= result.fpp <= 1
    assert not result.degenerate
    assert not result.claim_eligible
    assert len(result.diagnostics) > 0


def test_generate_diagnostics_small_array():
    lnZ = np.array([-10.0, -12.0, -8.0, -15.0])
    result = generate_diagnostics(lnZ)
    assert not result.degenerate


def test_trex_result_top_scenarios():
    result = TrexResult()
    result.probs = np.array([0.8, 0.15, 0.03, 0.02])
    top = result.top_scenarios(2)
    assert len(top) == 2
    assert top[0][1] > top[1][1]
