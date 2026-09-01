"""Unit tests for TREX native statistical validation engine.

Target-neutral synthetic tests exercising all TREX modules.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from exonym.vetting.trex.constants import Msun, Rsun, Rearth, G, au, pi
from exonym.vetting.trex._numerics import _log_mean_exp, _normalize_probabilities
from exonym.vetting.trex.funcs import (
    stellar_relations, J_Ks_to_Tmag, companion_flux_ratio, dilute_flux,
    separation_at_contrast, delta_mag_to_flux_ratio, semi_major_axis_cgs,
    a_over_Rs, impact_parameter,
)
from exonym.vetting.trex.priors import (
    sample_rp, sample_inc, sample_ecc, sample_w, sample_q,
    lnprior_bound, lnprior_background,
)
from exonym.vetting.trex.target import TargetScene
from exonym.vetting.trex.diagnostics import TrexResult, generate_diagnostics


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


def test_sample_q_solar():
    rng = np.random.default_rng(42)
    q = sample_q(rng.random(500), M_s=1.0)
    assert np.all((q >= 0.1) & (q <= 1.0))


def test_lnprior_bound_finite():
    lnp = lnprior_bound(1.0, np.array([1.0]), np.array([0.1, 2.0]), np.array([0.0, 3.0]), 5.0)
    assert np.all(np.isfinite(lnp))


def test_lnprior_background_finite():
    lnp = lnprior_background(100, np.array([1.0]), np.array([0.1, 2.0]), np.array([0.0, 3.0]))
    assert np.all(np.isfinite(lnp))


# ============================================================================
# TargetScene
# ============================================================================

def test_target_scene_basic():
    scene = TargetScene(tic_id=123, ra_deg=90.0, dec_deg=-60.0)
    assert scene.tic_id == 123
    assert scene.n_neighbors == 0


def test_target_scene_neighbors():
    scene = TargetScene(
        tic_id=123, ra_deg=90.0, dec_deg=-60.0,
        resolved_neighbors=[{"M_s": 0.5, "R_s": 0.45, "delta_mag": 2.0}],
    )
    assert scene.n_neighbors == 1
    m, r, _ = scene.neighbor_masses_radii()
    assert m[0] == 0.5
    assert r[0] == 0.45


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