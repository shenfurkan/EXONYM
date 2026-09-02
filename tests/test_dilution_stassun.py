"""Target-neutral unit tests for Stassun et al. (2019, TIC v8) TESS magnitude conversion."""

from __future__ import annotations

import math
import pytest

from exonym.dilution import (
    STASSUN_BP_RP_MAX,
    STASSUN_BP_RP_MIN,
    gaia_contamination_factor,
    gaia_g_to_tess_mag,
)


def test_gaia_g_to_tess_solar_twin():
    """Verify conversion for Sun-like FGK star (G_BP - G_RP ~ 0.82)."""
    g_mag = 10.0
    color = 0.82
    t_mag = gaia_g_to_tess_mag(g_mag, color)
    # Stassun 2019: T ~ G - 0.43 for solar twins
    assert abs((t_mag - g_mag) - (-0.43)) < 0.02


def test_gaia_g_to_tess_cool_m_dwarf():
    """Verify conversion for cool M dwarf star (G_BP - G_RP ~ 3.5)."""
    g_mag = 15.0
    color = 3.5
    t_mag = gaia_g_to_tess_mag(g_mag, color)
    # M dwarfs are much brighter in red/IR TESS band: delta T - G ~ -1.3 mag
    assert t_mag < g_mag - 1.0
    assert abs((t_mag - g_mag) - (-1.319)) < 0.05


def test_gaia_g_to_tess_hot_star():
    """Verify conversion for hot A/B star (G_BP - G_RP ~ 0.0)."""
    g_mag = 8.0
    color = 0.0
    t_mag = gaia_g_to_tess_mag(g_mag, color)
    # Polynomial constant term is +0.0324473
    assert abs((t_mag - g_mag) - 0.03245) < 0.001


def test_gaia_g_to_tess_rejects_outside_stassun_calibration():
    """The conversion must not extrapolate the published color calibration."""
    g_mag = 12.0
    assert gaia_g_to_tess_mag(g_mag, STASSUN_BP_RP_MAX + 0.01) is None
    assert gaia_g_to_tess_mag(g_mag, STASSUN_BP_RP_MIN - 0.01) is None


def test_gaia_g_to_tess_missing_color_is_unavailable():
    """A colorless Gaia row cannot support a TESS-band conversion."""
    g_mag = 11.5
    assert gaia_g_to_tess_mag(g_mag, None) is None
    assert gaia_g_to_tess_mag(g_mag, float("nan")) is None


def test_gaia_contamination_factor_with_stassun_colors():
    """Verify neighborhood contamination factor evaluation with varied neighbor colors."""
    neighbors = [
        {"separation_arcsec": 0.0, "is_target": True, "phot_g_mean_mag": 10.0, "bp_rp": 0.82},
        {"separation_arcsec": 15.0, "is_target": False, "phot_g_mean_mag": 12.0, "bp_rp": 3.0},
        {"separation_arcsec": 30.0, "is_target": False, "phot_g_mean_mag": 14.0, "bp_rp": 1.2},
        {"separation_arcsec": 90.0, "is_target": False, "phot_g_mean_mag": 8.0, "bp_rp": 0.5},  # Outside 60"
    ]
    summary = gaia_contamination_factor(
        neighbors,
        search_radius_arcsec=60.0,
        target_g_mag=10.0,
        target_bp_rp_color=0.82,
    )
    assert summary["n_neighbors_in_aperture"] == 2
    assert summary["availability"] == "available"
    assert summary["target_g_mag"] == 10.0
    assert summary["contamination_ratio"] > 0.0


def test_gaia_contamination_is_unavailable_when_a_neighbor_lacks_bp_rp():
    summary = gaia_contamination_factor(
        [{"separation_arcsec": 15.0, "is_target": False, "phot_g_mean_mag": 12.0, "bp_rp": None}],
        target_g_mag=10.0,
        target_bp_rp_color=0.82,
    )

    assert summary["availability"] == "unavailable"
    assert summary["contamination_factor"] is None
    assert summary["n_neighbors_omitted"] == 1
