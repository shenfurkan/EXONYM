"""Target-neutral unit tests for Morris (1985) physical ellipsoidal variation modeling."""

from __future__ import annotations

import math
import pytest

from exonym.vetting.ellipsoidal import (
    ellipsoidal_gate,
    ellipsoidal_variation_amplitude_ppm,
    gravity_darkening_exponent,
    morris_ellipsoidal_coefficient,
)


def test_gravity_darkening_envelope_regimes():
    """Verify convective vs radiative gravity darkening transition at ~6500 K."""
    assert gravity_darkening_exponent(5500.0) == 0.32
    assert gravity_darkening_exponent(6499.0) == 0.32
    assert gravity_darkening_exponent(6500.0) == 1.0
    assert gravity_darkening_exponent(8500.0) == 1.0

    with pytest.raises(ValueError, match="positive finite"):
        gravity_darkening_exponent(-100.0)
    with pytest.raises(ValueError, match="positive finite"):
        gravity_darkening_exponent(float("nan"))


def test_morris_coefficient_solar_twin():
    """Verify Morris (1985) coefficient for solar convective envelope (u=0.6, g=0.32)."""
    alpha = morris_ellipsoidal_coefficient(u_linear=0.6, g_darkening=0.32)
    # alpha = 0.15 * (15 + 0.6) * (1 + 0.32) / (3 - 0.6) = 3.0888 / 2.40 = 1.287
    assert abs(alpha - 1.287) < 0.001


def test_morris_coefficient_hot_star():
    """Verify Morris (1985) coefficient for hot radiative envelope (u=0.4, g=1.0)."""
    alpha = morris_ellipsoidal_coefficient(u_linear=0.4, g_darkening=1.0)
    # alpha = 0.15 * (15 + 0.4) * (1 + 1.0) / (3 - 0.4) = 4.62 / 2.60 = 1.7769
    assert abs(alpha - 1.7769) < 0.001


def test_morris_coefficient_domain_guards():
    """Verify domain error checking for limb and gravity darkening."""
    with pytest.raises(ValueError, match="0 <= u < 3"):
        morris_ellipsoidal_coefficient(u_linear=-0.1)
    with pytest.raises(ValueError, match="0 <= u < 3"):
        morris_ellipsoidal_coefficient(u_linear=3.5)
    with pytest.raises(ValueError, match="non-negative"):
        morris_ellipsoidal_coefficient(g_darkening=-0.5)


def test_ellipsoidal_variation_dynamic_temperature_scaling():
    """Verify dynamic Teff scaling in ellipsoidal amplitude."""
    # System: 0.1 Msun companion around 1.0 Msun host, 1.0 Rsun, a = 0.05 AU (hot Jupiter/M-dwarf regime)
    amp_cool = ellipsoidal_variation_amplitude_ppm(
        m_companion_solar=0.1,
        m_host_solar=1.0,
        r_host_solar=1.0,
        semi_major_axis_au=0.05,
        teff_k=5000.0,
        u_linear=0.6,
    )
    amp_hot = ellipsoidal_variation_amplitude_ppm(
        m_companion_solar=0.1,
        m_host_solar=1.0,
        r_host_solar=1.0,
        semi_major_axis_au=0.05,
        teff_k=7500.0,
        u_linear=0.4,
    )
    # Radiative star has significantly higher tidal amplitude due to g=1.0
    assert amp_hot > amp_cool
    assert amp_cool > 0.0


def test_ellipsoidal_gate_pass_and_veto():
    """Verify threshold gating logic."""
    passed, amp = ellipsoidal_gate(45.0, threshold_ppm=100.0)
    assert passed is True
    assert amp == 45.0

    failed, amp = ellipsoidal_gate(150.0, threshold_ppm=100.0)
    assert failed is False
    assert amp == 150.0

    with pytest.raises(ValueError, match="non-negative"):
        ellipsoidal_gate(-1.0)
