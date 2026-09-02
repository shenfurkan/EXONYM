"""Target-neutral unit and stress tests for dynamic stellar activity period search bounds."""

from __future__ import annotations

import numpy as np
import pytest

from exonym.activity import gls_periodogram, sinusoid_amplitude_ppm


def test_gls_fast_rotator_recovery():
    """Verify that rapid rotators (P < 1.0 day) are successfully recovered dynamically."""
    rng = np.random.default_rng(42)
    # 2-minute cadence for 10 days
    time = np.linspace(0.0, 10.0, 7200)
    true_period = 0.45  # days (< 1.0 day default)
    true_amp_ppm = 2500.0
    flux = 1.0 + (true_amp_ppm * 1e-6) * np.sin(2.0 * np.pi * time / true_period)
    flux += rng.normal(0.0, 200e-6, size=time.size)

    periods, powers, fap = gls_periodogram(time, flux)
    best_period = float(periods[np.argmax(powers)])

    assert abs(best_period - true_period) < 0.02
    assert fap < 1e-5


def test_gls_slow_rotator_recovery():
    """Verify that slow rotators (P > 20.0 days) are recovered on multi-sector baselines with >= 2 cycles."""
    rng = np.random.default_rng(123)
    # 30-minute cadence for 80 days (80 / 32 = 2.5 cycles)
    time = np.linspace(0.0, 80.0, 3840)
    true_period = 32.0  # days (> 20.0 day default)
    true_amp_ppm = 5000.0
    flux = 1.0 + (true_amp_ppm * 1e-6) * np.cos(2.0 * np.pi * time / true_period)
    flux += rng.normal(0.0, 300e-6, size=time.size)

    periods, powers, fap = gls_periodogram(time, flux)
    best_period = float(periods[np.argmax(powers)])

    assert abs(best_period - true_period) < 0.8
    assert fap < 1e-5


def test_gls_two_cycle_boundary_enforcement():
    """Verify that the search space strictly enforces the >= 2.0 full cycle rule."""
    # 20-day baseline -> maximum period evaluated should be 20.0 / 2.0 = 10.0 days
    time = np.linspace(0.0, 20.0, 1000)
    flux = 1.0 + 0.001 * np.sin(2.0 * np.pi * time / 5.0)
    periods, powers, _ = gls_periodogram(time, flux)
    assert np.max(periods) <= 10.001


def test_gls_correlated_noise_and_gaps_stress_test():
    """Stress test GLS with red-noise drift and multi-day observation gaps."""
    rng = np.random.default_rng(999)
    # Segment 1: days 0 to 12
    t1 = np.linspace(0.0, 12.0, 1000)
    # 4-day gap (days 12 to 16)
    # Segment 2: days 16 to 28
    t2 = np.linspace(16.0, 28.0, 1000)
    time = np.concatenate((t1, t2))
    
    true_period = 2.4  # days
    signal = 0.002 * np.sin(2.0 * np.pi * time / true_period)
    
    # Red-noise instrumental polynomial drift
    drift = 0.0005 * ((time - 14.0) / 14.0) ** 2
    # White noise component
    noise = rng.normal(0.0, 300e-6, size=time.size)
    flux = 1.0 + signal + drift + noise

    periods, powers, fap = gls_periodogram(time, flux)
    best_period = float(periods[np.argmax(powers)])

    # Peak must be within 5% of true period despite gap and drift
    assert abs(best_period - true_period) < 0.15
    assert fap < 1e-4


def test_gls_explicit_override():
    """Verify that explicit period bounds override dynamic boundaries."""
    time = np.linspace(0.0, 20.0, 1000)
    flux = 1.0 + 0.001 * np.sin(2.0 * np.pi * time / 3.5)
    periods, powers, _ = gls_periodogram(
        time, flux, period_min_days=2.0, period_max_days=5.0
    )
    assert np.min(periods) >= 1.99
    assert np.max(periods) <= 5.01


def test_gls_invalid_bounds_rejected():
    """Verify that invalid period search ranges raise ValueError."""
    time = np.linspace(0.0, 10.0, 500)
    flux = np.ones(500)
    with pytest.raises(ValueError, match="invalid period search bounds"):
        gls_periodogram(time, flux, period_min_days=5.0, period_max_days=2.0)
