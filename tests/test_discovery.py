"""Synthetic tests for deterministic survey robustness primitives."""

import json

import numpy as np
import pytest

from exonym.discovery import (
    detrend_by_sector,
    inject_box_transit,
    inverted_flux,
    recovered_period,
    injection_recovery_diagnostics,
    mask_box_transit,
    robustness_diagnostics,
    scrambled_flux,
    search_duration_grid,
)
from exonym.search import BLSSearchResult


def test_sector_detrending_is_deterministic_and_independent():
    time = np.concatenate((np.linspace(0.0, 5.0, 200), np.linspace(10.0, 15.0, 200)))
    sectors = np.concatenate((np.ones(200, dtype=int), np.full(200, 2, dtype=int)))
    flux = np.concatenate((1.0 + 0.01 * time[:200], 1.0 - 0.01 * (time[200:] - 10.0)))

    first = detrend_by_sector(time, flux, sectors, window_days=1.0)
    second = detrend_by_sector(time, flux, sectors, window_days=1.0)

    assert np.allclose(first, second)
    assert abs(np.median(first[:200]) - 1.0) < 0.001
    assert abs(np.median(first[200:]) - 1.0) < 0.001


def test_controls_are_deterministic_and_invert_flux():
    flux = np.array([0.99, 1.0, 1.01])

    assert np.allclose(inverted_flux(flux), np.array([1.01, 1.0, 0.99]))
    assert np.array_equal(scrambled_flux(flux, 5), scrambled_flux(flux, 5))


def test_box_injection_and_period_recovery():
    time = np.linspace(0.0, 20.0, 1000)
    injected = inject_box_transit(time, np.ones_like(time), 4.0, 1.0, 3.0, 1000.0)

    assert injected.min() == 0.999
    assert recovered_period(4.0, 4.01, tolerance=0.01)
    assert not recovered_period(4.0, 2.0, tolerance=0.01)


def test_mask_box_transit_removes_the_detected_event_window():
    # Arrange
    time = np.linspace(0.0, 12.0, 500)
    flux = inject_box_transit(time, np.ones_like(time), 3.0, 1.0, 2.0, 1000.0)

    # Act
    masked, masked_cadences = mask_box_transit(time, flux, 3.0, 1.0, 2.0)

    # Assert
    assert masked_cadences > 0
    assert np.count_nonzero(np.isnan(masked)) == masked_cadences


def test_duration_grid_returns_each_trial_and_best_result():
    time = np.linspace(0.0, 20.0, 800)
    flux = inject_box_transit(time, np.ones_like(time), 4.0, 1.0, 3.0, 3000.0)

    best, trials = search_duration_grid(
        time,
        flux,
        duration_grid_hours=[1.5, 3.0, 4.5],
        period_min_days=2.0,
        period_max_days=6.0,
        n_periods=100,
    )

    assert len(trials) == 3
    assert best.snr == max(trial["snr"] for trial in trials)


def test_robustness_diagnostics_records_all_variants_and_controls():
    time = np.linspace(0.0, 12.0, 500)
    sectors = np.ones(time.size, dtype=int)
    flux = inject_box_transit(time, np.ones_like(time), 3.0, 1.0, 3.0, 3000.0)

    diagnostics = robustness_diagnostics(
        time,
        flux,
        sectors,
        duration_grid_hours=[2.0, 3.0],
        period_min_days=2.0,
        period_max_days=4.0,
        n_periods=80,
        detrend_window_days=1.0,
        scramble_seeds=[5, 7],
    )

    assert set(diagnostics["variants"]) == {"normalized", "running-median"}
    assert len(diagnostics["controls"]["scrambles"]) == 2
    assert diagnostics["controls"]["max_snr"] >= diagnostics["controls"]["inverted"]["snr"]
    json.dumps(diagnostics)


def test_injection_recovery_records_the_declared_injection():
    time = np.linspace(0.0, 12.0, 500)

    results = injection_recovery_diagnostics(
        time, np.ones_like(time), [{"period_days": 3.0, "duration_hours": 3.0, "depth_ppm": 5000.0}],
        [3.0], 2.0, 4.0, 80, 0.05,
    )

    assert results[0]["injection"]["depth_ppm"] == 5000.0
    assert results[0]["recovered"] is True
    assert results[0]["period_match"] is True
    assert results[0]["epoch_match"] is True
    assert results[0]["snr_pass"] is True
    json.dumps(results)


@pytest.mark.parametrize(
    "best_epoch, best_snr, expected",
    [(1.0, 7.0, True), (1.0, 5.0, False), (1.1, 7.0, False)],
)
def test_injection_recovery_requires_period_epoch_and_snr(
    monkeypatch, best_epoch, best_snr, expected
):
    # Arrange
    time = np.linspace(0.0, 12.0, 500)

    def fake_duration_grid(*args, **kwargs):
        return (
            BLSSearchResult(
                best_period=3.0,
                best_epoch=best_epoch,
                best_depth_ppm=1000.0,
                best_duration_hours=2.0,
                snr=best_snr,
            ),
            [],
        )

    monkeypatch.setattr("exonym.discovery.search_duration_grid", fake_duration_grid)

    # Act
    result = injection_recovery_diagnostics(
        time,
        np.ones_like(time),
        [{"period_days": 3.0, "epoch_btjd": 1.0, "duration_hours": 2.0, "depth_ppm": 1000.0}],
        [2.0],
        2.0,
        4.0,
        80,
        0.05,
        minimum_snr=6.0,
    )[0]

    # Assert
    assert result["period_match"] is True
    assert result["recovered"] is expected
