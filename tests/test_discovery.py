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


def test_sector_detrending_splits_large_intra_sector_time_gaps(monkeypatch):
    """A sample-index filter must never borrow trend values across a visit gap."""
    calls = []

    def recording_filter(values, size, mode):
        calls.append(np.asarray(values).copy())
        return np.asarray(values)

    monkeypatch.setattr("exonym.discovery.median_filter", recording_filter)
    time = np.array([0.0, 0.1, 0.2, 10.0, 10.1, 10.2])
    flux = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    result = detrend_by_sector(time, flux, np.ones(time.size, dtype=int), window_days=1.0)

    assert [call.size for call in calls] == [3, 3]
    assert np.allclose(result, np.ones_like(result))


def test_controls_are_deterministic_and_invert_flux():
    flux = np.array([0.99, 1.0, 1.01])

    assert np.allclose(inverted_flux(flux), np.array([1.01, 1.0, 0.99]))
    assert np.array_equal(scrambled_flux(flux, 5), scrambled_flux(flux, 5))

    sector_flux = np.array([0.98, 0.99, 1.0, 1.01, 1.02, 1.03])
    sectors = np.array([1, 1, 1, 2, 2, 2])
    sector_scramble = scrambled_flux(sector_flux, 5, sectors=sectors)
    assert np.array_equal(sector_scramble, scrambled_flux(sector_flux, 5, sectors=sectors))
    for sector in np.unique(sectors):
        assert np.array_equal(
            np.sort(sector_scramble[sectors == sector]), np.sort(sector_flux[sectors == sector])
        )


def test_box_injection_and_period_recovery():
    time = np.linspace(0.0, 20.0, 1000)
    injected = inject_box_transit(time, np.ones_like(time), 4.0, 1.0, 3.0, 1000.0)

    assert injected.min() == 0.999
    assert recovered_period(4.0, 4.01, tolerance=0.01)
    assert not recovered_period(4.0, 2.0, tolerance=0.01)


def test_box_injection_integrates_the_partial_cadence_overlap():
    """A cadence crossing ingress receives a fractional, not full, depth."""
    time = np.array([0.075])

    injected = inject_box_transit(
        time,
        np.ones_like(time),
        period_days=2.0,
        epoch_btjd=0.0,
        duration_hours=2.4,
        depth_ppm=1000.0,
        exposure_days=0.1,
    )

    # The 0.1-day exposure overlaps a 0.1-day transit for only 0.025 day.
    assert injected[0] == pytest.approx(1.0 - 1000.0e-6 * 0.25)


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
    assert set(diagnostics["controls"]["by_variant"]) == {"normalized", "running-median"}
    assert diagnostics["controls"]["scramble_method"] == "independent-sector-circular-shift"
    assert all(
        len(branch["scrambles"]) == 2
        for branch in diagnostics["controls"]["by_variant"].values()
    )
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
    "best_epoch, best_snr, event_count, expected",
    [(1.0, 7.0, 3, True), (1.0, 5.0, 3, False), (1.1, 7.0, 3, False), (1.0, 7.0, 1, False)],
)
def test_injection_recovery_requires_period_epoch_score_and_multiple_events(
    monkeypatch, best_epoch, best_snr, event_count, expected
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
                n_distinct_transit_events=event_count,
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
    assert result["recovered"] == expected


def test_injection_recovery_no_detection_safe(monkeypatch):
    """Injection recovery must complete without TypeError when SNR is None."""
    import numpy as np

    from exonym.discovery import injection_recovery_diagnostics
    from exonym.search import BLSSearchResult

    time = np.linspace(0.0, 10.0, 200)
    flux = np.ones_like(time)

    # Return a result with snr=None to simulate a non-detection
    def fake_duration_grid(*args, **kwargs):
        return (
            BLSSearchResult(
                best_period=3.0,
                best_epoch=1.0,
                best_depth_ppm=0.0,
                best_duration_hours=2.0,
                snr=None,
                n_distinct_transit_events=0,
            ),
            [],
        )

    monkeypatch.setattr("exonym.discovery.search_duration_grid", fake_duration_grid)

    injections = [
        {"epoch_btjd": 1.0, "period_days": 3.0, "duration_hours": 2.0, "depth_ppm": 1e-6}
    ]

    # Must not raise TypeError from `best.snr is None` or np.isfinite(None)
    results = injection_recovery_diagnostics(
        time,
        flux,
        injections,
        duration_grid_hours=[2.0],
        period_min_days=1.0,
        period_max_days=10.0,
        n_periods=10,
        tolerance=0.1,
    )
    assert len(results) == 1
    assert results[0]["recovered"] is False


def test_injection_recovery_requires_both_preprocessing_branches(monkeypatch):
    # Arrange
    time = np.linspace(0.0, 12.0, 500)
    sectors = np.ones(time.size, dtype=int)
    results = iter(
        (
            BLSSearchResult(3.0, 1.0, 1000.0, 2.0, 7.0, 3),
            BLSSearchResult(3.0, 1.0, 1000.0, 2.0, 5.0, 3),
        )
    )

    def fake_duration_grid(*args, **kwargs):
        return next(results), []

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
        sectors=sectors,
        detrend_window_days=1.0,
    )[0]

    # Assert
    assert set(result["branches"]) == {"normalized", "running-median"}
    assert result["branches"]["normalized"]["recovered"] is True
    assert result["branches"]["running-median"]["recovered"] is False
    assert result["recovered"] is False
