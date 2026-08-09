"""Tests for candidate-local fixed-ephemeris photometric screening."""

import json

import numpy as np
import pytest

from exonym.lightcurve import phase_hours
from exonym.screening import fixed_ephemeris_screen, run_fixed_ephemeris_screen
from exonym.workspace import create_candidate


def _synthetic_transit_series(odd_depth_ppm=1000.0, even_depth_ppm=1000.0):
    rng = np.random.default_rng(47)
    time = np.arange(0.0, 30.0, 2.0 / 1440.0)
    period = 2.6
    epoch = 0.75
    duration_hours = 2.4
    flux = 1.0 + rng.normal(0.0, 120e-6, time.size)
    phase = phase_hours(time, period, epoch)
    cycles = np.floor((time - epoch) / period + 0.5).astype(int)
    in_transit = np.abs(phase) < 0.5 * duration_hours
    flux[in_transit & ((cycles % 2) == 0)] -= even_depth_ppm * 1e-6
    flux[in_transit & ((cycles % 2) != 0)] -= odd_depth_ppm * 1e-6
    return time, flux, period, epoch, duration_hours


def test_fixed_ephemeris_screen_recovers_primary_and_consistent_odd_even_depths():
    time, flux, period, epoch, duration = _synthetic_transit_series()

    result = fixed_ephemeris_screen(time, flux, period, epoch, duration)

    assert result["primary"]["status"] == "measured"
    assert result["primary"]["depth_ppm"] == pytest.approx(1000.0, abs=120.0)
    assert result["primary"]["depth_significance_sigma"] > 5.0
    assert result["odd_even"]["status"] == "measured"
    assert result["odd_even"]["z"] < 3.0
    assert result["odd_even"]["consistent_at_threshold"] is True
    assert abs(result["half_phase_control"]["depth_significance_sigma"]) < 3.0


def test_fixed_ephemeris_screen_flags_odd_even_inconsistency_without_validation_claim():
    time, flux, period, epoch, duration = _synthetic_transit_series(
        odd_depth_ppm=2600.0, even_depth_ppm=600.0
    )

    result = fixed_ephemeris_screen(time, flux, period, epoch, duration)

    assert result["odd_even"]["status"] == "measured"
    assert result["odd_even"]["z"] > 3.0
    assert result["odd_even"]["consistent_at_threshold"] is False
    harmonic = result["double_period_hypothesis"]
    assert harmonic["period_days"] == pytest.approx(2.0 * period)
    assert harmonic["primary"]["depth_ppm"] == pytest.approx(600.0, abs=120.0)
    assert harmonic["alternating_event"]["depth_ppm"] == pytest.approx(
        2600.0, abs=160.0
    )
    assert harmonic["alternating_event"]["depth_significance_sigma"] > 5.0
    assert "validate a planet" in harmonic["interpretation"]


def test_run_fixed_ephemeris_screen_requires_a_signal_prior_and_writes_scoped_output(
    tmp_path, monkeypatch
):
    workspace = create_candidate(tmp_path, "candidate-screen-test")
    signals = workspace.path / "config" / "signals"
    signals.mkdir(parents=True)
    (signals / "transit_config.01.json").write_text(
        json.dumps(
            {
                "transit": {
                    "period_days": 2.6,
                    "epoch_btjd": 0.75,
                    "duration_hours": 2.4,
                }
            }
        ),
        encoding="utf-8",
    )
    time, flux, _, _, _ = _synthetic_transit_series()
    monkeypatch.setattr(
        "exonym.screening.load_light_curve_table",
        lambda _workspace, max_points: {
            "time": time,
            "flux": flux,
            "sector": np.full(time.size, 31, dtype=int),
        },
    )

    output = run_fixed_ephemeris_screen(workspace, signal=".01")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert output.name == "fixed_ephemeris_screen.01.json"
    assert payload["source"] == "candidate-data"
    assert payload["signal"] == ".01"
    assert payload["screen"]["primary"]["status"] == "measured"
    with pytest.raises(ValueError, match="no readable signal prior"):
        run_fixed_ephemeris_screen(workspace, signal=".02")
