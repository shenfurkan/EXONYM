"""Target-neutral truth-fixture regression pins for scientific CI.

These tests keep deterministic injected signals close to the numerical paths
they exercise.  Optional detrending engines are represented by dependency-free
adapters so CI checks the mask contract without requiring external packages.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from exonym.asteroseismology import estimate_oscillation_envelope
from exonym.catalog import calculate_radial_velocity_semi_amplitude
from exonym.detrending import (
    _celerite_trend,
    _running_median_trend,
    _wotan_trend,
    transit_mask_from_ephemeris,
)
from exonym.ephemeris_matching import _compare_record
from tests.fixtures.synthetic_observations import _synthetic_oscillation_table


def _synthetic_masked_transit_light_curve():
    """Return a deterministic injected transit and its continuum truth."""
    time = np.linspace(0.0, 8.0, 401)
    continuum = 1.0 + 0.015 * np.sin(2.0 * np.pi * time / 8.0)
    injected_depth = 0.01
    ephemeris = {
        "period_days": 20.0,
        "epoch_btjd": 4.0,
        "duration_days": 0.12,
        "time_system": "BTJD_TDB",
        "source": "candidate-config",
        "field_sources": {
            "period_days": "candidate-config",
            "epoch_btjd": "candidate-config",
            "duration_days": "candidate-config",
        },
    }
    transit_mask = transit_mask_from_ephemeris(time, ephemeris)
    flux = continuum * np.where(transit_mask, 1.0 - injected_depth, 1.0)
    return time, continuum, flux, transit_mask, injected_depth


def _assert_depth_recovery(detrended_flux, transit_mask, injected_depth):
    measured_depth = 1.0 - float(np.median(detrended_flux[transit_mask]))
    fractional_error = abs(measured_depth - injected_depth) / injected_depth
    assert fractional_error <= 0.05, (
        "masked detrending changed an injected depth of {0:.6f} to {1:.6f} "
        "({2:.1%} error)".format(injected_depth, measured_depth, fractional_error)
    )


@pytest.mark.parametrize("backend", ("running-median", "wotan", "celerite"))
def test_masked_detrending_backends_preserve_injected_depth(monkeypatch, backend):
    """Every supported backend preserves a target-neutral 1-percent transit."""
    time, continuum, flux, transit_mask, injected_depth = _synthetic_masked_transit_light_curve()

    if backend == "running-median":
        trend = _running_median_trend(time, flux, 0.5, transit_mask=transit_mask)
    elif backend == "wotan":
        observed = {}

        def flatten(observed_time, observed_flux, **kwargs):
            observed["time"] = np.asarray(observed_time).copy()
            observed["mask"] = np.asarray(kwargs["mask"], dtype=bool).copy()
            return None, continuum

        monkeypatch.setitem(sys.modules, "wotan", SimpleNamespace(flatten=flatten))
        trend = _wotan_trend(time, flux, 0.5, transit_mask=transit_mask)
        assert np.array_equal(observed["time"], time)
        assert np.array_equal(observed["mask"], transit_mask)
    else:
        observed = {}
        baseline = float(np.median(flux[~transit_mask]))

        class FakeTerm:
            def __init__(self, log_sigma, log_rho):
                self.log_sigma = log_sigma
                self.log_rho = log_rho

        class FakeGP:
            def __init__(self, kernel, mean):
                self.kernel = kernel
                self.mean = mean

            def compute(self, conditioned_time, yerr):
                observed["conditioned_time"] = np.asarray(conditioned_time).copy()
                observed["conditioned_error"] = np.asarray(yerr).copy()

            def predict(self, residuals, prediction_time, return_cov=False):
                observed["residuals"] = np.asarray(residuals).copy()
                observed["prediction_time"] = np.asarray(prediction_time).copy()
                return continuum - baseline

        fake_celerite = SimpleNamespace(
            terms=SimpleNamespace(Matern32Term=FakeTerm),
            GP=FakeGP,
        )
        monkeypatch.setitem(sys.modules, "celerite", fake_celerite)
        errors = np.full(time.size, 1.0e-4)
        trend = _celerite_trend(time, flux, errors, 0.5, transit_mask=transit_mask)
        assert np.array_equal(observed["conditioned_time"], time[~transit_mask])
        assert np.array_equal(observed["conditioned_error"], errors[~transit_mask])
        assert np.allclose(observed["residuals"], (flux - baseline)[~transit_mask])
        assert np.array_equal(observed["prediction_time"], time)

    _assert_depth_recovery(flux / trend, transit_mask, injected_depth)


def test_injected_oscillation_comb_recovers_numax_and_dnu():
    """The source-owned p-mode comb recovers its envelope and spacing family."""
    table = _synthetic_oscillation_table()
    result = estimate_oscillation_envelope(table["time"], table["flux"], 100.0, 1600.0)

    assert result["numax_candidate_uhz"] == pytest.approx(250.0, abs=15.0)
    assert any(
        result["dnu_candidate_uhz"] == pytest.approx(expected, abs=6.0)
        for expected in (40.0, 80.0, 120.0)
    )
    assert result["dnu_correlation"] > 0.5


@pytest.mark.parametrize(
    ("period_days", "expected_k_m_per_s"),
    ((3.0, 0.4435875603552931), (365.25, 0.0895)),
)
def test_rv_semi_amplitude_matches_two_period_truth_pins(period_days, expected_k_m_per_s):
    """Earth-mass circular-orbit K values retain the day-to-year conversion."""
    amplitude = calculate_radial_velocity_semi_amplitude(
        m_planet_earth=1.0,
        m_star_solar=1.0,
        period_days=period_days,
        inclination_deg=90.0,
    )

    assert amplitude == pytest.approx(expected_k_m_per_s, rel=1.0e-6)


@pytest.mark.parametrize(
    ("known_period_days", "expected_harmonic"),
    ((5.0, 0.5), (10.0, 1.0), (20.0, 2.0)),
)
def test_ephemeris_truth_ratios_fold_to_reviewable_matches(known_period_days, expected_harmonic):
    """P/2, P, and 2P catalog records retain an unambiguous review route."""
    candidate = {
        "period_days": 10.0,
        "period_uncertainty_days": 0.01,
        "epoch_btjd": 100.0,
        "duration_hours": 2.0,
    }
    source = {
        "pl_name": "synthetic-known-signal",
        "pl_orbper": known_period_days,
        "pl_orbpererr1": 0.01,
        "pl_orbpererr2": -0.01,
        "pl_tranmid": 2457000.0 + 100.0 + known_period_days,
        "pl_trandur": 2.0,
        "pl_tranmid_systemref": "BJD_TDB",
    }
    snapshot = {
        "provider": "nasa-exoplanet-archive",
        "retrieval_id": "synthetic-truth-fixture",
        "snapshot": {"path": "synthetic.csv", "sha256": "0" * 64},
    }

    comparison = _compare_record(candidate, source, 0, snapshot)

    assert comparison is not None
    assert comparison["period_ratio_known_over_candidate"] == pytest.approx(expected_harmonic)
    assert comparison["nearest_harmonic_factor"] == expected_harmonic
    assert comparison["period_harmonic_match"] is True
    assert comparison["epoch_phase_delta_days"] == pytest.approx(0.0)
    assert comparison["epoch_match"] is True
    assert comparison["harmonic_parity_ambiguous"] is False
    assert comparison["review_required"] is True
