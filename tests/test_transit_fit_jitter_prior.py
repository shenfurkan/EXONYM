"""Regression coverage for the data-independent transit-fit jitter prior."""

import math

import numpy as np
import pytest

from exonym.transit_fit import (
    JITTER_HALF_CAUCHY_SCALE,
    _half_cauchy_log_jitter_log_density,
    _initial_fit_parameters,
    _jitter_prior_assumption,
    _log_prior,
    _make_dynesty_prior_transform,
)


def test_jitter_prior_is_data_independent_and_shared_by_both_samplers():
    """A change in quoted photometric errors must not move the prior centre."""
    assumption = _jitter_prior_assumption()

    assert assumption["distribution"] == "half-cauchy-on-jitter"
    assert assumption["scale_normalized_flux"] == pytest.approx(JITTER_HALF_CAUCHY_SCALE)
    assert assumption["data_dependent"] is False
    assert assumption["empirical_bayes"] is False

    theta = _initial_fit_parameters(1200.0, 1.0, 80e-6, eccentric=False)
    theta[4] = math.log(400e-6)
    expected = _half_cauchy_log_jitter_log_density(float(theta[4]))
    assert _log_prior(theta, 1.0, 0.1, eccentric=False) == pytest.approx(expected)

    transform = _make_dynesty_prior_transform(1.0, 0.1, False, None)
    transformed = transform(np.full(theta.size, 0.5))
    assert math.isfinite(float(transformed[4]))
    assert _log_prior(transformed, 1.0, 0.1, eccentric=False) > -np.inf


def test_half_cauchy_jitter_prior_retains_wide_synthetic_noise_support():
    """A synthetic no-information likelihood retains a broad jitter interval."""
    transform = _make_dynesty_prior_transform(1.0, 0.1, False, None)
    unit_cube = np.full(7, 0.5)
    jitter_draws = []
    for quantile in (0.16, 0.84):
        unit_cube[4] = quantile
        jitter_draws.append(math.exp(float(transform(unit_cube)[4])))

    lower, upper = jitter_draws
    assert lower > 0.0
    assert upper > lower
    # The 68-percent prior interval spans more than an order of magnitude, so
    # the prior does not create an artificially narrow synthetic posterior.
    assert upper / lower > 10.0
