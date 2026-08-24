"""Tests for asteroseismic uncertainty reporting."""

import numpy as np
import pytest

from exonym.asteroseismology import (
    CPD_PER_UHZ,
    seismic_mass_radius,
    seismic_uncertainty_summary,
)


def test_seismic_uncertainty_summary_reports_deterministic_percentiles():
    summary = seismic_uncertainty_summary(
        {
            "numax_candidate_uhz": 1200.0,
            "dnu_candidate_uhz": 60.0,
            "rayleigh_uhz": 0.5,
        },
        {"teff_k": 5700.0, "teff_k_err": 75.0},
        draws=128,
    )

    assert summary["status"] == "resolution-and-temperature-monte-carlo"
    assert summary["draws"] == 128
    assert summary["mass_solar"]["p16"] < summary["mass_solar"]["median"] < summary["mass_solar"]["p84"]
    assert summary["radius_solar"]["minus"] > 0.0


def test_resolution_draws_are_bounded_by_half_rayleigh_intervals_and_corrected_consistently():
    envelope = {
        "numax_candidate_uhz": 1200.0,
        "dnu_candidate_uhz": 60.0,
        "rayleigh_uhz": 0.5,
    }
    stellar = {"teff_k": 5700.0, "teff_k_err": 75.0}

    identity = seismic_uncertainty_summary(envelope, stellar, draws=128)
    corrected = seismic_uncertainty_summary(
        envelope,
        stellar,
        draws=128,
        dnu_correction_factor=0.98,
    )

    sampling = corrected["frequency_resolution_sampling"]
    assert sampling["distribution"] == "uniform-within-one-Rayleigh-resolution-element"
    assert sampling["numax_interval_uhz"] == pytest.approx([1199.75, 1200.25])
    assert sampling["dnu_interval_uhz"] == pytest.approx([59.75, 60.25])
    assert 1199.75 <= corrected["numax_uhz"]["p16"] <= 1200.25
    assert 1199.75 <= corrected["numax_uhz"]["p84"] <= 1200.25
    assert 59.75 <= corrected["dnu_uhz"]["p16"] <= 60.25
    assert 59.75 <= corrected["dnu_uhz"]["p84"] <= 60.25
    assert corrected["dnu_corrected_uhz"]["median"] == pytest.approx(
        0.98 * corrected["dnu_uhz"]["median"]
    )
    assert corrected["radius_solar"]["median"] == pytest.approx(
        identity["radius_solar"]["median"] / 0.98**2
    )
    assert corrected["mass_solar"]["median"] == pytest.approx(
        identity["mass_solar"]["median"] / 0.98**4
    )


@pytest.mark.parametrize("factor", [0.0, -0.1, np.nan, np.inf])
def test_seismic_scaling_rejects_invalid_dnu_correction_factors(factor):
    with pytest.raises(ValueError, match="positive finite"):
        seismic_mass_radius(1200.0, 60.0, 5700.0, dnu_correction_factor=factor)


def test_cpd_per_uhz_has_the_correct_conversion_direction():
    frequency_uhz = 135.1
    frequency_cpd = frequency_uhz * CPD_PER_UHZ

    assert CPD_PER_UHZ == pytest.approx(0.0864)
    assert frequency_cpd / CPD_PER_UHZ == pytest.approx(frequency_uhz)


def test_seismic_uncertainty_summary_does_not_fabricate_missing_errors():
    summary = seismic_uncertainty_summary(
        {"numax_candidate_uhz": 1200.0, "dnu_candidate_uhz": 60.0, "rayleigh_uhz": 0.5},
        {"teff_k": 5700.0},
    )

    assert summary["status"] == "unavailable-missing-input-uncertainty"
