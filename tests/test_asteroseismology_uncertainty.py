"""Tests for asteroseismic uncertainty reporting."""

from exonym.asteroseismology import seismic_uncertainty_summary


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


def test_seismic_uncertainty_summary_does_not_fabricate_missing_errors():
    summary = seismic_uncertainty_summary(
        {"numax_candidate_uhz": 1200.0, "dnu_candidate_uhz": 60.0, "rayleigh_uhz": 0.5},
        {"teff_k": 5700.0},
    )

    assert summary["status"] == "unavailable-missing-input-uncertainty"
