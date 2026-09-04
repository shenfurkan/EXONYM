"""Public numerical-domain regression coverage for astrophysical helpers."""

import math

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("rho_solar", "period_days"),
    [
        (np.nan, 3.0),
        (np.inf, 3.0),
        (1.0, np.nan),
        (1.0, np.inf),
    ],
)
def test_stellar_density_scaling_rejects_nonfinite_inputs(rho_solar, period_days):
    from exonym.transit_fit import stellar_density_a_rs

    with pytest.raises(ValueError, match="finite and positive"):
        stellar_density_a_rs(rho_solar, period_days)


@pytest.mark.parametrize(
    ("numax_uhz", "dnu_uhz", "teff_k", "mass_prior_solar", "radius_prior_solar"),
    [
        (np.nan, None, 5772.0, 1.0, 1.0),
        (np.inf, None, 5772.0, 1.0, 1.0),
        (-1.0, None, 5772.0, 1.0, 1.0),
        (0.0, np.nan, 5772.0, 1.0, 1.0),
        (0.0, np.inf, 5772.0, 1.0, 1.0),
        (0.0, -1.0, 5772.0, 1.0, 1.0),
        (0.0, None, np.nan, 1.0, 1.0),
        (0.0, None, np.inf, 1.0, 1.0),
        (0.0, None, 5772.0, np.nan, 1.0),
        (0.0, None, 5772.0, 1.0, np.inf),
    ],
)
def test_seismic_scaling_rejects_nonfinite_or_unphysical_inputs(
    numax_uhz,
    dnu_uhz,
    teff_k,
    mass_prior_solar,
    radius_prior_solar,
):
    from exonym.asteroseismology import seismic_mass_radius

    with pytest.raises(ValueError):
        seismic_mass_radius(
            numax_uhz,
            dnu_uhz,
            teff_k,
            mass_prior_solar=mass_prior_solar,
            radius_prior_solar=radius_prior_solar,
        )


def test_seismic_scaling_preserves_explicit_missing_measurement_fallback():
    from exonym.asteroseismology import seismic_mass_radius

    result = seismic_mass_radius(
        0.0,
        None,
        5772.0,
        mass_prior_solar=1.2,
        radius_prior_solar=1.4,
    )

    assert result["method"] == "stellar-priors-only"
    assert result["mass_solar"] == pytest.approx(1.2)
    assert result["radius_solar"] == pytest.approx(1.4)


def test_mist_sed_runner_rejects_an_unbound_or_missing_manifest(tmp_path):
    from exonym.sed import run_sed_fit
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "mist-manifest-required")

    with pytest.raises(RuntimeError, match="MIST v1.2 bolometric-correction SED input manifest is missing"):
        run_sed_fit(workspace)
    assert not (workspace.path / "outputs" / "sed_fit_results.json").exists()


@pytest.mark.parametrize(
    ("period_inner_days", "period_outer_days", "j_resonance"),
    [
        (np.nan, 4.0, 2),
        (2.0, np.inf, 2),
        (2.0, 4.0, 2.0),
        (2.0, 4.0, True),
        (2.0, 4.0, 1),
    ],
)
def test_ttv_super_period_rejects_nonfinite_or_invalid_resonance_inputs(
    period_inner_days, period_outer_days, j_resonance
):
    from exonym.search import calculate_ttv_super_period

    with pytest.raises(ValueError):
        calculate_ttv_super_period(period_inner_days, period_outer_days, j_resonance)


def test_ttv_super_period_preserves_exact_resonance_infinity():
    from exonym.search import calculate_ttv_super_period

    assert math.isinf(calculate_ttv_super_period(2.0, 4.0, np.int64(2)))
