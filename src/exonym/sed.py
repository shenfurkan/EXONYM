"""Target-neutral spectral energy distribution (SED) engine.

Fits multi-band broadband photometry (Gaia DR3 G/BP/RP, 2MASS J/H/Ks, AllWISE W1-W4)
against synthetic stellar atmosphere models (BT-Settl / Kurucz) or a reddened
Planck blackbody model to derive fundamental host star parameters:
- Effective temperature (Teff)
- Surface gravity (log g)
- Metallicity ([Fe/H])
- Angular diameter / scaled radius (R_star / d)
- Visual interstellar extinction (A_V)

Physical Model:
    Observed flux density at Earth:
        F_nu = pi * B_nu(Teff) * (R_star / d)^2 * 10^(-0.4 * A_lambda)
    where A_lambda is computed via standard interstellar extinction laws
    (Cardelli, Clayton & Mathis 1989, Fitzpatrick 1999) with R_V = 3.1.

Contains zero target-specific identifiers or constants; all catalog magnitudes and
parallaxes are loaded dynamically from the candidate workspace.

Scientific Boundary:
    The grid path requires a candidate-supplied atmosphere table; the fallback
    is a pivot-wavelength reddened-blackbody approximation.  Neither path is a
    response-integrated atmosphere posterior or an automatic validation
    constraint.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.constants import c, h, k

from .constants import (
    NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K as TEFF_SUN_K,
    NOMINAL_SOLAR_LOGG_CGS as LOGG_SUN_CGS,
    NOMINAL_SOLAR_RADIUS_M as RSUN_M,
    PARSEC_M as PC_M,
)
from .inputs import load_photometry, load_stellar_parameters
from .workspace import CandidateWorkspace

MAG_SYSTEMATIC_FLOOR = 0.05             # Minimum systematic magnitude uncertainty floor

# 2MASS (Cohen et al. 2003) & AllWISE (Wright et al. 2010) bandpass pivot wavelengths and Vega zero-point fluxes (Jy)
BAND_ZERO_POINTS: Dict[str, Tuple[float, float]] = {
    "J": (1.235, 1594.0),               # 2MASS J-band (pivot: 1.235 um, F_0: 1594.0 Jy)
    "H": (1.662, 1024.0),               # 2MASS H-band (pivot: 1.662 um, F_0: 1024.0 Jy)
    "Ks": (2.159, 666.7),               # 2MASS Ks-band (pivot: 2.159 um, F_0: 666.7 Jy)
    "W1": (3.3526, 309.540),            # WISE W1-band (pivot: 3.35 um, F_0: 309.54 Jy)
    "W2": (4.6028, 171.787),            # WISE W2-band (pivot: 4.60 um, F_0: 171.79 Jy)
    "W3": (11.5608, 31.674),            # WISE W3-band (pivot: 11.56 um, F_0: 31.67 Jy)
    "W4": (22.0883, 8.363),             # WISE W4-band (pivot: 22.09 um, F_0: 8.36 Jy)
}

# Standard Milky Way interstellar extinction ratios A_lambda / A_V (Fitzpatrick 1999, R_V = 3.1)
EXTINCTION_RATIOS: Dict[str, float] = {
    "J": 0.282,
    "H": 0.190,
    "Ks": 0.114,
    "W1": 0.067,
    "W2": 0.054,
    "W3": 0.024,
    "W4": 0.015,
}


def percentile_summary(samples: np.ndarray) -> Dict[str, float]:
    """Summarize sampled values with central percentile and asymmetric errors.

    Args:
        samples (np.ndarray): Numeric posterior or Monte Carlo samples in one
            declared physical unit.

    Returns:
        Dict[str, float]: Lower percentile, median, upper percentile, and
        positive and negative offsets in the same unit as ``samples``.

    Note:
        This is a descriptive quantile summary; it does not assess sampler
        convergence or turn a model approximation into a calibrated posterior.
    """
    quantiles = np.quantile(np.asarray(samples), [0.16, 0.50, 0.84])
    return {
        "p16": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p84": float(quantiles[2]),
        "plus": float(quantiles[2] - quantiles[1]),
        "minus": float(quantiles[1] - quantiles[0]),
    }


def blackbody_model_magnitudes(
    teff_k: float,
    log_radius_over_distance: float,
    av_mag: float,
    band_data: Sequence[Tuple[str, float, float]],
) -> np.ndarray:
    """Evaluate a reddened blackbody approximation at band pivot wavelengths.

    Mathematical Formulation:
        The model evaluates ``F_nu = pi B_nu(Teff) (R / d)**2`` at each pivot
        wavelength, converts to a Vega magnitude, then adds
        ``A_lambda = A_V (A_lambda / A_V)``.  This matches the monochromatic
        approximation described in the project's stellar-physics note.

    Args:
        teff_k (float): Effective temperature in kelvin.
        log_radius_over_distance (float): Natural logarithm of the
            dimensionless radius-to-distance scale.
        av_mag (float): Visual extinction in magnitudes.
        band_data (Sequence[Tuple[str, float, float]]): Band name, pivot
            wavelength in microns, and Vega zero-point flux in janskys.

    Returns:
        np.ndarray: Model Vega magnitudes in the same order as ``band_data``.

    Note:
        Pivot-wavelength fluxes approximate passband-integrated photometry and
        should not be interpreted as a response-integrated atmosphere model.
    """
    radius_distance = math.exp(log_radius_over_distance)
    model = []
    for name, wavelength_micron, zero_jy in band_data:
        wavelength = wavelength_micron * 1e-6
        frequency = c / wavelength
        intensity = (
            2.0
            * h
            * frequency**3
            / c**2
            / np.expm1(h * frequency / (k * float(teff_k)))
        )
        flux_jy = np.pi * intensity * radius_distance**2 / 1e-26
        magnitude = -2.5 * np.log10(flux_jy / zero_jy) + av_mag * EXTINCTION_RATIOS[name]
        model.append(magnitude)
    return np.asarray(model)


def load_atmosphere_grid_model(
    workspace: CandidateWorkspace,
    band_names: Sequence[str],
) -> Optional[Callable[[float, float, float], np.ndarray]]:
    """Return a candidate-local generic atmosphere-grid interpolator, or ``None``.

    The grid CSV must contain ``teff_k``, ``logg_cgs``, ``feh`` columns plus
    magnitude columns named ``mag_<band>`` for the observed bands. Returns a
    callable mapping (teff_k, logg_cgs, feh) -> magnitude array.

    Args:
        workspace (CandidateWorkspace): Workspace that may own the optional
            atmosphere-grid CSV.
        band_names (Sequence[str]): Required observed band labels.  Each must
            have a corresponding grid magnitude column.

    Returns:
        Optional[Callable[[float, float, float], np.ndarray]]: Interpolator
        from effective temperature in kelvin, surface gravity in cgs log units,
        and metallicity in dex to model magnitudes, or ``None`` if the optional
        grid is unavailable or unusable.

    Note:
        Queries outside the linear interpolation hull use nearest-grid fallback.
        This makes availability explicit but does not establish model adequacy.
    """
    path = workspace.path / "data" / "external" / "atmosphere_grid.csv"
    if not path.is_file():
        return None
    try:
        import pandas as pd
        from scipy.interpolate import griddata

        frame = pd.read_csv(path)
    except (ImportError, OSError, ValueError, pd.errors.ParserError) as exc:
        logging.warning("atmosphere grid load failed for %s: %s", path.name, exc)
        return None
    except Exception as exc:
        logging.warning("atmosphere grid load failed for %s: %s", path.name, exc)
        return None
    required = ("teff_k", "logg_cgs", "feh")
    if not all(column in frame.columns for column in required):
        return None
    mag_columns = [f"mag_{name}" for name in band_names]
    if not all(column in frame.columns for column in mag_columns):
        return None
    points = np.column_stack(
        (
            frame["teff_k"].to_numpy(float),
            frame["logg_cgs"].to_numpy(float),
            frame["feh"].to_numpy(float),
        )
    )
    values = np.column_stack([frame[column].to_numpy(float) for column in mag_columns])

    def model(teff_k: float, logg_cgs: float, feh: float) -> np.ndarray:
        query = np.array([[teff_k, logg_cgs, feh]], dtype=float)
        interpolated = griddata(points, values, query, method="linear")
        if np.any(~np.isfinite(interpolated)):
            interpolated = griddata(points, values, query, method="nearest")
        return np.asarray(interpolated[0], dtype=float)

    return model


def _run_emcee(
    log_probability: Callable[[np.ndarray], float],
    start: np.ndarray,
    n_walkers: int,
    burn_in: int,
    production: int,
    seed: int,
) -> Tuple[np.ndarray, Any]:
    import emcee

    ndim = int(start.size)
    rng = np.random.default_rng(seed=seed)
    walkers = start + rng.normal(size=(n_walkers, ndim)) * 1e-3
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability)
    sampler.random_state = np.random.RandomState(seed).get_state()
    state = sampler.run_mcmc(walkers, burn_in, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, production, progress=False)
    samples = sampler.get_chain(flat=True)
    return samples, sampler


def _fit_blackbody(
    observations: Sequence[Tuple[str, float, float]],
    stellar: Dict[str, Any],
    n_walkers: int = 32,
    burn_in: int = 200,
    production: int = 500,
    seed: int = 7,
) -> Dict[str, Any]:
    band_data = [
        (name, *BAND_ZERO_POINTS[name]) for name, _, _ in observations
    ]
    magnitudes = np.array([row[1] for row in observations], dtype=float)
    # NUMERICAL_GUARD: The declared systematic floor prevents arbitrarily small
    # catalog errors from dominating this approximate pivot-wavelength model.
    errors = np.sqrt(
        np.array([row[2] for row in observations], dtype=float) ** 2
        + MAG_SYSTEMATIC_FLOOR**2
    )
    parallax = float(stellar["parallax_mas"])
    parallax_error = float(stellar.get("parallax_mas_err", float("nan")))
    if not math.isfinite(parallax) or parallax <= 0 or not math.isfinite(parallax_error) or parallax_error <= 0:
        raise RuntimeError("blackbody SED fitting requires positive candidate-owned parallax_mas and parallax_mas_err")
    distance_pc = 1000.0 / parallax
    teff_prior = float(stellar["teff_k"])
    initial_scale = 1.0 * RSUN_M / (distance_pc * PC_M)

    def log_probability(theta: np.ndarray) -> float:
        teff, log_scale, av = float(theta[0]), float(theta[1]), float(theta[2])
        if not 3500.0 < teff < 8000.0 or not 0.0 < av < 0.5:
            return -np.inf
        if not np.log(initial_scale / 3.0) < log_scale < np.log(initial_scale * 3.0):
            return -np.inf
        model = blackbody_model_magnitudes(teff, log_scale, av, band_data)
        likelihood = -0.5 * np.sum(
            ((magnitudes - model) / errors) ** 2 + np.log(2.0 * np.pi * errors**2)
        )
        temperature_prior = -0.5 * ((teff - teff_prior) / 200.0) ** 2
        extinction_prior = -0.5 * (av / 0.05) ** 2
        return float(likelihood + temperature_prior + extinction_prior)

    start = np.array([teff_prior, np.log(initial_scale), 0.02])
    samples, sampler = _run_emcee(
        log_probability, start, n_walkers, burn_in, production, seed
    )

    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0 or not np.all(np.isfinite(samples)):
        raise RuntimeError("blackbody SED sampler returned no finite posterior samples")
    draw_rng = np.random.default_rng(seed=seed + 1)
    proposed_parallax_draws = np.asarray(
        draw_rng.normal(parallax, parallax_error, samples.shape[0]), dtype=float
    )
    # NUMERICAL_GUARD: Never invert non-positive or non-finite parallax draws.
    valid_parallax_draws = np.isfinite(proposed_parallax_draws) & (
        proposed_parallax_draws > 0.0
    )
    parallax_draws = proposed_parallax_draws[valid_parallax_draws]
    derived_samples = samples[valid_parallax_draws]
    rejected_parallax_draws = int(proposed_parallax_draws.size - parallax_draws.size)
    if parallax_draws.size == 0:
        raise RuntimeError(
            "blackbody SED fitting rejected every parallax draw as non-positive or non-finite"
        )
    distance_draws = 1000.0 / parallax_draws
    radius_draws = np.exp(derived_samples[:, 1]) * distance_draws * PC_M / RSUN_M
    luminosity_draws = radius_draws**2 * (derived_samples[:, 0] / TEFF_SUN_K) ** 4
    mass_prior = float(stellar["mass_solar"])
    if not math.isfinite(mass_prior) or mass_prior <= 0.0:
        raise RuntimeError("blackbody SED fitting requires positive candidate-owned mass_solar")
    logg_draws = LOGG_SUN_CGS + np.log10(mass_prior) - 2.0 * np.log10(radius_draws)
    if not (
        np.all(np.isfinite(distance_draws))
        and np.all(distance_draws > 0.0)
        and np.all(np.isfinite(radius_draws))
        and np.all(radius_draws > 0.0)
        and np.all(np.isfinite(luminosity_draws))
        and np.all(luminosity_draws > 0.0)
        and np.all(np.isfinite(logg_draws))
    ):
        raise RuntimeError("blackbody SED fitting produced invalid derived posterior samples")

    median = np.median(samples, axis=0)
    model_at_median = blackbody_model_magnitudes(
        float(median[0]), float(median[1]), float(median[2]), band_data
    )
    residuals = magnitudes - model_at_median
    return {
        "model": "reddened blackbody at catalog pivot wavelengths",
        "posterior": {
            "teff_k": percentile_summary(samples[:, 0]),
            "av_mag": percentile_summary(samples[:, 2]),
            "distance_pc": percentile_summary(distance_draws),
            "radius_solar": percentile_summary(radius_draws),
            "luminosity_solar": percentile_summary(luminosity_draws),
            "logg_cgs": percentile_summary(logg_draws),
        },
        "photometry": [
            {
                "band": name,
                "observed_mag": float(observed),
                "total_error_mag": float(total_error),
                "model_mag_at_posterior_median": float(model),
                "residual_mag": float(residual),
            }
            for (name, observed, _), total_error, model, residual in zip(
                observations, errors, model_at_median, residuals
            )
        ],
        "fit_quality": {
            "chi_square_at_posterior_median": float(np.sum((residuals / errors) ** 2)),
            "degrees_of_freedom": len(observations) - 3,
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "retained_samples": int(len(samples)),
            "parallax_draws": {
                "proposed_count": int(proposed_parallax_draws.size),
                "accepted_positive_finite_count": int(parallax_draws.size),
                "rejected_nonpositive_or_nonfinite_count": rejected_parallax_draws,
                "rejection_rate": float(
                    rejected_parallax_draws / proposed_parallax_draws.size
                ),
                "policy": (
                    "non-positive or non-finite draws are rejected before "
                    "distance-derived summaries"
                ),
            },
        },
        "samples": samples,
    }


def _fit_grid(
    observations: Sequence[Tuple[str, float, float]],
    grid_model: Callable[[float, float, float], np.ndarray],
    stellar: Dict[str, Any],
    n_walkers: int = 32,
    burn_in: int = 200,
    production: int = 500,
    seed: int = 7,
) -> Dict[str, Any]:
    magnitudes = np.array([row[1] for row in observations], dtype=float)
    # NUMERICAL_GUARD: Apply the same magnitude-error floor to both model paths
    # so a grid lookup is not given unbounded weight over catalog systematics.
    errors = np.sqrt(
        np.array([row[2] for row in observations], dtype=float) ** 2
        + MAG_SYSTEMATIC_FLOOR**2
    )
    teff_prior = float(stellar["teff_k"])
    logg_prior = float(stellar["logg_cgs"])
    feh_prior = float(stellar["feh"])

    def log_probability(theta: np.ndarray) -> float:
        teff, logg, feh, offset = (
            float(theta[0]),
            float(theta[1]),
            float(theta[2]),
            float(theta[3]),
        )
        if not 3500.0 < teff < 8000.0 or not 2.0 < logg < 5.5:
            return -np.inf
        if not -2.0 < feh < 1.0 or not -5.0 < offset < 5.0:
            return -np.inf
        model = np.asarray(grid_model(teff, logg, feh), dtype=float) + offset
        likelihood = -0.5 * np.sum(
            ((magnitudes - model) / errors) ** 2 + np.log(2.0 * np.pi * errors**2)
        )
        prior = (
            -0.5 * ((teff - teff_prior) / 200.0) ** 2
            - 0.5 * ((logg - logg_prior) / 0.25) ** 2
            - 0.5 * ((feh - feh_prior) / 0.2) ** 2
        )
        return float(likelihood + prior)

    start = np.array([teff_prior, logg_prior, feh_prior, 0.0])
    samples, sampler = _run_emcee(
        log_probability, start, n_walkers, burn_in, production, seed
    )
    median = np.median(samples, axis=0)
    model_at_median = (
        np.asarray(grid_model(float(median[0]), float(median[1]), float(median[2])))
        + float(median[3])
    )
    residuals = magnitudes - model_at_median
    return {
        "model": "generic atmosphere-grid interpolation with free magnitude offset",
        "posterior": {
            "teff_k": percentile_summary(samples[:, 0]),
            "logg_cgs": percentile_summary(samples[:, 1]),
            "feh": percentile_summary(samples[:, 2]),
            "magnitude_offset": percentile_summary(samples[:, 3]),
        },
        "photometry": [
            {
                "band": name,
                "observed_mag": float(observed),
                "total_error_mag": float(total_error),
                "model_mag_at_posterior_median": float(model),
                "residual_mag": float(residual),
            }
            for (name, observed, _), total_error, model, residual in zip(
                observations, errors, model_at_median, residuals
            )
        ],
        "fit_quality": {
            "chi_square_at_posterior_median": float(np.sum((residuals / errors) ** 2)),
            "degrees_of_freedom": len(observations) - 4,
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "retained_samples": int(len(samples)),
        },
        "samples": samples,
    }


def _collect_observations(
    photometry: Optional[Dict[str, Any]]
) -> Tuple[Optional[List[Tuple[str, float, float]]], str]:
    """Extract (band, mag, error) rows from the generic photometry JSON."""
    if photometry is None:
        return None, "no-photometry-file"
    rows: List[Tuple[str, float, float]] = []
    for catalog_name in ("2MASS", "AllWISE"):
        catalog = photometry.get(catalog_name)
        if not isinstance(catalog, dict):
            continue
        for band, value in catalog.items():
            if not isinstance(value, dict):
                continue
            mag = value.get("mag")
            error = value.get("error")
            if mag is None or error is None:
                continue
            try:
                rows.append((str(band), float(mag), float(error)))
            except (TypeError, ValueError):
                continue
    if not rows:
        return None, "no-readable-photometry"
    return rows, "candidate-data"


def _synthetic_photometry(stellar: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    """Deterministic demonstration photometry from a reddened blackbody."""
    rng = np.random.default_rng(seed=7)
    teff = float(stellar["teff_k"])
    radius = float(stellar["radius_solar"])
    distance_pc = 1000.0 / float(stellar["parallax_mas"])
    log_scale = math.log(radius * RSUN_M / (distance_pc * PC_M))
    av = 0.02
    band_names = list(BAND_ZERO_POINTS)
    band_data = [(name, *BAND_ZERO_POINTS[name]) for name in band_names]
    model = blackbody_model_magnitudes(teff, log_scale, av, band_data)
    rows = []
    for name, magnitude in zip(band_names, model):
        observed = magnitude + rng.normal(0.0, 0.02)
        rows.append((name, float(observed), 0.02))
    return rows


def run_sed_fit(workspace: CandidateWorkspace) -> Path:
    """Run the candidate-local exploratory broadband SED fit.

    The runner uses a candidate-supplied atmosphere grid when one has the
    required columns; otherwise it uses the documented reddened-blackbody
    pivot-wavelength approximation.  It records posterior quantiles, input
    photometry, residuals, sampler diagnostics, and model caveats.

    Args:
        workspace (CandidateWorkspace): Workspace that owns broadband
            photometry, stellar parameters, optional grid data, and outputs.

    Returns:
        Path: Candidate-local ``outputs/sed_fit_results.json``.  The result
        records the model path and its known approximation limits.

    Raises:
        RuntimeError: Candidate-owned photometry or required parallax data for
            the blackbody path is unavailable or invalid.
        OSError: Candidate-local output artifacts cannot be written.

    Note:
        The fit is an exploratory color diagnostic.  It does not provide a
        calibrated atmosphere posterior, a validation constraint, or a
        lifecycle decision.
    """
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    stellar = load_stellar_parameters(workspace)
    photometry = load_photometry(workspace)
    observations, source = _collect_observations(photometry)
    grid_used = False
    if observations is None:
        raise RuntimeError("SED fitting requires candidate-owned broadband photometry")
    grid_model = load_atmosphere_grid_model(
        workspace, [name for name, _, _ in observations]
    )
    if grid_model is not None:
        grid_used = True
    grid_load_failed = not grid_used and (workspace.path / "data" / "external" / "atmosphere_grid.csv").is_file()

    # SCIENTIFIC_BOUNDARY: The fallback is recorded as a blackbody approximation
    # rather than being presented as an atmosphere-grid inference.
    fit = (
        _fit_grid(observations, grid_model, stellar)  # type: ignore[arg-type]
        if grid_used
        else _fit_blackbody(observations, stellar)
    )
    samples = fit.pop("samples", None)

    payload = {
        "schema_version": "1.0",
        "work_package": "SED_FIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "scientific_status": "exploratory-color-fit",
        "validation_eligible": False,
        "grid_used": grid_used,
        "grid_load_failed": grid_load_failed,
        "grid_source": (
            "candidate-data/external/atmosphere_grid.csv" if grid_used else "blackbody-fallback"
        ),
        "method": fit["model"],
        "input_photometry": [
            {"band": name, "mag": mag, "error": error}
            for name, mag, error in observations
        ],
        "posterior": fit["posterior"],
        "photometry": fit["photometry"],
        "fit_quality": fit["fit_quality"],
        "caveats": [
            "Pivot-wavelength monochromatic models approximate passband-integrated photometry.",
            "Radius and luminosity are derived only for the blackbody path via the parallax prior.",
            "Grid magnitudes carry an unknown absolute normalization; a free offset absorbs it.",
            "This exploratory fit is not a response-integrated atmosphere posterior or a validation constraint.",
        ],
    }
    output_path = outputs_dir / "sed_fit_results.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if samples is not None:
        np.save(str(outputs_dir / "sed_fit_chain.npy"), samples)
    return output_path
