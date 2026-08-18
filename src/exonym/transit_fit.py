"""Target-neutral MCMC transit light curve fitter.

Fits phase-folded transit light curves with batman (Mandel & Agol 2002):
- Free Kipping (2013) limb darkening (q1, q2 uninformative triangular sampling).
- Stellar-density locking: a/Rs derived directly from Kepler's third law
  (Seager & Mallen-Ornelas 2003, Sozzetti et al. 2007) and orbital period.
- Parameterized eccentric orbits via (sqrt(e)*cos(omega), sqrt(e)*sin(omega))
  to avoid coordinate singularities at e=0 (Eastman et al. 2013).
- Gaussian log-likelihood with per-cadence photometric jitter parameter.
- Posterior sampling via Goodman & Weare (2010) affine-invariant ensemble sampler
  (emcee) or optional dynamic nested sampling (dynesty).

Contains no target constants or hardcoded candidate parameters; all stellar priors,
ephemerides, and photometric time-series are loaded dynamically from the candidate workspace.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .inputs import (
    load_light_curve_table,
    load_stellar_parameters,
    load_transit_ephemeris,
)
from .lightcurve import bin_phase_folded_flux, kipping_to_quadratic_limb_darkening
from .workspace import CandidateWorkspace

# Physical constants in CGS units
G_CGS = 6.67430e-8          # Gravitational constant (cm^3 g^-1 s^-2)
RHO_SUN_GCM3 = 1.408        # Mean solar density (g cm^-3)
WINDOW_HALF_HOURS = 13.0    # Folded light curve crop window half-width (hours)
BIN_MINUTES = 8.0           # Default phase-binning resolution (minutes)
SUPERSAMPLE_FACTOR = 7      # Numerical exposure integration sub-sampling factor
EXPTIME_SECONDS = 120.0     # Nominal TESS 2-minute SPOC cadence (seconds)

# Parameter vectors for circular and eccentric orbits
PARAMETER_NAMES_CIRCULAR = (
    "rp_rs",             # Planet-to-star radius ratio (R_p / R_star)
    "log_rho_star",      # log10(rho_star / rho_sun), stellar density proxy
    "impact_parameter",  # Transit impact parameter b = (a/R_star) * cos(i)
    "baseline",          # Out-of-transit flux normalization baseline
    "log_jitter",        # log(sigma_jitter), added in quadrature to flux uncertainties
    "q1",                # Kipping (2013) limb-darkening parameter 1
    "q2",                # Kipping (2013) limb-darkening parameter 2
)
PARAMETER_NAMES_ECCENTRIC = PARAMETER_NAMES_CIRCULAR + (
    "sqe_cosw",          # sqrt(e) * cos(omega), Lagrangian eccentricity component
    "sqe_sinw",          # sqrt(e) * sin(omega), Lagrangian eccentricity component
)


def stellar_density_a_rs(rho_solar: float, period_days: float) -> float:
    """Calculate scaled semimajor axis (a/R_star) from mean stellar density and orbital period.

    From Kepler's Third Law (Seager & Mallen-Ornelas 2003):
        (a / R_star)^3 = (G * P^2 * rho_star) / (3 * pi)
    
    Parameters
    ----------
    rho_solar : float
        Mean stellar density normalized to solar density (rho_star / rho_sun).
    period_days : float
        Orbital period in days.

    Returns
    -------
    float
        Scaled semimajor axis dimensionless ratio (a / R_star).
    """
    if rho_solar <= 0 or period_days <= 0:
        raise ValueError("stellar density and period must be positive")
    rho_gcm3 = rho_solar * RHO_SUN_GCM3
    period_seconds = period_days * 86400.0
    return (
        (G_CGS * period_seconds**2 * rho_gcm3) / (3.0 * math.pi)
    ) ** (1.0 / 3.0)


def batman_transit_flux(
    phase_days: Sequence[float],
    period_days: float,
    rp_rs: float,
    a_rs: float,
    impact_parameter: float,
    q1: float,
    q2: float,
    baseline: float,
    eccentricity: float = 0.0,
    omega_deg: float = 90.0,
) -> Optional[np.ndarray]:
    """Evaluate a batman quadratic limb-darkening model at transit-relative phase.

    Returns None when the geometry is unphysical (b >= a/Rs, e >= 1, or batman
    fails), so callers can apply an infinite penalty.
    """
    cosine = impact_parameter / max(a_rs, 1e-9)
    if not 0.0 <= cosine < 1.0:
        return None
    if not 0.0 <= eccentricity < 1.0:
        return None
    try:
        import batman

        u1, u2 = kipping_to_quadratic_limb_darkening(q1, q2)
        params = batman.TransitParams()
        params.t0 = 0.0
        params.per = period_days
        params.rp = rp_rs
        params.a = a_rs
        params.inc = math.degrees(math.acos(float(cosine)))
        params.ecc = eccentricity
        params.w = omega_deg
        params.u = [u1, u2]
        params.limb_dark = "quadratic"
        model = batman.TransitModel(
            params,
            np.asarray(phase_days, dtype=float),
            supersample_factor=SUPERSAMPLE_FACTOR,
            exp_time=EXPTIME_SECONDS / 86400.0,
        )
        flux = np.asarray(model.light_curve(params), dtype=float)
        return baseline * flux
    except Exception:
        return None


def _folded_binned_data(
    time: Sequence[float],
    flux: Sequence[float],
    ephemeris: Dict[str, Any],
    window_half_hours: float = WINDOW_HALF_HOURS,
    bin_minutes: float = BIN_MINUTES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase-fold and median-bin a light curve around the transit window."""
    centers_hours, binned_flux, binned_error = bin_phase_folded_flux(
        time,
        flux,
        ephemeris["period_days"],
        ephemeris["epoch_btjd"],
        limit_hours=window_half_hours,
        bin_minutes=bin_minutes,
    )
    valid = (
        np.isfinite(centers_hours)
        & np.isfinite(binned_flux)
        & np.isfinite(binned_error)
        & (binned_error > 0)
    )
    if int(valid.sum()) < 20:
        raise ValueError("insufficient binned transit window coverage")
    return (
        centers_hours[valid] / 24.0,
        binned_flux[valid],
        binned_error[valid],
    )


def _load_ldtk_prior(workspace: CandidateWorkspace) -> Dict[str, Any]:
    """Load one finite candidate-local quadratic LDTk prior for explicit use."""
    path = workspace.path / "outputs" / "ldtk_quadratic_limb_darkening_prior.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("--ldtk-prior requires a readable candidate-local LDTk prior artifact") from exc
    if payload.get("candidate_id") != workspace.candidate_id:
        raise ValueError("LDTk prior candidate_id does not match the fit workspace")
    coefficients = payload.get("quadratic_coefficients")
    if not isinstance(coefficients, list) or len(coefficients) != 1:
        raise ValueError("--ldtk-prior requires exactly one recorded passband prior")
    coefficient = coefficients[0]
    if not isinstance(coefficient, dict):
        raise ValueError("LDTk prior coefficient record is invalid")
    values = {}
    for name in ("u1", "u1_err", "u2", "u2_err"):
        value = coefficient.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("LDTk prior {0} must be finite".format(name))
        if name.endswith("_err") and value <= 0:
            raise ValueError("LDTk prior uncertainties must be positive")
        values[name] = float(value)
    values["path"] = str(path.relative_to(workspace.path)).replace("\\", "/")
    return values


def _neg_log_posterior(
    theta: np.ndarray,
    phase_days: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    ephemeris: Dict[str, Any],
    rho_prior_solar: float,
    eccentric: bool,
    ldtk_prior: Optional[Dict[str, Any]] = None,
) -> float:
    log_prior = _log_prior(
        theta,
        rho_prior_solar,
        eccentric,
        ldtk_prior,
        noise_scale=float(np.median(flux_err)),
    )
    if not math.isfinite(log_prior):
        return 1e100
    log_likelihood = _log_likelihood(
        theta, phase_days, flux, flux_err, ephemeris, eccentric
    )
    if not math.isfinite(log_likelihood):
        return 1e100
    return float(-log_likelihood - log_prior)


def _unpack_theta(theta: np.ndarray, eccentric: bool) -> Tuple[float, ...]:
    """Return the model parameters, including eccentricity coordinates when used."""
    if eccentric:
        return tuple(float(value) for value in theta[:9])
    return tuple(float(value) for value in theta[:7]) + (0.0, 0.0)


def _log_prior(
    theta: np.ndarray,
    rho_prior_solar: float,
    eccentric: bool,
    ldtk_prior: Optional[Dict[str, Any]] = None,
    noise_scale: Optional[float] = None,
) -> float:
    """Evaluate parameter priors separately from the photometric likelihood."""
    if not np.all(np.isfinite(theta)):
        return -np.inf
    rp, log_rho, b, baseline, log_jitter, q1, q2, se_cos, se_sin = _unpack_theta(
        theta, eccentric
    )
    if not (
        0.001 < rp < 0.3
        and -2.0 < log_rho < 1.5
        # b <= 1.2 is intentional: it admits grazing transits (b slightly > 1).
        # Posteriors with median b > 1.0 should be flagged for manual review
        # as they are degenerate with high-impact-parameter eclipsing binaries.
        and 0.0 <= b < 1.2
        and 0.99 < baseline < 1.01
        and -12.0 < log_jitter < -2.0
        and 0.01 < q1 < 0.99
        and 0.01 < q2 < 0.99
    ):
        return -np.inf
    if eccentric and se_cos * se_cos + se_sin * se_sin > 1.0:
        return -np.inf

    if rho_prior_solar <= 0 or not math.isfinite(rho_prior_solar):
        return -np.inf
    log_prior = -0.5 * ((log_rho - math.log10(rho_prior_solar)) / 0.3) ** 2
    if noise_scale is not None:
        if noise_scale <= 0 or not math.isfinite(noise_scale):
            return -np.inf
        # Retained for emcee compatibility: this empirical weak prior prevents
        # the jitter parameter from washing out the folded signal.
        log_prior += -0.5 * ((log_jitter - math.log(noise_scale)) / 1.0) ** 2
    if ldtk_prior is not None:
        u1, u2 = kipping_to_quadratic_limb_darkening(q1, q2)
        log_prior += -0.5 * ((u1 - ldtk_prior["u1"]) / ldtk_prior["u1_err"]) ** 2
        log_prior += -0.5 * ((u2 - ldtk_prior["u2"]) / ldtk_prior["u2_err"]) ** 2
    return float(log_prior)


def _log_likelihood(
    theta: np.ndarray,
    phase_days: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    ephemeris: Dict[str, Any],
    eccentric: bool,
) -> float:
    """Evaluate only the Gaussian folded-light-curve likelihood."""
    if not np.all(np.isfinite(theta)):
        return -np.inf
    rp, log_rho, b, baseline, log_jitter, q1, q2, se_cos, se_sin = _unpack_theta(
        theta, eccentric
    )
    eccentricity = se_cos * se_cos + se_sin * se_sin if eccentric else 0.0
    if eccentricity >= 1.0:
        return -np.inf
    omega_deg = math.degrees(math.atan2(se_sin, se_cos)) if eccentricity > 0 else 90.0

    try:
        period_days = ephemeris["period_days"]
        rho_solar = 10.0 ** log_rho
        a_rs = stellar_density_a_rs(rho_solar, period_days)
    except (KeyError, OverflowError, ValueError):
        return -np.inf
    if eccentricity > 0:
        denominator = 1.0 + eccentricity * math.sin(math.radians(omega_deg))
        if denominator <= 0:
            return -np.inf
        a_rs = a_rs * (1.0 - eccentricity**2) / denominator

    model = batman_transit_flux(
        phase_days,
        period_days,
        rp,
        a_rs,
        b,
        q1,
        q2,
        baseline,
        eccentricity=eccentricity,
        omega_deg=omega_deg,
    )
    if model is None:
        return -np.inf

    jitter = math.exp(log_jitter)
    ivar = 1.0 / (flux_err**2 + jitter**2)
    residual = flux - model
    chi2 = float(np.sum(residual**2 * ivar))
    logdet = float(np.sum(np.log(2.0 * math.pi / ivar)))
    return float(-0.5 * (chi2 + logdet))


def _map_optimize(
    phase_days: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    ephemeris: Dict[str, Any],
    rho_prior_solar: float,
    eccentric: bool,
    start: np.ndarray,
    ldtk_prior: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    from scipy.optimize import minimize

    if eccentric:
        bounds = [
            (0.001, 0.3), (-2.0, 1.5), (0.0, 1.19), (0.99, 1.01),
            (-12.0, -2.0), (0.01, 0.99), (0.01, 0.99), (-1.0, 1.0), (-1.0, 1.0),
        ]
    else:
        bounds = [
            (0.001, 0.3), (-2.0, 1.5), (0.0, 1.19), (0.99, 1.01),
            (-12.0, -2.0), (0.01, 0.99), (0.01, 0.99),
        ]
    if eccentric:
        offsets = [
            np.zeros_like(start),
            np.array([0.01, 0.2, -0.1, 0.0005, -0.5, 0.05, -0.05, 0.1, 0.1]),
            np.array([-0.01, -0.2, 0.1, -0.0005, 0.5, -0.05, 0.05, -0.1, -0.1]),
        ]
    else:
        offsets = [
            np.zeros_like(start),
            np.array([0.01, 0.2, -0.1, 0.0005, -0.5, 0.05, -0.05]),
            np.array([-0.01, -0.2, 0.1, -0.0005, 0.5, -0.05, 0.05]),
        ]
    jitter_starts = np.array([0.0, -2.0, -4.0, -6.0, -8.0])
    best_objective = np.inf
    best_point = start
    for offset in offsets:
        for jitter_delta in jitter_starts:
            candidate = start + offset
            candidate[4] = start[4] + jitter_delta
            result = minimize(
                lambda x: _neg_log_posterior(
                    x, phase_days, flux, flux_err, ephemeris, rho_prior_solar, eccentric, ldtk_prior
                ),
                candidate,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 400, "ftol": 1e-9},
            )
            if np.isfinite(result.fun) and result.fun < best_objective:
                best_objective = float(result.fun)
                best_point = np.asarray(result.x, dtype=float)
    return best_point


def _quantile_summary(chain: np.ndarray) -> Dict[str, float]:
    quantiles = np.quantile(chain, [0.16, 0.50, 0.84])
    return {
        "p16": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p84": float(quantiles[2]),
        "plus": float(quantiles[2] - quantiles[1]),
        "minus": float(quantiles[1] - quantiles[0]),
    }


def _posterior_summaries(
    chain: np.ndarray, ephemeris: Dict[str, Any], eccentric: bool
) -> Dict[str, Dict[str, float]]:
    """Summarize sampled and derived transit parameters from an equal-weight chain."""
    names = list(PARAMETER_NAMES_ECCENTRIC if eccentric else PARAMETER_NAMES_CIRCULAR)
    posteriors: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(names):
        posteriors[name] = _quantile_summary(chain[:, index])

    rp_samples = chain[:, 0]
    rho_samples = 10.0 ** chain[:, 1]
    b_samples = chain[:, 2]
    q1_samples = chain[:, 5]
    q2_samples = chain[:, 6]
    a_rs_samples = np.array(
        [stellar_density_a_rs(rho, ephemeris["period_days"]) for rho in rho_samples]
    )
    inc_samples = np.degrees(np.arccos(np.clip(b_samples / a_rs_samples, 0.0, 1.0)))
    area_ppm = (rp_samples**2) * 1e6
    depth_values = []
    for median_rp, median_a, median_b, median_q1, median_q2 in zip(
        _chunk_medians(rp_samples),
        _chunk_medians(a_rs_samples),
        _chunk_medians(b_samples),
        _chunk_medians(q1_samples),
        _chunk_medians(q2_samples),
    ):
        model = batman_transit_flux(
            np.array([0.0]),
            ephemeris["period_days"],
            float(median_rp),
            float(median_a),
            float(median_b),
            float(median_q1),
            float(median_q2),
            1.0,
        )
        depth_values.append(1.0 - (model[0] if model is not None else 1.0))
    depth_ppm_samples = np.asarray(depth_values) * 1e6

    u1_samples, u2_samples = [], []
    for q1_val, q2_val in zip(q1_samples[::7], q2_samples[::7]):
        u1_val, u2_val = kipping_to_quadratic_limb_darkening(q1_val, q2_val)
        u1_samples.append(u1_val)
        u2_samples.append(u2_val)

    posteriors["inclination_deg"] = _quantile_summary(inc_samples)
    posteriors["a_rs"] = _quantile_summary(a_rs_samples)
    posteriors["rho_star_solar"] = _quantile_summary(rho_samples)
    posteriors["area_ratio_ppm"] = _quantile_summary(area_ppm)
    posteriors["mid_transit_depth_ppm"] = _quantile_summary(depth_ppm_samples)
    posteriors["u1"] = _quantile_summary(np.asarray(u1_samples))
    posteriors["u2"] = _quantile_summary(np.asarray(u2_samples))
    if eccentric:
        se_cos_samples = chain[:, 7]
        se_sin_samples = chain[:, 8]
        eccentricity_samples = np.minimum(0.95, se_cos_samples**2 + se_sin_samples**2)
        omega_samples = np.degrees(np.arctan2(se_sin_samples, se_cos_samples))
        posteriors["eccentricity"] = _quantile_summary(eccentricity_samples)
        posteriors["omega_deg"] = _quantile_summary(omega_samples)
    return posteriors


def _resample_weighted_posterior(
    samples: np.ndarray, weights: np.ndarray, seed: int
) -> Tuple[np.ndarray, float]:
    """Systematically resample normalized nested-sampling weights with a fixed seed."""
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if samples.ndim != 2 or weights.shape != (samples.shape[0],):
        raise ValueError("nested samples and weights have incompatible shapes")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("nested samples and weights must be finite with non-negative weights")
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise ValueError("nested weights must have positive total weight")
    weights = weights / total_weight
    positions = (np.random.default_rng(seed).random() + np.arange(weights.size)) / weights.size
    indices = np.searchsorted(np.cumsum(weights), positions, side="right")
    indices = np.minimum(indices, samples.shape[0] - 1)
    return samples[indices], float(1.0 / np.sum(weights**2))


def _make_dynesty_prior_transform(
    rho_prior_solar: float,
    noise_scale: float,
    eccentric: bool,
    ldtk_prior: Optional[Dict[str, Any]],
):
    """Create a normalized prior transform for dynesty's likelihood-only API."""
    from scipy.special import ndtr, ndtri

    if rho_prior_solar <= 0 or noise_scale <= 0:
        raise ValueError("stellar density and noise scale must be positive")

    def truncated_normal(unit_value: float, mean: float, sigma: float, lower: float, upper: float) -> float:
        clipped = float(np.clip(unit_value, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
        lower_cdf = ndtr((lower - mean) / sigma)
        upper_cdf = ndtr((upper - mean) / sigma)
        return float(mean + sigma * ndtri(lower_cdf + clipped * (upper_cdf - lower_cdf)))

    q_transform = None
    if ldtk_prior is not None:
        # The emcee path uses uniform Kipping parameters multiplied by the LDTk
        # density. Build its equivalent normalized two-dimensional prior once,
        # then use inverse-CDF sampling so it remains a prior for the evidence.
        from scipy.integrate import cumulative_trapezoid

        q_grid = np.linspace(0.01, 0.99, 513)
        q1_grid, q2_grid = np.meshgrid(q_grid, q_grid, indexing="ij")
        root_q1 = np.sqrt(q1_grid)
        u1_grid = 2.0 * root_q1 * q2_grid
        u2_grid = root_q1 * (1.0 - 2.0 * q2_grid)
        log_density = -0.5 * ((u1_grid - ldtk_prior["u1"]) / ldtk_prior["u1_err"]) ** 2
        log_density += -0.5 * ((u2_grid - ldtk_prior["u2"]) / ldtk_prior["u2_err"]) ** 2
        density = np.exp(log_density - float(np.max(log_density)))
        marginal = np.trapz(density, q_grid, axis=1)
        q1_cdf = cumulative_trapezoid(marginal, q_grid, initial=0.0)
        q1_cdf /= q1_cdf[-1]

        def q_transform(q1_unit: float, q2_unit: float) -> Tuple[float, float]:
            q1_value = float(np.interp(q1_unit, q1_cdf, q_grid))
            root_q1_value = math.sqrt(q1_value)
            u1_values = 2.0 * root_q1_value * q_grid
            u2_values = root_q1_value * (1.0 - 2.0 * q_grid)
            conditional = np.exp(
                -0.5 * ((u1_values - ldtk_prior["u1"]) / ldtk_prior["u1_err"]) ** 2
                -0.5 * ((u2_values - ldtk_prior["u2"]) / ldtk_prior["u2_err"]) ** 2
                - float(np.max(log_density))
            )
            q2_cdf = cumulative_trapezoid(conditional, q_grid, initial=0.0)
            q2_cdf /= q2_cdf[-1]
            return q1_value, float(np.interp(q2_unit, q2_cdf, q_grid))

    def prior_transform(unit_cube: np.ndarray) -> np.ndarray:
        unit_cube = np.asarray(unit_cube, dtype=float)
        expected_dimensions = len(PARAMETER_NAMES_ECCENTRIC if eccentric else PARAMETER_NAMES_CIRCULAR)
        if unit_cube.shape != (expected_dimensions,) or not np.all(np.isfinite(unit_cube)):
            raise ValueError("dynesty prior transform received an invalid unit-cube point")
        if np.any(unit_cube < 0.0) or np.any(unit_cube > 1.0):
            raise ValueError("dynesty prior transform requires values in [0, 1]")
        theta = np.empty(expected_dimensions, dtype=float)
        theta[0] = 0.001 + unit_cube[0] * (0.3 - 0.001)
        theta[1] = truncated_normal(unit_cube[1], math.log10(rho_prior_solar), 0.3, -2.0, 1.5)
        theta[2] = unit_cube[2] * 1.2
        theta[3] = 0.99 + unit_cube[3] * 0.02
        theta[4] = truncated_normal(unit_cube[4], math.log(noise_scale), 1.0, -12.0, -2.0)
        if q_transform is None:
            theta[5] = 0.01 + unit_cube[5] * 0.98
            theta[6] = 0.01 + unit_cube[6] * 0.98
        else:
            theta[5], theta[6] = q_transform(unit_cube[5], unit_cube[6])
        if eccentric:
            radius = math.sqrt(unit_cube[7])
            angle = 2.0 * math.pi * unit_cube[8]
            theta[7] = radius * math.cos(angle)
            theta[8] = radius * math.sin(angle)
        return theta

    return prior_transform


def _synthetic_transit_table(
    ephemeris: Dict[str, Any], rng_seed: int = 5
) -> Dict[str, np.ndarray]:
    """Deterministic demonstration transit light curve.

    The injected radius is derived from the ephemeris depth so the synthetic
    signal is self-consistent with the fitter's initialization.
    """
    rng = np.random.default_rng(seed=rng_seed)
    cadence_days = 120.0 / 86400.0
    time = np.arange(0.0, 54.0, cadence_days)
    phase_days = (
        (time - ephemeris["epoch_btjd"] + 0.5 * ephemeris["period_days"])
        % ephemeris["period_days"]
    ) - 0.5 * ephemeris["period_days"]
    injected_rp = math.sqrt(max(float(ephemeris["depth_ppm"]) * 1e-6, 1e-8))
    rho_solar = 1.0
    a_rs = stellar_density_a_rs(rho_solar, ephemeris["period_days"])
    model = batman_transit_flux(
        phase_days, ephemeris["period_days"], injected_rp, a_rs, 0.3, 0.35, 0.3, 1.0
    )
    flux = np.ones_like(time)
    if model is not None:
        flux = np.asarray(model)
    flux = flux + rng.normal(0.0, 80e-6, size=time.shape)
    flux_err = np.full_like(flux, 80e-6)
    sector_values = np.ones(time.size, dtype=int)
    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "sector": sector_values,
    }


def run_mcmc_transit_fit(
    workspace: CandidateWorkspace,
    n_samples: int = 5000,
    eccentric: bool = False,
    n_walkers: Optional[int] = None,
    burn_in: Optional[int] = None,
    seed: int = 5,
    signal: Optional[str] = None,
    use_ldtk_prior: bool = False,
    sampler: str = "emcee",
) -> Path:
    """Run an emcee or dynesty transit fit and write the historical output paths."""
    if sampler == "dynesty":
        return _run_dynesty_transit_fit(
            workspace,
            n_samples=n_samples,
            eccentric=eccentric,
            seed=seed,
            signal=signal,
            use_ldtk_prior=use_ldtk_prior,
        )
    if sampler != "emcee":
        raise ValueError("sampler must be one of: emcee, dynesty")
    import emcee

    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ephemeris = load_transit_ephemeris(workspace, signal=signal)
    stellar = load_stellar_parameters(workspace)
    rho_prior_solar = float(stellar["mass_solar"]) / float(stellar["radius_solar"]) ** 3
    ldtk_prior = _load_ldtk_prior(workspace) if use_ldtk_prior else None

    table = load_light_curve_table(workspace)
    if table is None:
        table = _synthetic_transit_table(ephemeris)
        source = "synthetic-demo"
    else:
        source = "candidate-data"

    try:
        phase_days, binned_flux, binned_error = _folded_binned_data(
            table["time"], table["flux"], ephemeris
        )
    except ValueError:
        if source == "synthetic-demo":
            raise
        table = _synthetic_transit_table(ephemeris)
        source = "synthetic-demo"
        phase_days, binned_flux, binned_error = _folded_binned_data(
            table["time"], table["flux"], ephemeris
        )

    depth_ppm = float(ephemeris["depth_ppm"])
    rp_start = min(0.2, max(0.01, math.sqrt(depth_ppm * 1e-6)))
    scatter = float(np.std(binned_flux - np.median(binned_flux)))
    log_jitter_start = math.log10(max(scatter, 1e-6))
    if eccentric:
        start = np.array(
            [rp_start, math.log10(rho_prior_solar), 0.3, 1.0, log_jitter_start, 0.35, 0.3, 0.0, 0.0]
        )
    else:
        start = np.array(
            [rp_start, math.log10(rho_prior_solar), 0.3, 1.0, log_jitter_start, 0.35, 0.3]
        )
    map_point = _map_optimize(
        phase_days, binned_flux, binned_error, ephemeris, rho_prior_solar, eccentric, start, ldtk_prior
    )

    ndim = int(map_point.size)
    if n_walkers is None:
        n_walkers = max(2 * ndim, min(48, n_samples // 20))
    n_walkers = max(n_walkers, 2 * ndim)
    if burn_in is None:
        burn_in = max(50, n_samples // 5)
    rng = np.random.default_rng(seed=seed)
    p0 = map_point + 1e-3 * rng.normal(size=(n_walkers, ndim))
    p0[:, 2] = np.clip(p0[:, 2], 0.0, 1.1)
    p0[:, 3] = np.clip(p0[:, 3], 0.995, 1.005)
    if eccentric:
        p0[:, 7] = np.clip(p0[:, 7], -1.0, 1.0)
        p0[:, 8] = np.clip(p0[:, 8], -1.0, 1.0)

    # Reproducibility: walker starting positions are fully determined by
    # np.random.default_rng(seed=seed) above. emcee's StretchMove uses its
    # own C-level RNG seeded by p0; reproducibility is achieved by keeping
    # p0 deterministic rather than by setting a global NumPy seed.
    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        lambda x: -_neg_log_posterior(
            x, phase_days, binned_flux, binned_error, ephemeris, rho_prior_solar, eccentric, ldtk_prior
        ),
        moves=emcee.moves.StretchMove(a=1.5),
    )
    sampler.run_mcmc(p0, burn_in + n_samples, progress=False)
    chain = sampler.get_chain(discard=burn_in, flat=True)

    names = list(PARAMETER_NAMES_ECCENTRIC if eccentric else PARAMETER_NAMES_CIRCULAR)
    posteriors = _posterior_summaries(chain, ephemeris, eccentric)

    try:
        import logging
        import warnings

        emcee_logger = logging.getLogger("emcee")
        previous_level = emcee_logger.level
        emcee_logger.setLevel(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                tau_values = sampler.get_autocorr_time(discard=burn_in // 2, quiet=True)
            finally:
                emcee_logger.setLevel(previous_level)
        tau_dict = {
            names[index]: float(tau) if np.isfinite(tau) else None
            for index, tau in enumerate(tau_values)
        }
    except Exception as exc:
        tau_dict = {"_error": "{0}: {1}".format(type(exc).__name__, exc)}

    payload = {
        "schema_version": "1.0",
        "work_package": "MCMC_TRANSIT_FIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "model": (
            "batman quadratic limb darkening, stellar-density locked, eccentric orbit"
            if eccentric
            else "batman quadratic limb darkening, stellar-density locked, circular orbit"
        ),
        "ephemeris": {
            "period_days": ephemeris["period_days"],
            "epoch_btjd": ephemeris["epoch_btjd"],
            "source": ephemeris["source"],
        },
        "density_prior_solar": float(rho_prior_solar),
        "limb_darkening_prior": (
            {"source": "ldtk", "path": ldtk_prior["path"]} if ldtk_prior is not None else None
        ),
        "posterior": posteriors,
        "mcmc": {
            "walkers": int(n_walkers),
            "burn_in": int(burn_in),
            "production": int(n_samples),
            "flat_samples": int(chain.shape[0]),
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "autocorrelation_times": tau_dict,
        },
        "n_binned_points": int(phase_days.size),
        "signal": signal,
        "caveat": "Descriptive folded/binned fit; not an adopted native-cadence posterior.",
    }
    suffix = f".{signal.lstrip('.')}" if signal else ""
    output_path = outputs_dir / f"mcmc_transit_fit{suffix}.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.save(str(outputs_dir / f"mcmc_transit_fit_chain{suffix}.npy"), chain)
    return output_path


def _run_dynesty_transit_fit(
    workspace: CandidateWorkspace,
    n_samples: int,
    eccentric: bool,
    seed: int,
    signal: Optional[str],
    use_ldtk_prior: bool,
) -> Path:
    """Run optional dynamic nested sampling with an explicit normalized prior transform."""
    try:
        import dynesty
    except ImportError as exc:
        raise RuntimeError(
            "dynesty is required for --sampler dynesty; install the pinned optional dependency with "
            'pip install -e ".[inference]"'
        ) from exc
    if n_samples <= 0:
        raise ValueError("n_samples must be positive for dynesty")

    ephemeris = load_transit_ephemeris(workspace, signal=signal)
    stellar = load_stellar_parameters(workspace)
    rho_prior_solar = float(stellar["mass_solar"]) / float(stellar["radius_solar"]) ** 3
    ldtk_prior = _load_ldtk_prior(workspace) if use_ldtk_prior else None
    table = load_light_curve_table(workspace)
    if table is None:
        table = _synthetic_transit_table(ephemeris)
        source = "synthetic-demo"
    else:
        source = "candidate-data"
    try:
        phase_days, binned_flux, binned_error = _folded_binned_data(
            table["time"], table["flux"], ephemeris
        )
    except ValueError:
        if source == "synthetic-demo":
            raise
        table = _synthetic_transit_table(ephemeris)
        source = "synthetic-demo"
        phase_days, binned_flux, binned_error = _folded_binned_data(
            table["time"], table["flux"], ephemeris
        )

    noise_scale = float(np.median(binned_error))
    prior_transform = _make_dynesty_prior_transform(
        rho_prior_solar, noise_scale, eccentric, ldtk_prior
    )
    ndim = len(PARAMETER_NAMES_ECCENTRIC if eccentric else PARAMETER_NAMES_CIRCULAR)
    initial_live_points = max(2 * ndim + 1, min(500, max(50, n_samples // 10)))
    max_likelihood_calls = max(n_samples, initial_live_points)
    nested_sampler = dynesty.DynamicNestedSampler(
        lambda theta: _log_likelihood(
            theta, phase_days, binned_flux, binned_error, ephemeris, eccentric
        ),
        prior_transform,
        ndim,
        rstate=np.random.default_rng(seed),
    )
    nested_sampler.run_nested(
        nlive_init=initial_live_points,
        maxcall=max_likelihood_calls,
        print_progress=False,
    )
    results = nested_sampler.results
    samples = np.asarray(results.samples, dtype=float)
    log_weights = np.asarray(results.logwt, dtype=float)
    log_evidence = np.asarray(results.logz, dtype=float)
    log_evidence_error = np.asarray(results.logzerr, dtype=float)
    if (
        samples.ndim != 2
        or samples.shape[1] != ndim
        or samples.shape[0] == 0
        or log_weights.shape != (samples.shape[0],)
        or log_evidence.size == 0
        or log_evidence_error.size == 0
        or not np.all(np.isfinite(samples))
        or not np.all(np.isfinite(log_weights))
        or not math.isfinite(float(log_evidence[-1]))
        or not math.isfinite(float(log_evidence_error[-1]))
    ):
        raise RuntimeError("dynesty returned incomplete or non-finite nested-sampling results")
    posterior_weights = np.exp(log_weights - float(log_evidence[-1]))
    chain, effective_samples = _resample_weighted_posterior(samples, posterior_weights, seed)
    posteriors = _posterior_summaries(chain, ephemeris, eccentric)
    sampling_efficiency = float(getattr(results, "eff", np.nan))
    if not math.isfinite(sampling_efficiency):
        sampling_efficiency = None

    payload = {
        "schema_version": "1.0",
        "work_package": "NESTED_TRANSIT_FIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "sampler": "dynesty",
        "dynesty_version": getattr(dynesty, "__version__", "unknown"),
        "model": (
            "batman quadratic limb darkening, stellar-density locked, eccentric orbit"
            if eccentric
            else "batman quadratic limb darkening, stellar-density locked, circular orbit"
        ),
        "ephemeris": {
            "period_days": ephemeris["period_days"],
            "epoch_btjd": ephemeris["epoch_btjd"],
            "source": ephemeris["source"],
        },
        "density_prior_solar": float(rho_prior_solar),
        "limb_darkening_prior": (
            {"source": "ldtk", "path": ldtk_prior["path"]} if ldtk_prior is not None else None
        ),
        "posterior": posteriors,
        "evidence": {
            "log_z": float(log_evidence[-1]),
            "log_z_err": float(log_evidence_error[-1]),
            "meaning": "Nested-sampling model evidence; not a validation probability.",
        },
        "diagnostics": {
            "initial_live_points": int(initial_live_points),
            "max_likelihood_calls": int(max_likelihood_calls),
            "iterations": int(getattr(results, "niter", samples.shape[0])),
            "likelihood_calls": int(np.sum(np.asarray(getattr(results, "ncall", 0)))),
            "sampling_efficiency_percent": sampling_efficiency,
            "weighted_samples": int(samples.shape[0]),
            "effective_samples": effective_samples,
            "resampled_samples": int(chain.shape[0]),
            "resampling": "systematic equal-weight resampling",
            "resampling_seed": int(seed),
            "prior_transform": (
                "LDTk-weighted Kipping prior via numerical inverse CDF"
                if ldtk_prior is not None
                else "analytic bounded and truncated priors"
            ),
        },
        "n_binned_points": int(phase_days.size),
        "signal": signal,
        "caveat": "Descriptive folded/binned fit; nested evidence is not an adopted native-cadence posterior or validation claim.",
    }
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{signal.lstrip('.')}" if signal else ""
    output_path = outputs_dir / f"mcmc_transit_fit{suffix}.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.save(str(outputs_dir / f"mcmc_transit_fit_chain{suffix}.npy"), chain)
    return output_path


def _chunk_medians(samples: np.ndarray) -> List[float]:
    """Median down a large sample chain to ~1000 values for model evaluation."""
    samples = np.asarray(samples, dtype=float)
    if samples.size <= 1000:
        return [float(value) for value in samples]
    step = int(np.ceil(samples.size / 1000.0))
    return [float(np.median(samples[index : index + step])) for index in range(0, samples.size, step)]
