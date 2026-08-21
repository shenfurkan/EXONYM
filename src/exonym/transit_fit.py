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
from .workspace import CandidateWorkspace, validate_signal_suffix

# Physical constants in CGS units
G_CGS = 6.67430e-8          # Gravitational constant (cm^3 g^-1 s^-2)
RHO_SUN_GCM3 = 1.408        # Mean solar density (g cm^-3)
WINDOW_HALF_HOURS = 13.0    # Folded light curve crop window half-width (hours)
BIN_MINUTES = 8.0           # Default phase-binning resolution (minutes)
SUPERSAMPLE_FACTOR = 7      # Numerical exposure integration sub-sampling factor
EXPTIME_SECONDS = 120.0     # Nominal TESS 2-minute SPOC cadence (seconds)
FITTED_BIN_EXPOSURE_SECONDS = BIN_MINUTES * 60.0

# Parameter vectors for circular and eccentric orbits
PARAMETER_NAMES_CIRCULAR = (
    "rp_rs",             # Planet-to-star radius ratio (R_p / R_star)
    "log_rho_star",      # log10(rho_star / rho_sun), stellar density proxy
    "impact_parameter",  # Transit impact parameter at inferior conjunction
    "baseline",          # Out-of-transit flux normalization baseline
    "log_jitter",        # natural log(sigma_jitter), added in quadrature to flux uncertainties
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


def conjunction_distance_a_rs(a_rs: float, eccentricity: float, omega_deg: float) -> Optional[float]:
    """Return the planet-star separation in stellar radii at inferior conjunction."""
    if (
        not math.isfinite(a_rs)
        or not math.isfinite(eccentricity)
        or not math.isfinite(omega_deg)
        or a_rs <= 0
        or not 0.0 <= eccentricity < 1.0
    ):
        return None
    denominator = 1.0 + eccentricity * math.sin(math.radians(omega_deg))
    if denominator <= 0:
        return None
    return a_rs * (1.0 - eccentricity**2) / denominator


def inclination_deg_from_impact_parameter(
    a_rs: float, impact_parameter: float, eccentricity: float = 0.0, omega_deg: float = 90.0
) -> Optional[float]:
    """Convert conjunction impact parameter to inclination for a Keplerian orbit."""
    conjunction_distance = conjunction_distance_a_rs(a_rs, eccentricity, omega_deg)
    if conjunction_distance is None or not math.isfinite(impact_parameter) or impact_parameter < 0:
        return None
    cosine = impact_parameter / conjunction_distance
    if not 0.0 <= cosine < 1.0:
        return None
    return math.degrees(math.acos(cosine))


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
    exposure_seconds: float = FITTED_BIN_EXPOSURE_SECONDS,
) -> Optional[np.ndarray]:
    """Evaluate a batman quadratic limb-darkening model at transit-relative phase.

    ``a_rs`` remains the Keplerian semimajor axis expected by batman. For an
    eccentric orbit, ``impact_parameter`` is converted through the
    conjunction-distance relation before setting the inclination.
    """
    if not math.isfinite(exposure_seconds) or exposure_seconds <= 0:
        return None
    inclination_deg = inclination_deg_from_impact_parameter(
        a_rs, impact_parameter, eccentricity, omega_deg
    )
    if inclination_deg is None:
        return None
    try:
        import batman

        u1, u2 = kipping_to_quadratic_limb_darkening(q1, q2)
        params = batman.TransitParams()
        params.t0 = 0.0
        params.per = period_days
        params.rp = rp_rs
        params.a = a_rs
        params.inc = inclination_deg
        params.ecc = eccentricity
        params.w = omega_deg
        params.u = [u1, u2]
        params.limb_dark = "quadratic"

        model = batman.TransitModel(
            params,
            np.asarray(phase_days, dtype=float),
            supersample_factor=SUPERSAMPLE_FACTOR,
            exp_time=float(exposure_seconds) / 86400.0,
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


def _native_transit_window_data(
    table: Dict[str, Any],
    ephemeris: Dict[str, Any],
    window_half_hours: float = WINDOW_HALF_HOURS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], np.ndarray]:
    """Return native-cadence transit windows with sector-specific integrations.

    The light-curve loader supplies normalized, quality-filtered observations.
    This helper retains every finite cadence inside the declared fit window,
    maps each cadence to a compact sector index, and estimates one integration
    time per sector from within-sector sampling intervals.  It does not mix
    cadences across sectors or phase-bin them before inference.
    """
    try:
        time = np.asarray(table["time"], dtype=float)
        flux = np.asarray(table["flux"], dtype=float)
        flux_err = np.asarray(table["flux_err"], dtype=float)
        sectors = np.asarray(table["sector"], dtype=int)
        period_days = float(ephemeris["period_days"])
        epoch_btjd = float(ephemeris["epoch_btjd"])
        duration_days = float(ephemeris["duration_days"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("candidate photometry or ephemeris is malformed") from exc
    if (
        time.ndim != 1
        or flux.shape != time.shape
        or flux_err.shape != time.shape
        or sectors.shape != time.shape
        or not math.isfinite(period_days)
        or period_days <= 0
        or not math.isfinite(epoch_btjd)
        or not math.isfinite(duration_days)
        or duration_days <= 0
        or not math.isfinite(window_half_hours)
        or window_half_hours <= 0
    ):
        raise ValueError("candidate photometry cannot define a native transit window")

    effective_window = min(window_half_hours, max(2.5, duration_days * 24.0 * 2.5))
    phase_days = ((time - epoch_btjd + 0.5 * period_days) % period_days) - 0.5 * period_days
    valid = (
        np.isfinite(time)
        & np.isfinite(phase_days)
        & np.isfinite(flux)
        & np.isfinite(flux_err)
        & (flux_err > 0)
        & (np.abs(phase_days) <= effective_window / 24.0)
    )
    if int(np.count_nonzero(valid)) < 100:
        raise ValueError("insufficient native-cadence transit-window coverage")
    phase_days = phase_days[valid]
    flux = flux[valid]
    flux_err = flux_err[valid]
    time = time[valid]
    sectors = sectors[valid]

    sector_labels = sorted(int(value) for value in np.unique(sectors))
    if not sector_labels:
        raise ValueError("candidate transit window has no sector ownership")
    sector_index = np.empty(sectors.size, dtype=int)
    exposure_seconds = []
    for index, sector_label in enumerate(sector_labels):
        selected = sectors == sector_label
        sector_time = np.sort(time[selected])
        cadence_seconds = np.diff(sector_time) * 86400.0
        cadence_seconds = cadence_seconds[np.isfinite(cadence_seconds) & (cadence_seconds > 0)]
        if cadence_seconds.size == 0:
            raise ValueError("candidate sector has no measurable integration cadence")
        median_cadence_seconds = float(np.median(cadence_seconds))
        if not math.isfinite(median_cadence_seconds) or median_cadence_seconds <= 0:
            raise ValueError("candidate sector has an invalid integration cadence")
        sector_index[selected] = index
        exposure_seconds.append(median_cadence_seconds)
    return phase_days, flux, flux_err, sector_index, sector_labels, np.asarray(exposure_seconds)


def _parameter_names(
    eccentric: bool,
    n_sectors: int = 1,
    sector_labels: Optional[Sequence[int]] = None,
) -> List[str]:
    """Return the sampled-parameter names for one or more observed sectors."""
    if n_sectors <= 0:
        raise ValueError("transit likelihood requires at least one sector")
    if n_sectors == 1:
        return list(PARAMETER_NAMES_ECCENTRIC if eccentric else PARAMETER_NAMES_CIRCULAR)
    labels = list(sector_labels) if sector_labels is not None else list(range(n_sectors))
    if len(labels) != n_sectors:
        raise ValueError("sector labels do not match the transit likelihood")
    names = ["rp_rs", "log_rho_star", "impact_parameter"]
    names.extend("baseline_sector_{0}".format(int(label)) for label in labels)
    names.extend(["log_jitter", "q1", "q2"])
    if eccentric:
        names.extend(["sqe_cosw", "sqe_sinw"])
    return names


def _parameter_count(eccentric: bool, n_sectors: int = 1) -> int:
    """Return the number of sampled parameters for the selected sectors."""
    return 6 + n_sectors + (2 if eccentric else 0)


def _initial_fit_parameters(
    depth_ppm: float,
    rho_prior_solar: float,
    scatter: float,
    eccentric: bool,
    n_sectors: int = 1,
) -> np.ndarray:
    """Build a finite starting point with explicit logarithm conventions.

    ``log_rho_star`` is base-10 log density in solar units, while
    ``log_jitter`` is the natural log of normalized flux scatter because the
    likelihood recovers jitter as ``exp(log_jitter)``.
    """
    if (
        not math.isfinite(depth_ppm)
        or not math.isfinite(rho_prior_solar)
        or not math.isfinite(scatter)
        or depth_ppm < 0
        or rho_prior_solar <= 0
        or scatter < 0
    ):
        raise ValueError("fit initialization requires finite non-negative depth and scatter plus positive density")
    rp_start = min(0.2, max(0.01, math.sqrt(depth_ppm * 1e-6)))
    log_jitter_start = math.log(max(scatter, 1e-6))
    if n_sectors <= 0:
        raise ValueError("transit initialization requires at least one sector")
    common = [rp_start, math.log10(rho_prior_solar), 0.3]
    common.extend([1.0] * n_sectors)
    common.extend([log_jitter_start, 0.35, 0.3])
    if eccentric:
        common.extend([0.0, 0.0])
    return np.asarray(common, dtype=float)


def _stellar_density_prior(stellar: Dict[str, Any]) -> Dict[str, float]:
    """Propagate candidate-supplied stellar mass and radius uncertainties.

    The fit needs symmetric one-sigma ``mass_solar_err`` and
    ``radius_solar_err`` values.  It approximates their errors as independent
    and propagates ``rho_star = M_star / R_star**3`` into base-10 log-density
    space.  No fixed generic density width is substituted when that evidence
    is absent.
    """
    try:
        mass_solar = float(stellar["mass_solar"])
        mass_solar_err = float(stellar["mass_solar_err"])
        radius_solar = float(stellar["radius_solar"])
        radius_solar_err = float(stellar["radius_solar_err"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "transit fitting requires candidate stellar mass_solar_err and radius_solar_err"
        ) from exc
    if not all(
        math.isfinite(value)
        for value in (mass_solar, mass_solar_err, radius_solar, radius_solar_err)
    ) or mass_solar <= 0 or mass_solar_err <= 0 or radius_solar <= 0 or radius_solar_err <= 0:
        raise RuntimeError(
            "transit fitting requires positive finite candidate stellar mass and radius uncertainties"
        )
    rho_solar = mass_solar / radius_solar**3
    relative_density_err = math.sqrt(
        (mass_solar_err / mass_solar) ** 2 + (3.0 * radius_solar_err / radius_solar) ** 2
    )
    log10_sigma = relative_density_err / math.log(10.0)
    if not math.isfinite(rho_solar) or rho_solar <= 0 or not math.isfinite(log10_sigma) or log10_sigma <= 0:
        raise RuntimeError("candidate stellar uncertainties cannot produce a finite density prior")
    return {
        "rho_solar": float(rho_solar),
        "log10_sigma": float(log10_sigma),
        "mass_solar": mass_solar,
        "mass_solar_err": mass_solar_err,
        "radius_solar": radius_solar,
        "radius_solar_err": radius_solar_err,
    }


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
    rho_prior_log10_sigma: float,
    eccentric: bool,
    ldtk_prior: Optional[Dict[str, Any]] = None,
    sector_index: Optional[np.ndarray] = None,
    exposure_seconds_by_sector: Optional[np.ndarray] = None,
    n_sectors: int = 1,
) -> float:
    log_prior = _log_prior(
        theta,
        rho_prior_solar,
        rho_prior_log10_sigma,
        eccentric,
        ldtk_prior,
        noise_scale=float(np.median(flux_err)),
        n_sectors=n_sectors,
    )
    if not math.isfinite(log_prior):
        return float("inf")
    log_likelihood = _log_likelihood(
        theta,
        phase_days,
        flux,
        flux_err,
        ephemeris,
        eccentric,
        sector_index=sector_index,
        exposure_seconds_by_sector=exposure_seconds_by_sector,
        n_sectors=n_sectors,
    )
    if not math.isfinite(log_likelihood):
        return float("inf")
    return float(-log_likelihood - log_prior)


def _unpack_theta(
    theta: np.ndarray, eccentric: bool, n_sectors: int = 1
) -> Tuple[Any, ...]:
    """Return physical parameters and one flux normalization per data sector."""
    expected = _parameter_count(eccentric, n_sectors)
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (expected,):
        raise ValueError("transit parameter vector has an invalid sector layout")
    baseline_stop = 3 + n_sectors
    baselines = theta[3:baseline_stop]
    log_jitter, q1, q2 = (float(value) for value in theta[baseline_stop:baseline_stop + 3])
    if eccentric:
        se_cos, se_sin = (float(value) for value in theta[baseline_stop + 3:baseline_stop + 5])
    else:
        se_cos, se_sin = 0.0, 0.0
    return (
        float(theta[0]),
        float(theta[1]),
        float(theta[2]),
        baselines,
        log_jitter,
        q1,
        q2,
        se_cos,
        se_sin,
    )


def _log_prior(
    theta: np.ndarray,
    rho_prior_solar: float,
    rho_prior_log10_sigma: float,
    eccentric: bool,
    ldtk_prior: Optional[Dict[str, Any]] = None,
    noise_scale: Optional[float] = None,
    n_sectors: int = 1,
) -> float:
    """Evaluate parameter priors separately from the photometric likelihood."""
    if not np.all(np.isfinite(theta)):
        return -np.inf
    try:
        rp, log_rho, b, baselines, log_jitter, q1, q2, se_cos, se_sin = _unpack_theta(
            theta, eccentric, n_sectors
        )
    except ValueError:
        return -np.inf
    if not (
        0.001 < rp < 0.3
        and -2.0 < log_rho < 1.5
        # b <= 1.2 is intentional: it admits grazing transits (b slightly > 1).
        # Posteriors with median b > 1.0 should be flagged for manual review
        # as they are degenerate with high-impact-parameter eclipsing binaries.
        and 0.0 <= b < 1.2
        and bool(np.all((0.99 < baselines) & (baselines < 1.01)))
        and -12.0 < log_jitter < -2.0
        and 0.01 < q1 < 0.99
        and 0.01 < q2 < 0.99
    ):
        return -np.inf
    if eccentric and se_cos * se_cos + se_sin * se_sin > 1.0:
        return -np.inf

    if (
        rho_prior_solar <= 0
        or not math.isfinite(rho_prior_solar)
        or rho_prior_log10_sigma <= 0
        or not math.isfinite(rho_prior_log10_sigma)
    ):
        return -np.inf
    log_prior = -0.5 * (
        (log_rho - math.log10(rho_prior_solar)) / rho_prior_log10_sigma
    ) ** 2
    if noise_scale is not None:
        if noise_scale <= 0 or not math.isfinite(noise_scale):
            return -np.inf
        # A weak empirical prior prevents jitter from washing out the transit
        # signal while retaining the candidate-reported flux uncertainties.
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
    sector_index: Optional[np.ndarray] = None,
    exposure_seconds_by_sector: Optional[np.ndarray] = None,
    n_sectors: int = 1,
) -> float:
    """Evaluate a Gaussian native-cadence likelihood with sector baselines."""
    if not np.all(np.isfinite(theta)):
        return -np.inf
    try:
        rp, log_rho, b, baselines, log_jitter, q1, q2, se_cos, se_sin = _unpack_theta(
            theta, eccentric, n_sectors
        )
    except ValueError:
        return -np.inf
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
    if sector_index is None:
        sector_index = np.zeros(np.asarray(phase_days).size, dtype=int)
    else:
        sector_index = np.asarray(sector_index, dtype=int)
    if sector_index.shape != np.asarray(phase_days).shape or np.any(sector_index < 0) or np.any(sector_index >= n_sectors):
        return -np.inf
    if exposure_seconds_by_sector is None:
        exposure_seconds_by_sector = np.full(n_sectors, FITTED_BIN_EXPOSURE_SECONDS)
    else:
        exposure_seconds_by_sector = np.asarray(exposure_seconds_by_sector, dtype=float)
    if (
        exposure_seconds_by_sector.shape != (n_sectors,)
        or not np.all(np.isfinite(exposure_seconds_by_sector))
        or np.any(exposure_seconds_by_sector <= 0)
    ):
        return -np.inf
    model = np.empty(np.asarray(phase_days).shape, dtype=float)
    for sector_number in range(n_sectors):
        selected = sector_index == sector_number
        if not np.any(selected):
            return -np.inf
        sector_model = batman_transit_flux(
            np.asarray(phase_days)[selected],
            period_days,
            rp,
            a_rs,
            b,
            q1,
            q2,
            1.0,
            eccentricity=eccentricity,
            omega_deg=omega_deg,
            exposure_seconds=float(exposure_seconds_by_sector[sector_number]),
        )
        if sector_model is None:
            return -np.inf
        model[selected] = baselines[sector_number] * sector_model

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
    rho_prior_log10_sigma: float,
    eccentric: bool,
    start: np.ndarray,
    ldtk_prior: Optional[Dict[str, Any]] = None,
    sector_index: Optional[np.ndarray] = None,
    exposure_seconds_by_sector: Optional[np.ndarray] = None,
    n_sectors: int = 1,
) -> np.ndarray:
    from scipy.optimize import minimize

    if start.shape != (_parameter_count(eccentric, n_sectors),):
        raise ValueError("transit MAP start has an invalid sector layout")
    bounds = [(0.001, 0.3), (-2.0, 1.5), (0.0, 1.19)]
    bounds.extend([(0.99, 1.01)] * n_sectors)
    bounds.extend([(-12.0, -2.0), (0.01, 0.99), (0.01, 0.99)])
    if eccentric:
        bounds.extend([(-1.0, 1.0), (-1.0, 1.0)])
    offsets = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.01, 0.2, -0.1, 0.05, -0.05, 0.1),
        (-0.01, -0.2, 0.1, -0.05, 0.05, -0.1),
    )
    jitter_starts = np.array([0.0, -2.0, -4.0, -6.0, -8.0])
    best_objective = np.inf
    best_point = start

    def optimizer_objective(candidate_theta: np.ndarray) -> float:
        """Keep numerical optimization finite while MCMC retains strict support."""
        value = _neg_log_posterior(
            candidate_theta,
            phase_days,
            flux,
            flux_err,
            ephemeris,
            rho_prior_solar,
            rho_prior_log10_sigma,
            eccentric,
            ldtk_prior,
            sector_index=sector_index,
            exposure_seconds_by_sector=exposure_seconds_by_sector,
            n_sectors=n_sectors,
        )
        return value if math.isfinite(value) else 1e100

    baseline_stop = 3 + n_sectors
    for rp_delta, rho_delta, impact_delta, q1_delta, q2_delta, eccentric_delta in offsets:
        for jitter_delta in jitter_starts:
            candidate = start.copy()
            candidate[0] += rp_delta
            candidate[1] += rho_delta
            candidate[2] += impact_delta
            candidate[baseline_stop] += jitter_delta
            candidate[baseline_stop + 1] += q1_delta
            candidate[baseline_stop + 2] += q2_delta
            if eccentric:
                candidate[baseline_stop + 3] += eccentric_delta
                candidate[baseline_stop + 4] += eccentric_delta
            result = minimize(
                optimizer_objective,
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
    chain: np.ndarray,
    ephemeris: Dict[str, Any],
    eccentric: bool,
    n_sectors: int = 1,
    sector_labels: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Summarize sampled and derived transit parameters from an equal-weight chain."""
    names = _parameter_names(eccentric, n_sectors, sector_labels)
    if chain.ndim != 2 or chain.shape[1] != len(names):
        raise ValueError("transit posterior chain has an invalid sector layout")
    posteriors: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(names):
        posteriors[name] = _quantile_summary(chain[:, index])

    rp_samples = chain[:, 0]
    rho_samples = 10.0 ** chain[:, 1]
    b_samples = chain[:, 2]
    q1_index = 4 + n_sectors
    q2_index = 5 + n_sectors
    q1_samples = chain[:, q1_index]
    q2_samples = chain[:, q2_index]
    a_rs_samples = np.array(
        [stellar_density_a_rs(rho, ephemeris["period_days"]) for rho in rho_samples]
    )
    if eccentric:
        se_cos_samples = chain[:, 6 + n_sectors]
        se_sin_samples = chain[:, 7 + n_sectors]
        eccentricity_samples = se_cos_samples**2 + se_sin_samples**2
        omega_samples = np.degrees(np.arctan2(se_sin_samples, se_cos_samples))
    else:
        eccentricity_samples = np.zeros_like(rp_samples)
        omega_samples = np.full_like(rp_samples, 90.0)
    conjunction_distance_samples = np.asarray(
        [
            conjunction_distance_a_rs(a_rs, eccentricity_value, omega_value)
            for a_rs, eccentricity_value, omega_value in zip(
                a_rs_samples, eccentricity_samples, omega_samples
            )
        ],
        dtype=float,
    )
    inc_samples = np.degrees(
        np.arccos(np.clip(b_samples / conjunction_distance_samples, 0.0, 1.0))
    )
    area_ppm = (rp_samples**2) * 1e6
    depth_values = []
    for median_rp, median_a, median_b, median_q1, median_q2, median_eccentricity, median_omega in zip(
        _chunk_medians(rp_samples),
        _chunk_medians(a_rs_samples),
        _chunk_medians(b_samples),
        _chunk_medians(q1_samples),
        _chunk_medians(q2_samples),
        _chunk_medians(eccentricity_samples),
        _chunk_medians(omega_samples),
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
            eccentricity=float(median_eccentricity),
            omega_deg=float(median_omega),
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
    posteriors["conjunction_distance_a_rs"] = _quantile_summary(conjunction_distance_samples)
    posteriors["rho_star_solar"] = _quantile_summary(rho_samples)
    posteriors["area_ratio_ppm"] = _quantile_summary(area_ppm)
    posteriors["mid_transit_depth_ppm"] = _quantile_summary(depth_ppm_samples)
    posteriors["u1"] = _quantile_summary(np.asarray(u1_samples))
    posteriors["u2"] = _quantile_summary(np.asarray(u2_samples))
    if eccentric:
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
    rho_prior_log10_sigma: float,
    noise_scale: float,
    eccentric: bool,
    ldtk_prior: Optional[Dict[str, Any]],
    n_sectors: int = 1,
):
    """Create a normalized prior transform for dynesty's likelihood-only API."""
    from scipy.special import ndtr, ndtri

    if rho_prior_solar <= 0 or rho_prior_log10_sigma <= 0 or noise_scale <= 0:
        raise ValueError("stellar density, density uncertainty, and noise scale must be positive")

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
        expected_dimensions = _parameter_count(eccentric, n_sectors)
        if unit_cube.shape != (expected_dimensions,) or not np.all(np.isfinite(unit_cube)):
            raise ValueError("dynesty prior transform received an invalid unit-cube point")
        if np.any(unit_cube < 0.0) or np.any(unit_cube > 1.0):
            raise ValueError("dynesty prior transform requires values in [0, 1]")
        theta = np.empty(expected_dimensions, dtype=float)
        theta[0] = 0.001 + unit_cube[0] * (0.3 - 0.001)
        theta[1] = truncated_normal(
            unit_cube[1], math.log10(rho_prior_solar), rho_prior_log10_sigma, -2.0, 1.5
        )
        theta[2] = unit_cube[2] * 1.2
        baseline_stop = 3 + n_sectors
        theta[3:baseline_stop] = 0.99 + unit_cube[3:baseline_stop] * 0.02
        theta[baseline_stop] = truncated_normal(
            unit_cube[baseline_stop], math.log(noise_scale), 1.0, -12.0, -2.0
        )
        q1_index = baseline_stop + 1
        q2_index = baseline_stop + 2
        if q_transform is None:
            theta[q1_index] = 0.01 + unit_cube[q1_index] * 0.98
            theta[q2_index] = 0.01 + unit_cube[q2_index] * 0.98
        else:
            theta[q1_index], theta[q2_index] = q_transform(
                unit_cube[q1_index], unit_cube[q2_index]
            )
        if eccentric:
            radius = math.sqrt(unit_cube[baseline_stop + 3])
            angle = 2.0 * math.pi * unit_cube[baseline_stop + 4]
            theta[baseline_stop + 3] = radius * math.cos(angle)
            theta[baseline_stop + 4] = radius * math.sin(angle)
        return theta

    return prior_transform


def _synthetic_transit_table(
    ephemeris: Dict[str, Any], rng_seed: int = 5
) -> Dict[str, np.ndarray]:
    """Deterministic test-only transit light curve.

    The injected radius is derived from the ephemeris depth so the synthetic
    signal is self-consistent with the fitter's initialization.
    """
    rng = np.random.default_rng(seed=rng_seed)
    cadence_days = 120.0 / 86400.0
    duration_days = max(12.0, 4.0 * float(ephemeris["period_days"]))
    time = np.arange(0.0, duration_days, cadence_days)
    phase_days = (
        (time - ephemeris["epoch_btjd"] + 0.5 * ephemeris["period_days"])
        % ephemeris["period_days"]
    ) - 0.5 * ephemeris["period_days"]
    injected_rp = math.sqrt(max(float(ephemeris["depth_ppm"]) * 1e-6, 1e-8))
    rho_solar = 1.0
    a_rs = stellar_density_a_rs(rho_solar, ephemeris["period_days"])
    model = batman_transit_flux(
        phase_days,
        ephemeris["period_days"],
        injected_rp,
        a_rs,
        0.3,
        0.35,
        0.3,
        1.0,
        exposure_seconds=EXPTIME_SECONDS,
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


def _mcmc_convergence_diagnostics(
    raw_chain: np.ndarray, tau_values: Optional[np.ndarray], parameter_names: Sequence[str]
) -> Dict[str, Any]:
    """Assess ensemble-chain mixing without claiming independent-chain validation."""
    chain = np.asarray(raw_chain, dtype=float)
    if chain.ndim != 3 or chain.shape[0] < 4 or chain.shape[1] < 2 or not np.all(np.isfinite(chain)):
        return {
            "status": "not-demonstrated",
            "scientific_posterior_eligible": False,
            "reason": "insufficient finite post-burn-in ensemble chain for convergence diagnostics",
        }
    half = chain.shape[0] // 2
    split = np.concatenate((chain[:half], chain[-half:]), axis=1)
    within = np.mean(np.var(split, axis=0, ddof=1), axis=0)
    means = np.mean(split, axis=0)
    between = half * np.var(means, axis=0, ddof=1)
    variance_plus = ((half - 1.0) / half) * within + between / half
    rhat = np.sqrt(np.divide(variance_plus, within, out=np.full_like(within, np.nan), where=within > 0))
    tau = np.asarray(tau_values, dtype=float) if tau_values is not None else np.full(chain.shape[2], np.nan)
    ess = np.divide(
        chain.shape[0] * chain.shape[1], tau, out=np.full(chain.shape[2], np.nan), where=tau > 0
    )
    ratios = np.divide(chain.shape[0], tau, out=np.full(chain.shape[2], np.nan), where=tau > 0)
    diagnostics = {
        "split_r_hat": {parameter_names[index]: float(value) if np.isfinite(value) else None for index, value in enumerate(rhat)},
        "effective_samples": {parameter_names[index]: float(value) if np.isfinite(value) else None for index, value in enumerate(ess)},
        "chain_length_over_tau": {parameter_names[index]: float(value) if np.isfinite(value) else None for index, value in enumerate(ratios)},
        "thresholds": {"split_r_hat_max": 1.01, "effective_samples_min": 400.0, "chain_length_over_tau_min": 50.0},
    }
    basic_pass = bool(np.all(rhat <= 1.01) and np.all(ess >= 400.0) and np.all(ratios >= 50.0))
    diagnostics.update({
        "basic_mixing_passed": basic_pass,
        "status": "basic-ensemble-mixing-passed" if basic_pass else "not-demonstrated",
        "scientific_posterior_eligible": False,
        "reason": "ensemble-walker diagnostics are not a substitute for independently initialized chains or a calibrated correlated-noise likelihood",
    })
    return diagnostics


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
    signal = validate_signal_suffix(signal)
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
    if ephemeris["source"] == "synthetic-demo" or any(
        value == "synthetic-demo" for value in ephemeris.get("field_sources", {}).values()
    ):
        raise RuntimeError("transit fitting requires a complete candidate-derived transit ephemeris")
    stellar = load_stellar_parameters(workspace)
    if stellar["source"] != "candidate-data":
        raise RuntimeError("transit fitting requires complete candidate-derived stellar parameters")
    density_prior = _stellar_density_prior(stellar)
    rho_prior_solar = density_prior["rho_solar"]
    rho_prior_log10_sigma = density_prior["log10_sigma"]
    ldtk_prior = _load_ldtk_prior(workspace) if use_ldtk_prior else None

    table = load_light_curve_table(workspace, max_points=None, require_raw_provenance=True)
    if table is None:
        raise RuntimeError("transit fitting requires observed candidate photometry")
    source = "candidate-data"

    try:
        (
            phase_days,
            native_flux,
            native_error,
            sector_index,
            sector_labels,
            exposure_seconds_by_sector,
        ) = _native_transit_window_data(
            table, ephemeris
        )
    except ValueError as exc:
        raise RuntimeError("candidate photometry cannot support a transit fit") from exc

    depth_ppm = float(ephemeris["depth_ppm"])
    n_sectors = len(sector_labels)
    scatter = float(np.std(native_flux - np.median(native_flux)))
    start = _initial_fit_parameters(
        depth_ppm, rho_prior_solar, scatter, eccentric, n_sectors=n_sectors
    )
    map_point = _map_optimize(
        phase_days,
        native_flux,
        native_error,
        ephemeris,
        rho_prior_solar,
        rho_prior_log10_sigma,
        eccentric,
        start,
        ldtk_prior,
        sector_index=sector_index,
        exposure_seconds_by_sector=exposure_seconds_by_sector,
        n_sectors=n_sectors,
    )

    ndim = int(map_point.size)
    if n_walkers is None:
        n_walkers = max(2 * ndim, min(48, n_samples // 20))
    n_walkers = max(n_walkers, 2 * ndim)
    if burn_in is None:
        burn_in = max(50, n_samples // 5)
    rng = np.random.default_rng(seed=seed)

    def valid_walker_start(candidate_theta: np.ndarray) -> bool:
        return math.isfinite(
            _neg_log_posterior(
                candidate_theta,
                phase_days,
                native_flux,
                native_error,
                ephemeris,
                rho_prior_solar,
                rho_prior_log10_sigma,
                eccentric,
                ldtk_prior,
                sector_index=sector_index,
                exposure_seconds_by_sector=exposure_seconds_by_sector,
                n_sectors=n_sectors,
            )
        )

    p0 = np.empty((n_walkers, ndim), dtype=float)
    for walker_index in range(n_walkers):
        accepted_start = None
        for center in (map_point, start):
            for _attempt in range(100):
                candidate_theta = center + 1e-3 * rng.normal(size=ndim)
                candidate_theta[2] = np.clip(candidate_theta[2], 0.0, 1.1)
                baseline_stop = 3 + n_sectors
                candidate_theta[3:baseline_stop] = np.clip(
                    candidate_theta[3:baseline_stop], 0.995, 1.005
                )
                candidate_theta[baseline_stop + 1] = np.clip(
                    candidate_theta[baseline_stop + 1], 0.01, 0.99
                )
                candidate_theta[baseline_stop + 2] = np.clip(
                    candidate_theta[baseline_stop + 2], 0.01, 0.99
                )
                if eccentric:
                    eccentric_radius = math.hypot(
                        candidate_theta[baseline_stop + 3], candidate_theta[baseline_stop + 4]
                    )
                    if eccentric_radius >= 0.99:
                        candidate_theta[baseline_stop + 3:baseline_stop + 5] *= 0.99 / eccentric_radius
                if valid_walker_start(candidate_theta):
                    accepted_start = candidate_theta
                    break
            if accepted_start is not None:
                break
        if accepted_start is None:
            raise RuntimeError("could not initialize a physically valid transit-fit walker")
        p0[walker_index] = accepted_start

    # Reproducibility: walker positions and emcee's StretchMove RNG are both
    # explicitly seeded without mutating NumPy's process-global RNG.
    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        lambda x: -_neg_log_posterior(
            x,
            phase_days,
            native_flux,
            native_error,
            ephemeris,
            rho_prior_solar,
            rho_prior_log10_sigma,
            eccentric,
            ldtk_prior,
            sector_index=sector_index,
            exposure_seconds_by_sector=exposure_seconds_by_sector,
            n_sectors=n_sectors,
        ),
        moves=emcee.moves.StretchMove(a=1.5),
    )
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(p0, burn_in + n_samples, progress=False)
    raw_chain = sampler.get_chain(discard=burn_in, flat=False)
    chain = raw_chain.reshape((-1, raw_chain.shape[-1]))

    names = _parameter_names(eccentric, n_sectors, sector_labels)
    posteriors = _posterior_summaries(
        chain, ephemeris, eccentric, n_sectors=n_sectors, sector_labels=sector_labels
    )

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
        convergence = _mcmc_convergence_diagnostics(raw_chain, np.asarray(tau_values), names)
    except Exception as exc:
        tau_dict = {"_error": "{0}: {1}".format(type(exc).__name__, exc)}
        convergence = _mcmc_convergence_diagnostics(raw_chain, None, names)

    payload = {
        "schema_version": "1.0",
        "work_package": "MCMC_TRANSIT_FIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "scientific_status": "exploratory-native-cadence-inference",
        "validation_eligible": False,
        "validation_reason": (
            "This likelihood has per-sector flux normalizations but no "
            "calibrated correlated-noise model or independent-chain analysis."
        ),
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
        "density_prior": {
            **density_prior,
            "propagation": "first-order independent symmetric mass and radius uncertainties",
        },
        "limb_darkening_prior": (
            {"source": "ldtk", "path": ldtk_prior["path"]} if ldtk_prior is not None else None
        ),
        "posterior": posteriors,
        "mcmc": {
            "walkers": int(n_walkers),
            "burn_in": int(burn_in),
            "production": int(n_samples),
            "flat_samples": int(chain.shape[0]),
            "random_seed": int(seed),
            "random_generator": "numpy.random.RandomState (MT19937)",
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "autocorrelation_times": tau_dict,
            "convergence": convergence,
        },
        "likelihood": {
            "cadence": "native",
            "n_points": int(phase_days.size),
            "sector_labels": sector_labels,
            "exposure_seconds_by_sector": {
                str(label): float(exposure_seconds_by_sector[index])
                for index, label in enumerate(sector_labels)
            },
            "flux_err_sources": list(table.get("flux_err_sources", [])),
        },
        "signal": signal,
        "caveat": (
            "Exploratory native-cadence fit with independent Gaussian residuals; "
            "not an adopted posterior or validation claim."
        ),
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
    signal = validate_signal_suffix(signal)
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
    if ephemeris["source"] == "synthetic-demo" or any(
        value == "synthetic-demo" for value in ephemeris.get("field_sources", {}).values()
    ):
        raise RuntimeError("transit fitting requires a complete candidate-derived transit ephemeris")
    stellar = load_stellar_parameters(workspace)
    if stellar["source"] != "candidate-data":
        raise RuntimeError("transit fitting requires complete candidate-derived stellar parameters")
    density_prior = _stellar_density_prior(stellar)
    rho_prior_solar = density_prior["rho_solar"]
    rho_prior_log10_sigma = density_prior["log10_sigma"]
    ldtk_prior = _load_ldtk_prior(workspace) if use_ldtk_prior else None
    table = load_light_curve_table(workspace, max_points=None, require_raw_provenance=True)
    if table is None:
        raise RuntimeError("transit fitting requires observed candidate photometry")
    source = "candidate-data"
    try:
        (
            phase_days,
            native_flux,
            native_error,
            sector_index,
            sector_labels,
            exposure_seconds_by_sector,
        ) = _native_transit_window_data(
            table, ephemeris
        )
    except ValueError as exc:
        raise RuntimeError("candidate photometry cannot support a transit fit") from exc

    n_sectors = len(sector_labels)
    noise_scale = float(np.median(native_error))
    prior_transform = _make_dynesty_prior_transform(
        rho_prior_solar,
        rho_prior_log10_sigma,
        noise_scale,
        eccentric,
        ldtk_prior,
        n_sectors=n_sectors,
    )
    ndim = _parameter_count(eccentric, n_sectors)
    initial_live_points = max(2 * ndim + 1, min(500, max(50, n_samples // 10)))
    max_likelihood_calls = max(n_samples, initial_live_points)
    nested_sampler = dynesty.DynamicNestedSampler(
        lambda theta: _log_likelihood(
            theta,
            phase_days,
            native_flux,
            native_error,
            ephemeris,
            eccentric,
            sector_index=sector_index,
            exposure_seconds_by_sector=exposure_seconds_by_sector,
            n_sectors=n_sectors,
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
    posteriors = _posterior_summaries(
        chain, ephemeris, eccentric, n_sectors=n_sectors, sector_labels=sector_labels
    )
    sampling_efficiency = float(getattr(results, "eff", np.nan))
    if not math.isfinite(sampling_efficiency):
        sampling_efficiency = None

    payload = {
        "schema_version": "1.0",
        "work_package": "NESTED_TRANSIT_FIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "scientific_status": "exploratory-native-cadence-inference",
        "validation_eligible": False,
        "validation_reason": (
            "Nested sampling does not make this an adopted posterior without a "
            "calibrated correlated-noise model and independent reproducibility checks."
        ),
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
        "density_prior": {
            **density_prior,
            "propagation": "first-order independent symmetric mass and radius uncertainties",
        },
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
        "likelihood": {
            "cadence": "native",
            "n_points": int(phase_days.size),
            "sector_labels": sector_labels,
            "exposure_seconds_by_sector": {
                str(label): float(exposure_seconds_by_sector[index])
                for index, label in enumerate(sector_labels)
            },
            "flux_err_sources": list(table.get("flux_err_sources", [])),
        },
        "signal": signal,
        "caveat": (
            "Exploratory native-cadence fit with independent Gaussian residuals; "
            "nested evidence is not a validation probability or an adopted posterior."
        ),
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
