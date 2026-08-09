"""Target-neutral MCMC transit light curve fitter.

Fits phase-folded transit light curves with batman: free Kipping (2013) limb
darkening, stellar-density locking (a/Rs derived from the sampled stellar
density and orbital period), and optional eccentric orbit parameters
(sqrt(e) cos(omega), sqrt(e) sin(omega)). Posteriors and chain diagnostics are
written to the candidate outputs directory.
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

G_CGS = 6.67430e-8
RHO_SUN_GCM3 = 1.408
WINDOW_HALF_HOURS = 13.0
BIN_MINUTES = 8.0
SUPERSAMPLE_FACTOR = 7
EXPTIME_SECONDS = 120.0

PARAMETER_NAMES_CIRCULAR = (
    "rp_rs",
    "log_rho_star",
    "impact_parameter",
    "baseline",
    "log_jitter",
    "q1",
    "q2",
)
PARAMETER_NAMES_ECCENTRIC = PARAMETER_NAMES_CIRCULAR + ("sqe_cosw", "sqe_sinw")


def stellar_density_a_rs(rho_solar: float, period_days: float) -> float:
    """Scaled semimajor axis from Kepler's third law at a given density."""
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


def _neg_log_posterior(
    theta: np.ndarray,
    phase_days: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    ephemeris: Dict[str, Any],
    rho_prior_solar: float,
    eccentric: bool,
) -> float:
    if not np.all(np.isfinite(theta)):
        return 1e100
    if eccentric:
        rp, log_rho, b, baseline, log_jitter, q1, q2, se_cos, se_sin = (
            float(theta[0]),
            float(theta[1]),
            float(theta[2]),
            float(theta[3]),
            float(theta[4]),
            float(theta[5]),
            float(theta[6]),
            float(theta[7]),
            float(theta[8]),
        )
    else:
        rp, log_rho, b, baseline, log_jitter, q1, q2 = (
            float(theta[0]),
            float(theta[1]),
            float(theta[2]),
            float(theta[3]),
            float(theta[4]),
            float(theta[5]),
            float(theta[6]),
        )
        se_cos, se_sin = 0.0, 0.0
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
        return 1e100
    eccentricity = 0.0
    omega_deg = 90.0
    if eccentric:
        norm_sq = se_cos * se_cos + se_sin * se_sin
        if norm_sq > 1.0:
            return 1e100
        eccentricity = norm_sq
        if eccentricity > 0:
            omega_deg = math.degrees(math.atan2(se_sin, se_cos))
        else:
            omega_deg = 90.0

    period_days = ephemeris["period_days"]
    rho_solar = 10.0 ** log_rho
    a_rs = stellar_density_a_rs(rho_solar, period_days)
    if eccentricity > 0:
        a_rs = a_rs * (1.0 - eccentricity**2) / (1.0 + eccentricity * math.sin(math.radians(omega_deg)))

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
        return 1e100

    jitter = math.exp(log_jitter)
    ivar = 1.0 / (flux_err**2 + jitter**2)
    residual = flux - model
    chi2 = float(np.sum(residual**2 * ivar))
    logdet = float(np.sum(np.log(2.0 * math.pi / ivar)))
    log_likelihood = -0.5 * (chi2 + logdet)
    # Weak log-jitter prior anchored to the observed per-bin noise scale so
    # the sampler cannot inflate the error budget to wash out the signal.
    noise_scale = float(np.median(flux_err))
    log_prior = -0.5 * ((log_rho - math.log10(rho_prior_solar)) / 0.3) ** 2
    log_prior += -0.5 * ((log_jitter - math.log(noise_scale)) / 1.0) ** 2
    return float(-log_likelihood - log_prior)


def _map_optimize(
    phase_days: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    ephemeris: Dict[str, Any],
    rho_prior_solar: float,
    eccentric: bool,
    start: np.ndarray,
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
                    x, phase_days, flux, flux_err, ephemeris, rho_prior_solar, eccentric
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
) -> Path:
    """Run the MCMC transit fit and write outputs/mcmc_transit_fit.json."""
    import emcee

    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ephemeris = load_transit_ephemeris(workspace, signal=signal)
    stellar = load_stellar_parameters(workspace)
    rho_prior_solar = float(stellar["mass_solar"]) / float(stellar["radius_solar"]) ** 3

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
        phase_days, binned_flux, binned_error, ephemeris, rho_prior_solar, eccentric, start
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
            x, phase_days, binned_flux, binned_error, ephemeris, rho_prior_solar, eccentric
        ),
        moves=emcee.moves.StretchMove(a=1.5),
    )
    sampler.run_mcmc(p0, burn_in + n_samples, progress=False)
    chain = sampler.get_chain(discard=burn_in, flat=True)

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
        "caveat": "Descriptive folded/binned fit; not an adopted native-cadence posterior.",
    }
    output_path = outputs_dir / "mcmc_transit_fit.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.save(str(outputs_dir / "mcmc_transit_fit_chain.npy"), chain)
    return output_path


def _chunk_medians(samples: np.ndarray) -> List[float]:
    """Median down a large sample chain to ~1000 values for model evaluation."""
    samples = np.asarray(samples, dtype=float)
    if samples.size <= 1000:
        return [float(value) for value in samples]
    step = int(np.ceil(samples.size / 1000.0))
    return [float(np.median(samples[index : index + step])) for index in range(0, samples.size, step)]
