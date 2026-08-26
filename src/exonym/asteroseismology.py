"""Target-neutral asteroseismology engine.

Estimates solar-like p-mode stellar oscillation envelope parameters:
- Frequency of maximum oscillation power (nu_max)
- Large frequency separation (Delta_nu)

From high-cadence light curves via background-whitened Lomb-Scargle power spectral
densities (PSD), and derives fundamental stellar properties (M_star, R_star, rho_star, log_g)
using canonical asteroseismic scaling relations (Kjeldsen & Bedding 1995, Huber et al. 2011,
Chaplin et al. 2014).

Scaling Relations:
    (R / R_sun) = (nu_max / nu_max_sun) * (Delta_nu / Delta_nu_sun)^-2 * (Teff / Teff_sun)^(1/2)
    (M / M_sun) = (nu_max / nu_max_sun)^3 * (Delta_nu / Delta_nu_sun)^-4 * (Teff / Teff_sun)^(3/2)
    (g / g_sun) = (nu_max / nu_max_sun) * (Teff / Teff_sun)^(1/2)
    (rho / rho_sun) = (Delta_nu / Delta_nu_sun)^2

Optionally cross-checks with pySYD when installed. Contains no target identifiers
or hardcoded candidate constants.

Scientific Boundary:
    The output is an exploratory scaling diagnostic.  It is not mode
    identification, a calibrated stellar inference, planet validation, or a
    lifecycle decision.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K as TEFF_SUN_K,
    SECONDS_PER_DAY,
)
from .inputs import (
    load_light_curve_table,
    load_stellar_parameters,
    load_transit_ephemeris,
)
from .lightcurve import phase_hours
from .workspace import CandidateWorkspace

# Canonical Solar Asteroseismic Reference Values (Huber et al. 2011, Chaplin et al. 2014)
NUMAX_SUN_UHZ = 3090.0      # Solar frequency of maximum oscillation power (microHz)
DNU_SUN_UHZ = 135.1         # Solar large frequency separation (microHz)

PSD_MIN_UHZ = 100.0         # Default minimum frequency for stellar PSD search (microHz)
PSD_MAX_UHZ = 2000.0        # Default maximum frequency for stellar PSD search (microHz)
DNU_MIN_UHZ = 30.0          # Minimum trial Delta-nu lag (microHz)
DNU_MAX_UHZ = 200.0         # Maximum trial Delta-nu lag (microHz) — Solar Δν☉ ≈ 135.1 µHz; must exceed it — Chaplin & Miglio 2013
# 1 microhertz is exactly 0.0864 cycles per day.  The prior name reversed
# this conversion direction even though callers used its numeric value correctly.
CPD_PER_UHZ = 0.0864


def _odd_bins(value: float) -> int:
    bins = max(3, int(round(value)))
    return bins if bins % 2 else bins + 1


def compute_power_spectrum(
    time: Sequence[float],
    flux: Sequence[float],
    frequency_min_uhz: float = PSD_MIN_UHZ,
    frequency_max_uhz: float = PSD_MAX_UHZ,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a background-whitened Lomb-Scargle power spectrum.

    Mathematical Formulation:
        The native PSD uses Astropy's ``normalization="psd"``.  A smoothed
        local background ``B(nu)`` defines dimensionless whitened power
        ``W(nu) = P(nu) / B(nu)``; a Gaussian-smoothed ``W`` is the envelope
        statistic used by :func:`estimate_oscillation_envelope`.

    Args:
        time (Sequence[float]): Cadence times in a consistent day-based unit;
            non-finite paired cadences are removed.
        flux (Sequence[float]): Flux samples paired with ``time``.  The mean is
            removed before PSD estimation.
        frequency_min_uhz (float): Lower frequency bound in microhertz.
        frequency_max_uhz (float): Upper frequency bound in microhertz.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Frequency in
        microhertz, Astropy PSD power, dimensionless whitened power, and the
        dimensionless smoothed envelope, all aligned on the frequency grid.

    Raises:
        ValueError: Fewer than the required finite cadence pairs remain, or
            the underlying periodogram cannot use the requested frequency range.

    Note:
        Window functions, granulation, instrumental systematics, and finite
        cadence can produce misleading envelope structure.
    """
    from astropy.timeseries import LombScargle
    from scipy.ndimage import gaussian_filter1d, median_filter

    time_arr = np.asarray(time, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    finite = np.isfinite(time_arr) & np.isfinite(flux_arr)
    time_arr = time_arr[finite]
    flux_arr = flux_arr[finite]
    if time_arr.size < 50:
        raise ValueError("insufficient data for power spectrum estimation")

    frequency_day, power = LombScargle(
        time_arr,
        flux_arr - np.nanmean(flux_arr),
        normalization="psd",
    ).autopower(
        minimum_frequency=frequency_min_uhz * CPD_PER_UHZ,
        maximum_frequency=frequency_max_uhz * CPD_PER_UHZ,
        samples_per_peak=1,
        method="fast",
    )
    frequency_uhz = np.asarray(frequency_day) / CPD_PER_UHZ
    power = np.asarray(power, dtype=float)
    spacing = float(np.nanmedian(np.diff(frequency_uhz)))
    background_bins = _odd_bins(100.0 / max(spacing, 1e-6))
    background = median_filter(power, size=background_bins, mode="nearest")
    background = np.maximum(background, np.finfo(float).tiny)
    whitened = power / background
    smooth_sigma = max(1.0, 20.0 / max(spacing, 1e-6))
    envelope = gaussian_filter1d(whitened, smooth_sigma, mode="nearest")
    return frequency_uhz, power, whitened, envelope


def fit_harvey_granulation_background(
    frequency_uhz: np.ndarray,
    power_psd: np.ndarray,
    numax_guess: Optional[float] = None,
) -> Dict[str, Any]:
    """Fit a multi-component Harvey convective background model with Gaussian oscillation envelope.

    Mathematical Formulation:
        The total stellar power spectrum P(nu) is modeled as:
            P_model(nu) = P_bg(nu) + P_osc(nu)
        where the convective granulation background is:
            P_bg(nu) = W + A_1 / (1 + (2*pi*1e-6 * tau_1 * nu)^2) + A_2 / (1 + (2*pi*1e-6 * tau_2 * nu)^4)
        with c_1 = 2 (mesogranulation/activity) and c_2 = 4 (granulation), tau in seconds, nu in uHz.
        The p-mode oscillation envelope is:
            P_osc(nu) = H_osc * exp( - (nu - nu_max)^2 / (2 * sigma_env^2) )

    Args:
        frequency_uhz (np.ndarray): Frequency array in microhertz.
        power_psd (np.ndarray): Power spectral density array.
        numax_guess (float, optional): Initial guess for nu_max in microhertz.

    Returns:
        Dict[str, Any]: Fitted model parameters, background PSD, whitened PSD, and fit metrics.
    """
    from scipy.optimize import least_squares

    freq = np.asarray(frequency_uhz, dtype=float)
    power = np.asarray(power_psd, dtype=float)
    finite = np.isfinite(freq) & np.isfinite(power) & (freq > 0.0) & (power > 0.0)
    freq = freq[finite]
    power = power[finite]

    if freq.size < 20:
        raise ValueError("insufficient finite power spectrum data for background fitting")

    # Initial parameter estimates
    high_freq_cutoff = float(np.percentile(freq, 85))
    high_freq_mask = freq >= high_freq_cutoff
    w_init = float(np.median(power[high_freq_mask])) if np.any(high_freq_mask) else float(np.min(power))
    w_init = max(1e-12, w_init)

    low_freq_cutoff = min(100.0, float(np.percentile(freq, 20)))
    low_freq_mask = freq <= low_freq_cutoff
    a1_init = float(np.max(power[low_freq_mask]) - w_init) if np.any(low_freq_mask) else float(np.max(power))
    a1_init = max(1e-6, a1_init)
    tau1_init = 20000.0  # seconds (~5.5 hours)

    mid_freq_cutoff = min(500.0, float(np.percentile(freq, 60)))
    mid_freq_mask = (freq > 50.0) & (freq <= mid_freq_cutoff)
    a2_init = float(np.median(power[mid_freq_mask]) - w_init) if np.any(mid_freq_mask) else float(a1_init * 0.1)
    a2_init = max(1e-6, a2_init)
    tau2_init = 600.0  # seconds (10 minutes)

    if numax_guess is not None and math.isfinite(float(numax_guess)):
        numax_init = float(np.clip(numax_guess, float(np.min(freq)), float(np.max(freq))))
    else:
        numax_init = float(np.median(freq))
    h_osc_init = float(np.max(power) * 0.5)
    sigma_env_init = max(5.0, 0.25 * numax_init)

    # Initial parameter vector: [W, A1, tau1, A2, tau2, H_osc, nu_max, sigma_env]
    p0 = np.array([w_init, a1_init, tau1_init, a2_init, tau2_init, h_osc_init, numax_init, sigma_env_init], dtype=float)
    lower_bounds = np.array([0.0, 0.0, 10.0, 0.0, 1.0, 0.0, float(np.min(freq)), 1.0], dtype=float)
    upper_bounds = np.array([
        float(np.max(power) * 10.0),
        float(np.max(power) * 100.0),
        1e7,
        float(np.max(power) * 100.0),
        1e6,
        float(np.max(power) * 50.0),
        float(np.max(freq)),
        float(np.max(freq) - np.min(freq)),
    ], dtype=float)

    def _model(params, nu):
        w, a1, t1, a2, t2, h_osc, nu_max, sig_env = params
        scale1 = 2.0 * math.pi * 1e-6 * t1 * nu
        scale2 = 2.0 * math.pi * 1e-6 * t2 * nu
        p_bg = w + a1 / (1.0 + scale1**2) + a2 / (1.0 + scale2**4)
        p_osc = h_osc * np.exp(-0.5 * ((nu - nu_max) / max(sig_env, 1e-6))**2)
        return p_bg, p_osc

    log_power = np.log(power)

    def _residuals(params):
        p_bg, p_osc = _model(params, freq)
        p_total = np.maximum(p_bg + p_osc, 1e-30)
        return np.log(p_total) - log_power

    try:
        opt = least_squares(
            _residuals,
            p0,
            bounds=(lower_bounds, upper_bounds),
            loss="soft_l1",
            f_scale=0.5,
            max_nfev=2000,
        )
        p_opt = opt.x
        status = "converged" if opt.success else "optimization_suboptimal"
    except Exception:
        p_opt = p0
        status = "optimization_fallback"

    w_fit, a1_fit, t1_fit, a2_fit, t2_fit, h_osc_fit, numax_fit, sig_env_fit = p_opt
    bg_power, osc_power = _model(p_opt, freq)
    total_power = bg_power + osc_power
    whitened = power / np.maximum(bg_power, 1e-30)

    # Statistical goodness-of-fit
    residuals = power - total_power
    chi2 = float(np.sum((residuals / np.maximum(total_power, 1e-30))**2))
    k = 8
    n = int(freq.size)
    dof = max(1, n - k)
    chi2_red = float(chi2 / dof)
    aic = float(2.0 * k + n * math.log(max(1e-12, float(np.sum(residuals**2) / n))))
    bic = float(k * math.log(n) + n * math.log(max(1e-12, float(np.sum(residuals**2) / n))))

    return {
        "status": status,
        "white_noise_floor_w": float(w_fit),
        "amplitude_a1": float(a1_fit),
        "timescale_tau1_seconds": float(t1_fit),
        "amplitude_a2": float(a2_fit),
        "timescale_tau2_seconds": float(t2_fit),
        "envelope_amplitude_h_osc": float(h_osc_fit),
        "numax_uhz": float(numax_fit),
        "envelope_sigma_uhz": float(sig_env_fit),
        "chi2": chi2,
        "reduced_chi2": chi2_red,
        "aic": aic,
        "bic": bic,
        "frequency_uhz": [float(f) for f in freq],
        "background_power": [float(b) for b in bg_power],
        "whitened_power": [float(w) for w in whitened],
    }


def spacing_correlation(
    frequency_uhz: np.ndarray,
    whitened: np.ndarray,
    numax_uhz: float,
    dnu_min_uhz: float = DNU_MIN_UHZ,
    dnu_max_uhz: float = DNU_MAX_UHZ,
) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
    """Correlate a local whitened PSD with shifted copies to estimate ``Delta_nu``.

    Mathematical Formulation:
        For each trial lag, the function interpolates ``W(nu + Delta_nu)`` on
        the local frequency grid and returns its normalized dot product with
        ``W(nu) - 1``.  The best finite correlation selects the reported lag.

    Args:
        frequency_uhz (np.ndarray): PSD frequency grid in microhertz.
        whitened (np.ndarray): Dimensionless PSD divided by its background.
        numax_uhz (float): Envelope-peak frequency in microhertz.
        dnu_min_uhz (float): Lower trial large-separation lag in microhertz.
        dnu_max_uhz (float): Upper trial large-separation lag in microhertz.

    Returns:
        Tuple[Optional[float], Optional[float], Optional[np.ndarray]]: Best
        lag in microhertz, its dimensionless correlation, and the trial grid.
        Unsupported local coverage returns ``None`` estimates rather than a
        fabricated separation.
    """
    envelope_half_width = 0.66 * numax_uhz**0.88
    use = np.abs(frequency_uhz - numax_uhz) <= envelope_half_width
    local_frequency = frequency_uhz[use]
    local = whitened[use] - 1.0
    if local.size < 20:
        return None, None, None
    lags = np.linspace(dnu_min_uhz, dnu_max_uhz, 801)
    scores = np.empty_like(lags)
    for index, lag in enumerate(lags):
        shifted = np.interp(
            local_frequency + lag,
            local_frequency,
            local,
            left=np.nan,
            right=np.nan,
        )
        valid = np.isfinite(shifted)
        if valid.sum() < 10:
            scores[index] = np.nan
            continue
        x = local[valid]
        y = shifted[valid]
        denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
        scores[index] = np.sum(x * y) / denominator if denominator else np.nan
    if not np.any(np.isfinite(scores)):
        return None, None, lags
    best = int(np.nanargmax(scores))
    return float(lags[best]), float(scores[best]), lags


def estimate_oscillation_envelope(
    time: Sequence[float],
    flux: Sequence[float],
    numax_min_uhz: float,
    numax_max_uhz: float,
) -> Dict[str, Any]:
    """Estimate an oscillation envelope and local large-separation diagnostic.

    Args:
        time (Sequence[float]): Candidate light-curve cadence times.
        flux (Sequence[float]): Paired candidate flux samples.
        numax_min_uhz (float): Requested lower envelope-search bound in
            microhertz.
        numax_max_uhz (float): Requested upper envelope-search bound in
            microhertz.

    Returns:
        Dict[str, Any]: Candidate envelope metadata including requested and
        effective bounds, clipping flags, PSD resolution, ``nu_max`` estimate,
        and a nullable ``Delta_nu`` correlation result.

    Raises:
        ValueError: Requested bounds are non-finite or collapse after clipping,
            or there are insufficient usable cadences for the PSD.

    Note:
        Effective bounds are recorded separately so review can distinguish the
        requested astrophysical range from the supported native PSD range.
    """
    import warnings

    numax_min_requested = float(numax_min_uhz)
    numax_max_requested = float(numax_max_uhz)
    if not math.isfinite(numax_min_requested) or not math.isfinite(numax_max_requested):
        raise ValueError("numax search bounds must be finite")
    numax_min_used = numax_min_requested
    numax_max_used = numax_max_requested
    # NUMERICAL_GUARD: Keep the envelope search within the PSD support instead
    # of extrapolating a frequency grid beyond its declared native range.
    numax_min_clipped = numax_min_requested < PSD_MIN_UHZ
    numax_max_clipped = numax_max_requested > PSD_MAX_UHZ
    if numax_min_clipped:
        warnings.warn(
            "numax_min_uhz {0:.1f} uHz clamped to search floor {1:.1f} uHz".format(
                numax_min_used, PSD_MIN_UHZ
            )
        )
        numax_min_used = PSD_MIN_UHZ
    if numax_max_clipped:
        warnings.warn(
            "numax_max_uhz {0:.1f} uHz clamped to search ceiling {1:.1f} uHz".format(
                numax_max_used, PSD_MAX_UHZ
            )
        )
        numax_max_used = PSD_MAX_UHZ
    search_low = numax_min_used
    search_high = numax_max_used
    if search_high <= search_low:
        raise ValueError("invalid numax search bounds")
    frequency, power, whitened, envelope = compute_power_spectrum(
        time, flux, search_low, search_high
    )
    search = (frequency >= search_low) & (frequency <= search_high)
    peak_index = int(np.flatnonzero(search)[int(np.nanargmax(envelope[search]))])
    numax_candidate = float(frequency[peak_index])
    dnu_candidate, dnu_correlation, _ = spacing_correlation(
        frequency, whitened, numax_candidate
    )
    try:
        granulation_background = fit_harvey_granulation_background(
            frequency, power, numax_guess=numax_candidate
        )
    except Exception:
        granulation_background = None
    return {
        "n_points": int(len(time)),
        "baseline_days": float(np.max(time) - np.min(time)),
        "rayleigh_uhz": float(
            1e6 / ((np.max(time) - np.min(time)) * SECONDS_PER_DAY)
        ),
        "numax_min_requested_uhz": numax_min_requested,
        "numax_max_requested_uhz": numax_max_requested,
        "numax_min_used": numax_min_used,
        "numax_max_used": numax_max_used,
        "numax_min_clipped": numax_min_clipped,
        "numax_max_clipped": numax_max_clipped,
        "numax_candidate_uhz": numax_candidate,
        "envelope_peak_ratio": float(envelope[peak_index]),
        "dnu_candidate_uhz": dnu_candidate,
        "dnu_correlation": dnu_correlation,
        "granulation_background_model": granulation_background,
    }


def seismic_mass_radius(
    numax_uhz: float,
    dnu_uhz: Optional[float],
    teff_k: float,
    mass_prior_solar: Optional[float] = None,
    radius_prior_solar: Optional[float] = None,
    dnu_correction_factor: float = 1.0,
) -> Dict[str, Any]:
    """Derive asteroseismic stellar mass and radius from scaling relations.

    Mathematical Formulation:
        The full solution combines ``nu_max / nu_max_sun = (M / M_sun)
        (R / R_sun)**-2 (Teff / Teff_sun)**-0.5`` with
        ``Delta_nu / Delta_nu_sun = (rho / rho_sun)**0.5``.  If an observable
        is unavailable, a supplied mass or radius prior closes the reduced
        system rather than supplying a new measurement.

    Args:
        numax_uhz (float): Envelope maximum in microhertz.
        dnu_uhz (Optional[float]): Large separation in microhertz, or ``None``
            when no finite spacing correlation is available.
        teff_k (float): Effective temperature in kelvin.
        mass_prior_solar (Optional[float]): Solar-unit mass used only when the
            available observables do not independently determine it.
        radius_prior_solar (Optional[float]): Solar-unit radius used only when
            the available observables do not independently determine it.
        dnu_correction_factor (float): Dimensionless multiplier applied to the
            measured large separation before scaling.

    Returns:
        Dict[str, Any]: Rounded mass and radius in solar units plus a method
        label that states whether the result used both observables or a prior.

    Note:
        Solar scaling relations can carry systematic error, particularly where
        a model-dependent large-separation correction is needed.  This helper
        is a descriptive estimate, not a calibrated stellar solution.

    Uses the classic relations
        nu_max / nu_max_sun = M/M_sun (R/R_sun)^-2 (Teff/Teff_sun)^-1/2
        Delta-nu / Delta-nu_sun = (rho / rho_sun)^1/2
    When one of the two observables is missing, the corresponding stellar
    prior (solar reference by default) closes the system.

    .. note:: Systematic bias in Delta-nu
        The classic Kjeldsen & Bedding (1995) scaling relation for Delta-nu
        carries a known 5–15% systematic offset driven by near-surface effects.
        A caller may apply ``dnu_correction_factor`` only after retaining its
        own candidate-owned calibration evidence. This helper does not select
        a correction grid or infer a factor from stellar parameters.
        ``dnu_correction_factor`` multiplies the raw Lomb-Scargle Delta-nu
         estimate before the ratio is computed (default 1.0 = no correction).
    """
    try:
        numax_value = float(numax_uhz)
        teff_value = float(teff_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("numax_uhz and teff_k must be finite physical values") from exc
    if (
        not math.isfinite(numax_value)
        or numax_value < 0.0
        or not math.isfinite(teff_value)
        or teff_value <= 0.0
    ):
        raise ValueError("numax_uhz must be finite and non-negative; teff_k must be finite and positive")

    dnu_value: Optional[float] = None
    if dnu_uhz is not None:
        try:
            dnu_value = float(dnu_uhz)
        except (TypeError, ValueError) as exc:
            raise ValueError("dnu_uhz must be a finite positive number or None") from exc
        if not math.isfinite(dnu_value) or dnu_value <= 0.0:
            raise ValueError("dnu_uhz must be a finite positive number or None")

    mass_prior = 1.0
    if mass_prior_solar is not None:
        try:
            mass_prior = float(mass_prior_solar)
        except (TypeError, ValueError) as exc:
            raise ValueError("mass_prior_solar must be finite and positive when supplied") from exc
        if not math.isfinite(mass_prior) or mass_prior <= 0.0:
            raise ValueError("mass_prior_solar must be finite and positive when supplied")

    radius_prior = 1.0
    if radius_prior_solar is not None:
        try:
            radius_prior = float(radius_prior_solar)
        except (TypeError, ValueError) as exc:
            raise ValueError("radius_prior_solar must be finite and positive when supplied") from exc
        if not math.isfinite(radius_prior) or radius_prior <= 0.0:
            raise ValueError("radius_prior_solar must be finite and positive when supplied")

    if isinstance(dnu_correction_factor, bool):
        raise ValueError("dnu_correction_factor must be a positive finite number")
    try:
        correction_factor = float(dnu_correction_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError("dnu_correction_factor must be a positive finite number") from exc
    if not math.isfinite(correction_factor) or correction_factor <= 0.0:
        raise ValueError("dnu_correction_factor must be a positive finite number")

    numax_ratio = numax_value / NUMAX_SUN_UHZ
    teff_ratio = teff_value / TEFF_SUN_K
    method = "scaling-relations"
    dnu_corrected: Optional[float] = None
    if dnu_value is not None:
        dnu_corrected = dnu_value * correction_factor
        if not math.isfinite(dnu_corrected) or dnu_corrected <= 0.0:
            raise ValueError("dnu_correction_factor produces an invalid corrected Delta-nu")
        dnu_ratio = dnu_corrected / DNU_SUN_UHZ
        if numax_value > 0.0:
            radius = numax_ratio * math.sqrt(teff_ratio) / (dnu_ratio**2)
            mass = (radius**3) * (dnu_ratio**2)
            method = "full-numax-dnu-scaling"
        else:
            radius = radius_prior
            mass = (radius**3) * (dnu_ratio**2)
            method = "dnu-density-scaling-with-radius-prior"
    elif numax_value > 0.0:
        mass = mass_prior
        radius = math.sqrt(mass / (numax_ratio * math.sqrt(teff_ratio)))
        method = "numax-scaling-with-mass-prior"
    else:
        mass = mass_prior
        radius = radius_prior
        method = "stellar-priors-only"
    if not math.isfinite(mass) or mass <= 0.0 or not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("asteroseismic scaling inputs produce non-finite stellar parameters")
    return {
        "mass_solar": round(float(mass), 4),
        "radius_solar": round(float(radius), 4),
        "method": method,
        "dnu_correction_factor": correction_factor,
        "dnu_corrected_uhz": dnu_corrected,
    }


def _percentile_summary(samples: np.ndarray) -> Dict[str, float]:
    quantiles = np.quantile(np.asarray(samples, dtype=float), [0.16, 0.50, 0.84])
    return {
        "p16": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p84": float(quantiles[2]),
        "plus": float(quantiles[2] - quantiles[1]),
        "minus": float(quantiles[1] - quantiles[0]),
    }


def _rayleigh_resolution_interval(
    center_uhz: float, rayleigh_uhz: float
) -> Tuple[float, float, bool]:
    """Return the physically positive frequency interval for one resolution element.

    A Rayleigh resolution is a finite-baseline bin width, not a Gaussian
    one-sigma error.  A peak selected from that grid is represented by the
    half-bin interval around its reported center.  The lower edge is truncated
    only when that interval would cross the positive-frequency boundary.
    """
    half_width = rayleigh_uhz / 2.0
    lower = center_uhz - half_width
    upper = center_uhz + half_width
    if not math.isfinite(upper) or upper <= 0.0:
        raise ValueError("Rayleigh resolution interval has no positive support")
    lower_truncated_at_zero = lower <= 0.0
    return (
        max(float(np.finfo(float).tiny), lower),
        upper,
        lower_truncated_at_zero,
    )


def seismic_uncertainty_summary(
    envelope: Dict[str, Any],
    stellar: Dict[str, Any],
    draws: int = 2048,
    dnu_correction_factor: float = 1.0,
) -> Dict[str, Any]:
    """Propagate resolution-interval and temperature errors through scaling.

    Args:
        envelope (Dict[str, Any]): Envelope record with finite ``nu_max``,
            ``Delta_nu``, and Rayleigh resolution in microhertz.
        stellar (Dict[str, Any]): Candidate stellar record with effective
            temperature and its uncertainty in kelvin.
        draws (int): Number of deterministic Monte Carlo draws used for the
            reported percentile summaries.
        dnu_correction_factor (float): Positive finite, evidence-backed
            multiplier applied to the raw ``Delta_nu`` draws before the mass
            and radius scaling relations.

    Returns:
        Dict[str, Any]: Status plus percentile summaries in microhertz and
        solar units, or an unavailable status when required uncertainties are
        missing or invalid.

    Note:
        Each frequency is sampled uniformly within its one-Rayleigh-resolution
        element, rather than treating the full resolution as a Gaussian
        one-sigma error. Temperature remains a candidate-supplied Gaussian
        uncertainty. This deliberately excludes systematic scaling-relation
        error and is not a complete stellar posterior.
    """
    try:
        numax = float(envelope["numax_candidate_uhz"])
        dnu = float(envelope["dnu_candidate_uhz"])
        rayleigh = float(envelope["rayleigh_uhz"])
        teff = float(stellar["teff_k"])
        teff_error = float(stellar["teff_k_err"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "unavailable-missing-input-uncertainty",
            "reason": "Requires finite numax, dnu, Rayleigh resolution, teff_k, and teff_k_err.",
        }
    if not all(math.isfinite(value) and value > 0 for value in (numax, dnu, rayleigh, teff, teff_error)):
        return {
            "status": "unavailable-invalid-input-uncertainty",
            "reason": "Frequency resolution and stellar-temperature uncertainty must be positive and finite.",
        }
    if isinstance(dnu_correction_factor, bool):
        return {
            "status": "unavailable-invalid-dnu-correction-factor",
            "reason": "dnu_correction_factor must be a positive finite number.",
        }
    try:
        correction_factor = float(dnu_correction_factor)
    except (TypeError, ValueError):
        return {
            "status": "unavailable-invalid-dnu-correction-factor",
            "reason": "dnu_correction_factor must be a positive finite number.",
        }
    if not math.isfinite(correction_factor) or correction_factor <= 0.0:
        return {
            "status": "unavailable-invalid-dnu-correction-factor",
            "reason": "dnu_correction_factor must be a positive finite number.",
        }
    try:
        numax_lower, numax_upper, numax_lower_truncated = _rayleigh_resolution_interval(
            numax, rayleigh
        )
        dnu_lower, dnu_upper, dnu_lower_truncated = _rayleigh_resolution_interval(
            dnu, rayleigh
        )
    except ValueError as exc:
        return {
            "status": "unavailable-invalid-resolution-interval",
            "reason": str(exc),
        }
    rng = np.random.default_rng(seed=41)
    numax_draws = rng.uniform(numax_lower, numax_upper, draws)
    dnu_draws = rng.uniform(dnu_lower, dnu_upper, draws)
    dnu_corrected_draws = dnu_draws * correction_factor
    if not np.all(np.isfinite(dnu_corrected_draws)) or not np.all(dnu_corrected_draws > 0.0):
        return {
            "status": "unavailable-invalid-dnu-correction-factor",
            "reason": "dnu_correction_factor produces invalid corrected Delta-nu draws.",
        }
    teff_draws = np.clip(rng.normal(teff, teff_error, draws), np.finfo(float).eps, None)
    numax_ratio = numax_draws / NUMAX_SUN_UHZ
    dnu_ratio = dnu_corrected_draws / DNU_SUN_UHZ
    teff_ratio = teff_draws / TEFF_SUN_K
    radius_draws = numax_ratio * np.sqrt(teff_ratio) / dnu_ratio**2
    # ASTROPHYSICAL_NOTE: mass is derived from radius (not sampled independently),
    # so rho = M / R^3 reduces exactly to dnu_ratio^2 with no Teff variance.
    # This preserves the analytic cancellation required by the seismic scaling
    # relations; do not replace this with an independent mass draw.
    mass_draws = radius_draws**3 * dnu_ratio**2
    return {
        "status": "resolution-and-temperature-monte-carlo",
        "draws": int(draws),
        "numax_uhz": _percentile_summary(numax_draws),
        "dnu_uhz": _percentile_summary(dnu_draws),
        "dnu_corrected_uhz": _percentile_summary(dnu_corrected_draws),
        "dnu_correction_factor": correction_factor,
        "mass_solar": _percentile_summary(mass_draws),
        "radius_solar": _percentile_summary(radius_draws),
        "frequency_resolution_sampling": {
            "distribution": "uniform-within-one-Rayleigh-resolution-element",
            "rayleigh_uhz": rayleigh,
            "numax_interval_uhz": [numax_lower, numax_upper],
            "dnu_interval_uhz": [dnu_lower, dnu_upper],
            "numax_lower_bound_truncated_at_zero": numax_lower_truncated,
            "dnu_lower_bound_truncated_at_zero": dnu_lower_truncated,
        },
        "assumptions": (
            "Independent uniform draws within one Rayleigh-resolution element for raw "
            "numax and dnu, a fixed supplied dnu correction factor, and a candidate-supplied "
            "Gaussian teff error; systematic scaling-relation error is excluded."
        ),
    }


# ASTROPHYSICAL_HEURISTIC: Broad plausibility bounds prevent an uncalibrated
# PSD peak from being propagated as an unreviewed stellar solution.
SEISMIC_MASS_BOUNDS_SOLAR = (0.05, 20.0)
SEISMIC_RADIUS_BOUNDS_SOLAR = (0.05, 20.0)
SEISMIC_PRIOR_RATIO_TOLERANCE = 2.0


def seismic_sanity_check(
    seismic: Dict[str, Any],
    radius_prior_solar: Optional[float] = None,
    prior_is_catalog: bool = False,
) -> Dict[str, Any]:
    """Flag scaling-relation results that are physically implausible.

    Args:
        seismic (Dict[str, Any]): Mapping with mass and radius in solar units.
        radius_prior_solar (Optional[float]): Positive solar-unit external
            radius prior used for a consistency check when it is catalog-based.
        prior_is_catalog (bool): Whether the supplied radius prior is eligible
            for the catalog-consistency heuristic.

    Returns:
        Dict[str, Any]: ``plausible`` flag and human-readable rejection reasons.

    Note:
        The bounds and prior-ratio comparison are triage heuristics for noisy
        PSD peaks.  Passing them does not establish mode identification or a
        physically calibrated stellar characterization.

    Scaling relations applied to noise peaks can return absurd stellar
    parameters (e.g., a 26 Msun A star from two 120-s sectors). Results outside
    plausible bounds, or inconsistent with a catalog/SED radius prior by more
    than the tolerance factor, are flagged so the caller can reject them.
    """
    reasons: List[str] = []
    mass = float(seismic.get("mass_solar", 0.0))
    radius = float(seismic.get("radius_solar", 0.0))
    mass_lo, mass_hi = SEISMIC_MASS_BOUNDS_SOLAR
    radius_lo, radius_hi = SEISMIC_RADIUS_BOUNDS_SOLAR
    if not (mass_lo <= mass <= mass_hi):
        reasons.append("mass outside plausible range")
    if not (radius_lo <= radius <= radius_hi):
        reasons.append("radius outside plausible range")
    if prior_is_catalog and radius_prior_solar and radius_prior_solar > 0 and radius > 0:
        ratio = radius / float(radius_prior_solar)
        if not (1.0 / SEISMIC_PRIOR_RATIO_TOLERANCE <= ratio <= SEISMIC_PRIOR_RATIO_TOLERANCE):
            reasons.append("scaling radius inconsistent with catalog radius prior")
    return {"plausible": not reasons, "reasons": reasons}


def _highpass_segments(
    time: np.ndarray,
    flux: np.ndarray,
    cadence_seconds: float,
    window_days: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove slow trends per contiguous segment (Savitzky-Golay)."""
    from scipy.signal import savgol_filter

    order = np.argsort(time)
    time = np.asarray(time, dtype=float)[order]
    flux = np.asarray(flux, dtype=float)[order]
    residual = np.full_like(flux, np.nan)
    gaps = np.flatnonzero(
        np.diff(time) > 5.0 * cadence_seconds / SECONDS_PER_DAY
    ) + 1
    edges = np.r_[0, gaps, len(time)]
    nominal_window = int(round(window_days * SECONDS_PER_DAY / cadence_seconds))
    if nominal_window % 2 == 0:
        nominal_window += 1
    for start, stop in zip(edges[:-1], edges[1:]):
        count = stop - start
        window = min(nominal_window, count if count % 2 else count - 1)
        if window < 11:
            continue
        trend = savgol_filter(flux[start:stop], window, 2, mode="interp")
        good = np.isfinite(trend) & (trend != 0)
        segment = np.full(count, np.nan)
        segment[good] = (flux[start:stop][good] / trend[good] - 1.0) * 1e6
        residual[start:stop] = segment
    finite = np.isfinite(residual)
    return time[finite], residual[finite]


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one candidate-local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stellar_parameters_artifact(workspace: CandidateWorkspace) -> Optional[Dict[str, str]]:
    """Return provenance for the optional candidate-owned stellar input file."""
    path = workspace.path / "data" / "external" / "stellar_params.json"
    if not path.is_file() or path.is_symlink():
        return None
    return {
        "path": path.relative_to(workspace.path).as_posix(),
        "sha256": _sha256(path),
    }


def _nonempty_text(value: Any) -> Optional[str]:
    """Normalize a nonblank evidence field without coercing arbitrary values."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _resolve_dnu_correction(
    stellar: Dict[str, Any],
    dnu_uhz: Optional[float],
    input_artifact: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Resolve an optional Δν correction only from explicit candidate evidence.

    The candidate-owned ``stellar_params.json`` may provide a nested record of
    the form ``{"factor": number, "evidence": {"reference": str,
    "applicability": str}}``.  This function deliberately does not infer a
    correction from temperature, gravity, or a named literature relation: that
    would turn an exploratory scaling diagnostic into an uncalibrated model
    interpolation.
    """
    try:
        raw_dnu = float(dnu_uhz) if dnu_uhz is not None else None
    except (TypeError, ValueError):
        raw_dnu = None
    base: Dict[str, Any] = {
        "factor": 1.0,
        "applied": False,
        "raw_dnu_uhz": raw_dnu if raw_dnu is not None and math.isfinite(raw_dnu) and raw_dnu > 0.0 else None,
        "scaling_dnu_uhz": raw_dnu if raw_dnu is not None and math.isfinite(raw_dnu) and raw_dnu > 0.0 else None,
        "input_artifact": input_artifact,
    }
    if base["raw_dnu_uhz"] is None:
        base.update(
            {
                "status": "unavailable-no-measured-dnu",
                "reason": "No finite positive native Delta-nu measurement is available for scaling.",
            }
        )
        return base

    record = stellar.get("dnu_correction")
    if not isinstance(record, dict):
        base.update(
            {
                "status": "identity-no-evidence-backed-input",
                "reason": "No candidate-owned dnu_correction record supplied a factor and evidence.",
            }
        )
        return base

    factor_value = record.get("factor")
    if isinstance(factor_value, bool) or not isinstance(factor_value, (int, float)):
        base.update(
            {
                "status": "identity-invalid-evidence-record",
                "reason": "dnu_correction.factor must be a positive finite number.",
            }
        )
        return base
    factor = float(factor_value)
    if not math.isfinite(factor) or factor <= 0.0:
        base.update(
            {
                "status": "identity-invalid-evidence-record",
                "reason": "dnu_correction.factor must be a positive finite number.",
            }
        )
        return base

    evidence = record.get("evidence")
    if not isinstance(evidence, dict):
        base.update(
            {
                "status": "identity-invalid-evidence-record",
                "reason": "dnu_correction.evidence must describe the correction reference and applicability.",
            }
        )
        return base
    reference = _nonempty_text(evidence.get("reference"))
    applicability = _nonempty_text(evidence.get("applicability"))
    if reference is None or applicability is None:
        base.update(
            {
                "status": "identity-invalid-evidence-record",
                "reason": "dnu_correction.evidence requires nonblank reference and applicability fields.",
            }
        )
        return base

    scaling_dnu = float(base["raw_dnu_uhz"]) * factor
    if not math.isfinite(scaling_dnu) or scaling_dnu <= 0.0:
        base.update(
            {
                "status": "identity-invalid-evidence-record",
                "reason": "dnu_correction.factor produces an invalid corrected Delta-nu.",
            }
        )
        return base
    base.update(
        {
            "factor": factor,
            "applied": factor != 1.0,
            "scaling_dnu_uhz": scaling_dnu,
            "evidence": {"reference": reference, "applicability": applicability},
            "status": (
                "corrected-evidence-backed-input"
                if factor != 1.0
                else "identity-evidence-backed-input"
            ),
        }
    )
    return base


def _adapter_run_dir(workspace: CandidateWorkspace, engine: str) -> Tuple[str, Path]:
    """Create one unique candidate-local directory for an optional adapter run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f").lower()
    run_id = "{0}-{1}".format(timestamp, engine)
    run_dir = workspace.path / "runs" / engine / run_id
    suffix = 1
    while run_dir.exists():
        run_id = "{0}-{1}-{2}".format(timestamp, engine, suffix)
        run_dir = workspace.path / "runs" / engine / run_id
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_id, run_dir


def _adapter_artifact(workspace: CandidateWorkspace, path: Path, role: Optional[str] = None) -> Dict[str, str]:
    """Build a schema-compatible artifact record for a file below a workspace."""
    relative = path.resolve().relative_to(workspace.path.resolve()).as_posix()
    artifact = {"path": relative, "sha256": _sha256(path)}
    if role is not None:
        artifact["role"] = role
    return artifact


def _adapter_outputs(run_dir: Path, input_path: Path) -> List[Path]:
    """Return adapter-produced files, excluding the generated adapter input."""
    return sorted(
        path for path in run_dir.rglob("*") if path.is_file() and path != input_path
    )


def _write_adapter_manifest(
    workspace: CandidateWorkspace,
    engine: str,
    run_id: str,
    run_dir: Path,
    started_at: str,
    runtime: Dict[str, str],
    input_path: Path,
    status: str,
    failure: Optional[Dict[str, str]] = None,
) -> Path:
    """Write the engine-run record after hashing all candidate-local adapter files."""
    output_paths = _adapter_outputs(run_dir, input_path)
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "engine": engine,
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "inputs": [_adapter_artifact(workspace, input_path, "adapter-input")],
        "outputs": [_adapter_artifact(workspace, path) for path in output_paths],
    }
    if failure is not None:
        manifest["failure"] = failure
    manifest_path = run_dir / "engine-run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest_path


def _runtime_version(module: Any, distribution: str) -> str:
    """Return installed package metadata when available without requiring a dependency pin."""
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:
        module_version = getattr(module, "__version__", None)
        return str(module_version) if module_version else "unknown"


def _read_pysyd_estimates(path: Path) -> List[Dict[str, Any]]:
    """Read pySYD's CSV result without adding a pandas runtime requirement."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = []
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("pySYD estimates have no CSV header")
        for row in reader:
            parsed: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None or not value.strip():
                    parsed[key] = None
                    continue
                try:
                    number = float(value)
                except ValueError:
                    parsed[key] = value
                    continue
                if not math.isfinite(number):
                    raise ValueError("pySYD estimates contain a non-finite value")
                parsed[key] = number
            rows.append(parsed)
    return rows


def _run_pysyd_adapter(
    workspace: CandidateWorkspace,
    time: np.ndarray,
    flux: np.ndarray,
    numax_min_uhz: float,
    numax_max_uhz: float,
) -> Dict[str, Any]:
    """Run pySYD in a candidate-local directory and normalize its adapter status."""
    started_at = datetime.now(timezone.utc).isoformat()
    run_id, run_dir = _adapter_run_dir(workspace, "pysyd")
    input_path = run_dir / "asteroseismic_input_LC.txt"
    np.savetxt(
        input_path,
        np.column_stack((time, np.asarray(flux) / 1e6)),
        fmt="%.10f %.12f",
    )
    runtime = {"kind": "direct", "version": "unavailable", "executable": "pysyd"}
    try:
        pysyd = importlib.import_module("pysyd")
    except ModuleNotFoundError as exc:
        manifest_path = _write_adapter_manifest(
            workspace, "pysyd", run_id, run_dir, started_at, runtime, input_path,
            "unavailable", {"code": "module-unavailable", "message": str(exc)},
        )
        return {"status": "unavailable", "manifest_path": manifest_path, "crosscheck": None}
    except Exception as exc:
        manifest_path = _write_adapter_manifest(
            workspace, "pysyd", run_id, run_dir, started_at, runtime, input_path,
            "failed", {"code": "module-import-failed", "message": str(exc)},
        )
        return {"status": "failed", "manifest_path": manifest_path, "crosscheck": None}

    runtime["version"] = _runtime_version(pysyd, "pysyd")
    main_func = getattr(pysyd, "main", None)
    if not callable(main_func):
        manifest_path = _write_adapter_manifest(
            workspace, "pysyd", run_id, run_dir, started_at, runtime, input_path,
            "unavailable", {
                "code": "unsupported-interface",
                "message": "The installed pySYD module does not expose a callable main entry point.",
            },
        )
        return {"status": "unavailable", "manifest_path": manifest_path, "crosscheck": None}

    try:
        previous_directory = Path.cwd()
        try:
            os.chdir(run_dir)
            main_func(["-f", str(input_path)])
        finally:
            os.chdir(previous_directory)
        estimates_path = run_dir / "estimates.csv"
        if not estimates_path.is_file():
            manifest_path = _write_adapter_manifest(
                workspace, "pysyd", run_id, run_dir, started_at, runtime, input_path,
                "unavailable", {
                    "code": "unsupported-output-interface",
                    "message": "pySYD completed without its documented estimates.csv output.",
                },
            )
            return {"status": "unavailable", "manifest_path": manifest_path, "crosscheck": None}
        crosscheck = {
            "pipeline": "pysyd",
            "estimates": _read_pysyd_estimates(estimates_path),
            "requested_search_range_uhz": [float(numax_min_uhz), float(numax_max_uhz)],
        }
    except Exception as exc:
        manifest_path = _write_adapter_manifest(
            workspace, "pysyd", run_id, run_dir, started_at, runtime, input_path,
            "failed", {"code": "adapter-execution-failed", "message": str(exc)},
        )
        return {"status": "failed", "manifest_path": manifest_path, "crosscheck": None}

    manifest_path = _write_adapter_manifest(
        workspace, "pysyd", run_id, run_dir, started_at, runtime, input_path, "succeeded"
    )
    return {"status": "succeeded", "manifest_path": manifest_path, "crosscheck": crosscheck}


def _record_tess_atl_adapter(workspace: CandidateWorkspace) -> Dict[str, Any]:
    """Record tess-atl availability without querying or inventing stellar values."""
    started_at = datetime.now(timezone.utc).isoformat()
    run_id, run_dir = _adapter_run_dir(workspace, "tess-atl")
    input_path = run_dir / "adapter-request.json"
    input_path.write_text(
        json.dumps(
            {
                "adapter": "tess-atl",
                "purpose": "availability and interface provenance only",
                "network_requests": "not-attempted",
            },
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    runtime = {"kind": "external", "version": "unavailable", "executable": "tess-atl"}
    try:
        module_spec = importlib.util.find_spec("tess_atl")
    except Exception as exc:
        manifest_path = _write_adapter_manifest(
            workspace, "tess-atl", run_id, run_dir, started_at, runtime, input_path,
            "failed", {"code": "availability-check-failed", "message": str(exc)},
        )
        return {"status": "failed", "manifest_path": manifest_path}
    if module_spec is None:
        failure = {
            "code": "module-unavailable",
            "message": "No local tess-atl module is installed; no request was attempted.",
        }
    else:
        failure = {
            "code": "unsupported-interface",
            "message": "A tess-atl module is present, but no supported local analysis interface is configured; no request was attempted.",
        }
        runtime["version"] = _runtime_version(None, "tess-atl")
    manifest_path = _write_adapter_manifest(
        workspace, "tess-atl", run_id, run_dir, started_at, runtime, input_path,
        "unavailable", failure,
    )
    return {"status": "unavailable", "manifest_path": manifest_path}


def _synthetic_oscillation_table() -> Dict[str, np.ndarray]:
    """Deterministic demonstration light curve with an injected p-mode comb.

    The comb carries a Gaussian amplitude envelope so the whitened PSD
    envelope peaks near the injected nu_max.
    """
    rng = np.random.default_rng(seed=23)
    numax_demo_uhz = 250.0
    dnu_demo_uhz = 40.0
    envelope_sigma_uhz = 2.5 * dnu_demo_uhz
    cadence_days = 120.0 / SECONDS_PER_DAY
    time = np.arange(0.0, 27.0, cadence_days)
    flux = np.ones_like(time)
    for harmonic in range(-4, 5):
        amplitude = 120e-6 * math.exp(
            -((harmonic * dnu_demo_uhz) ** 2) / (2.0 * envelope_sigma_uhz**2)
        )
        frequency_cpd = (numax_demo_uhz + harmonic * dnu_demo_uhz) * CPD_PER_UHZ
        flux = flux + amplitude * np.sin(2.0 * np.pi * frequency_cpd * time)
    flux = flux + rng.normal(0.0, 30e-6, size=time.shape)
    flux_err = np.full_like(flux, 30e-6)
    sector_values = np.ones(time.size, dtype=int)
    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "sector": sector_values,
    }


def run_asteroseismology(
    workspace: CandidateWorkspace,
    numax_min_uhz: float = 100.0,
    numax_max_uhz: float = 1600.0,
) -> Path:
    """Run the candidate-local exploratory asteroseismology workflow.

    The runner loads provenance-bound photometry, masks transits only when a
    complete candidate-derived ephemeris is available, estimates the native
    PSD envelope, records scaling results and sanity checks, and preserves
    optional adapter status.

    Args:
        workspace (CandidateWorkspace): Workspace that owns photometry, stellar
            parameters, provenance, adapter runs, and output artifacts.
        numax_min_uhz (float): Requested lower envelope-search bound in
            microhertz.
        numax_max_uhz (float): Requested upper envelope-search bound in
            microhertz.

    Returns:
        Path: Candidate-local ``outputs/asteroseismic_results.json`` with
        provenance, native diagnostics, optional adapter manifests, and stated
        calibration limits.

    Raises:
        RuntimeError: Required candidate photometry or stellar parameters are
            unavailable, or a required candidate ephemeris is unsuitable.
        ValueError: The requested search bounds or usable PSD inputs are invalid.
        OSError: Candidate-local output or adapter artifacts cannot be written.

    Note:
        The result is explicitly exploratory.  It does not provide calibrated
        detection probabilities, an automatic validation constraint, or a
        lifecycle transition.
    """
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    table = load_light_curve_table(workspace, require_raw_provenance=True)
    if table is None:
        raise RuntimeError("asteroseismology requires observed candidate photometry")
    source = "candidate-data"

    time = table["time"]
    flux = table["flux"]
    ephemeris = load_transit_ephemeris(workspace)
    cadence_seconds = 120.0
    if time.size > 1:
        cadence_seconds = float(np.median(np.diff(np.sort(time)))) * SECONDS_PER_DAY
    required_fields = ("period_days", "epoch_btjd", "duration_days")
    can_mask_transits = ephemeris.get("source") != "synthetic-demo" and all(
        ephemeris.get("field_sources", {}).get(field) != "synthetic-demo"
        for field in required_fields
    )
    if can_mask_transits:
        phase_days = phase_hours(time, ephemeris["period_days"], ephemeris["epoch_btjd"]) / 24.0
        transit_mask = np.abs(phase_days) >= 0.75 * ephemeris["duration_days"]
        masked_time = time[transit_mask]
        masked_flux = flux[transit_mask]
        transit_mask_status = "applied-candidate-ephemeris"
    else:
        masked_time = time
        masked_flux = flux
        transit_mask_status = "not-applied-no-candidate-ephemeris"
    if masked_time.size < 100:
        masked_time = time
        masked_flux = flux
        transit_mask_status = "not-applied-insufficient-post-mask-cadences"

    detrended_time, detrended_flux = _highpass_segments(
        masked_time, masked_flux, cadence_seconds, window_days=1.0
    )
    if detrended_time.size < 100:
        detrended_time, detrended_flux = masked_time, masked_flux

    envelope = estimate_oscillation_envelope(
        detrended_time, detrended_flux, numax_min_uhz, numax_max_uhz
    )
    stellar_params = load_stellar_parameters(workspace)
    dnu_correction = _resolve_dnu_correction(
        stellar_params,
        envelope["dnu_candidate_uhz"],
        _stellar_parameters_artifact(workspace),
    )
    seismic = seismic_mass_radius(
        envelope["numax_candidate_uhz"],
        envelope["dnu_candidate_uhz"],
        stellar_params["teff_k"],
        mass_prior_solar=stellar_params["mass_solar"],
        radius_prior_solar=stellar_params["radius_solar"],
        dnu_correction_factor=dnu_correction["factor"],
    )
    sanity = seismic_sanity_check(
        seismic,
        radius_prior_solar=stellar_params["radius_solar"],
        prior_is_catalog=stellar_params.get("source") == "candidate-data",
    )
    uncertainty = seismic_uncertainty_summary(
        envelope,
        stellar_params,
        dnu_correction_factor=dnu_correction["factor"],
    )

    pysyd_adapter = _run_pysyd_adapter(
        workspace, detrended_time, detrended_flux, numax_min_uhz, numax_max_uhz
    )
    tess_atl_adapter = _record_tess_atl_adapter(workspace)
    pysyd_result = pysyd_adapter["crosscheck"]

    payload = {
        "schema_version": "1.0",
        "work_package": "ASTEROSEISMOLOGY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "scientific_status": "exploratory-scaling-diagnostic",
        "validation_eligible": False,
        "validation_reason": (
            "The envelope and solar scaling relations are not a calibrated "
            "mode-identification or stellar-parameter inference. They cannot "
            "supply an automatic validation constraint."
        ),
        "transit_mask_status": transit_mask_status,
        "status": (
            "scaling_rejected_unphysical"
            if not sanity["plausible"]
            else (
                "oscillation_envelope_estimated"
                if envelope["dnu_candidate_uhz"] is not None
                else "envelope_estimated_dnu_undetermined"
            )
        ),
        "pipeline": "pysyd-crosscheck" if pysyd_result else "whitened-gls-psd",
        "search_range_uhz": [
            float(envelope["numax_min_used"]),
            float(envelope["numax_max_used"]),
        ],
        "requested_search_range_uhz": [
            float(envelope["numax_min_requested_uhz"]),
            float(envelope["numax_max_requested_uhz"]),
        ],
        "numax_search_bounds": {
            "supported_range_uhz": [PSD_MIN_UHZ, PSD_MAX_UHZ],
            "lower_clipped": bool(envelope["numax_min_clipped"]),
            "upper_clipped": bool(envelope["numax_max_clipped"]),
        },
        "numax_uhz": envelope["numax_candidate_uhz"],
        "envelope_peak_ratio": envelope["envelope_peak_ratio"],
        "dnu_uhz": envelope["dnu_candidate_uhz"],
        "dnu_correlation": envelope["dnu_correlation"],
        "dnu_correction": dnu_correction,
        "rayleigh_uhz": envelope["rayleigh_uhz"],
        "n_points_analyzed": int(detrended_time.size),
        "baseline_days": envelope["baseline_days"],
        "stellar_parameters": {
            "mass_solar": seismic["mass_solar"],
            "radius_solar": seismic["radius_solar"],
            "method": seismic["method"],
            "teff_k_prior": stellar_params["teff_k"],
            "mass_prior_solar": stellar_params["mass_solar"],
            "radius_prior_solar": stellar_params["radius_solar"],
            "validity": sanity,
        },
        "uncertainty": uncertainty,
        "pysyd_crosscheck": pysyd_result,
        "external_adapters": {
            "pysyd": {
                "status": pysyd_adapter["status"],
                "manifest_path": pysyd_adapter["manifest_path"].relative_to(workspace.path).as_posix(),
            },
            "tess-atl": {
                "status": tess_atl_adapter["status"],
                "manifest_path": tess_atl_adapter["manifest_path"].relative_to(workspace.path).as_posix(),
            },
        },
        "caveat": (
            "Candidate envelope peaks and spacing correlations are preliminary "
            "diagnostics; calibrated detection probabilities require null "
            "simulations and injection/recovery gates. Missing candidate "
            "ephemerides leave transit cadences unmasked rather than synthetic-masked."
        ),
    }
    output_path = outputs_dir / "asteroseismic_results.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_path
