r"""Target-neutral stellar rotational activity engine.

Measures stellar rotation periods ($P_{\mathrm{rot}}$) and active region (starspot)
modulation amplitudes from out-of-transit photometric time-series.

Key Scientific Steps:
1. Planetary Transit Masking: Masks all primary transit windows ($\pm 0.75 \times T_{14}$)
   to prevent transit box harmonics from creating spurious rotation signals.
2. Generalized Lomb-Scargle (GLS) Periodogram (Zechmeister & Kürster 2009):
   Fits a floating-mean sinusoid across trial periods $P \in [1, 20]$ days.
3. Sampling-window and cross-sector harmonic diagnostics:
   Preserve cadence-window features and whether segment peaks are compatible with
   one fundamental frequency or its first harmonic.
4. Analytic White-noise FAP and Harmonic Modulation Semi-Amplitude:
   The Baluev-style GLS probability is retained as a white-noise reference only;
   it is not a red-noise or population-calibrated activity probability.
   Fits out-of-transit flux $y(t) = A_1 \cos(2\pi t / P) + A_2 \sin(2\pi t / P) + C$,
   extracting total starspot amplitude $A = \sqrt{A_1^2 + A_2^2}$ in ppm.

Contains zero candidate-specific identifiers or constants; all searches operate
strictly within candidate-provided photometric inputs.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .inputs import load_light_curve_table, load_transit_ephemeris
from .lightcurve import phase_hours
from .workspace import CandidateWorkspace

PERIOD_MIN_DAYS = 1.0               # Minimum rotation period for GLS search grid (days)
PERIOD_MAX_DAYS = 20.0              # Maximum rotation period for GLS search grid (days)
SAMPLES_PER_PEAK = 10               # Frequency oversampling factor per Rayleigh peak
TRANSIT_MASK_HALF_DURATIONS = 0.75  # Planetary transit exclusion window half-width
WINDOW_PEAK_LIMIT = 5               # Retained cadence-window peaks per segment
WINDOW_COMPLEX_ELEMENT_LIMIT = 1500000  # Bound temporary complex-array size
HARMONIC_FREQUENCY_FACTORS = (0.5, 1.0, 2.0)


def gls_periodogram(
    time: Sequence[float],
    flux: Sequence[float],
    flux_err: Optional[Sequence[float]] = None,
    period_min_days: float = PERIOD_MIN_DAYS,
    period_max_days: float = PERIOD_MAX_DAYS,
    samples_per_peak: int = SAMPLES_PER_PEAK,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return GLS periods, powers, and its analytic white-noise FAP reference."""
    from astropy.timeseries import LombScargle

    time_arr = np.asarray(time, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    error_arr = None if flux_err is None else np.asarray(flux_err, dtype=float)
    if error_arr is not None and error_arr.shape != flux_arr.shape:
        raise ValueError("flux_err must match flux")
    finite = np.isfinite(time_arr) & np.isfinite(flux_arr)
    if error_arr is not None:
        finite &= np.isfinite(error_arr) & (error_arr > 0)
    time_arr = time_arr[finite]
    flux_arr = flux_arr[finite]
    if error_arr is not None:
        error_arr = error_arr[finite]
    if time_arr.size < 50:
        raise ValueError("insufficient data for periodogram analysis")
    ls = LombScargle(time_arr, flux_arr - np.nanmean(flux_arr), dy=error_arr)
    frequency, power = ls.autopower(
        minimum_frequency=1.0 / period_max_days,
        maximum_frequency=1.0 / period_min_days,
        samples_per_peak=samples_per_peak,
    )
    periods = 1.0 / np.asarray(frequency)
    power = np.asarray(power, dtype=float)
    analytic_white_noise_fap = float(ls.false_alarm_probability(float(np.max(power))))
    return periods, power, analytic_white_noise_fap


def sampling_window_periodogram(
    time_btjd: Sequence[float], frequency_days_inverse: Sequence[float]
) -> Tuple[np.ndarray, float]:
    """Evaluate the normalized spectral window at supplied frequencies.

    The window is ``|mean(exp(-2 pi i f (t - mean(t))))|^2``. It describes
    cadence and gap structure only; it is not a null distribution and does not
    establish that a photometric peak is an alias.
    """
    time = np.asarray(time_btjd, dtype=float)
    frequency = np.asarray(frequency_days_inverse, dtype=float)
    time = time[np.isfinite(time)]
    valid_frequency = np.isfinite(frequency) & (frequency > 0.0)
    if time.size < 2 or not np.any(valid_frequency):
        raise ValueError("sampling-window analysis requires finite times and positive frequencies")
    baseline_days = float(np.max(time) - np.min(time))
    if not math.isfinite(baseline_days) or baseline_days <= 0.0:
        raise ValueError("sampling-window analysis requires a positive time baseline")
    centered_time = time - float(np.mean(time))
    powers = np.full(frequency.shape, np.nan, dtype=float)
    usable_frequency = frequency[valid_frequency]
    chunk_size = max(1, int(WINDOW_COMPLEX_ELEMENT_LIMIT // centered_time.size))
    evaluated = np.empty(usable_frequency.size, dtype=float)
    for start in range(0, usable_frequency.size, chunk_size):
        stop = min(start + chunk_size, usable_frequency.size)
        phase = -2.0j * np.pi * np.outer(usable_frequency[start:stop], centered_time)
        evaluated[start:stop] = np.clip(
            np.abs(np.mean(np.exp(phase), axis=1)) ** 2, 0.0, 1.0
        )
    powers[valid_frequency] = evaluated
    return powers, baseline_days


def _top_window_peaks(
    frequency_days_inverse: np.ndarray,
    window_power: np.ndarray,
    frequency_resolution_days_inverse: float,
    limit: int = WINDOW_PEAK_LIMIT,
) -> List[Dict[str, float]]:
    """Retain separated local maxima of a sampling window for reviewer context."""
    valid = np.isfinite(frequency_days_inverse) & np.isfinite(window_power)
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return []
    local_maxima = [
        index
        for index in candidates
        if 0 < index < window_power.size - 1
        and window_power[index] >= window_power[index - 1]
        and window_power[index] >= window_power[index + 1]
    ]
    if not local_maxima:
        local_maxima = [int(candidates[np.argmax(window_power[candidates])])]
    selected: List[Dict[str, float]] = []
    for index in sorted(local_maxima, key=lambda value: float(window_power[value]), reverse=True):
        current_frequency = float(frequency_days_inverse[index])
        if any(
            abs(current_frequency - item["frequency_days_inverse"])
            < frequency_resolution_days_inverse
            for item in selected
        ):
            continue
        selected.append(
            {
                "frequency_days_inverse": current_frequency,
                "period_days": float(1.0 / current_frequency),
                "window_power": float(window_power[index]),
            }
        )
        if len(selected) == limit:
            break
    return selected


def sampling_window_diagnostics(
    time_btjd: Sequence[float],
    frequency_days_inverse: Sequence[float],
    best_frequency_days_inverse: float,
) -> Dict[str, Any]:
    """Describe the closest resolved sampling-window feature to one GLS peak."""
    window_power, baseline_days = sampling_window_periodogram(
        time_btjd, frequency_days_inverse
    )
    frequency = np.asarray(frequency_days_inverse, dtype=float)
    frequency_resolution = 1.0 / baseline_days
    peaks = _top_window_peaks(frequency, window_power, frequency_resolution)
    if not peaks or not math.isfinite(best_frequency_days_inverse) or best_frequency_days_inverse <= 0.0:
        nearest = None
        direct_proximity = None
    else:
        nearest = min(
            peaks,
            key=lambda item: abs(best_frequency_days_inverse - item["frequency_days_inverse"]),
        )
        nearest = dict(nearest)
        nearest["frequency_difference_days_inverse"] = abs(
            best_frequency_days_inverse - nearest["frequency_days_inverse"]
        )
        direct_proximity = (
            nearest["frequency_difference_days_inverse"] <= frequency_resolution
        )
    return {
        "method": "normalized-spectral-window-v1",
        "baseline_days": baseline_days,
        "frequency_resolution_days_inverse": frequency_resolution,
        "top_window_peaks": peaks,
        "nearest_window_peak": nearest,
        "direct_frequency_proximity_within_resolution": direct_proximity,
        "interpretation": (
            "Sampling-window proximity is a reviewer diagnostic only; it does not "
            "prove that the GLS peak is instrumental or an astrophysical rotation signal."
        ),
    }


def segment_harmonic_persistence(
    segment_peaks: Sequence[Dict[str, float]]
) -> Dict[str, Any]:
    """Find the descriptive fundamental/harmonic family supported by segments.

    Every observed segment frequency is divided by each allowed harmonic factor
    to propose a reference frequency. The deterministic choice maximizes the
    number of segment peaks compatible at their own Rayleigh-like resolution;
    ties prefer the lower frequency rather than silently promoting a harmonic.
    This is descriptive persistence evidence, not a rotation-period uncertainty
    or a statistical activity detection.
    """
    usable = [
        item
        for item in segment_peaks
        if all(
            isinstance(item.get(key), (int, float))
            and math.isfinite(float(item[key]))
            and float(item[key]) > 0.0
            for key in ("best_period_days", "baseline_days")
        )
    ]
    if len(usable) < 2:
        return {
            "status": "unresolved-insufficient-segments",
            "reference_frequency_days_inverse": None,
            "reference_period_days": None,
            "segments": [],
            "consistent_segment_count": 0,
            "interpretation": "At least two usable segments are required for cross-segment harmonic comparison.",
        }
    frequencies = np.asarray(
        [1.0 / float(item["best_period_days"]) for item in usable], dtype=float
    )
    candidate_references = sorted(
        float(frequency / factor)
        for frequency in frequencies
        for factor in HARMONIC_FREQUENCY_FACTORS
    )
    best_candidate: Optional[Tuple[int, float, float]] = None
    reference_frequency = None
    for candidate in candidate_references:
        compatible_count = 0
        normalized_residual_sum = 0.0
        for item, frequency in zip(usable, frequencies):
            factor = min(
                HARMONIC_FREQUENCY_FACTORS,
                key=lambda value: abs(float(frequency) / candidate - value),
            )
            residual = abs(float(frequency) - factor * candidate)
            resolution = 1.0 / float(item["baseline_days"])
            compatible_count += int(residual <= resolution)
            normalized_residual_sum += residual / resolution
        score = (compatible_count, -normalized_residual_sum, -candidate)
        if best_candidate is None or score > best_candidate:
            best_candidate = score
            reference_frequency = candidate
    if reference_frequency is None:
        raise RuntimeError("activity harmonic persistence failed to select a finite reference frequency")
    comparisons: List[Dict[str, Any]] = []
    consistent_count = 0
    for item, frequency in zip(usable, frequencies):
        factor = min(
            HARMONIC_FREQUENCY_FACTORS,
            key=lambda value: abs(float(frequency) / reference_frequency - value),
        )
        residual = abs(float(frequency) - factor * reference_frequency)
        resolution = 1.0 / float(item["baseline_days"])
        consistent = residual <= resolution
        consistent_count += int(consistent)
        comparisons.append(
            {
                "sector": int(item["sector"]),
                "best_period_days": float(item["best_period_days"]),
                "best_frequency_days_inverse": float(frequency),
                "nearest_harmonic_frequency_factor": float(factor),
                "frequency_residual_days_inverse": residual,
                "frequency_resolution_days_inverse": resolution,
                "compatible_with_reference_or_first_harmonic": consistent,
            }
        )
    return {
        "status": "descriptive-harmonic-consistency",
        "reference_frequency_days_inverse": float(reference_frequency),
        "reference_period_days": float(1.0 / reference_frequency),
        "segments": comparisons,
        "consistent_segment_count": consistent_count,
        "interpretation": (
            "Compatibility uses each segment's frequency resolution and permits "
            "half/double-frequency modulation; evolving spots and red noise remain unmodeled."
        ),
    }


def sinusoid_amplitude_ppm(
    time: Sequence[float], flux: Sequence[float], period_days: float
) -> float:
    """Fit a fixed-period sinusoid and return its amplitude in ppm."""
    time_arr = np.asarray(time, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    if period_days <= 0:
        raise ValueError("period must be positive")
    angle = 2.0 * np.pi * time_arr / period_days
    design = np.column_stack((np.cos(angle), np.sin(angle)))
    coefficients, _, _, _ = np.linalg.lstsq(design, flux_arr - np.nanmean(flux_arr), rcond=None)
    amplitude = math.hypot(float(coefficients[0]), float(coefficients[1]))
    return amplitude * 1e6


def weighted_period_summary(
    periods_days: Sequence[float], powers: Sequence[float]
) -> Dict[str, float]:
    """Weighted mean and standard deviation of per-segment period peaks."""
    periods = np.asarray(periods_days, dtype=float)
    weights = np.asarray(powers, dtype=float)
    if periods.size == 0:
        raise ValueError("no periodogram peaks to summarize")
    if float(np.sum(weights)) <= 0:
        weights = np.ones_like(weights)
    weights = weights / float(np.sum(weights))
    mean_period = float(np.sum(periods * weights))
    variance = float(np.sum(weights * (periods - mean_period) ** 2))
    return {
        "weighted_mean_period_days": round(mean_period, 4),
        "weighted_std_period_days": round(math.sqrt(variance), 4),
        "n_segments": int(periods.size),
    }


def weighted_percentile_summary(
    values: Sequence[float], weights: Sequence[float]
) -> Dict[str, float]:
    """Return weighted 16th, 50th, and 84th percentile summaries."""
    samples = np.asarray(values, dtype=float)
    sample_weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(samples) & np.isfinite(sample_weights) & (sample_weights >= 0)
    samples, sample_weights = samples[valid], sample_weights[valid]
    if samples.size == 0:
        raise ValueError("no finite values are available for a weighted percentile summary")
    if float(np.sum(sample_weights)) <= 0:
        sample_weights = np.ones(samples.size, dtype=float)
    order = np.argsort(samples)
    samples, sample_weights = samples[order], sample_weights[order]
    cumulative = np.cumsum(sample_weights) / float(np.sum(sample_weights))
    quantiles = [float(np.interp(level, cumulative, samples)) for level in (0.16, 0.50, 0.84)]
    return {
        "p16": quantiles[0],
        "median": quantiles[1],
        "p84": quantiles[2],
        "plus": quantiles[2] - quantiles[1],
        "minus": quantiles[1] - quantiles[0],
    }


def sinusoid_amplitude_posterior(
    time: Sequence[float],
    flux: Sequence[float],
    flux_err: Sequence[float],
    period_days: float,
) -> Dict[str, Any]:
    """Propagate weighted sinusoid-fit covariance into an amplitude interval."""
    time_arr = np.asarray(time, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    error_arr = np.asarray(flux_err, dtype=float)
    valid = np.isfinite(time_arr) & np.isfinite(flux_arr) & np.isfinite(error_arr) & (error_arr > 0)
    if int(np.count_nonzero(valid)) < 4 or period_days <= 0:
        raise ValueError("finite time, flux, and flux_err values are required for amplitude propagation")
    angle = 2.0 * np.pi * time_arr[valid] / period_days
    design = np.column_stack((np.cos(angle), np.sin(angle), np.ones(int(np.count_nonzero(valid)))))
    weights = 1.0 / error_arr[valid] ** 2
    normal = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError as exc:
        raise ValueError("sinusoid design matrix is singular") from exc
    coefficients = covariance @ (design.T @ (weights * flux_arr[valid]))
    residual = flux_arr[valid] - design @ coefficients
    degrees_of_freedom = max(1, int(np.count_nonzero(valid)) - design.shape[1])
    covariance *= float(np.sum(weights * residual**2) / degrees_of_freedom)
    amplitude = math.hypot(float(coefficients[0]), float(coefficients[1]))
    gradient = np.array([coefficients[0], coefficients[1], 0.0], dtype=float) / max(amplitude, 1e-15)
    variance = float(gradient @ covariance @ gradient)
    sigma = math.sqrt(max(variance, 0.0))
    return {
        "p16": (amplitude - sigma) * 1e6,
        "median": amplitude * 1e6,
        "p84": (amplitude + sigma) * 1e6,
        "plus": sigma * 1e6,
        "minus": sigma * 1e6,
        "covariance_cos_sin_baseline": covariance.tolist(),
        "covariance_parameter_order": ["cosine", "sine", "baseline"],
        "method": "weighted-linear-sinusoid covariance with residual variance scaling",
    }


def _synthetic_rotation_table() -> Dict[str, np.ndarray]:
    """Deterministic demonstration light curve with an injected rotation signal."""
    rng = np.random.default_rng(seed=29)
    rotation_period_days = 5.0
    amplitude = 400e-6
    cadence_days = 120.0 / 86400.0
    time = np.arange(0.0, 27.0, cadence_days)
    flux = 1.0 + amplitude * np.sin(2.0 * np.pi * time / rotation_period_days)
    flux = flux + rng.normal(0.0, 150e-6, size=time.shape)
    flux_err = np.full_like(flux, 150e-6)
    sector_values = np.ones(time.size, dtype=int)
    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "sector": sector_values,
    }


def run_stellar_activity(workspace: CandidateWorkspace) -> Path:
    """Run the stellar activity analysis and write outputs/stellar_activity_results.json."""
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    table = load_light_curve_table(workspace, require_raw_provenance=True)
    if table is None:
        raise RuntimeError("stellar activity analysis requires observed candidate photometry")
    source = "candidate-data"

    ephemeris = load_transit_ephemeris(workspace)
    required_fields = ("period_days", "epoch_btjd", "duration_days")
    can_mask_transits = ephemeris.get("source") != "synthetic-demo" and all(
        ephemeris.get("field_sources", {}).get(field) != "synthetic-demo"
        for field in required_fields
    )
    if can_mask_transits:
        phase_days = phase_hours(
            table["time"], ephemeris["period_days"], ephemeris["epoch_btjd"]
        ) / 24.0
        transit_mask = np.abs(phase_days) >= TRANSIT_MASK_HALF_DURATIONS * ephemeris["duration_days"]
        time = table["time"][transit_mask]
        flux = table["flux"][transit_mask]
        flux_err = table["flux_err"][transit_mask]
        sector_values = table["sector"][transit_mask]
        transit_mask_status = "applied-candidate-ephemeris"
    else:
        time = table["time"]
        flux = table["flux"]
        flux_err = table["flux_err"]
        sector_values = table["sector"]
        transit_mask_status = "not-applied-no-candidate-ephemeris"
    if time.size < 100:
        time = table["time"]
        flux = table["flux"]
        flux_err = table["flux_err"]
        sector_values = table["sector"]
        transit_mask_status = "not-applied-insufficient-post-mask-cadences"

    segment_results: List[Dict[str, Any]] = []
    period_peaks: List[float] = []
    power_peaks: List[float] = []
    for sector_value in sorted(int(value) for value in np.unique(sector_values)):
        mask = sector_values == sector_value
        if int(np.sum(mask)) < 100:
            continue
        try:
            periods, powers, analytic_white_noise_fap = gls_periodogram(
                time[mask], flux[mask], flux_err[mask]
            )
        except ValueError:
            continue
        best_index = int(np.argmax(powers))
        best_period = float(periods[best_index])
        best_power = float(powers[best_index])
        frequency = 1.0 / periods
        try:
            window = sampling_window_diagnostics(
                time[mask], frequency, float(frequency[best_index])
            )
        except ValueError:
            continue
        baseline_days = float(np.max(time[mask]) - np.min(time[mask]))
        segment_results.append(
            {
                "sector": int(sector_value),
                "n_points": int(np.sum(mask)),
                "baseline_days": baseline_days,
                "best_period_days": round(best_period, 4),
                "max_power": round(best_power, 4),
                "analytic_white_noise_false_alarm_probability": analytic_white_noise_fap,
                "sampling_window": window,
            }
        )
        period_peaks.append(best_period)
        power_peaks.append(best_power)

    if not segment_results:
        raise RuntimeError("no usable light curve segments for activity analysis")

    summary = weighted_period_summary(period_peaks, power_peaks)
    rotation_period = summary["weighted_mean_period_days"]
    rotation_posterior = weighted_percentile_summary(period_peaks, power_peaks)
    amplitude_posterior = sinusoid_amplitude_posterior(time, flux, flux_err, rotation_period)
    best_analytic_white_noise_fap = min(
        segment["analytic_white_noise_false_alarm_probability"]
        for segment in segment_results
    )
    harmonic_persistence = segment_harmonic_persistence(segment_results)

    payload = {
        "schema_version": "1.0",
        "work_package": "STELLAR_ACTIVITY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "scientific_status": "exploratory-activity-diagnostic",
        "validation_eligible": False,
        "transit_mask_status": transit_mask_status,
        "method": "Generalized Lomb-Scargle per segment with sampling-window and harmonic-persistence diagnostics",
        "period_search_range_days": [PERIOD_MIN_DAYS, PERIOD_MAX_DAYS],
        "rotation_period_days": round(rotation_period, 4),
        "rotation_period_std_days": summary["weighted_std_period_days"],
        "rotation_period_posterior_days": rotation_posterior,
        "modulation_amplitude_ppm": round(amplitude_posterior["median"], 2),
        "modulation_amplitude_posterior_ppm": amplitude_posterior,
        "uncertainty_method": (
            "Power-weighted segment peak percentiles for rotation and weighted linear "
            "sinusoid covariance for amplitude; neither model includes evolving spots or red noise."
        ),
        "best_analytic_white_noise_false_alarm_probability": best_analytic_white_noise_fap,
        "n_segments": summary["n_segments"],
        "segments": segment_results,
        "harmonic_persistence": harmonic_persistence,
        "caveat": (
            "The reported FAP assumes independent white noise and is not a calibrated "
            "activity probability. Sampling-window and harmonic diagnostics are descriptive; "
            "a rotation claim still requires red-noise, evolving-spot, and persistence analysis. "
            "A missing candidate ephemeris leaves transits unmasked rather than using a synthetic mask."
        ),
    }
    output_path = outputs_dir / "stellar_activity_results.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_path
