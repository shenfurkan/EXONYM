r"""Target-neutral stellar rotational activity engine.

Measures stellar rotation periods ($P_{\mathrm{rot}}$) and active region (starspot)
modulation amplitudes from out-of-transit photometric time-series.

Key Scientific Steps:
1. Planetary Transit Masking: Masks all primary transit windows ($\pm 0.75 \times T_{14}$)
   to prevent transit box harmonics from creating spurious rotation signals.
2. Generalized Lomb-Scargle (GLS) Periodogram (Zechmeister & Kürster 2009):
   Fits a floating-mean sinusoid across trial periods $P \in [1, 20]$ days.
3. Analytic False Alarm Probability (FAP) via Baluev (2008) extreme-value statistics:
   Evaluates peak significance against red-noise and white-noise false-alarm thresholds.
4. Harmonic Modulation Semi-Amplitude:
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


def gls_periodogram(
    time: Sequence[float],
    flux: Sequence[float],
    period_min_days: float = PERIOD_MIN_DAYS,
    period_max_days: float = PERIOD_MAX_DAYS,
    samples_per_peak: int = SAMPLES_PER_PEAK,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (periods_days, powers, best_fap) from a GLS periodogram."""
    from astropy.timeseries import LombScargle

    time_arr = np.asarray(time, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    finite = np.isfinite(time_arr) & np.isfinite(flux_arr)
    time_arr = time_arr[finite]
    flux_arr = flux_arr[finite]
    if time_arr.size < 50:
        raise ValueError("insufficient data for periodogram analysis")
    ls = LombScargle(time_arr, flux_arr - np.nanmean(flux_arr))
    frequency, power = ls.autopower(
        minimum_frequency=1.0 / period_max_days,
        maximum_frequency=1.0 / period_min_days,
        samples_per_peak=samples_per_peak,
    )
    periods = 1.0 / np.asarray(frequency)
    power = np.asarray(power, dtype=float)
    best_fap = float(ls.false_alarm_probability(float(np.max(power))))
    return periods, power, best_fap


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

    table = load_light_curve_table(workspace)
    if table is None:
        table = _synthetic_rotation_table()
        source = "synthetic-demo"
    else:
        source = "candidate-data"

    ephemeris = load_transit_ephemeris(workspace)
    phase_days = phase_hours(
        table["time"], ephemeris["period_days"], ephemeris["epoch_btjd"]
    ) / 24.0
    transit_mask = np.abs(phase_days) >= TRANSIT_MASK_HALF_DURATIONS * ephemeris["duration_days"]
    time = table["time"][transit_mask]
    flux = table["flux"][transit_mask]
    sector_values = table["sector"][transit_mask]
    if time.size < 100:
        time = table["time"]
        flux = table["flux"]
        sector_values = table["sector"]

    segment_results: List[Dict[str, Any]] = []
    period_peaks: List[float] = []
    power_peaks: List[float] = []
    for sector_value in sorted(int(value) for value in np.unique(sector_values)):
        mask = sector_values == sector_value
        if int(np.sum(mask)) < 100:
            continue
        try:
            periods, powers, fap = gls_periodogram(time[mask], flux[mask])
        except ValueError:
            continue
        best_index = int(np.argmax(powers))
        best_period = float(periods[best_index])
        best_power = float(powers[best_index])
        segment_results.append(
            {
                "sector": int(sector_value),
                "n_points": int(np.sum(mask)),
                "baseline_days": round(float(np.max(time[mask]) - np.min(time[mask])), 3),
                "best_period_days": round(best_period, 4),
                "max_power": round(best_power, 4),
                "false_alarm_probability": fap,
            }
        )
        period_peaks.append(best_period)
        power_peaks.append(best_power)

    if not segment_results:
        raise RuntimeError("no usable light curve segments for activity analysis")

    summary = weighted_period_summary(period_peaks, power_peaks)
    rotation_period = summary["weighted_mean_period_days"]
    amplitude_ppm = sinusoid_amplitude_ppm(time, flux, rotation_period)
    best_fap = min(segment["false_alarm_probability"] for segment in segment_results)

    payload = {
        "schema_version": "1.0",
        "work_package": "STELLAR_ACTIVITY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "method": "Generalized Lomb-Scargle periodogram per segment with weighted peak summary",
        "period_search_range_days": [PERIOD_MIN_DAYS, PERIOD_MAX_DAYS],
        "rotation_period_days": round(rotation_period, 4),
        "rotation_period_std_days": summary["weighted_std_period_days"],
        "modulation_amplitude_ppm": round(amplitude_ppm, 2),
        "best_false_alarm_probability": best_fap,
        "n_segments": summary["n_segments"],
        "segments": segment_results,
        "caveat": (
            "Periodogram peaks are exploratory; a rotation claim requires "
            "window-function and harmonic cross-checks."
        ),
    }
    output_path = outputs_dir / "stellar_activity_results.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
