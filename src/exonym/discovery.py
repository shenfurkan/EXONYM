"""Deterministic survey-search robustness primitives.

These functions produce diagnostics for candidate-local survey artifacts. They
do not assign a scientific disposition or calibrate a population-level false
alarm rate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter

from .lightcurve import phase_hours
from .search import BLSSearchResult, find_transits


def detrend_by_sector(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    sectors: Sequence[int],
    window_days: float,
) -> np.ndarray:
    """Divide each sector by a deterministic running median trend.

    Args:
        time_btjd: Observation times in BTJD.
        flux: Normalized flux values.
        sectors: TESS sector label for each cadence.
        window_days: Median-filter window width in days.

    Returns:
        Locally detrended normalized flux. The attenuation of transit signals
        must be measured by injection recovery for each frozen configuration.
    """
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    labels = np.asarray(sectors, dtype=int)
    if time.shape != values.shape or time.shape != labels.shape:
        raise ValueError("time, flux, and sectors must have matching shapes")
    if not np.isfinite(window_days) or window_days <= 0:
        raise ValueError("window_days must be positive and finite")

    result = np.full_like(values, np.nan)
    for sector in np.unique(labels):
        mask = labels == sector
        sector_time = time[mask]
        sector_flux = values[mask]
        order = np.argsort(sector_time)
        sorted_time = sector_time[order]
        sorted_flux = sector_flux[order]
        finite = np.isfinite(sorted_time) & np.isfinite(sorted_flux)
        if finite.sum() < 3:
            continue
        cadence_days = np.median(np.diff(sorted_time[finite]))
        width = max(3, int(round(window_days / cadence_days))) if cadence_days > 0 else 3
        if width % 2 == 0:
            width += 1
        trend = median_filter(sorted_flux, size=width, mode="nearest")
        trend[~np.isfinite(trend) | (trend == 0)] = 1.0
        detrended = sorted_flux / trend
        restored = np.empty_like(detrended)
        restored[order] = detrended
        result[mask] = restored
    return result


def search_duration_grid(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    duration_grid_hours: Sequence[float],
    period_min_days: float,
    period_max_days: float,
    n_periods: int,
) -> Tuple[BLSSearchResult, List[Dict[str, float]]]:
    """Search each declared duration and return the highest-SNR result."""
    durations = [float(value) for value in duration_grid_hours]
    if not durations or any(not np.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("duration_grid_hours must contain positive finite values")
    if len(set(durations)) != len(durations):
        raise ValueError("duration_grid_hours must not contain duplicates")
    results: List[Tuple[BLSSearchResult, Dict[str, float]]] = []
    for duration_hours in durations:
        result = find_transits(
            time_btjd,
            flux,
            period_min=period_min_days,
            period_max=period_max_days,
            n_periods=n_periods,
            duration_hours=duration_hours,
        )
        results.append((result, result.to_dict()))
    best, _ = max(results, key=lambda item: item[0].snr)
    return best, [payload for _, payload in results]


def inverted_flux(flux: Sequence[float]) -> np.ndarray:
    """Reflect normalized flux about unity for an inverted-event control."""
    values = np.asarray(flux, dtype=float)
    return 2.0 - values


def scrambled_flux(flux: Sequence[float], seed: int) -> np.ndarray:
    """Return a deterministic time-scrambled flux control."""
    return np.random.default_rng(seed).permutation(np.asarray(flux, dtype=float))


def inject_box_transit(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
    depth_ppm: float,
) -> np.ndarray:
    """Inject a multiplicative box transit into normalized candidate flux."""
    if (
        not np.isfinite(period_days)
        or not np.isfinite(epoch_btjd)
        or not np.isfinite(duration_hours)
        or not np.isfinite(depth_ppm)
        or period_days <= 0
        or duration_hours <= 0
        or depth_ppm <= 0
    ):
        raise ValueError("injection period, duration, and depth must be positive")
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float).copy()
    in_transit = np.abs(phase_hours(time, period_days, epoch_btjd)) <= 0.5 * duration_hours
    values[in_transit] *= 1.0 - depth_ppm * 1e-6
    return values


def mask_box_transit(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
) -> Tuple[np.ndarray, int]:
    """Mask a detected box event before testing an injected replacement.

    Masking prevents a recovery search from crediting the pre-existing BLS peak
    instead of the injected event. NaN values are removed by the BLS loader,
    retaining the original time sampling and its gaps.
    """
    if (
        not np.isfinite(period_days)
        or not np.isfinite(epoch_btjd)
        or not np.isfinite(duration_hours)
        or period_days <= 0
        or duration_hours <= 0
    ):
        raise ValueError("masked period and duration must be positive and finite")
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    if time.shape != values.shape:
        raise ValueError("time and flux must have matching shapes")
    in_transit = np.abs(phase_hours(time, period_days, epoch_btjd)) <= 0.5 * duration_hours
    masked = values.copy()
    masked[in_transit] = np.nan
    return masked, int(np.count_nonzero(in_transit))


def recovered_period(injected_period_days: float, recovered_period_days: float, tolerance: float) -> bool:
    """Return whether a recovered period agrees with an injected period."""
    if (
        not np.isfinite(injected_period_days)
        or not np.isfinite(recovered_period_days)
        or not np.isfinite(tolerance)
        or injected_period_days <= 0
        or recovered_period_days <= 0
        or tolerance <= 0
    ):
        raise ValueError("periods and tolerance must be positive")
    ratio = recovered_period_days / injected_period_days
    return abs(ratio - 1.0) <= tolerance


def recovered_epoch(
    injected_period_days: float,
    injected_epoch_btjd: float,
    recovered_epoch_btjd: float,
    tolerance_hours: float,
) -> bool:
    """Return whether a recovered epoch is within the injected event window."""
    if (
        not np.isfinite(injected_period_days)
        or not np.isfinite(injected_epoch_btjd)
        or not np.isfinite(recovered_epoch_btjd)
        or not np.isfinite(tolerance_hours)
        or injected_period_days <= 0
        or tolerance_hours <= 0
    ):
        raise ValueError("epoch recovery inputs must be finite with positive tolerances")
    offset_hours = phase_hours(
        np.asarray([recovered_epoch_btjd]), injected_period_days, injected_epoch_btjd
    )[0]
    return bool(abs(offset_hours) <= tolerance_hours)


def robustness_diagnostics(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    sectors: Sequence[int],
    duration_grid_hours: Sequence[float],
    period_min_days: float,
    period_max_days: float,
    n_periods: int,
    detrend_window_days: float,
    scramble_seeds: Sequence[int],
) -> Dict[str, Any]:
    """Run frozen duration, detrending, and null-control diagnostics.

    The returned result is diagnostic evidence. It does not estimate a
    population false-alarm probability or validate an astrophysical source.
    """
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    labels = np.asarray(sectors, dtype=int)
    variants = {
        "normalized": values,
        "running-median": detrend_by_sector(time, values, labels, detrend_window_days),
    }
    variant_results: Dict[str, Dict[str, Any]] = {}
    for name, variant_flux in variants.items():
        best, trials = search_duration_grid(
            time,
            variant_flux,
            duration_grid_hours,
            period_min_days,
            period_max_days,
            n_periods,
        )
        variant_results[name] = {"best": best.to_dict(), "trials": trials}

    inverted, _ = search_duration_grid(
        time,
        inverted_flux(values),
        duration_grid_hours,
        period_min_days,
        period_max_days,
        n_periods,
    )
    scrambled_results = []
    for seed in scramble_seeds:
        result, _ = search_duration_grid(
            time,
            scrambled_flux(values, int(seed)),
            duration_grid_hours,
            period_min_days,
            period_max_days,
            n_periods,
        )
        scrambled_results.append({"seed": int(seed), "best": result.to_dict()})
    null_snr = [inverted.snr] + [entry["best"]["snr"] for entry in scrambled_results]
    return {
        "variants": variant_results,
        "controls": {
            "inverted": inverted.to_dict(),
            "scrambles": scrambled_results,
            "max_snr": float(max(null_snr)),
        },
    }


def injection_recovery_diagnostics(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    injections: Sequence[Dict[str, float]],
    duration_grid_hours: Sequence[float],
    period_min_days: float,
    period_max_days: float,
    n_periods: int,
    tolerance: float,
    minimum_snr: Optional[float] = None,
    epoch_tolerance_duration_fraction: float = 1.0,
) -> List[Dict[str, Any]]:
    """Measure recovery of declared synthetic transits in real candidate flux.

    A recovered trial must match its injected period and epoch. When
    ``minimum_snr`` is supplied, it must also clear that pre-registered BLS
    threshold. This is an internal search-sensitivity diagnostic, not a
    completeness estimate or scientific disposition.
    """
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    if time.size == 0:
        raise ValueError("injection recovery requires at least one cadence")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("period recovery tolerance must be positive and finite")
    if (
        not np.isfinite(epoch_tolerance_duration_fraction)
        or epoch_tolerance_duration_fraction <= 0
    ):
        raise ValueError("epoch tolerance fraction must be positive and finite")
    if minimum_snr is not None and (
        not np.isfinite(minimum_snr) or minimum_snr <= 0
    ):
        raise ValueError("minimum_snr must be positive and finite when supplied")
    epoch = float(np.nanmin(time))
    results: List[Dict[str, Any]] = []
    for injection in injections:
        injected_epoch = float(injection.get("epoch_btjd", epoch))
        injected_period = float(injection["period_days"])
        injected_duration = float(injection["duration_hours"])
        injected_depth = float(injection["depth_ppm"])
        injected = inject_box_transit(
            time,
            values,
            injected_period,
            injected_epoch,
            injected_duration,
            injected_depth,
        )
        best, _ = search_duration_grid(
            time, injected, duration_grid_hours, period_min_days, period_max_days, n_periods
        )
        period_match = recovered_period(injected_period, best.best_period, tolerance)
        epoch_tolerance_hours = injected_duration * epoch_tolerance_duration_fraction
        epoch_match = recovered_epoch(
            injected_period, injected_epoch, best.best_epoch, epoch_tolerance_hours
        )
        snr_pass = minimum_snr is None or best.snr >= minimum_snr
        declared_injection = dict(injection)
        declared_injection["epoch_btjd"] = injected_epoch
        results.append(
            {
                "injection": declared_injection,
                "period_match": period_match,
                "epoch_match": epoch_match,
                "snr_pass": snr_pass,
                "epoch_tolerance_hours": epoch_tolerance_hours,
                "recovered": period_match and epoch_match and snr_pass,
                "best": best.to_dict(),
            }
        )
    return results
