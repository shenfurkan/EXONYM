"""Candidate-local robustness controls for bounded photometric searches.

These functions generate deterministic duration-grid, inverted-flux,
sector-preserving scramble, and injection-recovery diagnostics for survey
artifacts. They keep preprocessing choices and null controls inspectable
without assigning a lifecycle state or astrophysical disposition.

Scientific Boundary:
    A recovery fraction, BLS ranking statistic, or null-control comparison is
    configuration-specific evidence. It is not a survey selection function,
    population false-alarm calibration, discovery claim, or validation result.

References:
    methods/tls_search.md and methods/detrending-and-transit-inference.md
    describe the search and preprocessing limitations used by these controls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter

from .lightcurve import phase_hours
from .search import BLSSearchResult, find_transits_duration_grid


def detrend_by_sector(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    sectors: Sequence[int],
    window_days: float,
    gap_break_window_fraction: float = 0.5,
) -> np.ndarray:
    """Divide each contiguous sector visit by a running-median trend.

    Large cadence gaps are split before filtering so edge extension cannot
    borrow a baseline from a physically disconnected visit.

    Args:
        time_btjd: Observation times in BTJD_TDB days.
        flux: Normalized relative flux values.
        sectors: Positive mission-sector label for every cadence.
        window_days: Positive median-filter window width in days.
        gap_break_window_fraction: Positive fraction of window_days used when
            defining a gap between contiguous visits.

    Returns:
        Locally detrended normalized flux aligned with the input cadence order.

    Raises:
        ValueError: If input shapes differ or detrending controls are invalid.

    Notes:
        Transit attenuation must be measured by injection recovery for each
        frozen preprocessing configuration.
    """
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    labels = np.asarray(sectors, dtype=int)
    if time.shape != values.shape or time.shape != labels.shape:
        raise ValueError("time, flux, and sectors must have matching shapes")
    if not np.isfinite(window_days) or window_days <= 0:
        raise ValueError("window_days must be positive and finite")
    if (
        not np.isfinite(gap_break_window_fraction)
        or gap_break_window_fraction <= 0
        or gap_break_window_fraction > 1.0
    ):
        raise ValueError("gap_break_window_fraction must be in (0, 1]")

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
        finite_time = sorted_time[finite]
        cadence_days = np.median(np.diff(finite_time))
        width = max(3, int(round(window_days / cadence_days))) if cadence_days > 0 else 3
        if width % 2 == 0:
            width += 1
        # NUMERICAL_GUARD: Filtering in sample index is acceptable only within a contiguous
        # cadence run. Split large gaps so the edge extension of a median
        # filter cannot borrow values from a physically disconnected visit.
        gap_days = max(5.0 * float(cadence_days), gap_break_window_fraction * window_days)
        run_starts = np.r_[0, np.flatnonzero(np.diff(sorted_time) > gap_days) + 1]
        run_stops = np.r_[run_starts[1:], sorted_time.size]
        detrended = np.full_like(sorted_flux, np.nan)
        for start, stop in zip(run_starts, run_stops):
            run_flux = sorted_flux[start:stop]
            valid_flux = np.isfinite(run_flux)
            if valid_flux.sum() < 3:
                detrended[start:stop] = run_flux
                continue
            trend_input = run_flux.copy()
            if not np.all(valid_flux):
                indices = np.arange(run_flux.size)
                valid_idx = indices[valid_flux]
                valid_vals = run_flux[valid_flux]
                # NUMERICAL_GUARD: linear slope extrapolation instead of
                # np.interp flat-clamping, mirroring the sector-edge guard in
                # detrending._running_median_trend.
                if valid_idx.size >= 2:
                    from scipy.interpolate import interp1d

                    interpolator = interp1d(
                        valid_idx,
                        valid_vals,
                        kind="linear",
                        fill_value="extrapolate",
                        assume_sorted=True,
                    )
                    trend_input[~valid_flux] = interpolator(indices[~valid_flux])
                else:
                    trend_input[~valid_flux] = np.interp(
                        indices[~valid_flux], valid_idx, valid_vals
                    )
            trend = median_filter(trend_input, size=width, mode="nearest")
            patched = ~np.isfinite(trend) | (trend == 0)
            if patched.any():
                import logging

                logging.warning(
                    "detrend_by_sector: trend contains %d non-finite/zero values "
                    "in sector %d (start=%d, stop=%d); patching to 1.0",
                    int(np.count_nonzero(patched)), int(sector), start, stop,
                )
            trend[patched] = 1.0
            detrended[start:stop] = run_flux / trend
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
    flux_err: Optional[Sequence[float]] = None,
) -> Tuple[BLSSearchResult, List[Dict[str, Any]]]:
    """Run the shared BLS duration-grid search with survey-style arguments.

    Args:
        time_btjd: Observation times in BTJD_TDB days.
        flux: Normalized relative flux values.
        duration_grid_hours: Trial transit durations in hours.
        period_min_days: Inclusive lower search-period bound in days.
        period_max_days: Inclusive upper search-period bound in days.
        n_periods: Requested trial-grid density.
        flux_err: Optional positive normalized-flux uncertainties.

    Returns:
        The best BLS result and JSON-safe records for all duration trials.

    Notes:
        This wrapper preserves one shared search implementation; ranking
        statistics remain configuration-specific diagnostic quantities.
    """
    return find_transits_duration_grid(
        time_btjd,
        flux,
        duration_grid_hours,
        period_min=period_min_days,
        period_max=period_max_days,
        n_periods=n_periods,
        flux_err=flux_err,
    )


def inverted_flux(flux: Sequence[float]) -> np.ndarray:
    """Reflect normalized flux about unity for an inverted-event control.

    Args:
        flux: Normalized relative flux values centered near unity.

    Returns:
        A float array in which dimmings become brightenings and vice versa.

    Notes:
        An inverted search is a null-control comparison, not a calibrated
        false-alarm probability.
    """
    values = np.asarray(flux, dtype=float)
    return 2.0 - values


def scrambled_flux(
    flux: Sequence[float], seed: int, sectors: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Create a deterministic null control while retaining sector structure.

    Without labels, this performs the historical full-series permutation used
    by standalone tests. With labels, it applies an independent nonzero
    circular shift within each sector, preserving cadence ordering and local
    noise correlation while disrupting coherent cross-sector phase alignment.

    Args:
        flux: Normalized relative flux values.
        seed: Seed used for deterministic control construction.
        sectors: Optional sector label for every cadence.

    Returns:
        A flux array aligned with the original cadence order.

    Raises:
        ValueError: If supplied sector labels do not match flux shape.
    """
    values = np.asarray(flux, dtype=float)
    generator = np.random.default_rng(seed)
    if sectors is None:
        return generator.permutation(values)
    labels = np.asarray(sectors, dtype=int)
    if values.shape != labels.shape:
        raise ValueError("flux and sectors must have matching shapes")
    shifted = values.copy()
    for sector in np.unique(labels):
        indices = np.flatnonzero(labels == sector)
        if indices.size < 2:
            continue
        offset = int(generator.integers(1, indices.size))
        shifted[indices] = np.roll(values[indices], offset)
    return shifted


def _finite_exposure_days(
    time_btjd: Sequence[float], exposure_days: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Return one positive finite integration time for every cadence.

    Explicit exposure values are preferred. When a generic array caller does
    not have product metadata, a robust median positive cadence spacing is a
    conservative proxy: it keeps an injection from spanning a data gap, unlike
    using neighboring timestamps as an implicit exposure window would do.
    """
    time = np.asarray(time_btjd, dtype=float)
    if time.ndim != 1 or time.size == 0 or not np.all(np.isfinite(time)):
        raise ValueError("injection times must be a non-empty finite one-dimensional array")
    if exposure_days is not None:
        exposure = np.asarray(exposure_days, dtype=float)
        if exposure.ndim == 0:
            exposure = np.full(time.size, float(exposure), dtype=float)
        if exposure.shape != time.shape:
            raise ValueError("exposure_days must be a scalar or match the time shape")
        if not np.all(np.isfinite(exposure) & (exposure > 0)):
            raise ValueError("exposure_days must contain only positive finite values")
        return exposure

    ordered = np.sort(time)
    intervals = np.diff(ordered)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if intervals.size == 0:
        raise ValueError("injection times must contain more than one distinct cadence")
    cadence = float(np.median(intervals))
    return np.full(time.size, cadence, dtype=float)


def _box_exposure_fraction(
    time_btjd: Sequence[float],
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
    exposure_days: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Integrate a periodic box over each finite cadence exposure."""
    time = np.asarray(time_btjd, dtype=float)
    exposure = _finite_exposure_days(time, exposure_days)
    duration_days = float(duration_hours) / 24.0
    if np.any(exposure >= period_days - duration_days):
        raise ValueError("cadence exposure must be shorter than the out-of-transit interval")
    event_numbers = np.rint((time - epoch_btjd) / period_days).astype(int)
    event_centers = epoch_btjd + event_numbers * period_days
    starts = time - 0.5 * exposure
    ends = time + 0.5 * exposure
    transit_starts = event_centers - 0.5 * duration_days
    transit_ends = event_centers + 0.5 * duration_days
    overlap = np.maximum(0.0, np.minimum(ends, transit_ends) - np.maximum(starts, transit_starts))
    return np.clip(overlap / exposure, 0.0, 1.0)


def inject_box_transit(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
    depth_ppm: float,
    exposure_days: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Inject a finite-exposure-integrated box signal into normalized flux.

    The model is deliberately a box, not a limb-darkened transit. Each cadence
    receives its finite-exposure overlap fraction so injection and recovery do
    not treat every observation as an instantaneous sample.

    Args:
        time_btjd: Observation times in BTJD_TDB days.
        flux: Normalized relative flux values.
        period_days: Positive injected period in days.
        epoch_btjd: Injected central epoch in BTJD_TDB days.
        duration_hours: Positive injected box duration in hours.
        depth_ppm: Positive injected relative depth in ppm.
        exposure_days: Optional positive integration time for each cadence in
            days; a robust cadence estimate is used when omitted.

    Returns:
        A new normalized-flux array containing the injected box signal.

    Raises:
        ValueError: If inputs have incompatible shapes or an injected physical
            scale is non-finite, non-positive, or longer than its period.
    """
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
    if duration_hours / 24.0 >= period_days:
        raise ValueError("injection duration must be shorter than its period")
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float).copy()
    if time.shape != values.shape:
        raise ValueError("time and flux must have matching shapes")
    fraction = _box_exposure_fraction(
        time, period_days, epoch_btjd, duration_hours, exposure_days=exposure_days
    )
    values *= 1.0 - depth_ppm * 1e-6 * fraction
    return values


def mask_box_transit(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
    exposure_days: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, int]:
    """Mask a declared box event before testing an injected replacement.

    Masking prevents a recovery search from crediting a pre-existing periodic
    peak instead of the injected signal. The BLS loader later removes NaNs
    while retaining the original cadence sampling and gaps.

    Args:
        time_btjd: Observation times in BTJD_TDB days.
        flux: Normalized relative flux values.
        period_days: Positive period of the event to mask in days.
        epoch_btjd: Central epoch of the event to mask in BTJD_TDB days.
        duration_hours: Positive box duration to mask in hours.
        exposure_days: Optional positive integration time for each cadence in
            days.

    Returns:
        A copied flux array with overlapping cadences set to NaN and the
        number of masked cadences.

    Raises:
        ValueError: If arrays are incompatible or mask ephemeris values are
            invalid.
    """
    if (
        not np.isfinite(period_days)
        or not np.isfinite(epoch_btjd)
        or not np.isfinite(duration_hours)
        or period_days <= 0
        or duration_hours <= 0
    ):
        raise ValueError("masked period and duration must be positive and finite")
    if duration_hours / 24.0 >= period_days:
        raise ValueError("masked duration must be shorter than its period")
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    if time.shape != values.shape:
        raise ValueError("time and flux must have matching shapes")
    in_transit = _box_exposure_fraction(
        time, period_days, epoch_btjd, duration_hours, exposure_days=exposure_days
    ) > 0.0
    masked = values.copy()
    masked[in_transit] = np.nan
    return masked, int(np.count_nonzero(in_transit))


def recovered_period(injected_period_days: float, recovered_period_days: float, tolerance: float) -> bool:
    """Test fractional agreement between injected and recovered periods.

    Args:
        injected_period_days: Positive injected period in days.
        recovered_period_days: Positive recovered period in days.
        tolerance: Positive fractional agreement tolerance.

    Returns:
        True when the recovered-to-injected period ratio lies within tolerance
        of unity.

    Raises:
        ValueError: If either period or the tolerance is non-finite or
            non-positive.
    """
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
    """Test phase-wrapped epoch agreement against an injected event.

    Args:
        injected_period_days: Positive injected period in days.
        injected_epoch_btjd: Injected central epoch in BTJD_TDB days.
        recovered_epoch_btjd: Recovered central epoch in BTJD_TDB days.
        tolerance_hours: Positive phase-wrapped agreement window in hours.

    Returns:
        True when the wrapped epoch offset is no greater than tolerance_hours.

    Raises:
        ValueError: If periods, epochs, or tolerance are non-finite, or if the
            period or tolerance is non-positive.
    """
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
    flux_err: Optional[Sequence[float]] = None,
    gap_break_window_fraction: float = 0.5,
) -> Dict[str, Any]:
    """Run frozen search variants and null controls for one light curve.

    The routine compares normalized and per-sector running-median branches,
    then runs inverted-flux and sector-preserving scramble controls through
    the same duration-grid search. This makes preprocessing sensitivity and
    common null peaks inspectable together.

    Args:
        time_btjd: Observation times in BTJD_TDB days.
        flux: Normalized relative flux values.
        sectors: Sector label for every cadence.
        duration_grid_hours: Trial transit durations in hours.
        period_min_days: Inclusive lower trial-period bound in days.
        period_max_days: Inclusive upper trial-period bound in days.
        n_periods: Requested number of trial periods.
        detrend_window_days: Positive running-median window in days.
        scramble_seeds: Deterministic seeds for sector-local null controls.
        flux_err: Optional positive normalized-flux uncertainties.
        gap_break_window_fraction: Positive fraction used to split large gaps.

    Returns:
        JSON-safe results for preprocessing variants and null controls.

    Raises:
        ValueError: If uncertainty or sector arrays are incompatible with flux,
            or detrending controls are invalid.

    Notes:
        The maximum control statistic is diagnostic evidence only; it is not a
        population false-alarm probability or validation result.
    """
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    labels = np.asarray(sectors, dtype=int)
    errors = None if flux_err is None else np.asarray(flux_err, dtype=float)
    if errors is not None and errors.shape != values.shape:
        raise ValueError("flux_err must match the time and flux shapes")
    def preprocess(branch: str, raw_flux: np.ndarray) -> np.ndarray:
        if branch == "normalized":
            return raw_flux
        return detrend_by_sector(
            time,
            raw_flux,
            labels,
            detrend_window_days,
            gap_break_window_fraction=gap_break_window_fraction,
        )

    variants = {
        "normalized": preprocess("normalized", values),
        "running-median": preprocess("running-median", values),
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
            flux_err=errors,
        )
        variant_results[name] = {"best": best.to_dict(), "trials": trials}

    branch_controls: Dict[str, Dict[str, Any]] = {}
    inverted_values = inverted_flux(values)
    for name in variants:
        inverted, _ = search_duration_grid(
            time,
            preprocess(name, inverted_values),
            duration_grid_hours,
            period_min_days,
            period_max_days,
            n_periods,
            flux_err=errors,
        )
        scrambled_results = []
        for seed in scramble_seeds:
            result, _ = search_duration_grid(
                time,
                preprocess(name, scrambled_flux(values, int(seed), sectors=labels)),
                duration_grid_hours,
                period_min_days,
                period_max_days,
                n_periods,
                flux_err=errors,
            )
            scrambled_results.append({"seed": int(seed), "best": result.to_dict()})
        branch_snr_raw = [inverted.snr] + [entry["best"]["snr"] for entry in scrambled_results]
        branch_snr = [value for value in branch_snr_raw if value is not None and np.isfinite(value)]
        if not branch_snr:
            branch_snr = [0.0]
        branch_controls[name] = {
            "inverted": inverted.to_dict(),
            "scrambles": scrambled_results,
            "max_snr": float(max(branch_snr)),
        }
    normalized_controls = branch_controls["normalized"]
    null_snr = [entry["max_snr"] for entry in branch_controls.values()]
    return {
        "variants": variant_results,
        "controls": {
            "inverted": normalized_controls["inverted"],
            "scrambles": normalized_controls["scrambles"],
            "by_variant": branch_controls,
            "scramble_method": "independent-sector-circular-shift",
            "max_snr": float(max(null_snr)),
        },
    }


def _non_detection_best_record(
    best: Optional[BLSSearchResult], attempted_period_trials: int
) -> Dict[str, Any]:
    """Serialize a no-detection result without discarding BLS grid metadata."""
    if best is None:
        payload = BLSSearchResult(
            best_period=None,
            best_epoch=None,
            best_depth_ppm=None,
            best_duration_hours=None,
            snr=None,
            n_distinct_transit_events=0,
            n_period_trials=int(attempted_period_trials),
            detection_status="no-detection",
        ).to_dict()
    else:
        payload = best.to_dict()
        # The caller reached this path because the result cannot support a
        # recovery decision, regardless of a stale/default status label.
        payload["detection_status"] = "no-detection"
    return payload


def _is_recoverable_detection(best: Optional[BLSSearchResult]) -> bool:
    """Return whether a search result carries finite detected ephemeris values."""
    if best is None or getattr(best, "detection_status", None) != "detected":
        return False
    try:
        period_days = float(best.best_period)
        epoch_btjd = float(best.best_epoch)
        snr = float(best.snr)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(period_days)
        and period_days > 0.0
        and np.isfinite(epoch_btjd)
        and np.isfinite(snr)
    )


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
    sectors: Optional[Sequence[int]] = None,
    detrend_window_days: Optional[float] = None,
    flux_err: Optional[Sequence[float]] = None,
    exposure_days: Optional[Sequence[float]] = None,
    gap_break_window_fraction: float = 0.5,
) -> List[Dict[str, Any]]:
    """Measure recovery of declared synthetic box signals in observed flux.

    A recovered trial must agree in period and phase-wrapped epoch. When a
    pre-registered minimum SNR is supplied, it must also clear that threshold
    and contain multiple distinct transit events. With sectors and a detrending
    window, each injection must recover from both normalized and per-sector
    running-median branches.

    Args:
        time_btjd: Observation times in BTJD_TDB days.
        flux: Observed normalized relative flux values.
        injections: Declared box-signal mappings with period, duration, depth,
            and optional epoch values in documented units.
        duration_grid_hours: BLS trial durations in hours.
        period_min_days: Inclusive lower trial-period bound in days.
        period_max_days: Inclusive upper trial-period bound in days.
        n_periods: Requested number of BLS period trials.
        tolerance: Positive fractional period-recovery tolerance.
        minimum_snr: Optional pre-registered BLS SNR requirement.
        epoch_tolerance_duration_fraction: Positive multiple of injected
            duration used for the epoch-recovery window.
        sectors: Optional sector label for every cadence.
        detrend_window_days: Required positive running-median window when
            sectors are supplied.
        flux_err: Optional positive normalized-flux uncertainties.
        exposure_days: Optional positive cadence integration times in days.
        gap_break_window_fraction: Positive gap-splitting fraction for the
            running-median branch.

    Returns:
        One JSON-safe recovery record per declared injection, including
        branch-level results when preprocessing is requested.

    Raises:
        ValueError: If input arrays, injection controls, or optional
            preprocessing arguments are incompatible.

    Notes:
        These outcomes characterize only the declared grid and processing
        branches. They do not estimate population completeness, a selection
        function, or scientific disposition.
    """
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    errors = None if flux_err is None else np.asarray(flux_err, dtype=float)
    if time.size == 0:
        raise ValueError("injection recovery requires at least one cadence")
    if errors is not None and errors.shape != values.shape:
        raise ValueError("flux_err must match the time and flux shapes")
    exposures = _finite_exposure_days(time, exposure_days)
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
    if (sectors is None) != (detrend_window_days is None):
        raise ValueError("sectors and detrend_window_days must be supplied together")
    labels: Optional[np.ndarray] = None
    if sectors is not None:
        labels = np.asarray(sectors, dtype=int)
        if labels.shape != time.shape:
            raise ValueError("sectors must match the time and flux shapes")
        if not np.isfinite(detrend_window_days) or float(detrend_window_days) <= 0:
            raise ValueError("detrend_window_days must be positive and finite")
    epoch = float(np.nanmin(time))
    results: List[Dict[str, Any]] = []
    for injection in injections:
        injected_epoch = float(injection.get("epoch_btjd", epoch))
        injected_period = float(injection["period_days"])
        injected_duration = float(injection["duration_hours"])
        injected_depth = float(injection["depth_ppm"])
        # DIAGNOSTIC_REASONING: Inject before optional detrending so the
        # alternate branch cannot receive an unrealistically preserved signal.
        injected = inject_box_transit(
            time,
            values,
            injected_period,
            injected_epoch,
            injected_duration,
            injected_depth,
            exposure_days=exposures,
        )
        epoch_tolerance_hours = float(injected_duration * epoch_tolerance_duration_fraction)
        branch_fluxes = {"normalized": injected}
        if labels is not None:
            branch_fluxes["running-median"] = detrend_by_sector(
                time,
                injected,
                labels,
                float(detrend_window_days),
                gap_break_window_fraction=gap_break_window_fraction,
            )
        branch_results: Dict[str, Dict[str, Any]] = {}
        for branch_name, branch_flux in branch_fluxes.items():
            best, _ = search_duration_grid(
                time,
                branch_flux,
                duration_grid_hours,
                period_min_days,
                period_max_days,
                n_periods,
                flux_err=errors,
            )
            if not _is_recoverable_detection(best):
                branch_results[branch_name] = {
                    "period_match": False,
                    "epoch_match": False,
                    "snr_pass": False,
                    "recovered": False,
                    "best": _non_detection_best_record(best, n_periods),
                }
                continue
            period_match = bool(recovered_period(injected_period, best.best_period, tolerance))
            epoch_match = bool(
                recovered_epoch(
                    injected_period, injected_epoch, best.best_epoch, epoch_tolerance_hours
                )
            )
            snr_pass = bool(
                (minimum_snr is None or best.snr >= minimum_snr)
                and best.n_distinct_transit_events >= 2
            )
            branch_results[branch_name] = {
                "period_match": period_match,
                "epoch_match": epoch_match,
                "snr_pass": snr_pass,
                "recovered": bool(period_match and epoch_match and snr_pass),
                "best": best.to_dict(),
            }
        normalized = branch_results["normalized"]
        branch_recovered = list(branch_results.values())
        declared_injection = dict(injection)
        declared_injection["epoch_btjd"] = injected_epoch
        result: Dict[str, Any] = {
            "injection": declared_injection,
            "period_match": bool(all(entry["period_match"] for entry in branch_recovered)),
            "epoch_match": bool(all(entry["epoch_match"] for entry in branch_recovered)),
            "snr_pass": bool(all(entry["snr_pass"] for entry in branch_recovered)),
            "epoch_tolerance_hours": epoch_tolerance_hours,
            "recovered": bool(all(entry["recovered"] for entry in branch_recovered)),
            "best": normalized["best"],
        }
        if labels is not None:
            result["branches"] = branch_results
        results.append(result)
    return results
