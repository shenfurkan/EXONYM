"""Target-neutral transit search engine supporting BLS and native TLS.

Implements blind and targeted periodic transit detection algorithms across photometric
time-series without hardcoded candidate designations or ephemerides:

1. Box Least Squares (BLS) Search (Kovács, Zucker & Mazeh 2002):
   Uses Astropy's weighted ``BoxLeastSquares`` implementation to fit periodic
   step functions (top-hat boxes) defined by:
   - Trial period P in [P_min, P_max] days
   - Fractional transit duration q = T_14 / P
   - Transit epoch / center time T_0 (BTJD)
   - Transit depth delta = <y_out> - <y_in>
   - A fitted depth and formal uncertainty. The reported ``snr`` is their
     ratio, not a calibrated false-alarm probability or detection reliability.

2. Grid Resolution:
   - Astropy's baseline-and-duration-aware frequency grid prevents a requested
     sparse scan from under-resolving a multi-sector light curve.

3. Optional Transit Least Squares (TLS) (Hippke & Heller 2019):
   Integrates realistic physical limb-darkened transit shapes (Mandel & Agol 2002) with ingress/egress
   morphology, yielding higher sensitivity for shallow small-planet transits.
"""

from __future__ import annotations

import json
import re
import hashlib
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lightcurve import phase_hours
from .workspace import CandidateWorkspace, validate_signal_suffix


@dataclass
class BLSSearchResult:
    """Standardized result container for periodic transit searches."""

    best_period: float          # Optimal orbital period (days)
    best_epoch: float           # Optimal transit epoch (BTJD)
    best_depth_ppm: float       # Optimal transit depth (parts per million)
    best_duration_hours: float  # Optimal total transit duration T_14 (hours)
    snr: float                  # Fitted depth / formal BLS depth uncertainty
    n_distinct_transit_events: int
    n_period_trials: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_period": float(self.best_period),
            "best_epoch": float(self.best_epoch),
            "best_depth_ppm": float(self.best_depth_ppm),
            "best_duration_hours": float(self.best_duration_hours),
            "snr": float(self.snr),
            "n_distinct_transit_events": int(self.n_distinct_transit_events),
            "n_period_trials": int(self.n_period_trials),
        }


def _frequency_period_grid(period_min: float, period_max: float, n_periods: int) -> np.ndarray:
    """Return trial periods uniformly sampled in orbital frequency.

    A uniform period grid under-resolves long-period signals on a multi-sector
    baseline.  Keeping the number of samples fixed in frequency gives the
    periodogram approximately constant phase drift resolution instead.  The
    returned periods decrease from ``period_max`` to ``period_min``; callers
    must not infer a ranking from their order.
    """
    if isinstance(n_periods, bool) or not isinstance(n_periods, (int, np.integer)):
        raise ValueError("n_periods must be an integer of at least two")
    if n_periods < 2:
        raise ValueError("n_periods must be an integer of at least two")
    frequencies = np.linspace(1.0 / period_max, 1.0 / period_min, int(n_periods))
    return 1.0 / frequencies


def _uncertainties_for_bls(
    values: np.ndarray, flux_err: Optional[Sequence[float]]
) -> np.ndarray:
    """Return finite positive per-cadence uncertainties for weighted BLS.

    Candidate-product loading normally supplies reported uncertainties. Public
    array callers may omit them, in which case a robust constant scatter is
    used solely to retain a dimensionless ranking statistic; the
    candidate-facing runner records that fallback.
    """
    if flux_err is not None:
        errors = np.asarray(flux_err, dtype=float)
        if errors.shape != values.shape:
            raise ValueError("flux_err must match the time and flux shapes")
        if not np.all(np.isfinite(errors) & (errors > 0)):
            raise ValueError("flux_err must contain only positive finite values")
        return errors

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scatter = 1.4826 * mad
    if not np.isfinite(scatter) or scatter <= 0:
        scatter = 1e-4
    return np.full(values.size, scatter, dtype=float)


def _baseline_aware_frequency_factor(
    time: np.ndarray,
    period_min: float,
    period_max: float,
    duration_days: float,
    requested_minimum_trials: int,
) -> float:
    """Select an Astropy BLS frequency factor without under-resolving a grid.

    ``BoxLeastSquares.autopower`` uses a duration/baseline-squared frequency
    scale. A caller can request *more* samples through ``n_periods`` but never
    a coarser-than-standard scan, preventing a fixed sparse grid from missing
    narrow, long-baseline signals.
    """
    baseline_days = float(np.ptp(time))
    if not np.isfinite(baseline_days) or baseline_days <= 0:
        raise ValueError("BLS requires observations spanning a positive time baseline")
    natural_step = duration_days / (baseline_days * baseline_days)
    frequency_span = 1.0 / period_min - 1.0 / period_max
    natural_trials = max(2, int(np.ceil(frequency_span / natural_step)) + 1)
    return min(1.0, natural_trials / float(requested_minimum_trials))


def _distinct_transit_events(
    time: np.ndarray, period: float, epoch: float, duration_hours: float
) -> int:
    """Count observed event windows containing at least one cadence."""
    in_transit = np.abs(phase_hours(time, period, epoch)) <= 0.5 * duration_hours
    if not np.any(in_transit):
        return 0
    event_numbers = np.rint((time[in_transit] - epoch) / period).astype(int)
    return int(np.unique(event_numbers).size)


def find_transits(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    period_min: float = 0.5,
    period_max: float = 15.0,
    n_periods: int = 2000,
    duration_hours: float = 3.0,
    flux_err: Optional[Sequence[float]] = None,
) -> BLSSearchResult:
    """Run a target-neutral BLS periodogram search over a light curve.

    Returns the optimal (period, epoch, depth_ppm, duration_hours, score).
    ``snr`` is the fitted weighted BLS depth divided by its formal uncertainty.
    It is retained as a compatibility field name only and is not a calibrated
    detection significance, false-alarm probability, or reliability estimate.

    The trial grid is generated by Astropy using the observed time baseline and
    requested transit duration. ``n_periods`` is treated as a minimum requested
    density: it can add samples but cannot make the standard BLS grid coarser.
    A selected peak must contain at least two observed transit events.
    """
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)

    if time.shape != values.shape:
        raise ValueError("time and flux must have matching shapes")
    finite = np.isfinite(time) & np.isfinite(values)
    if flux_err is not None:
        raw_errors = np.asarray(flux_err, dtype=float)
        if raw_errors.shape != values.shape:
            raise ValueError("flux_err must match the time and flux shapes")
        finite &= np.isfinite(raw_errors) & (raw_errors > 0)
        raw_errors = raw_errors[finite]
    else:
        raw_errors = None
    time = time[finite]
    values = values[finite]

    if time.size < 50:
        raise ValueError("insufficient data points for BLS transit search")
    if period_min <= 0 or period_max <= period_min:
        raise ValueError("invalid period search bounds")
    _frequency_period_grid(period_min, period_max, n_periods)

    try:
        from astropy.timeseries import BoxLeastSquares
    except ImportError as exc:  # pragma: no cover - declared core dependency
        raise RuntimeError("BLS search requires the core astropy dependency") from exc

    duration_days = float(duration_hours) / 24.0
    if not np.isfinite(duration_days) or duration_days <= 0:
        raise ValueError("duration_hours must be positive and finite")
    errors = _uncertainties_for_bls(values, raw_errors)
    frequency_factor = _baseline_aware_frequency_factor(
        time, period_min, period_max, duration_days, n_periods
    )
    periodogram = BoxLeastSquares(time, values, dy=errors).autopower(
        duration_days,
        objective="likelihood",
        method="fast",
        minimum_n_transit=2,
        minimum_period=period_min,
        maximum_period=period_max,
        frequency_factor=frequency_factor,
    )
    valid = (
        np.isfinite(periodogram.power)
        & np.isfinite(periodogram.period)
        & np.isfinite(periodogram.transit_time)
        & np.isfinite(periodogram.depth)
        & np.isfinite(periodogram.depth_err)
        & (periodogram.depth > 0)
        & (periodogram.depth_err > 0)
    )
    best: Optional[Dict[str, float]] = None
    for index in np.argsort(periodogram.power)[::-1]:
        if not valid[index]:
            continue
        period = float(periodogram.period[index])
        epoch = float(periodogram.transit_time[index])
        n_events = _distinct_transit_events(time, period, epoch, duration_hours)
        if n_events < 2:
            continue
        depth = float(periodogram.depth[index])
        depth_err = float(periodogram.depth_err[index])
        best = {
            "period": period,
            "epoch": epoch,
            "depth": depth,
            "snr": depth / depth_err,
            "n_events": n_events,
        }
        break
    if best is None:
        return BLSSearchResult(
            best_period=round(period_min, 5),
            best_epoch=round(float(np.min(time)), 5),
            best_depth_ppm=0.0,
            best_duration_hours=round(duration_hours, 2),
            snr=0.0,
            n_distinct_transit_events=0,
            n_period_trials=0,
        )

    return BLSSearchResult(
        best_period=round(best["period"], 5),
        best_epoch=round(best["epoch"], 5),
        best_depth_ppm=round(best["depth"] * 1e6, 2),
        best_duration_hours=round(duration_hours, 2),
        snr=round(max(best["snr"], 0.0), 2),
        n_distinct_transit_events=int(best["n_events"]),
        n_period_trials=int(periodogram.period.size),
    )


def find_transits_duration_grid(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    duration_grid_hours: Sequence[float],
    period_min: float = 0.5,
    period_max: float = 15.0,
    n_periods: int = 2000,
    flux_err: Optional[Sequence[float]] = None,
) -> Tuple[BLSSearchResult, List[Dict[str, Any]]]:
    """Run the declared BLS duration grid and retain every trial result.

    The returned best result is selected by the same uncalibrated ranking
    statistic as an individual BLS search.  This is a deterministic model
    selection step, not an additional significance calibration.
    """
    durations = [float(value) for value in duration_grid_hours]
    if not durations or any(not np.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("duration_grid_hours must contain positive finite values")
    if len(set(durations)) != len(durations):
        raise ValueError("duration_grid_hours must not contain duplicates")

    results: List[Tuple[BLSSearchResult, Dict[str, Any]]] = []
    for duration_hours in durations:
        result = find_transits(
            time_btjd,
            flux,
            flux_err=flux_err,
            period_min=period_min,
            period_max=period_max,
            n_periods=n_periods,
            duration_hours=duration_hours,
        )
        results.append((result, result.to_dict()))
    best, _ = max(results, key=lambda item: item[0].snr)
    return best, [payload for _, payload in results]


def find_transits_tls(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    flux_err: Sequence[float],
    period_min: float = 0.5,
    period_max: float = 15.0,
    use_threads: int = 1,
) -> Dict[str, float]:
    """Run a weighted, native-cadence Transit Least Squares search.

    Args:
        time_btjd: Observation times in BTJD.
        flux: Normalized flux values.
        flux_err: Per-cadence normalized flux uncertainties.
        period_min: Minimum searched orbital period in days.
        period_max: Maximum searched orbital period in days.
        use_threads: TLS worker count. The default of one avoids TLS's
            multiprocessing path, which is unreliable in constrained Windows
            shells.

    Returns:
        TLS best period, epoch, depth, duration, and SDE. This is a discovery
        statistic, not a planetary-validation result.
    """
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(flux_err, dtype=float)
    finite = np.isfinite(time) & np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    time = time[finite]
    values = values[finite]
    errors = errors[finite]
    if time.size < 50:
        raise ValueError("insufficient data points for TLS transit search")
    if period_min <= 0 or period_max <= period_min:
        raise ValueError("invalid period search bounds")
    if use_threads < 1:
        raise ValueError("TLS use_threads must be at least one")

    try:
        from transitleastsquares import transitleastsquares
    except ImportError as exc:
        raise RuntimeError(
            "TLS search requires the optional 'discovery' dependency group"
        ) from exc

    result = transitleastsquares(time, values, errors, verbose=False).power(
        period_min=period_min,
        period_max=period_max,
        show_progress_bar=False,
        use_threads=use_threads,
    )
    bottom_flux = float(result.depth)
    depth_relative = 1.0 - bottom_flux
    values_to_check = (
        float(result.period),
        float(result.T0),
        float(result.duration),
        float(result.SDE),
        bottom_flux,
    )
    if not np.all(np.isfinite(values_to_check)) or not 0.0 < depth_relative < 1.0:
        raise RuntimeError("TLS did not return a physical transit solution")
    return {
        "best_period": float(result.period),
        "best_epoch": float(result.T0),
        "best_depth_ppm": depth_relative * 1e6,
        "best_duration_hours": float(result.duration) * 24.0,
        "sde": float(result.SDE),
    }


def _median_bin(time: np.ndarray, flux: np.ndarray, n_bins: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """Median-bin a time-sorted light curve down to at most n_bins samples."""
    if time.size <= n_bins:
        return time, flux
    order = np.argsort(time)
    time_sorted = time[order]
    flux_sorted = flux[order]
    edges = np.linspace(0, time_sorted.size, n_bins + 1).astype(int)
    bin_times = np.empty(n_bins, dtype=float)
    bin_flux = np.empty(n_bins, dtype=float)
    for index in range(n_bins):
        start, stop = edges[index], edges[index + 1]
        if stop > start:
            bin_times[index] = np.mean(time_sorted[start:stop])
            bin_flux[index] = np.median(flux_sorted[start:stop])
    return bin_times, bin_flux


def load_candidate_light_curve(
    workspace: CandidateWorkspace,
    max_points: int = 4000,
    sectors: Optional[Sequence[int]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (time_btjd, normalized_flux) from candidate FITS data, or None.

    Products are read from ``data/processed/`` first, then ``data/raw/``.
    Multiple products are concatenated (per-sector binning) so multi-sector
    baselines are searched jointly. Returns None when no readable FITS light
    curve with at least 50 points exists.
    """
    from .inputs import load_light_curve_table

    table = load_light_curve_table(workspace, max_points=max_points, sectors=sectors)
    if table is None:
        return None
    time = np.asarray(table["time"], dtype=float)
    flux = np.asarray(table["flux"], dtype=float)
    if time.size < 50 or time.size != flux.size:
        return None
    return time, flux


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for a candidate-local input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bls_runtime_provenance() -> Dict[str, str]:
    """Return the exact core BLS implementation and installed package version."""
    try:
        version = importlib.metadata.version("astropy")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - core dependency
        version = "unknown"
    return {
        "implementation": "astropy.timeseries.BoxLeastSquares",
        "package": "astropy",
        "version": version,
    }


def _input_manifest_records(
    workspace: CandidateWorkspace, sectors: Optional[Sequence[int]] = None
) -> List[Dict[str, Any]]:
    """Describe the exact light-curve products selected by the input loader."""
    from .inputs import load_light_curve_table

    table = load_light_curve_table(workspace, sectors=sectors)
    if table is None:
        return []

    records: List[Dict[str, Any]] = []
    for path in table.get("input_files", []):
        product_path = Path(path)
        sidecar_path = product_path.with_name(product_path.stem + ".provenance.json")
        provenance: Optional[Dict[str, Any]] = None
        if sidecar_path.is_file():
            try:
                provenance = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                provenance = None
        records.append(
            {
                "path": product_path.relative_to(workspace.path).as_posix(),
                "sha256": _sha256(product_path),
                "provenance_path": (
                    sidecar_path.relative_to(workspace.path).as_posix()
                    if sidecar_path.is_file()
                    else None
                ),
                "provenance": provenance,
            }
        )
    return records


def run_bls_on_candidate(
    workspace: CandidateWorkspace,
    period_min: float = 0.5,
    period_max: float = 15.0,
    n_periods: int = 2000,
    signal: Optional[str] = None,
    engine: str = "bls",
    sectors: Optional[Sequence[int]] = None,
    result_suffix: Optional[str] = None,
    duration_grid_hours: Optional[Sequence[float]] = None,
) -> Path:
    """Run BLS transit search on candidate data and save JSON summary to outputs/.

    If ``signal`` is provided, the search reads the matching per-signal prior,
    uses its duration, and restricts the period grid to +/- 0.1 days around
    its period. Targeted runs are written to
    ``outputs/bls_search_results<signal>.json`` so independent signals cannot
    overwrite one another. A run without ``signal`` retains the historical
    ``outputs/bls_search_results.json`` path and behavior.

    Candidate searches require real, readable light-curve photometry.
    """
    if engine not in ("bls", "tls"):
        raise ValueError("search engine must be 'bls' or 'tls'")
    if duration_grid_hours is not None and engine != "bls":
        raise ValueError("duration_grid_hours is supported only by BLS searches")
    if engine == "bls":
        _frequency_period_grid(period_min, period_max, n_periods)

    duration_grid: Optional[List[float]] = None
    if duration_grid_hours is not None:
        duration_grid = [float(value) for value in duration_grid_hours]
        if not duration_grid or any(not np.isfinite(value) or value <= 0 for value in duration_grid):
            raise ValueError("duration_grid_hours must contain positive finite values")
        if len(set(duration_grid)) != len(duration_grid):
            raise ValueError("duration_grid_hours must not contain duplicates")

    duration_hours: Optional[float] = None
    signal_provenance: Optional[Dict[str, Any]] = None
    if signal is not None:
        validate_signal_suffix(signal)
        if duration_grid is not None:
            raise ValueError("duration_grid_hours cannot be combined with a targeted signal search")

        from .inputs import load_transit_ephemeris

        ephem = load_transit_ephemeris(workspace, signal=signal)
        if ephem.get("source") != "candidate-config-signal":
            raise ValueError(
                "no readable signal prior at config/signals/transit_config{0}.json".format(
                    signal
                )
            )

        prior_p = float(ephem["period_days"])
        duration_hours = float(ephem["duration_days"]) * 24.0
        if not np.isfinite(prior_p) or prior_p <= 0:
            raise ValueError("signal prior period_days must be positive and finite")
        if not np.isfinite(duration_hours) or duration_hours <= 0:
            raise ValueError("signal prior duration must be positive and finite")

        period_min = max(0.5, prior_p - 0.1)
        period_max = prior_p + 0.1
        if period_max <= period_min:
            raise ValueError("signal prior period is below the supported BLS range")
        signal_provenance = {
            "mode": "targeted-prior",
            "signal": signal,
            "prior_path": "config/signals/transit_config{0}.json".format(signal),
            "prior_source": ephem["source"],
            "prior_period_days": prior_p,
            "prior_epoch_btjd": float(ephem["epoch_btjd"]),
            "prior_duration_hours": duration_hours,
            "period_min_days": period_min,
            "period_max_days": period_max,
        }

    if result_suffix is not None:
        if signal is not None:
            raise ValueError("result_suffix cannot be combined with a signal search")
        if not re.fullmatch(r"\.[a-z0-9][a-z0-9-]*", result_suffix):
            raise ValueError("result_suffix must use the .label format")

    tls_errors: Optional[np.ndarray] = None
    bls_errors: Optional[np.ndarray] = None
    bls_error_sources: Optional[List[str]] = None
    if engine == "tls":
        from .inputs import load_light_curve_table

        native_table = load_light_curve_table(workspace, max_points=None, sectors=sectors)
        loaded = None
        if native_table is not None:
            loaded = (
                np.asarray(native_table["time"], dtype=float),
                np.asarray(native_table["flux"], dtype=float),
            )
            tls_errors = np.asarray(native_table["flux_err"], dtype=float)
    else:
        loaded = load_candidate_light_curve(workspace, sectors=sectors)
        if loaded is not None:
            # Keep the compact public loader compatible while acquiring the
            # per-cadence uncertainties needed by the weighted BLS engine.
            # A test or third-party caller may provide only (time, flux); in
            # that case find_transits uses its explicit robust-scatter fallback.
            from .inputs import load_light_curve_table

            bls_table = load_light_curve_table(workspace, sectors=sectors)
            if bls_table is not None:
                candidate_time = np.asarray(bls_table["time"], dtype=float)
                candidate_errors = np.asarray(bls_table["flux_err"], dtype=float)
                if candidate_time.shape == loaded[0].shape and np.array_equal(candidate_time, loaded[0]):
                    bls_errors = candidate_errors
                    bls_error_sources = list(bls_table.get("flux_err_sources", []))
    if loaded is None:
        raise ValueError("no readable candidate light-curve photometry available for BLS transit search")
    time, flux = loaded
    source = "candidate-data"
    input_records = _input_manifest_records(workspace, sectors=sectors)

    if engine == "tls":
        if tls_errors is None:
            raise ValueError("TLS transit search requires per-cadence flux uncertainties")
        payload = find_transits_tls(
            time,
            flux,
            tls_errors,
            period_min=period_min,
            period_max=period_max,
        )
    else:
        search_kwargs: Dict[str, Any] = {
            "period_min": period_min,
            "period_max": period_max,
            "n_periods": n_periods,
        }
        duration_trials: Optional[List[Dict[str, Any]]] = None
        if duration_grid is not None:
            result, duration_trials = find_transits_duration_grid(
                time,
                flux,
                duration_grid,
                period_min=period_min,
                period_max=period_max,
                n_periods=n_periods,
                flux_err=bls_errors,
            )
            payload = result.to_dict()
        else:
            if duration_hours is not None:
                search_kwargs["duration_hours"] = duration_hours
            payload = find_transits(time, flux, flux_err=bls_errors, **search_kwargs).to_dict()
        if duration_trials is not None:
            payload["duration_grid_trials"] = duration_trials
    payload["source"] = source
    payload["n_points"] = int(time.size)
    payload["statistic"] = {
        "name": (
            "weighted BLS fitted-depth signal-to-noise"
            if engine == "bls"
            else "TLS signal detection efficiency"
        ),
        "value_field": "snr" if engine == "bls" else "sde",
        "calibrated_false_alarm_probability": None,
        "population_detection_reliability": None,
        "scientific_use": "ranking and human-review triage only",
    }
    if engine == "bls":
        payload["statistic"]["uncertainty_source"] = (
            bls_error_sources if bls_errors is not None else ["robust-scatter-fallback"]
        )
    if signal_provenance is not None:
        payload["signal"] = signal
        payload["search_provenance"] = signal_provenance

    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_name = "{0}_search_results{1}.json".format(engine, signal or result_suffix or "")
    output_path = outputs_dir / output_name
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    manifest: Dict[str, Any] = {
        "schema": "exonym-{0}-search-manifest-1".format(engine),
        "candidate_id": workspace.candidate_id,
        "result_path": output_path.relative_to(workspace.path).as_posix(),
        "source": source,
        "inputs": input_records,
        "configuration": {
            "period_min_days": period_min,
            "period_max_days": period_max,
            "duration_hours": duration_hours if duration_grid is None else None,
            "duration_grid_hours": duration_grid,
            "n_periods": n_periods if engine == "bls" else None,
            "n_periods_role": (
                "minimum requested trial density; Astropy baseline-duration grid may use more"
                if engine == "bls"
                else None
            ),
            "period_grid": "astropy-autopower-baseline-duration-resolved" if engine == "bls" else None,
            "max_points": 4000 if engine == "bls" else None,
            "quality_filter": "quality == 0 when available",
            "normalization": "lightkurve.remove_nans().normalize()",
            "binning": (
                "per-product median binning; no global rebinning"
                if engine == "bls"
                else "none; native cadence"
            ),
            "signal": signal,
            "engine": engine,
            "cadence": "native" if engine == "tls" else "median-binned",
            "use_threads": 1 if engine == "tls" else None,
            "uncertainty_source": bls_error_sources if engine == "bls" else None,
            "sectors": list(sectors) if sectors is not None else None,
        },
        "search_statistic": payload["statistic"],
        "runtime": _bls_runtime_provenance() if engine == "bls" else None,
    }
    if signal_provenance is not None:
        prior_path = workspace.path / signal_provenance["prior_path"]
        manifest["targeted_prior"] = {
            "path": signal_provenance["prior_path"],
            "sha256": _sha256(prior_path),
            "search_provenance": signal_provenance,
        }
    manifest_path = outputs_dir / "{0}_search_manifest{1}.json".format(
        engine, signal or result_suffix or ""
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_path


def calculate_ttv_super_period(
    period_inner_days: float,
    period_outer_days: float,
    j_resonance: int = 2,
) -> float:
    """Return TTV super-period P_ttv in days for a j:j-1 resonance."""
    if period_inner_days <= 0 or period_outer_days <= period_inner_days:
        raise ValueError("periods must satisfy 0 < P_inner < P_outer")
    if j_resonance <= 1:
        raise ValueError("j_resonance must be an integer >= 2")
    freq_inner = j_resonance / period_outer_days
    freq_outer = (j_resonance - 1) / period_inner_days
    delta_freq = abs(freq_inner - freq_outer)
    if delta_freq == 0:
        return float("inf")
    return 1.0 / delta_freq


def compute_linear_ephemeris_residuals(
    transit_times_btjd: Sequence[float],
    period_days: float,
    epoch_btjd: float,
) -> List[float]:
    """Return list of (O - C) TTV timing residuals in minutes."""
    if period_days <= 0:
        raise ValueError("period_days must be positive")
    residuals_min = []
    for t_obs in transit_times_btjd:
        n_epoch = round((float(t_obs) - float(epoch_btjd)) / float(period_days))
        t_calc = float(epoch_btjd) + n_epoch * float(period_days)
        omc_days = float(t_obs) - t_calc
        residuals_min.append(round(omc_days * 1440.0, 4))
    return residuals_min
