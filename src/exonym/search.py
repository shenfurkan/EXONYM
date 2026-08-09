"""Target-neutral Box Least Squares (BLS) transit search engine.

Search routines evaluate transit candidate periodograms across light curves without
hardcoding target designations or ephemerides.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lightcurve import phase_hours
from .workspace import CandidateWorkspace


@dataclass
class BLSSearchResult:
    best_period: float
    best_epoch: float
    best_depth_ppm: float
    best_duration_hours: float
    snr: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_period": float(self.best_period),
            "best_epoch": float(self.best_epoch),
            "best_depth_ppm": float(self.best_depth_ppm),
            "best_duration_hours": float(self.best_duration_hours),
            "snr": float(self.snr),
        }


def _epoch_score(
    time: np.ndarray,
    values: np.ndarray,
    period: float,
    epoch: float,
    duration_hours: float,
) -> Optional[Dict[str, float]]:
    """Score one (period, epoch) trial; returns depth/snr counts or None."""
    ph = phase_hours(time, period, epoch)
    in_transit = np.abs(ph) <= 0.5 * duration_hours
    out_transit = (np.abs(ph) > 1.0 * duration_hours) & (np.abs(ph) < 3.0 * duration_hours)

    in_vals = values[in_transit]
    out_vals = values[out_transit]
    if in_vals.size < 1 or out_vals.size < 3:
        return None

    depth = float(np.median(out_vals) - np.median(in_vals))
    std_out = np.std(out_vals) if np.std(out_vals) > 1e-8 else 1e-4
    n_eff = (in_vals.size * out_vals.size) / (in_vals.size + out_vals.size)
    snr = (depth * np.sqrt(n_eff)) / std_out
    return {"depth": depth, "snr": snr, "n_in": int(in_vals.size), "n_out": int(out_vals.size)}


def _epoch_trial_count(period: float, duration_hours: float, cap: int = 500) -> int:
    """Trial epochs dense enough that spacing never exceeds half the duration."""
    return max(8, min(int(np.ceil(period / (duration_hours / 48.0))), cap))


def find_transits(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    period_min: float = 0.5,
    period_max: float = 15.0,
    n_periods: int = 2000,
    duration_hours: float = 3.0,
) -> BLSSearchResult:
    """Run a target-neutral BLS periodogram search over a light curve.

    Returns the optimal (period, epoch, depth_ppm, duration_hours, snr).

    The search is two-pass: a coarse period grid locates the strongest peak,
    then a fine refinement re-scans around it so long baselines are not
    smeared by grid quantization. Integer-multiple aliases are resolved by
    testing for additional transits at fractional phase offsets: when a
    folded period k*P shows transits at every j/k offset, the fundamental
    period P is adopted instead.
    """
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)

    finite = np.isfinite(time) & np.isfinite(values)
    time = time[finite]
    values = values[finite]

    if time.size < 50:
        raise ValueError("insufficient data points for BLS transit search")
    if period_min <= 0 or period_max <= period_min:
        raise ValueError("invalid period search bounds")

    t_min = float(np.min(time))
    coarse_periods = np.linspace(period_min, period_max, n_periods)
    grid_step = (period_max - period_min) / max(n_periods - 1, 1)

    def scan(period_values: np.ndarray) -> Optional[Dict[str, float]]:
        best: Optional[Dict[str, float]] = None
        for p in period_values:
            n_epochs = _epoch_trial_count(p, duration_hours)
            for trial_epoch in np.linspace(t_min, t_min + p, n_epochs):
                scored = _epoch_score(time, values, float(p), float(trial_epoch), duration_hours)
                if scored is None:
                    continue
                if best is None or scored["snr"] > best["snr"]:
                    best = {"period": float(p), "epoch": float(trial_epoch), **scored}
        return best

    best = scan(coarse_periods)
    if best is None:
        return BLSSearchResult(
            best_period=round(period_min, 5),
            best_epoch=round(t_min, 5),
            best_depth_ppm=0.0,
            best_duration_hours=round(duration_hours, 2),
            snr=0.0,
        )

    def refine(center: float) -> Optional[Dict[str, float]]:
        local = np.linspace(center - 2.0 * grid_step, center + 2.0 * grid_step, 241)
        local = local[(local >= period_min) & (local <= period_max)]
        if local.size == 0:
            return None
        return scan(local)

    refined = refine(best["period"])
    if refined is not None and refined["snr"] >= best["snr"]:
        best = refined

    for _ in range(4):
        divided = False
        for divisor in (2, 3):
            sub_period = best["period"] / divisor
            if sub_period < period_min:
                continue
            offset_ok = True
            for j in range(1, divisor):
                offset_score = _epoch_score(
                    time, values, best["period"], best["epoch"] + j * sub_period, duration_hours
                )
                if (
                    offset_score is None
                    or offset_score["depth"] <= 0.5 * best["depth"]
                    or offset_score["snr"] < 5.0
                ):
                    offset_ok = False
                    break
            if not offset_ok:
                continue
            sub_refined = refine(sub_period)
            if sub_refined is not None and sub_refined["snr"] >= 0.8 * best["snr"]:
                best = sub_refined
                divided = True
                break
        if not divided:
            break

    return BLSSearchResult(
        best_period=round(best["period"], 5),
        best_epoch=round(best["epoch"], 5),
        best_depth_ppm=round(best["depth"] * 1e6, 2),
        best_duration_hours=round(duration_hours, 2),
        snr=round(max(best["snr"], 0.0), 2),
    )


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
    workspace: CandidateWorkspace, max_points: int = 4000
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (time_btjd, normalized_flux) from candidate FITS data, or None.

    Products are read from ``data/processed/`` first, then ``data/raw/``.
    Multiple products are concatenated (per-sector binning) so multi-sector
    baselines are searched jointly. Returns None when no readable FITS light
    curve with at least 50 points exists, so callers can fall back to a
    synthetic demonstration grid.
    """
    from .inputs import load_light_curve_table

    table = load_light_curve_table(workspace, max_points=max_points)
    if table is None:
        return None
    time = np.asarray(table["time"], dtype=float)
    flux = np.asarray(table["flux"], dtype=float)
    if time.size < 50 or time.size != flux.size:
        return None
    return time, flux


def run_bls_on_candidate(
    workspace: CandidateWorkspace,
    period_min: float = 0.5,
    period_max: float = 15.0,
    signal: Optional[str] = None,
) -> Path:
    """Run BLS transit search on candidate data and save JSON summary to outputs/.

    If ``signal`` is provided, the search reads the matching per-signal prior,
    uses its duration, and restricts the period grid to +/- 0.1 days around
    its period. Targeted runs are written to
    ``outputs/bls_search_results<signal>.json`` so independent signals cannot
    overwrite one another. A run without ``signal`` retains the historical
    ``outputs/bls_search_results.json`` path and behavior.

    Real candidate light curves are used when present; otherwise a synthetic
    demonstration grid is analyzed and the payload is marked ``source``.
    """
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    duration_hours: Optional[float] = None
    signal_provenance: Optional[Dict[str, Any]] = None
    if signal is not None:
        if not re.fullmatch(r"\.\d+", signal):
            raise ValueError("signal must use the .NN format")

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

    loaded = load_candidate_light_curve(workspace)
    if loaded is None:
        time = np.linspace(0, 30, 1000)
        flux = 1.0 - 0.001 * (np.abs((time - 2.0) % 3.5) < 0.05).astype(float)
        source = "synthetic-demo"
    else:
        time, flux = loaded
        source = "candidate-data"

    search_kwargs: Dict[str, float] = {
        "period_min": period_min,
        "period_max": period_max,
    }
    if duration_hours is not None:
        search_kwargs["duration_hours"] = duration_hours
    result = find_transits(time, flux, **search_kwargs)
    payload = result.to_dict()
    payload["source"] = source
    payload["n_points"] = int(time.size)
    if signal_provenance is not None:
        payload["signal"] = signal
        payload["search_provenance"] = signal_provenance

    output_name = "bls_search_results{0}.json".format(signal or "")
    output_path = outputs_dir / output_name
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
