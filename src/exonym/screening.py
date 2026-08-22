"""Fixed-ephemeris photometric screening suite for candidate transit verification.

Applies deterministic physical checks on candidate light curves at a declared ephemeris (P, T_0, T_14)
to rapidly identify eclipsing binaries, centroid contaminants, and period aliases before
computationally expensive MCMC or TRICERATOPS runs:

1. Odd-Even Depth Consistency Check:
   Computes depth significance difference:
       z = |delta_odd - delta_even| / sqrt(sigma_odd^2 + sigma_even^2)
   Flags candidates with z >= 3.0 sigma as probable Eclipsing Binaries (EB) whose true
   orbital period is doubled (P_true = 2 * P_trial).

2. Half-Phase Secondary Eclipse Screen (phi = 0.5):
   Measures flux deficit at the anti-transit phase to detect companion occultations (depth_sec).
   Significant secondary eclipses (> 3 sigma) rule out planetary nature unless proven thermal emission.

3. Doubled-Period Screen (2 * P):
   Evaluates alternating primary and secondary events at double the declared period.

Note:
    Produces a screening diagnostic artifact only, NOT a statistical validation claim.
    Scatter-based uncertainties do not fully model correlated red noise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .inputs import BTJD_TIME_SYSTEM, load_light_curve_table, load_transit_ephemeris
from .workspace import validate_signal_suffix
from .lightcurve import phase_hours
from .workspace import CandidateWorkspace

MIN_SAMPLES_PER_WINDOW = 3                      # Minimum in-transit cadences required for depth evaluation
ODD_EVEN_CONSISTENCY_THRESHOLD_SIGMA = 3.0      # Significance threshold (sigma) for odd-even depth difference
SCREENING_MAX_POINTS_PER_PRODUCT = 12000        # Maximum cadences loaded per product to bound memory
OUT_OF_TRANSIT_INNER_DURATIONS = 1.2            # Inner buffer boundary for out-of-transit baseline
OUT_OF_TRANSIT_OUTER_DURATIONS = 2.5            # Outer buffer boundary for out-of-transit baseline
DOUBLE_PERIOD_HARMONIC_MULTIPLIER = 2.0         # Period multiplier for harmonic sub-harmonic check


def _finite_float(value: object) -> Optional[float]:
    """Return a finite float or ``None`` without serializing NaNs to JSON."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _require_positive_finite(name: str, value: object) -> float:
    """Validate a positive physical scale read from a candidate workspace."""
    number = _finite_float(value)
    if number is None or number <= 0.0:
        raise ValueError("{0} must be positive and finite".format(name))
    return number


def _depth_measurement(
    flux: np.ndarray, in_transit: np.ndarray, out_of_transit: np.ndarray
) -> Dict[str, Any]:
    """Estimate a median depth and scatter-based uncertainty in ppm."""
    in_values = flux[in_transit]
    out_values = flux[out_of_transit]
    measurement: Dict[str, Any] = {
        "n_in_transit": int(in_values.size),
        "n_out_of_transit": int(out_values.size),
        "depth_ppm": None,
        "uncertainty_ppm": None,
        "depth_significance_sigma": None,
        "status": "insufficient_coverage",
    }
    if (
        in_values.size < MIN_SAMPLES_PER_WINDOW
        or out_values.size < MIN_SAMPLES_PER_WINDOW
    ):
        return measurement

    depth = float(np.median(out_values) - np.median(in_values)) * 1e6
    uncertainty = float(
        math.hypot(
            1.253 * float(np.std(in_values)) / math.sqrt(in_values.size),
            1.253 * float(np.std(out_values)) / math.sqrt(out_values.size),
        )
        * 1e6
    )
    measurement["depth_ppm"] = _finite_float(depth)
    measurement["uncertainty_ppm"] = _finite_float(uncertainty)
    if measurement["depth_ppm"] is None:
        measurement["status"] = "nonfinite_measurement"
        return measurement
    if measurement["uncertainty_ppm"] is None or measurement["uncertainty_ppm"] <= 0.0:
        measurement["uncertainty_ppm"] = None
        measurement["status"] = "uncertainty_unavailable"
        return measurement

    measurement["depth_significance_sigma"] = _finite_float(
        measurement["depth_ppm"] / measurement["uncertainty_ppm"]
    )
    measurement["status"] = "measured"
    return measurement


def _window_measurement(
    time_btjd: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
    cycle_selection: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Measure a transit-shaped window, optionally in a parity subset."""
    hours = phase_hours(time_btjd, period_days, epoch_btjd)
    valid = np.isfinite(hours) & np.isfinite(flux)
    if cycle_selection is not None:
        if cycle_selection.shape != valid.shape:
            raise ValueError("cycle_selection must match time and flux shapes")
        valid &= cycle_selection
    in_transit = valid & (np.abs(hours) < 0.5 * duration_hours)
    out_of_transit = valid & (np.abs(hours) > OUT_OF_TRANSIT_INNER_DURATIONS * duration_hours)
    out_of_transit &= np.abs(hours) < OUT_OF_TRANSIT_OUTER_DURATIONS * duration_hours
    return _depth_measurement(flux, in_transit, out_of_transit)


def _odd_even_summary(odd: Dict[str, Any], even: Dict[str, Any]) -> Dict[str, Any]:
    """Return a guarded odd-even depth comparison with no validation inference."""
    odd_depth = _finite_float(odd.get("depth_ppm"))
    even_depth = _finite_float(even.get("depth_ppm"))
    odd_error = _finite_float(odd.get("uncertainty_ppm"))
    even_error = _finite_float(even.get("uncertainty_ppm"))
    result: Dict[str, Any] = {
        "parity_definition": "nearest declared ephemeris epoch is even",
        "odd": odd,
        "even": even,
        "z": None,
        "consistency_threshold_sigma": ODD_EVEN_CONSISTENCY_THRESHOLD_SIGMA,
        "consistent_at_threshold": None,
        "status": "unresolved",
    }
    if (
        odd_depth is None
        or even_depth is None
        or odd_error is None
        or even_error is None
        or odd_error <= 0.0
        or even_error <= 0.0
    ):
        return result

    denominator = math.hypot(odd_error, even_error)
    if not math.isfinite(denominator) or denominator <= 0.0:
        return result
    z_value = abs(odd_depth - even_depth) / denominator
    z = _finite_float(z_value)
    if z is None:
        return result
    result["z"] = z
    result["consistent_at_threshold"] = z < ODD_EVEN_CONSISTENCY_THRESHOLD_SIGMA
    result["status"] = "measured"
    return result


def fixed_ephemeris_screen(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
) -> Dict[str, Any]:
    """Screen primary, odd/even, harmonic, and half-period windows.

    Inputs are normalized relative flux and BTJD times.  This function is
    deliberately deterministic and target-neutral so that the result can be
    re-run from the candidate-owned light curve and declared ephemeris.
    """
    period = _require_positive_finite("period_days", period_days)
    duration = _require_positive_finite("duration_hours", duration_hours)
    epoch = _finite_float(epoch_btjd)
    if epoch is None:
        raise ValueError("epoch_btjd must be finite")

    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    if time.shape != values.shape or time.ndim != 1:
        raise ValueError("time_btjd and flux must be one-dimensional arrays of equal shape")

    finite_time = np.isfinite(time)
    cycles = np.zeros(time.shape, dtype=np.int64)
    cycles[finite_time] = np.floor((time[finite_time] - epoch) / period + 0.5).astype(
        np.int64
    )
    even_cycles = finite_time & ((cycles % 2) == 0)
    odd_cycles = finite_time & ((cycles % 2) != 0)

    primary = _window_measurement(time, values, period, epoch, duration)
    odd = _window_measurement(time, values, period, epoch, duration, odd_cycles)
    even = _window_measurement(time, values, period, epoch, duration, even_cycles)
    half_phase = _window_measurement(time, values, period, epoch + 0.5 * period, duration)
    double_period = period * DOUBLE_PERIOD_HARMONIC_MULTIPLIER
    double_primary = _window_measurement(time, values, double_period, epoch, duration)
    alternating_event = _window_measurement(
        time,
        values,
        double_period,
        epoch + 0.5 * double_period,
        duration,
    )

    return {
        "primary": primary,
        "odd_even": _odd_even_summary(odd, even),
        "half_phase_control": half_phase,
        "double_period_hypothesis": {
            "period_days": double_period,
            "primary": double_primary,
            "alternating_event": alternating_event,
            "interpretation": (
                "diagnostic only: expresses alternating declared-period events "
                "as primary and half-phase events at twice the period; it does "
                "not identify an eclipsing binary or validate a planet"
            ),
        },
    }


def _rounded_payload(value: object, digits: int = 6) -> object:
    """Recursively make a JSON-safe, concise payload without nonfinite values."""
    if isinstance(value, dict):
        return {str(key): _rounded_payload(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded_payload(item, digits) for item in value]
    if isinstance(value, (float, np.floating)):
        number = _finite_float(value)
        return None if number is None else round(number, digits)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    return value


def run_fixed_ephemeris_screen(
    workspace: CandidateWorkspace,
    signal: Optional[str] = None,
    detrending_method: Optional[str] = None,
) -> Path:
    """Write a candidate-local fixed-ephemeris screening artifact.

    A named signal requires its own readable candidate configuration.  Unlike
    exploratory commands, this workflow refuses a synthetic ephemeris so that
    no science-looking screen can be produced from demonstration defaults.
    """
    validate_signal_suffix(signal)
    ephemeris = load_transit_ephemeris(workspace, signal=signal)
    if signal is not None and ephemeris.get("source") != "candidate-config-signal":
        raise ValueError(
            "no readable signal prior at config/signals/transit_config{0}.json".format(signal)
        )
    required_fields = ("period_days", "epoch_btjd", "duration_days", "depth_ppm")
    field_sources = ephemeris.get("field_sources", {})
    if not isinstance(field_sources, dict) or any(
        field_sources.get(field) in (None, "synthetic-demo") for field in required_fields
    ):
        raise ValueError("fixed-ephemeris screening requires complete candidate-derived ephemeris data")
    if ephemeris.get("time_system") != BTJD_TIME_SYSTEM:
        raise ValueError("fixed-ephemeris screening requires a BTJD_TDB epoch")

    table = load_light_curve_table(
        workspace,
        max_points=SCREENING_MAX_POINTS_PER_PRODUCT,
        require_raw_provenance=True,
        detrending_method=detrending_method,
    )
    if table is None:
        raise ValueError("fixed-ephemeris screening requires a readable candidate light curve")

    period = _require_positive_finite("period_days", ephemeris.get("period_days"))
    duration_days = _require_positive_finite("duration_days", ephemeris.get("duration_days"))
    _require_positive_finite("depth_ppm", ephemeris.get("depth_ppm"))
    epoch = _finite_float(ephemeris.get("epoch_btjd"))
    if epoch is None:
        raise ValueError("epoch_btjd must be finite")
    duration_hours = duration_days * 24.0
    result = fixed_ephemeris_screen(table["time"], table["flux"], period, epoch, duration_hours)

    sectors = np.asarray(table.get("sector", []), dtype=int)
    payload = {
        "method": "fixed_ephemeris_photometric_screen",
        "interpretation": (
            "screening only: scatter-based depth estimates are not a false-positive "
            "probability or validation claim"
        ),
        "candidate_id": workspace.candidate_id,
        "signal": signal,
        "source": "candidate-data",
        "preprocessing": table.get("detrending", {"kind": "pipeline-normalization"}),
        "n_points": int(np.asarray(table["time"]).size),
        "sectors": sorted({int(value) for value in sectors if int(value) > 0}),
        "ephemeris": {
            "period_days": period,
            "epoch_btjd": epoch,
            "epoch_time_system": BTJD_TIME_SYSTEM,
            "duration_hours": duration_hours,
            "source": ephemeris.get("source"),
        },
        "screen": result,
    }
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output = outputs_dir / "fixed_ephemeris_screen{0}.json".format(signal or "")
    output.write_text(
        json.dumps(_rounded_payload(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
