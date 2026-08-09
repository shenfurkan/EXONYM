"""Target-neutral transit timing variation (O-C) engine.

Fits individual transit epoch central times against a fixed batman transit
template, computes (O - C) timing residuals in minutes, renders a timing
diagram, and evaluates resonant companion super-periods when a companion
period is declared in the candidate config.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .inputs import load_light_curve_table, load_stellar_parameters, load_transit_ephemeris
from .lightcurve import kipping_to_quadratic_limb_darkening
from .search import calculate_ttv_super_period
from .transit_fit import stellar_density_a_rs
from .workspace import CandidateWorkspace

MIN_POINTS_PER_TRANSIT = 30
WINDOW_DAYS = 0.35
GRID_HALF_WINDOW_DAYS = 0.02
GRID_STEP_DAYS = 0.001


def transit_template_parameters(
    ephemeris: Dict[str, Any], a_rs: float
) -> Dict[str, Any]:
    """Build fixed template parameters for per-transit epoch fitting."""
    rp_rs = math.sqrt(max(float(ephemeris["depth_ppm"]) * 1e-6, 1e-8))
    u1, u2 = kipping_to_quadratic_limb_darkening(0.3, 0.3)
    return {
        "period_days": float(ephemeris["period_days"]),
        "rp_rs": rp_rs,
        "a_rs": a_rs,
        "impact_parameter": 0.3,
        "u1": u1,
        "u2": u2,
    }


def _template_flux(
    template: Dict[str, Any], time: np.ndarray, t0_value: float
) -> Optional[np.ndarray]:
    """Evaluate the batman template with the transit center shifted to t0."""
    try:
        import batman

        params = batman.TransitParams()
        params.t0 = float(t0_value)
        params.per = template["period_days"]
        params.rp = template["rp_rs"]
        params.a = template["a_rs"]
        params.inc = math.degrees(
            math.acos(template["impact_parameter"] / template["a_rs"])
        )
        params.ecc = 0.0
        params.w = 90.0
        params.u = [template["u1"], template["u2"]]
        params.limb_dark = "quadratic"
        model = batman.TransitModel(params, np.asarray(time, dtype=float))
        return np.asarray(model.light_curve(params), dtype=float)
    except Exception:
        return None


def fit_transit_epoch(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    template: Dict[str, Any],
    t0_expected: float,
    window_days: float = WINDOW_DAYS,
    grid_half_window_days: float = GRID_HALF_WINDOW_DAYS,
    grid_step_days: float = GRID_STEP_DAYS,
) -> Optional[Tuple[float, float]]:
    """Fit one transit epoch by grid search plus parabolic refinement.

    Returns (t0_fit, sigma_t0) in days or None when the transit is not
    measurable in the window.
    """
    mask = (time > t0_expected - window_days) & (time < t0_expected + window_days)
    t_window = time[mask]
    f_window = flux[mask]
    e_window = flux_err[mask]
    if t_window.size < MIN_POINTS_PER_TRANSIT:
        return None

    def chi2(t0_trial: float) -> float:
        model = _template_flux(template, t_window, t0_trial)
        if model is None:
            return 1e100
        return float(np.sum(((f_window - model) / e_window) ** 2))

    trials = np.arange(
        t0_expected - grid_half_window_days,
        t0_expected + grid_half_window_days + grid_step_days,
        grid_step_days,
    )
    values = np.array([chi2(trial) for trial in trials])
    best_index = int(np.argmin(values))
    t0_fit = float(trials[best_index])
    if values[best_index] >= 1e99:
        return None

    eps = grid_step_days
    if 0 < best_index < len(trials) - 1:
        a = values[best_index - 1]
        b = values[best_index]
        c = values[best_index + 1]
        denominator = a - 2.0 * b + c
        if abs(denominator) > 1e-12:
            t0_fit = t0_fit + 0.5 * eps * (a - c) / denominator
    curvature = (chi2(t0_fit + eps) - 2.0 * chi2(t0_fit) + chi2(t0_fit - eps)) / (eps**2)
    sigma_t0 = math.sqrt(2.0 / curvature) if curvature > 0 else grid_step_days
    sigma_t0 = float(np.clip(sigma_t0, 0.0005, 0.05))
    return t0_fit, sigma_t0


def transit_timing_analysis(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    ephemeris: Dict[str, Any],
    a_rs: float,
    window_days: float = WINDOW_DAYS,
) -> Dict[str, Any]:
    """Fit every measurable transit epoch and return the O-C report."""
    period_days = float(ephemeris["period_days"])
    t0_reference = float(ephemeris["epoch_btjd"])
    template = transit_template_parameters(ephemeris, a_rs)
    n_min = int(np.floor((np.min(time) - t0_reference) / period_days))
    n_max = int(np.ceil((np.max(time) - t0_reference) / period_days))

    epochs: List[int] = []
    t_observed: List[float] = []
    t_calculated: List[float] = []
    t_errors: List[float] = []
    for epoch in range(n_min, n_max + 1):
        t_expected = t0_reference + epoch * period_days
        fit = fit_transit_epoch(
            time, flux, flux_err, template, t_expected, window_days=window_days
        )
        if fit is None:
            continue
        t0_fit, sigma_t0 = fit
        epochs.append(epoch)
        t_observed.append(t0_fit)
        t_calculated.append(t_expected)
        t_errors.append(sigma_t0)

    epochs_arr = np.asarray(epochs, dtype=int)
    t_observed_arr = np.asarray(t_observed, dtype=float)
    t_calculated_arr = np.asarray(t_calculated, dtype=float)
    t_errors_arr = np.asarray(t_errors, dtype=float)
    oc_minutes = (t_observed_arr - t_calculated_arr) * 1440.0
    oc_errors_minutes = t_errors_arr * 1440.0
    rms_oc = float(np.sqrt(np.mean(oc_minutes**2))) if oc_minutes.size else None
    mean_uncertainty = float(np.mean(oc_errors_minutes)) if oc_errors_minutes.size else None
    return {
        "epochs": [int(epoch) for epoch in epochs_arr],
        "t_observed_btjd": [float(value) for value in t_observed_arr],
        "t_calculated_btjd": [float(value) for value in t_calculated_arr],
        "oc_minutes": [float(value) for value in oc_minutes],
        "oc_error_minutes": [float(value) for value in oc_errors_minutes],
        "oc_rms_minutes": rms_oc,
        "mean_uncertainty_minutes": mean_uncertainty,
        "n_transits_fit": int(epochs_arr.size),
    }


def plot_timing_diagram(
    epochs: Sequence[int],
    oc_minutes: Sequence[float],
    oc_errors_minutes: Sequence[float],
    output_path: Path,
) -> Path:
    """Render the O-C timing diagram to a PNG file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        list(epochs),
        list(oc_minutes),
        yerr=list(oc_errors_minutes),
        fmt="o",
        color="gray",
        markeredgecolor="black",
        capsize=3,
        alpha=0.6,
    )
    ax.axhline(0.0, color="red", linestyle="--")
    ax.set_xlabel("Transit Epoch N")
    ax.set_ylabel("O-C (minutes)")
    ax.set_title("Transit Timing Diagram")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _companion_periods(workspace: CandidateWorkspace) -> List[float]:
    """Return declared companion orbital periods from the candidate config."""
    periods: List[float] = []
    for config_name in ("transit_config.json", "ephemeris.json"):
        config_path = workspace.path / "config" / config_name
        if not config_path.is_file():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        declared = payload.get("ttv_companion_period_days")
        if isinstance(declared, (int, float)) and not isinstance(declared, bool):
            if float(declared) > 0:
                periods.append(float(declared))
        companions = payload.get("companions")
        if isinstance(companions, list):
            for companion in companions:
                if not isinstance(companion, dict):
                    continue
                for key in ("period_days", "period"):
                    value = companion.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if float(value) > 0:
                            periods.append(float(value))
                        break
    return sorted(set(round(value, 6) for value in periods))


def _synthetic_timing_table(
    ttv_amplitude_minutes: float = 0.0,
    rng_seed: int = 17,
) -> Dict[str, np.ndarray]:
    """Deterministic demonstration light curve with injected transits.

    Transits are generated with the same batman template used by the fitter,
    optionally with a sinusoidal per-epoch TTV shift.
    """
    rng = np.random.default_rng(seed=rng_seed)
    demo_period_days = 3.5
    demo_epoch_btjd = 2.0
    depth_ppm = 2500.0
    ttv_cycles = 6
    cadence_days = 20.0 / 1440.0
    time = np.arange(0.0, 35.0, cadence_days)
    rho_solar = 1.0
    a_rs = stellar_density_a_rs(rho_solar, demo_period_days)
    ephemeris = {
        "period_days": demo_period_days,
        "epoch_btjd": demo_epoch_btjd,
        "duration_days": 0.12,
        "depth_ppm": depth_ppm,
    }
    template = transit_template_parameters(ephemeris, a_rs)
    ttv_amplitude_days = ttv_amplitude_minutes / 1440.0
    n_min = int(np.floor((np.min(time) - demo_epoch_btjd) / demo_period_days))
    n_max = int(np.ceil((np.max(time) - demo_epoch_btjd) / demo_period_days))
    flux = np.ones_like(time)
    for epoch in range(n_min, n_max + 1):
        t0_epoch = demo_epoch_btjd + epoch * demo_period_days
        shift = ttv_amplitude_days * math.sin(2.0 * np.pi * epoch / ttv_cycles)
        mask = (time > t0_epoch - 0.35) & (time < t0_epoch + 0.35)
        model = _template_flux(template, time[mask], t0_epoch + shift)
        if model is not None:
            flux[mask] = model
    flux = flux + rng.normal(0.0, 250e-6, size=time.shape)
    flux_err = np.full_like(flux, 250e-6)
    sector_values = np.ones(time.size, dtype=int)
    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "sector": sector_values,
        "_period_days": demo_period_days,
        "_epoch_btjd": demo_epoch_btjd,
        "_duration_days": 0.12,
        "_depth_ppm": depth_ppm,
    }


def run_ttv_analysis(workspace: CandidateWorkspace, signal: Optional[str] = None) -> Path:
    """Run the TTV analysis and write outputs/ttv_analysis_results.json."""
    outputs_dir = workspace.path / "outputs"
    figures_dir = workspace.path / "figures"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = load_light_curve_table(workspace)
    if table is None:
        table = _synthetic_timing_table()
        source = "synthetic-demo"
        ephemeris = {
            "period_days": table.pop("_period_days"),
            "epoch_btjd": table.pop("_epoch_btjd"),
            "duration_days": table.pop("_duration_days"),
            "depth_ppm": table.pop("_depth_ppm"),
            "source": source,
        }
    else:
        source = "candidate-data"
        ephemeris = load_transit_ephemeris(workspace, signal=signal)

    stellar = load_stellar_parameters(workspace)
    rho_prior_solar = float(stellar["mass_solar"]) / float(stellar["radius_solar"]) ** 3
    a_rs = stellar_density_a_rs(rho_prior_solar, ephemeris["period_days"])

    analysis = transit_timing_analysis(
        table["time"], table["flux"], table["flux_err"], ephemeris, a_rs
    )
    timing_diagram = figures_dir / "ttv_timing_diagram.png"
    if analysis["n_transits_fit"] > 0:
        plot_timing_diagram(
            analysis["epochs"],
            analysis["oc_minutes"],
            analysis["oc_error_minutes"],
            timing_diagram,
        )
    else:
        timing_diagram = None

    companion_periods = _companion_periods(workspace)
    inner_period = ephemeris["period_days"]
    super_periods = []
    for outer_period in companion_periods:
        if outer_period <= inner_period:
            continue
        for resonance in (2, 3, 4):
            try:
                super_periods.append(
                    {
                        "companion_period_days": round(outer_period, 6),
                        "resonance_j": int(resonance),
                        "super_period_days": round(
                            calculate_ttv_super_period(
                                inner_period, outer_period, j_resonance=resonance
                            ),
                            4,
                        ),
                    }
                )
            except ValueError:
                continue

    payload = {
        "schema_version": "1.0",
        "work_package": "TTV_ANALYSIS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "ephemeris": {
            "period_days": ephemeris["period_days"],
            "epoch_btjd": ephemeris["epoch_btjd"],
            "source": ephemeris["source"],
        },
        "timing": {
            "n_transits_fit": analysis["n_transits_fit"],
            "oc_rms_minutes": analysis["oc_rms_minutes"],
            "mean_uncertainty_minutes": analysis["mean_uncertainty_minutes"],
            "epochs": analysis["epochs"],
            "t_observed_btjd": analysis["t_observed_btjd"],
            "t_calculated_btjd": analysis["t_calculated_btjd"],
            "oc_minutes": analysis["oc_minutes"],
            "oc_error_minutes": analysis["oc_error_minutes"],
        },
        "companion_super_periods": super_periods,
        "timing_diagram": (
            str(timing_diagram.relative_to(workspace.path)).replace("\\", "/")
            if timing_diagram is not None
            else None
        ),
        "caveat": (
            "Per-transit timing in SNR-limited cadences is noise-dominated; "
            "no TTV detection is claimed without a significance threshold."
        ),
    }
    output_path = outputs_dir / "ttv_analysis_results.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
