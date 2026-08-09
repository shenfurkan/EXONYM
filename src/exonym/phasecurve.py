"""Target-neutral phase curve and secondary eclipse engine.

Decomposes out-of-transit light curves into reflection/emission, Doppler
beaming, ellipsoidal, and harmonic control components via weighted linear
regression with a block-clustered sandwich covariance, and reports the
significance of a fixed-phase secondary eclipse box.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .inputs import load_light_curve_table, load_transit_ephemeris
from .lightcurve import phase_hours
from .workspace import CandidateWorkspace

PRIMARY_MASK_HALF_DURATIONS = 0.65
BLOCK_DAYS = 0.5

PHYSICAL_COMPONENTS = (
    "reflection_semiamplitude",
    "beaming_semiamplitude",
    "ellipsoidal_semiamplitude",
    "second_harmonic_sine_control",
    "secondary_eclipse_depth",
)


def cluster_sandwich_covariance(
    design: np.ndarray,
    residual: np.ndarray,
    sigma: np.ndarray,
    cluster: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Return a finite-sample-corrected cluster-sandwich covariance.

    The bread matrix uses a pseudo-inverse so collinear design columns (e.g.
    duplicate sector offsets when the same TESS sector appears in more than
    one product) do not raise a singular-matrix error. With fewer than two
    clusters the finite-sample correction degrades to the HC1 form.
    """
    weighted_design = design / sigma[:, None]
    weighted_residual = residual / sigma
    bread = np.linalg.pinv(weighted_design.T @ weighted_design)
    meat = np.zeros((design.shape[1], design.shape[1]))
    groups = np.unique(cluster)
    for group in groups:
        mask = cluster == group
        score = weighted_design[mask].T @ weighted_residual[mask]
        meat += np.outer(score, score)
    n_points, n_params = design.shape
    n_groups = len(groups)
    if n_groups < 2 or n_points <= n_params:
        correction = float(n_points) / max(n_points - n_params, 1)
    else:
        correction = n_groups / (n_groups - 1.0) * (n_points - 1.0) / (n_points - n_params)
    return correction * bread @ meat @ bread, n_groups


def build_design_matrix(
    time: np.ndarray,
    phase_days: np.ndarray,
    period_days: float,
    duration_days: float,
    sector_values: np.ndarray,
    block_days: float = BLOCK_DAYS,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Build the phase-curve regression design matrix.

    Columns are per-sector offsets and linear slopes, the four orbital harmonic
    components, and a fixed secondary-eclipse box at phase 0.5. Returns
    (design, names, cluster) with cluster grouping 0.5-day blocks per sector.
    """
    unique_sectors = sorted(int(value) for value in np.unique(sector_values))
    columns: List[np.ndarray] = []
    names: List[str] = []
    cluster = np.empty(len(time), dtype=int)
    group_offset = 0
    for sector_value in unique_sectors:
        in_sector = sector_values == sector_value
        columns.append(in_sector.astype(float))
        names.append(f"sector_{sector_value}_offset")
        centered_time = np.zeros(len(time))
        centered_time[in_sector] = time[in_sector] - np.median(time[in_sector])
        columns.append(centered_time)
        names.append(f"sector_{sector_value}_slope")
        local_block = np.floor(
            (time[in_sector] - np.min(time[in_sector])) / block_days
        ).astype(int)
        cluster[in_sector] = group_offset + local_block
        group_offset += int(np.max(local_block)) + 1

    angle = 2.0 * np.pi * phase_days / period_days
    half_orbit_distance = np.abs(np.abs(phase_days) - 0.5 * period_days)
    eclipse = (half_orbit_distance < 0.5 * duration_days).astype(float)
    physical_values = {
        "reflection_semiamplitude": -np.cos(angle),
        "beaming_semiamplitude": np.sin(angle),
        "ellipsoidal_semiamplitude": -np.cos(2.0 * angle),
        "second_harmonic_sine_control": np.sin(2.0 * angle),
        "secondary_eclipse_depth": -eclipse,
    }
    for name, values in physical_values.items():
        columns.append(values)
        names.append(name)
    return np.column_stack(columns), names, cluster


def fit_phase_curve_components(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    sector_values: np.ndarray,
    ephemeris: Dict[str, Any],
    block_days: float = BLOCK_DAYS,
    primary_mask_half_durations: float = PRIMARY_MASK_HALF_DURATIONS,
) -> Dict[str, Any]:
    """Fit harmonic + eclipse components and return the component report."""
    period_days = ephemeris["period_days"]
    epoch_btjd = ephemeris["epoch_btjd"]
    duration_days = ephemeris["duration_days"]
    phase_days = phase_hours(time, period_days, epoch_btjd) / 24.0
    keep = (
        np.abs(phase_days) > primary_mask_half_durations * duration_days
    ) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    time = time[keep]
    phase_days = phase_days[keep]
    flux = flux[keep]
    flux_err = flux_err[keep]
    sector_values = sector_values[keep]
    if time.size < 100:
        raise ValueError("insufficient out-of-transit coverage for phase curve analysis")

    design, names, cluster = build_design_matrix(
        time, phase_days, period_days, duration_days, sector_values, block_days=block_days
    )
    sigma = np.asarray(flux_err, dtype=float)
    weighted_design = design / sigma[:, None]
    coefficients = np.linalg.lstsq(weighted_design, flux / sigma, rcond=None)[0]
    model = design @ coefficients
    residual = flux - model
    covariance, n_clusters = cluster_sandwich_covariance(design, residual, sigma, cluster)
    errors = np.sqrt(np.diag(covariance))

    components: Dict[str, Dict[str, float]] = {}
    for name in PHYSICAL_COMPONENTS:
        index = names.index(name)
        value_ppm = float(coefficients[index] * 1e6)
        error_ppm = float(errors[index] * 1e6)
        components[name] = {
            "value_ppm": round(value_ppm, 3),
            "block_robust_error_ppm": round(error_ppm, 3),
            "significance_sigma": round(value_ppm / error_ppm if error_ppm > 0 else 0.0, 2),
            "three_sigma_absolute_upper_bound_ppm": round(abs(value_ppm) + 3.0 * error_ppm, 3),
        }

    max_significance = max(
        abs(item["significance_sigma"]) for item in components.values()
    )
    reflection = components["reflection_semiamplitude"]
    unphysical_reflection = (
        reflection["value_ppm"] < 0.0
        and abs(reflection["significance_sigma"]) >= 3.0
    )
    if unphysical_reflection:
        status = "unphysical_phase_harmonic_detected_systematics_limited"
    elif max_significance < 3.0:
        status = "no_significant_phase_curve_component"
    else:
        status = "component_above_three_sigma_requires_followup"

    return {
        "status": status,
        "period_days": float(period_days),
        "epoch_btjd": float(epoch_btjd),
        "n_points_after_primary_transit_mask": int(time.size),
        "n_sectors": int(len(np.unique(sector_values))),
        "n_covariance_clusters": int(n_clusters),
        "primary_mask_half_width_hours": round(primary_mask_half_durations * duration_days * 24.0, 3),
        "secondary_box_duration_hours": round(duration_days * 24.0, 3),
        "components": components,
        "maximum_absolute_significance_sigma": round(max_significance, 2),
    }


def _synthetic_phase_curve_table() -> Dict[str, np.ndarray]:
    """Deterministic demonstration light curve with an injected reflection signal."""
    rng = np.random.default_rng(seed=13)
    demo_period_days = 3.5
    demo_epoch_btjd = 2.0
    demo_duration_days = 0.12
    cadence_days = 120.0 / 86400.0
    time = np.arange(0.0, 27.0, cadence_days)
    phase_days = (
        (time - demo_epoch_btjd + 0.5 * demo_period_days) % demo_period_days
    ) - 0.5 * demo_period_days
    angle = 2.0 * np.pi * phase_days / demo_period_days
    reflection = 150e-6 * (-np.cos(angle))
    flux = 1.0 + reflection + rng.normal(0.0, 400e-6, size=time.shape)
    flux_err = np.full_like(flux, 400e-6)
    sector_values = np.ones(time.size, dtype=int)
    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "sector": sector_values,
        "_duration_days": demo_duration_days,
        "_epoch_btjd": demo_epoch_btjd,
        "_period_days": demo_period_days,
    }


def run_phase_curve_search(workspace: CandidateWorkspace) -> Path:
    """Run the phase curve search and write outputs/phase_curve_results.json."""
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    table = load_light_curve_table(workspace)
    if table is None:
        table = _synthetic_phase_curve_table()
        source = "synthetic-demo"
        ephemeris = {
            "period_days": table.pop("_period_days"),
            "epoch_btjd": table.pop("_epoch_btjd"),
            "duration_days": table.pop("_duration_days"),
            "source": source,
        }
    else:
        source = "candidate-data"
        ephemeris = load_transit_ephemeris(workspace)

    result = fit_phase_curve_components(
        table["time"], table["flux"], table["flux_err"], table["sector"], ephemeris
    )
    payload = {
        "schema_version": "1.0",
        "work_package": "PHASE_CURVE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "method": (
            "weighted simultaneous harmonic and box-eclipse regression with "
            "sector offsets/slopes and 0.5-day cluster-sandwich covariance"
        ),
        "status": result["status"],
        "period_days": result["period_days"],
        "epoch_btjd": result["epoch_btjd"],
        "n_points_after_primary_transit_mask": result["n_points_after_primary_transit_mask"],
        "n_sectors": result["n_sectors"],
        "n_covariance_clusters": result["n_covariance_clusters"],
        "primary_mask_half_width_hours": result["primary_mask_half_width_hours"],
        "secondary_box_duration_hours": result["secondary_box_duration_hours"],
        "components": result["components"],
        "maximum_absolute_significance_sigma": result["maximum_absolute_significance_sigma"],
        "interpretation": (
            "Photometric vetting constraint; no physical phase-curve amplitude "
            "or secondary eclipse detection is claimed without followup."
        ),
    }
    output_path = outputs_dir / "phase_curve_results.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
