"""Target-neutral aperture robustness and dilution engine.

Evaluates transit depth stability and third-light flux contamination across spatial
photometric extraction apertures on TESS Target Pixel Files (TPF):

1. Multi-Size Box Aperture Sensitivity (3x3, 5x5, 7x7 pixels):
   Tests whether transit depth delta(r) varies with aperture size:
   - On-target planetary transits maintain stable depth across expanding apertures.
   - Off-target background blends (BEB/NEB) exhibit increasing depth in larger apertures
     as the contaminating star enters the extraction mask.

2. Gaia DR3 Neighborhood Contamination & Dilution Correction:
   Computes the flux contamination ratio:
       C_contam = (sum_k F_neighbor,k) / F_target
   and the corresponding photometric dilution factor:
       r_dil = F_target / (F_target + sum_k F_neighbor,k) = 1 / (1 + C_contam)
   allowing conversion from observed depth to true undiluted depth:
       delta_true = delta_observed * (1 + C_contam) = delta_observed / r_dil.

Contains zero target-specific constants; all calculations operate dynamically on
candidate TPF data cubes and archival neighbor tables.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .archive import load_validated_archival_gaia_sources
from .inputs import load_tpf_cubes, load_transit_ephemeris
from .lightcurve import phase_hours, robust_transit_depth
from .workspace import CandidateWorkspace

QUALITY_HARD_MASK = 24319        # TESS quality bitmask rejecting severe cadences
APERTURE_HALF_SIZES = (1, 2, 3)  # Half-widths for 3x3, 5x5, and 7x7 pixel bounding boxes


def _box_apertures(
    shape: Tuple[int, int], centroid_x: float, centroid_y: float, half_sizes: Sequence[int]
) -> Dict[str, np.ndarray]:
    """Square pixel boxes of given half-sizes centered on the aperture centroid."""
    yy, xx = np.indices(shape, dtype=float)
    boxes: Dict[str, np.ndarray] = {}
    for half_size in half_sizes:
        mask = (np.abs(xx - centroid_x) <= half_size) & (
            np.abs(yy - centroid_y) <= half_size
        )
        boxes[f"box_{2 * half_size + 1}x{2 * half_size + 1}"] = mask
    return boxes


def aperture_depth_ppm(
    time: np.ndarray,
    flux_1d: np.ndarray,
    ephemeris: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """Median in/out transit depth for a one-dimensional aperture light curve.

    Aperture pixel sums are count-based, so the light curve is normalized to
    its median before the fractional depth is computed.
    """
    duration_hours = float(ephemeris["duration_days"]) * 24.0
    if duration_hours <= 0:
        return None
    flux_1d = np.asarray(flux_1d, dtype=float)
    median_flux = float(np.median(flux_1d))
    if not np.isfinite(median_flux) or median_flux <= 0:
        return None
    try:
        depth_ppm, uncertainty_ppm, n_in, n_out = robust_transit_depth(
            time, flux_1d / median_flux, ephemeris["period_days"], ephemeris["epoch_btjd"], duration_hours
        )
    except ValueError:
        return None
    return {
        "depth_ppm": round(depth_ppm, 2),
        "uncertainty_ppm": round(uncertainty_ppm, 2),
        "n_in_transit": int(n_in),
        "n_out_transit": int(n_out),
    }


def _extract_cube_light_curves(
    cube: Dict[str, Any],
    ephemeris: Dict[str, Any],
    quality_mask: int = QUALITY_HARD_MASK,
) -> Optional[Dict[str, Any]]:
    """Build per-aperture light curves from one TPF cube."""
    time = cube["time"]
    quality = cube["quality"]
    flux = cube["flux"]
    aperture = cube["aperture"]
    shape = flux.shape[1:]
    good = (
        np.isfinite(time)
        & ((quality & quality_mask) == 0)
        & np.all(np.isfinite(flux), axis=(1, 2))
    )
    time = time[good]
    flux = flux[good]
    if time.size < 100:
        return None
    pipeline = (np.asarray(aperture) & 2) != 0
    if int(np.sum(pipeline)) == 0:
        pipeline = np.ones(shape, dtype=bool)
    yy, xx = np.indices(shape, dtype=float)
    centroid_x = float(np.mean(xx[pipeline]))
    centroid_y = float(np.mean(yy[pipeline]))
    boxes = _box_apertures(shape, centroid_x, centroid_y, APERTURE_HALF_SIZES)
    light_curves: Dict[str, Any] = {}
    for name, mask in boxes.items():
        flux_1d = np.sum(flux[:, mask], axis=1)
        if int(np.sum(mask)) > 0:
            light_curves[name] = flux_1d
    light_curves["pipeline"] = np.sum(flux[:, pipeline], axis=1)
    return {"time": time, "light_curves": light_curves}


def gaia_contamination_factor(
    neighbors: Sequence[Dict[str, Any]],
    search_radius_arcsec: float = 60.0,
    target_g_mag: Optional[float] = None,
) -> Dict[str, Any]:
    """Flux contamination factor from Gaia G-band neighbor flux ratios.

    Returns the summed contamination ratio C_contam = sum(neighbor_flux /
    target_flux) over neighbors within the search radius.
    """
    total_ratio = 0.0
    included = 0
    valid_target_g_mag = _finite_float(target_g_mag)
    for row in neighbors:
        separation_arcsec = _finite_float(row.get("separation_arcsec"))
        if separation_arcsec is None or separation_arcsec > search_radius_arcsec:
            continue
        if row.get("is_target"):
            continue
        ratio = _finite_float(row.get("flux_ratio"))
        if ratio is None and valid_target_g_mag is not None:
            g_mag = _finite_float(row.get("g_mag"))
            if g_mag is not None:
                ratio = 10.0 ** (-0.4 * (g_mag - valid_target_g_mag))
        if ratio is None or ratio < 0.0:
            continue
        total_ratio += ratio
        included += 1
    return {
        "contamination_factor": round(total_ratio, 6),
        "n_neighbors_included": int(included),
        "search_radius_arcsec": float(search_radius_arcsec),
    }


def _finite_float(value: Any) -> Optional[float]:
    """Return a finite numeric value, or ``None`` for invalid measurements."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _load_archival_gaia_neighbor_rows(
    workspace: CandidateWorkspace,
) -> Tuple[List[Dict[str, Any]], Optional[float], Dict[str, Any]]:
    """Map validated archival Gaia photometry into dilution input rows."""
    target, sources, metadata = load_validated_archival_gaia_sources(workspace)
    if target is None:
        return [], None, metadata
    target_g_mag = _finite_float(target.get("phot_g_mean_mag"))
    if target_g_mag is None:
        return [], None, metadata

    rows: List[Dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        separation_arcsec = _finite_float(source.get("separation_arcsec"))
        g_mag = _finite_float(source.get("phot_g_mean_mag"))
        if separation_arcsec is None or g_mag is None:
            continue
        rows.append(
            {
                "g_mag": g_mag,
                "separation_arcsec": separation_arcsec,
                "flux_ratio": None,
                "is_target": False,
            }
        )

    metadata["target_g_mag"] = target_g_mag
    metadata["target_g_mag_unit"] = "mag"
    return rows, target_g_mag, metadata


def _synthetic_tpf_cube() -> Dict[str, Any]:
    """Deterministic test-only TPF cube with a blended neighbor."""
    rng = np.random.default_rng(seed=31)
    shape = (11, 11)
    target_x, target_y = 5.0, 5.0
    neighbor_x, neighbor_y = 7.5, 5.0
    neighbor_ratio = 0.02
    fwhm = 1.2
    sigma = fwhm / 2.3548
    yy, xx = np.indices(shape, dtype=float)
    target_psf = np.exp(
        -((xx - target_x) ** 2 + (yy - target_y) ** 2) / (2.0 * sigma**2)
    )
    neighbor_psf = np.exp(
        -((xx - neighbor_x) ** 2 + (yy - neighbor_y) ** 2) / (2.0 * sigma**2)
    )
    base_image = 1800.0 + 1200.0 * target_psf / np.max(target_psf) + 24.0 * neighbor_psf
    deficit_psf = 60.0 * target_psf / np.max(target_psf)

    demo_period_days = 3.5
    demo_epoch_btjd = 2.0
    demo_duration_days = 0.12
    cadence_days = 120.0 / 86400.0
    time = np.arange(0.0, 27.0, cadence_days)
    hours = phase_hours(time, demo_period_days, demo_epoch_btjd)
    in_transit = np.abs(hours) < 0.5 * demo_duration_days * 24.0
    flux_cube = np.zeros((time.size, *shape), dtype=float)
    for index in range(time.size):
        image = base_image + rng.normal(0.0, 1.0, size=shape)
        if in_transit[index]:
            image = image - deficit_psf
        flux_cube[index] = image
    aperture = np.zeros(shape, dtype=int)
    aperture[1:-1, 1:-1] = 2
    return {
        "path": None,
        "sector": 1,
        "time": time,
        "quality": np.zeros(time.size, dtype=np.int64),
        "flux": flux_cube,
        "aperture": aperture,
        "header": {},
        "_neighbor_rows": [
            {
                "g_mag": 14.24,
                "separation_arcsec": round(2.5 * 21.0, 2),
                "flux_ratio": neighbor_ratio,
                "is_target": False,
            }
        ],
        "_period_days": demo_period_days,
        "_epoch_btjd": demo_epoch_btjd,
        "_duration_days": demo_duration_days,
    }


def run_dilution_sensitivity(workspace: CandidateWorkspace) -> Path:
    """Run the aperture robustness analysis and write dilution_sensitivity_results.json."""
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    cubes = load_tpf_cubes(workspace, require_raw_provenance=True)
    if not cubes:
        raise RuntimeError("dilution sensitivity requires observed candidate target-pixel data")
    ephemeris = load_transit_ephemeris(workspace)
    required_fields = ("period_days", "epoch_btjd", "duration_days")
    if ephemeris.get("source") == "synthetic-demo" or any(
        ephemeris.get("field_sources", {}).get(field) == "synthetic-demo"
        for field in required_fields
    ):
        raise RuntimeError("dilution sensitivity requires a complete candidate-derived transit ephemeris")
    source = "candidate-data"
    neighbor_rows, target_g_mag, contamination_metadata = _load_archival_gaia_neighbor_rows(
        workspace
    )

    aperture_rows: List[Dict[str, Any]] = []
    for cube in cubes:
        extracted = _extract_cube_light_curves(cube, ephemeris)
        if extracted is None:
            continue
        for name, flux_1d in extracted["light_curves"].items():
            depth = aperture_depth_ppm(extracted["time"], flux_1d, ephemeris)
            if depth is None:
                continue
            aperture_rows.append(
                {
                    "sector": int(cube["sector"]),
                    "aperture": name,
                    **depth,
                }
            )

    contamination = gaia_contamination_factor(neighbor_rows, target_g_mag=target_g_mag)
    contamination.update(contamination_metadata)
    if not aperture_rows:
        raise RuntimeError("no measurable aperture light curves produced")

    depths = [row["depth_ppm"] for row in aperture_rows]
    median_depth = float(np.median(depths))
    stability = (float(np.max(depths)) - float(np.min(depths))) / max(median_depth, 1e-9)
    per_aperture_summary: Dict[str, Any] = {}
    for row in aperture_rows:
        key = row["aperture"]
        if key not in per_aperture_summary:
            per_aperture_summary[key] = []
        per_aperture_summary[key].append(row["depth_ppm"])

    payload = {
        "schema_version": "1.0",
        "work_package": "DILUTION_SENSITIVITY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "scientific_status": "exploratory-aperture-sensitivity-diagnostic",
        "validation_eligible": False,
        "validation_reason": (
            "Gaia-band neighbor context and aperture-depth variation are not a "
            "calibrated TESS-band dilution correction or validation constraint."
        ),
        "apertures": aperture_rows,
        "aperture_summary_ppm": {
            name: {
                "n_sectors": len(values),
                "median_depth_ppm": round(float(np.median(values)), 2),
                "min_depth_ppm": round(float(np.min(values)), 2),
                "max_depth_ppm": round(float(np.max(values)), 2),
            }
            for name, values in per_aperture_summary.items()
        },
        "depth_stability": {
            "median_depth_ppm": round(median_depth, 2),
            "max_variation_relative_to_median": round(stability, 4),
            "interpretation": (
                "stable"
                if stability < 0.3
                else "aperture-sensitive"
            ),
        },
        "contamination": contamination,
        "caveat": (
            "Gaia G-band neighbor flux sums are conservative sensitivity "
            "bounds rather than exact TESS-band aperture corrections."
        ),
    }
    output_path = outputs_dir / "dilution_sensitivity_results.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
