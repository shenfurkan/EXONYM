"""Candidate-local aperture-depth and archival-neighbor sensitivity diagnostics.

The module extracts several spatial apertures from observed target-pixel cubes,
measures fixed-ephemeris depths, and summarizes catalog-neighbor flux ratios.
Depth changes across apertures can make blend-sensitive behavior visible for
review, while the neighbor sum reports a band-limited contamination context.

Scientific Boundary:
    Aperture variation and catalog-band flux ratios are exploratory
    diagnostics. They are not a calibrated instrument-band dilution correction,
    source-localization result, or validation constraint.

References:
    methods/literature_notes/perryman_handbook/
    04_false_positives_vetting_diagnostics.md describes dilution and
    blend-related false-positive context.

Verified catalog context, units, and fail-closed boundary
----------------------------------------------------------
The Gaia/TESS catalog context is Stassun et al. (2019), ADS
``2019AJ....158..138S``, DOI ``10.3847/1538-3881/ab3467``.  Time is
``BTJD_TDB`` days; pixel flux remains source-native then becomes dimensionless
normalized aperture flux; depth/error is ppm; Gaia/TESS magnitudes are mag;
separation is arcsec; and contamination factors/flux ratios are dimensionless.
The color transform is accepted only in its declared calibration interval;
missing color, invalid neighbor, incompatible band, or insufficient aperture
coverage yields unavailable rather than an invented dilution correction.  This
is neither a calibrated TESS scene model nor source localization and cannot set
``claim_eligible``.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .archive import load_validated_archival_gaia_sources
from .inputs import is_complete_candidate_ephemeris, load_tpf_cubes, load_transit_ephemeris
from .lightcurve import phase_hours, robust_transit_depth
from .workspace import CandidateWorkspace

QUALITY_HARD_MASK = 24319        # TESS quality bitmask rejecting severe cadences
APERTURE_HALF_SIZES = (1, 2, 3)  # Half-widths for 3x3, 5x5, and 7x7 pixel bounding boxes
# ASTROPHYSICAL_HEURISTIC: Maximum fractional transit depth variation across apertures
# (30%) before flagging potential contamination/blending from an off-target
# source. This is a declared review policy, not a calibrated contamination law.
APERTURE_VARIATION_STABILITY_THRESHOLD = 0.3


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
    """Measure a robust fixed-ephemeris depth in one aperture light curve.

    Pixel sums are count-based, so the series is normalized by its finite
    median before calculating the relative in-transit deficit.

    Args:
        time: Observation times in BTJD_TDB days.
        flux_1d: One-dimensional aperture-summed flux values.
        ephemeris: Candidate period in days, epoch in BTJD_TDB days, and
            duration in days.

    Returns:
        Depth and scatter-based uncertainty in ppm plus in- and
        out-of-transit cadence counts, or None when a measurement is
        unavailable.

    Notes:
        The uncertainty describes sampled aperture scatter only; it does not
        calibrate correlated noise or flux dilution.
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
        "depth_ppm": float(depth_ppm),
        "uncertainty_ppm": float(uncertainty_ppm),
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


STASSUN_BP_RP_MIN = -0.1
STASSUN_BP_RP_MAX = 4.5


def gaia_g_to_tess_mag(g_mag: float, bp_rp_color: Optional[float]) -> Optional[float]:
    """Convert a Gaia G magnitude to the approximate TESS T magnitude.

    Uses the Stassun et al. (2019) cubic polynomial in ``G_BP - G_RP`` when a
    color is available within its calibrated color range. Returns ``None``
    when either magnitude, color, or calibration applicability is unavailable.
    """
    magnitude = _finite_float(g_mag)
    color = _finite_float(bp_rp_color)
    if (
        magnitude is None
        or color is None
        or not STASSUN_BP_RP_MIN <= color <= STASSUN_BP_RP_MAX
    ):
        return None
    delta_t_g = (
        -0.00522555 * color**3 + 0.0891337 * color**2 - 0.633923 * color + 0.0324473
    )
    return magnitude + delta_t_g


def gaia_contamination_factor(
    neighbors: Sequence[Dict[str, Any]],
    search_radius_arcsec: float = 60.0,
    target_g_mag: Optional[float] = None,
    target_bp_rp_color: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarize usable catalog-band neighbor flux ratios near the target.

    A row can supply a direct non-negative flux ratio or a magnitude from
    which a ratio is derived when the target magnitude is available. The
    routine excludes the target row and neighbors outside the requested
    angular radius.

    Args:
        neighbors: Archival neighbor rows with separation, target marker, and
        either a direct TESS-band flux ratio or Gaia magnitude and BP/RP color
        information.
        search_radius_arcsec: Maximum retained angular separation in arcsec.
        target_g_mag: Optional target catalog magnitude used to derive ratios.

    Returns:
        A mapping with an available dimensionless summed contamination factor,
        or explicit unavailable fields when any retained neighbor cannot be
        converted within the Stassun calibration.

    Notes:
        Catalog-band ratios are retained as sensitivity context, not exact
        aperture- or instrument-band corrections.
    """
    total_ratio = 0.0
    included = 0
    omitted: List[Dict[str, Any]] = []
    valid_target_g_mag = _finite_float(target_g_mag)
    valid_target_bp_rp = _finite_float(target_bp_rp_color)
    target_t_mag = gaia_g_to_tess_mag(valid_target_g_mag, valid_target_bp_rp)
    for row in neighbors:
        separation_arcsec = _finite_float(row.get("separation_arcsec"))
        if separation_arcsec is None or separation_arcsec > search_radius_arcsec:
            continue
        if row.get("is_target"):
            continue
        ratio = _finite_float(row.get("flux_ratio"))
        if ratio is None:
            g_mag = _finite_float(row.get("phot_g_mean_mag"))
            if g_mag is None:
                g_mag = _finite_float(row.get("g_mag"))
            neighbor_bp_rp = _finite_float(row.get("bp_rp"))
            if neighbor_bp_rp is None:
                neighbor_bp_rp = _finite_float(row.get("bp_rp_color"))
            neighbor_t_mag = gaia_g_to_tess_mag(g_mag, neighbor_bp_rp)
            if target_t_mag is not None and neighbor_t_mag is not None:
                ratio = 10.0 ** (-0.4 * (neighbor_t_mag - target_t_mag))
            else:
                omitted.append(
                    {
                        "separation_arcsec": separation_arcsec,
                        "reason": "stassun-gaia-to-tess-input-or-calibration-unavailable",
                    }
                )
                continue
        if ratio is None or ratio < 0.0:
            omitted.append(
                {"separation_arcsec": separation_arcsec, "reason": "invalid-flux-ratio"}
            )
            continue
        total_ratio += ratio
        included += 1
    available = not omitted
    return {
        "availability": "available" if available else "unavailable",
        "contamination_factor": float(total_ratio) if available else None,
        "n_neighbors_included": int(included),
        "n_neighbors_omitted": len(omitted),
        "omitted_neighbors": omitted,
        "search_radius_arcsec": float(search_radius_arcsec),
        "contamination_ratio": float(total_ratio) if available else None,
        "n_neighbors_in_aperture": int(included),
        "target_g_mag": valid_target_g_mag,
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
) -> Tuple[List[Dict[str, Any]], Optional[float], Dict[str, Any], Optional[float]]:
    """Map validated archival Gaia photometry into dilution input rows."""
    target, sources, metadata = load_validated_archival_gaia_sources(workspace)
    if target is None:
        return [], None, metadata, None
    target_g_mag = _finite_float(target.get("phot_g_mean_mag"))
    if target_g_mag is None:
        return [], None, metadata, None
    # Optional BP-RP color enables the Stassun et al. (2019) Gaia G -> TESS T
    # transformation; absent colors retain the conservative G-band ratios.
    target_bp = _finite_float(target.get("phot_bp_mean_mag"))
    target_rp = _finite_float(target.get("phot_rp_mean_mag"))
    target_bp_rp_color: Optional[float] = None
    if target_bp is not None and target_rp is not None:
        target_bp_rp_color = target_bp - target_rp

    rows: List[Dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        separation_arcsec = _finite_float(source.get("separation_arcsec"))
        g_mag = _finite_float(source.get("phot_g_mean_mag"))
        if separation_arcsec is None:
            continue
        neighbor_bp_rp_color: Optional[float] = None
        neighbor_bp = _finite_float(source.get("phot_bp_mean_mag"))
        neighbor_rp = _finite_float(source.get("phot_rp_mean_mag"))
        if neighbor_bp is not None and neighbor_rp is not None:
            neighbor_bp_rp_color = neighbor_bp - neighbor_rp
        rows.append(
            {
                "g_mag": g_mag,
                "bp_rp_color": neighbor_bp_rp_color,
                "separation_arcsec": separation_arcsec,
                "flux_ratio": None,
                "is_target": False,
            }
        )

    metadata["target_g_mag"] = target_g_mag
    metadata["target_g_mag_unit"] = "mag"
    metadata["target_bp_rp_color"] = target_bp_rp_color
    return rows, target_g_mag, metadata, target_bp_rp_color


def run_dilution_sensitivity(workspace: CandidateWorkspace) -> Path:
    """Write a candidate-local aperture and neighbor sensitivity artifact.

    Observed, provenance-validated target-pixel data and a complete
    candidate-derived ephemeris are required. The output retains individual
    aperture measurements, their descriptive spread, and archival-neighbor
    context for later review.

    Args:
        workspace: Candidate workspace that owns observed inputs and the
            resulting output record.

    Returns:
        Path to the dilution_sensitivity_results.json artifact.

    Raises:
        RuntimeError: If observed pixel data, a complete ephemeris, or
            measurable aperture light curves are unavailable.
    """
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    cubes = load_tpf_cubes(workspace, require_raw_provenance=True)
    if not cubes:
        raise RuntimeError("dilution sensitivity requires observed candidate target-pixel data")
    ephemeris = load_transit_ephemeris(workspace)
    if not is_complete_candidate_ephemeris(ephemeris):
        raise RuntimeError("dilution sensitivity requires a complete candidate-derived transit ephemeris")
    source = "candidate-data"
    neighbor_rows, target_g_mag, contamination_metadata, target_bp_rp_color = _load_archival_gaia_neighbor_rows(
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

    contamination = gaia_contamination_factor(
        neighbor_rows, target_g_mag=target_g_mag, target_bp_rp_color=target_bp_rp_color
    )
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

    # SCIENTIFIC_BOUNDARY: Preserve aperture and neighbor evidence without
    # promoting a catalog-band estimate into a calibrated dilution correction.
    payload = {
        "schema_version": "1.0",
        "work_package": "DILUTION_SENSITIVITY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "scientific_status": "exploratory-aperture-sensitivity-diagnostic",
        "validation_eligible": False,
        "validation_reason": (
            "Gaia-derived neighbor context and aperture-depth variation are not a "
            "calibrated TESS-band dilution correction or validation constraint."
        ),
        "apertures": aperture_rows,
        "aperture_summary_ppm": {
            name: {
                "n_sectors": len(values),
                "median_depth_ppm": float(np.median(values)),
                "min_depth_ppm": float(np.min(values)),
                "max_depth_ppm": float(np.max(values)),
            }
            for name, values in per_aperture_summary.items()
        },
        "depth_stability": {
            "median_depth_ppm": float(median_depth),
            "max_variation_relative_to_median": float(stability),
            "interpretation": (
                "stable"
                if stability < APERTURE_VARIATION_STABILITY_THRESHOLD
                else "aperture-sensitive"
            ),
        },
        "contamination": contamination,
        "caveat": (
            "Gaia-to-TESS color conversions are catalog context rather than "
            "exact TESS-band aperture corrections."
        ),
    }
    output_path = outputs_dir / "dilution_sensitivity_results.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_path
