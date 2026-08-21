"""Target-neutral sub-pixel PRF transit source localization engine.

Determines the spatial origin of the transit signal on TESS Target Pixel Files (TPF)
to distinguish true planetary transits from off-target Background Eclipsing Binaries (BEB)
and Nearby Eclipsing Binaries (NEB) (Bryson et al. 2013, Twicken et al. 2018).

Key Methodological Steps:
1. In-Transit vs Out-of-Transit Difference Imaging:
   Computes the flux deficit map Delta_I(x, y) = <I_oot(x, y)> - <I_in(x, y)>.
2. Non-Negative Least Squares (NNLS) PRF Decomposition:
   Models Delta_I(x, y) as a linear superposition of Gaussian Pixel Response Function (PRF)
   templates centered on the target and all Gaia DR3 sources within the field of view:
       Delta_I(x, y) = sum_k A_k * PRF(x - x_k, y - y_k)   with A_k >= 0.
3. Sub-pixel Centroid Offset & Significance:
   Computes the astrometric offset (Delta_RA, Delta_Dec) and its 3-sigma localization
   confidence ellipse relative to the nominal host star position.

Instrument Constants:
- TESS Pixel Scale: 21.0 arcseconds / pixel (Ricker et al. 2015).
- TESS Nominal PRF FWHM: ~0.75 pixels.

Contains zero target-specific constants; all celestial positions and TPF cubes are
loaded dynamically from candidate-local workspace files.
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
from .lightcurve import phase_hours
from .workspace import CandidateWorkspace

PIXEL_SCALE_ARCSEC = 21.0       # TESS spatial plate scale (arcseconds per detector pixel)
PRF_FWHM_PIXELS = 0.75          # Nominal isotropic Gaussian PRF Full-Width at Half-Maximum
QUALITY_HARD_MASK = 24319       # TESS bitmask rejecting severe momentum dumps, desat, and cosmic rays
MIN_DIFFERENCE_CORE_PIXELS = 3  # Minimum core-pixel count for difference-image screening


def _finite_float(value: Any) -> Optional[float]:
    """Return a finite number, or ``None`` for an invalid catalog field."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _json_safe(value: Any) -> Any:
    """Recursively replace nonfinite numerical diagnostics with JSON ``null``."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        numeric = _finite_float(value)
        return numeric if numeric is not None else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def gaussian_prf_kernel(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x0: float,
    y0: float,
    fwhm_pixels: float = PRF_FWHM_PIXELS,
) -> np.ndarray:
    """Return a normalized isotropic Gaussian PRF template on a pixel grid."""
    sigma = fwhm_pixels / 2.3548
    kernel = np.exp(
        -((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2.0 * sigma**2)
    )
    total = float(np.sum(kernel))
    return kernel / total if total > 0 else kernel


def fit_difference_image_prf(
    difference_image: np.ndarray,
    pixel_mask: np.ndarray,
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    fwhm_pixels: float = PRF_FWHM_PIXELS,
) -> Tuple[Optional[np.ndarray], Optional[float], int]:
    """NNLS-fit Gaussian PRF amplitudes to an absolute difference image.

    Returns (amplitudes, residual, n_pixels_used) or (None, None, 0) when the
    system is degenerate or has too few usable pixels.
    """
    from scipy.optimize import nnls

    valid_mask = np.asarray(pixel_mask, dtype=bool) & np.isfinite(difference_image)
    if int(valid_mask.sum()) < 5 or len(x_positions) == 0:
        return None, None, 0
    yy, xx = np.indices(difference_image.shape, dtype=float)
    rows = []
    for x0, y0 in zip(x_positions, y_positions):
        rows.append(gaussian_prf_kernel(xx[valid_mask], yy[valid_mask], float(x0), float(y0), fwhm_pixels))
    design = np.asarray(rows).T
    differences = difference_image[valid_mask]
    try:
        amplitudes, residual = nnls(design, differences, maxiter=5000)
    except Exception:
        return None, None, 0
    return amplitudes, float(residual), int(valid_mask.sum())


def fit_depth_map_prf(
    depth_map: np.ndarray,
    pixel_mask: np.ndarray,
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    fwhm_pixels: float = PRF_FWHM_PIXELS,
) -> Tuple[Optional[np.ndarray], Optional[float], int]:
    """Compatibility wrapper for :func:`fit_difference_image_prf`.

    ``depth_map`` is retained only as a legacy parameter name. Callers must
    pass an absolute out-of-transit minus in-transit difference image, not a
    fractional-depth map.
    """
    return fit_difference_image_prf(
        depth_map, pixel_mask, x_positions, y_positions, fwhm_pixels
    )


def build_difference_image(
    in_image: np.ndarray, out_image: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return an absolute out-of-transit minus in-transit difference image.

    A transit source has the point-spread morphology in this image. A
    fractional map would divide that morphology by the local stellar scene and
    is therefore not suitable for PRF source fitting.
    """
    shape = in_image.shape
    valid = np.isfinite(in_image) & np.isfinite(out_image)
    difference_image = np.full(shape, np.nan, dtype=float)
    difference_image[valid] = out_image[valid] - in_image[valid]
    return difference_image, valid


def build_depth_map(
    in_image: np.ndarray, out_image: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning an absolute difference image.

    The former fractional-depth implementation was physically unsuitable for
    PRF decomposition and has intentionally been removed.
    """
    return build_difference_image(in_image, out_image)


def localize_difference_image(
    difference_image: np.ndarray,
    pixel_mask: np.ndarray,
    target_x: float,
    target_y: float,
    pixel_scale_arcsec: float = PIXEL_SCALE_ARCSEC,
    cos_dec: float = 1.0,
    core_fraction: float = 0.2,
) -> Dict[str, float]:
    """Centroid the positive difference-image core and offset it from target.

    Only pixels above ``core_fraction`` of the maximum positive difference participate so
    the centroid is not pulled toward noise-dominated periphery. Returns
    RA/Dec offsets in arcseconds (RA offset uses the provided cos(dec)
    projection factor) plus the total separation.
    """
    valid_mask = np.asarray(pixel_mask, dtype=bool) & np.isfinite(difference_image)
    differences = difference_image[valid_mask]
    if differences.size < 3 or float(np.max(differences)) <= 0:
        return {
            "ra_offset_arcsec": float("nan"),
            "dec_offset_arcsec": float("nan"),
            "offset_arcsec": float("nan"),
            "n_difference_pixels": 0,
        }
    threshold = core_fraction * float(np.max(differences))
    core_mask = valid_mask & (difference_image >= threshold)
    yy, xx = np.indices(difference_image.shape, dtype=float)
    core_differences = difference_image[core_mask]
    if core_differences.size < MIN_DIFFERENCE_CORE_PIXELS:
        return {
            "ra_offset_arcsec": float("nan"),
            "dec_offset_arcsec": float("nan"),
            "offset_arcsec": float("nan"),
            "n_difference_pixels": int(core_differences.size),
        }
    weights = core_differences / float(np.sum(core_differences))
    centroid_x = float(np.sum(xx[core_mask] * weights))
    centroid_y = float(np.sum(yy[core_mask] * weights))
    ra_offset = (centroid_x - float(target_x)) * pixel_scale_arcsec * max(cos_dec, 0.01)
    dec_offset = (centroid_y - float(target_y)) * pixel_scale_arcsec
    return {
        "ra_offset_arcsec": round(ra_offset, 4),
        "dec_offset_arcsec": round(dec_offset, 4),
        "offset_arcsec": round(math.hypot(ra_offset, dec_offset), 4),
        "n_difference_pixels": int(np.count_nonzero(core_mask)),
    }


def localize_depth_deficit(
    depth_map: np.ndarray,
    pixel_mask: np.ndarray,
    target_x: float,
    target_y: float,
    pixel_scale_arcsec: float = PIXEL_SCALE_ARCSEC,
    cos_dec: float = 1.0,
    core_fraction: float = 0.2,
) -> Dict[str, float]:
    """Compatibility wrapper for absolute difference-image centroiding."""
    result = localize_difference_image(
        depth_map, pixel_mask, target_x, target_y, pixel_scale_arcsec, cos_dec, core_fraction
    )
    result["n_depth_pixels"] = result.pop("n_difference_pixels")
    return result


def _header_position(header: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Return (ra_deg, dec_deg) from a generic TPF primary header, if present."""
    try:
        ra = float(header["RA_OBJ"])
        dec = float(header["DEC_OBJ"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(ra) or not np.isfinite(dec):
        return None
    return ra, dec


def extract_tpf_difference_image(
    cube: Dict[str, Any],
    ephemeris: Dict[str, Any],
    quality_mask: int = QUALITY_HARD_MASK,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float], int, int]:
    """Build an absolute in/out-transit difference image from one TPF cube.

    Returns (difference_image, pipeline_aperture, target_x, target_y, n_in, n_out) or
    (None, None, None, None, 0, 0) when coverage is insufficient.
    """
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError:  # pragma: no cover - optional dependency
        return None, None, None, None, 0, 0

    path = cube["path"]
    try:
        with fits.open(path, memmap=False) as hdul:
            pix_hdu, ap_hdu = hdul[1], hdul[2]
            aperture = np.asarray(ap_hdu.data)
            pipeline = (aperture & 2) != 0
            wcs = WCS(ap_hdu.header)
            header = dict(hdul[0].header)
    except Exception:
        return None, None, None, None, 0, 0

    time = cube["time"]
    quality = cube["quality"]
    flux = cube["flux"]
    shape = pipeline.shape
    good = (
        np.isfinite(time)
        & ((quality & quality_mask) == 0)
        & np.all(np.isfinite(flux), axis=(1, 2))
    )
    if int(good.sum()) < 20:
        return None, None, None, None, 0, 0
    time = time[good]
    flux = flux[good]

    period_days = ephemeris["period_days"]
    epoch_btjd = ephemeris["epoch_btjd"]
    duration_days = ephemeris["duration_days"]
    hours = phase_hours(time, period_days, epoch_btjd)
    in_mask = np.abs(hours) < 0.5 * duration_days * 24.0
    out_mask = (np.abs(hours) > 1.2 * duration_days * 24.0) & (
        np.abs(hours) < 2.5 * duration_days * 24.0
    )
    if int(in_mask.sum()) < 10 or int(out_mask.sum()) < 10:
        return None, None, None, None, 0, 0

    in_image = np.nanmedian(flux[in_mask], axis=0)
    out_image = np.nanmedian(flux[out_mask], axis=0)
    difference_image, _ = build_difference_image(in_image, out_image)

    position = _header_position(header)
    if position is None:
        target_x = float(np.mean(np.flatnonzero(np.any(pipeline, axis=0))))
        target_y = float(np.mean(np.flatnonzero(np.any(pipeline, axis=1))))
    else:
        ra_deg, dec_deg = position
        target_x, target_y = wcs.world_to_pixel_values(ra_deg, dec_deg)
        target_x = float(np.asarray(target_x))
        target_y = float(np.asarray(target_y))
    return (
        difference_image,
        pipeline,
        target_x,
        target_y,
        int(in_mask.sum()),
        int(out_mask.sum()),
    )


def extract_tpf_depth_map(
    cube: Dict[str, Any],
    ephemeris: Dict[str, Any],
    quality_mask: int = QUALITY_HARD_MASK,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float], int, int]:
    """Compatibility wrapper returning the corrected absolute difference image."""
    return extract_tpf_difference_image(cube, ephemeris, quality_mask)


def _load_archival_gaia_neighbors(
    workspace: CandidateWorkspace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build localization inputs from a validated archival Gaia report."""
    target, source_rows, metadata = load_validated_archival_gaia_sources(workspace)
    if target is None:
        return [], metadata

    target_ra_deg = _finite_float(target.get("ra_deg"))
    target_dec_deg = _finite_float(target.get("dec_deg"))
    target_g_mag = _finite_float(target.get("phot_g_mean_mag"))
    if target_ra_deg is None or target_dec_deg is None or target_g_mag is None:
        metadata["position_availability"] = "unavailable"
        return [], metadata

    rows: List[Dict[str, Any]] = [
        {
            "source_id": str(target.get("source_id", "archival-target")),
            "ra": target_ra_deg,
            "dec": target_dec_deg,
            "g_mag": target_g_mag,
            "separation_arcsec": _finite_float(target.get("separation_arcsec")) or 0.0,
            "flux_ratio": 1.0,
            "is_target": True,
        }
    ]
    for source in source_rows:
        ra_deg = _finite_float(source.get("ra_deg"))
        dec_deg = _finite_float(source.get("dec_deg"))
        g_mag = _finite_float(source.get("phot_g_mean_mag"))
        separation_arcsec = _finite_float(source.get("separation_arcsec"))
        if None in (ra_deg, dec_deg, g_mag, separation_arcsec):
            continue
        rows.append(
            {
                "source_id": str(source.get("source_id", "archival-neighbor")),
                "ra": ra_deg,
                "dec": dec_deg,
                "g_mag": g_mag,
                "separation_arcsec": separation_arcsec,
                "flux_ratio": 10.0 ** (-0.4 * (g_mag - target_g_mag)),
                "is_target": False,
            }
        )
    metadata["position_availability"] = "available"
    metadata["n_catalog_neighbors"] = len(rows) - 1
    return rows, metadata


def _select_sources(
    difference_image: np.ndarray,
    pipeline: np.ndarray,
    target_x: float,
    target_y: float,
    neighbors: Sequence[Dict[str, Any]],
    search_radius_arcsec: float,
    wcs: Any,
    cos_dec: float,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """Return (sources, cos_dec, pixel_scale) with target always first."""
    shape = pipeline.shape
    sources: List[Dict[str, Any]] = []
    for row in neighbors:
        if float(row.get("separation_arcsec", 0.0)) > search_radius_arcsec:
            continue
        if float(row.get("flux_ratio", 0.0)) < 1e-5:
            continue
        try:
            sx, sy = wcs.world_to_pixel_values(float(row["ra"]), float(row["dec"]))
            sx = float(np.asarray(sx))
            sy = float(np.asarray(sy))
        except Exception:
            continue
        if not (-3 <= sx < shape[1] + 2 and -3 <= sy < shape[0] + 2):
            continue
        sources.append(
            {
                "source_id": row.get("source_id", "neighbor"),
                "x_pix": sx,
                "y_pix": sy,
                "g_mag": float(row.get("g_mag", 20.0)),
                "flux_ratio": float(row.get("flux_ratio", 0.0)),
                "separation_arcsec": float(row.get("separation_arcsec", 0.0)),
                "is_target": bool(row.get("is_target", False)),
            }
        )
    target_entry: Optional[Dict[str, Any]] = next(
        (src for src in sources if src["is_target"]), None
    )
    if target_entry is None:
        target_entry = {
            "source_id": "catalog-target",
            "x_pix": float(target_x),
            "y_pix": float(target_y),
            "g_mag": 0.0,
            "flux_ratio": 1.0,
            "separation_arcsec": 0.0,
            "is_target": True,
        }
    target_entry["x_pix"] = float(target_x)
    target_entry["y_pix"] = float(target_y)
    non_target = sorted(
        (src for src in sources if not src["is_target"]),
        key=lambda src: src["g_mag"],
    )[:5]
    return [target_entry] + non_target, cos_dec, PIXEL_SCALE_ARCSEC


def _fit_one_difference_image(
    difference_image: np.ndarray,
    pipeline: np.ndarray,
    target_x: float,
    target_y: float,
    sources: Sequence[Dict[str, Any]],
    cos_dec: float,
    n_in: int,
    n_out: int,
    sector: int,
) -> Dict[str, Any]:
    pixel_mask = pipeline & np.isfinite(difference_image)
    amplitudes, residual, n_pixels = fit_difference_image_prf(
        difference_image,
        pixel_mask,
        [src["x_pix"] for src in sources],
        [src["y_pix"] for src in sources],
    )
    centroid = localize_difference_image(
        difference_image, pipeline, target_x, target_y, cos_dec=cos_dec
    )
    if amplitudes is None:
        return {
            "sector": int(sector),
            "skipped": True,
            "reason": "insufficient aperture pixels",
        }
    per_source = {}
    for index, src in enumerate(sources):
        amplitude = float(amplitudes[index]) if amplitudes[index] > 0 else 0.0
        flux_ratio = _finite_float(src.get("flux_ratio"))
        per_source[str(src["source_id"])] = {
            "g_mag": src["g_mag"],
            "separation_arcsec": src["separation_arcsec"],
            "is_target": src["is_target"],
            "g_band_flux_ratio_vs_target": round(flux_ratio, 8)
            if flux_ratio is not None
            else None,
            "difference_flux_amplitude": round(amplitude, 6),
        }
    target_amplitude = per_source[str(sources[0]["source_id"])]["difference_flux_amplitude"]
    other_amplitudes = [
        per_source[str(src["source_id"])]["difference_flux_amplitude"]
        for src in sources[1:]
        if str(src["source_id"]) in per_source
    ]
    max_other = max(other_amplitudes) if other_amplitudes else None
    fit_dominant_id = max(per_source, key=lambda key: per_source[key]["difference_flux_amplitude"])
    fit_dominant = per_source[fit_dominant_id]
    fit_dominant_is_target = bool(fit_dominant["is_target"])
    difference_core_resolved = (
        int(centroid["n_difference_pixels"]) >= MIN_DIFFERENCE_CORE_PIXELS
    )
    if not difference_core_resolved:
        source_assignment_status = "unresolved_insufficient_difference_core"
    else:
        source_assignment_status = "screening_only_uncalibrated_prf"
    source_assignment_interpretable = False
    ratio = target_amplitude / max_other if max_other is not None and max_other > 0.0 else None
    return {
        "sector": int(sector),
        "skipped": False,
        "n_aperture_pixels_used": int(n_pixels),
        "n_in_transit_cadences": int(n_in),
        "n_out_transit_cadences": int(n_out),
        "n_modeled_sources": int(len(sources)),
        "n_modeled_neighbors": int(len(other_amplitudes)),
        "nnls_residual": float(residual) if residual is not None else None,
        "difference_flux_units": "TPF FLUX native units",
        "target_difference_flux_amplitude": target_amplitude,
        "max_other_difference_flux_amplitude": max_other,
        "target_to_max_other_difference_ratio": round(ratio, 3) if ratio is not None else None,
        "source_assignment_status": source_assignment_status,
        "source_assignment_interpretable": source_assignment_interpretable,
        "fit_dominant_source_id": str(fit_dominant_id),
        "fit_dominant_is_target": fit_dominant_is_target,
        "difference_centroid_ra_offset_arcsec": centroid["ra_offset_arcsec"],
        "difference_centroid_dec_offset_arcsec": centroid["dec_offset_arcsec"],
        "difference_centroid_offset_arcsec": centroid["offset_arcsec"],
        "n_difference_pixels": centroid["n_difference_pixels"],
        "per_source_difference_flux_amplitudes": per_source,
    }


def run_prf_localization(
    workspace: CandidateWorkspace, search_radius_arcsec: float = 60.0
) -> Path:
    """Run PRF localization on candidate TPFs and write prf_localization_results.json."""
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ephemeris = load_transit_ephemeris(workspace)
    required_ephemeris_fields = ("period_days", "epoch_btjd", "duration_days")
    field_sources = ephemeris.get("field_sources", {})
    candidate_ephemeris = all(
        field_sources.get(field) not in (None, "synthetic-demo")
        for field in required_ephemeris_fields
    )
    neighbors, source_catalog = _load_archival_gaia_neighbors(workspace)

    cubes = load_tpf_cubes(workspace, require_raw_provenance=True)
    if not cubes:
        source = "not-run-no-candidate-tpf"
    elif not candidate_ephemeris:
        source = "not-run-no-candidate-ephemeris"
    else:
        source = "candidate-data"

    sector_results: List[Dict[str, Any]] = []
    if source == "candidate-data":
        try:
            from astropy.wcs import WCS
        except ImportError:  # pragma: no cover - optional dependency
            WCS = None  # type: ignore[assignment]
        for cube in cubes:
            difference_image, pipeline, target_x, target_y, n_in, n_out = extract_tpf_difference_image(
                cube, ephemeris
            )
            if difference_image is None:
                continue
            header = cube["header"]
            position = _header_position(header)
            cos_dec = math.cos(math.radians(float(position[1]))) if position else 1.0
            wcs = None
            if WCS is not None:
                try:
                    from astropy.io import fits

                    with fits.open(cube["path"], memmap=False) as hdul:
                        wcs = WCS(hdul[2].header)
                except Exception:
                    wcs = None
            if wcs is None:
                sources = [
                    {
                        "source_id": "catalog-target",
                        "x_pix": float(target_x),
                        "y_pix": float(target_y),
                        "g_mag": 0.0,
                        "flux_ratio": 1.0,
                        "separation_arcsec": 0.0,
                        "is_target": True,
                    }
                ]
            else:
                sources, cos_dec, _ = _select_sources(
                    difference_image,
                    pipeline,
                    float(target_x),
                    float(target_y),
                    neighbors,
                    search_radius_arcsec,
                    wcs,
                    cos_dec,
                )
            sector_results.append(
                _fit_one_difference_image(
                    difference_image,
                    pipeline,
                    float(target_x),
                    float(target_y),
                    sources,
                    cos_dec,
                    n_in,
                    n_out,
                    cube["sector"],
                )
            )

    completed = [row for row in sector_results if not row.get("skipped", False)]
    sectors_with_neighbors = [
        row for row in completed if row.get("n_modeled_neighbors", 0) > 0
    ]
    unresolved_difference_core = sum(
        1
        for row in completed
        if row.get("source_assignment_status") == "unresolved_insufficient_difference_core"
    )
    offsets = [
        row["difference_centroid_offset_arcsec"]
        for row in completed
        if np.isfinite(row["difference_centroid_offset_arcsec"])
    ]
    median_offset = float(np.median(offsets)) if offsets else None
    median_ratio = (
        float(
            np.median(
                [row["target_to_max_other_difference_ratio"] for row in sectors_with_neighbors]
            )
        )
        if sectors_with_neighbors
        else None
    )
    if source == "not-run-no-candidate-tpf":
        status = "inconclusive_no_candidate_tpf"
    elif source == "not-run-no-candidate-ephemeris":
        status = "inconclusive_no_candidate_ephemeris"
    elif not completed:
        status = "inconclusive_no_complete_depth_maps"
    elif unresolved_difference_core:
        status = "inconclusive_insufficient_difference_core"
    else:
        status = "inconclusive_uncalibrated_prf"
    payload = {
        "schema_version": "1.0",
        "work_package": "PRF_SOURCE_LOCALIZATION",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": workspace.candidate_id,
        "source": source,
        "calibration_status": "uncalibrated",
        "validation_eligible": False,
        "ephemeris_provenance": {
            "source": ephemeris.get("source"),
            "field_sources": {
                field: field_sources.get(field) for field in required_ephemeris_fields
            },
        },
        "method": (
            "Gaussian-PRF non-negative least-squares screening of absolute "
            "out-of-transit minus in-transit difference images"
        ),
        "prf_model": "isotropic Gaussian, FWHM=0.75 TESS pixels",
        "search_radius_arcsec": float(search_radius_arcsec),
        "source_catalog": source_catalog,
        "sector_results": sector_results,
        "summary": {
            "n_sectors": len(sector_results),
            "n_completed": len(completed),
            "sectors_with_competing_sources_modeled": len(sectors_with_neighbors),
            "sectors_with_unresolved_difference_core": int(unresolved_difference_core),
            "sectors_with_uncalibrated_prf": int(len(completed)),
            "median_target_to_other_difference_ratio": median_ratio,
            "median_difference_image_offset_arcsec": median_offset,
            "conclusion": status,
        },
        "caveats": [
            "Gaussian PRF approximation; formal TESS PRF library templates are not used.",
            "Absolute difference-flux amplitudes are in native TPF units, not transit depths or ppm.",
            "No covariance, pixel-response, or injected-source calibration supports an automated source assignment.",
            "Gaia G-band flux ratios are retained only as catalog context, not eclipse-depth constraints.",
            "PRF wings beyond the modeled core cannot exclude a deeply eclipsed distant neighbor.",
        ],
    }
    output_path = outputs_dir / "prf_localization_results.json"
    output_path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output_path
