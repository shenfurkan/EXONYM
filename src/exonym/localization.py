"""Target-neutral sub-pixel PRF transit source localization.

Fits the in-transit depth map of a TESS target pixel file as a non-negative
linear combination of Gaussian PRF templates centered on the catalog target and
nearby Gaia sources, and reports the precise RA/Dec offset of the transit flux
deficit centroid. All target positions are read from file headers or workspace
data, never hardcoded.
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

PIXEL_SCALE_ARCSEC = 21.0
PRF_FWHM_PIXELS = 0.75
QUALITY_HARD_MASK = 24319
MIN_DEPTH_CORE_PIXELS = 3


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


def fit_depth_map_prf(
    depth_map: np.ndarray,
    pixel_mask: np.ndarray,
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    fwhm_pixels: float = PRF_FWHM_PIXELS,
) -> Tuple[Optional[np.ndarray], Optional[float], int]:
    """NNLS-fit Gaussian PRF amplitudes for each candidate source.

    Returns (amplitudes, residual, n_pixels_used) or (None, None, 0) when the
    system is degenerate or has too few usable pixels.
    """
    from scipy.optimize import nnls

    valid_mask = np.asarray(pixel_mask, dtype=bool)
    if int(valid_mask.sum()) < 5 or len(x_positions) == 0:
        return None, None, 0
    yy, xx = np.indices(depth_map.shape, dtype=float)
    rows = []
    for x0, y0 in zip(x_positions, y_positions):
        rows.append(gaussian_prf_kernel(xx[valid_mask], yy[valid_mask], float(x0), float(y0), fwhm_pixels))
    design = np.asarray(rows).T
    depths = depth_map[valid_mask]
    try:
        amplitudes, residual = nnls(design, depths, maxiter=5000)
    except Exception:
        return None, None, 0
    return amplitudes, float(residual), int(valid_mask.sum())


def build_depth_map(
    in_image: np.ndarray, out_image: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (depth_map, valid_mask) from median in/out transit images."""
    shape = in_image.shape
    valid = (
        np.isfinite(in_image)
        & np.isfinite(out_image)
        & (out_image > 0)
    )
    depth_map = np.full(shape, np.nan, dtype=float)
    depth_map[valid] = (out_image[valid] - in_image[valid]) / out_image[valid]
    return depth_map, valid


def localize_depth_deficit(
    depth_map: np.ndarray,
    pixel_mask: np.ndarray,
    target_x: float,
    target_y: float,
    pixel_scale_arcsec: float = PIXEL_SCALE_ARCSEC,
    cos_dec: float = 1.0,
    core_fraction: float = 0.2,
) -> Dict[str, float]:
    """Centroid the transit depth deficit core and offset it from the target.

    Only pixels above ``core_fraction`` of the maximum depth participate so
    the centroid is not pulled toward noise-dominated periphery. Returns
    RA/Dec offsets in arcseconds (RA offset uses the provided cos(dec)
    projection factor) plus the total separation.
    """
    valid_mask = np.asarray(pixel_mask, dtype=bool) & np.isfinite(depth_map)
    depths = depth_map[valid_mask]
    if depths.size < 3 or float(np.max(depths)) <= 0:
        return {
            "ra_offset_arcsec": float("nan"),
            "dec_offset_arcsec": float("nan"),
            "offset_arcsec": float("nan"),
            "n_depth_pixels": 0,
        }
    threshold = core_fraction * float(np.max(depths))
    core_mask = valid_mask & (depth_map >= threshold)
    yy, xx = np.indices(depth_map.shape, dtype=float)
    core_depths = depth_map[core_mask]
    if core_depths.size < MIN_DEPTH_CORE_PIXELS:
        return {
            "ra_offset_arcsec": float("nan"),
            "dec_offset_arcsec": float("nan"),
            "offset_arcsec": float("nan"),
            "n_depth_pixels": int(core_depths.size),
        }
    weights = core_depths / float(np.sum(core_depths))
    centroid_x = float(np.sum(xx[core_mask] * weights))
    centroid_y = float(np.sum(yy[core_mask] * weights))
    ra_offset = (centroid_x - float(target_x)) * pixel_scale_arcsec * max(cos_dec, 0.01)
    dec_offset = (centroid_y - float(target_y)) * pixel_scale_arcsec
    return {
        "ra_offset_arcsec": round(ra_offset, 4),
        "dec_offset_arcsec": round(dec_offset, 4),
        "offset_arcsec": round(math.hypot(ra_offset, dec_offset), 4),
        "n_depth_pixels": int(np.count_nonzero(core_mask)),
    }


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


def extract_tpf_depth_map(
    cube: Dict[str, Any],
    ephemeris: Dict[str, Any],
    quality_mask: int = QUALITY_HARD_MASK,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float], int, int]:
    """Build an in-transit depth map from one TPF cube.

    Returns (depth_map, pipeline_aperture, target_x, target_y, n_in, n_out) or
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
    depth_map, _ = build_depth_map(in_image, out_image)

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
        depth_map,
        pipeline,
        target_x,
        target_y,
        int(in_mask.sum()),
        int(out_mask.sum()),
    )


def _load_gaia_neighbors(workspace: CandidateWorkspace) -> List[Dict[str, Any]]:
    """Read the optional generic Gaia neighbor CSV from external data."""
    path = workspace.path / "data" / "external" / "gaia_neighbors.csv"
    if not path.is_file():
        return []
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - optional dependency
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    required = ("ra", "dec")
    if not all(column in frame.columns for column in required):
        return []
    rows: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        try:
            rows.append(
                {
                    "source_id": str(row.get("source_id", "")),
                    "ra": float(row["ra"]),
                    "dec": float(row["dec"]),
                    "g_mag": float(row.get("phot_g_mean_mag", 20.0)),
                    "separation_arcsec": float(row.get("separation_arcsec", 0.0)),
                    "flux_ratio": float(row.get("flux_ratio_vs_target", 0.0)),
                    "is_target": str(row.get("is_target_match", "")).lower() == "true",
                }
            )
        except (TypeError, ValueError):
            continue
    return rows


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
    depth_map: np.ndarray,
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


def _synthetic_tpf_results() -> List[Dict[str, Any]]:
    """Deterministic demonstration depth map with a known source offset."""
    rng = np.random.default_rng(seed=11)
    shape = (11, 11)
    target_x, target_y = 5.0, 5.0
    deficit_x, deficit_y = target_x + 0.3, target_y - 0.2
    sigma = 0.85
    yy, xx = np.indices(shape, dtype=float)
    out_image = 2000.0 + 20.0 * np.exp(
        -((xx - target_x) ** 2 + (yy - target_y) ** 2) / (2.0 * sigma**2)
    )
    out_image = out_image + 30.0 * np.exp(
        -((xx - 3.0) ** 2 + (yy - 6.5) ** 2) / (2.0 * sigma**2)
    )
    deficit = 14.0 * np.exp(
        -((xx - deficit_x) ** 2 + (yy - deficit_y) ** 2) / (2.0 * sigma**2)
    )
    rng_gauss = rng.normal(0.0, 1.5, size=shape)
    in_image = out_image - deficit + rng_gauss
    out_image = out_image + rng_gauss
    depth_map, valid = build_depth_map(in_image, out_image)
    aperture = np.zeros(shape, dtype=int)
    aperture[2:-2, 2:-2] = 2
    return [
        {
            "sector": 1,
            "depth_map": depth_map,
            "valid": valid,
            "aperture": aperture.astype(bool),
            "target_x": target_x,
            "target_y": target_y,
            "cos_dec": 1.0,
            "n_in": 60,
            "n_out": 300,
        }
    ]


def _fit_one_map(
    depth_map: np.ndarray,
    pipeline: np.ndarray,
    target_x: float,
    target_y: float,
    sources: Sequence[Dict[str, Any]],
    cos_dec: float,
    n_in: int,
    n_out: int,
    sector: int,
) -> Dict[str, Any]:
    pixel_mask = pipeline & np.isfinite(depth_map) & (depth_map != 0)
    amplitudes, residual, n_pixels = fit_depth_map_prf(
        depth_map,
        pixel_mask,
        [src["x_pix"] for src in sources],
        [src["y_pix"] for src in sources],
    )
    centroid = localize_depth_deficit(
        depth_map, pipeline, target_x, target_y, cos_dec=cos_dec
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
        amplitude_ppm = amplitude * 1e6
        flux_ratio = _finite_float(src.get("flux_ratio"))
        g_band_limit_ppm = None
        fit_to_g_band_limit_ratio = None
        if flux_ratio is not None and flux_ratio > 0.0:
            g_band_limit_ppm = flux_ratio * 1e6
            fit_to_g_band_limit_ratio = amplitude_ppm / g_band_limit_ppm
        per_source[str(src["source_id"])] = {
            "g_mag": src["g_mag"],
            "separation_arcsec": src["separation_arcsec"],
            "is_target": src["is_target"],
            "g_band_flux_ratio_vs_target": round(flux_ratio, 8)
            if flux_ratio is not None
            else None,
            "g_band_full_eclipse_limit_ppm": round(g_band_limit_ppm, 2)
            if g_band_limit_ppm is not None
            else None,
            "fit_to_g_band_limit_ratio": round(fit_to_g_band_limit_ratio, 3)
            if fit_to_g_band_limit_ratio is not None
            else None,
            "depth_amplitude_ppm": round(amplitude_ppm, 2),
        }
    target_amplitude = per_source[str(sources[0]["source_id"])]["depth_amplitude_ppm"]
    other_amplitudes = [
        per_source[str(src["source_id"])]["depth_amplitude_ppm"]
        for src in sources[1:]
        if str(src["source_id"]) in per_source
    ]
    max_other = max(other_amplitudes) if other_amplitudes else None
    fit_dominant_id = max(per_source, key=lambda key: per_source[key]["depth_amplitude_ppm"])
    fit_dominant = per_source[fit_dominant_id]
    fit_dominant_is_target = bool(fit_dominant["is_target"])
    depth_core_resolved = int(centroid["n_depth_pixels"]) >= MIN_DEPTH_CORE_PIXELS
    g_band_bound_exceeded = bool(
        not fit_dominant_is_target
        and fit_dominant.get("fit_to_g_band_limit_ratio") is not None
        and float(fit_dominant["fit_to_g_band_limit_ratio"]) > 1.0
    )
    if not depth_core_resolved:
        source_assignment_status = "unresolved_insufficient_depth_core"
    elif g_band_bound_exceeded:
        source_assignment_status = "unresolved_g_band_flux_bound_exceeded"
    else:
        source_assignment_status = "screening_only"
    source_assignment_interpretable = source_assignment_status == "screening_only"
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
        "target_depth_amplitude_ppm": round(target_amplitude, 2),
        "max_other_depth_amplitude_ppm": round(max_other, 2) if max_other is not None else None,
        "target_to_max_other_ratio": round(ratio, 3) if ratio is not None else None,
        "source_assignment_status": source_assignment_status,
        "source_assignment_interpretable": source_assignment_interpretable,
        "fit_dominant_source_id": str(fit_dominant_id),
        "fit_dominant_is_target": fit_dominant_is_target,
        "dominant_source_id": str(fit_dominant_id)
        if source_assignment_interpretable
        else None,
        "dominant_is_target": fit_dominant_is_target
        if source_assignment_interpretable
        else None,
        "ra_offset_arcsec": centroid["ra_offset_arcsec"],
        "dec_offset_arcsec": centroid["dec_offset_arcsec"],
        "offset_arcsec": centroid["offset_arcsec"],
        "n_depth_pixels": centroid["n_depth_pixels"],
        "per_source_amplitudes_ppm": per_source,
    }


def run_prf_localization(
    workspace: CandidateWorkspace, search_radius_arcsec: float = 60.0
) -> Path:
    """Run PRF localization on candidate TPFs and write prf_localization_results.json."""
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ephemeris = load_transit_ephemeris(workspace)
    neighbors = _load_gaia_neighbors(workspace)
    source_catalog: Dict[str, Any] = {
        "availability": "available" if neighbors else "unavailable",
        "source": "data/external/gaia_neighbors.csv",
        "n_catalog_neighbors": sum(1 for row in neighbors if not row.get("is_target")),
    }
    if not neighbors:
        neighbors, source_catalog = _load_archival_gaia_neighbors(workspace)

    cubes = load_tpf_cubes(workspace)
    if cubes:
        source = "candidate-data"
        synthetic = False
    else:
        source = "synthetic-demo"
        synthetic = True

    sector_results: List[Dict[str, Any]] = []
    if synthetic:
        for entry in _synthetic_tpf_results():
            sources = [
                {
                    "source_id": "catalog-target",
                    "x_pix": entry["target_x"],
                    "y_pix": entry["target_y"],
                    "g_mag": 0.0,
                    "flux_ratio": 1.0,
                    "separation_arcsec": 0.0,
                    "is_target": True,
                }
            ]
            sector_results.append(
                _fit_one_map(
                    entry["depth_map"],
                    entry["aperture"],
                    entry["target_x"],
                    entry["target_y"],
                    sources,
                    entry["cos_dec"],
                    entry["n_in"],
                    entry["n_out"],
                    entry["sector"],
                )
            )
    else:
        try:
            from astropy.wcs import WCS
        except ImportError:  # pragma: no cover - optional dependency
            WCS = None  # type: ignore[assignment]
        for cube in cubes:
            depth_map, pipeline, target_x, target_y, n_in, n_out = extract_tpf_depth_map(
                cube, ephemeris
            )
            if depth_map is None:
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
                    depth_map,
                    pipeline,
                    float(target_x),
                    float(target_y),
                    neighbors,
                    search_radius_arcsec,
                    wcs,
                    cos_dec,
                )
            sector_results.append(
                _fit_one_map(
                    depth_map,
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
    compared = [
        row
        for row in completed
        if row.get("n_modeled_neighbors", 0) > 0
        and row.get("target_to_max_other_ratio") is not None
        and row.get("source_assignment_interpretable", False)
    ]
    unresolved_depth_core = sum(
        1
        for row in completed
        if row.get("source_assignment_status") == "unresolved_insufficient_depth_core"
    )
    unresolved_g_band_bound = sum(
        1
        for row in completed
        if row.get("source_assignment_status") == "unresolved_g_band_flux_bound_exceeded"
    )
    target_dominant_count = sum(
        1 for row in compared if row.get("dominant_is_target", False)
    )
    offsets = [row["offset_arcsec"] for row in completed if np.isfinite(row["offset_arcsec"])]
    median_offset = float(np.median(offsets)) if offsets else None
    median_ratio = (
        float(np.median([row["target_to_max_other_ratio"] for row in compared]))
        if compared
        else None
    )
    if not completed:
        status = "no_complete_depth_maps"
    elif unresolved_depth_core:
        status = "inconclusive_insufficient_depth_core"
    elif unresolved_g_band_bound:
        status = "inconclusive_g_band_flux_bound_exceeded"
    elif not compared:
        status = "inconclusive_no_competing_sources_modeled"
    elif target_dominant_count >= 0.5 * len(compared) and (median_ratio or 0.0) > 1.0:
        status = "target_dominant_among_modeled_sources"
    else:
        status = "inconclusive_or_other"
    payload = {
        "schema_version": "1.0",
        "work_package": "PRF_SOURCE_LOCALIZATION",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "method": (
            "Gaussian-PRF non-negative least-squares fit of in-transit depth "
            "maps plus depth-deficit centroid offsets"
        ),
        "prf_model": "isotropic Gaussian, FWHM=0.75 TESS pixels",
        "search_radius_arcsec": float(search_radius_arcsec),
        "source_catalog": source_catalog,
        "sector_results": sector_results,
        "summary": {
            "n_sectors": len(sector_results),
            "n_completed": len(completed),
            "sectors_with_competing_sources_modeled": len(compared),
            "sectors_with_unresolved_depth_core": int(unresolved_depth_core),
            "sectors_with_g_band_flux_bound_exceeded": int(unresolved_g_band_bound),
            "sectors_target_dominant": int(target_dominant_count),
            "median_target_to_other_ratio": median_ratio,
            "median_deficit_offset_arcsec": median_offset,
            "conclusion": status,
        },
        "caveats": [
            "Gaussian PRF approximation; formal TESS PRF library templates are not used.",
            "PRF wings beyond the modeled core cannot exclude a deeply eclipsed distant neighbor.",
            "Target dominance is never reported when no competing source is modeled.",
            "A source assignment is unresolved when fewer than three depth-core pixels are available.",
            "Gaia G-band flux ratios are screening bounds, not exact TESS-band eclipse limits.",
        ],
    }
    output_path = outputs_dir / "prf_localization_results.json"
    output_path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output_path
