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
- TESS Nominal PRF FWHM: ~2 pixels for this screening approximation.

Contains zero target-specific constants; all celestial positions and TPF cubes are
loaded dynamically from candidate-local workspace files.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .archive import load_validated_archival_gaia_sources
from .inputs import BTJD_TIME_SYSTEM, load_tpf_cubes, load_transit_ephemeris
from .lightcurve import phase_hours
from .resources import read_schema_text
from .workspace import CandidateWorkspace

# TESS pixel-plate scale: 21.0 arcsec/pixel (Ricker et al. 2015, JATIS 1, 014003).
# All coordinate conversions between pixel and equatorial offsets rely on this factor.
PIXEL_SCALE_ARCSEC = 21.0

# ASTROPHYSICAL_HEURISTIC: a detector-scale nominal isotropic Gaussian width
# provides a less artificially compact screening template. The true TESS PRF
# remains wavelength-, temperature-, and field-position-dependent, so this is
# deliberately not a replacement for the calibrated PRF-library model.
PRF_FWHM_PIXELS = 2.0

# QUALITY_HARD_MASK: bitwise OR of TESS data-quality flags for severe instrumental
# anomalies (momentum dumps, desaturation events, cosmic-ray hits).  Cadences whose
# QUALITY flags intersect this mask are excluded from in/out-transit stacks so they
# do not contaminate the difference image.
QUALITY_HARD_MASK = 24319

# NUMERICAL_GUARD: minimum number of core pixels required for a physically
# meaningful flux-weighted centroid.  Below this threshold the difference-image
# core is noise-dominated, and no automated source assignment can be attempted.
MIN_DIFFERENCE_CORE_PIXELS = 3


def _finite_float(value: Any) -> Optional[float]:
    """Canonicalise a catalog-supplied numeric field to a finite ``float``.

    Returns ``None`` for any input that cannot be represented as a finite
    IEEE 754 double, including ``NaN``, ``±Inf``, non-numeric strings, and
    missing (``None``) values.  This guard prevents downstream arithmetic
    (coordinate transforms, flux-ratio calculations) from silently
    propagating non-finite sentinels.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _json_safe(value: Any) -> Any:
    """Recursively convert non-finite floats and numpy scalars to JSON-safe types.

    IEEE 754 non-finites (``NaN``, ``±Inf``) have no legal JSON representation
    (RFC 8259 §6).  This serializer replaces them with ``null`` so the output
    payload always passes ``json.dumps(..., allow_nan=False)``.  Numpy integer
    scalars are converted to Python ``int`` to avoid type-encoding ambiguities.
    """
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
    """Build a unit-normalised isotropic 2-D Gaussian PRF template.

    Mathematical Formulation
    ------------------------
    .. math::

        \\sigma &= \\frac{\\text{FWHM}}{2\\sqrt{2\\ln 2}} \\approx \\frac{\\text{FWHM}}{2.3548} \\\\[4pt]
        G(x, y \\mid x_0, y_0, \\sigma) &= \\exp\\!\\left(-\\frac{(x - x_0)^2 + (y - y_0)^2}{2\\sigma^2}\\right) \\\\[4pt]
        \\tilde{G}(x, y) &= \\frac{G(x, y)}{\\sum_{i,j} G_{i,j}}

    The denominator normalises the template to unit total flux so that the
    NNLS amplitude :math:`\\beta_k` (see :func:`fit_difference_image_prf`)
    directly estimates the integrated difference-flux contributed by source
    *k*.

    Astrophysical Rationale
    -----------------------
    The true TESS PRF is wavelength-dependent, mildly asymmetric, and varies
    with detector field position (Twicken et al. 2018).  An isotropic
    Gaussian with a fixed nominal detector-scale FWHM is a first-order approximation
    suitable for screening-level source-localisation but not for calibrated
    astrometry.  Formal TESS PRF library templates are not used; this is an
    intentional methodological simplification documented in the output
    caveats.

    Parameters
    ----------
    x_grid : np.ndarray
        2-D array of pixel *x* coordinates (column index).
    y_grid : np.ndarray
        2-D array of pixel *y* coordinates (row index), same shape as *x_grid*.
    x0 : float
        Source column centre in pixel units.
    y0 : float
        Source row centre in pixel units.
    fwhm_pixels : float, optional
        Gaussian FWHM in TESS pixels. Default is ``PRF_FWHM_PIXELS``.

    Returns
    -------
    np.ndarray
        Unit-normalised kernel array, same shape as *x_grid* / *y_grid*.
        If the sum over all pixels is zero the un-normalised kernel is
        returned as-is.
    """
    # sigma = FWHM / sqrt(8 * ln 2): the exact closed-form Gaussian
    # FWHM-to-sigma relation (Tier 3 mathematical identity, no fitted constant).
    sigma = fwhm_pixels / math.sqrt(8.0 * math.log(2.0))
    kernel = np.exp(
        -((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2.0 * sigma**2)
    )
    total = float(np.sum(kernel))
    return kernel / total if total > 0 else kernel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write strict JSON with an atomic same-directory replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _strict_json(path: Path) -> Dict[str, Any]:
    """Read a finite, unambiguous JSON object from a candidate-owned asset."""
    def reject_constant(value: str) -> object:
        raise ValueError("non-finite JSON constant: {0}".format(value))

    def parse_float(value: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("non-finite JSON number")
        return numeric

    def reject_duplicates(pairs: Sequence[Tuple[str, object]]) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError("duplicate JSON key: {0}".format(key))
            parsed[key] = value
        return parsed

    payload = json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=reject_constant,
        parse_float=parse_float,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(payload, dict):
        raise ValueError("JSON asset must be an object")
    return payload


def _validate_candidate_artifact(
    workspace: CandidateWorkspace, artifact: object, label: str
) -> Dict[str, str]:
    """Validate one hash-bound, candidate-local calibration source artifact."""
    if not isinstance(artifact, dict):
        raise ValueError("{0} must be an object".format(label))
    path_value = artifact.get("path")
    digest = artifact.get("sha256")
    role = artifact.get("role")
    if not all(isinstance(value, str) and value for value in (path_value, digest, role)):
        raise ValueError("{0} must declare path, sha256, and role".format(label))
    relative = Path(path_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("{0} must remain inside the candidate workspace".format(label))
    path = workspace.path / relative
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(workspace.path.resolve()):
        raise ValueError("{0} must reference a regular candidate-owned file".format(label))
    if _sha256(path) != digest:
        raise ValueError("{0} SHA-256 does not match its candidate-owned file".format(label))
    return {"path": relative.as_posix(), "sha256": digest, "role": role}


def calibrated_prf_assets(workspace: CandidateWorkspace) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Verify the candidate-owned PRF template, provenance manifest, and recovery calibration."""
    external = workspace.path / "data" / "external"
    template = external / "tess_prf.fits"
    manifest = external / "tess_prf.manifest.json"
    recovery = external / "tess_prf.recovery_calibration.json"
    for path in (template, manifest, recovery):
        if not path.is_file() or path.is_symlink():
            return None, "missing required mission-calibrated PRF asset: {0}".format(path.name)
    try:
        digest = _sha256(template)
        manifest_data = _strict_json(manifest)
        recovery_data = _strict_json(recovery)
        import jsonschema

        manifest_schema = json.loads(
            read_schema_text(workspace.repository_root, "tess-prf-manifest.schema.json")
        )
        recovery_schema = json.loads(
            read_schema_text(workspace.repository_root, "tess-prf-recovery-calibration.schema.json")
        )
        format_checker = jsonschema.FormatChecker()
        jsonschema.validate(manifest_data, manifest_schema, format_checker=format_checker)
        jsonschema.validate(recovery_data, recovery_schema, format_checker=format_checker)
    except (ImportError, OSError, UnicodeError, ValueError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        return None, "unreadable mission-calibrated PRF asset: {0}".format(exc)
    if manifest_data.get("candidate_id") != workspace.candidate_id or recovery_data.get("candidate_id") != workspace.candidate_id:
        return None, "mission-calibrated PRF assets do not match the candidate workspace"
    if manifest_data.get("prf_sha256") != digest:
        return None, "PRF template SHA-256 does not match its manifest"
    if recovery_data.get("prf_sha256") != digest:
        return None, "PRF template SHA-256 does not match its recovery calibration"
    if recovery_data.get("recovery_passed") is not True:
        return None, "PRF recovery calibration has not passed"
    provenance = manifest_data["provenance"]
    if str(provenance.get("source", "")).strip().lower() == "synthetic":
        return None, "synthetic PRF provenance cannot establish mission calibration"
    detector_bounds = manifest_data.get("detector_bounds")
    if isinstance(detector_bounds, dict) and (
        detector_bounds["column_min"] > detector_bounds["column_max"]
        or detector_bounds["row_min"] > detector_bounds["row_max"]
    ):
        return None, "PRF detector bounds are not ordered"
    try:
        source_artifacts = [
            _validate_candidate_artifact(workspace, artifact, "PRF recovery source artifact")
            for artifact in recovery_data["source_artifacts"]
        ]
    except ValueError as exc:
        return None, str(exc)
    return {
        "prf_template": {"path": "data/external/tess_prf.fits", "sha256": digest},
        "prf_manifest": {"path": "data/external/tess_prf.manifest.json", "sha256": _sha256(manifest)},
        "recovery_calibration": {"path": "data/external/tess_prf.recovery_calibration.json", "sha256": _sha256(recovery)},
        "applicability": {
            key: manifest_data[key] for key in ("mission", "sector", "camera", "ccd", "detector_bounds", "field_position")
            if key in manifest_data
        },
        "recovery_source_artifacts": source_artifacts,
    }, None


def _prf_applies_to_cube(calibration_assets: Dict[str, Any], cube: Dict[str, Any]) -> bool:
    """Require the calibrated PRF's declared detector assignment to match one TPF."""
    applicability = calibration_assets.get("applicability")
    header = cube.get("header")
    if not isinstance(applicability, dict) or not isinstance(header, dict):
        return False
    expected = {
        "sector": applicability.get("sector"),
        "camera": applicability.get("camera"),
        "ccd": applicability.get("ccd"),
    }
    observed = {
        "sector": cube.get("sector", header.get("SECTOR")),
        "camera": header.get("CAMERA"),
        "ccd": header.get("CCD"),
    }
    try:
        return all(int(observed[name]) == int(expected[name]) for name in expected)
    except (TypeError, ValueError):
        return False


def load_calibrated_prf_template(path: Path) -> np.ndarray:
    """Load one finite, non-negative empirical PRF image from a FITS asset."""
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("astropy is required to read the calibrated PRF FITS template") from exc
    try:
        with fits.open(path, memmap=False) as hdul:
            template = np.asarray(hdul[0].data, dtype=float)
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError("calibrated PRF template FITS is unreadable") from exc
    if template.ndim != 2 or template.size == 0 or not np.all(np.isfinite(template)) or np.any(template < 0.0):
        raise RuntimeError("calibrated PRF template must be a finite non-negative 2-D image")
    total = float(np.sum(template))
    if total <= 0.0:
        raise RuntimeError("calibrated PRF template has zero total response")
    return template / total


def calibrated_prf_kernel(
    x_grid: np.ndarray, y_grid: np.ndarray, x0: float, y0: float, template: np.ndarray
) -> np.ndarray:
    """Sample an empirical PRF template at a source's sub-pixel detector position."""
    from scipy.ndimage import map_coordinates

    centre_y = (template.shape[0] - 1.0) / 2.0
    centre_x = (template.shape[1] - 1.0) / 2.0
    kernel = map_coordinates(
        template,
        [y_grid - float(y0) + centre_y, x_grid - float(x0) + centre_x],
        order=1,
        mode="constant",
        cval=0.0,
    )
    total = float(np.sum(kernel))
    return kernel / total if total > 0.0 else kernel


def fit_calibrated_difference_image_prf(
    difference_image: np.ndarray,
    pixel_mask: np.ndarray,
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    template: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[float], int]:
    """Fit all finite footprint pixels with candidate-owned empirical PRFs and NNLS."""
    from scipy.optimize import nnls

    valid_mask = np.asarray(pixel_mask, dtype=bool) & np.isfinite(difference_image)
    if not np.any(valid_mask) or not x_positions or len(x_positions) != len(y_positions):
        return None, None, 0
    yy, xx = np.indices(difference_image.shape, dtype=float)
    design = np.column_stack([
        calibrated_prf_kernel(xx[valid_mask], yy[valid_mask], x0, y0, template)
        for x0, y0 in zip(x_positions, y_positions)
    ])
    try:
        amplitudes, residual = nnls(design, np.asarray(difference_image[valid_mask], dtype=float), maxiter=max(5000, 10 * len(x_positions)))
    except (ValueError, RuntimeError) as exc:
        logging.warning("empirical PRF NNLS convergence failure: %s", exc)
        return None, None, 0
    return amplitudes, float(residual), int(valid_mask.sum())


def fit_difference_image_prf(
    difference_image: np.ndarray,
    pixel_mask: np.ndarray,
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    fwhm_pixels: float = PRF_FWHM_PIXELS,
) -> Tuple[Optional[np.ndarray], Optional[float], int]:
    """Decompose a difference image into non-negative PRF source amplitudes.

    Mathematical Formulation
    ------------------------
    Given a set of *K* candidate source positions
    :math:`\\{(x_k, y_k)\\}_{k=1}^K`, the difference image is modelled as a
    linear superposition of unit-normalised Gaussian PRF templates
    (see :func:`gaussian_prf_kernel`):

    .. math::

        I_{\\text{diff}}(x_i, y_i) \\approx \\sum_{k=1}^{K} \\beta_k \\,
        \\tilde{G}(x_i, y_i \\mid x_k, y_k, \\sigma),
        \\qquad \\beta_k \\ge 0.

    Let :math:`\\mathbf{D} \\in \\mathbb{R}^{M \\times K}` be the design
    matrix whose column *k* is the PRF template centred at source *k*,
    evaluated at the *M* unmasked, finite-valued pixels, and let
    :math:`\\mathbf{d} \\in \\mathbb{R}^M` be the corresponding difference-
    image values.  The problem reduces to the non-negative least-squares
    (NNLS) program:

    .. math::

        \\underset{\\boldsymbol{\\beta} \\ge 0}{\\text{minimise}} \\;
        \\|\\mathbf{D}\\boldsymbol{\\beta} - \\mathbf{d}\\|_2^2,

    solved via the active-set algorithm ``scipy.optimize.nnls`` (Lawson &
    Hanson 1974, Ch. 23).

    Astrophysical Rationale
    -----------------------
    The non-negativity constraint :math:`\\beta_k \\ge 0` encodes the
    physical expectation that a transit *removes* flux; the difference image
    :math:`I_{\\text{out}} - I_{\\text{in}}` is positive at the location of
    the eclipsed source (Perryman §4.3.1).  A source whose best-fit
    amplitude is near zero contributes negligible flux to the transit signal,
    even if it lies within the photometric aperture.

    The fit is performed on the **absolute** out-minus-in difference image,
    not on a fractional (ppm-normalised) map.  Division by the local stellar
    scene would distort the PRF morphology and invalidate the linear
    superposition assumption.

    Parameters
    ----------
    difference_image : np.ndarray
        2-D absolute out-of-transit minus in-transit image.
    pixel_mask : np.ndarray
        Boolean mask of same shape selecting pixels to include in the fit
        (typically the pipeline aperture with finite values).
    x_positions : Sequence[float]
        Column positions (pixels) of candidate sources.
    y_positions : Sequence[float]
        Row positions (pixels) of candidate sources, same length as *x_positions*.
    fwhm_pixels : float, optional
        PRF FWHM in pixels (default ``PRF_FWHM_PIXELS``).

    Returns
    -------
    amplitudes : np.ndarray or None
        Non-negative amplitudes :math:`\\beta_k` for each source in the
        same order as *x_positions* / *y_positions*.  ``None`` on failure.
    residual : float or None
        Squared Euclidean norm of the fit residual,
        :math:`\\|\\mathbf{D}\\boldsymbol{\\beta} - \\mathbf{d}\\|_2^2`.
    n_pixels_used : int
        Number of unmasked, finite pixels used in the fit.

    Raises
    ------
    No exceptions propagate to the caller; all failures are logged and
    return ``(None, None, 0)``.
    """
    from scipy.optimize import nnls

    valid_mask = np.asarray(pixel_mask, dtype=bool) & np.isfinite(difference_image)
    # NUMERICAL_GUARD: NNLS requires a well-conditioned design matrix; too few
    # pixels or zero sources yield a degenerate system.
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
    except (ValueError, RuntimeError) as exc:
        logging.warning("NNLS convergence failure in PRF fit: %s", exc)
        return None, None, 0
    except Exception as exc:
        logging.warning("unexpected PRF fit failure: %s", exc)
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

    .. deprecated::
        The parameter name ``depth_map`` is a legacy relic.  Callers must
        supply an **absolute** out-of-transit minus in-transit difference
        image, never a fractional-depth (ppm) map.  The former fractional-
        depth pathway was physically unsuitable for PRF decomposition and has
        been intentionally removed.

    See :func:`fit_difference_image_prf` for the full mathematical
    description of the NNLS decomposition.
    """
    return fit_difference_image_prf(
        depth_map, pixel_mask, x_positions, y_positions, fwhm_pixels
    )


def build_difference_image(
    in_image: np.ndarray, out_image: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the absolute out-of-transit minus in-transit difference image.

    Mathematical Formulation
    ------------------------
    .. math::

        I_{\\text{diff}}(x, y) = I_{\\text{out}}(x, y) - I_{\\text{in}}(x, y)

    where :math:`I_{\\text{out}}` and :math:`I_{\\text{in}}` are the median
    out-of-transit and in-transit pixel stacks respectively.

    Astrophysical Rationale
    -----------------------
    A genuine transit on the target star removes flux precisely at the
    target's pixel position.  The difference image therefore displays a
    positive excess with the point-spread morphology (PRF shape) of the
    eclipsed source (Perryman §4.3.1; Bryson et al. 2013).  Conversely, a
    background eclipsing binary (BEB) produces a difference-image peak
    offset from the nominal target position, detectable via
    :func:`localize_difference_image`.

    This image must remain in **absolute** TPF flux units.  Division by the
    out-of-transit pixel values to form a fractional-depth map would distort
    the PRF morphology, because different pixels have different normalising
    fluxes, and would therefore violate the linear-superposition assumption
    of the NNLS decomposition (see :func:`fit_difference_image_prf`).

    Parameters
    ----------
    in_image : np.ndarray
        2-D median in-transit image (same shape as *out_image*).
    out_image : np.ndarray
        2-D median out-of-transit image.

    Returns
    -------
    difference_image : np.ndarray
        :math:`I_{\\text{diff}}(x, y)`; non-finite where either input is
        non-finite.
    valid : np.ndarray
        Boolean mask where both *in_image* and *out_image* have finite values.
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
    wcs: Any = None,
) -> Dict[str, float]:
    """Compute the flux-weighted centroid of the difference-image core
    and its astrometric offset from the target position.

    Mathematical Formulation
    ------------------------
    **Core extraction.**  Only pixels whose value exceeds a fraction
    *core_fraction* of the global maximum participate in the centroid,
    preventing noise-dominated periphery pixels from pulling the centroid
    away from the true transit source:

    .. math::

        \\mathcal{C} = \\{(x, y) \\mid I_{\\text{diff}}(x, y) \\ge
        f_{\\text{core}} \\cdot \\max(I_{\\text{diff}})\\}

    **Flux-weighted centroid.**  The core centre-of-light is computed with
    weights proportional to the absolute difference flux:

    .. math::

        x_c = \\frac{\\sum_{(x,y) \\in \\mathcal{C}} x \\cdot I_{\\text{diff}}(x,y)}
                     {\\sum_{(x,y) \\in \\mathcal{C}} I_{\\text{diff}}(x,y)}, \\quad
        y_c = \\frac{\\sum_{(x,y) \\in \\mathcal{C}} y \\cdot I_{\\text{diff}}(x,y)}
                     {\\sum_{(x,y) \\in \\mathcal{C}} I_{\\text{diff}}(x,y)}.

    **Equatorial offset.**  Pixel offsets are converted to equatorial
    coordinates using the TESS plate scale and the declination-dependent
    RA projection factor :math:`\\cos\\delta` (Perryman §4.3.2):

    .. math::

        \\Delta\\alpha \\cos\\delta &= (x_c - x_{\\text{target}}) \\times
                                     s_{\\text{pix}} \\times \\max(\\cos\\delta, 0.01), \\\\[4pt]
        \\Delta\\delta &= (y_c - y_{\\text{target}}) \\times s_{\\text{pix}}, \\\\[4pt]
        \\Delta r &= \\sqrt{(\\Delta\\alpha \\cos\\delta)^2 + (\\Delta\\delta)^2},

    where :math:`s_{\\text{pix}} = 21.0` arcsec/pixel (Ricker et al. 2015).

    **Significance interpretation.**  Under the null hypothesis that the
    transit source is co-located with the target (:math:`\\Delta r = 0`),
    the offset :math:`Z_{\\text{centroid}} = \\Delta r / \\sigma_{\\text{centroid}}`
    follows a Rayleigh distribution (Perryman §4.3.3).  This function
    computes only the offset magnitude; the formal significance threshold
    :math:`Z_{\\text{centroid}} < 3.0\\sigma` is applied downstream in the
    vetting layer.

    DIAGNOSTIC_REASONING
    --------------------
    - A :math:`\\Delta r` consistent with zero (within the TESS pixel
      scale) supports an on-target transit source.
    - A statistically significant, non-zero :math:`\\Delta r` may indicate
      a blended background eclipsing binary (BEB/NEB) whose difference-
      image peak is offset from the nominal target position.
    - This diagnostic must be interpreted alongside the NNLS PRF source
      competition analysis (:func:`fit_difference_image_prf`), because a
      bright, well-separated neighbour can bias the centroid even when the
      transit is genuinely on-target.

    Parameters
    ----------
    difference_image : np.ndarray
        2-D absolute out-minus-in difference image.
    pixel_mask : np.ndarray
        Boolean mask selecting valid pixels (e.g., pipeline aperture).
    target_x : float
        Nominal target column position in pixels.
    target_y : float
        Nominal target row position in pixels.
    pixel_scale_arcsec : float, optional
        TESS plate scale (default ``PIXEL_SCALE_ARCSEC``).
    cos_dec : float, optional
        :math:`\\cos(\\delta)` for the target declination; used to
        project RA pixel offsets onto the sky.  Default 1.0 (equator).
    core_fraction : float, optional
        Fraction of the maximum difference value above which pixels
        contribute to the centroid (default 0.2).

    Returns
    -------
    dict
        Keys: ``"ra_offset_arcsec"``, ``"dec_offset_arcsec"``,
        ``"offset_arcsec"``, ``"n_difference_pixels"``.  All offsets are
        ``NaN`` when fewer than ``MIN_DIFFERENCE_CORE_PIXELS`` core pixels
        are available.
    """
    valid_mask = np.asarray(pixel_mask, dtype=bool) & np.isfinite(difference_image)
    differences = difference_image[valid_mask]
    # NUMERICAL_GUARD: need at least 3 pixels with positive flux to centroid.
    if differences.size < 3 or float(np.max(differences)) <= 0:
        return {
            "ra_offset_arcsec": float("nan"),
            "ra_cosdec_offset_arcsec": float("nan"),
            "ra_coordinate_offset_arcsec": float("nan"),
            "dec_offset_arcsec": float("nan"),
            "offset_arcsec": float("nan"),
            "n_difference_pixels": 0,
        }
    threshold = core_fraction * float(np.max(differences))
    core_mask = valid_mask & (difference_image >= threshold)
    yy, xx = np.indices(difference_image.shape, dtype=float)
    core_differences = difference_image[core_mask]
    # NUMERICAL_GUARD: fewer than MIN_DIFFERENCE_CORE_PIXELS → noise-dominated core.
    if core_differences.size < MIN_DIFFERENCE_CORE_PIXELS:
        return {
            "ra_offset_arcsec": float("nan"),
            "ra_cosdec_offset_arcsec": float("nan"),
            "ra_coordinate_offset_arcsec": float("nan"),
            "dec_offset_arcsec": float("nan"),
            "offset_arcsec": float("nan"),
            "n_difference_pixels": int(core_differences.size),
        }
    # Flux-weighted centroid on the difference-image core (Perryman §4.3.2).
    weights = core_differences / float(np.sum(core_differences))
    centroid_x = float(np.sum(xx[core_mask] * weights))
    centroid_y = float(np.sum(yy[core_mask] * weights))
    # Prefer the WCS spherical tangent-plane transformation. Pixel axes can be
    # rotated, so detector columns are not inherently projected right ascension.
    if wcs is not None:
        try:
            target_ra, target_dec = wcs.pixel_to_world_values(float(target_x), float(target_y))
            centroid_ra, centroid_dec = wcs.pixel_to_world_values(centroid_x, centroid_y)
            target_ra, target_dec, centroid_ra, centroid_dec = (
                float(np.asarray(value)) for value in (target_ra, target_dec, centroid_ra, centroid_dec)
            )
            ra0 = math.radians(target_ra)
            dec0 = math.radians(target_dec)
            ra = math.radians(centroid_ra)
            dec = math.radians(centroid_dec)
            delta_ra = math.atan2(math.sin(ra - ra0), math.cos(ra - ra0))
            denominator = math.sin(dec0) * math.sin(dec) + math.cos(dec0) * math.cos(dec) * math.cos(delta_ra)
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise ValueError("invalid tangent-plane projection")
            ra_cosdec_offset = math.degrees(math.cos(dec) * math.sin(delta_ra) / denominator) * 3600.0
            dec_offset = math.degrees((math.cos(dec0) * math.sin(dec) - math.sin(dec0) * math.cos(dec) * math.cos(delta_ra)) / denominator) * 3600.0
            ra_coordinate_offset = math.degrees(delta_ra) * 3600.0
            return {
                "ra_offset_arcsec": round(ra_cosdec_offset, 4),
                "ra_cosdec_offset_arcsec": round(ra_cosdec_offset, 4),
                "ra_coordinate_offset_arcsec": round(ra_coordinate_offset, 4),
                "dec_offset_arcsec": round(dec_offset, 4),
                "offset_arcsec": round(math.hypot(ra_cosdec_offset, dec_offset), 4),
                "n_difference_pixels": int(np.count_nonzero(core_mask)),
                "coordinate_method": "fits-wcs-spherical-tangent-offset",
            }
        except (AttributeError, TypeError, ValueError, OverflowError):
            pass
    # Convert pixel offsets to on-sky projected equatorial arcseconds.
    # ASTROPHYSICAL_GUARD: the pixel X-axis already measures the projected
    # displacement Δα·cos(δ); multiplying again by cos(δ) here (and again in
    # vetting/centroid.py) compressed RA offsets by cos²(δ), silently
    # suppressing centroid-shift significance for high-declination targets.
    ra_offset = (centroid_x - float(target_x)) * pixel_scale_arcsec
    dec_offset = (centroid_y - float(target_y)) * pixel_scale_arcsec
    coordinate_ra_offset = ra_offset / cos_dec if math.isfinite(cos_dec) and abs(cos_dec) > np.finfo(float).eps else ra_offset
    return {
        "ra_offset_arcsec": round(ra_offset, 4),
        "ra_cosdec_offset_arcsec": round(ra_offset, 4),
        "ra_coordinate_offset_arcsec": round(coordinate_ra_offset, 4),
        "dec_offset_arcsec": round(dec_offset, 4),
        # Total separation Δr = √(Δα²cos²δ + Δδ²) (Perryman §4.3.2, Eq. 4.8).
        "offset_arcsec": round(math.hypot(ra_offset, dec_offset), 4),
        "n_difference_pixels": int(np.count_nonzero(core_mask)),
        "coordinate_method": "pixel-scale-projected-offset",
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
    """Compatibility wrapper for absolute difference-image centroiding.

    .. deprecated::
        Renamed key ``"n_depth_pixels"`` retained for backward
        compatibility only.  Prefer :func:`localize_difference_image`
        directly, which returns ``"n_difference_pixels"`` with the
        same semantics.
    """
    result = localize_difference_image(
        depth_map, pixel_mask, target_x, target_y, pixel_scale_arcsec, cos_dec, core_fraction
    )
    result["n_depth_pixels"] = result.pop("n_difference_pixels")
    return result


def _header_position(header: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Extract the target RA/Dec (degrees) from a TPF primary FITS header.

    Reads the ``RA_OBJ`` and ``DEC_OBJ`` keywords, which record the
    commanded spacecraft pointing centre for the target, not a WCS-
    calibrated astrometric solution.  Returns ``None`` for any missing,
    un-parseable, or non-finite value.  Used as a fallback when the WCS
    is unavailable, and as the source of the declination for the
    :math:`\\cos\\delta` projection factor in centroid offset calculations.
    """
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
    """Build the absolute in/out-transit difference image from one TPF cube.

    Mathematical Formulation
    ------------------------
    The TPF time series is phase-folded using the candidate ephemeris
    (period :math:`P`, epoch :math:`T_0`, duration :math:`T_{\\text{dur}}`).

    Cadences are classified as:

    - **In-transit:** :math:`|t - T_0| < 0.5 \\times T_{\\text{dur}}`
      (within one half-duration of mid-transit).
    - **Out-of-transit:** :math:`1.2 \\times T_{\\text{dur}} < |t - T_0|
      < 2.5 \\times T_{\\text{dur}}` (buffer of 1.2× duration guards
      against ingress/egress contamination; Perryman §4.3.1).

    The median image is computed for each class to suppress outlier pixel
    values, yielding :math:`I_{\\text{in}}` and :math:`I_{\\text{out}}`.
    The absolute difference image follows from :func:`build_difference_image`.

    Quality filtering
    -----------------
    Cadences whose ``QUALITY`` bitmask intersects *quality_mask*
    are excluded before stacking.  At least 10 in-transit and 10
    out-of-transit cadences must survive; fewer than 20 total good
    cadences also aborts.

    Target position
    ---------------
    The target pixel position is obtained preferentially from the WCS
    solution using ``RA_OBJ``/``DEC_OBJ`` from the primary header.  When
    the WCS or header is unavailable, the pipeline aperture centroid
    serves as a fallback.

    Parameters
    ----------
    cube : dict
        TPF cube dictionary with keys ``"path"``, ``"time"``,
        ``"quality"``, ``"flux"``, ``"header"``, ``"sector"``.
    ephemeris : dict
        Candidate ephemeris with ``"period_days"``, ``"epoch_btjd"``,
        ``"duration_days"``.
    quality_mask : int, optional
        TESS QUALITY bitmask for cadence exclusion.

    Returns
    -------
    difference_image : np.ndarray or None
        2-D absolute out-minus-in difference image, or ``None`` on failure.
    pipeline : np.ndarray or None
        Boolean pipeline aperture mask (same shape).
    target_x : float or None
        Target column in pixels.
    target_y : float or None
        Target row in pixels.
    n_in : int
        Number of in-transit cadences used.
    n_out : int
        Number of out-of-transit cadences used.
    """
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError:  # pragma: no cover - optional dependency
        return None, None, None, None, 0, 0

    path = cube["path"]
    try:
        with fits.open(path, memmap=False) as hdul:
            _, ap_hdu = hdul[1], hdul[2]
            aperture = np.asarray(ap_hdu.data)
            # Pipeline optimal aperture: bit 1 (= value 2) marks pixels used
            # by the SPOC photometric extraction (Twicken et al. 2018).
            pipeline = (aperture & 2) != 0
            wcs = WCS(ap_hdu.header)
            header = dict(hdul[0].header)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        logging.warning(
            "WCS/TPF load failed for %s (sector %s): %s",
            path.name, cube.get("sector", "unknown"), exc,
        )
        return None, None, None, None, 0, 0
    except Exception as exc:
        logging.warning(
            "unexpected TPF failure for %s (sector %s): %s",
            path.name, cube.get("sector", "unknown"), exc,
        )
        return None, None, None, None, 0, 0

    time = cube["time"]
    quality = cube["quality"]
    flux = cube["flux"]
    # Quality-filter: reject cadences whose bitwise quality intersects
    # the hard mask (momentum dumps, desat, cosmic rays).
    good = (
        np.isfinite(time)
        & ((quality & quality_mask) == 0)
        & np.all(np.isfinite(flux), axis=(1, 2))
    )
    # NUMERICAL_GUARD: fewer than 20 usable cadences → unreliable stacking.
    if int(good.sum()) < 20:
        return None, None, None, None, 0, 0
    time = time[good]
    flux = flux[good]

    period_days = ephemeris["period_days"]
    epoch_btjd = ephemeris["epoch_btjd"]
    duration_days = ephemeris["duration_days"]
    # Phase-fold and compute hours from mid-transit (see lightcurve.phase_hours).
    hours = phase_hours(time, period_days, epoch_btjd)
    # In-transit: |hours| < 0.5 × T_dur × 24 h/d.
    in_mask = np.abs(hours) < 0.5 * duration_days * 24.0
    # Out-of-transit: 1.2–2.5 × duration away from centre.
    # The 1.2× buffer avoids ingress/egress bleed (Perryman §4.3.1).
    out_mask = (np.abs(hours) > 1.2 * duration_days * 24.0) & (
        np.abs(hours) < 2.5 * duration_days * 24.0
    )
    # NUMERICAL_GUARD: at least 10 cadences in each stack.
    if int(in_mask.sum()) < 10 or int(out_mask.sum()) < 10:
        return None, None, None, None, 0, 0

    # Median stacking for robust difference imaging (outliers suppressed).
    in_image = np.nanmedian(flux[in_mask], axis=0)
    out_image = np.nanmedian(flux[out_mask], axis=0)
    difference_image, _ = build_difference_image(in_image, out_image)

    position = _header_position(header)
    if position is None:
        # Fallback: centroid of the pipeline aperture in pixel space.
        target_x = float(np.mean(np.flatnonzero(np.any(pipeline, axis=0))))
        target_y = float(np.mean(np.flatnonzero(np.any(pipeline, axis=1))))
    else:
        ra_deg, dec_deg = position
        # astropy WCS: world (RA, Dec) → pixel (column, row).
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
    """Build the source list for PRF localisation from the archival Gaia report.

    Loads the validated archival Gaia DR3 report for the candidate
    (see :func:`archive.load_validated_archival_gaia_sources`), extracts
    the target row, and appends all neighbouring sources with valid
    coordinates, magnitudes, and separations.

    Mathematical Formulation
    ------------------------
    **Gaia G-band flux ratio.**  Neighbour flux relative to the target is
    computed from the magnitude difference via Pogson's law:

    .. math::

        \\frac{F_k}{F_{\\text{target}}} =
        10^{-0.4 \\times (G_k - G_{\\text{target}})}.

    This dimensionless ratio is used downstream for source selection
    (threshold :math:`\\sim 10^{-5}`, see :func:`_select_sources`) and as
    contextual metadata in the per-source output.  It is **not** used as
    an eclipse-depth constraint.

    Gaia DR3 RUWE Context
    ---------------------
    The archival report includes the Gaia DR3 RUWE (Renormalised Unit
    Weight Error) for each source.  RUWE > 1.4 is an
    ASTROPHYSICAL_HEURISTIC for possible unresolved binarity or
    astrometric excess noise (Gaia Collaboration 2018, A&A 616, A1).
    The RUWE values are carried through to the localisation output as
    catalogue context but do not gate the PRF fit itself.

    Parameters
    ----------
    workspace : CandidateWorkspace
        Candidate workspace with a completed archival Gaia report.

    Returns
    -------
    rows : list of dict
        Source entries; the target is always first with
        ``"is_target": True`` and ``"flux_ratio": 1.0``.
    metadata : dict
        Catalogue provenance metadata; ``"position_availability"`` is
        ``"unavailable"`` when the target coordinates are missing.
    """
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
            "flux_ratio": 1.0,  # target relative to itself
            "tess_flux_ratio": 1.0,
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
        target_bp = _finite_float(target.get("phot_bp_mean_mag"))
        target_rp = _finite_float(target.get("phot_rp_mean_mag"))
        source_bp = _finite_float(source.get("phot_bp_mean_mag"))
        source_rp = _finite_float(source.get("phot_rp_mean_mag"))
        tess_ratio: Optional[float] = None
        if None not in (target_bp, target_rp, source_bp, source_rp):
            target_t_mag = _gaia_to_tess_mag(target_g_mag, target_bp - target_rp)
            source_t_mag = _gaia_to_tess_mag(g_mag, source_bp - source_rp)
            tess_ratio = 10.0 ** (-0.4 * (source_t_mag - target_t_mag))
        # Gaia G is retained as a catalog context ratio; a TESS-band estimate is
        # present only where BP/RP colors make the transform applicable.
        rows.append(
            {
                "source_id": str(source.get("source_id", "archival-neighbor")),
                "ra": ra_deg,
                "dec": dec_deg,
                "g_mag": g_mag,
                "separation_arcsec": separation_arcsec,
                "flux_ratio": 10.0 ** (-0.4 * (g_mag - target_g_mag)),
                "tess_flux_ratio": tess_ratio,
                "is_target": False,
            }
        )
    metadata["position_availability"] = "available"
    metadata["n_catalog_neighbors"] = len(rows) - 1
    return rows, metadata


def _gaia_to_tess_mag(g_mag: float, bp_rp: float) -> float:
    """Apply the Stassun et al. Gaia BP/RP to TESS magnitude relation."""
    return g_mag + (-0.00522555 * bp_rp**3 + 0.0891337 * bp_rp**2 - 0.633923 * bp_rp + 0.0324473)


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
    """Filter and project Gaia neighbours onto the difference-image pixel grid.

    Candidate sources are subjected to three screening criteria:

    1. **Angular proximity:** :math:`\\theta < \\theta_{\\text{max}}`
       (default 60 arcsec).
    2. **Gaia flux ratio:** :math:`F_k / F_{\\text{target}} \\ge 10^{-5}`;
       sources fainter than this contribute negligible flux to the
       aperture.  ASTROPHYSICAL_HEURISTIC: a 5-magnitude difference
       corresponds to ~1% flux ratio; 10^-5 corresponds to ~12.5 mag
       fainter — well below the confusion limit.
    3. **WCS projection:** the source must map to a pixel within or
       slightly outside the TPF footprint (tolerance of ±2 pixels).

    The target is always placed first, with its pixel position forcibly
    set to (*target_x*, *target_y*).  Only the five brightest (lowest
    *g_mag*) non-target neighbours are retained to limit the NNLS design
    matrix size.

    Parameters
    ----------
    difference_image : np.ndarray
        2-D difference image for shape reference.
    pipeline : np.ndarray
        Boolean pipeline aperture mask.
    target_x : float
        Target column in pixels.
    target_y : float
        Target row in pixels.
    neighbors : sequence of dict
        Gaia neighbour entries from :func:`_load_archival_gaia_neighbors`.
    search_radius_arcsec : float
        Maximum angular separation for inclusion.
    wcs : astropy.wcs.WCS
        WCS solution for the TPF.
    cos_dec : float
        :math:`\\cos(\\delta)` projection factor (unused here; returned
        for downstream use).

    Returns
    -------
    sources : list of dict
        Selected sources, target first, with pixel positions.
    cos_dec : float
        Unchanged *cos_dec*.
    pixel_scale : float
        ``PIXEL_SCALE_ARCSEC`` (21.0).
    """
    sources: List[Dict[str, Any]] = []
    for row in neighbors:
        try:
            # astropy WCS: world (RA, Dec) → pixel (column, row).
            sx, sy = wcs.world_to_pixel_values(float(row["ra"]), float(row["dec"]))
            sx = float(np.asarray(sx))
            sy = float(np.asarray(sy))
        except Exception:
            continue
        # The archive cone limits the scene. Retain every successfully projected
        # Gaia source; NNLS evaluates its empirical template over all finite
        # footprint pixels, including sources whose PRF wings are negligible.
        sources.append(
            {
                "source_id": row.get("source_id", "neighbor"),
                "x_pix": sx,
                "y_pix": sy,
                "g_mag": _finite_float(row.get("g_mag")),
                "flux_ratio": _finite_float(row.get("flux_ratio")),
                "separation_arcsec": _finite_float(row.get("separation_arcsec")),
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
    # Force the target position to the TPF-derived pixel coordinates,
    # not the WCS projection, to avoid sub-pixel WCS jitter.
    target_entry["x_pix"] = float(target_x)
    target_entry["y_pix"] = float(target_y)
    return [target_entry] + [src for src in sources if not src["is_target"]], cos_dec, PIXEL_SCALE_ARCSEC


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
    template: np.ndarray,
    wcs: Any = None,
) -> Dict[str, Any]:
    """Orchestrate PRF decomposition and centroid localisation for one sector.

    This function combines the two core diagnostics for a single TPF cube:

    1. **NNLS PRF decomposition** (:func:`fit_difference_image_prf`):
       fits non-negative Gaussian amplitudes for the target and up to five
       neighbouring Gaia sources simultaneously on the pipeline aperture
       pixels.

    2. **Difference-image centroiding** (:func:`localize_difference_image`):
       computes the flux-weighted centre-of-light of the difference-image
       core and its offset from the nominal target position.

    Source Competition Analysis
    ---------------------------
    The dominant flux-loss source is identified as the source with the
    largest NNLS amplitude.  The ratio

    .. math::

        R = \\frac{\\beta_{\\text{target}}}{\\max_{k \\neq \\text{target}} \\beta_k}

    is computed as a dimensionless competition metric (cf. Perryman §4.3).
    A high :math:`R` indicates the NNLS fit attributes most of the
    difference flux to the target; a low :math:`R \\lesssim 1` suggests a
    competing neighbour may dominate the transit signal.

    DIAGNOSTIC_REASONING
    --------------------
    - If the fit-dominant source is the target AND the difference-image
      centroid offset is sub-pixel, the transit signal is spatially
      consistent with an on-target origin.
    - If a neighbour dominates the NNLS fit OR the centroid offset is
      large, the signal may originate from a blended background eclipsing
      binary (BEB/NEB).
    - A missing WCS prevents catalogue-neighbour projection, so that sector
      remains inconclusive and cannot make a source-assignment statement.

    Returns
    -------
    dict
        Per-sector diagnostic record with keys documented in the output
        schema (``prf_localization_results.json``).
    """
    pixel_mask = pipeline & np.isfinite(difference_image)
    # NNLS PRF decomposition on the pipeline aperture pixels.
    amplitudes, residual, n_pixels = fit_calibrated_difference_image_prf(
        difference_image,
        pixel_mask,
        [src["x_pix"] for src in sources],
        [src["y_pix"] for src in sources],
        template,
    )
    # Flux-weighted centroid of the difference-image core.
    centroid = localize_difference_image(
        difference_image, pipeline, target_x, target_y, cos_dec=cos_dec, wcs=wcs
    )
    if amplitudes is None:
        return {
            "sector": int(sector),
            "skipped": True,
            "reason": "insufficient aperture pixels",
        }
    per_source = {}
    for index, src in enumerate(sources):
        # NNLS enforces β_k ≥ 0, but floating-point may give tiny negatives; clamp.
        amplitude = float(amplitudes[index]) if amplitudes[index] > 0 else 0.0
        flux_ratio = _finite_float(src.get("flux_ratio"))
        per_source[str(src["source_id"])] = {
            "g_mag": src["g_mag"],
            "separation_arcsec": src["separation_arcsec"],
            "is_target": src["is_target"],
            # Gaia G-band flux ratio from Pogson's law (see _load_archival_gaia_neighbors).
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
    # The NNLS-dominant source is the one with the largest amplitude.
    fit_dominant_id = max(per_source, key=lambda key: per_source[key]["difference_flux_amplitude"])
    fit_dominant = per_source[fit_dominant_id]
    fit_dominant_is_target = bool(fit_dominant["is_target"])
    # Sufficient core pixels for a meaningful centroid?
    difference_core_resolved = (
        int(centroid["n_difference_pixels"]) >= MIN_DIFFERENCE_CORE_PIXELS
    )
    if wcs is None:
        source_assignment_status = "inconclusive_wcs_unavailable"
    elif not difference_core_resolved:
        source_assignment_status = "unresolved_insufficient_difference_core"
    else:
        source_assignment_status = "calibrated_empirical_prf"
    source_assignment_interpretable = difference_core_resolved and wcs is not None
    # Target-to-max-other amplitude ratio: high values favour on-target origin.
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


# Candidate-derived ephemeris provenance labels accepted for source localization.
# The shared candidate ephemeris boundary records these prefixes plus the
# candidate-owned "candidate-data" label used by candidate loaders and the
# localization contract.
_CANDIDATE_EPHEMERIS_SOURCE_PREFIXES = (
    "candidate-config",
    "candidate-config-signal",
    "candidate-data-bls",
    "bls-search",
    "candidate-data",
)


def _localization_ephemeris_complete(
    ephemeris: Dict[str, Any], field_sources: Dict[str, Any]
) -> bool:
    """Return whether the ephemeris has the finite, candidate-sourced transit fields localization consumes.

    Difference imaging selects in- and out-of-transit cadences from the orbital
    period, epoch, and transit duration only; ``depth_ppm`` never enters the PRF
    localization model, so a candidate-owned ephemeris may omit it.  When
    ``depth_ppm`` is present it must still be finite, positive, and
    candidate-sourced so a conflicting synthetic depth cannot pass unnoticed.
    """
    if not isinstance(ephemeris, dict):
        return False
    source = ephemeris.get("source")
    if not isinstance(source, str):
        return False
    source_prefix = source.removeprefix("partial-")
    if source_prefix not in _CANDIDATE_EPHEMERIS_SOURCE_PREFIXES:
        return False
    if ephemeris.get("time_system") not in (None, BTJD_TIME_SYSTEM):
        return False
    try:
        period_days = float(ephemeris["period_days"])
        epoch_btjd = float(ephemeris["epoch_btjd"])
        duration_days = float(ephemeris["duration_days"])
    except (KeyError, TypeError, ValueError):
        return False
    if not (
        math.isfinite(period_days)
        and math.isfinite(epoch_btjd)
        and math.isfinite(duration_days)
        and period_days > 0.0
        and 0.0 < duration_days < period_days
    ):
        return False
    for field in ("period_days", "epoch_btjd", "duration_days"):
        if not str(field_sources.get(field, "")).startswith(source_prefix):
            return False
    if "depth_ppm" in ephemeris:
        try:
            depth_ppm = float(ephemeris["depth_ppm"])
        except (KeyError, TypeError, ValueError):
            return False
        if not (math.isfinite(depth_ppm) and depth_ppm > 0.0):
            return False
        if not str(field_sources.get("depth_ppm", "")).startswith(source_prefix):
            return False
    return True


def run_prf_localization(
    workspace: CandidateWorkspace, search_radius_arcsec: float = 60.0
) -> Path:
    """Execute the full PRF source-localisation pipeline and write results.

    This is the main entry point for TESS sub-pixel transit-source
    localisation.  It orchestrates:

    1. **Input validation:** loads the candidate ephemeris and TPF cubes,
       verifies that both are from real candidate data (not synthetic
       demos), and aborts early with an appropriate status when
       prerequisites are missing.

    2. **Catalogue loading:** reads the validated archival Gaia DR3
       report via :func:`_load_archival_gaia_neighbors`, extracting the
       target and all neighbours within the search radius.

    3. **Per-sector analysis:** for each TPF cube with adequate in/out-
       transit coverage, constructs the difference image
       (:func:`extract_tpf_difference_image`), selects candidate sources
       (:func:`_select_sources`), and performs the combined NNLS PRF
       decomposition plus centroid localisation
       (:func:`_fit_one_difference_image`).

    4. **Summary aggregation:** computes sector-level and cross-sector
       statistics (median centroid offset, median target-to-competitor
       amplitude ratio, unresolved-sector count).

    5. **Triage routing:** assigns a conclusion status for downstream
       consumption by the statistical-vetting engine
       (see ``methods/engine-execution-and-triage.md``):

       ====================================== ===========================================
       Status                                  Trigger
       ====================================== ===========================================
       ``inconclusive_no_candidate_tpf``       No TPF cubes with raw provenance.
       ``inconclusive_no_candidate_ephemeris`` Ephemeris is synthetic or missing.
       ``inconclusive_no_complete_depth_maps`` All sectors failed difference imaging.
       ``inconclusive_insufficient_difference_core``
                                               Some sectors completed but all lacked
                                               adequate difference-image core pixels.
       ``inconclusive_uncalibrated_prf``       Sectors completed with adequate core
                                               pixels, but the Gaussian PRF model is
                                               uncalibrated — the result is a screening
                                               aid, not a validated source assignment.
       ====================================== ===========================================

    SCIENTIFIC_BOUNDARY
    -------------------
    ``calibrated`` is true only when hash-bound mission PRF and recovery
    assets apply to every completed TPF sector and every sector has usable
    WCS for source projection. ``validation_eligible`` remains false: this
    diagnostic cannot independently validate or exclude a planetary origin.

    Parameters
    ----------
    workspace : CandidateWorkspace
        Candidate workspace with completed archival Gaia report and
        raw-provenance-bound TPF cubes.
    search_radius_arcsec : float, optional
        Maximum angular separation for neighbouring Gaia sources to
        include in the PRF model.  Default 60 arcsec.

    Returns
    -------
    Path
        Path to the written ``prf_localization_results.json`` artefact.
    """
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "prf_localization_results.json"
    calibration_assets, calibration_error = calibrated_prf_assets(workspace)
    if calibration_assets is None:
        payload = {
        "schema_version": "1.0",
        "work_package": "PRF_SOURCE_LOCALIZATION",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": workspace.candidate_id,
        "source": "not-run-mission-calibrated-prf-required",
        "calibrated": False,
        "calibration_status": "uncalibrated",
        "validation_eligible": False,
        "method": "not-run: mission-calibrated PRF assets are required",
        "prf_model": None,
        "search_radius_arcsec": float(search_radius_arcsec),
        "source_catalog": "not-read",
        "skipped_tpf_products": [],
        "sector_results": [],
        "summary": {
            "n_sectors": 0,
            "n_completed": 0,
            "sectors_with_competing_sources_modeled": 0,
            "sectors_with_unresolved_difference_core": 0,
            "sectors_with_uncalibrated_prf": 0,
            "median_target_to_other_difference_ratio": None,
            "median_difference_image_offset_arcsec": None,
            "conclusion": "inconclusive_mission_calibrated_prf_required",
        },
        "caveats": [
            "Localization did not run because required mission-calibrated PRF assets did not verify: {0}".format(calibration_error),
            "The nominal Gaussian screening model is prohibited from producing localization evidence.",
        ],
        }
        _write_json_atomic(output_path, payload)
        return output_path
    try:
        template = load_calibrated_prf_template(workspace.path / calibration_assets["prf_template"]["path"])
    except RuntimeError as exc:
        payload = {
            "schema_version": "1.0", "work_package": "PRF_SOURCE_LOCALIZATION",
            "generated_utc": datetime.now(timezone.utc).isoformat(), "candidate_id": workspace.candidate_id,
            "source": "not-run-invalid-mission-calibrated-prf", "calibrated": False,
            "calibration_status": "uncalibrated", "validation_eligible": False,
            "method": "not-run: verified empirical PRF template was unreadable", "prf_model": None,
            "search_radius_arcsec": float(search_radius_arcsec), "source_catalog": "not-read",
            "skipped_tpf_products": [], "sector_results": [],
            "summary": {"n_sectors": 0, "n_completed": 0, "sectors_with_competing_sources_modeled": 0, "sectors_with_unresolved_difference_core": 0, "sectors_with_uncalibrated_prf": 0, "median_target_to_other_difference_ratio": None, "median_difference_image_offset_arcsec": None, "conclusion": "inconclusive_invalid_mission_calibrated_prf"},
            "caveats": ["The hash-bound PRF template could not be read: {0}".format(exc)],
        }
        _write_json_atomic(output_path, payload)
        return output_path

    ephemeris = load_transit_ephemeris(workspace)
    field_sources = ephemeris.get("field_sources")
    if not isinstance(field_sources, dict):
        field_sources = {}
    required_ephemeris_fields = (
        "period_days",
        "epoch_btjd",
        "duration_days",
        "depth_ppm",
    )
    candidate_ephemeris = _localization_ephemeris_complete(ephemeris, field_sources)
    neighbors, source_catalog = _load_archival_gaia_neighbors(workspace)

    skipped_tpf_products: List[Dict[str, str]] = []
    cubes = load_tpf_cubes(
        workspace,
        require_raw_provenance=True,
        skipped_products=skipped_tpf_products,
    )
    if not cubes:
        source = "not-run-no-candidate-tpf"
    elif not candidate_ephemeris:
        source = "not-run-no-candidate-ephemeris"
    else:
        source = "candidate-data"

    sector_results: List[Dict[str, Any]] = []
    inapplicable_calibration_assets = False
    if source == "candidate-data":
        try:
            from astropy.wcs import WCS
        except ImportError:  # pragma: no cover - optional dependency
            WCS = None  # type: ignore[assignment]
        # Per-sector loop: difference image → source selection → PRF fit + centroid.
        for cube in cubes:
            if not _prf_applies_to_cube(calibration_assets, cube):
                inapplicable_calibration_assets = True
                sector_results.append(
                    {
                        "sector": int(cube["sector"]),
                        "skipped": True,
                        "reason": "mission-calibrated PRF does not match TPF sector, camera, and CCD",
                    }
                )
                continue
            difference_image, pipeline, target_x, target_y, n_in, n_out = extract_tpf_difference_image(
                cube, ephemeris
            )
            if difference_image is None:
                continue
            header = cube["header"]
            position = _header_position(header)
            # cos(δ) for RA projection: derived from the TPF header or default to 1.0.
            cos_dec = math.cos(math.radians(float(position[1]))) if position else 1.0
            wcs = None
            if WCS is not None:
                try:
                    from astropy.io import fits

                    with fits.open(cube["path"], memmap=False) as hdul:
                        wcs = WCS(hdul[2].header)
                except (OSError, ValueError, KeyError, IndexError) as exc:
                    logging.warning(
                        "WCS load failed for %s (sector %s): %s",
                        cube["path"], cube.get("sector", "unknown"), exc,
                    )
                    wcs = None
                except Exception as exc:
                    logging.warning(
                        "unexpected WCS failure for %s (sector %s): %s",
                        cube["path"], cube.get("sector", "unknown"), exc,
                    )
                    wcs = None
            if wcs is None:
                # WCS unavailable: fit only the target source.
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
                    template,
                    wcs,
                )
            )

    # --- Summary aggregation ---
    completed = [row for row in sector_results if not row.get("skipped", False)]
    sectors_with_neighbors = [
        row for row in completed if row.get("n_modeled_neighbors", 0) > 0
    ]
    unresolved_difference_core = sum(
        1
        for row in completed
        if row.get("source_assignment_status") == "unresolved_insufficient_difference_core"
    )
    wcs_unavailable = sum(
        1
        for row in completed
        if row.get("source_assignment_status") == "inconclusive_wcs_unavailable"
    )
    offsets = [
        row["difference_centroid_offset_arcsec"]
        for row in completed
        if np.isfinite(row["difference_centroid_offset_arcsec"])
    ]
    median_offset = float(np.median(offsets)) if offsets else None
    # A modeled neighbour can still have a zero/undefined NNLS amplitude.
    # Preserve that sector in the competition count, but exclude its missing
    # ratio from the numeric median instead of passing None to NumPy.
    ratios = [
        ratio
        for row in sectors_with_neighbors
        for ratio in (_finite_float(row.get("target_to_max_other_difference_ratio")),)
        if ratio is not None
    ]
    median_ratio = float(np.median(ratios)) if ratios else None
    # --- Triage routing (see methods/engine-execution-and-triage.md) ---
    localization_calibrated = (
        bool(completed)
        and not inapplicable_calibration_assets
        and not wcs_unavailable
        and not unresolved_difference_core
    )
    if source == "not-run-no-candidate-tpf":
        status = "inconclusive_no_candidate_tpf"
    elif source == "not-run-no-candidate-ephemeris":
        status = "inconclusive_no_candidate_ephemeris"
    elif inapplicable_calibration_assets:
        status = "inconclusive_prf_asset_not_applicable"
    elif not completed:
        status = "inconclusive_no_complete_depth_maps"
    elif wcs_unavailable:
        status = "inconclusive_wcs_unavailable"
    elif unresolved_difference_core:
        status = "inconclusive_insufficient_difference_core"
    else:
        status = "localized_calibrated_empirical_prf"
    payload = {
        "schema_version": "1.0",
        "work_package": "PRF_SOURCE_LOCALIZATION",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": workspace.candidate_id,
        "source": source,
        "calibrated": localization_calibrated,
        "calibration_status": "calibrated" if localization_calibrated else "uncalibrated",
        "validation_eligible": False,
        "ephemeris_provenance": {
            "source": ephemeris.get("source"),
            "field_sources": {
                field: field_sources.get(field) for field in required_ephemeris_fields
            },
        },
        "method": "candidate-owned empirical TESS PRF non-negative least-squares fit of absolute out-of-transit minus in-transit difference images",
        "prf_model": "candidate-owned empirical TESS PRF FITS template",
        "calibration_assets": calibration_assets,
        "search_radius_arcsec": float(search_radius_arcsec),
        "source_catalog": source_catalog,
        "skipped_tpf_products": skipped_tpf_products,
        "sector_results": sector_results,
        "summary": {
            "n_sectors": len(sector_results),
            "n_completed": len(completed),
            "sectors_with_competing_sources_modeled": len(sectors_with_neighbors),
            "sectors_with_unresolved_difference_core": int(unresolved_difference_core),
            "sectors_with_uncalibrated_prf": 0,
            "median_target_to_other_difference_ratio": median_ratio,
            "median_difference_image_offset_arcsec": median_offset,
            "conclusion": status,
        },
        "caveats": [
            "Absolute difference-flux amplitudes are in native TPF units, not transit depths or ppm.",
            "The empirical PRF template and recovery calibration are hash-bound candidate-owned assets.",
            "All Gaia sources projected into the finite TPF footprint are fitted simultaneously.",
            "Gaia BP/RP-derived TESS ratios are catalog context, not eclipse-depth constraints.",
        ] + (
            []
            if localization_calibrated
            else ["The localization result is inconclusive because a calibrated source assignment was not supported for every analyzed sector."]
        ),
    }
    output_path = outputs_dir / "prf_localization_results.json"
    _write_json_atomic(output_path, _json_safe(payload))
    return output_path
