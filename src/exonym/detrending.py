"""Candidate-local, provenance-bound light-curve detrending.

The public entry point accepts caller-supplied normalized flux instead of
silently changing the shared light-curve loader. It supports a deterministic
running median and opt-in Wotan or Celerite backends, then writes a processed
array plus a manifest below the owning candidate workspace. Raw products are
never opened for writing.

Transit masks are derived from a complete candidate BTJD ephemeris and are
hash-bound to the exact cadence array. This preserves an auditable separation
between continuum estimation and the declared transit windows.

Scientific Boundary:
    Detrending can alter apparent transit depth and variability. Its products
    are descriptive preprocessing inputs, not an independent detection,
    completeness calibration, or validation result.

References:
    methods/detrending-and-transit-inference.md documents the backend models,
    units, and limitations used by this module.

Primary literature, units, and failure boundary
------------------------------------------------
The Wotan backend is Hippke et al. (2019), ADS ``2019AJ....158..143H``, DOI
``10.3847/1538-3881/ab3984``.  The Celerite Matérn-3/2 implementation is
Foreman-Mackey et al. (2017), ADS ``2017AJ....154..220F``, DOI
``10.3847/1538-3881/aa9332``.  Times are ``BTJD_TDB`` days, flux/trend/error
arrays are dimensionless normalized relative flux, windows and GP length scales
are days, and sector labels are positive mission integers.  Celerite requires
reported finite positive per-cadence errors; it never invents covariance from a
scatter estimate.  Missing raw provenance, sector labels, a complete
candidate-derived ephemeris for a mask, an optional backend, or a finite trend
fails without a processed science artifact.  Detrending remains preprocessing;
it cannot set ``claim_eligible``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import minimize

from .remediation import numerical_npz_sha256
from .workspace import CandidateWorkspace, load_candidate, validate_candidate_id


SUPPORTED_METHODS = ("running-median", "wotan", "celerite")
_TRANSIT_MASK_FIELDS = ("period_days", "epoch_btjd", "duration_days")
_CANDIDATE_TRANSIT_MASK_FIELD_SOURCES = frozenset(
    {
        "candidate-config",
        "candidate-config-signal",
        "candidate-data-bls",
        "bls-search",
    }
)
TRANSIT_MASK_DEFINITION = "nearest-ephemeris-centre-within-half-duration-v1"


class OptionalBackendUnavailable(RuntimeError):
    """Raised when an explicitly selected optional detrending backend is absent."""


@dataclass(frozen=True)
class DetrendingArtifacts:
    """Candidate-local files created by one successful detrending run.

    Attributes:
        artifact_path: Compressed processed cadence array below data/processed.
        manifest_path: JSON manifest that binds configuration, input products,
            transit-mask provenance, and artifact digests.
    """

    artifact_path: Path
    manifest_path: Path


def _trusted_workspace(workspace: CandidateWorkspace) -> Path:
    """Return the resolved workspace path only when it has the expected owner."""
    if not isinstance(workspace, CandidateWorkspace):
        raise TypeError("workspace must be a CandidateWorkspace")
    candidate_id = validate_candidate_id(workspace.candidate_id)
    trusted = load_candidate(workspace.repository_root, candidate_id)
    actual = workspace.path.resolve()
    if actual != trusted.path.resolve() or not actual.is_dir():
        raise ValueError("workspace path is not the registered candidate workspace")
    return actual


def _validated_inputs(
    time_btjd: Sequence[float], flux: Sequence[float], flux_err: Optional[Sequence[float]]
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Convert caller inputs to finite-compatible one-dimensional arrays."""
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    if time.ndim != 1 or values.ndim != 1 or time.shape != values.shape:
        raise ValueError("time_btjd and flux must be matching one-dimensional arrays")
    if time.size < 3:
        raise ValueError("at least three cadences are required for detrending")

    errors = None
    if flux_err is not None:
        errors = np.asarray(flux_err, dtype=float)
        if errors.ndim != 1 or errors.shape != values.shape:
            raise ValueError("flux_err must match flux when provided")
    return time, values, errors


def _validated_sectors(sector: Optional[Sequence[int]], length: int) -> Optional[np.ndarray]:
    """Validate per-cadence TESS sector ownership when supplied by the loader."""
    if sector is None:
        raise ValueError("detrending requires one positive TESS sector per cadence")
    sectors = np.asarray(sector, dtype=int)
    if sectors.ndim != 1 or sectors.size != length or np.any(sectors <= 0):
        raise ValueError("sector must contain one positive TESS sector per cadence")
    return sectors


def _validated_input_products(
    workspace: CandidateWorkspace,
    workspace_path: Path,
    input_products: Optional[Sequence[Mapping[str, str]]],
) -> List[Dict[str, str]]:
    """Retain hash-bound raw inputs so the derived array remains traceable."""
    if input_products is None:
        return []
    from .gatekeeper import has_valid_raw_product_provenance

    records: List[Dict[str, str]] = []
    for product in input_products:
        if not isinstance(product, Mapping):
            raise ValueError("input_products entries must be path and sha256 records")
        relative = product.get("path")
        expected_digest = product.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_digest, str):
            raise ValueError("input_products entries require string path and sha256 values")
        path = (workspace_path / relative).resolve()
        try:
            relative_path = path.relative_to(workspace_path).as_posix()
        except ValueError as exc:
            raise ValueError("input product must remain inside its candidate workspace") from exc
        if (
            not relative_path.startswith("data/raw/")
            or not path.is_file()
            or not has_valid_raw_product_provenance(workspace, path)
        ):
            raise ValueError("input product must be a provenance-valid file below data/raw/")
        actual_digest = _file_sha256(path)
        if actual_digest != expected_digest:
            raise ValueError("input product digest changed before detrending")
        records.append({"path": relative_path, "sha256": actual_digest})
    if len({record["path"] for record in records}) != len(records):
        raise ValueError("input_products must not repeat a raw product")
    return sorted(records, key=lambda record: record["path"])


def _finite_series(
    time: np.ndarray, values: np.ndarray, errors: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Sort finite cadences and retain the mapping needed to restore their order."""
    valid = np.isfinite(time) & np.isfinite(values)
    if valid.sum() < 3:
        raise ValueError("at least three finite time and flux values are required")
    indices = np.flatnonzero(valid)
    order = np.argsort(time[indices], kind="mergesort")
    sorted_indices = indices[order]
    sorted_time = time[sorted_indices]
    if np.any(np.diff(sorted_time) <= 0):
        raise ValueError("finite observation times must be unique")
    sorted_errors = errors[sorted_indices] if errors is not None else None
    return sorted_time, values[sorted_indices], sorted_errors, sorted_indices


def _canonical_transit_mask_ephemeris(ephemeris: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a canonical candidate-derived BTJD ephemeris for mask provenance."""
    from .inputs import BTJD_TIME_SYSTEM

    if not isinstance(ephemeris, Mapping):
        raise ValueError("detrending requires a complete candidate-derived BTJD ephemeris")
    if ephemeris.get("time_system") != BTJD_TIME_SYSTEM:
        raise ValueError("detrending requires a BTJD_TDB candidate ephemeris")

    field_sources = ephemeris.get("field_sources")
    if not isinstance(field_sources, Mapping) or any(
        not isinstance(field_sources.get(field), str)
        or field_sources[field] not in _CANDIDATE_TRANSIT_MASK_FIELD_SOURCES
        for field in _TRANSIT_MASK_FIELDS
    ):
        raise ValueError("detrending requires a complete candidate-derived BTJD ephemeris")
    source = ephemeris.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("detrending requires a candidate ephemeris source label")

    try:
        period_days = float(ephemeris["period_days"])
        epoch_btjd = float(ephemeris["epoch_btjd"])
        duration_days = float(ephemeris["duration_days"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("detrending requires finite period, epoch, and duration values") from exc
    if (
        not np.isfinite(period_days)
        or not np.isfinite(epoch_btjd)
        or not np.isfinite(duration_days)
        or period_days <= 0.0
        or duration_days <= 0.0
        or duration_days >= period_days
    ):
        raise ValueError(
            "detrending requires a positive duration shorter than the candidate period"
        )

    return {
        "schema_version": 1,
        "mask_definition": TRANSIT_MASK_DEFINITION,
        "time_system": BTJD_TIME_SYSTEM,
        "source": source,
        "period_days": period_days,
        "epoch_btjd": epoch_btjd,
        "duration_days": duration_days,
        "field_sources": {
            field: field_sources[field] for field in _TRANSIT_MASK_FIELDS
        },
    }


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical JSON so equivalent ephemeris records have one digest."""
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("transit mask provenance is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _transit_mask_sha256(mask: np.ndarray) -> str:
    """Hash one one-dimensional boolean cadence mask with its exact length."""
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 1:
        raise ValueError("transit mask provenance requires a one-dimensional mask")
    digest = hashlib.sha256()
    digest.update(b"exonym-transit-mask-v1\0")
    digest.update(int(array.size).to_bytes(8, byteorder="big", signed=False))
    digest.update(np.ascontiguousarray(array, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _transit_mask_from_canonical_ephemeris(
    time_btjd: Sequence[float], canonical_ephemeris: Mapping[str, Any]
) -> np.ndarray:
    """Return in-transit flags after the ephemeris provenance has been checked."""
    time = np.asarray(time_btjd, dtype=float)
    if time.ndim != 1:
        raise ValueError(
            "time_btjd must be one-dimensional when deriving a transit mask"
        )
    period_days = float(canonical_ephemeris["period_days"])
    epoch_btjd = float(canonical_ephemeris["epoch_btjd"])
    duration_days = float(canonical_ephemeris["duration_days"])

    phase_days = (
        (time - epoch_btjd + 0.5 * period_days) % period_days
    ) - 0.5 * period_days
    return np.abs(phase_days) <= 0.5 * duration_days


def transit_mask_from_ephemeris(
    time_btjd: Sequence[float], ephemeris: Mapping[str, Any]
) -> np.ndarray:
    """Derive in-transit cadence flags from a complete candidate BTJD ephemeris.

    Each cadence is assigned to its nearest declared transit centre modulo the
    period. A flag is true when that separation is no more than half the
    declared duration, so trend fitting can omit the expected transit window.

    Args:
        time_btjd: One-dimensional observation times in BTJD_TDB days.
        ephemeris: Candidate-derived period, epoch, duration, time-system, and
            field-source mapping.

    Returns:
        A one-dimensional boolean array aligned with time_btjd. True denotes
        an in-transit cadence.

    Raises:
        ValueError: If times are not one-dimensional or the ephemeris is
            incomplete, non-finite, non-BTJD, synthetic, or physically invalid.
    """
    return _transit_mask_from_canonical_ephemeris(
        time_btjd, _canonical_transit_mask_ephemeris(ephemeris)
    )


def transit_mask_provenance_from_ephemeris(
    time_btjd: Sequence[float], ephemeris: Mapping[str, Any]
) -> Dict[str, Any]:
    """Build canonical, hash-bound provenance for a derived transit mask.

    The ephemeris digest covers BTJD period, epoch, duration, source labels,
    and mask-definition version. The mask digest additionally binds those
    values to the exact cadence sequence written into a processed artifact.

    Args:
        time_btjd: One-dimensional observation times in BTJD_TDB days.
        ephemeris: Complete candidate-derived ephemeris used for the mask.

    Returns:
        A JSON-safe mapping containing canonical ephemeris values and SHA-256
        digests for the ephemeris and resulting cadence mask.

    Raises:
        ValueError: If the inputs cannot form a complete canonical provenance
            record.
    """
    canonical_ephemeris = _canonical_transit_mask_ephemeris(ephemeris)
    mask = _transit_mask_from_canonical_ephemeris(time_btjd, canonical_ephemeris)
    return {
        "schema_version": 1,
        "mask_definition": TRANSIT_MASK_DEFINITION,
        "ephemeris": canonical_ephemeris,
        "ephemeris_sha256": _canonical_json_sha256(canonical_ephemeris),
        "mask_sha256": _transit_mask_sha256(mask),
    }


def validate_transit_mask_provenance(
    time_btjd: Sequence[float], provenance: Mapping[str, Any], ephemeris: Mapping[str, Any]
) -> None:
    """Verify that a recorded detrending mask remains reproducible.

    Args:
        time_btjd: Processed artifact cadence times in BTJD_TDB days.
        provenance: Recorded canonical ephemeris and cadence-mask digests.
        ephemeris: Ephemeris expected to reproduce the recorded mask.

    Raises:
        ValueError: If provenance is absent, malformed, stale, or differs from
            the deterministic reconstruction.
    """
    if not isinstance(provenance, Mapping):
        raise ValueError("detrended input has no transit mask provenance")
    expected = transit_mask_provenance_from_ephemeris(time_btjd, ephemeris)
    if dict(provenance) != expected:
        raise ValueError("detrended input transit mask provenance is stale or mismatched")


def _running_median_trend(
    time: np.ndarray, values: np.ndarray, window_days: float,
    transit_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate a deterministic median trend over ``window_days`` in days.

    When ``transit_mask`` is provided, masked cadences are linearly
    interpolated before the median filter runs.
    """
    cadence_days = float(np.median(np.diff(time)))
    if not np.isfinite(cadence_days) or cadence_days <= 0:
        raise ValueError("observation times must have positive finite cadence")
    width = max(3, int(round(window_days / cadence_days)))
    if width % 2 == 0:
        width += 1
    working = values.copy()
    if transit_mask is not None:
        mask = np.asarray(transit_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("transit_mask must match values shape")
        if mask.any():
            if mask.all():
                raise ValueError("transit_mask must retain at least one out-of-transit cadence")
            indices = np.arange(values.size)
            unmasked_idx = indices[~mask]
            unmasked_vals = values[~mask]
            # NUMERICAL_GUARD: np.interp flat-clamps masked cadences beyond the
            # outermost unmasked samples (sector-edge transits), introducing an
            # artificial gradient discontinuity into the running median. Linear
            # slope extrapolation preserves the local trend at sector edges.
            if unmasked_idx.size >= 2:
                from scipy.interpolate import interp1d

                interpolator = interp1d(
                    unmasked_idx,
                    unmasked_vals,
                    kind="linear",
                    fill_value="extrapolate",
                    assume_sorted=True,
                )
                working[mask] = interpolator(indices[mask])
            else:
                working[mask] = np.interp(indices[mask], unmasked_idx, unmasked_vals)
    return median_filter(working, size=width, mode="nearest")


def _wotan_trend(
    time: np.ndarray, values: np.ndarray, window_days: float,
    transit_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate a trend with the optional Wotan package.

    Wotan's ``flatten`` API receives ``True`` for in-transit cadences and
    excludes those cadences internally.  Pass the in-transit mask directly;
    inverting it would instead remove the out-of-transit baseline.
    """
    try:
        wotan = importlib.import_module("wotan")
    except ImportError as exc:
        raise OptionalBackendUnavailable(
            "Wotan detrending was requested but the optional 'wotan' package is not installed"
        ) from exc
    try:
        wotan_kwargs: Dict[str, Any] = {
            "window_length": window_days,
            "method": "biweight",
            "return_trend": True,
        }
        if transit_mask is not None:
            mask = np.asarray(transit_mask, dtype=bool)
            if mask.shape != values.shape:
                raise ValueError("transit_mask must match values shape")
            wotan_kwargs["mask"] = mask
        _, trend = wotan.flatten(time, values, **wotan_kwargs)
    except Exception as exc:
        raise RuntimeError("Wotan detrending failed") from exc
    return np.asarray(trend, dtype=float)


def _resolved_errors(errors: Optional[np.ndarray]) -> np.ndarray:
    """Require measured finite positive uncertainties for the celerite GP.

    Celerite conditions its likelihood on the observational covariance. A
    median replacement or scatter-derived proxy would turn unmeasured noise
    into fabricated input evidence, so the optional GP backend is unavailable
    for a series without complete reported cadence uncertainties.
    """
    if errors is None:
        raise ValueError("celerite detrending requires reported per-cadence flux_err")
    resolved = np.asarray(errors, dtype=float)
    if resolved.ndim != 1 or not np.all(np.isfinite(resolved) & (resolved > 0.0)):
        raise ValueError("celerite detrending requires finite positive per-cadence flux_err")
    return resolved


def _celerite_trend(
    time: np.ndarray, values: np.ndarray, errors: Optional[np.ndarray], window_days: float,
    transit_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate a Matern-3/2 GP trend with the optional celerite package.

    When ``transit_mask`` is provided, masked cadences are excluded from
    the amplitude estimate and GP conditioning arrays.  The conditioned GP
    then predicts the trend at every cadence, including transit windows.

    The Matern-3/2 hyperparameters are maximum-a-posteriori optimized using
    the celerite marginal likelihood and analytic gradient following
    Foreman-Mackey et al. (2017, AJ, 154, 220; arXiv:1703.09710;
    DOI:10.3847/1538-3881/aa9332). The retained primary source is
    ``literature/foreman_mackey_2017_celerite.pdf``. Bounds arise only from
    the observed cadence, baseline, quoted uncertainty floor, and residual
    flux span of the unmasked candidate data.
    """
    unmasked = np.ones(values.size, dtype=bool)
    if transit_mask is not None:
        mask = np.asarray(transit_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("transit_mask must match values shape")
        unmasked = ~mask
    if not unmasked.any():
        raise ValueError("transit_mask must retain at least one out-of-transit cadence")

    try:
        celerite = importlib.import_module("celerite")
    except ImportError as exc:
        raise OptionalBackendUnavailable(
            "Celerite detrending was requested but the optional 'celerite' package is not installed"
        ) from exc

    baseline = float(np.median(values[unmasked]))
    residuals = values - baseline
    unmasked_residuals = residuals[unmasked]
    unmasked_time = time[unmasked]
    amplitude = max(float(np.std(unmasked_residuals)), np.finfo(float).eps)
    unmasked_errors = _resolved_errors(errors[unmasked] if errors is not None else None)
    try:
        kernel = celerite.terms.Matern32Term(
            log_sigma=float(np.log(amplitude)), log_rho=float(np.log(window_days))
        )
        gp = celerite.GP(kernel, mean=0.0)
        gp.compute(unmasked_time, unmasked_errors)
        initial_parameters = np.asarray(gp.get_parameter_vector(), dtype=float)
        parameter_names = tuple(gp.get_parameter_names())
        if initial_parameters.ndim != 1 or initial_parameters.size != len(parameter_names):
            raise RuntimeError("celerite GP parameter vector is invalid")

        cadence_days = float(np.min(np.diff(np.sort(unmasked_time))))
        baseline_days = float(np.max(unmasked_time) - np.min(unmasked_time))
        log_rho_bounds = (math.log(cadence_days), math.log(baseline_days))
        log_sigma_bounds = (
            math.log(max(float(np.min(unmasked_errors)), np.finfo(float).eps)),
            math.log(max(float(np.ptp(unmasked_residuals)), np.finfo(float).eps)),
        )
        if (
            not all(math.isfinite(value) for value in (*log_rho_bounds, *log_sigma_bounds))
            or log_rho_bounds[0] > log_rho_bounds[1]
            or log_sigma_bounds[0] > log_sigma_bounds[1]
        ):
            raise RuntimeError("Celerite GP hyperparameter optimization failed to converge")

        bounds = []
        for parameter_name in parameter_names:
            if parameter_name.endswith("log_sigma"):
                bounds.append(log_sigma_bounds)
            elif parameter_name.endswith("log_rho"):
                bounds.append(log_rho_bounds)
            else:
                raise RuntimeError("Celerite GP exposes unsupported Matern32 hyperparameters")

        def negative_log_likelihood(parameters: np.ndarray) -> Tuple[float, np.ndarray]:
            gp.set_parameter_vector(parameters)
            gp.compute(unmasked_time, unmasked_errors)
            log_likelihood = float(gp.log_likelihood(unmasked_residuals))
            gradient_result = gp.grad_log_likelihood(unmasked_residuals)
            gradient = (
                gradient_result[1]
                if isinstance(gradient_result, tuple)
                else gradient_result
            )
            gradient = np.asarray(gradient, dtype=float)
            if (
                not math.isfinite(log_likelihood)
                or gradient.shape != parameters.shape
                or not np.all(np.isfinite(gradient))
            ):
                return math.inf, np.full(parameters.shape, math.nan)
            return -log_likelihood, -gradient

        solution = minimize(
            negative_log_likelihood,
            initial_parameters,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
        )
        if (
            not solution.success
            or not np.all(np.isfinite(solution.x))
            or not math.isfinite(float(solution.fun))
        ):
            raise RuntimeError("Celerite GP hyperparameter optimization failed to converge")
        gp.set_parameter_vector(solution.x)
        gp.compute(unmasked_time, unmasked_errors)
        prediction = np.asarray(
            gp.predict(unmasked_residuals, time, return_cov=False), dtype=float
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Celerite detrending failed") from exc
    return baseline + prediction


def _method_version(method: str) -> Optional[str]:
    """Return the installed backend version when a package supplies the trend."""
    if method == "running-median":
        return None
    try:
        return importlib.metadata.version(method)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - package metadata varies
        return None


def _input_sha256(time: np.ndarray, values: np.ndarray, errors: Optional[np.ndarray]) -> str:
    """Hash exact numerical inputs so the manifest identifies the processed series."""
    digest = hashlib.sha256()
    for array in (time, values, errors):
        if array is not None:
            contiguous = np.ascontiguousarray(array, dtype=np.float64)
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one candidate-local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    """Replace one candidate-local artifact without leaving a partial output file."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def detrend_candidate(
    workspace: CandidateWorkspace,
    time_btjd: Sequence[float],
    flux: Sequence[float],
    *,
    method: str = "running-median",
    window_days: float,
    flux_err: Optional[Sequence[float]] = None,
    sector: Optional[Sequence[int]] = None,
    input_products: Optional[Sequence[Mapping[str, str]]] = None,
    transit_mask: Optional[np.ndarray] = None,
    transit_mask_ephemeris: Optional[Mapping[str, Any]] = None,
) -> DetrendingArtifacts:
    """Detrend caller-supplied normalized flux and write candidate-local artifacts.

    Args:
        workspace: Registered candidate workspace that owns the new artifacts.
        time_btjd: Observation times in BTJD.
        flux: Normalized flux values.
        method: ``running-median``, or explicit optional ``wotan``/``celerite``.
        window_days: Trend timescale in days; it must be positive and finite.
        flux_err: Optional normalized flux uncertainties for celerite.
        sector: Candidate-owned TESS sector labels, one for each cadence.
        input_products: Hash-bound raw light-curve records used for the input.
        transit_mask: Boolean mask of in-transit cadences to protect from
            detrending bias. Cadences where ``transit_mask`` is True are
            excluded from backend conditioning; each backend estimates or
            predicts the corresponding continuum trend at those cadences.
        transit_mask_ephemeris: Complete candidate-derived BTJD ephemeris
            used to derive ``transit_mask``. It is required whenever a mask
            is supplied and is hash-bound into the output manifest.

    Returns:
        Paths for a compressed processed array and its JSON provenance manifest.

    Raises:
        OptionalBackendUnavailable: The selected optional backend is unavailable.
        ValueError: Inputs or a fitted trend are unsuitable for scientific output.

    The function is opt-in and does not modify input arrays or files under
    ``data/raw/``. It writes only after the selected backend has completed.
    """
    workspace_path = _trusted_workspace(workspace)
    normalized_method = method.strip().lower()
    if normalized_method not in SUPPORTED_METHODS:
        raise ValueError("method must be one of: {0}".format(", ".join(SUPPORTED_METHODS)))
    if not np.isfinite(window_days) or window_days <= 0:
        raise ValueError("window_days must be positive and finite")

    time, values, errors = _validated_inputs(time_btjd, flux, flux_err)
    sectors = _validated_sectors(sector, values.size)
    products = _validated_input_products(workspace, workspace_path, input_products)
    if not products:
        raise ValueError("detrending requires hash-bound raw input products")
    sorted_time, sorted_values, sorted_errors, sorted_indices = _finite_series(time, values, errors)

    # SCIENTIFIC_BOUNDARY: Require a reproducible candidate ephemeris before
    # treating a protected cadence mask as suitable for scientific consumers.
    # Validate and sort the transit mask alongside the flux arrays.
    sorted_transit_mask: Optional[np.ndarray] = None
    mask_provenance: Optional[Dict[str, Any]] = None
    if transit_mask is not None:
        mask_arr = np.asarray(transit_mask, dtype=bool)
        if mask_arr.shape != values.shape:
            raise ValueError("transit_mask must be a boolean array matching flux shape")
        sorted_transit_mask = mask_arr[sorted_indices]
        if sorted_transit_mask.all():
            raise ValueError("transit_mask must retain at least one out-of-transit cadence")
        if transit_mask_ephemeris is None:
            raise ValueError("transit_mask requires its complete candidate-derived BTJD ephemeris")
        expected_mask = transit_mask_from_ephemeris(time, transit_mask_ephemeris)
        if not np.array_equal(mask_arr, expected_mask):
            raise ValueError("transit_mask does not match its candidate-derived BTJD ephemeris")
        mask_provenance = transit_mask_provenance_from_ephemeris(
            time, transit_mask_ephemeris
        )
    elif transit_mask_ephemeris is not None:
        raise ValueError("transit_mask_ephemeris requires a transit_mask")

    masked_fraction: float = 0.0
    transit_mask_applied: bool = False
    if sorted_transit_mask is not None:
        transit_mask_applied = True
        masked_fraction = float(np.count_nonzero(sorted_transit_mask)) / float(sorted_values.size)

    runners: Dict[str, Callable[..., np.ndarray]] = {
        "running-median": _running_median_trend,
        "wotan": _wotan_trend,
        "celerite": _celerite_trend,
    }
    if normalized_method == "celerite":
        sorted_trend = runners[normalized_method](
            sorted_time, sorted_values, sorted_errors, float(window_days),
            transit_mask=sorted_transit_mask,
        )
    else:
        sorted_trend = runners[normalized_method](
            sorted_time, sorted_values, float(window_days),
            transit_mask=sorted_transit_mask,
        )
    if sorted_trend.shape != sorted_values.shape:
        raise ValueError("detrending backend returned a trend with the wrong shape")
    if not np.all(np.isfinite(sorted_trend)) or np.any(np.abs(sorted_trend) <= np.finfo(float).eps):
        raise ValueError("detrending backend returned a non-finite or zero trend")

    trend = np.full(values.shape, np.nan)
    detrended_flux = np.full(values.shape, np.nan)
    trend[sorted_indices] = sorted_trend
    detrended_flux[sorted_indices] = sorted_values / sorted_trend
    detrended_err = None
    if errors is not None:
        detrended_err = np.full(errors.shape, np.nan)
        detrended_err[sorted_indices] = sorted_errors / np.abs(sorted_trend)

    processed_dir = workspace_path / "data" / "processed"
    outputs_dir = workspace_path / "outputs"
    artifact_path = processed_dir / "detrended-{0}.npz".format(normalized_method)
    manifest_path = outputs_dir / "detrending_manifest.{0}.json".format(normalized_method)
    artifact_buffer = io.BytesIO()
    payload: Dict[str, Any] = {
        "time_btjd": time,
        "flux": values,
        "trend": trend,
        "detrended_flux": detrended_flux,
    }
    if errors is not None:
        payload["flux_err"] = errors
        payload["detrended_flux_err"] = detrended_err
    payload["sector"] = sectors
    np.savez_compressed(artifact_buffer, **payload)

    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(artifact_path, artifact_buffer.getvalue())
    manifest = {
        "schema_version": 2,
        "candidate_id": workspace.candidate_id,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": normalized_method,
        "backend_version": _method_version(normalized_method),
        "configuration": {
            "window_days": float(window_days),
            "transit_mask_applied": transit_mask_applied,
            "masked_fraction": masked_fraction,
            "transit_mask_provenance": mask_provenance,
        },
        "input": {
            "cadences": int(values.size),
            "finite_cadences": int(sorted_indices.size),
            "sha256": _input_sha256(time, values, errors),
        },
        "artifact": {
            "path": artifact_path.relative_to(workspace_path).as_posix(),
            "sha256": _file_sha256(artifact_path),
            "data_sha256": numerical_npz_sha256(artifact_path),
        },
    }
    manifest["input_products"] = products
    _atomic_write(
        manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return DetrendingArtifacts(artifact_path=artifact_path, manifest_path=manifest_path)
