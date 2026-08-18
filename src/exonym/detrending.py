"""Opt-in, candidate-local light-curve detrending.

This module deliberately accepts arrays from the caller instead of changing the
shared light-curve loader.  Every successful run writes a processed array
artifact and a provenance manifest below the owning candidate workspace; raw
inputs are never opened for writing.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter

from .workspace import CandidateWorkspace, validate_candidate_id


SUPPORTED_METHODS = ("running-median", "wotan", "celerite")


class OptionalBackendUnavailable(RuntimeError):
    """Raised when an explicitly selected optional detrending backend is absent."""


@dataclass(frozen=True)
class DetrendingArtifacts:
    """Candidate-local files created by one successful detrending run."""

    artifact_path: Path
    manifest_path: Path


def _trusted_workspace(workspace: CandidateWorkspace) -> Path:
    """Return the resolved workspace path only when it has the expected owner."""
    if not isinstance(workspace, CandidateWorkspace):
        raise TypeError("workspace must be a CandidateWorkspace")
    candidate_id = validate_candidate_id(workspace.candidate_id)
    expected = (workspace.repository_root.resolve() / "candidate" / candidate_id).resolve()
    actual = workspace.path.resolve()
    if actual != expected or not actual.is_dir():
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


def _running_median_trend(time: np.ndarray, values: np.ndarray, window_days: float) -> np.ndarray:
    """Estimate a deterministic median trend over ``window_days`` in days."""
    cadence_days = float(np.median(np.diff(time)))
    if not np.isfinite(cadence_days) or cadence_days <= 0:
        raise ValueError("observation times must have positive finite cadence")
    width = max(3, int(round(window_days / cadence_days)))
    if width % 2 == 0:
        width += 1
    return median_filter(values, size=width, mode="nearest")


def _wotan_trend(time: np.ndarray, values: np.ndarray, window_days: float) -> np.ndarray:
    """Estimate a trend with the optional Wotan package."""
    try:
        wotan = importlib.import_module("wotan")
    except ImportError as exc:
        raise OptionalBackendUnavailable(
            "Wotan detrending was requested but the optional 'wotan' package is not installed"
        ) from exc
    try:
        _, trend = wotan.flatten(
            time, values, window_length=window_days, method="biweight", return_trend=True
        )
    except Exception as exc:
        raise RuntimeError("Wotan detrending failed") from exc
    return np.asarray(trend, dtype=float)


def _resolved_errors(values: np.ndarray, errors: Optional[np.ndarray]) -> np.ndarray:
    """Return finite positive uncertainties required by the celerite GP solver."""
    if errors is not None:
        finite_positive = errors[np.isfinite(errors) & (errors > 0)]
        if finite_positive.size:
            fallback = float(np.median(finite_positive))
            return np.where(np.isfinite(errors) & (errors > 0), errors, fallback)
    centered = values - np.median(values)
    robust_sigma = float(1.4826 * np.median(np.abs(centered)))
    return np.full(values.shape, max(robust_sigma, np.finfo(float).eps))


def _celerite_trend(
    time: np.ndarray, values: np.ndarray, errors: Optional[np.ndarray], window_days: float
) -> np.ndarray:
    """Estimate a Matern-3/2 GP trend with the optional celerite package."""
    try:
        celerite = importlib.import_module("celerite")
    except ImportError as exc:
        raise OptionalBackendUnavailable(
            "Celerite detrending was requested but the optional 'celerite' package is not installed"
        ) from exc

    baseline = float(np.median(values))
    residuals = values - baseline
    amplitude = max(float(np.std(residuals)), np.finfo(float).eps)
    try:
        kernel = celerite.terms.Matern32Term(
            log_sigma=float(np.log(amplitude)), log_rho=float(np.log(window_days))
        )
        gp = celerite.GP(kernel, mean=0.0)
        gp.compute(time, _resolved_errors(values, errors))
        prediction = np.asarray(gp.predict(residuals, time, return_cov=False), dtype=float)
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
) -> DetrendingArtifacts:
    """Detrend caller-supplied normalized flux and write candidate-local artifacts.

    Args:
        workspace: Registered candidate workspace that owns the new artifacts.
        time_btjd: Observation times in BTJD.
        flux: Normalized flux values.
        method: ``running-median``, or explicit optional ``wotan``/``celerite``.
        window_days: Trend timescale in days; it must be positive and finite.
        flux_err: Optional normalized flux uncertainties for celerite.

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
    sorted_time, sorted_values, sorted_errors, sorted_indices = _finite_series(time, values, errors)
    runners: Dict[str, Callable[..., np.ndarray]] = {
        "running-median": _running_median_trend,
        "wotan": _wotan_trend,
        "celerite": _celerite_trend,
    }
    if normalized_method == "celerite":
        sorted_trend = runners[normalized_method](
            sorted_time, sorted_values, sorted_errors, float(window_days)
        )
    else:
        sorted_trend = runners[normalized_method](sorted_time, sorted_values, float(window_days))
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
    np.savez_compressed(artifact_buffer, **payload)

    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(artifact_path, artifact_buffer.getvalue())
    manifest = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": normalized_method,
        "backend_version": _method_version(normalized_method),
        "configuration": {"window_days": float(window_days)},
        "input": {
            "cadences": int(values.size),
            "finite_cadences": int(sorted_indices.size),
            "sha256": _input_sha256(time, values, errors),
        },
        "artifact": {
            "path": artifact_path.relative_to(workspace_path).as_posix(),
            "sha256": _file_sha256(artifact_path),
        },
    }
    _atomic_write(
        manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return DetrendingArtifacts(artifact_path=artifact_path, manifest_path=manifest_path)
