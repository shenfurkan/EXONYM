"""Target-neutral asteroseismology engine.

Estimates solar-like p-mode stellar oscillation envelope parameters:
- Frequency of maximum oscillation power (nu_max)
- Large frequency separation (Delta_nu)

From high-cadence light curves via background-whitened Lomb-Scargle power spectral
densities (PSD), and derives fundamental stellar properties (M_star, R_star, rho_star, log_g)
using canonical asteroseismic scaling relations (Kjeldsen & Bedding 1995, Huber et al. 2011,
Chaplin et al. 2014).

Scaling Relations:
    (R / R_sun) = (nu_max / nu_max_sun) * (Delta_nu / Delta_nu_sun)^-2 * (Teff / Teff_sun)^(1/2)
    (M / M_sun) = (nu_max / nu_max_sun)^3 * (Delta_nu / Delta_nu_sun)^-4 * (Teff / Teff_sun)^(3/2)
    (g / g_sun) = (nu_max / nu_max_sun) * (Teff / Teff_sun)^(1/2)
    (rho / rho_sun) = (Delta_nu / Delta_nu_sun)^2

Optionally cross-checks with pySYD when installed. Contains no target identifiers
or hardcoded candidate constants.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .inputs import (
    load_light_curve_table,
    load_stellar_parameters,
    load_transit_ephemeris,
)
from .lightcurve import phase_hours
from .workspace import CandidateWorkspace

# Canonical Solar Asteroseismic Reference Values (Huber et al. 2011, Chaplin et al. 2014)
NUMAX_SUN_UHZ = 3090.0      # Solar frequency of maximum oscillation power (microHz)
DNU_SUN_UHZ = 135.1         # Solar large frequency separation (microHz)
TEFF_SUN_K = 5772.0         # Solar effective temperature (Kelvin)

PSD_MIN_UHZ = 100.0         # Default minimum frequency for stellar PSD search (microHz)
PSD_MAX_UHZ = 2000.0        # Default maximum frequency for stellar PSD search (microHz)
DNU_MIN_UHZ = 30.0          # Minimum trial Delta-nu lag (microHz)
DNU_MAX_UHZ = 70.0          # Maximum trial Delta-nu lag (microHz)
MICROHZ_PER_CPD = 0.0864    # Unit conversion factor: cycles per day to microHz (1 c/d = 11.574 microHz)


def _odd_bins(value: float) -> int:
    bins = max(3, int(round(value)))
    return bins if bins % 2 else bins + 1


def compute_power_spectrum(
    time: Sequence[float],
    flux: Sequence[float],
    frequency_min_uhz: float = PSD_MIN_UHZ,
    frequency_max_uhz: float = PSD_MAX_UHZ,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (frequency_uhz, power, whitened, envelope) for a light curve."""
    from astropy.timeseries import LombScargle
    from scipy.ndimage import gaussian_filter1d, median_filter

    time_arr = np.asarray(time, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    finite = np.isfinite(time_arr) & np.isfinite(flux_arr)
    time_arr = time_arr[finite]
    flux_arr = flux_arr[finite]
    if time_arr.size < 50:
        raise ValueError("insufficient data for power spectrum estimation")

    frequency_day, power = LombScargle(
        time_arr,
        flux_arr - np.nanmean(flux_arr),
        normalization="psd",
    ).autopower(
        minimum_frequency=frequency_min_uhz * MICROHZ_PER_CPD,
        maximum_frequency=frequency_max_uhz * MICROHZ_PER_CPD,
        samples_per_peak=1,
        method="fast",
    )
    frequency_uhz = np.asarray(frequency_day) / MICROHZ_PER_CPD
    power = np.asarray(power, dtype=float)
    spacing = float(np.nanmedian(np.diff(frequency_uhz)))
    background_bins = _odd_bins(100.0 / max(spacing, 1e-6))
    background = median_filter(power, size=background_bins, mode="nearest")
    background = np.maximum(background, np.finfo(float).tiny)
    whitened = power / background
    smooth_sigma = max(1.0, 20.0 / max(spacing, 1e-6))
    envelope = gaussian_filter1d(whitened, smooth_sigma, mode="nearest")
    return frequency_uhz, power, whitened, envelope


def spacing_correlation(
    frequency_uhz: np.ndarray,
    whitened: np.ndarray,
    numax_uhz: float,
    dnu_min_uhz: float = DNU_MIN_UHZ,
    dnu_max_uhz: float = DNU_MAX_UHZ,
) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
    """Return (best_dnu_uhz, correlation, lag_grid) around nu_max."""
    envelope_half_width = 0.66 * numax_uhz**0.88
    use = np.abs(frequency_uhz - numax_uhz) <= envelope_half_width
    local_frequency = frequency_uhz[use]
    local = whitened[use] - 1.0
    if local.size < 20:
        return None, None, None
    lags = np.linspace(dnu_min_uhz, dnu_max_uhz, 801)
    scores = np.empty_like(lags)
    for index, lag in enumerate(lags):
        shifted = np.interp(
            local_frequency + lag,
            local_frequency,
            local,
            left=np.nan,
            right=np.nan,
        )
        valid = np.isfinite(shifted)
        if valid.sum() < 10:
            scores[index] = np.nan
            continue
        x = local[valid]
        y = shifted[valid]
        denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
        scores[index] = np.sum(x * y) / denominator if denominator else np.nan
    best = int(np.nanargmax(scores))
    return float(lags[best]), float(scores[best]), lags


def estimate_oscillation_envelope(
    time: Sequence[float],
    flux: Sequence[float],
    numax_min_uhz: float,
    numax_max_uhz: float,
) -> Dict[str, Any]:
    """Return numax/dnu candidate estimates from a whitened PSD envelope."""
    search_low = max(PSD_MIN_UHZ, float(numax_min_uhz))
    search_high = min(PSD_MAX_UHZ, float(numax_max_uhz))
    if search_high <= search_low:
        raise ValueError("invalid numax search bounds")
    frequency, power, whitened, envelope = compute_power_spectrum(
        time, flux, search_low, search_high
    )
    search = (frequency >= search_low) & (frequency <= search_high)
    peak_index = int(np.flatnonzero(search)[int(np.nanargmax(envelope[search]))])
    numax_candidate = float(frequency[peak_index])
    dnu_candidate, dnu_correlation, _ = spacing_correlation(
        frequency, whitened, numax_candidate
    )
    return {
        "n_points": int(len(time)),
        "baseline_days": float(np.max(time) - np.min(time)),
        "rayleigh_uhz": float(1e6 / ((np.max(time) - np.min(time)) * 86400.0)),
        "numax_candidate_uhz": numax_candidate,
        "envelope_peak_ratio": float(envelope[peak_index]),
        "dnu_candidate_uhz": dnu_candidate,
        "dnu_correlation": dnu_correlation,
    }


def seismic_mass_radius(
    numax_uhz: float,
    dnu_uhz: Optional[float],
    teff_k: float,
    mass_prior_solar: Optional[float] = None,
    radius_prior_solar: Optional[float] = None,
    dnu_correction_factor: float = 1.0,
) -> Dict[str, Any]:
    """Derive asteroseismic stellar mass and radius from scaling relations.

    Uses the classic relations
        nu_max / nu_max_sun = M/M_sun (R/R_sun)^-2 (Teff/Teff_sun)^-1/2
        Delta-nu / Delta-nu_sun = (rho / rho_sun)^1/2
    When one of the two observables is missing, the corresponding stellar
    prior (solar reference by default) closes the system.

    .. note:: Systematic bias in Delta-nu
        The classic Kjeldsen & Bedding (1995) scaling relation for Delta-nu
        carries a known 5–15% systematic offset driven by near-surface effects.
        A partial correction can be applied via ``dnu_correction_factor`` using
        the tabulated values from Sharma et al. (2016, ApJ 822, 15).
        ``dnu_correction_factor`` multiplies the raw Lomb-Scargle Delta-nu
        estimate before the ratio is computed (default 1.0 = no correction).
    """
    numax_ratio = float(numax_uhz) / NUMAX_SUN_UHZ
    teff_ratio = float(teff_k) / TEFF_SUN_K
    method = "scaling-relations"
    if dnu_uhz is not None and dnu_uhz > 0:
        dnu_corrected = float(dnu_uhz) * float(dnu_correction_factor)
        dnu_ratio = dnu_corrected / DNU_SUN_UHZ
        if numax_uhz > 0:
            radius = numax_ratio * math.sqrt(teff_ratio) / (dnu_ratio**2)
            mass = (radius**3) * (dnu_ratio**2)
            method = "full-numax-dnu-scaling"
        else:
            radius = float(radius_prior_solar) if radius_prior_solar else 1.0
            mass = (radius**3) * (dnu_ratio**2)
            method = "dnu-density-scaling-with-radius-prior"
    elif numax_uhz > 0:
        mass = float(mass_prior_solar) if mass_prior_solar else 1.0
        radius = math.sqrt(mass / (numax_ratio * math.sqrt(teff_ratio)))
        method = "numax-scaling-with-mass-prior"
    else:
        mass = float(mass_prior_solar) if mass_prior_solar else 1.0
        radius = float(radius_prior_solar) if radius_prior_solar else 1.0
        method = "stellar-priors-only"
    return {
        "mass_solar": round(float(mass), 4),
        "radius_solar": round(float(radius), 4),
        "method": method,
    }


def _percentile_summary(samples: np.ndarray) -> Dict[str, float]:
    quantiles = np.quantile(np.asarray(samples, dtype=float), [0.16, 0.50, 0.84])
    return {
        "p16": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p84": float(quantiles[2]),
        "plus": float(quantiles[2] - quantiles[1]),
        "minus": float(quantiles[1] - quantiles[0]),
    }


def seismic_uncertainty_summary(
    envelope: Dict[str, Any], stellar: Dict[str, Any], draws: int = 2048
) -> Dict[str, Any]:
    """Propagate frequency-resolution and stellar-temperature errors by Monte Carlo."""
    try:
        numax = float(envelope["numax_candidate_uhz"])
        dnu = float(envelope["dnu_candidate_uhz"])
        rayleigh = float(envelope["rayleigh_uhz"])
        teff = float(stellar["teff_k"])
        teff_error = float(stellar["teff_k_err"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "unavailable-missing-input-uncertainty",
            "reason": "Requires finite numax, dnu, Rayleigh resolution, teff_k, and teff_k_err.",
        }
    if not all(math.isfinite(value) and value > 0 for value in (numax, dnu, rayleigh, teff, teff_error)):
        return {
            "status": "unavailable-invalid-input-uncertainty",
            "reason": "Frequency resolution and stellar-temperature uncertainty must be positive and finite.",
        }
    rng = np.random.default_rng(seed=41)
    numax_draws = np.clip(rng.normal(numax, rayleigh, draws), np.finfo(float).eps, None)
    dnu_draws = np.clip(rng.normal(dnu, rayleigh, draws), np.finfo(float).eps, None)
    teff_draws = np.clip(rng.normal(teff, teff_error, draws), np.finfo(float).eps, None)
    numax_ratio = numax_draws / NUMAX_SUN_UHZ
    dnu_ratio = dnu_draws / DNU_SUN_UHZ
    teff_ratio = teff_draws / TEFF_SUN_K
    radius_draws = numax_ratio * np.sqrt(teff_ratio) / dnu_ratio**2
    mass_draws = radius_draws**3 * dnu_ratio**2
    return {
        "status": "resolution-and-temperature-monte-carlo",
        "draws": int(draws),
        "numax_uhz": _percentile_summary(numax_draws),
        "dnu_uhz": _percentile_summary(dnu_draws),
        "mass_solar": _percentile_summary(mass_draws),
        "radius_solar": _percentile_summary(radius_draws),
        "assumptions": (
            "Independent Gaussian Rayleigh-resolution errors for numax and dnu, and a "
            "candidate-supplied Gaussian teff error; systematic scaling-relation error is excluded."
        ),
    }


SEISMIC_MASS_BOUNDS_SOLAR = (0.05, 20.0)
SEISMIC_RADIUS_BOUNDS_SOLAR = (0.05, 20.0)
SEISMIC_PRIOR_RATIO_TOLERANCE = 2.0


def seismic_sanity_check(
    seismic: Dict[str, Any],
    radius_prior_solar: Optional[float] = None,
    prior_is_catalog: bool = False,
) -> Dict[str, Any]:
    """Flag scaling-relation results that are physically implausible.

    Scaling relations applied to noise peaks can return absurd stellar
    parameters (e.g., a 26 Msun A star from two 120-s sectors). Results outside
    plausible bounds, or inconsistent with a catalog/SED radius prior by more
    than the tolerance factor, are flagged so the caller can reject them.
    """
    reasons: List[str] = []
    mass = float(seismic.get("mass_solar", 0.0))
    radius = float(seismic.get("radius_solar", 0.0))
    mass_lo, mass_hi = SEISMIC_MASS_BOUNDS_SOLAR
    radius_lo, radius_hi = SEISMIC_RADIUS_BOUNDS_SOLAR
    if not (mass_lo <= mass <= mass_hi):
        reasons.append("mass outside plausible range")
    if not (radius_lo <= radius <= radius_hi):
        reasons.append("radius outside plausible range")
    if prior_is_catalog and radius_prior_solar and radius_prior_solar > 0 and radius > 0:
        ratio = radius / float(radius_prior_solar)
        if not (1.0 / SEISMIC_PRIOR_RATIO_TOLERANCE <= ratio <= SEISMIC_PRIOR_RATIO_TOLERANCE):
            reasons.append("scaling radius inconsistent with catalog radius prior")
    return {"plausible": not reasons, "reasons": reasons}


def _highpass_segments(
    time: np.ndarray,
    flux: np.ndarray,
    cadence_seconds: float,
    window_days: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove slow trends per contiguous segment (Savitzky-Golay)."""
    from scipy.signal import savgol_filter

    order = np.argsort(time)
    time = np.asarray(time, dtype=float)[order]
    flux = np.asarray(flux, dtype=float)[order]
    residual = np.full_like(flux, np.nan)
    gaps = np.flatnonzero(np.diff(time) > 5.0 * cadence_seconds / 86400.0) + 1
    edges = np.r_[0, gaps, len(time)]
    nominal_window = int(round(window_days * 86400.0 / cadence_seconds))
    if nominal_window % 2 == 0:
        nominal_window += 1
    for start, stop in zip(edges[:-1], edges[1:]):
        count = stop - start
        window = min(nominal_window, count if count % 2 else count - 1)
        if window < 11:
            continue
        trend = savgol_filter(flux[start:stop], window, 2, mode="interp")
        good = np.isfinite(trend) & (trend != 0)
        segment = np.full(count, np.nan)
        segment[good] = (flux[start:stop][good] / trend[good] - 1.0) * 1e6
        residual[start:stop] = segment
    finite = np.isfinite(residual)
    return time[finite], residual[finite]


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one candidate-local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_run_dir(workspace: CandidateWorkspace, engine: str) -> Tuple[str, Path]:
    """Create one unique candidate-local directory for an optional adapter run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f").lower()
    run_id = "{0}-{1}".format(timestamp, engine)
    run_dir = workspace.path / "runs" / engine / run_id
    suffix = 1
    while run_dir.exists():
        run_id = "{0}-{1}-{2}".format(timestamp, engine, suffix)
        run_dir = workspace.path / "runs" / engine / run_id
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_id, run_dir


def _adapter_artifact(workspace: CandidateWorkspace, path: Path, role: Optional[str] = None) -> Dict[str, str]:
    """Build a schema-compatible artifact record for a file below a workspace."""
    relative = path.resolve().relative_to(workspace.path.resolve()).as_posix()
    artifact = {"path": relative, "sha256": _sha256(path)}
    if role is not None:
        artifact["role"] = role
    return artifact


def _adapter_outputs(run_dir: Path, input_path: Path) -> List[Path]:
    """Return adapter-produced files, excluding the generated adapter input."""
    return sorted(
        path for path in run_dir.rglob("*") if path.is_file() and path != input_path
    )


def _write_adapter_manifest(
    workspace: CandidateWorkspace,
    engine: str,
    run_id: str,
    run_dir: Path,
    started_at: str,
    runtime: Dict[str, str],
    input_path: Path,
    status: str,
    failure: Optional[Dict[str, str]] = None,
) -> Path:
    """Write the engine-run record after hashing all candidate-local adapter files."""
    output_paths = _adapter_outputs(run_dir, input_path)
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "engine": engine,
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "inputs": [_adapter_artifact(workspace, input_path, "adapter-input")],
        "outputs": [_adapter_artifact(workspace, path) for path in output_paths],
    }
    if failure is not None:
        manifest["failure"] = failure
    manifest_path = run_dir / "engine-run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _runtime_version(module: Any, distribution: str) -> str:
    """Return installed package metadata when available without requiring a dependency pin."""
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:
        module_version = getattr(module, "__version__", None)
        return str(module_version) if module_version else "unknown"


def _read_pysyd_estimates(path: Path) -> List[Dict[str, Any]]:
    """Read pySYD's CSV result without adding a pandas runtime requirement."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = []
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("pySYD estimates have no CSV header")
        for row in reader:
            parsed: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None or not value.strip():
                    parsed[key] = None
                    continue
                try:
                    number = float(value)
                except ValueError:
                    parsed[key] = value
                    continue
                if not math.isfinite(number):
                    raise ValueError("pySYD estimates contain a non-finite value")
                parsed[key] = number
            rows.append(parsed)
    return rows


def _run_pysyd_adapter(
    workspace: CandidateWorkspace,
    time: np.ndarray,
    flux: np.ndarray,
    numax_min_uhz: float,
    numax_max_uhz: float,
) -> Dict[str, Any]:
    """Run pySYD in a candidate-local directory and normalize its adapter status."""
    started_at = datetime.now(timezone.utc).isoformat()
    run_id, run_dir = _adapter_run_dir(workspace, "pysyd")
    input_path = run_dir / "asteroseismic_input_LC.txt"
    np.savetxt(
        input_path,
        np.column_stack((time, np.asarray(flux) / 1e6)),
        fmt="%.10f %.12f",
    )
    runtime = {"kind": "direct", "version": "unavailable", "executable": "pysyd"}
    try:
        pysyd = importlib.import_module("pysyd")
    except ModuleNotFoundError as exc:
        manifest_path = _write_adapter_manifest(
            workspace, "pysyd", run_id, run_dir, started_at, runtime, input_path,
            "unavailable", {"code": "module-unavailable", "message": str(exc)},
        )
        return {"status": "unavailable", "manifest_path": manifest_path, "crosscheck": None}
    except Exception as exc:
        manifest_path = _write_adapter_manifest(
            workspace, "pysyd", run_id, run_dir, started_at, runtime, input_path,
            "failed", {"code": "module-import-failed", "message": str(exc)},
        )
        return {"status": "failed", "manifest_path": manifest_path, "crosscheck": None}

    runtime["version"] = _runtime_version(pysyd, "pysyd")
    main_func = getattr(pysyd, "main", None)
    if not callable(main_func):
        manifest_path = _write_adapter_manifest(
            workspace, "pysyd", run_id, run_dir, started_at, runtime, input_path,
            "unavailable", {
                "code": "unsupported-interface",
                "message": "The installed pySYD module does not expose a callable main entry point.",
            },
        )
        return {"status": "unavailable", "manifest_path": manifest_path, "crosscheck": None}

    try:
        previous_directory = Path.cwd()
        try:
            os.chdir(run_dir)
            main_func(["-f", str(input_path)])
        finally:
            os.chdir(previous_directory)
        estimates_path = run_dir / "estimates.csv"
        if not estimates_path.is_file():
            manifest_path = _write_adapter_manifest(
                workspace, "pysyd", run_id, run_dir, started_at, runtime, input_path,
                "unavailable", {
                    "code": "unsupported-output-interface",
                    "message": "pySYD completed without its documented estimates.csv output.",
                },
            )
            return {"status": "unavailable", "manifest_path": manifest_path, "crosscheck": None}
        crosscheck = {
            "pipeline": "pysyd",
            "estimates": _read_pysyd_estimates(estimates_path),
            "search_range_uhz": [float(numax_min_uhz), float(numax_max_uhz)],
        }
    except Exception as exc:
        manifest_path = _write_adapter_manifest(
            workspace, "pysyd", run_id, run_dir, started_at, runtime, input_path,
            "failed", {"code": "adapter-execution-failed", "message": str(exc)},
        )
        return {"status": "failed", "manifest_path": manifest_path, "crosscheck": None}

    manifest_path = _write_adapter_manifest(
        workspace, "pysyd", run_id, run_dir, started_at, runtime, input_path, "succeeded"
    )
    return {"status": "succeeded", "manifest_path": manifest_path, "crosscheck": crosscheck}


def _record_tess_atl_adapter(workspace: CandidateWorkspace) -> Dict[str, Any]:
    """Record tess-atl availability without querying or inventing stellar values."""
    started_at = datetime.now(timezone.utc).isoformat()
    run_id, run_dir = _adapter_run_dir(workspace, "tess-atl")
    input_path = run_dir / "adapter-request.json"
    input_path.write_text(
        json.dumps(
            {
                "adapter": "tess-atl",
                "purpose": "availability and interface provenance only",
                "network_requests": "not-attempted",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    runtime = {"kind": "external", "version": "unavailable", "executable": "tess-atl"}
    try:
        module_spec = importlib.util.find_spec("tess_atl")
    except Exception as exc:
        manifest_path = _write_adapter_manifest(
            workspace, "tess-atl", run_id, run_dir, started_at, runtime, input_path,
            "failed", {"code": "availability-check-failed", "message": str(exc)},
        )
        return {"status": "failed", "manifest_path": manifest_path}
    if module_spec is None:
        failure = {
            "code": "module-unavailable",
            "message": "No local tess-atl module is installed; no request was attempted.",
        }
    else:
        failure = {
            "code": "unsupported-interface",
            "message": "A tess-atl module is present, but no supported local analysis interface is configured; no request was attempted.",
        }
        runtime["version"] = _runtime_version(None, "tess-atl")
    manifest_path = _write_adapter_manifest(
        workspace, "tess-atl", run_id, run_dir, started_at, runtime, input_path,
        "unavailable", failure,
    )
    return {"status": "unavailable", "manifest_path": manifest_path}


def _synthetic_oscillation_table() -> Dict[str, np.ndarray]:
    """Deterministic demonstration light curve with an injected p-mode comb.

    The comb carries a Gaussian amplitude envelope so the whitened PSD
    envelope peaks near the injected nu_max.
    """
    rng = np.random.default_rng(seed=23)
    numax_demo_uhz = 250.0
    dnu_demo_uhz = 40.0
    envelope_sigma_uhz = 2.5 * dnu_demo_uhz
    cadence_days = 120.0 / 86400.0
    time = np.arange(0.0, 27.0, cadence_days)
    flux = np.ones_like(time)
    for harmonic in range(-4, 5):
        amplitude = 120e-6 * math.exp(
            -((harmonic * dnu_demo_uhz) ** 2) / (2.0 * envelope_sigma_uhz**2)
        )
        frequency_cpd = (numax_demo_uhz + harmonic * dnu_demo_uhz) * MICROHZ_PER_CPD
        flux = flux + amplitude * np.sin(2.0 * np.pi * frequency_cpd * time)
    flux = flux + rng.normal(0.0, 30e-6, size=time.shape)
    flux_err = np.full_like(flux, 30e-6)
    sector_values = np.ones(time.size, dtype=int)
    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "sector": sector_values,
    }


def run_asteroseismology(
    workspace: CandidateWorkspace,
    numax_min_uhz: float = 100.0,
    numax_max_uhz: float = 1600.0,
) -> Path:
    """Run the asteroseismic pipeline and write outputs/asteroseismic_results.json."""
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    table = load_light_curve_table(workspace, require_raw_provenance=True)
    if table is None:
        raise RuntimeError("asteroseismology requires observed candidate photometry")
    source = "candidate-data"

    time = table["time"]
    flux = table["flux"]
    ephemeris = load_transit_ephemeris(workspace)
    cadence_seconds = 120.0
    if time.size > 1:
        cadence_seconds = float(np.median(np.diff(np.sort(time)))) * 86400.0
    required_fields = ("period_days", "epoch_btjd", "duration_days")
    can_mask_transits = ephemeris.get("source") != "synthetic-demo" and all(
        ephemeris.get("field_sources", {}).get(field) != "synthetic-demo"
        for field in required_fields
    )
    if can_mask_transits:
        phase_days = phase_hours(time, ephemeris["period_days"], ephemeris["epoch_btjd"]) / 24.0
        transit_mask = np.abs(phase_days) >= 0.75 * ephemeris["duration_days"]
        masked_time = time[transit_mask]
        masked_flux = flux[transit_mask]
        transit_mask_status = "applied-candidate-ephemeris"
    else:
        masked_time = time
        masked_flux = flux
        transit_mask_status = "not-applied-no-candidate-ephemeris"
    if masked_time.size < 100:
        masked_time = time
        masked_flux = flux
        transit_mask_status = "not-applied-insufficient-post-mask-cadences"

    detrended_time, detrended_flux = _highpass_segments(
        masked_time, masked_flux, cadence_seconds, window_days=1.0
    )
    if detrended_time.size < 100:
        detrended_time, detrended_flux = masked_time, masked_flux

    envelope = estimate_oscillation_envelope(
        detrended_time, detrended_flux, numax_min_uhz, numax_max_uhz
    )
    stellar_params = load_stellar_parameters(workspace)
    seismic = seismic_mass_radius(
        envelope["numax_candidate_uhz"],
        envelope["dnu_candidate_uhz"],
        stellar_params["teff_k"],
        mass_prior_solar=stellar_params["mass_solar"],
        radius_prior_solar=stellar_params["radius_solar"],
    )
    sanity = seismic_sanity_check(
        seismic,
        radius_prior_solar=stellar_params["radius_solar"],
        prior_is_catalog=stellar_params.get("source") == "candidate-data",
    )
    uncertainty = seismic_uncertainty_summary(envelope, stellar_params)

    pysyd_adapter = _run_pysyd_adapter(
        workspace, detrended_time, detrended_flux, numax_min_uhz, numax_max_uhz
    )
    tess_atl_adapter = _record_tess_atl_adapter(workspace)
    pysyd_result = pysyd_adapter["crosscheck"]

    payload = {
        "schema_version": "1.0",
        "work_package": "ASTEROSEISMOLOGY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "scientific_status": "exploratory-scaling-diagnostic",
        "validation_eligible": False,
        "validation_reason": (
            "The envelope and solar scaling relations are not a calibrated "
            "mode-identification or stellar-parameter inference. They cannot "
            "supply an automatic validation constraint."
        ),
        "transit_mask_status": transit_mask_status,
        "status": (
            "scaling_rejected_unphysical"
            if not sanity["plausible"]
            else (
                "oscillation_envelope_estimated"
                if envelope["dnu_candidate_uhz"] is not None
                else "envelope_estimated_dnu_undetermined"
            )
        ),
        "pipeline": "pysyd-crosscheck" if pysyd_result else "whitened-gls-psd",
        "search_range_uhz": [float(numax_min_uhz), float(numax_max_uhz)],
        "numax_uhz": envelope["numax_candidate_uhz"],
        "envelope_peak_ratio": envelope["envelope_peak_ratio"],
        "dnu_uhz": envelope["dnu_candidate_uhz"],
        "dnu_correlation": envelope["dnu_correlation"],
        "rayleigh_uhz": envelope["rayleigh_uhz"],
        "n_points_analyzed": int(detrended_time.size),
        "baseline_days": envelope["baseline_days"],
        "stellar_parameters": {
            "mass_solar": seismic["mass_solar"],
            "radius_solar": seismic["radius_solar"],
            "method": seismic["method"],
            "teff_k_prior": stellar_params["teff_k"],
            "mass_prior_solar": stellar_params["mass_solar"],
            "radius_prior_solar": stellar_params["radius_solar"],
            "validity": sanity,
        },
        "uncertainty": uncertainty,
        "pysyd_crosscheck": pysyd_result,
        "external_adapters": {
            "pysyd": {
                "status": pysyd_adapter["status"],
                "manifest_path": pysyd_adapter["manifest_path"].relative_to(workspace.path).as_posix(),
            },
            "tess-atl": {
                "status": tess_atl_adapter["status"],
                "manifest_path": tess_atl_adapter["manifest_path"].relative_to(workspace.path).as_posix(),
            },
        },
        "caveat": (
            "Candidate envelope peaks and spacing correlations are preliminary "
            "diagnostics; calibrated detection probabilities require null "
            "simulations and injection/recovery gates. Missing candidate "
            "ephemerides leave transit cadences unmasked rather than synthetic-masked."
        ),
    }
    output_path = outputs_dir / "asteroseismic_results.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
