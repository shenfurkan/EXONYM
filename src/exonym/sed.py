"""Target-neutral spectral energy distribution (SED) engine.

Fits multi-band broadband photometry (Gaia DR3 G/BP/RP, 2MASS J/H/Ks, AllWISE W1-W4)
against synthetic stellar atmosphere models (BT-Settl / Kurucz) or a reddened
Planck blackbody model to derive fundamental host star parameters:
- Effective temperature (Teff)
- Surface gravity (log g)
- Metallicity ([Fe/H])
- Angular diameter / scaled radius (R_star / d)
- Visual interstellar extinction (A_V)

Physical Model:
    Observed flux density at Earth:
        F_nu = pi * B_nu(Teff) * (R_star / d)^2 * 10^(-0.4 * A_lambda)
    where A_lambda is computed via standard interstellar extinction laws
    (Cardelli, Clayton & Mathis 1989, Fitzpatrick 1999) with R_V = 3.1.

Contains zero target-specific identifiers or constants; all catalog magnitudes and
parallaxes are loaded dynamically from the candidate workspace.

Scientific Boundary:
    The grid path requires a candidate-supplied atmosphere table; the fallback
    is a pivot-wavelength reddened-blackbody approximation.  Neither path is a
    response-integrated atmosphere posterior or an automatic validation
    constraint.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.constants import c, h, k
from scipy.interpolate import LinearNDInterpolator

from .constants import (
    NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K as TEFF_SUN_K,
    NOMINAL_SOLAR_LOGG_CGS as LOGG_SUN_CGS,
    NOMINAL_SOLAR_RADIUS_M as RSUN_M,
    PARSEC_M as PC_M,
)
from .inputs import load_photometry, load_stellar_parameters
from .resources import read_schema_text
from .workspace import CandidateWorkspace

MAG_SYSTEMATIC_FLOOR = 0.05             # Minimum systematic magnitude uncertainty floor
MIST_MAIN_SEQUENCE_INPUT = Path("data/external/mist_main_sequence_input.json")
MIST_ISOCHRONE_GRID = Path("data/external/mist_isochrone_grid.csv")
MIST_ABSOLUTE_MAGNITUDE_COLUMNS = {
    "gaia_g": "gaia_g_abs_mag",
    "gaia_bp": "gaia_bp_abs_mag",
    "gaia_rp": "gaia_rp_abs_mag",
    "twomass_j": "twomass_j_abs_mag",
    "twomass_h": "twomass_h_abs_mag",
    "twomass_ks": "twomass_ks_abs_mag",
}

# 2MASS (Cohen et al. 2003) & AllWISE (Wright et al. 2010) bandpass pivot wavelengths and Vega zero-point fluxes (Jy)
BAND_ZERO_POINTS: Dict[str, Tuple[float, float]] = {
    "J": (1.235, 1594.0),               # 2MASS J-band (pivot: 1.235 um, F_0: 1594.0 Jy)
    "H": (1.662, 1024.0),               # 2MASS H-band (pivot: 1.662 um, F_0: 1024.0 Jy)
    "Ks": (2.159, 666.7),               # 2MASS Ks-band (pivot: 2.159 um, F_0: 666.7 Jy)
    "W1": (3.3526, 309.540),            # WISE W1-band (pivot: 3.35 um, F_0: 309.54 Jy)
    "W2": (4.6028, 171.787),            # WISE W2-band (pivot: 4.60 um, F_0: 171.79 Jy)
    "W3": (11.5608, 31.674),            # WISE W3-band (pivot: 11.56 um, F_0: 31.67 Jy)
    "W4": (22.0883, 8.363),             # WISE W4-band (pivot: 22.09 um, F_0: 8.36 Jy)
}

# Standard Milky Way interstellar extinction ratios A_lambda / A_V (Fitzpatrick 1999, R_V = 3.1)
EXTINCTION_RATIOS: Dict[str, float] = {
    "J": 0.282,
    "H": 0.190,
    "Ks": 0.114,
    "W1": 0.067,
    "W2": 0.054,
    "W3": 0.024,
    "W4": 0.015,
}


def percentile_summary(samples: np.ndarray) -> Dict[str, float]:
    """Summarize sampled values with central percentile and asymmetric errors.

    Args:
        samples (np.ndarray): Numeric posterior or Monte Carlo samples in one
            declared physical unit.

    Returns:
        Dict[str, float]: Lower percentile, median, upper percentile, and
        positive and negative offsets in the same unit as ``samples``.

    Note:
        This is a descriptive quantile summary; it does not assess sampler
        convergence or turn a model approximation into a calibrated posterior.
    """
    quantiles = np.quantile(np.asarray(samples), [0.16, 0.50, 0.84])
    return {
        "p16": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p84": float(quantiles[2]),
        "plus": float(quantiles[2] - quantiles[1]),
        "minus": float(quantiles[1] - quantiles[0]),
    }


def blackbody_model_magnitudes(
    teff_k: float,
    log_radius_over_distance: float,
    av_mag: float,
    band_data: Sequence[Tuple[str, float, float]],
) -> np.ndarray:
    """Evaluate a reddened blackbody approximation at band pivot wavelengths.

    Mathematical Formulation:
        The model evaluates ``F_nu = pi B_nu(Teff) (R / d)**2`` at each pivot
        wavelength, converts to a Vega magnitude, then adds
        ``A_lambda = A_V (A_lambda / A_V)``.  This matches the monochromatic
        approximation described in the project's stellar-physics note.

    Args:
        teff_k (float): Effective temperature in kelvin.
        log_radius_over_distance (float): Natural logarithm of the
            dimensionless radius-to-distance scale.
        av_mag (float): Visual extinction in magnitudes.
        band_data (Sequence[Tuple[str, float, float]]): Band name, pivot
            wavelength in microns, and Vega zero-point flux in janskys.

    Returns:
        np.ndarray: Model Vega magnitudes in the same order as ``band_data``.

    Raises:
        ValueError: Inputs are non-finite, physically invalid, unsupported, or
            cannot produce finite model magnitudes.

    Note:
        Pivot-wavelength fluxes approximate passband-integrated photometry and
        should not be interpreted as a response-integrated atmosphere model.
    """
    try:
        teff_value = float(teff_k)
        log_scale = float(log_radius_over_distance)
        extinction = float(av_mag)
    except (TypeError, ValueError) as exc:
        raise ValueError("blackbody inputs must be finite physical values") from exc
    if (
        not math.isfinite(teff_value)
        or teff_value <= 0.0
        or not math.isfinite(log_scale)
        or not math.isfinite(extinction)
        or extinction < 0.0
    ):
        raise ValueError("teff_k must be finite and positive; scale finite; av_mag finite and non-negative")
    try:
        radius_distance = math.exp(log_scale)
    except OverflowError as exc:
        raise ValueError("log_radius_over_distance produces a non-finite scale") from exc
    if not math.isfinite(radius_distance) or radius_distance <= 0.0:
        raise ValueError("log_radius_over_distance produces a non-finite scale")

    model = []
    for row in band_data:
        try:
            name, wavelength_micron, zero_jy = row
            wavelength_micron = float(wavelength_micron)
            zero_jy = float(zero_jy)
        except (TypeError, ValueError) as exc:
            raise ValueError("each band_data row must contain a name, wavelength, and zero point") from exc
        if (
            not isinstance(name, str)
            or name not in EXTINCTION_RATIOS
            or not math.isfinite(wavelength_micron)
            or wavelength_micron <= 0.0
            or not math.isfinite(zero_jy)
            or zero_jy <= 0.0
        ):
            raise ValueError("band_data must use supported bands with finite positive wavelengths and zero points")

        wavelength = wavelength_micron * 1e-6
        frequency = c / wavelength
        exponent = h * frequency / (k * teff_value)
        if not math.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("band_data and teff_k produce an invalid Planck exponent")
        try:
            intensity = 2.0 * h * frequency**3 / c**2 / math.expm1(exponent)
        except OverflowError as exc:
            raise ValueError("band_data and teff_k produce an unrepresentable Planck exponent") from exc
        flux_jy = np.pi * intensity * radius_distance**2 / 1e-26
        if not math.isfinite(flux_jy) or flux_jy <= 0.0:
            raise ValueError("blackbody inputs produce a non-finite flux")
        magnitude = -2.5 * math.log10(flux_jy / zero_jy) + extinction * EXTINCTION_RATIOS[name]
        if not math.isfinite(magnitude):
            raise ValueError("blackbody inputs produce a non-finite magnitude")
        model.append(magnitude)
    if not model:
        raise ValueError("band_data must contain at least one supported band")
    return np.asarray(model, dtype=float)


def load_atmosphere_grid_model(
    workspace: CandidateWorkspace,
    band_names: Sequence[str],
) -> Optional[Callable[[float, float, float], np.ndarray]]:
    """Return a candidate-local generic atmosphere-grid interpolator, or ``None``.

    The grid CSV must contain ``teff_k``, ``logg_cgs``, ``feh`` columns plus
    magnitude columns named ``mag_<band>`` for the observed bands. Returns a
    callable mapping (teff_k, logg_cgs, feh) -> magnitude array.

    Args:
        workspace (CandidateWorkspace): Workspace that may own the optional
            atmosphere-grid CSV.
        band_names (Sequence[str]): Required observed band labels.  Each must
            have a corresponding grid magnitude column.

    Returns:
        Optional[Callable[[float, float, float], np.ndarray]]: Interpolator
        from effective temperature in kelvin, surface gravity in cgs log units,
        and metallicity in dex to model magnitudes, or ``None`` if the optional
        grid is unavailable or unusable.

    Note:
        Queries outside the linear interpolation hull use nearest-grid fallback.
        This makes availability explicit but does not establish model adequacy.
    """
    path = workspace.path / "data" / "external" / "atmosphere_grid.csv"
    if not path.is_file():
        return None
    try:
        import pandas as pd
        from scipy.interpolate import griddata

        frame = pd.read_csv(path)
    except (ImportError, OSError, ValueError, pd.errors.ParserError) as exc:
        logging.warning("atmosphere grid load failed for %s: %s", path.name, exc)
        return None
    except Exception as exc:
        logging.warning("atmosphere grid load failed for %s: %s", path.name, exc)
        return None
    required = ("teff_k", "logg_cgs", "feh")
    if not all(column in frame.columns for column in required):
        return None
    mag_columns = [f"mag_{name}" for name in band_names]
    if not all(column in frame.columns for column in mag_columns):
        return None
    points = np.column_stack(
        (
            frame["teff_k"].to_numpy(float),
            frame["logg_cgs"].to_numpy(float),
            frame["feh"].to_numpy(float),
        )
    )
    values = np.column_stack([frame[column].to_numpy(float) for column in mag_columns])

    def model(teff_k: float, logg_cgs: float, feh: float) -> np.ndarray:
        query = np.array([[teff_k, logg_cgs, feh]], dtype=float)
        interpolated = griddata(points, values, query, method="linear")
        if np.any(~np.isfinite(interpolated)):
            interpolated = griddata(points, values, query, method="nearest")
        return np.asarray(interpolated[0], dtype=float)

    return model


def _run_emcee(
    log_probability: Callable[[np.ndarray], float],
    start: np.ndarray,
    n_walkers: int,
    burn_in: int,
    production: int,
    seed: int,
) -> Tuple[np.ndarray, Any]:
    import emcee

    ndim = int(start.size)
    rng = np.random.default_rng(seed=seed)
    walkers = start + rng.normal(size=(n_walkers, ndim)) * 1e-3
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability)
    sampler.random_state = np.random.RandomState(seed).get_state()
    state = sampler.run_mcmc(walkers, burn_in, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, production, progress=False)
    samples = sampler.get_chain(flat=True)
    return samples, sampler


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one candidate-owned artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError("non-finite JSON constant: {0}".format(value))


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate JSON key: {0}".format(key))
        payload[key] = value
    return payload


def _candidate_artifact(
    workspace: CandidateWorkspace, artifact: Mapping[str, Any], label: str
) -> Dict[str, str]:
    """Verify and normalize one hash-bound candidate-local input artifact."""
    relative_value = artifact.get("path")
    digest = artifact.get("sha256")
    role = artifact.get("role")
    if not isinstance(relative_value, str) or not isinstance(digest, str) or not isinstance(role, str):
        raise RuntimeError("{0} must declare path, sha256, and role".format(label))
    relative_path = Path(relative_value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError("{0} must remain inside the candidate workspace".format(label))
    path = workspace.path / relative_path
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(workspace.path.resolve()):
        raise RuntimeError("{0} must reference a regular candidate-owned file".format(label))
    if _sha256(path) != digest:
        raise RuntimeError("{0} SHA-256 does not match its candidate-owned file".format(label))
    return {"path": relative_path.as_posix(), "sha256": digest, "role": role}


def _read_mist_main_sequence_input(
    workspace: CandidateWorkspace,
) -> Optional[Tuple[Path, Dict[str, Any], Path, List[Dict[str, str]]]]:
    """Load one schema-valid frozen MIST manifest and all bound source hashes."""
    manifest_path = workspace.path / MIST_MAIN_SEQUENCE_INPUT
    grid_path = workspace.path / MIST_ISOCHRONE_GRID
    if not manifest_path.exists() and not grid_path.exists():
        return None
    if not manifest_path.is_file() or manifest_path.is_symlink() or not grid_path.is_file() or grid_path.is_symlink():
        raise RuntimeError(
            "MIST main-sequence checking requires both candidate-owned manifest and frozen grid files"
        )
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8-sig"),
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("MIST main-sequence input is not valid finite UTF-8 JSON: {0}".format(exc)) from exc
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema is required to validate MIST main-sequence input") from exc
    try:
        schema = json.loads(
            read_schema_text(workspace.repository_root, "mist-main-sequence-input.schema.json")
        )
        jsonschema.validate(payload, schema, format_checker=jsonschema.FormatChecker())
    except (OSError, ValueError, jsonschema.ValidationError) as exc:
        raise RuntimeError("MIST main-sequence input schema violation: {0}".format(exc)) from exc
    if not isinstance(payload, dict) or payload.get("candidate_id") != workspace.candidate_id:
        raise RuntimeError("MIST main-sequence input candidate_id does not match the workspace")

    grid_artifact = _candidate_artifact(workspace, payload["grid_artifact"], "MIST grid artifact")
    if grid_artifact["path"] != MIST_ISOCHRONE_GRID.as_posix():
        raise RuntimeError("MIST grid artifact must reference data/external/mist_isochrone_grid.csv")
    source_artifacts = [
        _candidate_artifact(workspace, artifact, "MIST source artifact")
        for artifact in payload["provenance"]["input_artifacts"]
    ]
    return manifest_path, payload, grid_path, source_artifacts


def _mist_measurement(payload: Mapping[str, Any], catalog: str, band: str) -> Tuple[float, float]:
    """Return one finite frozen magnitude and uncertainty in magnitudes."""
    measurement = payload["photometry"][catalog][band]
    value = float(measurement["value"])
    uncertainty = float(measurement["uncertainty"])
    if not math.isfinite(value) or not math.isfinite(uncertainty) or uncertainty <= 0.0:
        raise RuntimeError("MIST photometry must contain finite magnitudes and positive uncertainties")
    return value, uncertainty


def _load_mist_main_sequence_grid(grid_path: Path) -> Dict[str, np.ndarray]:
    """Read main-sequence rows from a frozen MIST grid without nearest-grid fallback."""
    required_columns = {
        "evolutionary_stage",
        "teff_k",
        "logg_cgs",
        "feh",
        *MIST_ABSOLUTE_MAGNITUDE_COLUMNS.values(),
    }
    rows: Dict[str, List[float]] = {name: [] for name in required_columns if name != "evolutionary_stage"}
    try:
        with grid_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise RuntimeError("frozen MIST grid lacks required main-sequence columns")
            for row in reader:
                if row.get("evolutionary_stage") != "main_sequence":
                    continue
                parsed = {
                    name: float(row[name])
                    for name in rows
                }
                if not all(math.isfinite(value) for value in parsed.values()):
                    raise RuntimeError("frozen MIST grid contains a non-finite main-sequence row")
                for name, value in parsed.items():
                    rows[name].append(value)
    except (OSError, UnicodeError, TypeError, ValueError, csv.Error) as exc:
        raise RuntimeError("frozen MIST grid is unreadable or non-finite: {0}".format(exc)) from exc
    if len(rows["teff_k"]) < 4:
        raise RuntimeError("frozen MIST grid requires at least four finite main-sequence rows")
    return {name: np.asarray(values, dtype=float) for name, values in rows.items()}


def _mist_main_sequence_check(
    workspace: CandidateWorkspace, stellar: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare frozen Gaia/2MASS absolute magnitudes to MIST main-sequence rows."""
    loaded = _read_mist_main_sequence_input(workspace)
    if loaded is None:
        return {
            "status": "not-configured",
            "validation_eligible": False,
            "claim_eligible": False,
            "interpretation": "No frozen MIST main-sequence manifest and grid were supplied for this exploratory SED run.",
        }
    manifest_path, payload, grid_path, source_artifacts = loaded
    if stellar.get("source") != "candidate-data":
        return {
            "status": "insufficient-input",
            "input_artifact": {
                "path": MIST_MAIN_SEQUENCE_INPUT.as_posix(),
                "sha256": _sha256(manifest_path),
                "role": "mist-main-sequence-input",
            },
            "grid_artifact": {
                "path": MIST_ISOCHRONE_GRID.as_posix(),
                "sha256": _sha256(grid_path),
                "role": "mist-isochrone-grid",
            },
            "source_artifacts": source_artifacts,
            "validation_eligible": False,
            "claim_eligible": False,
            "interpretation": "MIST comparison requires complete candidate-owned stellar parameters.",
        }
    stellar_parameters_path = workspace.path / "data" / "external" / "stellar_params.json"
    if (
        not stellar_parameters_path.is_file()
        or stellar_parameters_path.is_symlink()
        or not stellar_parameters_path.resolve().is_relative_to(workspace.path.resolve())
    ):
        raise RuntimeError(
            "MIST comparison requires a regular candidate-owned stellar_params.json artifact"
        )
    stellar_parameters_artifact = {
        "path": "data/external/stellar_params.json",
        "sha256": _sha256(stellar_parameters_path),
        "role": "stellar-parameters",
    }
    try:
        parallax = float(stellar["parallax_mas"])
        parallax_error = float(stellar["parallax_mas_err"])
        target = np.array(
            [float(stellar["teff_k"]), float(stellar["logg_cgs"]), float(stellar["feh"])],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("MIST comparison requires finite candidate-owned parallax and stellar parameters") from exc
    if (
        not np.all(np.isfinite(target))
        or not math.isfinite(parallax)
        or not math.isfinite(parallax_error)
        or parallax <= 0.0
        or parallax_error <= 0.0
    ):
        raise RuntimeError("MIST comparison requires positive finite parallax and finite stellar parameters")
    parallax_snr = parallax / parallax_error
    if parallax_snr < 5.0:
        return {
            "status": "insufficient-input",
            "input_artifact": {
                "path": MIST_MAIN_SEQUENCE_INPUT.as_posix(),
                "sha256": _sha256(manifest_path),
                "role": "mist-main-sequence-input",
            },
            "grid_artifact": {
                "path": MIST_ISOCHRONE_GRID.as_posix(),
                "sha256": _sha256(grid_path),
                "role": "mist-isochrone-grid",
            },
            "source_artifacts": source_artifacts,
            "parallax_snr": parallax_snr,
            "validation_eligible": False,
            "claim_eligible": False,
            "interpretation": "MIST color-magnitude comparison requires parallax SNR of at least 5.",
        }

    grid = _load_mist_main_sequence_grid(grid_path)
    points = np.column_stack((grid["teff_k"], grid["logg_cgs"], grid["feh"]))
    base_result: Dict[str, Any] = {
        "input_artifact": {
            "path": MIST_MAIN_SEQUENCE_INPUT.as_posix(),
            "sha256": _sha256(manifest_path),
            "role": "mist-main-sequence-input",
        },
        "grid_artifact": {
            "path": MIST_ISOCHRONE_GRID.as_posix(),
            "sha256": _sha256(grid_path),
            "role": "mist-isochrone-grid",
        },
        "stellar_parameters_artifact": stellar_parameters_artifact,
        "source_artifacts": source_artifacts,
        "parallax_snr": parallax_snr,
        "interpolation_parameters": {
            "teff_k": float(target[0]),
            "logg_cgs": float(target[1]),
            "feh": float(target[2]),
        },
        "validation_eligible": False,
        "claim_eligible": False,
    }
    if any(target[index] < np.min(points[:, index]) or target[index] > np.max(points[:, index]) for index in range(points.shape[1])):
        return {
            **base_result,
            "status": "outside-grid",
            "interpretation": "Candidate stellar parameters fall outside the frozen MIST main-sequence grid; no nearest-grid fallback was used.",
        }
    try:
        bands = tuple(MIST_ABSOLUTE_MAGNITUDE_COLUMNS)
        values = np.column_stack(
            [grid[MIST_ABSOLUTE_MAGNITUDE_COLUMNS[band]] for band in bands]
        )
        interpolated = np.asarray(LinearNDInterpolator(points, values)(target), dtype=float)
        if interpolated.size != len(bands):
            raise RuntimeError("frozen MIST grid interpolation returned an invalid vector shape")
        predicted = {
            band: float(value)
            for band, value in zip(bands, interpolated.reshape(-1))
        }
    except Exception as exc:
        raise RuntimeError("frozen MIST grid cannot interpolate the declared stellar parameters: {0}".format(exc)) from exc
    if not all(math.isfinite(value) for value in predicted.values()):
        return {
            **base_result,
            "status": "outside-grid",
            "interpretation": "Candidate stellar parameters fall outside the frozen MIST interpolation hull; no nearest-grid fallback was used.",
        }

    distance_pc = 1000.0 / parallax
    distance_modulus = 5.0 * math.log10(distance_pc / 10.0)
    distance_modulus_uncertainty = 5.0 * parallax_error / (math.log(10.0) * parallax)
    band_locations = {
        "gaia_g": ("gaia_dr3", "g"),
        "gaia_bp": ("gaia_dr3", "bp"),
        "gaia_rp": ("gaia_dr3", "rp"),
        "twomass_j": ("twomass", "j"),
        "twomass_h": ("twomass", "h"),
        "twomass_ks": ("twomass", "ks"),
    }
    observed_absolute: Dict[str, float] = {}
    observed_uncertainty: Dict[str, float] = {}
    residuals: Dict[str, float] = {}
    extinction = payload["extinction"]["band_extinction_mag"]
    for band, (catalog, measurement_name) in band_locations.items():
        magnitude, magnitude_uncertainty = _mist_measurement(payload, catalog, measurement_name)
        extinction_mag = float(extinction[band])
        if not math.isfinite(extinction_mag) or extinction_mag < 0.0:
            raise RuntimeError("MIST extinction values must be finite and non-negative")
        absolute_magnitude = magnitude - distance_modulus - extinction_mag
        uncertainty = math.sqrt(magnitude_uncertainty**2 + distance_modulus_uncertainty**2)
        observed_absolute[band] = absolute_magnitude
        observed_uncertainty[band] = uncertainty
        residuals[band] = absolute_magnitude - predicted[band]
    chi_square = float(
        sum((residuals[band] / observed_uncertainty[band]) ** 2 for band in band_locations)
    )
    return {
        **base_result,
        "status": "evaluated",
        "method": "frozen-mist-main-sequence-linear-interpolation",
        "observed_absolute_magnitudes": observed_absolute,
        "absolute_magnitude_uncertainties": observed_uncertainty,
        "interpolated_main_sequence_absolute_magnitudes": predicted,
        "residuals_mag": residuals,
        "chi_square_fixed_stellar_parameters": chi_square,
        "interpretation": (
            "This frozen-grid color-magnitude comparison is descriptive only. It does not "
            "validate a main-sequence classification, infer an age, or replace an isochrone posterior."
        ),
    }


def _fit_blackbody(
    observations: Sequence[Tuple[str, float, float]],
    stellar: Dict[str, Any],
    n_walkers: int = 32,
    burn_in: int = 200,
    production: int = 500,
    seed: int = 7,
) -> Dict[str, Any]:
    band_data = [
        (name, *BAND_ZERO_POINTS[name]) for name, _, _ in observations
    ]
    magnitudes = np.array([row[1] for row in observations], dtype=float)
    # NUMERICAL_GUARD: The declared systematic floor prevents arbitrarily small
    # catalog errors from dominating this approximate pivot-wavelength model.
    errors = np.sqrt(
        np.array([row[2] for row in observations], dtype=float) ** 2
        + MAG_SYSTEMATIC_FLOOR**2
    )
    parallax = float(stellar["parallax_mas"])
    parallax_error = float(stellar.get("parallax_mas_err", float("nan")))
    if not math.isfinite(parallax) or parallax <= 0 or not math.isfinite(parallax_error) or parallax_error <= 0:
        raise RuntimeError("blackbody SED fitting requires positive candidate-owned parallax_mas and parallax_mas_err")
    # ASTROPHYSICAL_GUARD: direct 1/parallax inversion produces a severe
    # Lutz-Kelker-type distance bias when the fractional parallax error is
    # large (Bailer-Jones 2015). Require astrometric SNR >= 5 before trusting
    # the geometric distance prior in this approximate blackbody SED model.
    parallax_snr = parallax / parallax_error
    if parallax_snr < 5.0:
        raise RuntimeError(
            "blackbody SED fitting requires parallax SNR >= 5.0; "
            "observed SNR {0:.2f} would introduce a severe Lutz-Kelker "
            "distance bias into the derived stellar radius".format(parallax_snr)
        )
    distance_pc = 1000.0 / parallax
    teff_prior = float(stellar["teff_k"])
    initial_scale = 1.0 * RSUN_M / (distance_pc * PC_M)

    def log_probability(theta: np.ndarray) -> float:
        teff, log_scale, av = float(theta[0]), float(theta[1]), float(theta[2])
        if not 3500.0 < teff < 8000.0 or not 0.0 < av < 0.5:
            return -np.inf
        if not np.log(initial_scale / 3.0) < log_scale < np.log(initial_scale * 3.0):
            return -np.inf
        model = blackbody_model_magnitudes(teff, log_scale, av, band_data)
        likelihood = -0.5 * np.sum(
            ((magnitudes - model) / errors) ** 2 + np.log(2.0 * np.pi * errors**2)
        )
        temperature_prior = -0.5 * ((teff - teff_prior) / 200.0) ** 2
        extinction_prior = -0.5 * (av / 0.05) ** 2
        return float(likelihood + temperature_prior + extinction_prior)

    start = np.array([teff_prior, np.log(initial_scale), 0.02])
    samples, sampler = _run_emcee(
        log_probability, start, n_walkers, burn_in, production, seed
    )

    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0 or not np.all(np.isfinite(samples)):
        raise RuntimeError("blackbody SED sampler returned no finite posterior samples")
    draw_rng = np.random.default_rng(seed=seed + 1)
    proposed_parallax_draws = np.asarray(
        draw_rng.normal(parallax, parallax_error, samples.shape[0]), dtype=float
    )
    # NUMERICAL_GUARD: Never invert non-positive or non-finite parallax draws.
    valid_parallax_draws = np.isfinite(proposed_parallax_draws) & (
        proposed_parallax_draws > 0.0
    )
    parallax_draws = proposed_parallax_draws[valid_parallax_draws]
    derived_samples = samples[valid_parallax_draws]
    rejected_parallax_draws = int(proposed_parallax_draws.size - parallax_draws.size)
    if parallax_draws.size == 0:
        raise RuntimeError(
            "blackbody SED fitting rejected every parallax draw as non-positive or non-finite"
        )
    distance_draws = 1000.0 / parallax_draws
    radius_draws = np.exp(derived_samples[:, 1]) * distance_draws * PC_M / RSUN_M
    luminosity_draws = radius_draws**2 * (derived_samples[:, 0] / TEFF_SUN_K) ** 4
    mass_prior = float(stellar["mass_solar"])
    if not math.isfinite(mass_prior) or mass_prior <= 0.0:
        raise RuntimeError("blackbody SED fitting requires positive candidate-owned mass_solar")
    logg_draws = LOGG_SUN_CGS + np.log10(mass_prior) - 2.0 * np.log10(radius_draws)
    if not (
        np.all(np.isfinite(distance_draws))
        and np.all(distance_draws > 0.0)
        and np.all(np.isfinite(radius_draws))
        and np.all(radius_draws > 0.0)
        and np.all(np.isfinite(luminosity_draws))
        and np.all(luminosity_draws > 0.0)
        and np.all(np.isfinite(logg_draws))
    ):
        raise RuntimeError("blackbody SED fitting produced invalid derived posterior samples")

    median = np.median(samples, axis=0)
    model_at_median = blackbody_model_magnitudes(
        float(median[0]), float(median[1]), float(median[2]), band_data
    )
    residuals = magnitudes - model_at_median
    distance_model = {
        "method": "geometric-parallax-inversion",
        "parallax_snr": float(parallax_snr),
    }
    if parallax_snr <= 10.0:
        distance_model["caveat"] = (
            "At parallax SNR from 5 to 10, direct geometric 1/parallax distance "
            "inversion retains an approximately 1-4% second-order distance asymmetry. "
            "This approximate SED fit does not replace a Bayesian distance inference."
        )
    return {
        "model": "reddened blackbody at catalog pivot wavelengths",
        "posterior": {
            "teff_k": percentile_summary(samples[:, 0]),
            "av_mag": percentile_summary(samples[:, 2]),
            "distance_pc": percentile_summary(distance_draws),
            "radius_solar": percentile_summary(radius_draws),
            "luminosity_solar": percentile_summary(luminosity_draws),
            "logg_cgs": percentile_summary(logg_draws),
        },
        "photometry": [
            {
                "band": name,
                "observed_mag": float(observed),
                "total_error_mag": float(total_error),
                "model_mag_at_posterior_median": float(model),
                "residual_mag": float(residual),
            }
            for (name, observed, _), total_error, model, residual in zip(
                observations, errors, model_at_median, residuals
            )
        ],
        "fit_quality": {
            "chi_square_at_posterior_median": float(np.sum((residuals / errors) ** 2)),
            "degrees_of_freedom": len(observations) - 3,
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "retained_samples": int(len(samples)),
            "parallax_draws": {
                "proposed_count": int(proposed_parallax_draws.size),
                "accepted_positive_finite_count": int(parallax_draws.size),
                "rejected_nonpositive_or_nonfinite_count": rejected_parallax_draws,
                "rejection_rate": float(
                    rejected_parallax_draws / proposed_parallax_draws.size
                ),
                "policy": (
                    "non-positive or non-finite draws are rejected before "
                    "distance-derived summaries"
                ),
            },
        },
        "metadata": {"distance_model": distance_model},
        "samples": samples,
    }


def _fit_grid(
    observations: Sequence[Tuple[str, float, float]],
    grid_model: Callable[[float, float, float], np.ndarray],
    stellar: Dict[str, Any],
    n_walkers: int = 32,
    burn_in: int = 200,
    production: int = 500,
    seed: int = 7,
) -> Dict[str, Any]:
    magnitudes = np.array([row[1] for row in observations], dtype=float)
    # NUMERICAL_GUARD: Apply the same magnitude-error floor to both model paths
    # so a grid lookup is not given unbounded weight over catalog systematics.
    errors = np.sqrt(
        np.array([row[2] for row in observations], dtype=float) ** 2
        + MAG_SYSTEMATIC_FLOOR**2
    )
    teff_prior = float(stellar["teff_k"])
    logg_prior = float(stellar["logg_cgs"])
    feh_prior = float(stellar["feh"])

    def log_probability(theta: np.ndarray) -> float:
        teff, logg, feh, offset = (
            float(theta[0]),
            float(theta[1]),
            float(theta[2]),
            float(theta[3]),
        )
        if not 3500.0 < teff < 8000.0 or not 2.0 < logg < 5.5:
            return -np.inf
        if not -2.0 < feh < 1.0 or not -5.0 < offset < 5.0:
            return -np.inf
        model = np.asarray(grid_model(teff, logg, feh), dtype=float) + offset
        likelihood = -0.5 * np.sum(
            ((magnitudes - model) / errors) ** 2 + np.log(2.0 * np.pi * errors**2)
        )
        prior = (
            -0.5 * ((teff - teff_prior) / 200.0) ** 2
            - 0.5 * ((logg - logg_prior) / 0.25) ** 2
            - 0.5 * ((feh - feh_prior) / 0.2) ** 2
        )
        return float(likelihood + prior)

    start = np.array([teff_prior, logg_prior, feh_prior, 0.0])
    samples, sampler = _run_emcee(
        log_probability, start, n_walkers, burn_in, production, seed
    )
    median = np.median(samples, axis=0)
    model_at_median = (
        np.asarray(grid_model(float(median[0]), float(median[1]), float(median[2])))
        + float(median[3])
    )
    residuals = magnitudes - model_at_median
    return {
        "model": "generic atmosphere-grid interpolation with free magnitude offset",
        "posterior": {
            "teff_k": percentile_summary(samples[:, 0]),
            "logg_cgs": percentile_summary(samples[:, 1]),
            "feh": percentile_summary(samples[:, 2]),
            "magnitude_offset": percentile_summary(samples[:, 3]),
        },
        "photometry": [
            {
                "band": name,
                "observed_mag": float(observed),
                "total_error_mag": float(total_error),
                "model_mag_at_posterior_median": float(model),
                "residual_mag": float(residual),
            }
            for (name, observed, _), total_error, model, residual in zip(
                observations, errors, model_at_median, residuals
            )
        ],
        "fit_quality": {
            "chi_square_at_posterior_median": float(np.sum((residuals / errors) ** 2)),
            "degrees_of_freedom": len(observations) - 4,
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "retained_samples": int(len(samples)),
        },
        "samples": samples,
    }


def _collect_observations(
    photometry: Optional[Dict[str, Any]]
) -> Tuple[Optional[List[Tuple[str, float, float]]], str]:
    """Extract (band, mag, error) rows from the generic photometry JSON."""
    if photometry is None:
        return None, "no-photometry-file"
    rows: List[Tuple[str, float, float]] = []
    for catalog_name in ("2MASS", "AllWISE"):
        catalog = photometry.get(catalog_name)
        if not isinstance(catalog, dict):
            continue
        for band, value in catalog.items():
            if not isinstance(value, dict):
                continue
            mag = value.get("mag")
            error = value.get("error")
            if mag is None or error is None:
                continue
            try:
                rows.append((str(band), float(mag), float(error)))
            except (TypeError, ValueError):
                continue
    if not rows:
        return None, "no-readable-photometry"
    return rows, "candidate-data"


def _synthetic_photometry(stellar: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    """Deterministic demonstration photometry from a reddened blackbody."""
    rng = np.random.default_rng(seed=7)
    teff = float(stellar["teff_k"])
    radius = float(stellar["radius_solar"])
    distance_pc = 1000.0 / float(stellar["parallax_mas"])
    log_scale = math.log(radius * RSUN_M / (distance_pc * PC_M))
    av = 0.02
    band_names = list(BAND_ZERO_POINTS)
    band_data = [(name, *BAND_ZERO_POINTS[name]) for name in band_names]
    model = blackbody_model_magnitudes(teff, log_scale, av, band_data)
    rows = []
    for name, magnitude in zip(band_names, model):
        observed = magnitude + rng.normal(0.0, 0.02)
        rows.append((name, float(observed), 0.02))
    return rows


def run_sed_fit(workspace: CandidateWorkspace) -> Path:
    """Run the candidate-local exploratory broadband SED fit.

    The runner uses a candidate-supplied atmosphere grid when one has the
    required columns; otherwise it uses the documented reddened-blackbody
    pivot-wavelength approximation.  It records posterior quantiles, input
    photometry, residuals, sampler diagnostics, and model caveats.

    Args:
        workspace (CandidateWorkspace): Workspace that owns broadband
            photometry, stellar parameters, optional grid data, and outputs.

    Returns:
        Path: Candidate-local ``outputs/sed_fit_results.json``.  The result
        records the model path and its known approximation limits.

    Raises:
        RuntimeError: Candidate-owned photometry or required parallax data for
            the blackbody path is unavailable or invalid.
        OSError: Candidate-local output artifacts cannot be written.

    Note:
        The fit is an exploratory color diagnostic.  It does not provide a
        calibrated atmosphere posterior, a validation constraint, or a
        lifecycle decision.
    """
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    stellar = load_stellar_parameters(workspace)
    photometry = load_photometry(workspace)
    observations, source = _collect_observations(photometry)
    grid_used = False
    if observations is None:
        raise RuntimeError("SED fitting requires candidate-owned broadband photometry")
    mist_main_sequence_check = _mist_main_sequence_check(workspace, stellar)
    grid_model = load_atmosphere_grid_model(
        workspace, [name for name, _, _ in observations]
    )
    if grid_model is not None:
        grid_used = True
    grid_load_failed = not grid_used and (workspace.path / "data" / "external" / "atmosphere_grid.csv").is_file()

    # SCIENTIFIC_BOUNDARY: The fallback is recorded as a blackbody approximation
    # rather than being presented as an atmosphere-grid inference.
    fit = (
        _fit_grid(observations, grid_model, stellar)  # type: ignore[arg-type]
        if grid_used
        else _fit_blackbody(observations, stellar)
    )
    samples = fit.pop("samples", None)

    payload = {
        "schema_version": "1.0",
        "candidate_id": workspace.candidate_id,
        "work_package": "SED_FIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "scientific_status": "exploratory-color-fit",
        "validation_eligible": False,
        "grid_used": grid_used,
        "grid_load_failed": grid_load_failed,
        "grid_source": (
            "candidate-data/external/atmosphere_grid.csv" if grid_used else "blackbody-fallback"
        ),
        "method": fit["model"],
        "input_photometry": [
            {"band": name, "mag": mag, "error": error}
            for name, mag, error in observations
        ],
        "posterior": fit["posterior"],
        "photometry": fit["photometry"],
        "fit_quality": fit["fit_quality"],
        "metadata": fit.get("metadata", {}),
        "mist_main_sequence_check": mist_main_sequence_check,
        "caveats": [
            "Pivot-wavelength monochromatic models approximate passband-integrated photometry.",
            "Radius and luminosity are derived only for the blackbody path via the parallax prior.",
            "Grid magnitudes carry an unknown absolute normalization; a free offset absorbs it.",
            "This exploratory fit is not a response-integrated atmosphere posterior or a validation constraint.",
        ],
    }
    output_path = outputs_dir / "sed_fit_results.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if samples is not None:
        np.save(str(outputs_dir / "sed_fit_chain.npy"), samples)
    return output_path


def cross_match_isochrone_evolution(
    teff_k: float,
    logg_cgs: float,
    feh_dex: float,
    radius_solar: float,
) -> Dict[str, Any]:
    """Validate stellar parameters against canonical main-sequence scaling.

    Mathematical Formulation:
        For main-sequence dwarf stars, effective temperature sets the zero-age
        main-sequence (ZAMS) radius according to ``R_MS ~ (Teff / Teff_sun)**1.5``
        and the expected surface gravity ``log g_MS ~ log g_sun - 0.5 * log10(Teff / Teff_sun)``.
        Deviations ``|log g_obs - log g_MS| >= 0.4 dex`` or radius swellings
        ``R / R_MS > 1.5`` flag evolved subgiant or red giant stages.

    Args:
        teff_k (float): Effective temperature in kelvin.
        logg_cgs (float): Logarithmic surface gravity in cgs units.
        feh_dex (float): Metallicity in dex.
        radius_solar (float): Stellar radius in solar units.

    Returns:
        Dict[str, Any]: Main-sequence consistency metrics, including temperature
        ratio vs. Sun, expected main-sequence log(g), offset, and evolutionary stage.

    Raises:
        ValueError: If input values are non-finite or outside physical domains.
    """
    values = (teff_k, logg_cgs, feh_dex, radius_solar)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("stellar evolutionary parameters must be finite")
    if teff_k <= 0.0:
        raise ValueError("teff_k must be positive")
    if radius_solar <= 0.0:
        raise ValueError("radius_solar must be positive")

    teff_ratio = teff_k / TEFF_SUN_K
    expected_radius_ms = max(teff_ratio**1.5, 0.05)
    expected_logg_ms = LOGG_SUN_CGS - 0.5 * math.log10(teff_ratio)
    logg_offset = abs(logg_cgs - expected_logg_ms)
    radius_ratio_to_ms = radius_solar / expected_radius_ms

    is_ms = (logg_offset < 0.4) and (0.6 <= radius_ratio_to_ms <= 1.5)

    return {
        "method": "analytic-main-sequence-isochrone-consistency",
        "teff_ratio_vs_solar": round(float(teff_ratio), 4),
        "expected_main_sequence_radius_solar": round(float(expected_radius_ms), 3),
        "expected_main_sequence_logg": round(float(expected_logg_ms), 3),
        "observed_logg_offset": round(float(logg_offset), 3),
        "radius_ratio_to_main_sequence": round(float(radius_ratio_to_ms), 3),
        "evolutionary_stage": "main_sequence" if is_ms else "subgiant_or_evolved",
    }
