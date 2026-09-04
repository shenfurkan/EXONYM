"""Candidate-local MIST v1.2 bolometric-correction SED diagnostic.

The runner evaluates only retained, hash-bound MIST v1.2 bolometric-correction
archives supplied below a candidate workspace. It holds the candidate's
atmospheric parameters fixed, profiles apparent bolometric magnitude, and fits
visual extinction within the native MIST table domain. Pivot-wavelength Planck
surrogates, generic atmosphere grids, nearest-node interpolation, and analytic
main-sequence scaling relations are intentionally unsupported.

Scientific boundary:
    The diagnostic reports agreement between candidate-owned Vega photometry
    and MIST bolometric corrections.  It does not infer a stellar radius,
    luminosity, distance, atmosphere posterior, validation constraint, or
    lifecycle decision.

Primary literature, units, and fail-closed domain
--------------------------------------------------
The retained MIST sources are Dotter (2016), ADS ``2016ApJS..222....8D``, DOI
``10.3847/0067-0049/222/1/8``, and Choi et al. (2016), ADS
``2016ApJ...823..102C``, DOI ``10.3847/0004-637X/823/2/102``.  Extinction-table
context is Fitzpatrick (1999), ADS ``1999PASP..111...63F``, DOI
``10.1086/316293``.  Candidate table axes are ``Teff`` in K, ``logg`` in
``log10(cm s^-2)``, ``[Fe/H]`` in dex, and ``A_V`` in mag; observed/model Vega
magnitudes and profiled apparent bolometric magnitude are mag.  Interpolation
is permitted only inside the retained MIST v1.2 table domain.  A missing or
hash-mismatched manifest/table, unsupported band, nonfinite magnitude/error, or
out-of-domain node fails without an SED fit.  This agreement diagnostic does
not infer distance, radius, luminosity, or a claim and always remains
``claim_eligible: false``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.optimize import minimize_scalar

from .resources import read_schema_text
from .workspace import CandidateWorkspace

MIST_MAIN_SEQUENCE_INPUT = Path("data/external/mist_main_sequence_input.json")
MIST_ISOCHRONE_GRID = Path("data/external/mist_isochrone_grid.csv")
MIST_BC_MANIFEST = Path("data/external/sed_input_manifest.json")
MIST_BC_PARAMETER_COLUMNS = ("Teff", "logg", "[Fe/H]", "Av")
MIST_BC_RV_COLUMN = "Rv"
MIST_BC_FIT_PARAMETER_COUNT = len(("av_mag", "apparent_bolometric_mag"))
MIST_BC_TABLE_SUFFIXES = {
    "ubvriplus": "UBVRIplus",
    "wise": "WISE",
}
MIST_BC_METHOD = (
    "MIST v1.2 bolometric-correction table interpolation with profiled "
    "apparent bolometric magnitude and A_V"
)

MIST_ABSOLUTE_MAGNITUDE_COLUMNS = {
    "gaia_g": "gaia_g_abs_mag",
    "gaia_bp": "gaia_bp_abs_mag",
    "gaia_rp": "gaia_rp_abs_mag",
    "twomass_j": "twomass_j_abs_mag",
    "twomass_h": "twomass_h_abs_mag",
    "twomass_ks": "twomass_ks_abs_mag",
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one candidate-owned artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
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


def _read_mist_bc_sed_inputs(
    workspace: CandidateWorkspace,
) -> Tuple[
    List[Dict[str, Any]],
    Tuple[float, float, float],
    float,
    List[Dict[str, str]],
    Dict[str, str],
]:
    """Read schema-valid, hash-bound MIST BC inputs from a candidate workspace."""
    manifest_path = workspace.path / MIST_BC_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("MIST v1.2 bolometric-correction SED input manifest is missing")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8-sig"),
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("MIST v1.2 bolometric-correction SED input manifest is invalid: {0}".format(exc)) from exc
    if not isinstance(manifest, dict) or manifest.get("candidate_id") != workspace.candidate_id:
        raise RuntimeError("MIST v1.2 bolometric-correction SED manifest candidate_id does not match the workspace")
    try:
        import jsonschema

        schema = json.loads(
            read_schema_text(workspace.repository_root, "sed-input-manifest.schema.json")
        )
        jsonschema.validate(manifest, schema, format_checker=jsonschema.FormatChecker())
    except ImportError as exc:
        raise RuntimeError("jsonschema is required to validate MIST v1.2 bolometric-correction SED inputs") from exc
    except (OSError, ValueError, jsonschema.ValidationError) as exc:
        raise RuntimeError("MIST v1.2 bolometric-correction SED manifest schema violation: {0}".format(exc)) from exc

    photometry_artifact = _candidate_artifact(
        workspace, manifest.get("photometry_artifact", {}), "MIST SED photometry artifact"
    )
    stellar_artifact = _candidate_artifact(
        workspace,
        manifest.get("stellar_parameters_artifact", {}),
        "MIST SED stellar-parameters artifact",
    )
    mist = manifest.get("mist")
    if not isinstance(mist, dict):
        raise RuntimeError("MIST v1.2 bolometric-correction SED manifest lacks MIST metadata")
    try:
        declared_rv = float(mist["rv"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("MIST v1.2 bolometric-correction SED manifest has invalid R_V") from exc
    if not math.isfinite(declared_rv) or declared_rv <= 0.0:
        raise RuntimeError("MIST v1.2 bolometric-correction SED manifest R_V must be finite and positive")
    raw_table_artifacts = mist.get("table_artifacts")
    if not isinstance(raw_table_artifacts, list):
        raise RuntimeError("MIST v1.2 bolometric-correction SED manifest lacks table archives")
    table_kinds: set[str] = set()
    table_artifacts: List[Dict[str, str]] = []
    for raw_artifact in raw_table_artifacts:
        if not isinstance(raw_artifact, dict):
            raise RuntimeError("MIST v1.2 bolometric-correction SED table record is invalid")
        table_kind = raw_artifact.get("table_kind")
        if not isinstance(table_kind, str) or table_kind not in MIST_BC_TABLE_SUFFIXES:
            raise RuntimeError("MIST v1.2 bolometric-correction SED table kind is unsupported")
        if table_kind in table_kinds:
            raise RuntimeError("MIST v1.2 bolometric-correction SED manifest has duplicate table kinds")
        table_kinds.add(table_kind)
        table_artifacts.append(
            {
                **_candidate_artifact(workspace, raw_artifact, "MIST bolometric-correction archive"),
                "table_kind": table_kind,
            }
        )
    try:
        photometry = json.loads(
            (workspace.path / photometry_artifact["path"]).read_text(encoding="utf-8-sig"),
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        stellar = json.loads(
            (workspace.path / stellar_artifact["path"]).read_text(encoding="utf-8-sig"),
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("MIST v1.2 SED photometry or stellar parameters are unreadable") from exc
    if not isinstance(photometry, dict) or not isinstance(stellar, dict):
        raise RuntimeError("MIST v1.2 SED photometry and stellar parameters must be JSON objects")
    try:
        atmosphere = tuple(float(stellar[name]) for name in ("teff_k", "logg_cgs", "feh"))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("MIST v1.2 SED requires candidate teff_k, logg_cgs, and feh") from exc
    if not all(math.isfinite(value) for value in atmosphere):
        raise RuntimeError("MIST v1.2 SED atmospheric parameters must be finite")

    declared_photometry = manifest.get("photometry")
    if not isinstance(declared_photometry, dict) or not isinstance(
        declared_photometry.get("measurements"), list
    ):
        raise RuntimeError("MIST v1.2 SED manifest lacks declared photometric measurements")
    observations: List[Dict[str, Any]] = []
    mist_columns: set[str] = set()
    for declared in declared_photometry["measurements"]:
        if not isinstance(declared, dict):
            raise RuntimeError("MIST v1.2 SED declared photometric measurement is invalid")
        catalog = declared.get("catalog")
        catalog_band = declared.get("catalog_band")
        mist_column = declared.get("mist_column")
        if not all(isinstance(value, str) and value for value in (catalog, catalog_band, mist_column)):
            raise RuntimeError("MIST v1.2 SED measurement must declare catalog, catalog band, and MIST column")
        if mist_column in mist_columns:
            raise RuntimeError("MIST v1.2 SED measurements cannot reuse a MIST table column")
        mist_columns.add(mist_column)
        try:
            source_measurement = photometry[catalog][catalog_band]
            magnitude = float(source_measurement["mag"])
            uncertainty = float(source_measurement["error"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "MIST v1.2 SED photometry artifact lacks {0}.{1}".format(catalog, catalog_band)
            ) from exc
        if not math.isfinite(magnitude) or not math.isfinite(uncertainty) or uncertainty <= 0.0:
            raise RuntimeError("MIST v1.2 SED magnitudes must be finite with positive uncertainties")
        observations.append(
            {
                "catalog": catalog,
                "catalog_band": catalog_band,
                "mist_column": mist_column,
                "magnitude": magnitude,
                "uncertainty_mag": uncertainty,
            }
        )
    if len(observations) <= MIST_BC_FIT_PARAMETER_COUNT:
        raise RuntimeError(
            "MIST v1.2 SED requires more independent photometric measurements than fitted parameters"
        )
    manifest_artifact = {
        "path": MIST_BC_MANIFEST.as_posix(),
        "sha256": _sha256(manifest_path),
        "role": "mist-v1.2-bolometric-correction-sed-manifest",
    }
    return (
        observations,
        atmosphere,
        declared_rv,
        [photometry_artifact, stellar_artifact, *table_artifacts],
        manifest_artifact,
    )


def _iter_mist_bc_archive_rows(
    archive_path: Path,
    table_kind: str,
):
    """Yield validated header-index mappings and finite rows from one MIST archive."""
    suffix = ".{0}".format(MIST_BC_TABLE_SUFFIXES[table_kind])
    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError("MIST bolometric-correction archive is unreadable") from exc
    with archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile() and Path(member.name).name.endswith(suffix)
            ),
            key=lambda member: member.name,
        )
        if not members:
            raise RuntimeError("MIST bolometric-correction archive has no expected table members")
        for member in members:
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError("MIST bolometric-correction archive member is unreadable")
            with handle, io.TextIOWrapper(handle, encoding="utf-8") as text:
                header: Optional[Tuple[str, ...]] = None
                header_indexes: Dict[str, int] = {}
                row_count = 0
                for raw_line in text:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        declared_header = line[1:].strip()
                        if declared_header.startswith("Teff "):
                            parsed_header = tuple(declared_header.split())
                            if header is not None and header != parsed_header:
                                raise RuntimeError("MIST bolometric-correction archive has inconsistent repeated headers")
                            header = parsed_header
                            header_indexes = {name: index for index, name in enumerate(header)}
                        continue
                    if header is None:
                        raise RuntimeError("MIST bolometric-correction archive data precedes its table header")
                    values = tuple(line.split())
                    if len(values) != len(header):
                        raise RuntimeError("MIST bolometric-correction archive row width does not match its header")
                    try:
                        row = tuple(float(value) for value in values)
                    except ValueError as exc:
                        raise RuntimeError("MIST bolometric-correction archive contains a non-numeric row") from exc
                    if not all(math.isfinite(value) for value in row):
                        raise RuntimeError("MIST bolometric-correction archive contains a non-finite row")
                    row_count += 1
                    yield header_indexes, row
                if header is None or row_count == 0:
                    raise RuntimeError("MIST bolometric-correction archive member has no tabular data")


def _mist_bc_table_metadata(archive_path: Path, table_kind: str) -> Dict[str, Any]:
    """Derive and validate the native rectilinear coordinate axes of one MIST archive."""
    axes = [set() for _ in MIST_BC_PARAMETER_COLUMNS]
    rv_values: set[float] = set()
    header: Optional[Tuple[str, ...]] = None
    row_count = 0
    for indexes, row in _iter_mist_bc_archive_rows(archive_path, table_kind):
        current_header = tuple(name for name, _ in sorted(indexes.items(), key=lambda item: item[1]))
        required = (*MIST_BC_PARAMETER_COLUMNS, MIST_BC_RV_COLUMN)
        if not all(name in indexes for name in required):
            raise RuntimeError("MIST bolometric-correction archive lacks required coordinate columns")
        if header is None:
            header = current_header
        elif header != current_header:
            raise RuntimeError("MIST bolometric-correction archive members have inconsistent headers")
        for axis, name in zip(axes, MIST_BC_PARAMETER_COLUMNS):
            axis.add(row[indexes[name]])
        rv_values.add(row[indexes[MIST_BC_RV_COLUMN]])
        row_count += 1
    if header is None or len(rv_values) != 1:
        raise RuntimeError("MIST bolometric-correction archive must define exactly one R_V coordinate")
    native_axes = tuple(np.asarray(sorted(axis), dtype=float) for axis in axes)
    if any(axis.size == 0 for axis in native_axes):
        raise RuntimeError("MIST bolometric-correction archive has an empty interpolation axis")
    expected_rows = math.prod(int(axis.size) for axis in native_axes)
    if row_count != expected_rows:
        raise RuntimeError("MIST bolometric-correction archive does not form a complete rectilinear grid")
    return {
        "header": header,
        "axes": native_axes,
        "rv": next(iter(rv_values)),
    }


def _load_mist_bc_interpolators(
    workspace: CandidateWorkspace,
    table_artifacts: Sequence[Mapping[str, str]],
    mist_columns: Sequence[str],
    declared_rv: float,
) -> Tuple[Dict[str, Tuple[RegularGridInterpolator, int]], np.ndarray]:
    """Build no-extrapolation MIST BC interpolators only for declared columns."""
    metadata_by_path: Dict[str, Dict[str, Any]] = {}
    columns_by_path: Dict[str, List[str]] = {}
    for artifact in table_artifacts:
        path = str(artifact["path"])
        metadata = _mist_bc_table_metadata(
            workspace.path / path, str(artifact["table_kind"])
        )
        if metadata["rv"] != declared_rv:
            raise RuntimeError("MIST bolometric-correction archive R_V does not match the manifest declaration")
        metadata_by_path[path] = metadata
        columns_by_path[path] = []
    for column in mist_columns:
        owners = [
            path
            for path, metadata in metadata_by_path.items()
            if column in metadata["header"]
        ]
        if len(owners) != 1:
            raise RuntimeError(
                "MIST bolometric-correction column {0} must occur in exactly one declared archive".format(column)
            )
        columns_by_path[owners[0]].append(column)

    active_paths = [path for path, columns in columns_by_path.items() if columns]
    common_axes = metadata_by_path[active_paths[0]]["axes"]
    for path in active_paths[1:]:
        axes = metadata_by_path[path]["axes"]
        if any(not np.array_equal(axis, reference) for axis, reference in zip(axes, common_axes)):
            raise RuntimeError("MIST bolometric-correction archives have incompatible interpolation coordinates")

    interpolators: Dict[str, Tuple[RegularGridInterpolator, int]] = {}
    for artifact in table_artifacts:
        path = str(artifact["path"])
        selected_columns = columns_by_path[path]
        if not selected_columns:
            continue
        metadata = metadata_by_path[path]
        axes = metadata["axes"]
        shape = tuple(int(axis.size) for axis in axes)
        values = np.empty((*shape, len(selected_columns)), dtype=float)
        assigned = np.zeros(shape, dtype=bool)
        axis_indexes = [
            {float(value): index for index, value in enumerate(axis)}
            for axis in axes
        ]
        for indexes, row in _iter_mist_bc_archive_rows(
            workspace.path / path, str(artifact["table_kind"])
        ):
            coordinate = tuple(row[indexes[name]] for name in MIST_BC_PARAMETER_COLUMNS)
            try:
                location = tuple(
                    axis_indexes[index][float(value)]
                    for index, value in enumerate(coordinate)
                )
            except KeyError as exc:
                raise RuntimeError("MIST bolometric-correction archive changed while being read") from exc
            if assigned[location]:
                raise RuntimeError("MIST bolometric-correction archive has duplicate coordinate cells")
            assigned[location] = True
            values[location] = [row[indexes[column]] for column in selected_columns]
        if not np.all(assigned) or not np.all(np.isfinite(values)):
            raise RuntimeError("MIST bolometric-correction archive has missing or non-finite grid cells")
        interpolator = RegularGridInterpolator(
            axes,
            values,
            method="linear",
            bounds_error=True,
        )
        for index, column in enumerate(selected_columns):
            interpolators[column] = (interpolator, index)
    return interpolators, common_axes[-1]


def _fit_mist_bc_sed(
    observations: Sequence[Mapping[str, Any]],
    atmosphere: Tuple[float, float, float],
    interpolators: Mapping[str, Tuple[RegularGridInterpolator, int]],
    av_axis: np.ndarray,
) -> Dict[str, Any]:
    """Profile apparent bolometric magnitude and minimize chi square over native A_V intervals."""
    observed = np.asarray([float(row["magnitude"]) for row in observations], dtype=float)
    uncertainties = np.asarray([float(row["uncertainty_mag"]) for row in observations], dtype=float)
    columns = [str(row["mist_column"]) for row in observations]
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(uncertainties)) or np.any(uncertainties <= 0.0):
        raise RuntimeError("MIST v1.2 SED measurements must be finite with positive uncertainties")
    weights = uncertainties**-2
    if not math.isfinite(float(np.sum(weights))) or np.sum(weights) <= 0.0:
        raise RuntimeError("MIST v1.2 SED photometric uncertainties do not define finite weights")

    def profile(av_mag: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
        coordinate = np.asarray([(*atmosphere, av_mag)], dtype=float)
        bc_values = []
        try:
            for column in columns:
                interpolator, index = interpolators[column]
                value = np.asarray(interpolator(coordinate), dtype=float).reshape(-1)[index]
                bc_values.append(float(value))
        except (KeyError, ValueError) as exc:
            raise RuntimeError("MIST v1.2 SED atmospheric parameters fall outside the supplied table grid") from exc
        bolometric_estimates = observed + np.asarray(bc_values, dtype=float)
        apparent_bolometric_magnitude = float(
            np.sum(weights * bolometric_estimates) / np.sum(weights)
        )
        model = apparent_bolometric_magnitude - np.asarray(bc_values, dtype=float)
        residuals = observed - model
        chi_square = float(np.sum((residuals / uncertainties) ** 2))
        if not math.isfinite(chi_square) or not math.isfinite(apparent_bolometric_magnitude):
            raise RuntimeError("MIST v1.2 SED profile produced a non-finite fit")
        return chi_square, apparent_bolometric_magnitude, np.asarray(bc_values, dtype=float), model

    if av_axis.size <= 1 or np.any(np.diff(av_axis) <= 0.0):
        raise RuntimeError("MIST v1.2 bolometric-correction table must provide increasing A_V values")
    candidates: List[Tuple[float, float, np.ndarray, np.ndarray, float]] = []
    for lower, upper in zip(av_axis[:-1], av_axis[1:]):
        for av_mag in (float(lower), float(upper)):
            chi_square, apparent_bolometric_magnitude, bc_values, model = profile(av_mag)
            candidates.append((chi_square, apparent_bolometric_magnitude, bc_values, model, av_mag))
        optimum = minimize_scalar(
            lambda av_mag: profile(float(av_mag))[0],
            bounds=(float(lower), float(upper)),
            method="bounded",
        )
        if optimum.success and math.isfinite(float(optimum.fun)):
            chi_square, apparent_bolometric_magnitude, bc_values, model = profile(float(optimum.x))
            candidates.append(
                (chi_square, apparent_bolometric_magnitude, bc_values, model, float(optimum.x))
            )
    if not candidates:
        raise RuntimeError("MIST v1.2 SED could not evaluate a native A_V interval")
    chi_square, apparent_bolometric_magnitude, bc_values, model, av_mag = min(
        candidates, key=lambda candidate: candidate[0]
    )
    residuals = observed - model
    return {
        "fit": {
            "av_mag": av_mag,
            "apparent_bolometric_mag": apparent_bolometric_magnitude,
            "chi_square": chi_square,
            "degrees_of_freedom": len(observations) - MIST_BC_FIT_PARAMETER_COUNT,
        },
        "photometry": [
            {
                "catalog": str(row["catalog"]),
                "catalog_band": str(row["catalog_band"]),
                "mist_column": str(row["mist_column"]),
                "observed_mag": float(observed[index]),
                "uncertainty_mag": float(uncertainties[index]),
                "bc_mag_at_fit": float(bc_values[index]),
                "model_mag": float(model[index]),
                "residual_mag": float(residuals[index]),
            }
            for index, row in enumerate(observations)
        ],
    }


def run_sed_fit(workspace: CandidateWorkspace) -> Path:
    """Run the candidate-local MIST v1.2 bolometric-correction diagnostic.

    The runner accepts only a schema-valid manifest that binds the candidate's
    Vega photometry, fixed atmospheric parameters, and official MIST v1.2
    bolometric-correction archives. It never substitutes a Planck spectrum,
    generic atmosphere grid, nearest table node, or unbound filter curve.

    Args:
        workspace (CandidateWorkspace): Workspace that owns broadband
            photometry, stellar parameters, MIST archives, and outputs.

    Returns:
        Path: Candidate-local SED result artifact.

    Raises:
        RuntimeError: Candidate-owned photometry, atmospheric parameters, or
            hash-bound MIST v1.2 bolometric-correction archives are unavailable
            or incompatible.

    Note:
        The fit is an exploratory color diagnostic. It does not provide an
        atmosphere posterior, radius, luminosity, distance, validation
        constraint, or lifecycle decision.
    """
    observations, atmosphere, declared_rv, input_artifacts, manifest_artifact = _read_mist_bc_sed_inputs(
        workspace
    )
    interpolators, av_axis = _load_mist_bc_interpolators(
        workspace,
        [artifact for artifact in input_artifacts if "table_kind" in artifact],
        [str(observation["mist_column"]) for observation in observations],
        declared_rv,
    )
    fit = _fit_mist_bc_sed(observations, atmosphere, interpolators, av_axis)

    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "sed_fit_results.json"
    payload = {
        "schema_version": 2,
        "candidate_id": workspace.candidate_id,
        "work_package": "SED_FIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "candidate-data",
        "scientific_status": "exploratory-mist-v1.2-bolometric-correction-diagnostic",
        "validation_eligible": False,
        "claim_eligible": False,
        "method": MIST_BC_METHOD,
        "input_provenance": {
            "input_manifest_artifact": manifest_artifact,
            "input_artifacts": [
                {key: artifact[key] for key in ("path", "sha256", "role")}
                for artifact in input_artifacts
            ],
        },
        "table_metadata": {
            "release": "MIST v1.2",
            "bc_table_release": "BC_tables/v1",
            "rv": declared_rv,
            "magnitude_system": "Vega",
            "interpolation": "multilinear-rectilinear-no-extrapolation",
        },
        "fixed_atmosphere": {
            "teff_k": atmosphere[0],
            "logg_cgs": atmosphere[1],
            "feh_dex": atmosphere[2],
        },
        **fit,
        "caveats": [
            "MIST atmospheric parameters are fixed from the hash-bound candidate stellar-parameters artifact.",
            "The apparent bolometric magnitude is conditional on the MIST bolometric convention and is not converted to radius, luminosity, or distance.",
            "Chi square uses declared diagonal catalog uncertainties and excludes catalog covariance, blending, saturation, infrared excess, and MIST model uncertainty.",
            "This candidate-owned SED result is exploratory diagnostic evidence and is not a validation claim.",
        ],
    }
    _write_json_atomic(output_path, payload)
    return output_path


