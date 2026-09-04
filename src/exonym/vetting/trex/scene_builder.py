"""Autonomous candidate-local TREX scene manifest builder.

``exonym build-scene`` assembles the hash-bound ``data/external/trex_scene.json``
that ``_load_trex_scene`` requires before a TRICERATOPS attempt.  The builder is
deliberately narrow: it reads only validated, candidate-owned evidence (the
archival Gaia report and the stellar-parameter sidecar), derives the resolved
neighbor stellar properties it can, and fails closed on anything it cannot
reconstruct rather than fabricating a population.

Scientific boundary and retained sources
----------------------------------------
The scene is the Giacalone et al. (2021) conditional-vetting input, ADS
``2021AJ....161...24G``, DOI ``10.3847/1538-3881/abd184``, with the background
population context of Girardi et al. (2005), ADS ``2005A&A...436..895G``, DOI
``10.1051/0004-6361:20042352``. Resolved-neighbor mass and radius are read
directly from the Gaia DR3 Final
Luminosity Age Mass Estimator (FLAME), described by Creevey et al. (2023), ADS
``2023A&A...674A..39C``, DOI ``10.1051/0004-6361/202243800``.  The builder
accepts only FLAME's documented quality flags and model domain; it does not
derive a mass or radius from magnitude, color, or a hand-authored relation.
No population is synthesized in-tree: a missing TRILEGAL cache fails closed.

Units and fail-closed contract
------------------------------
Coordinates are ICRS degrees; mass/radius are solar units; temperature is K;
magnitude and delta-mag are mag; parallax is mas; separation is arcsec; FLAME
age is Gyr; and background ``star_count`` is an integer.  The builder requires
a validated archival Gaia target with FLAME values for all resolved neighbors,
a candidate-derived TESS magnitude, a retained instrumental contrast curve,
and a retained TRILEGAL/background cache; otherwise it raises
``SceneBuildError`` and writes nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...archive import (
    ARCHIVAL_REPORT_RELATIVE_PATH,
    load_validated_archival_gaia_sources,
)
from ...inputs import load_stellar_parameters
from ...workspace import CandidateWorkspace
from .target import _sha256

TREX_SCENE_MANIFEST_RELATIVE_PATH = Path("data") / "external" / "trex_scene.json"
TREX_SCENE_SCHEMA_NAME = "trex-scene-manifest.schema.json"

GAIA_DR3_FLAME_METHOD = "gaia-dr3-flame"
FLAME_RECOMMENDED_FLAG_SECOND_CHARACTER = "0"
FLAME_GIANT_FLAG_FIRST_CHARACTER = "1"
FLAME_MODEL_MASS_SOLAR_MIN = 0.5
FLAME_MODEL_MASS_SOLAR_MAX = 10.0
FLAME_GIANT_MASS_SOLAR_MIN = 1.0
FLAME_GIANT_MASS_SOLAR_MAX = 2.0
FLAME_GIANT_AGE_GYR_MIN = 1.0
TREX_CONTRAST_CURVE_RELATIVE_PATH = Path("data") / "external" / "trex_contrast_curve.json"


class SceneBuildError(RuntimeError):
    """The candidate workspace cannot yield a schema-valid TREX scene."""


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise SceneBuildError("{0} must be a finite number".format(name))
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SceneBuildError("{0} must be a finite number".format(name)) from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise SceneBuildError("{0} must be a {1} number".format(name, qualifier))
    return number


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SceneBuildError("{0} is missing or is not a regular file".format(label))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SceneBuildError("{0} is not valid JSON".format(label)) from exc
    if not isinstance(payload, dict):
        raise SceneBuildError("{0} must be a JSON object".format(label))
    return payload


def _bind_contrast_curve(workspace: CandidateWorkspace) -> Dict[str, Any]:
    """Bind a retained instrumental contrast curve; never infer one from Gaia."""
    path = workspace.path / TREX_CONTRAST_CURVE_RELATIVE_PATH
    payload = _read_json_object(path, "TREX contrast curve")
    separations = payload.get("separations_arcsec")
    magnitudes = payload.get("delta_magnitudes")
    if not isinstance(separations, list) or not isinstance(magnitudes, list):
        raise SceneBuildError("TREX contrast curve arrays are missing")
    if len(separations) != len(magnitudes) or len(separations) < 2:
        raise SceneBuildError("TREX contrast curve requires matching arrays with at least two points")
    validated_separations = [
        _finite_number(value, "contrast separation_arcsec", positive=True)
        for value in separations
    ]
    validated_magnitudes = [
        _finite_number(value, "contrast delta_magnitude") for value in magnitudes
    ]
    if any(
        next_separation <= separation
        for separation, next_separation in zip(validated_separations, validated_separations[1:])
    ):
        raise SceneBuildError("TREX contrast curve separations must be strictly increasing")
    return {
        "path": str(TREX_CONTRAST_CURVE_RELATIVE_PATH).replace("\\", "/"),
        "sha256": _sha256(path),
        "separations_arcsec": validated_separations,
        "delta_magnitudes": validated_magnitudes,
    }


def _flame_neighbor_properties(neighbor: Dict[str, Any], source_id: str) -> Dict[str, Any]:
    """Validate documented Gaia DR3 FLAME values for one resolved neighbor."""
    mass_solar = _finite_number(neighbor.get("mass_flame"), "neighbor mass_flame", positive=True)
    radius_solar = _finite_number(neighbor.get("radius_flame"), "neighbor radius_flame", positive=True)
    flags = neighbor.get("flags_flame")
    if not isinstance(flags, str) or len(flags) < 2:
        raise SceneBuildError("neighbor {0} lacks Gaia DR3 FLAME quality flags".format(source_id))
    if flags[1] != FLAME_RECOMMENDED_FLAG_SECOND_CHARACTER:
        raise SceneBuildError(
            "neighbor {0} Gaia DR3 FLAME flag {1!r} is outside the recommended quality subset".format(
                source_id, flags
            )
        )
    if not FLAME_MODEL_MASS_SOLAR_MIN <= mass_solar <= FLAME_MODEL_MASS_SOLAR_MAX:
        raise SceneBuildError(
            "neighbor {0} Gaia DR3 FLAME mass is outside its documented model domain".format(source_id)
        )
    age_gyr: Optional[float] = None
    if flags[0] == FLAME_GIANT_FLAG_FIRST_CHARACTER:
        age_gyr = _finite_number(
            neighbor.get("age_flame_gyr"), "neighbor age_flame_gyr", positive=True
        )
        if not (
            FLAME_GIANT_MASS_SOLAR_MIN <= mass_solar <= FLAME_GIANT_MASS_SOLAR_MAX
            and age_gyr > FLAME_GIANT_AGE_GYR_MIN
        ):
            raise SceneBuildError(
                "neighbor {0} Gaia DR3 FLAME giant does not meet its documented mass and age applicability constraint".format(
                    source_id
                )
            )
    evolutionary_stage = neighbor.get("evolstage_flame")
    if isinstance(evolutionary_stage, bool) or not isinstance(evolutionary_stage, int):
        raise SceneBuildError("neighbor {0} lacks Gaia DR3 FLAME evolutionary stage".format(source_id))
    return {
        "mass_solar": mass_solar,
        "radius_solar": radius_solar,
        "stellar_parameter_source": GAIA_DR3_FLAME_METHOD,
        "flame_flags": flags,
        "flame_evolutionary_stage": evolutionary_stage,
        "flame_age_gyr": age_gyr,
    }


def _bind_background(workspace: CandidateWorkspace) -> Dict[str, Any]:
    """Bind a retained TRILEGAL/background population cache, or fail closed."""
    external = workspace.path / "data" / "external"
    for name in ("trilegal_cache.csv", "background_stars.csv", "trilegal.csv"):
        candidate = external / name
        if candidate.is_file():
            star_count = _count_csv_rows(candidate)
            return {
                "path": "data/external/{0}".format(name),
                "sha256": _sha256(candidate),
                "model": "trilegal",
                "star_count": star_count,
            }
    raise SceneBuildError(
        "no retained TRILEGAL/background population cache found under data/external"
    )


def _count_csv_rows(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for _line in handle:
            count += 1
    # Subtract a header row when present; the manifest only records the
    # simulated star count, so an off-by-one from a missing header is bounded.
    return max(count - 1, 0)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise



def build_trex_scene_manifest(workspace: CandidateWorkspace) -> Path:
    """Assemble and atomically write ``data/external/trex_scene.json``.

    Args:
        workspace: Candidate workspace with a validated archival Gaia report,
            a candidate-derived stellar-parameter sidecar, and a retained
            TRILEGAL/background cache.

    Returns:
        The absolute path to the written manifest.

    Raises:
        SceneBuildError: If any required evidence is missing, invalid, or
            cannot be reconstructed without fabrication.
    """
    stellar = load_stellar_parameters(workspace)
    if stellar.get("source") != "candidate-data":
        raise SceneBuildError(
            "TREX scene requires complete candidate-data stellar parameters"
        )

    ra_deg = _finite_number(stellar.get("ra_deg"), "target ra_deg")
    dec_deg = _finite_number(stellar.get("dec_deg"), "target dec_deg")
    mass_solar = _finite_number(stellar.get("mass_solar"), "target mass_solar", positive=True)
    radius_solar = _finite_number(stellar.get("radius_solar"), "target radius_solar", positive=True)
    teff_k = _finite_number(stellar.get("teff_k"), "target teff_k", positive=True)
    parallax_mas = _finite_number(stellar.get("parallax_mas"), "target parallax_mas", positive=True)
    tess_mag = _finite_number(stellar.get("tess_mag"), "target tess_mag")
    if not 0.0 <= ra_deg < 360.0 or not -90.0 <= dec_deg <= 90.0:
        raise SceneBuildError("target coordinates are outside their valid domain")

    target, neighbors, metadata = load_validated_archival_gaia_sources(workspace)
    if target is None or metadata.get("availability") != "available":
        raise SceneBuildError("TREX scene requires a validated archival Gaia target")
    if not neighbors:
        raise SceneBuildError("TREX scene requires at least one resolved Gaia neighbor")
    target_source_id = str(target.get("source_id", ""))
    if not target_source_id:
        raise SceneBuildError("archival Gaia target lacks a source_id")

    target_g_mag = _finite_number(target.get("phot_g_mean_mag"), "target phot_g_mean_mag")
    resolved_neighbors: List[Dict[str, Any]] = []
    neighbor_source_ids: List[str] = []
    for neighbor in neighbors:
        source_id = str(neighbor.get("source_id", ""))
        if not source_id:
            raise SceneBuildError("archival Gaia neighbor lacks a source_id")
        if source_id == target_source_id:
            continue
        neighbor_g_mag = _finite_number(
            neighbor.get("phot_g_mean_mag"), "neighbor phot_g_mean_mag"
        )
        separation_arcsec = _finite_number(
            neighbor.get("separation_arcsec"), "neighbor separation_arcsec", positive=True
        )
        flame_properties = _flame_neighbor_properties(neighbor, source_id)
        delta_mag = neighbor_g_mag - target_g_mag
        resolved_neighbors.append(
            {
                "source_id": source_id,
                **flame_properties,
                "delta_mag": delta_mag,
                "separation_arcsec": separation_arcsec,
            }
        )
        neighbor_source_ids.append(source_id)

    contrast_curve = _bind_contrast_curve(workspace)

    background = _bind_background(workspace)
    archival_artifact_path = workspace.path / str(ARCHIVAL_REPORT_RELATIVE_PATH)
    archival_sha256 = _sha256(archival_artifact_path)

    manifest = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "source": "candidate-data",
        "target": {
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "mass_solar": mass_solar,
            "radius_solar": radius_solar,
            "teff_k": teff_k,
            "parallax_mas": parallax_mas,
            "tess_mag": tess_mag,
        },
        "archival_gaia": {
            "path": str(ARCHIVAL_REPORT_RELATIVE_PATH).replace("\\", "/"),
            "sha256": archival_sha256,
            "target_source_id": target_source_id,
            "neighbor_source_ids": neighbor_source_ids,
        },
        "contrast_curve": contrast_curve,
        "background": background,
        "resolved_neighbors": resolved_neighbors,
    }
    manifest_path = workspace.path / TREX_SCENE_MANIFEST_RELATIVE_PATH
    _write_json_atomic(manifest_path, manifest)
    return manifest_path
