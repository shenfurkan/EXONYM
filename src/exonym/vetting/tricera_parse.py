"""Interface to candidate-local TRICERATOPS statistical-vetting reports.

The parser validates a retained report, extracts finite model outputs, and
applies preregistered routing criteria to the recorded false-positive terms.
It preserves runtime and observed-input provenance so an unavailable or
fallback execution cannot be mistaken for a completed Monte Carlo result.

Scientific boundary:
    A finite false-positive probability remains evidence from its recorded
    assumptions. Claim creation is disabled until provenance-bound observed
    photometry and calibrated scene constraints are integrated.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from ..inputs import (
    EPHEMERIS_CONFIG_NAMES,
    _parse_finite_float,
    _reject_duplicate_json_keys,
    _reject_nonfinite_json_constant,
)
from ..workspace import validate_signal_suffix

# SCIENTIFIC_BOUNDARY: This threshold is retained for transparent routing; it
# does not enable a claim while the calibrated scene-model prerequisite is open.
FPP_THRESHOLD = 0.01
# SCIENTIFIC_BOUNDARY: A report-aware routing pass requires both retained
# TRICERATOPS false-positive terms; an FPP-only scalar is legacy utility input.
NFPP_THRESHOLD = 0.01
FPP_CLAIM_BLOCK_REASON = (
    "FPP claim creation is disabled until TRICERATOPS receives provenance-bound "
    "observed photometry and scene constraints."
)
DEFAULT_TRICERATOPS_SEED = 1729
TREX_SCENE_MANIFEST_RELATIVE_PATH = Path("data") / "external" / "trex_scene.json"


class TrexSceneUnavailableError(RuntimeError):
    """Required candidate-owned TREX scene evidence is unavailable or invalid."""




def _observed_sectors(workspace: Any) -> list:
    """Return the sorted list of TESS sectors observed for this workspace.

    Sector numbers are read from the workspace data (``data/raw`` product
    filenames first, ``data/external/tess_holdings.json`` as fallback) so the
    library stays target-neutral.
    """
    sectors: set = set()
    raw = workspace.path / "data" / "raw"
    if raw.is_dir():
        for path in sorted(raw.rglob("*")):
            if path.is_file() and path.name.startswith("s") and path.suffix.lower() in (".fits", ".fz"):
                stem = path.name[1:5]
                if stem.isdigit():
                    sectors.add(int(stem))
    if not sectors:
        holdings = workspace.path / "data" / "external" / "tess_holdings.json"
        try:
            payload = json.loads(holdings.read_text(encoding="utf-8"))
            for pipeline in payload.get("pipelines", {}).values():
                for entry in pipeline:
                    sector = entry.get("sector")
                    if isinstance(sector, int):
                        sectors.add(sector)
        except Exception:
            pass
    return sorted(sectors)


def load_fpp_report(path: Path) -> Dict[str, Any]:
    """Load one strict, finite TRICERATOPS JSON object."""
    try:
        data = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_float=_parse_finite_float,
            parse_constant=_reject_nonfinite_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("TRICERATOPS report must be strict finite JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("TRICERATOPS report must be a JSON object")
    return data


def _probability_value(report: Dict[str, Any], keys: Tuple[str, ...], label: str) -> float:
    for key in keys:
        value = report.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("{0} must be a JSON number".format(label))
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("{0} must be a finite probability in [0, 1]".format(label))
        return probability
    raise ValueError("no {0} value found in report".format(label))


def extract_fpp(report: Dict[str, Any]) -> float:
    """Return the FPP value from a report, probing common key layouts."""
    return _probability_value(
        report,
        ("fpp", "FPP", "fpp_value", "fpp_specific", "FPP_specific", "fpp_specific_value"),
        "FPP",
    )


def extract_nfpp(report: Dict[str, Any]) -> float:
    """Return the nearby-false-positive probability from a retained report.

    The report-aware gate requires this value alongside FPP. A missing,
    non-finite, or out-of-range value is not evidence that the nearby-source
    contribution is small and therefore cannot pass the joint screen.
    """
    return _probability_value(report, ("nfpp", "NFPP", "nfpp_value"), "NFPP")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for a candidate-local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()





def _notify_progress(
    progress_callback: Optional[Callable[[str, Optional[int], Optional[int]], None]],
    step: str,
    done: Optional[int] = None,
    total: Optional[int] = None,
) -> None:
    """Report a truthful backend milestone without affecting scientific work."""
    if progress_callback is None:
        return
    try:
        progress_callback(step, done, total)
    except Exception:
        # Presentation callbacks must never change a vetting outcome.
        pass



def _artifact_from_path(workspace: Any, path: Path) -> Dict[str, str]:
    """Hash one regular candidate-local artifact for an execution snapshot."""
    candidate_root = workspace.path.resolve()
    try:
        resolved = path.resolve()
        relative = resolved.relative_to(candidate_root)
    except (OSError, ValueError) as exc:
        raise ValueError("TRICERATOPS artifact must remain inside the candidate workspace") from exc
    if path.is_symlink() or not resolved.is_file():
        raise ValueError("TRICERATOPS artifact must be an available regular file")
    return {"path": relative.as_posix(), "sha256": _sha256(resolved)}


def _ephemeris_artifacts(workspace: Any, signal: Optional[str], source: object) -> list:
    """Bind every candidate config that could have supplied the ephemeris."""
    source_text = str(source)
    paths = []
    if "signal" in source_text and signal is not None:
        paths.append(workspace.path / "config" / "signals" / "transit_config{0}.json".format(signal))
    elif "candidate-config" in source_text:
        paths.extend(workspace.path / "config" / name for name in EPHEMERIS_CONFIG_NAMES)
    return [_artifact_from_path(workspace, path) for path in paths if path.is_file()]


def _snapshot_artifacts(workspace: Any, artifacts: Iterable[object]) -> list:
    """Normalize and rehash a complete, candidate-local execution input set."""
    snapshot = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError("TRICERATOPS input provenance contains a malformed artifact")
        record = _artifact_from_path(workspace, workspace.path / artifact["path"])
        expected = artifact.get("sha256")
        if expected is not None and expected != record["sha256"]:
            raise ValueError("TRICERATOPS input changed before execution: {0}".format(record["path"]))
        snapshot[record["path"]] = record
    return [snapshot[path] for path in sorted(snapshot)]


def _verify_snapshot(workspace: Any, artifacts: Iterable[object]) -> None:
    """Fail closed when an input changes during an optional-engine run."""
    _snapshot_artifacts(workspace, artifacts)


def _scene_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TrexSceneUnavailableError("TREX scene {0} must be a finite number".format(name))
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrexSceneUnavailableError("TREX scene {0} must be a finite number".format(name)) from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise TrexSceneUnavailableError(
            "TREX scene {0} must be a {1} number".format(name, qualifier)
        )
    return number


def _scene_artifact(workspace: Any, record: object, label: str) -> Dict[str, str]:
    """Validate one hash-bound candidate artifact referenced by a scene manifest."""
    if not isinstance(record, dict):
        raise TrexSceneUnavailableError("TREX scene {0} artifact is missing".format(label))
    path_value = record.get("path")
    digest = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise TrexSceneUnavailableError("TREX scene {0} artifact is malformed".format(label))
    try:
        artifact = _artifact_from_path(workspace, workspace.path / path_value)
    except (OSError, ValueError) as exc:
        raise TrexSceneUnavailableError(
            "TREX scene {0} artifact is unavailable".format(label)
        ) from exc
    if artifact["sha256"] != digest:
        raise TrexSceneUnavailableError(
            "TREX scene {0} artifact does not match its recorded hash".format(label)
        )
    return artifact


def _load_trex_scene(workspace: Any, tic_id: int, sectors: list) -> Tuple[Any, list]:
    """Build a fully evidenced TREX scene from candidate-owned immutable inputs.

    ``data/external/trex_scene.json`` binds explicit stellar values, every Gaia
    neighbor, a contrast curve, and a retained TRILEGAL/background population.
    Values are deliberately not inferred from incomplete catalog records: an
    absent field makes the optional FPP execution unresolved.
    """
    from ..archive import load_validated_archival_gaia_sources, load_validated_archival_report
    from .trex import TargetScene

    manifest_path = workspace.path / TREX_SCENE_MANIFEST_RELATIVE_PATH
    try:
        manifest = load_fpp_report(manifest_path)
    except ValueError as exc:
        raise TrexSceneUnavailableError(
            "TREX scene manifest is unavailable or invalid: {0}".format(exc)
        ) from exc
    if manifest.get("schema_version") != 1 or manifest.get("candidate_id") != workspace.candidate_id:
        raise TrexSceneUnavailableError("TREX scene manifest does not match this candidate")
    if manifest.get("source") != "candidate-data":
        raise TrexSceneUnavailableError("TREX scene manifest is not candidate-derived evidence")

    target = manifest.get("target")
    if not isinstance(target, dict):
        raise TrexSceneUnavailableError("TREX scene manifest lacks target parameters")
    target_values = {
        "ra_deg": _scene_number(target.get("ra_deg"), "target.ra_deg"),
        "dec_deg": _scene_number(target.get("dec_deg"), "target.dec_deg"),
        "M_s_Msun": _scene_number(target.get("mass_solar"), "target.mass_solar", positive=True),
        "R_s_Rsun": _scene_number(target.get("radius_solar"), "target.radius_solar", positive=True),
        "Teff_K": _scene_number(target.get("teff_k"), "target.teff_k", positive=True),
        "Tmag": _scene_number(target.get("tess_mag"), "target.tess_mag"),
        "plx_mas": _scene_number(target.get("parallax_mas"), "target.parallax_mas", positive=True),
    }

    archival_record = manifest.get("archival_gaia")
    archival_artifact = _scene_artifact(workspace, archival_record, "archival Gaia")
    if archival_artifact["path"] != "outputs/archival_vetting_report.json":
        raise TrexSceneUnavailableError("TREX scene must bind the archival Gaia report")
    archival_report = load_validated_archival_report(workspace)
    gaia = archival_report.get("gaia_astrometry") if isinstance(archival_report, dict) else None
    archival_target, archival_neighbors, _ = load_validated_archival_gaia_sources(workspace)
    if not isinstance(gaia, dict) or not isinstance(archival_target, dict):
        raise TrexSceneUnavailableError("TREX requires a validated archival Gaia target context")
    sources = gaia.get("sources")
    if (
        not isinstance(sources, list)
        or gaia.get("nearby_sources_count") != len(sources)
        or any(not isinstance(source, dict) for source in sources)
    ):
        raise TrexSceneUnavailableError("TREX archival Gaia source list is incomplete")
    target_source_id = str(archival_target.get("source_id", ""))
    neighbor_source_ids = [str(neighbor.get("source_id", "")) for neighbor in archival_neighbors]
    if not target_source_id or any(not source_id for source_id in neighbor_source_ids):
        raise TrexSceneUnavailableError("TREX archival Gaia sources require identifiers")
    if len(set([target_source_id] + neighbor_source_ids)) != len(sources):
        raise TrexSceneUnavailableError("TREX archival Gaia source identifiers are ambiguous")
    if not isinstance(archival_record, dict) or archival_record.get("target_source_id") != target_source_id:
        raise TrexSceneUnavailableError("TREX scene Gaia target does not match archival evidence")
    if archival_record.get("neighbor_source_ids") != neighbor_source_ids:
        raise TrexSceneUnavailableError("TREX scene does not retain every archival Gaia neighbor")

    # Archival coordinates are rounded to six decimals when the report is written.
    coordinates = archival_report.get("target_coordinates") if isinstance(archival_report, dict) else None
    if not isinstance(coordinates, dict):
        raise TrexSceneUnavailableError("TREX archival Gaia context lacks target coordinates")
    for key in ("ra_deg", "dec_deg"):
        archived = _scene_number(coordinates.get(key), "archival target.{0}".format(key))
        if not math.isclose(target_values[key], archived, rel_tol=0.0, abs_tol=0.5e-6):
            raise TrexSceneUnavailableError("TREX scene target coordinates do not match archival evidence")

    contrast = manifest.get("contrast_curve")
    contrast_artifact = _scene_artifact(workspace, contrast, "contrast curve")
    if not isinstance(contrast, dict):
        raise TrexSceneUnavailableError("TREX scene contrast curve is missing")
    separations = contrast.get("separations_arcsec")
    values = contrast.get("delta_magnitudes")
    if not isinstance(separations, list) or not isinstance(values, list):
        raise TrexSceneUnavailableError("TREX scene contrast curve arrays are missing")

    background = manifest.get("background")
    background_artifact = _scene_artifact(workspace, background, "TRILEGAL/background")
    if not isinstance(background, dict) or background.get("model") not in ("trilegal", "background"):
        raise TrexSceneUnavailableError("TREX requires a declared TRILEGAL/background population")
    star_count = background.get("star_count")
    if isinstance(star_count, bool) or not isinstance(star_count, int) or star_count < 0:
        raise TrexSceneUnavailableError("TREX background star_count must be a non-negative integer")

    manifest_neighbors = manifest.get("resolved_neighbors")
    if not isinstance(manifest_neighbors, list) or len(manifest_neighbors) != len(neighbor_source_ids):
        raise TrexSceneUnavailableError("TREX scene does not include every archival Gaia neighbor")
    by_source_id = {}
    for neighbor in manifest_neighbors:
        if not isinstance(neighbor, dict) or not isinstance(neighbor.get("source_id"), str):
            raise TrexSceneUnavailableError("TREX scene has a malformed Gaia neighbor")
        source_id = neighbor["source_id"]
        if source_id in by_source_id:
            raise TrexSceneUnavailableError("TREX scene repeats an archival Gaia neighbor")
        by_source_id[source_id] = neighbor
    if set(by_source_id) != set(neighbor_source_ids):
        raise TrexSceneUnavailableError("TREX scene Gaia neighbors do not match archival evidence")
    archival_neighbors_by_id = {
        str(neighbor["source_id"]): neighbor for neighbor in archival_neighbors
    }
    resolved_neighbors = []
    for source_id in neighbor_source_ids:
        manifest_neighbor = by_source_id[source_id]
        separation = _scene_number(
            manifest_neighbor.get("separation_arcsec"),
            "neighbor.separation_arcsec",
            positive=True,
        )
        archived_separation = _scene_number(
            archival_neighbors_by_id[source_id].get("separation_arcsec"),
            "archival neighbor.separation_arcsec",
            positive=True,
        )
        if not math.isclose(separation, archived_separation, rel_tol=0.0, abs_tol=0.5e-6):
            raise TrexSceneUnavailableError(
                "TREX scene neighbor separation does not match archival Gaia evidence"
            )
        resolved_neighbors.append(
            {
                "source_id": source_id,
                "M_s": _scene_number(
                    manifest_neighbor.get("mass_solar"), "neighbor.mass_solar", positive=True
                ),
                "R_s": _scene_number(
                    manifest_neighbor.get("radius_solar"), "neighbor.radius_solar", positive=True
                ),
                "delta_mag": _scene_number(
                    manifest_neighbor.get("delta_mag"), "neighbor.delta_mag"
                ),
                "separation_arcsec": separation,
            }
        )
    try:
        scene = TargetScene(
            tic_id=tic_id,
            sectors=sectors,
            contrast_separations=separations,
            contrast_values=values,
            resolved_neighbors=resolved_neighbors,
            N_background=star_count,
            trilegal_cache=workspace.path / background_artifact["path"],
            background_sha256=background_artifact["sha256"],
            **target_values,
        )
    except (TypeError, ValueError) as exc:
        raise TrexSceneUnavailableError("TREX scene parameters are invalid: {0}".format(exc)) from exc
    manifest_artifact = _artifact_from_path(workspace, manifest_path)
    return scene, [manifest_artifact, archival_artifact, contrast_artifact, background_artifact]


def _prepare_observed_transit_input(workspace: Any, signal: Optional[str]) -> Dict[str, Any]:
    """Return provenance-bound, phase-folded observed photometry for TRICERATOPS.

    The returned ``time_days`` values are measured from the nearest declared
    transit midpoint. Flux values are inverse-variance binned across phase,
    using a maximum of one hundred bins.  The central transit receives five
    equal-width bins, so its resolution is at most ``duration_days / 5`` even
    for a low-duty-cycle signal. The backend accepts only one flux uncertainty,
    so this function records and passes the mean uncertainty of the observed
    phase bins.
    """
    import numpy as np

    from ..inputs import BTJD_TIME_SYSTEM, load_light_curve_table, load_transit_ephemeris

    table = load_light_curve_table(workspace, max_points=None, require_raw_provenance=True)
    if table is None:
        raise ValueError("TRICERATOPS requires a readable candidate light curve")

    ephemeris = load_transit_ephemeris(workspace, signal=signal)
    field_sources = ephemeris.get("field_sources", {})
    required_fields = ("period_days", "epoch_btjd", "duration_days")
    if not isinstance(field_sources, dict) or any(
        field_sources.get(field) in (None, "synthetic-demo") for field in required_fields
    ):
        raise ValueError(
            "TRICERATOPS requires candidate-derived period, epoch, and duration values"
        )
    if ephemeris.get("time_system") != BTJD_TIME_SYSTEM:
        raise ValueError("TRICERATOPS requires a BTJD_TDB ephemeris")

    try:
        period_days = float(ephemeris["period_days"])
        epoch_btjd = float(ephemeris["epoch_btjd"])
        duration_days = float(ephemeris["duration_days"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("TRICERATOPS requires finite period, epoch, and duration values") from exc
    if (
        not math.isfinite(period_days)
        or not math.isfinite(epoch_btjd)
        or not math.isfinite(duration_days)
        or period_days <= 0.0
        or duration_days <= 0.0
        or duration_days >= 0.5 * period_days
    ):
        raise ValueError("TRICERATOPS requires a positive duration shorter than half the orbital period")

    flux_err_sources = table.get("flux_err_sources")
    if flux_err_sources != ["reported"]:
        raise ValueError("TRICERATOPS requires reported per-cadence flux uncertainties")

    time_btjd = np.asarray(table.get("time"), dtype=float)
    flux = np.asarray(table.get("flux"), dtype=float)
    flux_err = np.asarray(table.get("flux_err"), dtype=float)
    sectors = np.asarray(table.get("sector"), dtype=int)
    valid = np.isfinite(time_btjd) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0.0)
    time_btjd = time_btjd[valid]
    flux = flux[valid]
    flux_err = flux_err[valid]
    sectors = sectors[valid]
    if time_btjd.size < 50:
        raise ValueError("TRICERATOPS requires at least fifty finite observed cadences")

    time_days = np.remainder(time_btjd - epoch_btjd + 0.5 * period_days, period_days) - 0.5 * period_days
    max_bin_count = min(100, time_days.size)
    transit_bin_count = 5
    local_baseline_bin_count = 3
    broad_baseline_bin_count = (
        max_bin_count - transit_bin_count - 2 * local_baseline_bin_count
    ) // 2
    if broad_baseline_bin_count < 3:
        raise ValueError("TRICERATOPS requires sufficient phase coverage for transit and baseline bins")
    local_baseline_limit = min(3.0 * duration_days, 0.25 * period_days)
    # The central five bins have width duration_days / 5. Adjacent local
    # baseline bins measure depth without folding phase-curve modulation or a
    # secondary eclipse from the rest of the orbit into the transit depth.
    edges = np.concatenate(
        (
            np.linspace(-0.5 * period_days, -local_baseline_limit, broad_baseline_bin_count + 1),
            np.linspace(-local_baseline_limit, -0.5 * duration_days, local_baseline_bin_count + 1)[1:],
            np.linspace(-0.5 * duration_days, 0.5 * duration_days, transit_bin_count + 1)[1:],
            np.linspace(0.5 * duration_days, local_baseline_limit, local_baseline_bin_count + 1)[1:],
            np.linspace(local_baseline_limit, 0.5 * period_days, broad_baseline_bin_count + 1)[1:],
        )
    )
    binned_time: list = []
    binned_flux: list = []
    binned_err: list = []
    for index in range(edges.size - 1):
        if index + 1 == edges.size - 1:
            in_bin = (time_days >= edges[index]) & (time_days <= edges[index + 1])
        else:
            in_bin = (time_days >= edges[index]) & (time_days < edges[index + 1])
        if not np.any(in_bin):
            continue
        weights = np.reciprocal(np.square(flux_err[in_bin]))
        weight_sum = float(np.sum(weights))
        if not math.isfinite(weight_sum) or weight_sum <= 0.0:
            continue
        binned_time.append(float(np.sum(weights * time_days[in_bin]) / weight_sum))
        binned_flux.append(float(np.sum(weights * flux[in_bin]) / weight_sum))
        binned_err.append(float(math.sqrt(1.0 / weight_sum)))
    if len(binned_time) < 10:
        raise ValueError("TRICERATOPS phase folding produced fewer than ten populated bins")

    phase_days = np.asarray(binned_time, dtype=float)
    phase_flux = np.asarray(binned_flux, dtype=float)
    phase_err = np.asarray(binned_err, dtype=float)
    in_transit = np.abs(phase_days) <= 0.5 * duration_days
    out_of_transit = (np.abs(phase_days) >= duration_days) & (
        np.abs(phase_days) <= local_baseline_limit
    )
    if int(np.count_nonzero(out_of_transit)) < 3:
        out_of_transit = np.abs(phase_days) > 0.5 * duration_days
    if int(np.count_nonzero(in_transit)) < transit_bin_count or int(np.count_nonzero(out_of_transit)) < 3:
        raise ValueError("TRICERATOPS requires populated in-transit and out-of-transit phase bins")
    depth_ppm = float((np.median(phase_flux[out_of_transit]) - np.median(phase_flux[in_transit])) * 1e6)
    if not math.isfinite(depth_ppm) or depth_ppm <= 0.0:
        raise ValueError("TRICERATOPS could not measure a positive observed transit depth")

    exposure_steps = np.diff(np.sort(np.unique(time_btjd)))
    exposure_steps = exposure_steps[np.isfinite(exposure_steps) & (exposure_steps > 0.0)]
    if exposure_steps.size == 0:
        raise ValueError("TRICERATOPS could not determine the observed cadence")
    exposure_days = float(np.median(exposure_steps))
    flux_err_scalar = float(np.mean(phase_err))
    # NUMERICAL_GUARD: adopt the inverse-variance effective error so mixed-cadence
    # or multi-sector phase bins weight their uncertainties correctly instead of
    # over-trusting a simple arithmetic mean of reported standard errors.
    if np.all(phase_err > 0.0) and np.isfinite(phase_err).all():
        flux_err_scalar = float(np.sqrt(phase_err.size / float(np.sum(1.0 / (phase_err ** 2)))))
    if not math.isfinite(exposure_days) or exposure_days <= 0.0 or not math.isfinite(flux_err_scalar):
        raise ValueError("TRICERATOPS observed photometry has invalid cadence or uncertainty")

    candidate_root = workspace.path.resolve()
    input_files = []
    for item in table.get("input_files", []):
        path = Path(item).resolve()
        try:
            relative = path.relative_to(candidate_root)
        except ValueError as exc:
            raise ValueError("TRICERATOPS input photometry must belong to the candidate workspace") from exc
        if not path.is_file():
            raise FileNotFoundError("TRICERATOPS input photometry is unavailable: {0}".format(relative))
        input_files.append({"path": relative.as_posix(), "sha256": _sha256(path)})
    if not input_files:
        raise ValueError("TRICERATOPS requires candidate-local photometry provenance")

    return {
        "time_days": phase_days,
        "flux": phase_flux,
        "flux_err": flux_err_scalar,
        "period_days": period_days,
        "duration_hours": duration_days * 24.0,
        "depth_ppm": depth_ppm,
        "exposure_days": exposure_days,
        "sectors": sorted({int(value) for value in sectors}),
        "provenance": {
            "representation": "phase-folded observed candidate photometry",
            "time_reference": "days from nearest declared transit midpoint",
            "raw_cadence_count": int(time_btjd.size),
            "phase_bin_count": int(phase_days.size),
            "phase_binning": {
                "method": "transit-centered-nonuniform",
                "transit_bin_count": transit_bin_count,
                "transit_bin_width_days": duration_days / transit_bin_count,
                "local_baseline_limit_days": local_baseline_limit,
            },
            "flux_error_source": "reported per-cadence uncertainties",
            "flux_error_scalar": flux_err_scalar,
            "exposure_days": exposure_days,
            "input_files": input_files,
            "ephemeris_artifacts": _ephemeris_artifacts(
                workspace, signal, ephemeris.get("source")
            ),
            "ephemeris_source": ephemeris.get("source"),
            "ephemeris_field_sources": {
                field: field_sources.get(field) for field in required_fields
            },
            "observed_depth_ppm": depth_ppm,
        },
    }


def fpp_gate(
    report_or_value: Dict[str, Any],
    threshold: float = FPP_THRESHOLD,
    nfpp_threshold: float = NFPP_THRESHOLD,
) -> Tuple[bool, float]:
    """Return ``(passes, fpp)`` for an FPP or joint FPP/NFPP screen.

    A scalar preserves the historical FPP-only utility behavior. A report is
    stricter: both finite probabilities must lie in the unit interval and each
    must be below its registered threshold. The function remains a transparent
    routing helper; the global analysis claim gate is intentionally separate
    and remains closed pending calibrated scene constraints.
    """
    if isinstance(report_or_value, dict):
        fpp = extract_fpp(report_or_value)
        try:
            nfpp = extract_nfpp(report_or_value)
        except (TypeError, ValueError):
            return False, fpp
        if not (
            math.isfinite(fpp)
            and math.isfinite(nfpp)
            and 0.0 <= fpp <= 1.0
            and 0.0 <= nfpp <= 1.0
        ):
            return False, fpp
        return fpp < threshold and nfpp < nfpp_threshold, fpp
    else:
        fpp = float(report_or_value)
    return fpp < threshold, fpp


def run_triceratops_simulation(
    workspace: Any,
    n_draws: int = 2000,
    search_radius: int = 10,
    signal: Optional[str] = None,
    allow_fallback: bool = False,
    random_seed: int = DEFAULT_TRICERATOPS_SEED,
    n_jobs: int = 1,
    progress_callback: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None,
) -> Path:
    """Run TRICERATOPS Monte Carlo Bayesian false positive probability sampling target-neutrally.

    Reads candidate target metadata and either a per-signal transit config
    (``config/signals/transit_config<signal>.json`` when ``signal`` is given)
    or the BLS periodogram outputs, executes Monte Carlo sampling over
    candidate model scenarios, and writes outputs/triceratops_report.json.
    FPP claims are disabled until observed photometry and scene constraints are
    integrated and provenance-bound.

    Parameters
    ----------
    allow_fallback:
        When True, allow the report to be written even if the TRICERATOPS Monte
        Carlo could not run (e.g., the package is not installed). The report will
        carry ``source = "triceratops-failed-UNVALIDATED"`` and ``FPP = null``.
        When False (default), a RuntimeError is raised so the analysis gate is
        not silently satisfied by a placeholder value.
    random_seed:
        Seed applied only while calling the backend's legacy global NumPy RNG,
        then restored. The seed is recorded in the report. It makes this
        single realization replayable but does not quantify Monte Carlo
        variation across independent realizations.
    n_jobs:
        One preserves the canonical serial reference execution. Values above
        one opt into TRICERATOPS's in-process PyTransit vectorized mode and
        request that its Numba runtime use at most that many threads. Exonym
        never partitions scenarios or draw streams across processes.
    progress_callback:
        Optional milestone callback receiving ``(step, done, total)``. The
        optional backend does not expose per-draw progress, so only start and
        completion milestones are emitted.
    """
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")
    if random_seed < 0 or random_seed > 2**32 - 1:
        raise ValueError("random_seed must be between 0 and 2**32 - 1")
    if isinstance(n_draws, bool) or int(n_draws) < 1:
        raise ValueError("n_draws must be at least one")
    if isinstance(search_radius, bool) or int(search_radius) < 1:
        raise ValueError("search_radius must be at least one")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    signal = validate_signal_suffix(signal)
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    tic_str = workspace.metadata.get("identifiers", {}).get("tic")
    tic_id = int(tic_str) if tic_str and str(tic_str).isdigit() else None
    observed_input: Optional[Dict[str, Any]] = None
    _notify_progress(progress_callback, "Preparing observed TRICERATOPS input")
    from ..statistical_vetting import (
        triceratops_vetting_decision_path,
        write_triceratops_vetting_decision,
        _write_json_atomic,
    )

    def _existing_vetting_decision() -> Dict[str, Any]:
        path = triceratops_vetting_decision_path(workspace, signal)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    input_snapshot = []
    if tic_id is not None:
        # The public function is also an API entry point. Enforce the same
        # ordered readiness checks as the CLI before importing or invoking the
        # Monte Carlo backend, so callers cannot bypass the scientific guard.
        from ..statistical_vetting import require_vetting_readiness

        require_vetting_readiness(workspace, signal=signal)
        try:
            observed_input = _prepare_observed_transit_input(workspace, signal)
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            # The observed-input contract is part of execution, not a
            # scientific disposition. Keep the candidate unresolved and
            # replace a stale ``ready`` record with an explicit failure.
            write_triceratops_vetting_decision(
                workspace,
                signal=signal,
                execution_status="failed",
                triage_status=_existing_vetting_decision().get("triage_status", "not-run"),
                result_status="unresolved",
                fpp=None,
                nfpp=None,
                blocking_reasons=[],
                error={
                    "code": "triceratops-observed-input-failed",
                    "message": "{0}: {1}".format(type(exc).__name__, exc),
                },
            )
            raise

        previous = _existing_vetting_decision()
        try:
            input_snapshot = _snapshot_artifacts(
                workspace,
                list(previous.get("input_artifacts", []))
                + list(observed_input["provenance"].get("input_files", []))
                + list(observed_input["provenance"].get("ephemeris_artifacts", [])),
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            write_triceratops_vetting_decision(
                workspace,
                signal=signal,
                execution_status="failed",
                triage_status=previous.get("triage_status", "not-run"),
                result_status="unresolved",
                fpp=None,
                nfpp=None,
                blocking_reasons=[],
                error={
                    "code": "triceratops-input-snapshot-failed",
                    "message": "{0}: {1}".format(type(exc).__name__, exc),
                },
            )
            raise

    period, depth_ppm, duration_hrs, ephemeris_source = 2.50, 1250.0, 2.85, "defaults"
    if observed_input is not None:
        period = observed_input["period_days"]
        depth_ppm = observed_input["depth_ppm"]
        duration_hrs = observed_input["duration_hours"]
        ephemeris_source = observed_input["provenance"]["ephemeris_source"]
    else:
        # A fallback report may describe a candidate-owned ephemeris, but it
        # must use the same provenance checks as the actual observed-data path.
        try:
            from ..inputs import load_transit_ephemeris

            ephemeris = load_transit_ephemeris(workspace, signal=signal)
            field_sources = ephemeris.get("field_sources", {})
            required_fields = ("period_days", "duration_days", "depth_ppm")
            if not isinstance(field_sources, dict) or any(
                field_sources.get(field) == "synthetic-demo" for field in required_fields
            ):
                raise ValueError("ephemeris is incomplete or synthetic")
            period = float(ephemeris["period_days"])
            depth_ppm = float(ephemeris["depth_ppm"])
            duration_hrs = float(ephemeris["duration_days"]) * 24.0
            if (
                not math.isfinite(period)
                or not math.isfinite(depth_ppm)
                or not math.isfinite(duration_hrs)
                or period <= 0.0
                or depth_ppm <= 0.0
                or duration_hrs <= 0.0
            ):
                raise ValueError("ephemeris values are not physically usable")
            ephemeris_source = str(ephemeris.get("source"))
        except (KeyError, TypeError, ValueError) as exc:
            detail = (
                "could not read signal transit config transit_config{0}.json".format(signal)
                if signal is not None
                else "could not load candidate-derived transit ephemeris"
            )
            warnings.warn(
                "{0}: {1!r}".format(detail, exc),
                stacklevel=2,
            )

    # fpp is initialized to NaN so any code path that does not successfully
    # run the Monte Carlo produces an explicit sentinel rather than a
    # hardcoded value that could silently satisfy the FPP gate.
    fpp: float = float("nan")
    nfpp: float = float("nan")
    scenarios: Dict[str, float] = {}
    source = "not-run"
    triceratops_error: Optional[str] = None
    triceratops_exception_type: Optional[str] = None
    triceratops_error_code: Optional[str] = None
    backend: Optional[Dict[str, Any]] = None
    scene_artifacts = []
    runtime_compatible = True

    if tic_id is not None:
        from ..engines import check_engine

        runtime_compatible, runtime_message = check_engine("triceratops")
        if not runtime_compatible:
            triceratops_error = runtime_message
            triceratops_error_code = "triceratops-runtime-incompatible"
            source = "triceratops-failed-UNVALIDATED"

    if tic_id is not None and runtime_compatible:
        try:
            import numpy as np
            from exonym.vetting.trex import run_trex_vetting

            _verify_snapshot(workspace, input_snapshot)

            backend = {
                "package": "exonym.vetting.trex",
                "version": "native",
                "numpy_version": str(np.__version__),
            }

            if observed_input is None:
                raise RuntimeError("observed input preparation did not complete")

            scene, scene_artifacts = _load_trex_scene(
                workspace,
                tic_id,
                observed_input.get("sectors", []),
            )
            input_snapshot = _snapshot_artifacts(
                workspace, list(input_snapshot) + list(scene_artifacts)
            )
            _verify_snapshot(workspace, input_snapshot)

            time_val = observed_input.get("time_days") if "time_days" in observed_input else observed_input.get("time")
            flux_val = observed_input["flux"]
            sigma_val = observed_input.get("flux_err") if "flux_err" in observed_input else observed_input.get("sigma")
            if sigma_val is None:
                raise RuntimeError("observed input lacks a measured flux uncertainty")

            result = run_trex_vetting(
                scene,
                time=time_val,
                flux=flux_val,
                sigma=sigma_val,
                period_days=period,
                depth_ppm=depth_ppm,
                n_draws=n_draws,
                random_seed=random_seed,
                progress_callback=progress_callback,
                exptime_days=observed_input["exposure_days"],
            )

            fpp = result.fpp if result.fpp is not None else float("nan")
            nfpp = result.nfpp if result.nfpp is not None else float("nan")
            scenarios = dict(result.top_scenarios(99))
            source = "trex-monte-carlo"
            scene_artifacts = []

            _verify_snapshot(workspace, input_snapshot)

        except Exception as exc:
            fpp = float("nan")
            nfpp = float("nan")
            scenarios = {}
            triceratops_error = "{0}: {1}".format(type(exc).__name__, exc)
            triceratops_exception_type = type(exc).__name__
            triceratops_error_code = (
                "trex-scene-unavailable"
                if isinstance(exc, TrexSceneUnavailableError)
                else "trex-runtime-failed"
            )
            warning = (
                "TREX scene is unavailable: {0!r}. FPP will be marked UNVALIDATED."
                if isinstance(exc, TrexSceneUnavailableError)
                else "TREX Monte Carlo failed: {0!r}. FPP will be marked UNVALIDATED."
            )
            warnings.warn(warning.format(exc), stacklevel=2)
            source = "triceratops-failed-UNVALIDATED"
    if source in ("not-run", "triceratops-failed-UNVALIDATED"):
        unavailable = source == "not-run" or triceratops_exception_type in {
            "ImportError", "ModuleNotFoundError", "PackageNotFoundError",
        } or triceratops_error_code in {
            "triceratops-runtime-incompatible", "trex-scene-unavailable",
        }
        failure = triceratops_error or "TRICERATOPS could not run because no numeric TIC identifier is available."
        write_triceratops_vetting_decision(
            workspace,
            signal=signal,
            execution_status="unavailable" if unavailable else "failed",
            triage_status=_existing_vetting_decision().get("triage_status", "not-run"),
            blocking_reasons=[],
            input_artifacts=input_snapshot,
            error={
                "code": triceratops_error_code
                if triceratops_error_code in {
                    "triceratops-runtime-incompatible", "trex-scene-unavailable",
                }
                else "triceratops-unavailable"
                if unavailable
                else triceratops_error_code,
                "message": failure,
            },
        )

    # Raise before writing any files when the Monte Carlo was not run and the
    # caller has not explicitly opted in to an unvalidated fallback.
    if not allow_fallback and (source in ("not-run", "triceratops-failed-UNVALIDATED")):
        remediation = (
            "Install the 'triceratops' package or pass allow_fallback=True "
            "to write an unvalidated placeholder report."
            if unavailable
            else "The underlying failure was: {0}".format(failure)
        )
        raise RuntimeError(
            "TRICERATOPS Monte Carlo did not run (source={0!r}). "
            "{1}".format(source, remediation)
        )

    fpp_rounded: Optional[float] = round(fpp, 6) if math.isfinite(fpp) else None
    nfpp_rounded: Optional[float] = round(nfpp, 6) if math.isfinite(nfpp) else None

    report = {
        "method": "TRICERATOPS",
        "candidate_id": workspace.candidate_id,
        "tic_id": tic_id,
        "signal": signal,
        "n_draws": n_draws,
        "random_seed": random_seed,
        "backend": backend,
        "ephemeris": {
            "period_days": round(period, 6),
            "depth_ppm": round(depth_ppm, 2),
            "duration_hours": round(duration_hrs, 3),
            "source": ephemeris_source,
        },
        "FPP": fpp_rounded,
        "NFPP": nfpp_rounded,
        "scenarios": scenarios,
        "source": source,
        "triceratops_error": triceratops_error,
        "input_provenance": (
            {
                **observed_input["provenance"],
                "bound_artifacts": input_snapshot,
                "scene_artifacts": scene_artifacts,
            }
            if observed_input is not None
            else None
        ),
        "audit_status": "valid" if source == "trex-monte-carlo" else "invalid",
        "audit_invalid_reason": (
            None
            if source == "trex-monte-carlo"
            else "TRICERATOPS Monte Carlo did not complete; this report is not auditable scientific evidence."
        ),
        "claim_eligible": False,
        "claim_block_reason": FPP_CLAIM_BLOCK_REASON,
    }
    suffix = f".{signal.lstrip('.')}" if signal else ""
    report_path = outputs_dir / f"triceratops_report{suffix}.json"
    _write_json_atomic(report_path, report)

    if source == "trex-monte-carlo":
        passes = fpp_rounded is not None and nfpp_rounded is not None and fpp_rounded < FPP_THRESHOLD and nfpp_rounded < NFPP_THRESHOLD
        previous = _existing_vetting_decision()
        triage_status = previous.get("triage_status", "not-run")
        write_triceratops_vetting_decision(
            workspace,
            signal=signal,
            execution_status="succeeded",
            triage_status=triage_status,
            result_status=("review-required" if passes and triage_status == "review-required" else "fpp-pass" if passes else "fpp-fail"),
            fpp=fpp_rounded,
            nfpp=nfpp_rounded,
            input_artifacts=input_snapshot,
            triceratops_report={"path": report_path.relative_to(workspace.path).as_posix(), "sha256": _sha256(report_path)},
            audit_status="valid",
            audit_invalid_reason=None,
        )
    elif allow_fallback:
        previous = _existing_vetting_decision()
        write_triceratops_vetting_decision(
            workspace,
            signal=signal,
            execution_status=previous.get("execution_status", "failed"),
            triage_status=previous.get("triage_status", "not-run"),
            result_status="unresolved",
            error=previous.get("error") if isinstance(previous.get("error"), dict) else None,
            input_artifacts=input_snapshot,
            triceratops_report={"path": report_path.relative_to(workspace.path).as_posix(), "sha256": _sha256(report_path)},
            audit_status="invalid",
            audit_invalid_reason=report["audit_invalid_reason"],
        )

    return report_path
