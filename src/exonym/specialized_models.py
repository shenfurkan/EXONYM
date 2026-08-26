"""Run tightly scoped optional specialized-model adapters in candidate space.

``planetsynth`` receives a declared giant-planet characterization in Jupiter
mass and radius units, gigayears, and kelvin.  ``pyPplusS`` receives a declared
anomalous-transit hypothesis with cadence-aligned time in days and normalized
relative flux.  Both adapters preserve input hashes, raw package output, and
runtime metadata before issuing a normalized report.

The resulting diagnostics describe one supplied model or hypothesis.  They do
not compare all physical explanations, create a claim, validate a planet, or
change lifecycle state.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import numbers
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .resources import read_schema_text
from .workspace import CandidateWorkspace


PLANETSYNTH_ENGINE = "planetsynth"
PYPPLUSS_ENGINE = "pyppluss"
CATWOMAN_ENGINE = "catwoman"
SQUISHYPLANET_ENGINE = "squishyplanet"
PLANETSYNTH_INPUT = Path("data/external/planetsynth_characterization.json")
PYPPLUSS_INPUT = Path("data/external/anomalous_transit_hypothesis.json")
ASYMMETRIC_TRANSIT_INPUT = Path("data/external/asymmetric_transit_hypothesis.json")
PLANETSYNTH_OUTPUT_PREFIX = "planetsynth_interpretation"
PYPPLUSS_OUTPUT_PREFIX = "pyppluss_hypothesis_test"
CATWOMAN_OUTPUT_PREFIX = "catwoman_terminator_asymmetry_test"
SQUISHYPLANET_OUTPUT_PREFIX = "squishyplanet_terminator_asymmetry_test"
MAX_INPUT_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class AdapterRun:
    """Describe the candidate-local outcome of one optional adapter invocation.

    Attributes:
        status (str): Terminal adapter status such as succeeded, failed, or
            unavailable.  A non-success status still has a manifest path.
        manifest_path (Path): Candidate-local engine-run manifest that records
            inputs, runtime provenance, output digests, and any failure.
        report_path (Optional[Path]): Normalized scientific report when the
            adapter succeeds; otherwise ``None``.
    """

    status: str
    manifest_path: Path
    report_path: Optional[Path]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError("non-finite JSON number: {0}".format(value))


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError("duplicate JSON key: {0}".format(key))
        parsed[key] = value
    return parsed


def _read_input(workspace: CandidateWorkspace, relative_path: Path, schema_name: str) -> Tuple[Path, Dict[str, Any]]:
    """Load one fixed candidate-owned JSON input after strict schema validation."""
    path = workspace.path / relative_path
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("missing candidate-owned adapter input: {0}".format(relative_path.as_posix()))
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("adapter input exceeds the maximum supported JSON size")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("adapter input is not valid finite UTF-8 JSON: {0}".format(exc)) from exc
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema is required to validate specialized-model inputs") from exc
    schema = json.loads(read_schema_text(workspace.repository_root, schema_name))
    try:
        jsonschema.validate(payload, schema, format_checker=jsonschema.FormatChecker())
    except jsonschema.ValidationError as exc:
        raise ValueError("adapter input schema violation: {0}".format(exc.message)) from exc
    if not isinstance(payload, dict) or payload.get("candidate_id") != workspace.candidate_id:
        raise ValueError("adapter input candidate_id does not match the workspace")
    return path, payload


def _finite_value(value: object, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{0} must be a finite number".format(label)) from exc
    if not math.isfinite(numeric):
        raise ValueError("{0} must be a finite number".format(label))
    return numeric


def _validate_planetsynth_applicability(payload: Mapping[str, Any]) -> None:
    """Reject characterization outside the declared giant-planet model domain.

    The adapter accepts masses from 0.1 to 20 Jupiter masses, radii from 0.5 to
    2.5 Jupiter radii, ages from 0.001 to 20 Gyr, and equilibrium temperatures
    from 0 to 3000 K. These conservative bounds prevent extrapolation from being
    presented as a model result.
    """
    # SCIENTIFIC_BOUNDARY: The declared domain prevents an extrapolated package
    # result from being represented as a physical interpretation.
    characterization = payload["characterization"]
    mass_mjup = _finite_value(characterization["mass_mjup"]["value"], "mass_mjup")
    radius_rjup = _finite_value(characterization["radius_rjup"]["value"], "radius_rjup")
    age_gyr = _finite_value(characterization["age_gyr"]["value"], "age_gyr")
    temperature_k = _finite_value(
        characterization["equilibrium_temperature_k"]["value"], "equilibrium_temperature_k"
    )
    if not 0.1 <= mass_mjup <= 20.0:
        raise ValueError("planetsynth applicability requires mass_mjup between 0.1 and 20")
    if not 0.5 <= radius_rjup <= 2.5:
        raise ValueError("planetsynth applicability requires radius_rjup between 0.5 and 2.5")
    if not 0.001 <= age_gyr <= 20.0:
        raise ValueError("planetsynth applicability requires age_gyr between 0.001 and 20")
    if not 0.0 <= temperature_k <= 3000.0:
        raise ValueError("planetsynth applicability requires equilibrium_temperature_k between 0 and 3000")


def _validate_pyppluss_applicability(payload: Mapping[str, Any]) -> None:
    """Reject data and geometry outside this adapter's transit-model domain."""
    observation = payload["observation"]
    time_days = observation["time_days"]["values"]
    flux = observation["normalized_flux"]["values"]
    if len(time_days) != len(flux):
        raise ValueError("anomalous-transit time and normalized-flux arrays must have equal length")
    previous: Optional[float] = None
    for index, value in enumerate(time_days):
        current = _finite_value(value, "time_days[{0}]".format(index))
        if previous is not None and current <= previous:
            raise ValueError("anomalous-transit time_days values must be strictly increasing")
        previous = current
    for index, value in enumerate(flux):
        current = _finite_value(value, "normalized_flux[{0}]".format(index))
        if not 0.5 <= current <= 1.5:
            raise ValueError("pyPplusS applicability requires normalized flux between 0.5 and 1.5")
    hypothesis = payload["hypothesis"]
    if hypothesis["model"] == "ringed-planet":
        planet_radius = _finite_value(hypothesis["planet_radius_ratio"], "planet_radius_ratio")
        ring_inner = _finite_value(hypothesis["ring_inner_radius_ratio"], "ring_inner_radius_ratio")
        ring_outer = _finite_value(hypothesis["ring_outer_radius_ratio"], "ring_outer_radius_ratio")
        if not planet_radius < ring_inner < ring_outer:
            raise ValueError("ringed-transit applicability requires planet_radius_ratio < ring_inner_radius_ratio < ring_outer_radius_ratio")


def _validate_asymmetric_transit_applicability(
    workspace: CandidateWorkspace, payload: Mapping[str, Any]
) -> List[Dict[str, str]]:
    """Validate one declared terminator-asymmetry observation and its inputs."""
    observation = payload["observation"]
    time_days = observation["time_days"]["values"]
    flux = observation["normalized_flux"]["values"]
    flux_error = observation["normalized_flux_error"]["values"]
    if len(time_days) != len(flux) or len(time_days) != len(flux_error):
        raise ValueError("asymmetric-transit time, flux, and uncertainty arrays must have equal length")
    previous: Optional[float] = None
    for index, value in enumerate(time_days):
        current = _finite_value(value, "time_days[{0}]".format(index))
        if previous is not None and current <= previous:
            raise ValueError("asymmetric-transit time_days values must be strictly increasing")
        previous = current
    for index, (value, error) in enumerate(zip(flux, flux_error)):
        normalized_flux = _finite_value(value, "normalized_flux[{0}]".format(index))
        normalized_error = _finite_value(error, "normalized_flux_error[{0}]".format(index))
        if not 0.5 <= normalized_flux <= 1.5:
            raise ValueError("terminator-asymmetry applicability requires normalized flux between 0.5 and 1.5")
        if normalized_error <= 0.0:
            raise ValueError("terminator-asymmetry applicability requires positive flux uncertainties")
    geometry = payload["geometry"]
    period_days = _finite_value(geometry["period_days"], "period_days")
    scaled_semimajor_axis = _finite_value(
        geometry["scaled_semimajor_axis"], "scaled_semimajor_axis"
    )
    impact_parameter = _finite_value(geometry["impact_parameter"], "impact_parameter")
    limb_darkening = geometry["quadratic_limb_darkening"]
    _finite_value(limb_darkening["u1"], "quadratic_limb_darkening.u1")
    _finite_value(limb_darkening["u2"], "quadratic_limb_darkening.u2")
    if period_days <= 0.0 or scaled_semimajor_axis <= 1.0:
        raise ValueError("terminator-asymmetry geometry requires positive period and scaled semimajor axis above 1")
    if not 0.0 <= impact_parameter <= 1.5:
        raise ValueError("terminator-asymmetry impact_parameter must be between 0 and 1.5")
    hypothesis = payload["hypothesis"]
    morning_radius = _finite_value(hypothesis["morning_radius_ratio"], "morning_radius_ratio")
    evening_radius = _finite_value(hypothesis["evening_radius_ratio"], "evening_radius_ratio")
    if not 0.0 < morning_radius <= 0.3 or not 0.0 < evening_radius <= 0.3:
        raise ValueError("terminator-asymmetry radius ratios must be in (0, 0.3]")

    artifacts = payload["provenance"]["input_artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("terminator-asymmetry provenance requires candidate-owned input artifacts")
    verified: List[Dict[str, str]] = []
    workspace_root = workspace.path.resolve()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise ValueError("input_artifacts[{0}] must be an object".format(index))
        relative_value = artifact.get("path")
        digest = artifact.get("sha256")
        role = artifact.get("role")
        if not isinstance(relative_value, str) or not isinstance(digest, str) or not isinstance(role, str):
            raise ValueError("input_artifacts[{0}] must declare path, sha256, and role".format(index))
        relative_path = Path(relative_value)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("input_artifacts[{0}] path must remain in the candidate workspace".format(index))
        path = workspace.path / relative_path
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(workspace_root):
            raise ValueError("input_artifacts[{0}] must reference a regular candidate-owned file".format(index))
        if _sha256(path) != digest:
            raise ValueError("input_artifacts[{0}] SHA-256 does not match its candidate-owned file".format(index))
        verified.append({"path": relative_path.as_posix(), "sha256": digest, "role": role})
    return verified


def _finite_json(value: Any) -> Any:
    """Return a finite JSON-compatible external result or reject it before writing."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("external adapter result contains a non-finite number")
        return numeric
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    raise ValueError("external adapter result is not JSON-compatible")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _create_run_dir(workspace: CandidateWorkspace, engine: str) -> Tuple[str, Path]:
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


def _artifact(workspace: CandidateWorkspace, path: Path, role: str) -> Dict[str, str]:
    return {
        "path": path.resolve().relative_to(workspace.path.resolve()).as_posix(),
        "sha256": _sha256(path),
        "role": role,
    }


def _run_in_directory(run_dir: Path, function: Any, *args: Any, **kwargs: Any) -> Any:
    """Run an optional package with relative scratch files confined to its run directory."""
    previous_directory = Path.cwd()
    try:
        os.chdir(run_dir)
        return function(*args, **kwargs)
    finally:
        os.chdir(previous_directory)


def _clear_partial_run_outputs(run_dir: Path) -> None:
    """Remove package-created files after a failed or unsupported invocation."""
    for path in sorted(run_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def _package_output_paths(run_dir: Path, raw_path: Path) -> List[Path]:
    """Return package-created files for manifest hashing, excluding the normalized raw result."""
    return sorted(path for path in run_dir.rglob("*") if path.is_file() and path != raw_path)


def _write_manifest(
    workspace: CandidateWorkspace,
    engine: str,
    run_id: str,
    run_dir: Path,
    started_at: str,
    runtime: Mapping[str, str],
    input_path: Path,
    status: str,
    output_paths: Sequence[Tuple[Path, str]],
    failure: Optional[Mapping[str, str]] = None,
) -> Path:
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "engine": engine,
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime": dict(runtime),
        "inputs": [_artifact(workspace, input_path, "declared-candidate-input")],
        "outputs": [_artifact(workspace, path, role) for path, role in output_paths],
    }
    if failure is not None:
        manifest["failure"] = dict(failure)
    manifest_path = run_dir / "engine-run.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _resolve_runtime(module_name: str, distribution: str) -> Tuple[Dict[str, str], Any]:
    """Verify installed metadata and Python compatibility before importing a package."""
    try:
        package_metadata = metadata.metadata(distribution)
        package_version = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise LookupError("module-unavailable: {0}".format(exc)) from exc
    requires_python = package_metadata.get("Requires-Python")
    if not requires_python:
        raise LookupError("runtime-metadata-missing: installed package has no Requires-Python metadata")
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        current_python = Version("{0}.{1}.{2}".format(*sys.version_info[:3]))
        if current_python not in SpecifierSet(requires_python):
            raise LookupError(
                "python-version-unsupported: package requires Python {0}".format(requires_python)
            )
    except ImportError as exc:
        raise LookupError("runtime-compatibility-unavailable: packaging is unavailable") from exc
    except (TypeError, ValueError) as exc:
        raise LookupError("runtime-metadata-invalid: invalid Requires-Python metadata") from exc
    if importlib.util.find_spec(module_name) is None:
        raise LookupError("module-unavailable: package metadata exists but module is not importable")
    return (
        {"kind": "direct", "version": str(package_version), "executable": module_name},
        {"package": distribution, "version": str(package_version), "python_requires": requires_python},
    )


def _unavailable_run(
    workspace: CandidateWorkspace,
    engine: str,
    input_path: Path,
    started_at: str,
    code: str,
    message: str,
    runtime: Optional[Mapping[str, str]] = None,
    run_id: Optional[str] = None,
    run_dir: Optional[Path] = None,
) -> AdapterRun:
    if run_id is None or run_dir is None:
        run_id, run_dir = _create_run_dir(workspace, engine)
    manifest_path = _write_manifest(
        workspace,
        engine,
        run_id,
        run_dir,
        started_at,
        runtime or {"kind": "direct", "version": "unavailable", "executable": engine},
        input_path,
        "unavailable",
        [],
        {"code": code, "message": message},
    )
    return AdapterRun("unavailable", manifest_path, None)


def run_planetsynth(workspace: CandidateWorkspace) -> AdapterRun:
    """Run a declared giant-planet cooling interpretation when planetsynth supports it.

    The supported package interface is ``evolve_giant_planet`` with keyword
    arguments ``mass_mjup``, ``radius_rjup``, ``age_gyr``, and
    ``equilibrium_temperature_k``.  Inputs are passed in the units declared by
    the candidate-local schema; a successful result must provide finite radius
    in Jupiter radii and luminosity in solar luminosities.

    Args:
        workspace (CandidateWorkspace): Workspace that owns the declared
            characterization input and receives all run artifacts.

    Returns:
        AdapterRun: Status plus manifest path, and a normalized interpretation
        report path only when the installed package succeeds.

    Raises:
        FileNotFoundError: The candidate-owned characterization input is absent.
        ValueError: Input JSON, schema fields, ownership, or declared
            applicability values are invalid.
        RuntimeError: Schema validation support is unavailable.

    Note:
        Cooling and evolution output is descriptive downstream interpretation,
        not a planet claim or model-selection result.
    """
    input_path, payload = _read_input(
        workspace, PLANETSYNTH_INPUT, "planetsynth-characterization.schema.json"
    )
    _validate_planetsynth_applicability(payload)
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        runtime, runtime_report = _resolve_runtime("planetsynth", "planetsynth")
    except LookupError as exc:
        code, _, message = str(exc).partition(": ")
        return _unavailable_run(workspace, PLANETSYNTH_ENGINE, input_path, started_at, code, message or code)
    run_id, run_dir = _create_run_dir(workspace, PLANETSYNTH_ENGINE)
    try:
        module = _run_in_directory(run_dir, importlib.import_module, "planetsynth")
    except Exception as exc:
        _clear_partial_run_outputs(run_dir)
        manifest_path = _write_manifest(
            workspace, PLANETSYNTH_ENGINE, run_id, run_dir, started_at, runtime, input_path,
            "failed", [], {"code": "module-import-failed", "message": str(exc)},
        )
        return AdapterRun("failed", manifest_path, None)
    model = getattr(module, "evolve_giant_planet", None)
    if not callable(model):
        _clear_partial_run_outputs(run_dir)
        return _unavailable_run(
            workspace,
            PLANETSYNTH_ENGINE,
            input_path,
            started_at,
            "unsupported-interface",
            "planetsynth must expose callable evolve_giant_planet; no interpretation was written.",
            runtime,
            run_id,
            run_dir,
        )
    characterization = payload["characterization"]
    try:
        result = _finite_json(
            _run_in_directory(
                run_dir,
                model,
                mass_mjup=characterization["mass_mjup"]["value"],
                radius_rjup=characterization["radius_rjup"]["value"],
                age_gyr=characterization["age_gyr"]["value"],
                equilibrium_temperature_k=characterization["equilibrium_temperature_k"]["value"],
            )
        )
        if not isinstance(result, dict):
            raise ValueError("planetsynth result must be a mapping")
        radius_rjup = _finite_value(result.get("radius_rjup"), "planetsynth radius_rjup")
        luminosity_lsun = _finite_value(result.get("luminosity_lsun"), "planetsynth luminosity_lsun")
        if radius_rjup <= 0 or luminosity_lsun < 0:
            raise ValueError("planetsynth returned physically invalid radius or luminosity")
    except Exception as exc:
        _clear_partial_run_outputs(run_dir)
        manifest_path = _write_manifest(
            workspace, PLANETSYNTH_ENGINE, run_id, run_dir, started_at, runtime, input_path,
            "failed", [], {"code": "adapter-execution-failed", "message": str(exc)},
        )
        return AdapterRun("failed", manifest_path, None)
    raw_path = run_dir / "raw_result.json"
    _write_json(raw_path, result)
    package_outputs = _package_output_paths(run_dir, raw_path)
    report_path = workspace.path / "outputs" / "{0}.{1}.json".format(PLANETSYNTH_OUTPUT_PREFIX, run_id)
    report = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "succeeded",
        "input_artifact": _artifact(workspace, input_path, "planetsynth-characterization"),
        "raw_result_artifact": _artifact(workspace, raw_path, "planetsynth-raw-result"),
        "runtime": runtime_report,
        "interpretation": {
            "radius": {"value": radius_rjup, "unit": "R_jup"},
            "luminosity": {"value": luminosity_lsun, "unit": "L_sun"},
        },
        "caveat": "Cooling and evolution output is descriptive downstream interpretation, not a planet claim, validation result, or disposition.",
    }
    _write_json(report_path, report)
    manifest_path = _write_manifest(
        workspace,
        PLANETSYNTH_ENGINE,
        run_id,
        run_dir,
        started_at,
        runtime,
        input_path,
        "succeeded",
        [(raw_path, "planetsynth-raw-result")]
        + [(path, "package-output") for path in package_outputs]
        + [(report_path, "planetsynth-interpretation")],
    )
    return AdapterRun("succeeded", manifest_path, report_path)


def run_pyppluss(workspace: CandidateWorkspace) -> AdapterRun:
    """Test a declared anomalous-transit hypothesis with pyPplusS when supported.

    The supported package interface is ``model_anomalous_transit(time_days=...,
    hypothesis=...)``.  It must return finite cadence-aligned ``model_flux``.
    The normalized report records residual RMS and maximum absolute residual in
    dimensionless relative flux, matching the adapter-method note.

    Args:
        workspace (CandidateWorkspace): Workspace that owns the declared
            hypothesis, observed normalized flux, and output run directory.

    Returns:
        AdapterRun: Status plus manifest path, and a normalized hypothesis-test
        report path only when the package interface and output are usable.

    Raises:
        FileNotFoundError: The candidate-owned hypothesis input is absent.
        ValueError: Input JSON, schema fields, time ordering, normalized flux,
            geometry, or package output is invalid.
        RuntimeError: Schema validation support is unavailable.

    Note:
        This compares one declared geometry.  It is not evidence that the
        geometry is unique or that competing astrophysical explanations fail.
    """
    input_path, payload = _read_input(
        workspace, PYPPLUSS_INPUT, "anomalous-transit-hypothesis.schema.json"
    )
    _validate_pyppluss_applicability(payload)
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        runtime, runtime_report = _resolve_runtime("pyppluss", "pyppluss")
    except LookupError as exc:
        code, _, message = str(exc).partition(": ")
        return _unavailable_run(workspace, PYPPLUSS_ENGINE, input_path, started_at, code, message or code)
    run_id, run_dir = _create_run_dir(workspace, PYPPLUSS_ENGINE)
    try:
        module = _run_in_directory(run_dir, importlib.import_module, "pyppluss")
    except Exception as exc:
        _clear_partial_run_outputs(run_dir)
        manifest_path = _write_manifest(
            workspace, PYPPLUSS_ENGINE, run_id, run_dir, started_at, runtime, input_path,
            "failed", [], {"code": "module-import-failed", "message": str(exc)},
        )
        return AdapterRun("failed", manifest_path, None)
    model = getattr(module, "model_anomalous_transit", None)
    if not callable(model):
        _clear_partial_run_outputs(run_dir)
        return _unavailable_run(
            workspace,
            PYPPLUSS_ENGINE,
            input_path,
            started_at,
            "unsupported-interface",
            "pyPplusS must expose callable model_anomalous_transit; no hypothesis test was written.",
            runtime,
            run_id,
            run_dir,
        )
    observation = payload["observation"]
    try:
        result = _finite_json(
            _run_in_directory(
                run_dir,
                model,
                time_days=observation["time_days"]["values"],
                hypothesis=payload["hypothesis"],
            )
        )
        if not isinstance(result, dict) or not isinstance(result.get("model_flux"), list):
            raise ValueError("pyPplusS result must provide a model_flux array")
        model_flux = result["model_flux"]
        measured_flux = observation["normalized_flux"]["values"]
        if len(model_flux) != len(measured_flux):
            raise ValueError("pyPplusS model_flux length does not match the declared observation")
        residuals = [float(measured) - _finite_value(modeled, "pyPplusS model_flux") for measured, modeled in zip(measured_flux, model_flux)]
        rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
        maximum = max(abs(value) for value in residuals)
    except Exception as exc:
        _clear_partial_run_outputs(run_dir)
        manifest_path = _write_manifest(
            workspace, PYPPLUSS_ENGINE, run_id, run_dir, started_at, runtime, input_path,
            "failed", [], {"code": "adapter-execution-failed", "message": str(exc)},
        )
        return AdapterRun("failed", manifest_path, None)
    raw_path = run_dir / "raw_result.json"
    _write_json(raw_path, result)
    package_outputs = _package_output_paths(run_dir, raw_path)
    report_path = workspace.path / "outputs" / "{0}.{1}.json".format(PYPPLUSS_OUTPUT_PREFIX, run_id)
    report = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "succeeded",
        "model": payload["hypothesis"]["model"],
        "input_artifact": _artifact(workspace, input_path, "anomalous-transit-hypothesis"),
        "raw_result_artifact": _artifact(workspace, raw_path, "pyppluss-raw-result"),
        "runtime": runtime_report,
        "fit_diagnostics": {
            "rms_residual": {"value": rms, "unit": "relative_flux"},
            "max_abs_residual": {"value": maximum, "unit": "relative_flux"},
            "cadence_count": len(measured_flux),
        },
        "caveat": "A ringed or oblate transit comparison tests one declared hypothesis only. It does not establish a physical interpretation, planet claim, validation result, or disposition.",
    }
    _write_json(report_path, report)
    manifest_path = _write_manifest(
        workspace,
        PYPPLUSS_ENGINE,
        run_id,
        run_dir,
        started_at,
        runtime,
        input_path,
        "succeeded",
        [(raw_path, "pyppluss-raw-result")]
        + [(path, "package-output") for path in package_outputs]
        + [(report_path, "pyppluss-hypothesis-test")],
    )
    return AdapterRun("succeeded", manifest_path, report_path)


def _run_terminator_asymmetry_adapter(
    workspace: CandidateWorkspace,
    engine: str,
    module_name: str,
    distribution: str,
    output_prefix: str,
    required_interface: Optional[str],
) -> AdapterRun:
    """Run one pinned external terminator-asymmetry adapter.

    A supported package version must expose ``required_interface`` with keyword
    arguments for the declared time series, geometry, and asymmetric radius
    ratios. It must return a mapping containing cadence-aligned finite
    ``model_flux`` values. Unknown or unverified package APIs fail closed as
    unavailable rather than being guessed at runtime.
    """
    input_path, payload = _read_input(
        workspace, ASYMMETRIC_TRANSIT_INPUT, "asymmetric-transit-hypothesis.schema.json"
    )
    source_artifacts = _validate_asymmetric_transit_applicability(workspace, payload)
    started_at = datetime.now(timezone.utc).isoformat()
    if required_interface is None:
        return _unavailable_run(
            workspace,
            engine,
            input_path,
            started_at,
            "model-contract-unverified",
            (
                "{0} has no verified terminator-asymmetry model contract in this "
                "repository; no hypothesis test was written.".format(distribution)
            ),
        )
    try:
        runtime, runtime_report = _resolve_runtime(module_name, distribution)
    except LookupError as exc:
        code, _, message = str(exc).partition(": ")
        return _unavailable_run(
            workspace, engine, input_path, started_at, code, message or code
        )
    run_id, run_dir = _create_run_dir(workspace, engine)
    try:
        module = _run_in_directory(run_dir, importlib.import_module, module_name)
    except Exception as exc:
        _clear_partial_run_outputs(run_dir)
        manifest_path = _write_manifest(
            workspace,
            engine,
            run_id,
            run_dir,
            started_at,
            runtime,
            input_path,
            "failed",
            [],
            {"code": "module-import-failed", "message": str(exc)},
        )
        return AdapterRun("failed", manifest_path, None)
    model = getattr(module, required_interface, None)
    if not callable(model):
        _clear_partial_run_outputs(run_dir)
        return _unavailable_run(
            workspace,
            engine,
            input_path,
            started_at,
            "unsupported-interface",
            "{0} must expose callable {1}; no hypothesis test was written.".format(
                distribution, required_interface
            ),
            runtime,
            run_id,
            run_dir,
        )

    observation = payload["observation"]
    geometry = payload["geometry"]
    hypothesis = payload["hypothesis"]
    try:
        result = _finite_json(
            _run_in_directory(
                run_dir,
                model,
                time_days=observation["time_days"]["values"],
                geometry=geometry,
                morning_radius_ratio=hypothesis["morning_radius_ratio"],
                evening_radius_ratio=hypothesis["evening_radius_ratio"],
            )
        )
        if not isinstance(result, dict) or not isinstance(result.get("model_flux"), list):
            raise ValueError("terminator-asymmetry result must provide a model_flux array")
        model_flux = result["model_flux"]
        measured_flux = observation["normalized_flux"]["values"]
        flux_errors = observation["normalized_flux_error"]["values"]
        if len(model_flux) != len(measured_flux):
            raise ValueError("terminator-asymmetry model_flux length does not match the declared observation")
        residuals = [
            float(measured) - _finite_value(modeled, "terminator-asymmetry model_flux")
            for measured, modeled in zip(measured_flux, model_flux)
        ]
        chi_square = sum(
            (residual / float(error)) ** 2 for residual, error in zip(residuals, flux_errors)
        )
        rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
        maximum = max(abs(value) for value in residuals)
    except Exception as exc:
        _clear_partial_run_outputs(run_dir)
        manifest_path = _write_manifest(
            workspace,
            engine,
            run_id,
            run_dir,
            started_at,
            runtime,
            input_path,
            "failed",
            [],
            {"code": "adapter-execution-failed", "message": str(exc)},
        )
        return AdapterRun("failed", manifest_path, None)

    raw_path = run_dir / "raw_result.json"
    _write_json(raw_path, result)
    package_outputs = _package_output_paths(run_dir, raw_path)
    report_path = workspace.path / "outputs" / "{0}.{1}.json".format(output_prefix, run_id)
    morning_radius = float(hypothesis["morning_radius_ratio"])
    evening_radius = float(hypothesis["evening_radius_ratio"])
    mean_radius = 0.5 * (morning_radius + evening_radius)
    report = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "succeeded",
        "engine": engine,
        "model": "terminator-asymmetry",
        "input_artifact": _artifact(workspace, input_path, "asymmetric-transit-hypothesis"),
        "source_artifacts": source_artifacts,
        "raw_result_artifact": _artifact(workspace, raw_path, "{0}-raw-result".format(engine)),
        "runtime": runtime_report,
        "hypothesis": {
            "morning_radius_ratio": morning_radius,
            "evening_radius_ratio": evening_radius,
            "radius_difference_ratio": evening_radius - morning_radius,
            "fractional_radius_difference": abs(evening_radius - morning_radius) / mean_radius,
        },
        "fit_diagnostics": {
            "chi_square_fixed_hypothesis": chi_square,
            "rms_residual": {"value": rms, "unit": "relative_flux"},
            "max_abs_residual": {"value": maximum, "unit": "relative_flux"},
            "cadence_count": len(measured_flux),
        },
        "validation_eligible": False,
        "claim_eligible": False,
        "caveat": (
            "This fixed terminator-asymmetry comparison tests one declared geometry only. "
            "It does not establish clouds, rule out a symmetric planet, validate a planet, "
            "or change a candidate disposition."
        ),
    }
    _write_json(report_path, report)
    manifest_path = _write_manifest(
        workspace,
        engine,
        run_id,
        run_dir,
        started_at,
        runtime,
        input_path,
        "succeeded",
        [(raw_path, "{0}-raw-result".format(engine))]
        + [(path, "package-output") for path in package_outputs]
        + [(report_path, "{0}-terminator-asymmetry-test".format(engine))],
    )
    return AdapterRun("succeeded", manifest_path, report_path)


def run_catwoman(workspace: CandidateWorkspace) -> AdapterRun:
    """Run the candidate-owned Catwoman terminator-asymmetry diagnostic."""
    return _run_terminator_asymmetry_adapter(
        workspace,
        CATWOMAN_ENGINE,
        "catwoman",
        "catwoman",
        CATWOMAN_OUTPUT_PREFIX,
        "model_terminator_asymmetry",
    )


def run_squishyplanet(workspace: CandidateWorkspace) -> AdapterRun:
    """Run the candidate-owned terminator-asymmetry diagnostic, fail-closed.

    SquishyPlanet's declared model scope does not currently include a verified
    terminator-asymmetry contract in this repository. Until a supported
    interface is confirmed and pinned, every invocation is recorded as
    unavailable without importing or calling the package.
    """
    return _run_terminator_asymmetry_adapter(
        workspace,
        SQUISHYPLANET_ENGINE,
        "squishyplanet",
        "squishyplanet",
        SQUISHYPLANET_OUTPUT_PREFIX,
        None,
    )
