"""Target-neutral engine registry, execution runner, and automated triage engine.

Provides capability descriptors, optional group mappings, runtime availability checks,
reproducible run manifest generation, and pre-vetting decision triage for analytical
and vetting engines used by EXONYM.

Contains no candidate constants, sector numbers, or target identifiers.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .workspace import CandidateWorkspace, load_candidate


@dataclass(frozen=True)
class EngineDescriptor:
    """Static registration descriptor for an analytical or vetting engine."""

    name: str
    capability: str
    optional_group: str
    module_name: str
    description: str


@dataclass(frozen=True)
class EngineStatus:
    """Resolved runtime status for an engine."""

    name: str
    capability: str
    optional_group: str
    module_name: str
    description: str
    installed: bool
    version: Optional[str]


# Target-neutral canonical catalog of Exonym engines
_ENGINE_CATALOG: Tuple[EngineDescriptor, ...] = (
    EngineDescriptor(
        name="bls",
        capability="search",
        optional_group="core",
        module_name="astropy.timeseries",
        description="Box Least Squares transit search via Astropy.",
    ),
    EngineDescriptor(
        name="tls",
        capability="search",
        optional_group="discovery",
        module_name="transitleastsquares",
        description="Transit Least Squares native-cadence search engine.",
    ),
    EngineDescriptor(
        name="screen",
        capability="screening",
        optional_group="core",
        module_name="numpy",
        description="Fixed-ephemeris odd-even and secondary eclipse screening.",
    ),
    EngineDescriptor(
        name="batman",
        capability="fitting",
        optional_group="core",
        module_name="batman",
        description="Mandel-Agol transit light curve modeler.",
    ),
    EngineDescriptor(
        name="emcee",
        capability="sampler",
        optional_group="core",
        module_name="emcee",
        description="Affine-invariant ensemble MCMC sampler.",
    ),
    EngineDescriptor(
        name="dynesty",
        capability="sampler",
        optional_group="optional",
        module_name="dynesty",
        description="Dynamic nested sampling for Bayesian model comparison.",
    ),
    EngineDescriptor(
        name="triceratops",
        capability="vetting",
        optional_group="screening",
        module_name="triceratops",
        description="Bayesian transit validation and false positive probability calculation.",
    ),
    EngineDescriptor(
        name="pysyd",
        capability="asteroseismology",
        optional_group="asteroseismology",
        module_name="pysyd",
        description="Automated asteroseismic pipeline for solar-like oscillations.",
    ),
    EngineDescriptor(
        name="celerite",
        capability="detrending",
        optional_group="core",
        module_name="celerite",
        description="Fast 1D Gaussian Process regression for light curve modeling.",
    ),
    EngineDescriptor(
        name="wotan",
        capability="detrending",
        optional_group="optional",
        module_name="wotan",
        description="Comprehensive light curve detrending algorithms.",
    ),
    EngineDescriptor(
        name="ldtk",
        capability="priors",
        optional_group="core",
        module_name="ldtk",
        description="Limb Darkening Toolkit for stellar atmosphere profiles.",
    ),
    EngineDescriptor(
        name="corner",
        capability="plotting",
        optional_group="core",
        module_name="corner",
        description="Corner plot visualization for multidimensional posterior distributions.",
    ),
    EngineDescriptor(
        name="localization",
        capability="astrometry",
        optional_group="core",
        module_name="scipy",
        description="Sub-pixel PRF transit centroid source localization.",
    ),
    EngineDescriptor(
        name="activity",
        capability="activity",
        optional_group="core",
        module_name="astropy",
        description="Generalized Lomb-Scargle stellar rotational activity periodogram.",
    ),
    EngineDescriptor(
        name="sed",
        capability="sed",
        optional_group="core",
        module_name="scipy",
        description="Broadband multi-band spectral energy distribution fitting.",
    ),
    EngineDescriptor(
        name="dilution",
        capability="contamination",
        optional_group="core",
        module_name="numpy",
        description="Aperture depth stability and Gaia dilution sensitivity.",
    ),
    EngineDescriptor(
        name="ttv",
        capability="timing",
        optional_group="core",
        module_name="batman",
        description="Transit timing variation (O-C) diagram and resonance search.",
    ),
    EngineDescriptor(
        name="phasecurve",
        capability="phasecurve",
        optional_group="core",
        module_name="numpy",
        description="BEER harmonic orbital phase curve decomposition.",
    ),
    EngineDescriptor(
        name="asteroseismology",
        capability="asteroseismology",
        optional_group="core",
        module_name="astropy",
        description="Solar-like oscillation envelope and scaling relations.",
    ),
)


def _get_module_version(module_name: str) -> Optional[str]:
    """Retrieve installed version of a top-level package or module."""
    package_name = module_name.split(".")[0]
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        pass

    try:
        mod = importlib.import_module(package_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def get_engine_status(descriptor: EngineDescriptor) -> EngineStatus:
    """Inspect system runtime to determine availability of an engine."""
    package_name = descriptor.module_name.split(".")[0]
    spec = importlib.util.find_spec(package_name)
    installed = spec is not None
    version = _get_module_version(descriptor.module_name) if installed else None

    return EngineStatus(
        name=descriptor.name,
        capability=descriptor.capability,
        optional_group=descriptor.optional_group,
        module_name=descriptor.module_name,
        description=descriptor.description,
        installed=installed,
        version=version,
    )


def iter_engines() -> List[EngineStatus]:
    """List all registered engines with their current runtime installation status."""
    return [get_engine_status(desc) for desc in _ENGINE_CATALOG]


def get_engine(name: str) -> Optional[EngineStatus]:
    """Look up a specific engine by canonical name or alias."""
    normalized = name.strip().lower()
    alias_map = {
        "screening": "screen",
        "fit": "batman",
        "fitting": "batman",
        "vet": "triceratops",
        "vetting": "triceratops",
    }
    canonical = alias_map.get(normalized, normalized)
    for desc in _ENGINE_CATALOG:
        if desc.name == canonical:
            return get_engine_status(desc)
    return None


def check_engine(name: str) -> Tuple[bool, str]:
    """Validate runtime readiness of a named engine.

    Returns:
        Tuple of (is_ready: bool, message: str).
    """
    status = get_engine(name)
    if status is None:
        valid_names = ", ".join(d.name for d in _ENGINE_CATALOG)
        return False, f"Unknown engine '{name}'. Supported engines: {valid_names}"

    if not status.installed:
        return (
            False,
            f"Engine '{status.name}' ({status.module_name}) is not installed. Install with: pip install {status.module_name}",
        )

    ver_str = f" v{status.version}" if status.version else ""
    return True, f"Engine '{status.name}' ({status.capability}){ver_str} is installed and ready."


def _file_sha256(path: Path) -> str:
    """Compute standard SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


_RUNNABLE_ENGINES = {
    "bls",
    "screen",
    "batman",
    "localization",
    "activity",
    "sed",
    "dilution",
    "ttv",
    "phasecurve",
    "asteroseismology",
}


def _trusted_workspace(workspace: CandidateWorkspace) -> CandidateWorkspace:
    """Reload a workspace before an engine runner writes candidate-local artifacts."""
    trusted = load_candidate(workspace.repository_root, workspace.candidate_id)
    if workspace.path.resolve() != trusted.path.resolve():
        raise ValueError("engine runs require the registered candidate workspace path")
    return trusted


def _collect_input_artifacts(workspace: CandidateWorkspace) -> List[Dict[str, str]]:
    """Gather candidate-local input files and their SHA-256 hashes."""
    inputs: List[Dict[str, str]] = []
    
    # 1. Candidate metadata record
    cand_json = workspace.path / "candidate.json"
    if cand_json.is_file():
        inputs.append({
            "path": "candidate.json",
            "sha256": _file_sha256(cand_json),
            "role": "candidate_identity",
        })

    # 2. Transit ephemeris configs
    config_dir = workspace.path / "config"
    if config_dir.is_dir():
        for p in sorted(config_dir.rglob("*.json")):
            if p.is_file():
                rel = p.relative_to(workspace.path).as_posix()
                inputs.append({
                    "path": rel,
                    "sha256": _file_sha256(p),
                    "role": "transit_ephemeris_config",
                })

    # 3. Processed or raw light curves
    data_dir = workspace.path / "data"
    if data_dir.is_dir():
        for p in sorted(data_dir.rglob("*")):
            if p.is_file() and not p.name.endswith(".provenance.json"):
                rel = p.relative_to(workspace.path).as_posix()
                inputs.append({
                    "path": rel,
                    "sha256": _file_sha256(p),
                    "role": "photometric_input",
                })

    return inputs


def run_engine(
    workspace: CandidateWorkspace,
    engine_name: str,
    signal: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    """Execute a named analytical engine, preserving inputs, outputs, and manifest.

    Complies with schemas/engine-run.schema.json.
    Writes: candidate/<id>/runs/<engine>/<run_id>/engine-run.json
    """
    workspace = _trusted_workspace(workspace)
    engine_status = get_engine(engine_name)
    if engine_status is None:
        raise ValueError(f"Unknown engine '{engine_name}'.")

    if not engine_status.installed:
        raise RuntimeError(f"Engine '{engine_name}' ({engine_status.module_name}) is not installed.")

    engine_name = engine_status.name
    if engine_name not in _RUNNABLE_ENGINES:
        if engine_name == "triceratops":
            raise ValueError("TRICERATOPS must run through 'exonym vet' after the pre-vetting workflow.")
        raise ValueError(f"Engine '{engine_name}' has no candidate-local runner.")

    started_at = datetime.now(timezone.utc).isoformat()
    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f").lower()
    run_id = f"{timestamp_slug}-{engine_name}"

    runs_dir = workspace.path / "runs" / engine_name / run_id
    runs_dir.mkdir(parents=True)

    input_artifacts = _collect_input_artifacts(workspace)
    output_files: List[Path] = []
    failure_info: Optional[Dict[str, str]] = None
    status = "succeeded"

    try:
        if engine_name == "bls":
            from .search import run_bls_on_candidate
            out = run_bls_on_candidate(workspace, signal=signal, **kwargs)
            if out is not None:
                output_files.append(out)
        elif engine_name == "screen":
            from .screening import run_fixed_ephemeris_screen
            out = run_fixed_ephemeris_screen(workspace, signal=signal)
            if out is not None:
                output_files.append(out)
        elif engine_name == "batman":
            from .transit_fit import run_mcmc_transit_fit
            out = run_mcmc_transit_fit(workspace, signal=signal, **kwargs)
            if out is not None:
                output_files.append(out)
                suffix = f".{signal.lstrip('.')}" if signal else ""
                chain = workspace.path / "outputs" / f"mcmc_transit_fit_chain{suffix}.npy"
                if chain.is_file():
                    output_files.append(chain)
        elif engine_name == "sed":
            from .sed import run_sed_fit
            out = run_sed_fit(workspace)
            if out is not None:
                output_files.append(out)
        elif engine_name == "localization":
            from .localization import run_prf_localization
            out = run_prf_localization(workspace, **kwargs)
            if out is not None:
                output_files.append(out)
        elif engine_name == "activity":
            from .activity import run_stellar_activity
            out = run_stellar_activity(workspace)
            if out is not None:
                output_files.append(out)
        elif engine_name == "dilution":
            from .dilution import run_dilution_sensitivity
            out = run_dilution_sensitivity(workspace)
            if out is not None:
                output_files.append(out)
        elif engine_name == "ttv":
            from .ttv import run_ttv_analysis
            out = run_ttv_analysis(workspace, signal=signal)
            if out is not None:
                output_files.append(out)
        elif engine_name == "phasecurve":
            from .phasecurve import run_phase_curve_search
            out = run_phase_curve_search(workspace)
            if out is not None:
                output_files.append(out)
        elif engine_name == "asteroseismology":
            from .asteroseismology import run_asteroseismology
            out = run_asteroseismology(workspace, **kwargs)
            if out is not None:
                output_files.append(out)
        else:
            raise NotImplementedError(f"Engine '{engine_name}' runner is not yet configured.")
    except Exception as exc:
        status = "failed"
        failure_info = {
            "code": type(exc).__name__,
            "message": str(exc),
        }

    if status == "succeeded" and not output_files:
        status = "blocked"
        failure_info = {
            "code": "no-output-artifacts",
            "message": "The engine completed without producing a candidate-local output artifact.",
        }

    completed_at = datetime.now(timezone.utc).isoformat()

    output_artifacts: List[Dict[str, str]] = []
    for out_path in output_files:
        if out_path.is_file():
            try:
                rel_out = out_path.resolve().relative_to(workspace.path.resolve()).as_posix()
            except ValueError:
                status = "failed"
                failure_info = {
                    "code": "output-outside-workspace",
                    "message": "An engine returned an output outside the candidate workspace.",
                }
                output_artifacts = []
                break
            output_artifacts.append({
                "path": rel_out,
                "sha256": _file_sha256(out_path),
            })

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "engine": engine_name,
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "runtime": {
            "kind": "direct",
            "version": engine_status.version or "1.0.0",
            "executable": engine_status.module_name,
        },
        "inputs": input_artifacts,
        "outputs": output_artifacts,
    }
    if failure_info is not None:
        manifest["failure"] = failure_info

    manifest_path = runs_dir / "engine-run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def report_candidate_engines(workspace: CandidateWorkspace) -> List[Dict[str, Any]]:
    """Discover and summarize all recorded engine runs for a candidate workspace."""
    runs: List[Dict[str, Any]] = []
    runs_dir = workspace.path / "runs"
    if not runs_dir.is_dir():
        return runs

    for manifest_path in sorted(runs_dir.glob("*/*/engine-run.json")):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                runs.append({
                    "engine": data.get("engine"),
                    "run_id": data.get("run_id"),
                    "status": data.get("status"),
                    "started_at": data.get("started_at"),
                    "completed_at": data.get("completed_at"),
                    "inputs_count": len(data.get("inputs", [])),
                    "outputs_count": len(data.get("outputs", [])),
                    "path": str(manifest_path.relative_to(workspace.path)).replace("\\", "/"),
                })
        except Exception:
            continue

    return runs


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    """Read one JSON object, returning None for malformed or non-object content."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _producing_manifest(
    workspace: CandidateWorkspace,
    engine_name: str,
    artifact_path: Path,
) -> Optional[Tuple[Path, str]]:
    """Return the successful manifest that records the exact output artifact hash."""
    try:
        artifact_rel = artifact_path.resolve().relative_to(workspace.path.resolve()).as_posix()
    except ValueError:
        return None
    artifact_sha = _file_sha256(artifact_path)
    for manifest_path in sorted((workspace.path / "runs" / engine_name).glob("*/engine-run.json")):
        manifest = _load_json_object(manifest_path)
        if manifest is None:
            continue
        if (
            manifest.get("candidate_id") != workspace.candidate_id
            or manifest.get("engine") != engine_name
            or manifest.get("status") != "succeeded"
        ):
            continue
        outputs = manifest.get("outputs")
        if not isinstance(outputs, list):
            continue
        for output in outputs:
            if isinstance(output, dict) and output.get("path") == artifact_rel and output.get("sha256") == artifact_sha:
                return manifest_path, artifact_sha
    return None


def _triage_record(
    workspace: CandidateWorkspace,
    engine_name: str,
    artifact_path: Path,
    status: str,
    reason: str,
) -> Dict[str, str]:
    """Build one triage record only when its scientific artifact is traceable."""
    producer = _producing_manifest(workspace, engine_name, artifact_path)
    if producer is None:
        return {
            "engine": engine_name,
            "status": "blocked",
            "reason": "No successful engine manifest records the exact output artifact.",
        }
    manifest_path, artifact_sha = producer
    return {
        "engine": engine_name,
        "run_manifest_path": manifest_path.relative_to(workspace.path).as_posix(),
        "run_manifest_sha256": _file_sha256(manifest_path),
        "artifact_path": artifact_path.relative_to(workspace.path).as_posix(),
        "artifact_sha256": artifact_sha,
        "status": status,
        "reason": reason,
    }


def run_automated_triage(
    workspace: CandidateWorkspace,
    policy_id: str = "default-pre-vetting-triage",
    policy_version: str = "1.0.0",
) -> Path:
    """Aggregate pre-vetting findings into an automated triage decision.

    Evaluates only existing, manifest-backed fixed-ephemeris and PRF localization
    outputs. Missing, malformed, or unprovenanced evidence is blocked rather than
    inferred as a scientific pass.

    Output is strictly validated against schemas/automated-triage.schema.json
    and written to candidate/<id>/decisions/automated_triage.json.
    """
    workspace = _trusted_workspace(workspace)
    records: List[Dict[str, str]] = []

    # 1. Screening Evaluation
    screen_path = workspace.path / "outputs" / "fixed_ephemeris_screening.json"
    if not screen_path.is_file():
        # probe for generic screening outputs
        for p in sorted((workspace.path / "outputs").glob("fixed_ephemeris_screen*.json")):
            screen_path = p
            break

    if screen_path.is_file():
        screen_data = _load_json_object(screen_path)
        screen_obj = screen_data.get("screen", screen_data) if screen_data else None
        odd_even = screen_obj.get("odd_even") if isinstance(screen_obj, dict) else None
        consistent = odd_even.get("consistent_at_threshold") if isinstance(odd_even, dict) else None
        if consistent is False:
            records.append(_triage_record(
                workspace, "screen", screen_path, "review-required",
                "Significant odd-even transit depth asymmetry requires human review.",
            ))
        elif consistent is True:
            records.append(_triage_record(
                workspace, "screen", screen_path, "pass",
                "Odd-even depths are statistically consistent.",
            ))
        else:
            records.append({
                "engine": "screen",
                "status": "blocked",
                "reason": "Screening output lacks a conclusive odd-even consistency result.",
            })

    # 2. Centroid PRF Localization Evaluation
    prf_path = workspace.path / "outputs" / "prf_localization_report.json"
    if prf_path.is_file():
        prf_data = _load_json_object(prf_path)
        is_on_target = prf_data.get("on_target") if prf_data else None
        if is_on_target is False:
            records.append(_triage_record(
                workspace, "localization", prf_path, "review-required",
                "Transit flux deficit centroid is significantly offset from the target.",
            ))
        elif is_on_target is True:
            records.append(_triage_record(
                workspace, "localization", prf_path, "pass",
                "Centroid is consistent with an on-target signal.",
            ))
        else:
            records.append({
                "engine": "localization",
                "status": "blocked",
                "reason": "Localization output lacks a conclusive on-target result.",
            })

    if not records:
        records.append({
            "engine": "screen",
            "status": "blocked",
            "reason": "No manifest-backed screening or localization evidence is available.",
        })

    statuses = {record["status"] for record in records}
    if "blocked" in statuses:
        overall_status = "blocked"
    elif "review-required" in statuses:
        overall_status = "review-required"
    else:
        overall_status = "pass"

    decisions_dir = workspace.path / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)

    triage_payload: Dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy_id": policy_id,
        "policy_version": policy_version,
        "status": overall_status,
        "records": records,
    }

    triage_path = decisions_dir / "automated_triage.json"
    triage_path.write_text(json.dumps(triage_payload, indent=2) + "\n", encoding="utf-8")
    return triage_path
