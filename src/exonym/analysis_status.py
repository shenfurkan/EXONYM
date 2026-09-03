"""Candidate-local analysis coverage and applicability records.

The record distinguishes a completed diagnostic from one that could not be
run because its candidate-owned inputs were unavailable.  It is operational
coverage evidence only and never establishes a planetary claim.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .workspace import CandidateWorkspace


_MIST_BC_SED_STATUS = "exploratory-mist-v1.2-bolometric-correction-diagnostic"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage(
    workspace: CandidateWorkspace,
    name: str,
    output: Optional[str],
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    if output is not None:
        path = workspace.path / output
        if path.is_file():
            return {
                "name": name,
                "status": "succeeded",
                "evidence": [{"path": output, "sha256": _sha256(path)}],
                "reason": "Candidate-local analysis output is present and hash recorded.",
            }
    return {
        "name": name,
        "status": "unavailable",
        "evidence": [],
        "reason": reason or "Required candidate-local input or output is unavailable.",
    }


def _mist_bc_sed_stage(workspace: CandidateWorkspace) -> Dict[str, Any]:
    """Report SED coverage only for the current claim-ineligible MIST contract."""
    output = "outputs/sed_fit_results.json"
    path = workspace.path / output
    unavailable = {
        "name": "sed",
        "status": "unavailable",
        "evidence": [],
        "reason": "No current MIST v1.2 diagnostic contract output is available.",
    }
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(workspace.path.resolve())
    ):
        return unavailable
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return unavailable
    if not isinstance(payload, dict):
        return unavailable
    if (
        payload.get("schema_version") != 2
        or payload.get("candidate_id") != workspace.candidate_id
        or payload.get("source") != "candidate-data"
        or payload.get("scientific_status") != _MIST_BC_SED_STATUS
        or payload.get("validation_eligible") is not False
        or payload.get("claim_eligible") is not False
    ):
        return unavailable
    input_provenance = payload.get("input_provenance")
    if not isinstance(input_provenance, Mapping):
        return unavailable
    artifacts = input_provenance.get("input_artifacts")
    manifest_artifact = input_provenance.get("input_manifest_artifact")
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or not isinstance(manifest_artifact, Mapping)
        or not all(isinstance(artifact, Mapping) for artifact in artifacts)
    ):
        return unavailable
    for artifact in [manifest_artifact, *artifacts]:
        relative = artifact.get("path")
        expected_sha256 = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha256, str):
            return unavailable
        artifact_path = workspace.path / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not artifact_path.is_file()
            or artifact_path.is_symlink()
            or not artifact_path.resolve().is_relative_to(workspace.path.resolve())
            or _sha256(artifact_path) != expected_sha256
        ):
            return unavailable
    return {
        "name": "sed",
        "status": "succeeded",
        "evidence": [{"path": output, "sha256": _sha256(path)}],
        "reason": "Candidate-owned MIST v1.2 bolometric-correction diagnostic output is present and hash recorded.",
    }


def build_analysis_status(workspace: CandidateWorkspace) -> Path:
    """Write the candidate-local analysis coverage record."""
    raw_dir = workspace.path / "data" / "raw"
    light_curves = sorted(raw_dir.glob("*_lc.fits"))
    raw_status = "succeeded" if light_curves else "unavailable"
    raw_stage = {
        "name": "acquisition",
        "status": raw_status,
        "evidence": (
            [{"path": p.relative_to(workspace.path).as_posix(), "sha256": _sha256(p)} for p in light_curves]
            if light_curves
            else []
        ),
        "reason": (
            "Candidate-local TESS light curves are present and hash recorded."
            if light_curves
            else "No candidate-local TESS light-curve product was available after acquisition attempts."
        ),
    }
    stages = [
        raw_stage,
        _stage(workspace, "detrending", "data/processed/detrended-running-median.npz"),
        _stage(workspace, "search", "outputs/bls_search_results.json", "No valid BLS result is available."),
        _stage(workspace, "screening", "outputs/fixed_ephemeris_screen.json", "No valid fixed-ephemeris screen is available."),
        _stage(workspace, "archive", "outputs/archival_vetting_report.json"),
        _stage(workspace, "localization", "outputs/prf_localization_results.json", "TPF/PRF localization was not applicable or unavailable."),
        _stage(workspace, "activity", "outputs/stellar_activity_results.json"),
        _mist_bc_sed_stage(workspace),
        _stage(workspace, "dilution", "outputs/dilution_sensitivity_results.json"),
        _stage(workspace, "transit_fit", "outputs/mcmc_transit_fit.json", "Candidate-derived stellar parameters or fit inputs are unavailable."),
        _stage(workspace, "ttv", "outputs/ttv_analysis_results.json", "Candidate-derived stellar parameters or complete timing inputs are unavailable."),
        _stage(workspace, "phase_curve", "outputs/phase_curve_results.json"),
        _stage(workspace, "triage", "decisions/automated_triage.json"),
        _stage(workspace, "trex", "decisions/triceratops_vetting_decision.json"),
    ]
    complete = all(item["status"] == "succeeded" for item in stages)
    payload = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": "complete" if complete else "partial",
        "claim_eligible": False,
        "scientific_interpretation": "Coverage record only; unavailable stages are not scientific null results.",
        "stages": stages,
    }
    path = workspace.path / "decisions" / "analysis_completion.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path
