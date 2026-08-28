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
from typing import Any, Dict, List, Optional

from .workspace import CandidateWorkspace


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
        _stage(workspace, "sed", "outputs/sed_fit_results.json", "Candidate-owned broadband photometry or stellar priors are unavailable."),
        _stage(workspace, "dilution", "outputs/dilution_sensitivity_results.json"),
        _stage(workspace, "transit_fit", "outputs/mcmc_transit_fit.json", "Candidate-derived stellar parameters or fit inputs are unavailable."),
        _stage(workspace, "ttv", "outputs/ttv_analysis_results.json", "Candidate-derived stellar parameters or complete timing inputs are unavailable."),
        _stage(workspace, "phase_curve", "outputs/phase_curve_results.json"),
        _stage(workspace, "triage", "decisions/automated_triage.json"),
        _stage(workspace, "triceratops", "decisions/triceratops_vetting_decision.json"),
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
