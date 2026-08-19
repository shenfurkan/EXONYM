"""Candidate-local pre-TRICERATOPS statistical-vetting safeguards.

This module records what each pre-vetting diagnostic can and cannot establish.
It only routes work to pass, review-required, or blocked; it never writes a
scientific claim or assigns a candidate disposition.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .workspace import CandidateWorkspace, validate_signal_suffix


DIAGNOSTIC_NAMES = ("screening", "archive", "localization", "activity", "dilution")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _finite_number(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _artifact(workspace: CandidateWorkspace, path: Path) -> Optional[Dict[str, str]]:
    if not path.is_file():
        return None
    return {
        "path": path.relative_to(workspace.path).as_posix(),
        "sha256": _sha256(path),
    }


def _record(
    name: str,
    status: str,
    reason: str,
    artifact: Optional[Dict[str, str]],
    calibration_source: str,
    input_representation: str,
    score_name: str,
    score_value: Optional[float],
    score_unit: str,
    uncertainty: Optional[float],
    applicability_limits: List[str],
) -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "artifact": artifact,
        "calibration_source": calibration_source,
        "input_representation": input_representation,
        "score": {"name": score_name, "value": score_value, "unit": score_unit},
        "uncertainty": uncertainty,
        "applicability_limits": applicability_limits,
    }


def _missing_record(name: str, path: Path) -> Dict[str, Any]:
    return _record(
        name,
        "blocked",
        "Required diagnostic artifact is missing or unreadable: {0}".format(path.name),
        None,
        "No calibration can be assessed because the required artifact is unavailable.",
        "No candidate-local input representation is available.",
        "unavailable",
        None,
        "dimensionless",
        None,
        ["TRICERATOPS is not run without this preceding diagnostic."],
    )


def _screening_record(workspace: CandidateWorkspace, signal: Optional[str]) -> Dict[str, Any]:
    suffix = ".{0}".format(signal.lstrip(".")) if signal else ""
    path = workspace.path / "outputs" / "fixed_ephemeris_screen{0}.json".format(suffix)
    data = _load_object(path)
    artifact = _artifact(workspace, path)
    if data is None or artifact is None:
        return _missing_record("screening", path)
    screen = data.get("screen")
    odd_even = screen.get("odd_even") if isinstance(screen, dict) else None
    z = _finite_number(odd_even.get("z")) if isinstance(odd_even, dict) else None
    consistent = odd_even.get("consistent_at_threshold") if isinstance(odd_even, dict) else None
    threshold = _finite_number(odd_even.get("consistency_threshold_sigma")) if isinstance(odd_even, dict) else None
    primary = screen.get("primary") if isinstance(screen, dict) else None
    half_phase = screen.get("half_phase_control") if isinstance(screen, dict) else None
    double_period = screen.get("double_period_hypothesis") if isinstance(screen, dict) else None
    alternating = double_period.get("alternating_event") if isinstance(double_period, dict) else None
    control_significances = [
        _finite_number(item.get("depth_significance_sigma"))
        for item in (half_phase, alternating)
        if isinstance(item, dict)
    ]
    if data.get("candidate_id") != workspace.candidate_id or data.get("source") != "candidate-data":
        status, reason = "blocked", "Screening is not a candidate-data artifact owned by this workspace."
    elif not isinstance(primary, dict) or primary.get("status") != "measured" or threshold is None:
        status, reason = "blocked", "Screening lacks a measured primary event or recorded comparison threshold."
    elif consistent is not True or any(
        value is not None and abs(value) >= threshold for value in control_significances
    ):
        status, reason = "review-required", "Odd-even screening is unresolved or disagrees with the declared transit."
    else:
        status, reason = "pass", "Odd-even depths are consistent at the recorded diagnostic threshold."
    return _record(
        "screening", status, reason, artifact,
        "Scatter-based odd-even depth uncertainty; no population false-alarm calibration.",
        "Candidate light curve evaluated at the declared period, epoch, and duration.",
        "odd_even_z", z, "sigma", None,
        ["Correlated noise and half-phase/double-period interpretation require human review."],
    )


def _archive_record(workspace: CandidateWorkspace) -> Dict[str, Any]:
    path = workspace.path / "outputs" / "archival_vetting_report.json"
    data = _load_object(path)
    artifact = _artifact(workspace, path)
    if data is None or artifact is None:
        return _missing_record("archive", path)
    gaia = data.get("gaia_astrometry")
    assessment = data.get("scientific_assessment")
    binary = assessment.get("1_is_hidden_binary") if isinstance(assessment, dict) else None
    crowding = assessment.get("2_has_nearby_contaminants") if isinstance(assessment, dict) else None
    ruwe = _finite_number(gaia.get("ruwe")) if isinstance(gaia, dict) else None
    if data.get("candidate_id") != workspace.candidate_id or not isinstance(gaia, dict):
        status, reason = "blocked", "Archive report is not a complete candidate-owned Gaia diagnostic."
    elif gaia.get("validated") is not True or gaia.get("query_status") != "ok":
        status, reason = "blocked", "Gaia target identity or archive query is unsuitable for automated routing."
    elif not isinstance(binary, dict) or not isinstance(crowding, dict):
        status, reason = "blocked", "Archive report lacks the recorded binarity and crowding assessments."
    elif binary.get("answer") is True or crowding.get("answer") is True:
        status, reason = "review-required", "Archive context indicates possible multiplicity or nearby contaminating sources."
    else:
        status, reason = "pass", "Validated archive context has no recorded multiplicity or crowding warning."
    return _record(
        "archive", status, reason, artifact,
        "Validated Gaia target match and archive query status; catalog context is not an FPP calibration.",
        "Candidate coordinates, Gaia neighborhood, and archive follow-up metadata.",
        "gaia_ruwe", ruwe, "dimensionless", None,
        ["A catalog non-detection does not exclude contaminants or establish novelty."],
    )


def _localization_record(workspace: CandidateWorkspace) -> Dict[str, Any]:
    path = workspace.path / "outputs" / "prf_localization_results.json"
    data = _load_object(path)
    artifact = _artifact(workspace, path)
    if data is None or artifact is None:
        return _missing_record("localization", path)
    summary = data.get("summary")
    conclusion = summary.get("conclusion") if isinstance(summary, dict) else None
    ratio = (
        _finite_number(summary.get("median_target_to_other_difference_ratio"))
        if isinstance(summary, dict)
        else None
    )
    if data.get("source") != "candidate-data" or not isinstance(summary, dict):
        status, reason = "blocked", "Localization lacks candidate-data PRF inputs or a usable summary."
    elif data.get("calibration_status") != "uncalibrated":
        status, reason = "blocked", "Localization declares an unsupported calibration status."
    else:
        status, reason = "review-required", "Current localization is an uncalibrated PRF/scene diagnostic and cannot route a source automatically."
    return _record(
        "localization", status, reason, artifact,
        "Uncalibrated Gaussian-PRF difference-image screening with no scene-level FPP.",
        "Absolute in-transit versus out-of-transit TPF difference images with validated Gaia neighbors.",
        "target_to_max_other_difference_ratio", ratio, "dimensionless", None,
        ["No mission-calibrated PRF, scene model, or injection benchmark supports an automated transit-source assignment."],
    )


def _activity_record(workspace: CandidateWorkspace, screen_record: Dict[str, Any], signal: Optional[str]) -> Dict[str, Any]:
    path = workspace.path / "outputs" / "stellar_activity_results.json"
    data = _load_object(path)
    artifact = _artifact(workspace, path)
    if data is None or artifact is None:
        return _missing_record("activity", path)
    rotation = _finite_number(data.get("rotation_period_days"))
    spread = _finite_number(data.get("rotation_period_std_days"))
    screen_path = screen_record.get("artifact", {}).get("path") if isinstance(screen_record.get("artifact"), dict) else None
    screen_data = _load_object(workspace.path / screen_path) if screen_path else None
    ephemeris = screen_data.get("ephemeris") if isinstance(screen_data, dict) else None
    transit = _finite_number(ephemeris.get("period_days")) if isinstance(ephemeris, dict) else None
    aliases = False
    if rotation is not None and transit is not None and transit > 0.0:
        aliases = any(math.isclose(rotation, factor * transit, rel_tol=0.02) for factor in (0.5, 1.0, 2.0))
    if data.get("source") != "candidate-data" or rotation is None:
        status, reason = "blocked", "Activity analysis lacks a finite candidate-data rotation diagnostic."
    elif aliases:
        status, reason = "review-required", "The activity peak is compatible with the transit period or a simple harmonic."
    else:
        status, reason = "review-required", "Activity is an uncalibrated correlated-noise diagnostic and cannot automatically clear a transit signal."
    return _record(
        "activity", status, reason, artifact,
        "Analytic GLS false-alarm probability; no correlated-noise population calibration.",
        "Transit-masked candidate light curve segmented for GLS periodograms.",
        "rotation_period", rotation, "days", spread,
        ["Window functions, evolving spots, and aliases require manual interpretation."],
    )


def _dilution_record(workspace: CandidateWorkspace) -> Dict[str, Any]:
    path = workspace.path / "outputs" / "dilution_sensitivity_results.json"
    data = _load_object(path)
    artifact = _artifact(workspace, path)
    if data is None or artifact is None:
        return _missing_record("dilution", path)
    depth = data.get("depth_stability")
    contamination = data.get("contamination")
    stability = _finite_number(depth.get("max_variation_relative_to_median")) if isinstance(depth, dict) else None
    contamination_factor = _finite_number(contamination.get("contamination_factor")) if isinstance(contamination, dict) else None
    if data.get("source") != "candidate-data" or not isinstance(depth, dict) or not isinstance(contamination, dict):
        status, reason = "blocked", "Dilution analysis lacks candidate-data aperture and contamination inputs."
    elif contamination.get("availability") != "available":
        status, reason = "blocked", "Dilution contamination inputs are unavailable or unsuitable for automated routing."
    elif depth.get("interpretation") != "stable":
        status, reason = "review-required", "Transit depth is aperture-sensitive and requires human review."
    else:
        status, reason = "pass", "Recorded aperture depths are stable with available contamination context."
    return _record(
        "dilution", status, reason, artifact,
        "Aperture-depth comparison and Gaia G-band flux-ratio sensitivity bound.",
        "Candidate TPF aperture light curves and validated archival neighbor photometry.",
        "contamination_factor", contamination_factor, "dimensionless", stability,
        ["Gaia G-band flux ratios are not exact TESS-band dilution corrections."],
    )


def build_statistical_vetting_evidence(workspace: CandidateWorkspace, signal: Optional[str] = None) -> Path:
    """Write a complete pre-vetting evidence representation without a claim."""
    signal = validate_signal_suffix(signal)
    screening = _screening_record(workspace, signal)
    diagnostics = [
        screening,
        _archive_record(workspace),
        _localization_record(workspace),
        _activity_record(workspace, screening, signal),
        _dilution_record(workspace),
    ]
    statuses = {record["status"] for record in diagnostics}
    overall = "blocked" if "blocked" in statuses else "review-required" if "review-required" in statuses else "pass"
    suffix = ".{0}".format(signal.lstrip(".")) if signal else ""
    payload = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signal": signal,
        "status": overall,
        "diagnostics": diagnostics,
    }
    path = workspace.path / "outputs" / "statistical_vetting_evidence{0}.json".format(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def record_decisive_rejection(workspace: CandidateWorkspace, reason: str, evidence_path: str) -> Path:
    """Record an evidence-backed reason that makes TRICERATOPS inapplicable."""
    candidate_root = workspace.path.resolve()
    evidence = (candidate_root / evidence_path).resolve()
    try:
        relative = evidence.relative_to(candidate_root)
    except ValueError:
        raise ValueError("rejection evidence must be inside the candidate workspace")
    if not evidence.is_file():
        raise FileNotFoundError("rejection evidence does not exist: {0}".format(evidence_path))
    if not reason.strip():
        raise ValueError("decisive rejection reason must not be empty")
    payload = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "decisive-rejection",
        "reason": reason.strip(),
        "evidence": {"path": relative.as_posix(), "sha256": _sha256(evidence)},
    }
    path = workspace.path / "decisions" / "decisive_rejection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _require_real_data_prerequisites(workspace: CandidateWorkspace, signal: Optional[str]) -> None:
    """Require the ordered candidate-data outputs that must precede TRICERATOPS."""
    suffix = ".{0}".format(signal.lstrip(".")) if signal else ""
    required = (
        ("search", Path("outputs") / "bls_search_results{0}.json".format(suffix), "source", "candidate-data"),
        ("search manifest", Path("outputs") / "bls_search_manifest{0}.json".format(suffix), "source", "candidate-data"),
        ("screen", Path("outputs") / "fixed_ephemeris_screen{0}.json".format(suffix), "source", "candidate-data"),
        ("archive", Path("outputs") / "archival_vetting_report.json", "candidate_id", workspace.candidate_id),
        ("localization", Path("outputs") / "prf_localization_results.json", "source", "candidate-data"),
        ("activity", Path("outputs") / "stellar_activity_results.json", "source", "candidate-data"),
        ("asteroseismology", Path("outputs") / "asteroseismic_results.json", "source", "candidate-data"),
        ("SED", Path("outputs") / "sed_fit_results.json", "source", "candidate-data"),
        ("dilution", Path("outputs") / "dilution_sensitivity_results.json", "source", "candidate-data"),
        ("transit fit", Path("outputs") / "mcmc_transit_fit{0}.json".format(suffix), "source", "candidate-data"),
        ("timing", Path("outputs") / "ttv_analysis_results{0}.json".format(suffix), "source", "candidate-data"),
        ("phase curve", Path("outputs") / "phase_curve_results.json", "source", "candidate-data"),
    )
    candidate_root = workspace.path.resolve()
    missing: List[str] = []
    for name, relative, field, expected in required:
        path = (candidate_root / relative).resolve()
        try:
            path.relative_to(candidate_root)
        except ValueError:
            missing.append(name)
            continue
        data = _load_object(path)
        if data is None or data.get(field) != expected:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "TRICERATOPS requires real candidate-data prerequisite outputs: {0}.".format(
                ", ".join(missing)
            )
        )


def require_vetting_readiness(workspace: CandidateWorkspace, signal: Optional[str] = None) -> Path:
    """Stop before Monte Carlo unless all required diagnostics pass automated routing."""
    signal = validate_signal_suffix(signal)
    rejection = _load_object(workspace.path / "decisions" / "decisive_rejection.json")
    if rejection is not None and rejection.get("candidate_id") == workspace.candidate_id and rejection.get("status") == "decisive-rejection":
        raise RuntimeError("TRICERATOPS is prohibited by the candidate-local decisive rejection record.")
    # Preserve a candidate-local blocked evidence record even when the command
    # cannot proceed. This report does not run an engine or create a claim.
    evidence_path = build_statistical_vetting_evidence(workspace, signal=signal)
    # Establish candidate-data provenance before automated triage. Otherwise a
    # malformed or synthetic artifact could alter the routed decision before the
    # real-data gate rejects it.
    _require_real_data_prerequisites(workspace, signal)
    # Keep the human-visible automated triage decision synchronized with the
    # candidate-data evidence that passed the prerequisite gate. The import is
    # local to avoid a module cycle because ``triage`` itself builds evidence.
    from .engines import run_automated_triage

    run_automated_triage(workspace, signal=signal)
    evidence = _load_object(evidence_path)
    if evidence is None or evidence.get("status") != "pass":
        status = evidence.get("status") if evidence else "blocked"
        raise RuntimeError(
            "TRICERATOPS requires passing candidate-local statistical vetting evidence; current routing is {0}.".format(status)
        )
    _require_real_data_prerequisites(workspace, signal)
    return evidence_path
