"""Candidate-local routing safeguards before statistical vetting.

This module translates hash-bound screening, archive, localization, activity,
and dilution records into transparent routing states. The aggregate state is a
conservative maximum over required diagnostics: a missing or unsuitable input
must remain visible rather than being averaged into an apparently clean result.

Scientific boundary:
    The routing output records what each diagnostic can and cannot establish.
    It never writes a scientific claim, assigns a disposition, calibrates a
    false-positive probability, or substitutes for human review.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .inputs import BTJD_TIME_SYSTEM, _read_json, is_manifest_bound_bls_result
from .workspace import CandidateWorkspace, validate_signal_suffix


# ASTROPHYSICAL_HEURISTIC: Below a detector-pixel-scale archive cone, a
# crowding result cannot exclude sources relevant to the aperture.
MINIMUM_CROWDING_SEARCH_RADIUS_ARCSEC: float = 21.0


DIAGNOSTIC_NAMES = (
    "screening",
    "archive",
    "localization",
    "activity",
    "dilution",
    "radial-velocity",
)

# A descriptive model-comparison threshold only. Crossing it routes the
# candidate-owned RV series to human review; it never establishes a companion
# or a scientific claim.
RV_REVIEW_DELTA_BIC = 10.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> Optional[Dict[str, Any]]:
    return _read_json(path)


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


def _screening_matches_detected_bls(
    workspace: CandidateWorkspace, ephemeris: object, signal: Optional[str]
) -> bool:
    """Require screening to use the exact current hash-bound BLS ephemeris."""
    if not isinstance(ephemeris, dict):
        return False
    suffix = ".{0}".format(signal.lstrip(".")) if signal else ""
    result_path = workspace.path / "outputs" / "bls_search_results{0}.json".format(suffix)
    result = _load_object(result_path)
    if result is None or not is_manifest_bound_bls_result(workspace, result_path, result, signal):
        return False
    pairs = (
        (ephemeris.get("period_days"), result.get("best_period")),
        (ephemeris.get("epoch_btjd"), result.get("best_epoch")),
        (ephemeris.get("duration_hours"), result.get("best_duration_hours")),
    )
    for used, measured in pairs:
        used_value = _finite_number(used)
        measured_value = _finite_number(measured)
        if (
            used_value is None
            or measured_value is None
            or not math.isclose(used_value, measured_value, rel_tol=1e-12, abs_tol=1e-12)
        ):
            return False
    return True


def _screening_record(workspace: CandidateWorkspace, signal: Optional[str]) -> Dict[str, Any]:
    suffix = ".{0}".format(signal.lstrip(".")) if signal else ""
    path = workspace.path / "outputs" / "fixed_ephemeris_screen{0}.json".format(suffix)
    data = _load_object(path)
    artifact = _artifact(workspace, path)
    if data is None or artifact is None:
        return _missing_record("screening", path)
    screen = data.get("screen")
    ephemeris = data.get("ephemeris")
    odd_even = screen.get("odd_even") if isinstance(screen, dict) else None
    z = _finite_number(odd_even.get("z")) if isinstance(odd_even, dict) else None
    consistent = odd_even.get("consistent_at_threshold") if isinstance(odd_even, dict) else None
    threshold = _finite_number(odd_even.get("consistency_threshold_sigma")) if isinstance(odd_even, dict) else None
    primary = screen.get("primary") if isinstance(screen, dict) else None
    half_phase = screen.get("half_phase_control") if isinstance(screen, dict) else None
    half_significance = _finite_number(half_phase.get("depth_significance_sigma")) if isinstance(half_phase, dict) else None
    primary_depth = _finite_number(primary.get("depth_ppm")) if isinstance(primary, dict) else None
    primary_significance = _finite_number(primary.get("depth_significance_sigma")) if isinstance(primary, dict) else None
    double_period = screen.get("double_period_hypothesis") if isinstance(screen, dict) else None
    double_primary = double_period.get("primary") if isinstance(double_period, dict) else None
    alternating = double_period.get("alternating_event") if isinstance(double_period, dict) else None
    alternating_significance = (
        _finite_number(alternating.get("depth_significance_sigma"))
        if isinstance(alternating, dict)
        else None
    )
    complete_controls = (
        isinstance(primary, dict)
        and primary.get("status") == "measured"
        and primary_depth is not None
        and primary_depth > 0.0
        and primary_significance is not None
        and primary_significance > 0.0
        and isinstance(odd_even, dict)
        and odd_even.get("status") == "measured"
        and z is not None
        and threshold is not None
        and threshold > 0.0
        and isinstance(consistent, bool)
        and isinstance(half_phase, dict)
        and half_phase.get("status") == "measured"
        and half_significance is not None
        and isinstance(double_primary, dict)
        and double_primary.get("status") == "measured"
        and isinstance(alternating, dict)
        and alternating.get("status") == "measured"
        and alternating_significance is not None
    )
    if data.get("candidate_id") != workspace.candidate_id or data.get("source") != "candidate-data":
        status, reason = "blocked", "Screening is not a candidate-data artifact owned by this workspace."
    elif not isinstance(ephemeris, dict) or ephemeris.get("epoch_time_system") != BTJD_TIME_SYSTEM:
        status, reason = "blocked", "Screening does not declare a BTJD_TDB ephemeris."
    elif not _screening_matches_detected_bls(workspace, ephemeris, signal):
        status, reason = "blocked", "Screening ephemeris is not the current hash-bound BLS result."
    elif not complete_controls:
        status, reason = "blocked", "Screening lacks complete measured primary, odd-even, half-phase, or doubled-period controls."
    elif (
        consistent is not True
        or abs(half_significance) >= threshold
        or abs(alternating_significance) >= threshold
    ):
        status, reason = "review-required", "Odd-even screening is unresolved or secondary eclipse detected."
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
    search_radius = _finite_number(gaia.get("search_radius_arcsec")) if isinstance(gaia, dict) else None
    nearby_count = gaia.get("nearby_sources_count") if isinstance(gaia, dict) else None
    binary_ruwe = _finite_number(binary.get("ruwe")) if isinstance(binary, dict) else None
    crowding_radius = _finite_number(crowding.get("search_radius_arcsec")) if isinstance(crowding, dict) else None
    if data.get("candidate_id") != workspace.candidate_id or not isinstance(gaia, dict):
        status, reason = "blocked", "Archive report is not a complete candidate-owned Gaia diagnostic."
    elif gaia.get("validated") is not True or gaia.get("query_status") != "ok":
        status, reason = "blocked", "Gaia target identity or archive query is unsuitable for automated routing."
    elif not isinstance(binary, dict) or not isinstance(crowding, dict):
        status, reason = "blocked", "Archive report lacks the recorded binarity and crowding assessments."
    elif (
        ruwe is None
        or ruwe <= 0.0
        or binary_ruwe is None
        or not math.isclose(binary_ruwe, ruwe, rel_tol=1e-9, abs_tol=1e-12)
        or not isinstance(binary.get("answer"), bool)
        or not isinstance(gaia.get("suspected_binary"), bool)
        or binary.get("answer") is not gaia.get("suspected_binary")
        or not isinstance(nearby_count, int)
        or nearby_count < 1
        or search_radius is None
        or search_radius < MINIMUM_CROWDING_SEARCH_RADIUS_ARCSEC
        or crowding_radius is None
        or not math.isclose(crowding_radius, search_radius, rel_tol=1e-9, abs_tol=1e-12)
        or crowding.get("search_radius_sufficient_for_crowding") is not True
        or not isinstance(crowding.get("answer"), bool)
        or crowding.get("answer") != (nearby_count > 1)
    ):
        status, reason = "blocked", "Archive report lacks complete validated RUWE or sufficient-radius crowding evidence."
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
    modeled_competitor_count = (
        _finite_number(summary.get("sectors_with_competing_sources_modeled"))
        if isinstance(summary, dict)
        else None
    )
    if data.get("source") != "candidate-data" or not isinstance(summary, dict):
        status, reason = "blocked", "Localization lacks candidate-data PRF inputs or a usable summary."
    elif data.get("calibration_status") != "uncalibrated":
        status, reason = "blocked", "Localization declares an unsupported calibration status."
    elif (
        ratio is None
        and modeled_competitor_count is not None
        and modeled_competitor_count > 0.0
    ):
        status, reason = (
            "blocked",
            "Localization modeled competing sources but lacks a finite "
            "target-to-strongest-other amplitude ratio.",
        )
    elif ratio is None and modeled_competitor_count == 0.0:
        status, reason = (
            "review-required",
            "Uncalibrated localization modeled no competing sources, so no "
            "target-to-other amplitude ratio is available; human review is required.",
        )
    elif ratio is None:
        status, reason = (
            "review-required",
            "Uncalibrated localization lacks a reported competing-source amplitude "
            "ratio and requires human review.",
        )
    # This is target amplitude divided by the strongest other-source amplitude;
    # only values strictly above one favor the target.
    elif ratio > 1.0:
        status, reason = (
            "review-required",
            "Uncalibrated localization is target-favored by the difference-image "
            "amplitude ratio but requires human review.",
        )
    else:
        status, reason = (
            "review-required",
            "Uncalibrated localization is competitor-favored or tied by the "
            "difference-image amplitude ratio but requires human review.",
        )
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
    elif transit is None or transit <= 0.0:
        status, reason = (
            "blocked",
            "Activity alias triage requires a finite screening ephemeris transit period.",
        )
    elif aliases:
        status, reason = "review-required", "The activity peak is compatible with the transit period or a simple harmonic."
    else:
        status, reason = "review-required", "Activity period has no simple transit alias but requires human review."
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
    elif (
        contamination_factor is None
        or contamination_factor < 0.0
        or stability is None
        or stability < 0.0
        or depth.get("interpretation") not in ("stable", "aperture-sensitive")
    ):
        status, reason = "blocked", "Dilution analysis lacks a complete aperture-depth stability measurement."
    elif depth.get("interpretation") == "stable":
        status, reason = "pass", "Recorded aperture depths are stable with available contamination context."
    else:
        status, reason = "review-required", "Transit depth is aperture-sensitive and requires human review."
    return _record(
        "dilution", status, reason, artifact,
        "Aperture-depth comparison and Gaia G-band flux-ratio sensitivity bound.",
        "Candidate TPF aperture light curves and validated archival neighbor photometry.",
        "contamination_factor", contamination_factor, "dimensionless", stability,
        ["Gaia G-band flux ratios are not exact TESS-band dilution corrections."],
    )


def _radial_velocity_record(workspace: CandidateWorkspace) -> Optional[Dict[str, Any]]:
    """Route an RV fit only when the workspace actually contains RV data.

    RV remains optional for the broad workflow. Once a candidate owner adds
    canonical observations, however, a missing, stale, or malformed fit must
    stay visible in automated triage rather than leaving the RV path unused.
    """
    observations_path = workspace.path / "data" / "external" / "radial_velocity_observations.json"
    report_path = workspace.path / "outputs" / "rv_keplerian_fit.json"
    if not observations_path.is_file() and not report_path.is_file():
        return None

    report = _load_object(report_path)
    artifact = _artifact(workspace, report_path)
    if not observations_path.is_file() or report is None or artifact is None:
        return _record(
            "radial-velocity",
            "blocked",
            "Candidate-local RV observations require a current readable Keplerian comparison report.",
            artifact,
            "No RV model-comparison calibration is available until the canonical observations are fit.",
            "Canonical BJD_TDB radial-velocity observations and an optional activity indicator.",
            "delta_bic_constant_minus_keplerian",
            None,
            "BIC",
            None,
            ["RV observations are present but no current model-comparison artifact is available."],
        )

    try:
        from .radial_velocity import load_radial_velocity_observations

        load_radial_velocity_observations(workspace)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        return _record(
            "radial-velocity",
            "blocked",
            "Canonical RV observations cannot be revalidated: {0}".format(exc),
            artifact,
            "RV observations failed their canonical schema or ownership validation.",
            "Candidate-local radial-velocity observations.",
            "delta_bic_constant_minus_keplerian",
            None,
            "BIC",
            None,
            ["No RV interpretation is routed from malformed or untrusted observations."],
        )

    inputs = report.get("input_artifacts")
    matching_input = (
        isinstance(inputs, list)
        and any(
            isinstance(item, dict)
            and item.get("path") == "data/external/radial_velocity_observations.json"
            and item.get("sha256") == _sha256(observations_path)
            for item in inputs
        )
    )
    comparison = report.get("model_comparison")
    delta_bic = (
        _finite_number(comparison.get("delta_bic_constant_minus_keplerian"))
        if isinstance(comparison, dict)
        else None
    )
    if (
        report.get("candidate_id") != workspace.candidate_id
        or report.get("report_type") != "keplerian-rv-model-comparison"
        or report.get("observation_time_standard") != "BJD_TDB"
        or report.get("velocity_unit") != "m/s"
        or not matching_input
        or delta_bic is None
    ):
        status, reason = (
            "blocked",
            "RV comparison lacks matching candidate ownership, units, input digest, or finite BIC evidence.",
        )
    elif delta_bic >= RV_REVIEW_DELTA_BIC:
        status, reason = (
            "review-required",
            "RV data favor the fixed-period Keplerian model at the recorded descriptive BIC threshold.",
        )
    else:
        status, reason = (
            "pass",
            "RV comparison has no strong fixed-period Keplerian preference at the recorded descriptive threshold.",
        )
    return _record(
        "radial-velocity",
        status,
        reason,
        artifact,
        "Fixed-period Keplerian-versus-nuisance BIC comparison; not a mass posterior or validation calibration.",
        "Candidate-local BJD_TDB RV observations, quoted uncertainties, optional activity indicator, and fitted nuisance terms.",
        "delta_bic_constant_minus_keplerian",
        delta_bic,
        "BIC",
        None,
        [
            "The comparison is conditional on the fixed photometric period and does not prove a planetary companion.",
            "Activity, sampling, eccentricity, and model-selection uncertainty require human review.",
        ],
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
    radial_velocity = _radial_velocity_record(workspace)
    if radial_velocity is not None:
        diagnostics.append(radial_velocity)
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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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


def _has_detected_hash_bound_bls_result(
    workspace: CandidateWorkspace, signal: Optional[str]
) -> bool:
    """Require a detected BLS result and unchanged candidate-local inputs."""
    suffix = ".{0}".format(signal.lstrip(".")) if signal else ""
    result_path = workspace.path / "outputs" / "bls_search_results{0}.json".format(suffix)
    manifest_path = workspace.path / "outputs" / "bls_search_manifest{0}.json".format(suffix)
    result = _load_object(result_path)
    manifest = _load_object(manifest_path)
    if result is None or manifest is None:
        return False
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        return False
    period = _finite_number(result.get("best_period"))
    epoch = _finite_number(result.get("best_epoch"))
    duration_hours = _finite_number(result.get("best_duration_hours"))
    depth_ppm = _finite_number(result.get("best_depth_ppm"))
    events = result.get("n_distinct_transit_events")
    if (
        result.get("source") != "candidate-data"
        or result.get("detection_status") != "detected"
        or result.get("time_system") != BTJD_TIME_SYSTEM
        or period is None
        or period <= 0.0
        or epoch is None
        or duration_hours is None
        or duration_hours <= 0.0
        or duration_hours / 24.0 >= period
        or depth_ppm is None
        or depth_ppm <= 0.0
        or isinstance(events, bool)
        or not isinstance(events, int)
        or events < 2
        or manifest.get("schema") != "exonym-bls-search-manifest-1"
        or manifest.get("candidate_id") != workspace.candidate_id
        or manifest.get("source") != "candidate-data"
        or manifest.get("detection_status") != "detected"
        or manifest.get("result_path") != result_path.relative_to(workspace.path).as_posix()
        or manifest.get("result_sha256") != _sha256(result_path)
        or configuration.get("engine") != "bls"
        or configuration.get("signal") != signal
        or configuration.get("time_system") != BTJD_TIME_SYSTEM
    ):
        return False
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        return False
    candidate_root = workspace.path.resolve()
    for record in inputs:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            return False
        path = (candidate_root / record["path"]).resolve()
        try:
            path.relative_to(candidate_root)
        except ValueError:
            return False
        if not path.is_file() or record.get("sha256") != _sha256(path):
            return False
    return is_manifest_bound_bls_result(workspace, result_path, result, signal)


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
    if not _has_detected_hash_bound_bls_result(workspace, signal):
        missing.append("detected hash-bound BLS search")
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
