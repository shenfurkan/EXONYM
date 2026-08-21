"""Candidate-local survey registration and single-cohort search control.

Survey records live below ``candidate/_surveys/`` so target membership,
selection sectors, and outcomes never enter the target-neutral source tree.
They provide a complete denominator for a bounded search cohort, but do not
turn a photometric peak into a planetary claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .discovery import (
    injection_recovery_diagnostics,
    mask_box_transit,
    recovered_epoch,
    recovered_period,
    robustness_diagnostics,
)
from .inputs import MINIMUM_BLS_CANDIDATE_SNR, _read_json, load_light_curve_table
from .search import _input_manifest_records, run_bls_on_candidate
from .screening import fixed_ephemeris_screen
from .workspace import CandidateWorkspace, load_candidate, validate_candidate_id


SURVEY_DIRECTORY = "_surveys"
SURVEY_METADATA_FILENAME = "survey.json"
TARGET_METADATA_FILENAME = "target.json"
SURVEY_SCHEMA_VERSION = 1
_SURVEY_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
ROBUSTNESS_CONFIGURATION = {
    "duration_grid_hours": [1.0, 2.0, 4.0],
    "period_min_days": 0.5,
    "period_max_days": 20.0,
    "n_periods": 200,
    "period_grid": "astropy-autopower-baseline-duration-resolved",
    "injection_model": "finite-exposure-box-overlap",
    "exposure_model": "median-positive-cadence-inferred",
    "detrend_window_days": 1.0,
    "detrend_gap_break_window_fraction": 0.5,
    "scramble_seeds": [5, 7, 11],
    "period_agreement_fraction": 0.01,
    "injection_recovery": {
        "phase_offsets": [0.25, 0.5, 0.75],
        "minimum_recovered_trials": 2,
        "minimum_recovery_fraction": 2.0 / 3.0,
        "epoch_tolerance_duration_fraction": 1.0,
    },
}
SENSITIVITY_CONFIGURATION = {
    "period_days": [0.75, 3.0, 10.0],
    "duration_hours": [1.0, 2.0, 4.0],
    "depth_ppm": [250.0, 1000.0, 2500.0],
    "phase_offsets": [0.125, 0.375, 0.625, 0.875],
    "duration_grid_hours": [1.0, 2.0, 4.0],
    "period_min_days": 0.5,
    "period_max_days": 20.0,
    "n_periods": 200,
    "period_grid": "astropy-autopower-baseline-duration-resolved",
    "injection_model": "finite-exposure-box-overlap",
    "exposure_model": "median-positive-cadence-inferred",
    "period_agreement_fraction": 0.01,
    "epoch_tolerance_duration_fraction": 1.0,
    "detrend_window_days": 1.0,
    "detrend_gap_break_window_fraction": 0.5,
    "preprocessing_branches": ["normalized", "running-median"],
}
WILSON_CONFIDENCE_LEVEL = 0.95
WILSON_Z_95 = 1.959963984540054
ALIAS_CONTROL_METHOD = "fixed-ephemeris-odd-even-half-phase-double-period-v1"
ALIAS_CONTROL_INTERPRETATION = (
    "Diagnostic only: odd-even, half-phase, and doubled-period measurements "
    "preserve possible aliases for human review; they do not identify an "
    "eclipsing binary or validate a planet."
)
REFERENCE_BLS_CONFIGURATION = {
    "period_min_days": ROBUSTNESS_CONFIGURATION["period_min_days"],
    "period_max_days": ROBUSTNESS_CONFIGURATION["period_max_days"],
    "duration_hours": None,
    "duration_grid_hours": ROBUSTNESS_CONFIGURATION["duration_grid_hours"],
    "n_periods": ROBUSTNESS_CONFIGURATION["n_periods"],
    "n_periods_role": "minimum requested trial density; Astropy baseline-duration grid may use more",
    "period_grid": ROBUSTNESS_CONFIGURATION["period_grid"],
    "max_points": 4000,
    "quality_filter": "quality == 0 when available",
    "normalization": "lightkurve.remove_nans().normalize()",
    "binning": "per-product median binning; no global rebinning",
    "signal": None,
    "engine": "bls",
    "cadence": "median-binned",
    "use_threads": None,
}


@dataclass(frozen=True)
class SurveyWorkspace:
    """A bounded, candidate-local blind-search cohort."""

    repository_root: Path
    survey_id: str
    path: Path
    metadata: Dict[str, Any]


def _timestamp() -> str:
    """Return a UTC ISO-8601 timestamp without sub-second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_survey_id(survey_id: str) -> str:
    """Normalize a safe survey directory identifier."""
    normalized = survey_id.strip().lower()
    # Surveys are collection instances, so keep their IDs URL- and CLI-friendly.
    if not _SURVEY_ID.fullmatch(normalized):
        raise ValueError("survey_id must use lowercase letters, numbers, and hyphens")
    return normalized


def _survey_path(repository_root: Path, survey_id: str) -> Path:
    return repository_root.resolve() / "candidate" / SURVEY_DIRECTORY / validate_survey_id(survey_id)


def _validate_sectors(sectors: Sequence[int]) -> List[int]:
    """Validate positive, unique TESS sector selections from CLI input."""
    values = list(sectors)
    if not values:
        raise ValueError("a survey requires at least one sector")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("survey sectors must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError("survey sectors must be unique")
    return sorted(values)


def _validate_review_snr(review_snr: float) -> float:
    """Validate the preregistered BLS triage threshold."""
    if isinstance(review_snr, bool) or not isinstance(review_snr, (int, float)):
        raise ValueError("review_snr must be a positive finite number")
    value = float(review_snr)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("review_snr must be a positive finite number")
    return value


def _survey_readme(metadata: Dict[str, Any]) -> str:
    return """# {survey_id}

## Scope

- Mission: `{mission}`
- Sectors: `{sectors}`
- Created: `{created_at}`

This survey records every registered target and its search outcome. A
photometric alert remains an alert for human review until candidate-local
novelty, source-location, false-positive, and follow-up evidence supports a
separate scientific disposition.

## Files

- `survey.json`: Immutable cohort definition and survey metadata.
- `targets/<candidate-id>/target.json`: Registration, eligibility, and search outcome for one target.
- `targets/<candidate-id>/README.md`: Human-readable target status.

## Rules

Targets must have a TIC identifier, no assigned TOI in their workspace record,
and a current eligible novelty audit before the survey can search them. The
survey search only uses products from the sectors listed above.
""".format(
        survey_id=metadata["survey_id"],
        mission=metadata["mission"],
        sectors=", ".join(str(value) for value in metadata["sectors"]),
        created_at=metadata["created_at"],
    )


def create_survey(
    repository_root: Path,
    survey_id: str,
    mission: str,
    sectors: Sequence[int],
    review_snr: float = 6.0,
) -> SurveyWorkspace:
    """Create one bounded search cohort without modifying candidate records."""
    if mission != "tess":
        raise ValueError("blind survey search currently supports only the tess mission")
    normalized_id = validate_survey_id(survey_id)
    selected_sectors = _validate_sectors(sectors)
    selected_review_snr = _validate_review_snr(review_snr)
    path = _survey_path(repository_root, normalized_id)
    if path.exists():
        raise FileExistsError("survey already exists: {0}".format(path))

    metadata: Dict[str, Any] = {
        "schema_version": SURVEY_SCHEMA_VERSION,
        "survey_id": normalized_id,
        "mission": mission,
        "sectors": selected_sectors,
        "review_snr": selected_review_snr,
        "created_at": _timestamp(),
        "scientific_status": "unvalidated",
        "selection_policy": "Register every screened target and retain an explicit outcome.",
    }
    path.mkdir(parents=True)
    (path / "targets").mkdir()
    (path / "runs").mkdir()
    (path / SURVEY_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (path / "README.md").write_text(_survey_readme(metadata), encoding="utf-8")
    return SurveyWorkspace(Path(repository_root).resolve(), normalized_id, path, metadata)


def load_survey(repository_root: Path, survey_id: str) -> SurveyWorkspace:
    """Load a survey and validate its small, stable metadata contract."""
    normalized_id = validate_survey_id(survey_id)
    path = _survey_path(repository_root, normalized_id)
    metadata_path = path / SURVEY_METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError("survey metadata not found: {0}".format(metadata_path))
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid survey metadata: {0}".format(metadata_path)) from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SURVEY_SCHEMA_VERSION:
        raise ValueError("unsupported survey metadata schema")
    if metadata.get("survey_id") != normalized_id or metadata.get("mission") != "tess":
        raise ValueError("survey metadata does not match its directory or mission")
    metadata["sectors"] = _validate_sectors(metadata.get("sectors", []))
    metadata["review_snr"] = _validate_review_snr(metadata.get("review_snr"))
    return SurveyWorkspace(Path(repository_root).resolve(), normalized_id, path, metadata)


def _target_path(survey: SurveyWorkspace, candidate_id: str) -> Path:
    return survey.path / "targets" / validate_candidate_id(candidate_id)


def _target_readme(record: Dict[str, Any]) -> str:
    return """# {candidate_id}

- Registered: `{registered_at}`
- Status: `{status}`
- Search result: `{search_result_path}`

This record belongs to survey `{survey_id}`. Its status records survey
triage only and does not change the candidate workspace lifecycle or make a
planetary claim.
""".format(
        candidate_id=record["candidate_id"],
        registered_at=record["registered_at"],
        status=record["status"],
        search_result_path=record.get("search_result_path") or "not run",
        survey_id=record["survey_id"],
    )


def _atomic_write(path: Path, content: str) -> None:
    """Replace one candidate-local text record without exposing a partial file."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_target_record(survey: SurveyWorkspace, record: Dict[str, Any]) -> Path:
    target_dir = _target_path(survey, record["candidate_id"])
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = target_dir / TARGET_METADATA_FILENAME
    _atomic_write(metadata_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    _atomic_write(target_dir / "README.md", _target_readme(record))
    return metadata_path


def register_survey_target(
    survey: SurveyWorkspace, candidate: CandidateWorkspace
) -> Path:
    """Register one TOI-free TESS workspace in a survey denominator."""
    identifiers = candidate.metadata.get("identifiers", {})
    if identifiers.get("mission") != survey.metadata["mission"]:
        raise ValueError("candidate mission does not match the survey")
    if not identifiers.get("tic"):
        raise ValueError("survey targets require a TIC identifier")
    if identifiers.get("toi"):
        raise ValueError("known TOI workspaces cannot enter an independent discovery survey")

    metadata_path = _target_path(survey, candidate.candidate_id) / TARGET_METADATA_FILENAME
    if metadata_path.exists():
        raise FileExistsError("candidate is already registered in this survey")
    record = {
        "schema_version": SURVEY_SCHEMA_VERSION,
        "survey_id": survey.survey_id,
        "candidate_id": candidate.candidate_id,
        "registered_at": _timestamp(),
        "status": "pending-eligibility",
        "search_result_path": None,
        "search_reused": False,
        "reason": "A current eligible novelty audit is required before search.",
    }
    return _write_target_record(survey, record)


def _load_target_record(survey: SurveyWorkspace, candidate_id: str) -> Dict[str, Any]:
    path = _target_path(survey, candidate_id) / TARGET_METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError("candidate is not registered in this survey")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid survey target record: {0}".format(path)) from exc
    if (
        not isinstance(record, dict)
        or record.get("survey_id") != survey.survey_id
        or record.get("candidate_id") != validate_candidate_id(candidate_id)
    ):
        raise ValueError("survey target record does not match its directory")
    return record


def _current_eligible_audit(candidate: CandidateWorkspace) -> bool:
    """Apply the same novelty-audit contract used by workflow advancement."""
    from .gatekeeper import _gate_novelty_audit

    return _gate_novelty_audit(candidate)[0]


def _survey_result_suffix(survey: SurveyWorkspace) -> str:
    """Return the validated, candidate-local output suffix for one survey."""
    return ".survey-" + survey.survey_id


def _provenanced_input_manifest(
    candidate: CandidateWorkspace, table: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Return hash-bound raw inputs returned by one provenance-restricted loader call."""
    return _input_manifest_records(
        candidate,
        [Path(path) for path in table.get("input_files", [])],
        list(table.get("input_sha256s", [])),
    )


def _robustness_path(survey: SurveyWorkspace, candidate: CandidateWorkspace) -> Path:
    """Return the candidate-local robustness artifact path for one survey."""
    return candidate.path / "outputs" / ("survey_robustness" + _survey_result_suffix(survey) + ".json")


def _sensitivity_path(survey: SurveyWorkspace, candidate: CandidateWorkspace) -> Path:
    """Return the candidate-local sensitivity diagnostic path for one survey."""
    return candidate.path / "outputs" / ("survey_sensitivity" + _survey_result_suffix(survey) + ".json")


def _sensitivity_injections(
    time_btjd: Sequence[float], configuration: Dict[str, Any]
) -> List[Dict[str, float]]:
    """Construct a fixed candidate-level injection grid from observed timestamps."""
    finite_times = [float(value) for value in time_btjd if math.isfinite(float(value))]
    if not finite_times:
        raise RuntimeError("survey sensitivity diagnostics require finite candidate timestamps")
    anchor_epoch = min(finite_times)
    injections: List[Dict[str, float]] = []
    for period_days in configuration["period_days"]:
        for duration_hours in configuration["duration_hours"]:
            for depth_ppm in configuration["depth_ppm"]:
                for phase_offset in configuration["phase_offsets"]:
                    period = float(period_days)
                    injections.append(
                        {
                            "period_days": period,
                            "epoch_btjd": anchor_epoch + float(phase_offset) * period,
                            "duration_hours": float(duration_hours),
                            "depth_ppm": float(depth_ppm),
                            "phase_offset": float(phase_offset),
                        }
                    )
    return injections


def _sensitivity_summary(
    injections: Sequence[Dict[str, float]], recovery: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate a fixed grid without converting it into a completeness claim."""
    expected = {
        (
            float(injection["period_days"]),
            float(injection["duration_hours"]),
            float(injection["depth_ppm"]),
        ): []
        for injection in injections
    }
    for entry in recovery:
        injection = entry["injection"]
        key = (
            float(injection["period_days"]),
            float(injection["duration_hours"]),
            float(injection["depth_ppm"]),
        )
        if key in expected:
            expected[key].append(bool(entry["recovered"]))
    cells: List[Dict[str, Any]] = []
    recovered_count = 0
    for period_days, duration_hours, depth_ppm in sorted(expected):
        outcomes = expected[(period_days, duration_hours, depth_ppm)]
        count = sum(outcomes)
        recovered_count += count
        cells.append(
            {
                "period_days": period_days,
                "duration_hours": duration_hours,
                "depth_ppm": depth_ppm,
                "trial_count": len(outcomes),
                "recovered_count": count,
                "recovery_fraction": float(count) / len(outcomes) if outcomes else 0.0,
                "recovery_interval_95": _wilson_interval(count, len(outcomes)),
            }
        )
    trial_count = len(recovery)
    return {
        "grid_cell_count": len(cells),
        "trial_count": trial_count,
        "recovered_count": recovered_count,
        "recovery_fraction": float(recovered_count) / trial_count if trial_count else 0.0,
        "recovery_interval_95": _wilson_interval(recovered_count, trial_count),
        "cells": cells,
    }


def _wilson_interval(recovered_count: int, trial_count: int) -> Dict[str, Any]:
    """Return a finite-sample Wilson interval for a recovery proportion.

    The interval describes repeated injections from this fixed candidate-level
    grid only. It is included to expose the uncertainty caused by the limited
    number of phase trials, not to turn the grid into a population completeness
    calibration.
    """
    if trial_count < 1 or recovered_count < 0 or recovered_count > trial_count:
        raise ValueError("Wilson recovery interval requires bounded non-empty counts")
    proportion = float(recovered_count) / trial_count
    z_squared = WILSON_Z_95 * WILSON_Z_95
    denominator = 1.0 + z_squared / trial_count
    center = (proportion + z_squared / (2.0 * trial_count)) / denominator
    half_width = (
        WILSON_Z_95
        * math.sqrt(
            (proportion * (1.0 - proportion) + z_squared / (4.0 * trial_count))
            / trial_count
        )
        / denominator
    )
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    if recovered_count == 0:
        lower = 0.0
    if recovered_count == trial_count:
        upper = 1.0
    return {
        "method": "wilson-score",
        "confidence_level": WILSON_CONFIDENCE_LEVEL,
        "lower": lower,
        "upper": upper,
    }


def run_survey_sensitivity(
    survey: SurveyWorkspace, candidate: CandidateWorkspace
) -> Path:
    """Write a two-branch candidate-level injection-recovery grid.

    This operation is deliberately separate from survey alert routing. It uses
    a frozen box-transit grid on one candidate's observed light curve and
    reports only the recovery outcomes for that configuration. It does not
    estimate a survey selection function, population completeness, false-alarm
    probability, or scientific validation probability.
    """
    _load_target_record(survey, candidate.candidate_id)
    if not _current_eligible_audit(candidate):
        raise RuntimeError("survey sensitivity requires a current eligible novelty audit")
    table = load_light_curve_table(
        candidate,
        sectors=survey.metadata["sectors"],
        require_raw_provenance=True,
    )
    if table is None:
        raise RuntimeError("survey sensitivity diagnostics require real candidate photometry")
    input_manifest = _provenanced_input_manifest(candidate, table)
    if not input_manifest:
        raise RuntimeError("survey sensitivity diagnostics require hashable candidate inputs")
    configuration = json.loads(json.dumps(SENSITIVITY_CONFIGURATION))
    configuration["minimum_snr"] = _validate_review_snr(survey.metadata["review_snr"])
    injections = _sensitivity_injections(table["time"], configuration)
    recovery = injection_recovery_diagnostics(
        table["time"],
        table["flux"],
        injections,
        configuration["duration_grid_hours"],
        configuration["period_min_days"],
        configuration["period_max_days"],
        configuration["n_periods"],
        configuration["period_agreement_fraction"],
        minimum_snr=configuration["minimum_snr"],
        epoch_tolerance_duration_fraction=configuration[
            "epoch_tolerance_duration_fraction"
        ],
        sectors=table["sector"],
        detrend_window_days=configuration["detrend_window_days"],
        flux_err=table.get("flux_err"),
        gap_break_window_fraction=configuration["detrend_gap_break_window_fraction"],
    )
    artifact = {
        "schema_version": 1,
        "source": "candidate-data",
        "survey_id": survey.survey_id,
        "candidate_id": candidate.candidate_id,
        "sectors": list(survey.metadata["sectors"]),
        "scientific_status": "candidate-level-injection-recovery-diagnostic",
        "validation_eligible": False,
        "completeness_eligible": False,
        "configuration": configuration,
        "input_files": [path.relative_to(candidate.path).as_posix() for path in table["input_files"]],
        "input_manifest": input_manifest,
        "injection_recovery": recovery,
        "summary": _sensitivity_summary(injections, recovery),
        "calibration_limits": {
            "population_false_alarm_probability": None,
            "population_detection_reliability": None,
            "reason": "One candidate, fixed box injections, and deterministic BLS recovery do not calibrate a survey population selection function.",
        },
    }
    path = _sensitivity_path(survey, candidate)
    _atomic_write(path, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path


def _reference_signal(search_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the BLS scale that candidate-specific injections must reproduce."""
    try:
        result = {
            "best_period": float(search_result["best_period"]),
            "best_epoch": float(search_result["best_epoch"]),
            "best_duration_hours": float(search_result["best_duration_hours"]),
            "best_depth_ppm": float(search_result["best_depth_ppm"]),
            "snr": float(search_result["snr"]),
            "n_distinct_transit_events": int(search_result["n_distinct_transit_events"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("survey BLS result is missing a usable signal scale") from exc
    result["usable_for_injection_recovery"] = bool(
        all(math.isfinite(value) for value in result.values())
        and result["best_period"] > 0
        and result["best_duration_hours"] > 0
        and result["best_depth_ppm"] > 0
        and result["n_distinct_transit_events"] >= 2
    )
    return result


def _candidate_scale_injections(
    reference_signal: Dict[str, Any], phase_offsets: Sequence[float]
) -> List[Dict[str, float]]:
    """Create deterministic injections at the detected BLS period, duration, and depth."""
    if not reference_signal["usable_for_injection_recovery"]:
        return []
    return [
        {
            "period_days": reference_signal["best_period"],
            "epoch_btjd": reference_signal["best_epoch"]
            + float(offset) * reference_signal["best_period"],
            "duration_hours": reference_signal["best_duration_hours"],
            "depth_ppm": reference_signal["best_depth_ppm"],
            "phase_offset": float(offset),
        }
        for offset in phase_offsets
    ]


def _injection_entry_matches(
    entry: Dict[str, Any],
    reference_signal: Dict[str, Any],
    phase_offset: float,
    recovery_configuration: Dict[str, Any],
    period_agreement_fraction: float,
    require_all_branches: bool,
) -> bool:
    """Check one recovery entry against its declared injection and branches."""
    try:
        injection = entry["injection"]
        expected_injection = {
            "period_days": reference_signal["best_period"],
            "epoch_btjd": reference_signal["best_epoch"]
            + phase_offset * reference_signal["best_period"],
            "duration_hours": reference_signal["best_duration_hours"],
            "depth_ppm": reference_signal["best_depth_ppm"],
            "phase_offset": phase_offset,
        }
        if injection != expected_injection:
            return False
        epoch_tolerance_hours = (
            injection["duration_hours"]
            * recovery_configuration["epoch_tolerance_duration_fraction"]
        )
        if entry.get("epoch_tolerance_hours") != epoch_tolerance_hours:
            return False
        branches = entry.get("branches")
        if branches is None:
            if require_all_branches:
                return False
            branches = {
                "normalized": {
                    "period_match": entry["period_match"],
                    "epoch_match": entry["epoch_match"],
                    "snr_pass": entry["snr_pass"],
                    "recovered": entry["recovered"],
                    "best": entry["best"],
                }
            }
        expected_branch_names = {"normalized", "running-median"}
        if require_all_branches:
            if set(branches) != expected_branch_names:
                return False
        elif set(branches) != {"normalized"}:
            return False

        expected_outcomes: List[Dict[str, Any]] = []
        for branch in sorted(branches):
            branch_result = branches[branch]
            best = branch_result["best"]
            period_match = recovered_period(
                injection["period_days"], best["best_period"], period_agreement_fraction
            )
            epoch_match = recovered_epoch(
                injection["period_days"],
                injection["epoch_btjd"],
                best["best_epoch"],
                epoch_tolerance_hours,
            )
            snr_pass = bool(
                best["snr"] >= recovery_configuration["minimum_snr"]
                and int(best["n_distinct_transit_events"]) >= 2
            )
            expected = {
                "period_match": period_match,
                "epoch_match": epoch_match,
                "snr_pass": snr_pass,
                "recovered": bool(period_match and epoch_match and snr_pass),
                "best": best,
            }
            if branch_result != expected:
                return False
            expected_outcomes.append(expected)
        normalized = branches["normalized"]
        return bool(
            entry.get("best") == normalized["best"]
            and entry.get("period_match")
            == all(outcome["period_match"] for outcome in expected_outcomes)
            and entry.get("epoch_match")
            == all(outcome["epoch_match"] for outcome in expected_outcomes)
            and entry.get("snr_pass")
            == all(outcome["snr_pass"] for outcome in expected_outcomes)
            and entry.get("recovered")
            == all(outcome["recovered"] for outcome in expected_outcomes)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _injection_recovery_summary(
    reference_signal: Dict[str, Any],
    recovery: Sequence[Dict[str, Any]],
    recovery_configuration: Dict[str, Any],
    period_agreement_fraction: float,
    masked_cadences: int,
    require_all_branches: bool = False,
) -> Dict[str, Any]:
    """Summarize the frozen candidate-scale recovery rule for one artifact."""
    phase_offsets = recovery_configuration["phase_offsets"]
    trial_count = len(recovery)
    recovered_count = sum(entry["recovered"] is True for entry in recovery)
    recovery_fraction = float(recovered_count) / trial_count if trial_count else 0.0
    expected_trial_count = len(phase_offsets)
    matching_injections = trial_count == expected_trial_count
    if matching_injections:
        for entry, phase_offset in zip(recovery, phase_offsets):
            matching_injections = _injection_entry_matches(
                entry,
                reference_signal,
                phase_offset,
                recovery_configuration,
                period_agreement_fraction,
                require_all_branches,
            )
            if not matching_injections:
                break
    passed = bool(
        reference_signal["usable_for_injection_recovery"]
        and matching_injections
        and recovered_count >= recovery_configuration["minimum_recovered_trials"]
        and recovery_fraction >= recovery_configuration["minimum_recovery_fraction"]
    )
    return {
        "reference_signal_usable": reference_signal["usable_for_injection_recovery"],
        "masked_cadences": masked_cadences,
        "trial_count": trial_count,
        "expected_trial_count": expected_trial_count,
        "recovered_count": recovered_count,
        "recovery_fraction": recovery_fraction,
        "minimum_recovered_trials": recovery_configuration["minimum_recovered_trials"],
        "minimum_recovery_fraction": recovery_configuration["minimum_recovery_fraction"],
        "passed": passed,
    }


def _survey_alias_controls(
    table: Dict[str, Any], reference_signal: Dict[str, Any]
) -> Dict[str, Any]:
    """Preserve fixed-ephemeris alias diagnostics for a survey review.

    These measurements intentionally do not change the survey's triage outcome.
    A survey alert is still only a request for human review, and unresolved
    windows are retained instead of being treated as evidence against an alias.
    """
    if not reference_signal["usable_for_injection_recovery"]:
        return {
            "status": "not-run-unusable-reference-signal",
            "method": ALIAS_CONTROL_METHOD,
            "odd_even": None,
            "half_phase_control": None,
            "double_period_hypothesis": None,
            "interpretation": ALIAS_CONTROL_INTERPRETATION,
        }
    try:
        result = fixed_ephemeris_screen(
            table["time"],
            table["flux"],
            reference_signal["best_period"],
            reference_signal["best_epoch"],
            reference_signal["best_duration_hours"],
        )
    except (KeyError, TypeError, ValueError):
        return {
            "status": "unavailable-invalid-input",
            "method": ALIAS_CONTROL_METHOD,
            "odd_even": None,
            "half_phase_control": None,
            "double_period_hypothesis": None,
            "interpretation": ALIAS_CONTROL_INTERPRETATION,
        }
    return {
        "status": "computed",
        "method": ALIAS_CONTROL_METHOD,
        "odd_even": result["odd_even"],
        "half_phase_control": result["half_phase_control"],
        "double_period_hypothesis": result["double_period_hypothesis"],
        "interpretation": ALIAS_CONTROL_INTERPRETATION,
    }


def _run_survey_robustness(
    survey: SurveyWorkspace,
    candidate: CandidateWorkspace,
    search_result: Dict[str, Any],
    review_snr: float,
) -> Tuple[Path, Dict[str, Any]]:
    """Run and persist survey robustness diagnostics using real candidate data.

    Candidate-scale injections are masked away from the original BLS event and
    tested at three fixed phase offsets. A majority recovery is a triage
    criterion only; it does not estimate completeness or validate a source.
    """
    table = load_light_curve_table(
        candidate,
        sectors=survey.metadata["sectors"],
        require_raw_provenance=True,
    )
    if table is None:
        raise RuntimeError("survey robustness diagnostics require real candidate photometry")
    configuration = json.loads(json.dumps(ROBUSTNESS_CONFIGURATION))
    recovery_configuration = configuration["injection_recovery"]
    recovery_configuration["minimum_snr"] = review_snr
    diagnostics = robustness_diagnostics(
        table["time"],
        table["flux"],
        table["sector"],
        configuration["duration_grid_hours"],
        configuration["period_min_days"],
        configuration["period_max_days"],
        configuration["n_periods"],
        configuration["detrend_window_days"],
        configuration["scramble_seeds"],
        flux_err=table.get("flux_err"),
        gap_break_window_fraction=configuration["detrend_gap_break_window_fraction"],
    )
    reference_signal = _reference_signal(search_result)
    masked_cadences = 0
    injections: List[Dict[str, Any]] = []
    if reference_signal["usable_for_injection_recovery"]:
        masked_flux, masked_cadences = mask_box_transit(
            table["time"],
            table["flux"],
            reference_signal["best_period"],
            reference_signal["best_epoch"],
            reference_signal["best_duration_hours"],
        )
        injections = injection_recovery_diagnostics(
            table["time"],
            masked_flux,
            _candidate_scale_injections(reference_signal, recovery_configuration["phase_offsets"]),
            configuration["duration_grid_hours"],
            configuration["period_min_days"],
            configuration["period_max_days"],
            configuration["n_periods"],
            configuration["period_agreement_fraction"],
            minimum_snr=recovery_configuration["minimum_snr"],
            epoch_tolerance_duration_fraction=recovery_configuration[
                "epoch_tolerance_duration_fraction"
            ],
            sectors=table["sector"],
            detrend_window_days=configuration["detrend_window_days"],
            flux_err=table.get("flux_err"),
            gap_break_window_fraction=configuration["detrend_gap_break_window_fraction"],
        )
    artifact = {
        "schema_version": 1,
        "source": "candidate-data",
        "survey_id": survey.survey_id,
        "candidate_id": candidate.candidate_id,
        "sectors": list(survey.metadata["sectors"]),
        "configuration": configuration,
        "input_files": [path.relative_to(candidate.path).as_posix() for path in table["input_files"]],
        "reference_signal": reference_signal,
        "diagnostics": diagnostics,
        "alias_controls": _survey_alias_controls(table, reference_signal),
        "injection_recovery": injections,
        "injection_recovery_summary": _injection_recovery_summary(
            reference_signal,
            injections,
            recovery_configuration,
            configuration["period_agreement_fraction"],
            masked_cadences,
            require_all_branches=True,
        ),
    }
    path = _robustness_path(survey, candidate)
    _atomic_write(path, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path, artifact


def _robustness_passes(artifact: Dict[str, Any], review_snr: float) -> bool:
    """Require repeatability, quiet controls, and majority scale-matched recovery.

    The recovery gate accepts exactly the configured deterministic phase trials
    only when at least the configured majority reproduce the measured BLS
    period, epoch, duration, depth, and review SNR. This creates an internal
    human-review alert only and never a scientific claim.
    """
    try:
        diagnostics = artifact["diagnostics"]
        variants = diagnostics["variants"]
        normalized = variants["normalized"]["best"]
        detrended = variants["running-median"]["best"]
        configuration = artifact["configuration"]
        recovery_configuration = configuration["injection_recovery"]
        reference_signal = artifact["reference_signal"]
        recovery = artifact["injection_recovery"]
        summary = artifact["injection_recovery_summary"]
        controls = diagnostics["controls"]
        branch_controls = controls["by_variant"]
        tolerance = float(configuration["period_agreement_fraction"])
        reference = float(normalized["best_period"])
        comparison = float(detrended["best_period"])
        signal_period = float(reference_signal["best_period"])
        if not math.isfinite(reference) or reference <= 0:
            return False
        if not math.isfinite(signal_period) or signal_period <= 0:
            return False
        for field in (
            "duration_grid_hours",
            "period_min_days",
            "period_max_days",
            "n_periods",
            "period_grid",
            "detrend_window_days",
            "scramble_seeds",
            "period_agreement_fraction",
        ):
            if configuration[field] != ROBUSTNESS_CONFIGURATION[field]:
                return False
        expected_recovery_configuration = ROBUSTNESS_CONFIGURATION["injection_recovery"]
        for field in (
            "phase_offsets",
            "minimum_recovered_trials",
            "minimum_recovery_fraction",
            "epoch_tolerance_duration_fraction",
        ):
            if recovery_configuration[field] != expected_recovery_configuration[field]:
                return False
        if not math.isclose(float(recovery_configuration["minimum_snr"]), review_snr):
            return False
        if controls["scramble_method"] != "independent-sector-circular-shift":
            return False
        if set(branch_controls) != {"normalized", "running-median"}:
            return False
        if controls["inverted"] != branch_controls["normalized"]["inverted"]:
            return False
        if controls["scrambles"] != branch_controls["normalized"]["scrambles"]:
            return False
        control_scores: List[float] = []
        for branch in ("normalized", "running-median"):
            branch_control = branch_controls[branch]
            if [entry["seed"] for entry in branch_control["scrambles"]] != configuration["scramble_seeds"]:
                return False
            scores = [float(branch_control["inverted"]["snr"])] + [
                float(entry["best"]["snr"]) for entry in branch_control["scrambles"]
            ]
            if any(not math.isfinite(score) or score < 0 for score in scores):
                return False
            if not math.isclose(float(branch_control["max_snr"]), max(scores)):
                return False
            control_scores.extend(scores)
        if not math.isclose(float(controls["max_snr"]), max(control_scores)):
            return False
        recomputed_summary = _injection_recovery_summary(
            reference_signal,
            recovery,
            recovery_configuration,
            tolerance,
            int(summary["masked_cadences"]),
            require_all_branches=True,
        )
        normalized_agrees = abs(reference / signal_period - 1.0) <= tolerance
        detrended_agrees = abs(comparison / signal_period - 1.0) <= tolerance
        return bool(
            float(reference_signal["snr"]) >= review_snr
            and float(normalized["snr"]) >= review_snr
            and float(detrended["snr"]) >= review_snr
            and int(reference_signal["n_distinct_transit_events"]) >= 2
            and int(normalized["n_distinct_transit_events"]) >= 2
            and int(detrended["n_distinct_transit_events"]) >= 2
            and normalized_agrees
            and detrended_agrees
            and float(diagnostics["controls"]["max_snr"]) < review_snr
            and summary == recomputed_summary
            and recomputed_summary["passed"]
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def _existing_sector_bls_result(
    survey: SurveyWorkspace, candidate: CandidateWorkspace, sectors: Sequence[int]
) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Return a matching real-data BLS result left by an interrupted survey run."""
    suffix = _survey_result_suffix(survey)
    output = candidate.path / "outputs" / ("bls_search_results" + suffix + ".json")
    manifest = candidate.path / "outputs" / ("bls_search_manifest" + suffix + ".json")
    try:
        payload = _read_json(output)
        manifest_payload = _read_json(manifest)
        if payload is None or manifest_payload is None:
            return None
        configuration = manifest_payload["configuration"]
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    expected_configuration = dict(REFERENCE_BLS_CONFIGURATION)
    expected_configuration["sectors"] = list(sectors)
    if not isinstance(configuration, dict):
        return None
    table = load_light_curve_table(
        candidate, sectors=sectors, require_raw_provenance=True
    )
    if table is None:
        return None
    try:
        expected_inputs = _provenanced_input_manifest(candidate, table)
    except (OSError, ValueError):
        return None
    if not expected_inputs:
        return None
    try:
        snr = float(payload["snr"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(snr):
        return None
    expected_detection_status = (
        "detected" if snr >= MINIMUM_BLS_CANDIDATE_SNR else "no-detection"
    )
    if (
        not isinstance(payload, dict)
        or not isinstance(manifest_payload, dict)
        or payload.get("source") != "candidate-data"
        or payload.get("time_system") != "BTJD_TDB"
        or payload.get("detection_threshold_snr") != MINIMUM_BLS_CANDIDATE_SNR
        or payload.get("detection_status") != expected_detection_status
        or manifest_payload.get("schema") != "exonym-bls-search-manifest-1"
        or manifest_payload.get("candidate_id") != candidate.candidate_id
        or manifest_payload.get("source") != "candidate-data"
        or manifest_payload.get("detection_status") != expected_detection_status
        or manifest_payload.get("result_path") != output.relative_to(candidate.path).as_posix()
        or manifest_payload.get("result_sha256")
        != hashlib.sha256(output.read_bytes()).hexdigest()
        or manifest_payload.get("inputs") != expected_inputs
        or configuration.get("time_system") != "BTJD_TDB"
        or configuration.get("detection_threshold_snr") != MINIMUM_BLS_CANDIDATE_SNR
    ):
        return None
    for key, expected_value in expected_configuration.items():
        if configuration.get(key) != expected_value:
            return None
    uncertainty_source = configuration.get("uncertainty_source")
    if not isinstance(uncertainty_source, list) or not uncertainty_source:
        return None
    return output, payload


def run_survey_search(
    survey: SurveyWorkspace,
    candidate: CandidateWorkspace,
    review_snr: Optional[float] = None,
) -> Path:
    """Search one eligible target using only the survey's selected sectors.

    The outcome is a triage state. It does not set the candidate lifecycle,
    scientific disposition, or a false-positive claim.
    """
    configured_review_snr = _validate_review_snr(survey.metadata["review_snr"])
    if review_snr is not None and _validate_review_snr(review_snr) != configured_review_snr:
        raise ValueError("review_snr must match the preregistered survey threshold")
    record = _load_target_record(survey, candidate.candidate_id)
    if not _current_eligible_audit(candidate):
        record["status"] = "blocked-novelty-audit"
        record["reason"] = "Search blocked until a current eligible novelty audit exists."
        record["search_reused"] = False
        record["updated_at"] = _timestamp()
        return _write_target_record(survey, record)

    existing = _existing_sector_bls_result(survey, candidate, survey.metadata["sectors"])
    if existing is None:
        output = run_bls_on_candidate(
            candidate,
            period_min=REFERENCE_BLS_CONFIGURATION["period_min_days"],
            period_max=REFERENCE_BLS_CONFIGURATION["period_max_days"],
            n_periods=REFERENCE_BLS_CONFIGURATION["n_periods"],
            sectors=survey.metadata["sectors"],
            result_suffix=_survey_result_suffix(survey),
            duration_grid_hours=ROBUSTNESS_CONFIGURATION["duration_grid_hours"],
        )
        payload = _read_json(output)
        if payload is None:
            raise RuntimeError("survey BLS output is not a strict JSON object")
        record["search_reused"] = False
    else:
        output, payload = existing
        record["search_reused"] = True
    if payload.get("source") != "candidate-data":
        raise RuntimeError("survey search requires real candidate photometry")

    snr = float(payload.get("snr", 0.0))
    robustness_path, robustness = _run_survey_robustness(
        survey, candidate, payload, configured_review_snr
    )
    passes_robustness = _robustness_passes(robustness, configured_review_snr)
    record["status"] = (
        "alert-for-human-review"
        if snr >= configured_review_snr and passes_robustness
        else "searched-no-alert"
    )
    record["reason"] = (
        "Uncalibrated search-score and robustness checks exceeded the survey review threshold; complete candidate-local vetting."
        if record["status"] == "alert-for-human-review"
        else "The uncalibrated search score or required robustness checks did not exceed the survey review threshold."
    )
    record["search_result_path"] = output.relative_to(candidate.path).as_posix()
    record["search_snr"] = snr
    record["robustness_result_path"] = robustness_path.relative_to(candidate.path).as_posix()
    record["robustness_passed"] = passes_robustness
    record["review_snr"] = configured_review_snr
    record["updated_at"] = _timestamp()
    return _write_target_record(survey, record)


def exclude_survey_target(survey: SurveyWorkspace, candidate_id: str, reason: str) -> Path:
    """Record a pre-search exclusion while preserving the survey denominator."""
    if not reason.strip():
        raise ValueError("an exclusion reason is required")
    record = _load_target_record(survey, candidate_id)
    record["status"] = "excluded-before-search"
    record["reason"] = reason.strip()
    record["search_reused"] = False
    record["updated_at"] = _timestamp()
    return _write_target_record(survey, record)


def survey_summary(survey: SurveyWorkspace) -> Dict[str, Any]:
    """Return the recorded survey denominator and outcome counts."""
    counts: Dict[str, int] = {}
    targets: List[Dict[str, Any]] = []
    for metadata_path in sorted((survey.path / "targets").glob("*/" + TARGET_METADATA_FILENAME)):
        try:
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            record = {
                "candidate_id": metadata_path.parent.name,
                "status": "invalid-record",
                "reason": "Survey target record could not be read.",
            }
        else:
            candidate_id = str(record.get("candidate_id", ""))
            try:
                valid_record = (
                    record.get("survey_id") == survey.survey_id
                    and validate_candidate_id(candidate_id) == metadata_path.parent.name
                )
                if not valid_record:
                    raise ValueError("target record identity mismatch")
                load_candidate(survey.repository_root, candidate_id)
            except (FileNotFoundError, ValueError):
                record = dict(record)
                record["status"] = "orphaned-candidate-record"
                record["reason"] = "Referenced candidate workspace is unavailable."
        status = str(record.get("status", "invalid-record"))
        counts[status] = counts.get(status, 0) + 1
        targets.append(record)
    return {"survey": survey.metadata, "outcome_counts": counts, "targets": targets}


def load_survey_candidate(repository_root: Path, candidate_id: str) -> CandidateWorkspace:
    """Load a candidate for CLI survey actions without duplicating validation."""
    return load_candidate(repository_root, candidate_id)
