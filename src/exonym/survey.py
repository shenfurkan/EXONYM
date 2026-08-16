"""Candidate-local survey registration and single-cohort search control.

Survey records live below ``candidate/_surveys/`` so target membership,
selection sectors, and outcomes never enter the target-neutral source tree.
They provide a complete denominator for a bounded search cohort, but do not
turn a photometric peak into a planetary claim.
"""

from __future__ import annotations

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
from .inputs import load_light_curve_table
from .search import run_bls_on_candidate
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
    "detrend_window_days": 1.0,
    "scramble_seeds": [5, 7, 11],
    "period_agreement_fraction": 0.01,
    "injection_recovery": {
        "phase_offsets": [0.25, 0.5, 0.75],
        "minimum_recovered_trials": 2,
        "minimum_recovery_fraction": 2.0 / 3.0,
        "epoch_tolerance_duration_fraction": 1.0,
    },
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


def _robustness_path(survey: SurveyWorkspace, candidate: CandidateWorkspace) -> Path:
    """Return the candidate-local robustness artifact path for one survey."""
    return candidate.path / "outputs" / ("survey_robustness" + _survey_result_suffix(survey) + ".json")


def _reference_signal(search_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the BLS scale that candidate-specific injections must reproduce."""
    try:
        result = {
            "best_period": float(search_result["best_period"]),
            "best_epoch": float(search_result["best_epoch"]),
            "best_duration_hours": float(search_result["best_duration_hours"]),
            "best_depth_ppm": float(search_result["best_depth_ppm"]),
            "snr": float(search_result["snr"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("survey BLS result is missing a usable signal scale") from exc
    result["usable_for_injection_recovery"] = bool(
        all(math.isfinite(value) for value in result.values())
        and result["best_period"] > 0
        and result["best_duration_hours"] > 0
        and result["best_depth_ppm"] > 0
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


def _injection_recovery_summary(
    reference_signal: Dict[str, Any],
    recovery: Sequence[Dict[str, Any]],
    recovery_configuration: Dict[str, Any],
    period_agreement_fraction: float,
    masked_cadences: int,
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
            injection = entry["injection"]
            best = entry["best"]
            epoch_tolerance_hours = (
                injection["duration_hours"]
                * recovery_configuration["epoch_tolerance_duration_fraction"]
            )
            period_match = recovered_period(
                injection["period_days"], best["best_period"], period_agreement_fraction
            )
            epoch_match = recovered_epoch(
                injection["period_days"],
                injection["epoch_btjd"],
                best["best_epoch"],
                epoch_tolerance_hours,
            )
            snr_pass = bool(best["snr"] >= recovery_configuration["minimum_snr"])
            matching_injections = (
                injection["period_days"] == reference_signal["best_period"]
                and injection["duration_hours"] == reference_signal["best_duration_hours"]
                and injection["depth_ppm"] == reference_signal["best_depth_ppm"]
                and injection["phase_offset"] == phase_offset
                and injection["epoch_btjd"]
                == reference_signal["best_epoch"]
                + phase_offset * reference_signal["best_period"]
                and entry["epoch_tolerance_hours"] == epoch_tolerance_hours
                and entry["period_match"] is period_match
                and entry["epoch_match"] is epoch_match
                and entry["snr_pass"] is snr_pass
                and entry["recovered"] is (period_match and epoch_match and snr_pass)
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
    table = load_light_curve_table(candidate, sectors=survey.metadata["sectors"])
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
        "injection_recovery": injections,
        "injection_recovery_summary": _injection_recovery_summary(
            reference_signal,
            injections,
            recovery_configuration,
            configuration["period_agreement_fraction"],
            masked_cadences,
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
        recomputed_summary = _injection_recovery_summary(
            reference_signal,
            recovery,
            recovery_configuration,
            tolerance,
            int(summary["masked_cadences"]),
        )
        normalized_agrees = abs(reference / signal_period - 1.0) <= tolerance
        detrended_agrees = abs(comparison / signal_period - 1.0) <= tolerance
        return bool(
            float(reference_signal["snr"]) >= review_snr
            and float(normalized["snr"]) >= review_snr
            and float(detrended["snr"]) >= review_snr
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
        payload = json.loads(output.read_text(encoding="utf-8"))
        configuration = json.loads(manifest.read_text(encoding="utf-8"))["configuration"]
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("source") != "candidate-data"
        or configuration.get("engine") != "bls"
        or configuration.get("signal") is not None
        or configuration.get("sectors") != list(sectors)
    ):
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
            sectors=survey.metadata["sectors"],
            result_suffix=_survey_result_suffix(survey),
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
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
        "BLS and robustness checks exceeded the survey review threshold; complete candidate-local vetting."
        if record["status"] == "alert-for-human-review"
        else "BLS or required robustness checks did not exceed the survey review threshold."
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
