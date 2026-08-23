"""Candidate-local comparison of a detected ephemeris with known catalog signals.

Only catalog records with an explicitly documented field contract are compared.
The current implementation supports fresh NASA Exoplanet Archive ``pscomppars``
snapshots retained by :mod:`exonym.catalog_federation`. It is a review
diagnostic: an empty comparison does not establish novelty, while a period or
epoch agreement never creates a scientific disposition or claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .inputs import BTJD_TIME_SYSTEM, EPHEMERIS_CONFIG_NAMES, load_transit_ephemeris
from .workspace import CandidateWorkspace, validate_signal_suffix


SUPPORTED_PROVIDER = "nasa-exoplanet-archive"
TOI_PROVIDER = "nasa-exoplanet-archive-toi"
SUPPORTED_PROVIDERS = (SUPPORTED_PROVIDER, TOI_PROVIDER)
RECORDED_EVIDENCE_PROVIDER = "candidate-recorded-evidence"
RECORDED_EVIDENCE_FILENAME = "known_signal_ephemerides.json"
RECORDED_EVIDENCE_SOURCE_KINDS = (
    "eclipsing-binary-catalog",
    "variable-star-catalog",
    "exofop",
    "literature",
)
_RECORD_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
METHOD = "known-signal-ephemeris-period-epoch-duration-harmonic-v3"
PERIOD_RELATIVE_TOLERANCE = 0.001
EPOCH_TOLERANCE_DURATION_MULTIPLIER = 1.0
DURATION_RELATIVE_TOLERANCE = 0.5
HARMONIC_FACTORS = (0.5, 1.0, 2.0, 3.0)

# Every automatic source must state its field names, duration unit, and whether
# the retained epoch is safe to compare with a candidate BJD_TDB ephemeris.
# No conversion is inferred from a generic "BJD" label.
_PROVIDER_FIELD_CONTRACTS = {
    SUPPORTED_PROVIDER: {
        "period": "pl_orbper",
        "epoch": "pl_tranmid",
        "duration": "pl_trandur",
        "name": ("pl_name", "pl_letter"),
        "epoch_time_scale": "PER_RECORD_DECLARED_TIME_SCALE",
        "duration_unit": "hours",
    },
    TOI_PROVIDER: {
        "period": "pl_orbper",
        "epoch": "pl_tranmid",
        "duration": "pl_trandurh",
        "name": ("toi",),
        "epoch_time_scale": "BJD_UNSPECIFIED",
        "duration_unit": "hours",
    },
    RECORDED_EVIDENCE_PROVIDER: {
        "period": "pl_orbper",
        "epoch": "pl_tranmid",
        "duration": "pl_trandur",
        "name": ("pl_name",),
        "epoch_time_scale": "BJD_TDB",
        "duration_unit": "hours",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> object:
    raise ValueError("non-finite JSON number: {0}".format(value))


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number: {0}".format(value))
    return parsed


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {0}".format(key))
        result[key] = value
    return result


def _read_json(path: Path) -> Dict[str, Any]:
    parsed = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        parse_float=_finite_float,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(parsed, dict):
        raise ValueError("JSON artifact must be an object")
    return parsed


def _finite(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _parse_utc(value: object) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _relative(workspace: CandidateWorkspace, path: Path) -> str:
    return path.relative_to(workspace.path).as_posix()


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write a candidate-local JSON record without a partial replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _candidate_file(workspace: CandidateWorkspace, relative_path: str) -> Path:
    """Resolve one supplied artifact only when it remains inside the workspace."""
    supplied = Path(relative_path)
    if supplied.is_absolute() or any(part == ".." for part in supplied.parts):
        raise ValueError("raw artifact path must be candidate-relative and must not traverse upward")
    resolved_workspace = workspace.path.resolve()
    resolved_path = (resolved_workspace / supplied).resolve()
    try:
        resolved_path.relative_to(resolved_workspace)
    except ValueError as exc:
        raise ValueError("raw artifact path must remain inside the candidate workspace") from exc
    if not resolved_path.is_file():
        raise ValueError("raw artifact path must name an existing regular file")
    return resolved_path


def _parse_required_utc(name: str, value: object) -> datetime:
    parsed = _parse_utc(value)
    if parsed is None:
        raise ValueError("{0} must be an ISO-8601 UTC timestamp".format(name))
    return parsed.astimezone(timezone.utc)


def record_known_signal_ephemeris(
    workspace: CandidateWorkspace,
    record_id: str,
    source_kind: str,
    source_name: str,
    source_uri: str,
    raw_artifact_path: str,
    period_days: float,
    epoch_bjd_tdb: float,
    duration_hours: float,
    retrieved_at: str,
    expires_at: str,
) -> Path:
    """Append one explicitly sourced BJD_TDB known-signal ephemeris record.

    This deliberately does not parse web pages or infer time-system offsets.
    The caller supplies a reviewed source row and its retained raw artifact;
    the record binds both the values and the raw bytes before matching.
    """
    normalized_id = record_id.strip().lower()
    if not _RECORD_ID.fullmatch(normalized_id):
        raise ValueError("record_id must use lowercase letters, numbers, dots, hyphens, or underscores")
    if source_kind not in RECORDED_EVIDENCE_SOURCE_KINDS:
        raise ValueError("unsupported known-signal evidence source kind")
    if not source_name.strip():
        raise ValueError("source_name must not be empty")
    if not source_uri.startswith("https://"):
        raise ValueError("source_uri must use HTTPS")
    period = _finite(period_days)
    epoch = _finite(epoch_bjd_tdb)
    duration = _finite(duration_hours)
    if period is None or period <= 0 or epoch is None or duration is None or duration <= 0:
        raise ValueError("known-signal period, BJD_TDB epoch, and duration must be finite positive values")
    retrieved = _parse_required_utc("retrieved_at", retrieved_at)
    expires = _parse_required_utc("expires_at", expires_at)
    if retrieved > _now() or expires <= retrieved:
        raise ValueError("known-signal evidence timestamps must be current and expire after retrieval")
    raw_path = _candidate_file(workspace, raw_artifact_path)
    allowed_prefixes = (workspace.path / "data" / "external", workspace.path / "literature")
    if not any(raw_path.is_relative_to(prefix.resolve()) for prefix in allowed_prefixes):
        raise ValueError("raw artifact must be retained under data/external or literature")
    evidence_path = workspace.path / "decisions" / RECORDED_EVIDENCE_FILENAME
    if evidence_path.exists():
        existing = _read_json(evidence_path)
        if existing.get("candidate_id") != workspace.candidate_id or not isinstance(existing.get("records"), list):
            raise ValueError("existing known-signal evidence registry is invalid for this candidate")
        records = list(existing["records"])
    else:
        records = []
    if any(isinstance(record, dict) and record.get("record_id") == normalized_id for record in records):
        raise ValueError("known-signal evidence record_id already exists")
    records.append(
        {
            "record_id": normalized_id,
            "source_kind": source_kind,
            "source_name": source_name.strip(),
            "source_uri": source_uri,
            "retrieved_at": retrieved.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "expires_at": expires.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "raw_artifact": {"path": _relative(workspace, raw_path), "sha256": _sha256(raw_path)},
            "native_time_scale": "BJD_TDB",
            "ephemeris": {
                "period_days": period,
                "epoch_bjd_tdb": epoch,
                "duration_hours": duration,
            },
        }
    )
    _write_json_atomic(
        evidence_path,
        {
            "schema_version": 1,
            "candidate_id": workspace.candidate_id,
            "source": "candidate-evidence",
            "records": sorted(records, key=lambda record: str(record["record_id"])),
        },
    )
    return evidence_path


def _ephemeris_input_path(workspace: CandidateWorkspace, signal: Optional[str], source: str) -> Path:
    if source in ("candidate-config-signal", "partial-candidate-config-signal") and signal is not None:
        return workspace.path / "config" / "signals" / ("transit_config" + signal + ".json")
    if source in ("candidate-config", "partial-candidate-config"):
        for name in EPHEMERIS_CONFIG_NAMES:
            path = workspace.path / "config" / name
            if path.is_file():
                return path
    if source == "bls-search":
        return workspace.path / "outputs" / ("bls_search_results" + (signal or "") + ".json")
    raise ValueError("known-signal matching requires a candidate-derived ephemeris artifact")


def _candidate_ephemeris(
    workspace: CandidateWorkspace, signal: Optional[str]
) -> Dict[str, Any]:
    ephemeris = load_transit_ephemeris(workspace, signal=signal)
    source = str(ephemeris.get("source", ""))
    period = _finite(ephemeris.get("period_days"))
    epoch = _finite(ephemeris.get("epoch_btjd"))
    duration_days = _finite(ephemeris.get("duration_days"))
    field_sources = ephemeris.get("field_sources")
    if (
        source == "synthetic-demo"
        or period is None
        or period <= 0
        or epoch is None
        or duration_days is None
        or duration_days <= 0
        or ephemeris.get("time_system") != BTJD_TIME_SYSTEM
        or not isinstance(field_sources, dict)
        or any(field_sources.get(name) in (None, "synthetic-demo") for name in ("period_days", "epoch_btjd", "duration_days"))
    ):
        raise ValueError("known-signal matching requires a complete candidate-derived ephemeris")
    path = _ephemeris_input_path(workspace, signal, source)
    if not path.is_file():
        raise ValueError("known-signal matching cannot hash the selected ephemeris artifact")
    return {
        "source": source,
        "signal": signal,
        "period_days": period,
        "epoch_btjd": epoch,
        "duration_hours": duration_days * 24.0,
        "time_system": BTJD_TIME_SYSTEM,
        "input_artifact": {"path": _relative(workspace, path), "sha256": _sha256(path)},
    }


def _latest_fresh_snapshots(workspace: CandidateWorkspace) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Return the latest fresh supported snapshots and retained exclusion reasons."""
    latest: Dict[str, Tuple[datetime, Path, Dict[str, Any]]] = {}
    excluded: List[Dict[str, str]] = []
    for manifest_path in sorted((workspace.path / "runs" / "catalog").glob("*/*/query-manifest.json")):
        try:
            manifest = _read_json(manifest_path)
        except (OSError, UnicodeError, ValueError) as exc:
            excluded.append({"path": _relative(workspace, manifest_path), "reason": "invalid-query-manifest: {0}".format(exc)})
            continue
        provider = manifest.get("provider")
        if not isinstance(provider, str) or provider not in SUPPORTED_PROVIDERS:
            continue
        retrieved_at = _parse_utc(manifest.get("retrieved_at"))
        expires_at = _parse_utc(manifest.get("expires_at"))
        snapshot_path = manifest_path.with_name("snapshot.json")
        if manifest.get("candidate_id") != workspace.candidate_id:
            excluded.append({"path": _relative(workspace, manifest_path), "reason": "candidate-id-mismatch"})
            continue
        if manifest.get("status") != "available":
            excluded.append({"path": _relative(workspace, manifest_path), "reason": "retrieval-not-available"})
            continue
        if retrieved_at is None or expires_at is None or expires_at <= _now():
            excluded.append({"path": _relative(workspace, manifest_path), "reason": "retrieval-stale-or-invalid-time"})
            continue
        if not snapshot_path.is_file():
            excluded.append({"path": _relative(workspace, manifest_path), "reason": "snapshot-missing"})
            continue
        try:
            snapshot = _read_json(snapshot_path)
        except (OSError, UnicodeError, ValueError) as exc:
            excluded.append({"path": _relative(workspace, snapshot_path), "reason": "invalid-snapshot: {0}".format(exc)})
            continue
        if (
            snapshot.get("candidate_id") != workspace.candidate_id
            or snapshot.get("provider") != provider
            or snapshot.get("retrieval_id") != manifest.get("retrieval_id")
            or snapshot.get("status") != "available"
            or not isinstance(snapshot.get("records"), list)
        ):
            excluded.append({"path": _relative(workspace, snapshot_path), "reason": "snapshot-provenance-mismatch"})
            continue
        if provider not in latest or retrieved_at > latest[provider][0]:
            latest[provider] = (retrieved_at, manifest_path, snapshot)
    if not latest:
        return [], excluded
    snapshots: List[Dict[str, Any]] = []
    for provider in SUPPORTED_PROVIDERS:
        selected = latest.get(provider)
        if selected is None:
            continue
        _, manifest_path, snapshot = selected
        snapshot_path = manifest_path.with_name("snapshot.json")
        contract = _PROVIDER_FIELD_CONTRACTS[provider]
        snapshots.append(
            {
                "provider": provider,
                "retrieval_id": snapshot["retrieval_id"],
                "query_manifest": {"path": _relative(workspace, manifest_path), "sha256": _sha256(manifest_path)},
                "snapshot": {"path": _relative(workspace, snapshot_path), "sha256": _sha256(snapshot_path)},
                "native_time_scale": contract["epoch_time_scale"],
                "records": snapshot["records"],
            }
        )
    return snapshots, excluded


def _fresh_recorded_evidence(workspace: CandidateWorkspace) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Return only current, raw-hash-bound candidate-recorded signal rows."""
    evidence_path = workspace.path / "decisions" / RECORDED_EVIDENCE_FILENAME
    if not evidence_path.is_file():
        return [], []
    try:
        evidence = _read_json(evidence_path)
    except (OSError, UnicodeError, ValueError) as exc:
        return [], [{"path": _relative(workspace, evidence_path), "reason": "invalid-recorded-evidence: {0}".format(exc)}]
    if evidence.get("candidate_id") != workspace.candidate_id or not isinstance(evidence.get("records"), list):
        return [], [{"path": _relative(workspace, evidence_path), "reason": "recorded-evidence-candidate-or-records-mismatch"}]
    registry_artifact = {"path": _relative(workspace, evidence_path), "sha256": _sha256(evidence_path)}
    snapshots: List[Dict[str, Any]] = []
    excluded: List[Dict[str, str]] = []
    for record in evidence["records"]:
        if not isinstance(record, dict):
            excluded.append({"path": registry_artifact["path"], "reason": "recorded-evidence-record-not-object"})
            continue
        record_id = record.get("record_id")
        retrieved_at = _parse_utc(record.get("retrieved_at"))
        expires_at = _parse_utc(record.get("expires_at"))
        raw_artifact = record.get("raw_artifact")
        ephemeris = record.get("ephemeris")
        if (
            not isinstance(record_id, str)
            or not _RECORD_ID.fullmatch(record_id)
            or record.get("source_kind") not in RECORDED_EVIDENCE_SOURCE_KINDS
            or record.get("native_time_scale") != "BJD_TDB"
            or retrieved_at is None
            or expires_at is None
            or expires_at <= _now()
            or not isinstance(raw_artifact, dict)
            or not isinstance(ephemeris, dict)
        ):
            excluded.append({"path": registry_artifact["path"], "reason": "recorded-evidence-record-invalid-or-stale"})
            continue
        try:
            raw_path = _candidate_file(workspace, str(raw_artifact.get("path", "")))
        except ValueError:
            excluded.append({"path": registry_artifact["path"], "reason": "recorded-evidence-raw-artifact-missing-or-unsafe"})
            continue
        if raw_artifact.get("sha256") != _sha256(raw_path):
            excluded.append({"path": registry_artifact["path"], "reason": "recorded-evidence-raw-artifact-hash-mismatch"})
            continue
        period = _finite(ephemeris.get("period_days"))
        epoch = _finite(ephemeris.get("epoch_bjd_tdb"))
        duration = _finite(ephemeris.get("duration_hours"))
        if period is None or period <= 0 or epoch is None or duration is None or duration <= 0:
            excluded.append({"path": registry_artifact["path"], "reason": "recorded-evidence-ephemeris-invalid"})
            continue
        snapshots.append(
            {
                "provider": RECORDED_EVIDENCE_PROVIDER,
                "retrieval_id": record_id,
                "query_manifest": registry_artifact,
                "snapshot": registry_artifact,
                "native_time_scale": "BJD_TDB",
                "records": [
                    {
                        "pl_orbper": period,
                        "pl_tranmid": epoch,
                        "pl_trandur": duration,
                        "pl_name": record.get("source_name", ""),
                    }
                ],
            }
        )
    return snapshots, excluded


def _field(record: Mapping[str, Any], name: str) -> Optional[float]:
    lowered = {str(key).lower(): value for key, value in record.items()}
    return _finite(lowered.get(name.lower()))


def _epoch_time_scale(provider: str, source: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    """Return BJD_TDB only when the provider row itself declares that scale."""
    if provider != SUPPORTED_PROVIDER:
        return str(contract["epoch_time_scale"])
    lowered = {str(key).lower(): value for key, value in source.items()}
    declared = str(lowered.get("pl_tranmid_systemref", "")).strip().upper().replace(" ", "_")
    return "BJD_TDB" if declared == "BJD_TDB" else "BJD_UNSPECIFIED"


def _phase_delta_days(first_epoch: float, second_epoch: float, period_days: float) -> float:
    return abs(((first_epoch - second_epoch + 0.5 * period_days) % period_days) - 0.5 * period_days)


def _compare_record(
    candidate: Mapping[str, Any], source: Mapping[str, Any], record_index: int, snapshot: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    provider = str(snapshot["provider"])
    contract = _PROVIDER_FIELD_CONTRACTS.get(provider)
    if contract is None:
        return None
    period = _field(source, str(contract["period"]))
    if period is None or period <= 0:
        return None
    raw_epoch_bjd = _field(source, str(contract["epoch"]))
    epoch_time_scale = _epoch_time_scale(provider, source, contract)
    epoch_bjd = raw_epoch_bjd if epoch_time_scale == "BJD_TDB" else None
    duration_hours = _field(source, str(contract["duration"]))
    candidate_period = float(candidate["period_days"])
    candidate_epoch = float(candidate["epoch_btjd"])
    candidate_duration = float(candidate["duration_hours"])
    ratio = period / candidate_period
    harmonic = min(HARMONIC_FACTORS, key=lambda value: abs(ratio - value))
    period_difference = abs(ratio - harmonic) / harmonic
    period_harmonic_match = period_difference <= PERIOD_RELATIVE_TOLERANCE
    epoch_btjd = epoch_bjd - 2457000.0 if epoch_bjd is not None else None
    epoch_tolerance = max(candidate_duration, duration_hours or 0.0) * EPOCH_TOLERANCE_DURATION_MULTIPLIER / 24.0
    # For non-unity harmonic ratios the epoch folding must happen on the
    # shorter period so that offset epochs (e.g. odd/even aliases) still
    # register an epoch match. Without this an EB at P/2 of the candidate
    # period would always fail the epoch check and fall through to
    # "no-ephemeris-match".
    phase_period = candidate_period
    if harmonic != 1.0 and period_harmonic_match:
        phase_period = min(period, candidate_period)
    epoch_delta = (
        _phase_delta_days(epoch_btjd, candidate_epoch, phase_period)
        if epoch_btjd is not None
        else None
    )
    epoch_match_raw = epoch_delta <= epoch_tolerance if epoch_delta is not None else None
    # Harmonic-parity ambiguity: candidate period matches a harmonic but the
    # epoch does not fold correctly → route to human review rather than
    # silently falling through to "no-ephemeris-match".
    harmonic_parity_ambiguous = bool(
        period_harmonic_match and epoch_match_raw is False and harmonic != 1.0
    )
    epoch_match: Any = epoch_match_raw
    if harmonic_parity_ambiguous:
        epoch_match = "harmonic-parity-ambiguous"
    duration_ratio = duration_hours / candidate_duration if duration_hours is not None and candidate_duration > 0 else None
    duration_compatible = (
        abs(duration_ratio - 1.0) <= DURATION_RELATIVE_TOLERANCE
        if duration_ratio is not None
        else None
    )
    return {
        "provider": provider,
        "retrieval_id": snapshot["retrieval_id"],
        "snapshot_path": snapshot["snapshot"]["path"],
        "snapshot_sha256": snapshot["snapshot"]["sha256"],
        "source_record_index": record_index,
        "source_name": next(
            (str(source.get(field, "")) for field in contract["name"] if str(source.get(field, ""))), None
        ),
        "known_period_days": period,
        "known_epoch_bjd_tdb": epoch_bjd,
        "known_epoch_time_scale": epoch_time_scale,
        "known_duration_hours": duration_hours,
        "period_ratio_known_over_candidate": ratio,
        "nearest_harmonic_factor": harmonic,
        "period_relative_difference": period_difference,
        "period_harmonic_match": period_harmonic_match,
        "epoch_phase_delta_days": epoch_delta,
        "epoch_tolerance_days": epoch_tolerance if epoch_bjd is not None else None,
        "epoch_match": epoch_match,
        "duration_ratio_known_over_candidate": duration_ratio,
        "duration_compatible": duration_compatible,
        "review_required": bool(
            period_harmonic_match
            and (epoch_match is True or epoch_match is None or epoch_match == "harmonic-parity-ambiguous")
        ),
    }


def match_known_signal_ephemerides(
    workspace: CandidateWorkspace, signal: Optional[str] = None
) -> Path:
    """Write a hash-bound comparison with fresh supported catalog snapshots.

    A matched or unresolved harmonic routes to human review only. The absence of
    a match is strictly limited to the retained current catalog records and is
    never an independent-discovery or validation decision.
    """
    signal = validate_signal_suffix(signal)
    candidate = _candidate_ephemeris(workspace, signal)
    snapshots, excluded = _latest_fresh_snapshots(workspace)
    recorded_snapshots, recorded_excluded = _fresh_recorded_evidence(workspace)
    snapshots.extend(recorded_snapshots)
    excluded.extend(recorded_excluded)
    comparisons: List[Dict[str, Any]] = []
    source_snapshots: List[Dict[str, Any]] = []
    for snapshot in snapshots:
        source_snapshots.append({key: value for key, value in snapshot.items() if key != "records"})
        for index, record in enumerate(snapshot["records"]):
            if not isinstance(record, dict):
                continue
            comparison = _compare_record(candidate, record, index, snapshot)
            if comparison is not None:
                comparisons.append(comparison)
    full_matches = [
        entry
        for entry in comparisons
        if entry["period_harmonic_match"]
        and entry["epoch_match"] is True
    ]
    unresolved = [
        entry
        for entry in comparisons
        if entry["period_harmonic_match"]
        and (entry["epoch_match"] is None or entry["epoch_match"] == "harmonic-parity-ambiguous")
    ]
    if full_matches:
        status = "review-required-known-signal-match"
    elif unresolved:
        status = "review-required-period-harmonic"
    elif source_snapshots:
        status = "no-ephemeris-match-in-current-supported-catalog"
    else:
        status = "insufficient-current-supported-catalog-evidence"
    output = workspace.path / "outputs" / "known_signal_ephemeris_match{0}.json".format(signal or "")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "generated_at": _now().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": "candidate-data",
        "signal": signal,
        "method": METHOD,
        "status": status,
        "candidate_ephemeris": candidate,
        "configuration": {
            "supported_providers": [SUPPORTED_PROVIDER, TOI_PROVIDER, RECORDED_EVIDENCE_PROVIDER],
            "period_relative_tolerance": PERIOD_RELATIVE_TOLERANCE,
            "epoch_tolerance_duration_multiplier": EPOCH_TOLERANCE_DURATION_MULTIPLIER,
            "duration_relative_tolerance": DURATION_RELATIVE_TOLERANCE,
            "harmonic_factors": list(HARMONIC_FACTORS),
        },
        "source_snapshots": source_snapshots,
        "excluded_retrievals": excluded,
        "comparisons": comparisons,
        "limitations": (
            "This compares fresh retained NASA Exoplanet Archive pscomppars rows, TOI rows with period/duration-only epoch handling, and fresh candidate-recorded BJD_TDB evidence with raw-artifact hashes. "
            "No match does not establish novelty; automatic eclipsing-binary, variable-star, ExoFOP, and literature ephemeris retrievals remain unavailable."
        ),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output
