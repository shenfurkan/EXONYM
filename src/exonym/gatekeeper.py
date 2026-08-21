"""Quality Verification Gate (QVG) engine.

``exonym advance`` promotes a candidate to the next workflow phase only after
every mandatory gate item is checked and any phase-specific gate passes.
Gate sign-offs are recorded in ``candidate/<id>/gates/`` and lifecycle events
are appended to ``candidate/<id>/lifecycle/events.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .resources import ResourceUnavailableError, read_schema_text
from .schemas import NOVELTY_AUDIT_SCHEMA, PROVENANCE_SCHEMA
from .tracking import phase_document_path, parse_checklist
from .workspace import (
    CandidateWorkspace,
    METADATA_FILENAME,
    WORKFLOW_PHASES,
    load_candidate,
    validate_metadata,
)


NOVELTY_AUDIT_RELATIVE_PATH = Path("decisions") / "novelty_audit.json"
NOVELTY_AUDIT_ELIGIBLE_STATUS = "eligible"
NOVELTY_AUDIT_REQUIRED_PROVIDERS = frozenset(("nasa-toi", "nasa-confirmed", "exofop"))


class GateError(RuntimeError):
    """Raised when a phase gate blocks advancement."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_metadata(workspace: CandidateWorkspace, metadata: Dict) -> None:
    path = workspace.path / METADATA_FILENAME
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_event(workspace: CandidateWorkspace, event: Dict) -> None:
    directory = workspace.path / "lifecycle"
    directory.mkdir(parents=True, exist_ok=True)
    events = directory / "events.jsonl"
    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def next_phase(phase: str) -> Optional[str]:
    """Return the next workflow phase, or None for the terminal phase."""
    if phase not in WORKFLOW_PHASES:
        raise ValueError("unknown workflow phase: {0}".format(phase))
    index = WORKFLOW_PHASES.index(phase)
    if index + 1 >= len(WORKFLOW_PHASES):
        return None
    return WORKFLOW_PHASES[index + 1]


def _sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without loading a FITS product into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> object:
    """Reject non-finite JSON constants in an acquisition-sidecar record."""
    raise ValueError("non-finite JSON constant: {0}".format(value))


def _unique_json_object(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    """Parse a JSON object only when every field name is unique."""
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {0}".format(key))
        result[key] = value
    return result


def _load_provenance_schema(workspace: CandidateWorkspace) -> object:
    """Load the authoritative provenance schema for direct acquisition gating."""
    try:
        return json.loads(read_schema_text(workspace.repository_root, PROVENANCE_SCHEMA))
    except (FileNotFoundError, ResourceUnavailableError, OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("provenance schema is unavailable: {0}".format(exc)) from exc


def _gate_provenance_ready(workspace: CandidateWorkspace) -> Tuple[bool, str]:
    """Require schema-valid, hash-matched sidecars for every raw FITS product."""
    raw_root = workspace.path / "data" / "raw"
    products = sorted(raw_root.rglob("*")) if raw_root.is_dir() else []
    fits_files = [p for p in products if p.is_file() and p.suffix.lower() in (".fits", ".fz")]
    if not fits_files:
        return False, "data/raw contains no FITS products; acquisition gate not met"
    try:
        import jsonschema
    except ImportError:
        return False, "provenance schema validation is unavailable: jsonschema is not installed"
    try:
        schema = _load_provenance_schema(workspace)
    except RuntimeError as exc:
        return False, str(exc)
    errors: List[str] = []
    for product in fits_files:
        sidecar = product.with_name(product.stem + ".provenance.json")
        if not sidecar.is_file():
            errors.append("{0}: missing sidecar".format(product.name))
            continue
        try:
            record = json.loads(
                sidecar.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
            jsonschema.validate(record, schema, format_checker=jsonschema.FormatChecker())
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
            errors.append("{0}: invalid sidecar ({1})".format(product.name, str(exc).splitlines()[0]))
            continue
        except jsonschema.SchemaError as exc:
            return False, "provenance schema is invalid: {0}".format(exc.message)
        if not isinstance(record, dict) or record.get("sha256") != _sha256_file(product):
            errors.append("{0}: sidecar SHA-256 does not match product bytes".format(product.name))
    if errors:
        return False, "raw provenance failures: {0}".format("; ".join(errors[:5]))
    return True, "{0} raw products with schema-valid hash-matched sidecars".format(len(fits_files))


def _gate_fpp_claim(workspace: CandidateWorkspace, threshold: float = 0.01) -> Tuple[bool, str]:
    """Block FPP claims until the scene model is calibration-ready.

    A real Monte Carlo report is necessary but not sufficient for a validation
    claim. The current pipeline has provenance-bound observed photometry, but
    its PRF scene model is deliberately marked uncalibrated and is not yet an
    input to TRICERATOPS. No claim file, including a hand-written one, may
    advance the analysis gate before that gap is closed.
    """
    return (
        False,
        "FPP claims are disabled until provenance-bound observed photometry and "
        "calibrated scene constraints are integrated into TRICERATOPS.",
    )


def _parse_utc_timestamp(value: object) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp with an explicit timezone as UTC."""
    if not isinstance(value, str):
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _gate_novelty_evidence(workspace: CandidateWorkspace, audit: Dict) -> Tuple[bool, str]:
    """Require retained, hash-matched responses from all independent registries."""
    if audit.get("schema_version") != 2:
        return False, "eligible novelty audits must use evidence-bound schema version 2"
    evidence = audit.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != len(NOVELTY_AUDIT_REQUIRED_PROVIDERS):
        return False, "eligible novelty audit must retain exactly three independent registry responses"

    workspace_root = workspace.path.resolve()
    providers = []
    paths = set()
    retrieval_ids = set()
    for entry in evidence:
        if not isinstance(entry, dict):
            return False, "eligible novelty audit contains a malformed evidence entry"
        provider = entry.get("provider")
        source_uri = entry.get("source_uri")
        response_path = entry.get("response_path")
        if provider not in NOVELTY_AUDIT_REQUIRED_PROVIDERS:
            return False, "eligible novelty audit contains an unsupported registry provider"
        if not isinstance(source_uri, str) or not source_uri.startswith("https://"):
            return False, "eligible novelty audit evidence must use an HTTPS source URI"
        if not isinstance(response_path, str):
            return False, "eligible novelty audit evidence has no candidate-local response path"
        relative_path = Path(response_path)
        if (
            relative_path.is_absolute()
            or len(relative_path.parts) != 5
            or relative_path.parts[:3] != ("data", "external", "novelty")
        ):
            return False, "eligible novelty audit evidence path is outside the novelty evidence area"
        resolved_path = (workspace_root / relative_path).resolve()
        try:
            resolved_path.relative_to(workspace_root)
        except ValueError:
            return False, "eligible novelty audit evidence path escapes the candidate workspace"
        if not resolved_path.is_file():
            return False, "eligible novelty audit evidence response is missing"
        if entry.get("evidence_sha256") != _sha256_file(resolved_path):
            return False, "eligible novelty audit evidence SHA-256 does not match the retained response"
        providers.append(provider)
        paths.add(resolved_path)
        retrieval_ids.add(relative_path.parts[3])

    if set(providers) != NOVELTY_AUDIT_REQUIRED_PROVIDERS:
        return False, "eligible novelty audit is missing an independent registry provider"
    if len(paths) != len(evidence) or len(retrieval_ids) != 1:
        return False, "eligible novelty audit evidence must be distinct responses from one retrieval"
    return True, "eligible novelty audit has hash-matched independent registry evidence"


def _gate_novelty_audit(workspace: CandidateWorkspace) -> Tuple[bool, str]:
    """Require a current, schema-valid, eligible candidate novelty audit."""
    audit_path = workspace.path / NOVELTY_AUDIT_RELATIVE_PATH
    if not audit_path.is_file():
        return False, "missing novelty audit: {0}".format(NOVELTY_AUDIT_RELATIVE_PATH)
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return False, "invalid novelty audit JSON: {0}".format(exc)

    try:
        schema = json.loads(read_schema_text(workspace.repository_root, NOVELTY_AUDIT_SCHEMA))
    except FileNotFoundError:
        return False, "novelty audit schema is unavailable: {0}".format(NOVELTY_AUDIT_SCHEMA)
    except ResourceUnavailableError as exc:
        return False, "novelty audit schema is unavailable: {0}".format(exc)
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        return False, "invalid novelty audit schema: {0}".format(exc)
    try:
        import jsonschema
    except ImportError:
        return False, "novelty audit schema validation is unavailable: jsonschema is not installed"
    try:
        jsonschema.validate(audit, schema, format_checker=jsonschema.FormatChecker())
    except jsonschema.ValidationError as exc:
        return False, "novelty audit violates schema: {0}".format(exc.message)
    except jsonschema.SchemaError as exc:
        return False, "invalid novelty audit schema: {0}".format(exc.message)

    if audit.get("candidate_id") != workspace.candidate_id:
        return False, "novelty audit candidate_id does not match the workspace"
    if audit.get("status") != NOVELTY_AUDIT_ELIGIBLE_STATUS:
        return False, "novelty audit status is not eligible: {0}".format(audit.get("status"))
    evidence_ok, evidence_detail = _gate_novelty_evidence(workspace, audit)
    if not evidence_ok:
        return False, evidence_detail

    retrieved_at = _parse_utc_timestamp(audit.get("retrieved_at"))
    freshness = audit.get("freshness")
    expires_at = _parse_utc_timestamp(
        freshness.get("expires_at") if isinstance(freshness, dict) else None
    )
    now = datetime.now(timezone.utc)
    if retrieved_at is None or expires_at is None:
        return False, "novelty audit contains an invalid retrieval or freshness timestamp"
    if retrieved_at > now:
        return False, "novelty audit retrieval date is in the future"
    if expires_at <= retrieved_at:
        return False, "novelty audit freshness expiry must be later than retrieval date"
    if expires_at <= now:
        return False, "novelty audit is stale: freshness.expires_at has passed"
    return True, "novelty audit is eligible and current through {0}".format(
        audit["freshness"]["expires_at"]
    )


def gate_errors(workspace: CandidateWorkspace) -> List[str]:
    """Return a list of gate failures blocking the current phase."""
    metadata = workspace.metadata
    phase = metadata["workflow"]["phase"]
    errors: List[str] = []

    if metadata["lifecycle"]["state"] == "stopped":
        errors.append("candidate lifecycle is stopped; workflow advancement is disabled")

    document = phase_document_path(workspace, phase)
    if document is not None:
        telemetry = parse_checklist(document)
        if not telemetry.exists:
            errors.append("missing gate document: {0}".format(document.relative_to(workspace.path)))
        else:
            if telemetry.mandatory_total == 0:
                errors.append(
                    "gate document contains no mandatory checklist items: {0}".format(
                        document.relative_to(workspace.path)
                    )
                )
            for item in telemetry.items:
                if item.mandatory and not item.checked:
                    errors.append(
                        "unchecked mandatory item in {0}: {1}".format(
                            document.relative_to(workspace.path), item.text[:80]
                        )
                    )

    if phase in ("feasibility", "review"):
        ok, detail = _gate_novelty_audit(workspace)
        if not ok:
            errors.append(detail)
    if phase == "acquisition":
        ok, detail = _gate_provenance_ready(workspace)
        if not ok:
            errors.append(detail)
    elif phase == "analysis":
        ok, detail = _gate_fpp_claim(workspace)
        if not ok:
            errors.append(detail)
    elif phase == "review":
        if metadata["lifecycle"]["state"] in ("published", "archived"):
            errors.append("candidate is already locked; no further advancement")

    return errors


def set_lifecycle_state(
    workspace: CandidateWorkspace,
    state: str,
    reason: Optional[str] = None,
) -> Dict:
    """Set the lifecycle state of a candidate, recording the change as an event.

    ``state`` must be one of the registered lifecycle states. Changing the state
    of a stopped or locked candidate (``stopped``, ``published``, or
    ``archived``) requires a non-empty ``reason`` so audit history always
    explains why the workspace was reopened.
    """
    from .workspace import LIFECYCLE_STATES

    if state not in LIFECYCLE_STATES:
        raise ValueError("invalid lifecycle state: {0}".format(state))
    workspace = load_candidate(workspace.repository_root, workspace.candidate_id)
    metadata = dict(workspace.metadata)
    lifecycle = dict(metadata["lifecycle"])
    old_state = lifecycle["state"]
    if old_state == state:
        raise GateError("lifecycle state unchanged: {0}".format(state))
    has_reason = isinstance(reason, str) and bool(reason.strip())
    if old_state in ("stopped", "published", "archived") and not has_reason:
        raise GateError(
            "a reason is required to change the state of a stopped or locked candidate "
            "({0} -> {1})".format(old_state, state)
        )
    lifecycle["state"] = state
    lifecycle["state_since"] = _now()
    if reason:
        lifecycle["reason"] = reason
    metadata["lifecycle"] = lifecycle
    validate_metadata(metadata, workspace.candidate_id)
    _write_metadata(workspace, metadata)
    _append_event(
        workspace,
        {
            "event": "state_changed",
            "candidate_id": workspace.candidate_id,
            "from": old_state,
            "to": state,
            "reason": reason,
            "timestamp": _now(),
        },
    )
    return lifecycle


def advance(workspace: CandidateWorkspace) -> Dict:
    """Validate the current gate and promote the candidate one phase.

    Advancing from ``review`` additionally locks the lifecycle to
    ``published`` when it is still ``active`` or ``paused``.

    The candidate record is re-read from disk so repeated advances always
    operate on the current state.
    """
    workspace = load_candidate(workspace.repository_root, workspace.candidate_id)
    metadata = dict(workspace.metadata)
    phase = metadata["workflow"]["phase"]
    if metadata["lifecycle"]["state"] == "stopped":
        raise GateError("candidate lifecycle is stopped; workflow advancement is disabled")
    errors = gate_errors(workspace)
    if errors:
        raise GateError("; ".join(errors))

    next_phase_name = next_phase(phase)
    if next_phase_name is None and phase != "review":
        raise GateError("terminal phase reached: {0}".format(phase))

    event: Dict = {
        "event": "advanced",
        "candidate_id": workspace.candidate_id,
        "from": phase,
        "to": next_phase_name or phase,
        "timestamp": _now(),
    }
    if phase == "review":
        if metadata["lifecycle"]["state"] in ("published", "archived"):
            raise GateError(
                "candidate lifecycle is already locked: {0}".format(metadata["lifecycle"]["state"])
            )
        metadata["lifecycle"]["state"] = "published"
        metadata["lifecycle"]["state_since"] = _now()
        metadata["lifecycle"]["reason"] = "Review gate passed; lifecycle locked"
        event["lifecycle"] = metadata["lifecycle"]["state"]
        event["to"] = "review (locked)"

    metadata["workflow"]["phase"] = next_phase_name if next_phase_name is not None else phase
    validate_metadata(metadata, workspace.candidate_id)
    _write_metadata(workspace, metadata)
    _append_event(workspace, event)

    gate_record = {
        "gate": phase,
        "candidate_id": workspace.candidate_id,
        "result": "PASS",
        "timestamp": _now(),
        "next_phase": next_phase_name,
        "event": event["event"],
    }
    gates_dir = workspace.path / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)
    index = len(list(gates_dir.glob("gate-*.json")))
    (gates_dir / "gate-{0:03d}-{1}.json".format(index, phase)).write_text(
        json.dumps(gate_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return event
