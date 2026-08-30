"""Evidence-backed candidate classification reviews.

The review record is candidate-owned and immutable once written.  The small
summary fields in ``candidate.json`` make filtering cheap; the versioned
record under ``decisions/reviews/`` retains the evidence and the prior values
needed to audit the change.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from .isolation import is_reparse_point
from .workspace import (
    CandidateWorkspace,
    PUBLICATION_STATES,
    RETENTION_CLASSES,
    REVIEW_STATUSES,
    SCIENTIFIC_DISPOSITIONS,
    validate_metadata,
)


CLASSIFICATION_REVIEW_SCHEMA = "classification-review.schema.json"
REVIEW_DIRECTORY = Path("decisions") / "reviews"
DEFAULT_REVIEW_STATUS = "unreviewed"
DEFAULT_RETENTION_CLASS = "hot"


def _sha256(path: Path) -> str:
    """Hash one evidence file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write strict JSON with an atomic same-directory replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _candidate_evidence_path(workspace: CandidateWorkspace, supplied: str) -> Path:
    """Resolve one existing regular evidence file within its workspace."""
    relative = Path(supplied)
    if not supplied.strip() or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("review evidence must be a relative candidate-local path")
    candidate_root = workspace.path.resolve()
    path = (candidate_root / relative).resolve()
    try:
        path.relative_to(candidate_root)
    except ValueError as exc:
        raise ValueError("review evidence must remain inside the candidate workspace") from exc
    current = candidate_root
    for part in relative.parts:
        current = current / part
        if current.exists() and is_reparse_point(current):
            raise ValueError("review evidence cannot use a symlink or reparse point")
    if not path.is_file():
        raise FileNotFoundError("review evidence does not exist: {0}".format(supplied))
    return path


def _classification_snapshot(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Return the complete normalized classification summary."""
    return {
        "scientific_disposition": str(metadata.get("scientific_disposition", "unknown")),
        "publication": str(metadata.get("publication", "none")),
        "review_status": str(metadata.get("review_status", DEFAULT_REVIEW_STATUS)),
        "retention_class": str(metadata.get("retention_class", DEFAULT_RETENTION_CLASS)),
    }


def apply_classification_review(
    workspace: CandidateWorkspace,
    *,
    reviewer: str,
    reason: str,
    evidence_paths: Sequence[str],
    scientific_disposition: Optional[str] = None,
    publication: Optional[str] = None,
    review_status: Optional[str] = None,
    retention_class: Optional[str] = None,
) -> Path:
    """Record a human classification decision and update its metadata summary.

    At least one classification field must change.  Every review requires a
    non-empty reviewer, reason, and one or more existing candidate-local files;
    each evidence digest is captured before the record is written.
    """
    if not reviewer.strip():
        raise ValueError("reviewer must not be empty")
    if not reason.strip():
        raise ValueError("review reason must not be empty")
    requested = {
        "scientific_disposition": scientific_disposition,
        "publication": publication,
        "review_status": review_status,
        "retention_class": retention_class,
    }
    if not any(value is not None for value in requested.values()):
        raise ValueError("review must set at least one classification field")
    if not evidence_paths:
        raise ValueError("review requires at least one evidence path")
    for value, allowed, name in (
        (scientific_disposition, SCIENTIFIC_DISPOSITIONS, "scientific disposition"),
        (publication, PUBLICATION_STATES, "publication state"),
        (review_status, REVIEW_STATUSES, "review status"),
        (retention_class, RETENTION_CLASSES, "retention class"),
    ):
        if value is not None and value not in allowed:
            raise ValueError("invalid {0}".format(name))

    evidence: List[Dict[str, str]] = []
    seen_paths = set()
    for supplied in evidence_paths:
        path = _candidate_evidence_path(workspace, supplied)
        relative = path.relative_to(workspace.path.resolve()).as_posix()
        if relative in seen_paths:
            raise ValueError("review evidence paths must be unique")
        seen_paths.add(relative)
        evidence.append({"path": relative, "sha256": _sha256(path)})

    previous = _classification_snapshot(workspace.metadata)
    current = dict(previous)
    for name, value in requested.items():
        if value is not None:
            current[name] = value
    materializes_defaults = any(name not in workspace.metadata for name in current)
    if current == previous and not materializes_defaults:
        raise ValueError("review does not change the classification")

    updated_metadata = dict(workspace.metadata)
    updated_metadata.update(current)
    validate_metadata(updated_metadata, workspace.candidate_id)

    recorded_at = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "recorded_at": recorded_at.isoformat(),
        "reviewer": reviewer.strip(),
        "reason": reason.strip(),
        "evidence": evidence,
        "previous": previous,
        "current": current,
    }
    filename = "classification-review-{0}-{1}.json".format(
        recorded_at.strftime("%Y%m%dT%H%M%S.%fZ"), uuid4().hex
    )
    review_path = workspace.path / REVIEW_DIRECTORY / filename
    _write_json_atomic(review_path, payload)

    metadata_path = workspace.path / "candidate.json"
    _write_json_atomic(metadata_path, updated_metadata)
    return review_path
