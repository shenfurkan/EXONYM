"""Atomic, hash-bound workspace snapshots and rollback for candidate workspaces.

Checkpoint Model
----------------
A checkpoint is a compressed tarball of the mutable, analysis-relevant subset
of one candidate workspace plus a strict JSON manifest that binds the archive
to its SHA-256 digest, the candidate lifecycle state at capture time, and the
per-file hash inventory.  Checkpoints live inside the owning candidate:

    candidate/<candidate_id>/checkpoints/<timestamp>_<label>.tar.gz
    candidate/<candidate_id>/checkpoints/<timestamp>_<label>.manifest.json

Inclusion filter (analysis state only): ``candidate.json``, ``config/``,
``decisions/``, ``outputs/``, ``provenance/``, and ``data/processed/``.
``data/raw/`` is excluded by construction because immutable raw products are
already provenance-bound on disk and dominate archive size; ``checkpoints/``
is excluded to prevent recursive growth.  Restore swaps only the mutable
directories (``config/``, ``decisions/``, ``outputs/``, ``data/processed/``)
and ``candidate.json``; append-only evidence trees such as ``provenance/`` are
never rewritten by a restore.

Security
--------
Extraction validates every member against path traversal (CVE-2007-4559
pattern) before touching the filesystem, rejects links and device members,
and uses :mod:`tarfile`'s ``data`` extraction filter where the running Python
provides it.  The archive digest is verified *before* any existing workspace
file is modified.

Scientific boundary
-------------------
Checkpoints are operational safety nets.  They neither validate candidate
evidence nor alter lifecycle semantics beyond restoring the previously
recorded ``candidate.json``; every restore appends an audit record to
``candidate/<id>/audit_log.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .resources import read_schema_text
from .workspace import CandidateWorkspace

CHECKPOINT_SCHEMA = "checkpoint-manifest.schema.json"
CHECKPOINT_DIRECTORY = "checkpoints"
AUDIT_LOG_FILENAME = "audit_log.jsonl"
MANIFEST_SUFFIX = ".manifest.json"
ARCHIVE_SUFFIX = ".tar.gz"
RESTORE_STAGING_SUFFIX = ".restore-tmp"

# Analysis-state snapshot scope. Order defines archive layout; keep stable.
INCLUDE_ENTRIES: Tuple[str, ...] = (
    "candidate.json",
    "config",
    "decisions",
    "outputs",
    "provenance",
    "data/processed",
)

# Only these mutable paths are replaced by a restore. Append-only evidence
# trees (provenance/, data/raw/) are intentionally never rewritten.
RESTORE_ENTRIES: Tuple[str, ...] = (
    "candidate.json",
    "config",
    "decisions",
    "outputs",
    "data/processed",
)

_LABEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
_CHECKPOINT_ID_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[a-z0-9][a-z0-9_-]*$")

_SHA256_CHUNK_BYTES = 1024 * 1024


def format_archive_size(num_bytes: int) -> str:
    """Render an archive byte count for human-facing tables."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return "{0:.1f} {1}".format(value, unit)
        value /= 1024.0
    return "{0:.1f} GB".format(value)


def _checkpoints_dir(workspace: CandidateWorkspace) -> Path:
    """Return the owning candidate's checkpoints directory (not created)."""
    return workspace.path / CHECKPOINT_DIRECTORY


def _sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_SHA256_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_files(workspace: CandidateWorkspace) -> List[Path]:
    """Collect the deterministic, symlink-free include set relative to root."""
    collected: List[Path] = []
    for entry in INCLUDE_ENTRIES:
        base = workspace.path / entry
        if not base.exists():
            continue
        if base.is_file():
            collected.append(base)
            continue
        if not base.is_dir():
            raise ValueError("checkpoint source is neither file nor directory: {0}".format(base))
        for current, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if not (Path(current) / d).is_symlink())
            for filename in sorted(filenames):
                candidate_path = Path(current) / filename
                if candidate_path.is_symlink() or not candidate_path.is_file():
                    continue
                collected.append(candidate_path)
    return sorted(collected, key=lambda item: item.relative_to(workspace.path).as_posix())


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write strict finite JSON via a temporary file and an atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _validate_manifest(repository_root: Path, manifest: Dict[str, Any]) -> None:
    """Validate one manifest against the mirrored schema resource."""
    import jsonschema

    schema = json.loads(read_schema_text(repository_root, CHECKPOINT_SCHEMA))
    jsonschema.validate(instance=manifest, schema=schema)


def save_checkpoint(workspace: CandidateWorkspace, label: str) -> Path:
    """Create a snapshot of the candidate's analysis state.

    Args:
        workspace: Validated candidate workspace to snapshot.
        label: Short lowercase description; normalized before use.

    Returns:
        Path to the written manifest sidecar.

    Raises:
        ValueError: If the label is empty or contains unsupported characters.
        RuntimeError: If the produced archive or manifest fails validation.
    """
    normalized_label = str(label).strip().lower()
    if not _LABEL_PATTERN.match(normalized_label):
        raise ValueError(
            "checkpoint label must match [a-z0-9][a-z0-9_-]{0,47} after lowercasing"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_id = "{0}_{1}".format(timestamp, normalized_label)
    output_dir = _checkpoints_dir(workspace)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / (checkpoint_id + ARCHIVE_SUFFIX)
    manifest_path = output_dir / (checkpoint_id + MANIFEST_SUFFIX)
    if archive_path.exists() or manifest_path.exists():
        raise RuntimeError("checkpoint identifier already exists: {0}".format(checkpoint_id))

    files = _snapshot_files(workspace)
    if not files:
        raise RuntimeError("refusing to create an empty candidate checkpoint")
    with tarfile.open(archive_path, "w:gz") as archive:
        for file_path in files:
            archive.add(
                str(file_path),
                arcname=file_path.relative_to(workspace.path).as_posix(),
                recursive=False,
            )

    file_records = [
        {
            "path": item.relative_to(workspace.path).as_posix(),
            "sha256": _sha256_file(item),
            "bytes": int(item.stat().st_size),
        }
        for item in files
    ]
    metadata = workspace.metadata
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "checkpoint_id": checkpoint_id,
        "label": normalized_label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lifecycle_state": metadata["lifecycle"]["state"],
        "workflow_phase": metadata["workflow"]["phase"],
        "archive": {
            "filename": archive_path.name,
            "bytes": int(archive_path.stat().st_size),
            "sha256": _sha256_file(archive_path),
        },
        "files": file_records,
    }
    try:
        _validate_manifest(workspace.repository_root, manifest)
    except Exception as exc:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError("checkpoint manifest failed schema validation: {0}".format(exc)) from exc
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def list_checkpoints(workspace: CandidateWorkspace) -> List[Dict[str, Any]]:
    """Return parsed manifests for the candidate, newest first.

    Raises:
        RuntimeError: If any manifest sidecar exists but is unreadable or
            fails its own recorded ``checkpoint_id`` binding.
    """
    output_dir = _checkpoints_dir(workspace)
    if not output_dir.is_dir():
        return []
    records: List[Dict[str, Any]] = []
    for manifest_path in sorted(output_dir.glob("*" + MANIFEST_SUFFIX), reverse=True):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "unreadable checkpoint manifest {0}: {1}".format(manifest_path.name, exc)
            ) from exc
        if not isinstance(payload, dict) or payload.get("checkpoint_id") != manifest_path.name[
            : -len(MANIFEST_SUFFIX)
        ]:
            raise RuntimeError(
                "checkpoint manifest identity mismatch: {0}".format(manifest_path.name)
            )
        records.append(payload)
    return records


def _resolve_checkpoint_files(
    workspace: CandidateWorkspace, checkpoint_id: str
) -> Tuple[Path, Path]:
    """Bind a validated checkpoint id to its archive and manifest paths."""
    if not isinstance(checkpoint_id, str) or not _CHECKPOINT_ID_PATTERN.match(checkpoint_id):
        raise ValueError(
            "checkpoint id must look like <YYYYMMDDThhmmssZ>_<label> with a safe label"
        )
    output_dir = _checkpoints_dir(workspace)
    resolved_root = output_dir.resolve()
    archive_path = output_dir / (checkpoint_id + ARCHIVE_SUFFIX)
    manifest_path = output_dir / (checkpoint_id + MANIFEST_SUFFIX)
    for path in (archive_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError("unknown checkpoint id: {0}".format(checkpoint_id))
        if not str(path.resolve()).startswith(str(resolved_root)):
            raise ValueError("checkpoint path escapes the candidate checkpoints directory")
    return archive_path, manifest_path


def _extract_tar_safe(archive_path: Path, target_dir: Path) -> None:
    """Extract a gzip tarball after rejecting traversal, links, and devices."""
    target_resolved = target_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.islnk() or member.issym():
                raise RuntimeError("checkpoint archive contains link members")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError("checkpoint archive contains non-regular members")
            name = member.name.replace("\\", "/")
            parts = name.split("/")
            if name.startswith("/") or ".." in parts:
                raise RuntimeError("path traversal detected in checkpoint archive: {0}".format(name))
            destination = (target_dir / name).resolve()
            if not str(destination).startswith(str(target_resolved)):
                raise RuntimeError("path traversal detected in checkpoint archive: {0}".format(name))
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                source_handle = archive.extractfile(member)
                if source_handle is None:
                    raise RuntimeError("failed to read member data: {0}".format(name))
                with source_handle, destination.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle)


def _append_audit_record(
    workspace: CandidateWorkspace, payload: Dict[str, Any]
) -> Path:
    """Append one strict-JSON audit line to the candidate audit log."""
    audit_path = workspace.path / AUDIT_LOG_FILENAME
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, allow_nan=False, sort_keys=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return audit_path


def restore_checkpoint(
    workspace: CandidateWorkspace, checkpoint_id: str, assume_yes: bool = False
) -> Dict[str, Any]:
    """Atomically roll the mutable workspace state back to a checkpoint.

    The archive SHA-256 is verified before any workspace byte changes. Each
    restored path is swapped via rename-with-backup so a mid-operation failure
    rolls prior swaps back instead of leaving a mixed state.

    Args:
        workspace: Target candidate workspace.
        checkpoint_id: Identifier as shown by ``checkpoint list``.
        assume_yes: Skip the interactive confirmation prompt.

    Returns:
        Summary dictionary describing the applied restore.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        RuntimeError: On digest mismatch, unsafe archives, missing staged
            metadata, or a declined interactive confirmation.
    """
    archive_path, manifest_path = _resolve_checkpoint_files(workspace, checkpoint_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_digest = _sha256_file(archive_path)
    if actual_digest != manifest["archive"]["sha256"]:
        raise RuntimeError(
            "checkpoint archive digest mismatch for {0}; refusing to restore".format(checkpoint_id)
        )

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not assume_yes:
        if not interactive:
            raise RuntimeError(
                "restore requires --yes when no interactive terminal is available"
            )
        from rich.prompt import Confirm

        if not Confirm.ask(
            "Restore candidate '{0}' to checkpoint {1}? Current config/, decisions/, "
            "outputs/ and data/processed/ will be replaced".format(
                workspace.candidate_id, checkpoint_id
            ),
            default=False,
        ):
            raise RuntimeError("checkpoint restore declined by operator")

    staging = _checkpoints_dir(workspace) / (checkpoint_id + RESTORE_STAGING_SUFFIX)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _extract_tar_safe(archive_path, staging)
        staged_metadata = staging / "candidate.json"
        if not staged_metadata.is_file():
            raise RuntimeError("checkpoint archive is missing candidate.json")

        backups: List[Tuple[Path, Path]] = []
        applied: List[str] = []
        try:
            for entry in RESTORE_ENTRIES:
                target = workspace.path / entry
                backup = workspace.path / (entry.replace("/", "__") + ".restore-bak")
                if not target.exists():
                    continue
                if backup.exists():
                    if backup.is_dir():
                        shutil.rmtree(backup)
                    else:
                        backup.unlink()
                os.replace(target, backup)
                backups.append((target, backup))
            for entry in RESTORE_ENTRIES:
                staged = staging / entry
                target = workspace.path / entry
                if not staged.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, target)
                applied.append(entry)
        except Exception:
            for target, backup in reversed(backups):
                if not target.exists():
                    os.replace(backup, target)
            raise
        else:
            for _, backup in backups:
                if backup.is_dir():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()

        audit_payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "action": "checkpoint_restore",
            "candidate_id": workspace.candidate_id,
            "checkpoint_id": checkpoint_id,
            "archive_sha256": manifest["archive"]["sha256"],
            "restored_entries": applied,
            "result": "success",
        }
        audit_path = _append_audit_record(workspace, audit_payload)
        return {
            "checkpoint_id": checkpoint_id,
            "restored_entries": applied,
            "audit_log": audit_path.relative_to(workspace.path).as_posix(),
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def delete_checkpoint(workspace: CandidateWorkspace, checkpoint_id: str) -> None:
    """Remove one checkpoint archive and its manifest sidecar."""
    archive_path, manifest_path = _resolve_checkpoint_files(workspace, checkpoint_id)
    archive_path.unlink()
    manifest_path.unlink()
