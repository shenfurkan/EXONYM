"""Candidate-local acceleration cache for integrity verification.

The cache associates file size and nanosecond modification time with parsed
JSON and SHA-256 results. It avoids repeated reads during one verification
operation while keeping cache entries scoped to the owning candidate workspace.
Persisted entries are additionally bound to the resolved repository/workspace
location and filesystem identity, so a cache copied into another checkout is
discarded before it can satisfy a fingerprint lookup.

Scientific boundary:
    A fingerprint cache is an optimisation, not provenance or a scientific
    reproducibility guarantee. The workspace binding prevents accidental cache
    reuse after a copy or relocation; callers can request a fresh audit whenever
    the trustworthiness of local filesystem metadata is insufficient.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional


_CACHE_FILENAME = ".exonym-verify-cache.json"
_CACHE_VERSION = 2
_ACTIVE_CACHE: ContextVar[Optional["CandidateVerificationCache"]] = ContextVar(
    "exonym_verification_cache", default=None
)


def _file_fingerprint(path: Path) -> Dict[str, int]:
    stat = path.stat()
    return {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CandidateVerificationCache:
    """Persist workspace-bound file digests below owning candidate workspaces.

    Cache records are accepted only when their scope hash matches the current
    resolved repository and workspace paths plus their filesystem identities.
    Moving or copying a candidate workspace safely converts cache hits into
    misses; it does not alter candidate-owned scientific artifacts.
    """

    def __init__(self, repository_root: Path, *, enabled: bool = True) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.candidate_root = self.repository_root / "candidate"
        self.enabled = enabled
        self._states: Dict[Path, Dict[str, Any]] = {}
        self._dirty: set[Path] = set()
        self.hash_hits = 0
        self.hash_misses = 0
        self.json_hits = 0
        self.json_misses = 0

    def _workspace_scope(self, workspace: Path) -> str:
        """Return a stable local binding that invalidates copied cache records."""
        workspace = workspace.resolve()
        repository_stat = self.repository_root.stat()
        workspace_stat = workspace.stat()
        material = {
            "repository_root": str(self.repository_root),
            "repository_device": int(repository_stat.st_dev),
            "repository_inode": int(repository_stat.st_ino),
            "workspace": str(workspace),
            "workspace_device": int(workspace_stat.st_dev),
            "workspace_inode": int(workspace_stat.st_ino),
        }
        return hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _workspace_for(self, path: Path) -> Optional[Path]:
        try:
            relative = Path(path).resolve().relative_to(self.candidate_root)
        except (OSError, ValueError):
            return None
        if len(relative.parts) < 2 or relative.parts[0].startswith("_"):
            return None
        return self.candidate_root / relative.parts[0]

    def _state_for(self, workspace: Path) -> Dict[str, Any]:
        workspace = workspace.resolve()
        if workspace in self._states:
            return self._states[workspace]
        cache_path = workspace / "outputs" / _CACHE_FILENAME
        scope = self._workspace_scope(workspace)
        state: Dict[str, Any] = {
            "version": _CACHE_VERSION,
            "workspace_scope": scope,
            "files": {},
        }
        if self.enabled and cache_path.is_file():
            try:
                loaded = json.loads(cache_path.read_text(encoding="utf-8"))
                if (
                    isinstance(loaded, dict)
                    and loaded.get("version") == _CACHE_VERSION
                    and loaded.get("workspace_scope") == scope
                    and isinstance(loaded.get("files"), dict)
                ):
                    state = loaded
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        self._states[workspace] = state
        return state

    def _record_for(self, path: Path) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        workspace = self._workspace_for(path)
        if workspace is None:
            return None, None
        state = self._state_for(workspace)
        try:
            relative = Path(path).resolve().relative_to(workspace.resolve()).as_posix()
        except (OSError, ValueError):
            return None, None
        record = state["files"].get(relative)
        return state, record if isinstance(record, dict) else None

    def sha256(self, path: Path) -> str:
        """Return a cached SHA-256 only when size and mtime still match."""
        path = Path(path)
        if not self.enabled:
            self.hash_misses += 1
            return _sha256(path)
        state, record = self._record_for(path)
        fingerprint = _file_fingerprint(path)
        if (
            record is not None
            and record.get("mtime_ns") == fingerprint["mtime_ns"]
            and record.get("size") == fingerprint["size"]
            and isinstance(record.get("sha256"), str)
        ):
            self.hash_hits += 1
            return record["sha256"]
        digest = _sha256(path)
        self.hash_misses += 1
        if state is not None:
            workspace = self._workspace_for(path)
            assert workspace is not None
            relative = path.resolve().relative_to(workspace.resolve()).as_posix()
            state["files"][relative] = {**fingerprint, "sha256": digest}
            self._dirty.add(workspace.resolve())
        return digest

    def read_candidate_json(self, path: Path, parser: Callable[[str], object]) -> object:
        """Parse and cache a registered candidate metadata record by fingerprint."""
        path = Path(path)
        if path.name != "candidate.json" or not self.enabled:
            self.json_misses += 1
            return parser(path.read_text(encoding="utf-8"))
        state, record = self._record_for(path)
        fingerprint = _file_fingerprint(path)
        if (
            record is not None
            and record.get("mtime_ns") == fingerprint["mtime_ns"]
            and record.get("size") == fingerprint["size"]
            and "json" in record
        ):
            self.json_hits += 1
            return record["json"]
        value = parser(path.read_text(encoding="utf-8"))
        self.json_misses += 1
        if state is not None:
            workspace = self._workspace_for(path)
            assert workspace is not None
            relative = path.resolve().relative_to(workspace.resolve()).as_posix()
            existing_digest = (
                record.get("sha256")
                if (
                    isinstance(record, dict)
                    and record.get("mtime_ns") == fingerprint["mtime_ns"]
                    and record.get("size") == fingerprint["size"]
                    and isinstance(record.get("sha256"), str)
                )
                else None
            )
            state["files"][relative] = {
                **fingerprint,
                "json": value,
                **({"sha256": existing_digest} if existing_digest is not None else {}),
            }
            self._dirty.add(workspace.resolve())
        return value

    def save(self) -> None:
        """Atomically persist changed cache states without changing scientific outputs."""
        if not self.enabled:
            return
        for workspace in sorted(self._dirty):
            cache_path = workspace / "outputs" / _CACHE_FILENAME
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(cache_path.name + ".tmp")
            temporary.write_text(
                json.dumps(self._states[workspace], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, cache_path)

    def statistics(self) -> Dict[str, int]:
        return {
            "hash_cache_hits": self.hash_hits,
            "hash_cache_misses": self.hash_misses,
            "candidate_json_cache_hits": self.json_hits,
            "candidate_json_cache_misses": self.json_misses,
        }


def cached_sha256(path: Path) -> str:
    """Hash a file through the active candidate verifier cache when available."""
    cache = _ACTIVE_CACHE.get()
    return cache.sha256(path) if cache is not None else _sha256(Path(path))


def cached_candidate_json(path: Path, parser: Callable[[str], object]) -> object:
    """Parse registered candidate metadata through the active cache when available."""
    cache = _ACTIVE_CACHE.get()
    return cache.read_candidate_json(path, parser) if cache is not None else parser(Path(path).read_text(encoding="utf-8"))


@contextmanager
def candidate_verification_cache(
    repository_root: Path, *, enabled: bool = True
) -> Iterator[CandidateVerificationCache]:
    """Make candidate hash and metadata caching available to schema validation."""
    cache = CandidateVerificationCache(repository_root, enabled=enabled)
    token = _ACTIVE_CACHE.set(cache)
    try:
        yield cache
    finally:
        _ACTIVE_CACHE.reset(token)
        cache.save()
