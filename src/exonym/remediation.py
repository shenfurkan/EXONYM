"""Safe repairs for derived candidate records.

Repairs never alter raw products, candidate metadata, scientific values, or
claim status. A physical digest may be refreshed only when a stored semantic
digest proves the derived content is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from zipfile import BadZipFile

import numpy as np

from .workspace import CandidateWorkspace, discover_candidates


_VOLATILE_JSON_FIELDS = {"generated_at", "generated_utc", "completed_at", "started_at"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _semantic_value(item)
            for key, item in sorted(value.items())
            if key not in _VOLATILE_JSON_FIELDS
        }
    if isinstance(value, list):
        return [_semantic_value(item) for item in value]
    return value


def semantic_json_sha256(path: Path) -> str:
    """Hash JSON content after removing timestamps that do not change its meaning."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    encoded = json.dumps(_semantic_value(payload), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def numerical_npz_sha256(path: Path) -> str:
    """Hash NPZ arrays by numerical values, independent of archive container bytes."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            array = np.ascontiguousarray(np.asarray(archive[name]))
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _read_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_object(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_detrending_manifests(workspace: CandidateWorkspace) -> List[str]:
    actions: List[str] = []
    for manifest_path in sorted((workspace.path / "outputs").glob("detrending_manifest.*.json")):
        manifest = _read_object(manifest_path)
        if manifest is None:
            continue
        artifact = manifest.get("artifact")
        if not isinstance(artifact, dict):
            continue
        relative = artifact.get("path")
        expected_semantic = artifact.get("data_sha256")
        if not isinstance(relative, str) or not isinstance(expected_semantic, str):
            continue
        artifact_path = workspace.path / relative
        try:
            semantically_unchanged = (
                artifact_path.is_file()
                and numerical_npz_sha256(artifact_path) == expected_semantic
            )
        except (BadZipFile, EOFError, OSError, ValueError):
            semantically_unchanged = False
        if not semantically_unchanged:
            continue
        current_digest = _sha256(artifact_path)
        if artifact.get("sha256") == current_digest:
            continue
        artifact["sha256"] = current_digest
        _write_object(manifest_path, manifest)
        actions.append("refreshed {0} artifact digest".format(manifest_path.relative_to(workspace.path).as_posix()))
    return actions


def _refresh_search_manifests(workspace: CandidateWorkspace) -> List[str]:
    actions: List[str] = []
    for manifest_path in sorted((workspace.path / "outputs").glob("*_search_manifest*.json")):
        manifest = _read_object(manifest_path)
        if manifest is None:
            continue
        relative = manifest.get("result_path")
        expected_semantic = manifest.get("result_semantic_sha256")
        if not isinstance(relative, str) or not isinstance(expected_semantic, str):
            continue
        result_path = workspace.path / relative
        if not result_path.is_file() or semantic_json_sha256(result_path) != expected_semantic:
            continue
        current_digest = _sha256(result_path)
        if manifest.get("result_sha256") == current_digest:
            continue
        previous_result_digest = manifest.get("result_sha256")
        previous_manifest_digest = _sha256(manifest_path)
        manifest["result_sha256"] = current_digest
        _write_object(manifest_path, manifest)
        actions.append("refreshed {0} result digest".format(manifest_path.relative_to(workspace.path).as_posix()))
        if manifest.get("schema") == "exonym-bls-search-manifest-1":
            actions.extend(
                _refresh_bls_bound_transit_configs(
                    workspace,
                    manifest_path,
                    manifest,
                    previous_result_digest,
                    previous_manifest_digest,
                    current_digest,
                )
            )
    return actions


def _refresh_bls_bound_transit_configs(
    workspace: CandidateWorkspace,
    manifest_path: Path,
    manifest: Dict[str, Any],
    previous_result_digest: object,
    previous_manifest_digest: str,
    current_result_digest: str,
) -> List[str]:
    """Refresh only configs bound to the exact pre-repair BLS records."""
    result_path = manifest.get("result_path")
    if not isinstance(result_path, str) or not isinstance(previous_result_digest, str):
        return []
    manifest_relative = manifest_path.relative_to(workspace.path).as_posix()
    current_manifest_digest = _sha256(manifest_path)
    actions: List[str] = []
    config_paths = [workspace.path / "config" / "transit_config.json"]
    config_paths.extend(sorted((workspace.path / "config" / "signals").glob("transit_config.*.json")))
    for config_path in config_paths:
        payload = _read_object(config_path)
        provenance = payload.get("bls_provenance") if isinstance(payload, dict) else None
        result = provenance.get("result") if isinstance(provenance, dict) else None
        config_manifest = provenance.get("manifest") if isinstance(provenance, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("source") != "candidate-data-bls"
            or not isinstance(result, dict)
            or not isinstance(config_manifest, dict)
            or result.get("path") != result_path
            or result.get("sha256") != previous_result_digest
            or config_manifest.get("path") != manifest_relative
            or config_manifest.get("sha256") != previous_manifest_digest
        ):
            continue
        result["sha256"] = current_result_digest
        config_manifest["sha256"] = current_manifest_digest
        _write_object(config_path, payload)
        actions.append("refreshed {0} BLS provenance".format(config_path.relative_to(workspace.path).as_posix()))
    return actions


def _sync_triage(workspace: CandidateWorkspace) -> List[str]:
    triage_path = workspace.path / "decisions" / "automated_triage.json"
    if not triage_path.is_file():
        return []
    from .engines import run_automated_triage

    run_automated_triage(workspace)
    return ["synchronized decisions/automated_triage.json"]


def remediate_candidate_drift(repository_root: Path) -> Dict[str, List[str]]:
    """Repair derivation drift that preserves candidate-owned scientific values."""
    actions: Dict[str, List[str]] = {}
    for workspace in discover_candidates(repository_root):
        candidate_actions = _refresh_detrending_manifests(workspace)
        candidate_actions.extend(_refresh_search_manifests(workspace))
        try:
            candidate_actions.extend(_sync_triage(workspace))
        except (OSError, RuntimeError, ValueError) as exc:
            candidate_actions.append("triage sync skipped: {0}".format(exc))
        if candidate_actions:
            actions[workspace.candidate_id] = candidate_actions
    return actions
