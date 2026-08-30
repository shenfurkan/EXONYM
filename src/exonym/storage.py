"""Read-only candidate storage inventory.

The inventory uses filesystem metadata only.  It does not hash, move, delete,
or rewrite candidate artifacts, making it suitable for the first pass over a
large repository before any retention action is considered.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from stat import S_ISREG
from typing import Any, Dict, List, Optional, Tuple

from .isolation import is_reparse_point
from .workspace import (
    CandidateWorkspace,
    discover_candidates_with_outcomes,
    load_candidate,
)


STORAGE_BUCKETS = (
    "data/raw",
    "data/external",
    "data/interim",
    "data/processed",
    "data/other",
    "outputs",
    "scratch",
    "runs",
    "releases",
    "provenance",
    "figures",
    "other",
)


def _bucket_for(relative_path: Path) -> str:
    """Map a workspace-relative file path to a stable reporting bucket."""
    if relative_path.parts and relative_path.parts[0] == "data":
        if len(relative_path.parts) > 1:
            candidate = "data/{0}".format(relative_path.parts[1])
            if candidate in STORAGE_BUCKETS:
                return candidate
        return "data/other"
    if relative_path.parts and relative_path.parts[0] in STORAGE_BUCKETS:
        return relative_path.parts[0]
    return "other"


def _empty_measurement() -> Dict[str, int]:
    """Return one mutable file-count and byte-count measurement."""
    return {"files": 0, "bytes": 0}


def _inventory_tree(root: Path) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]], int, int]:
    """Measure regular files below ``root`` without reading their contents."""
    total = _empty_measurement()
    buckets = {name: _empty_measurement() for name in STORAGE_BUCKETS}
    reparse_points_skipped = 0
    stat_errors = 0

    def on_walk_error(_: OSError) -> None:
        nonlocal stat_errors
        stat_errors += 1

    for current, directories, files in os.walk(
        str(root), topdown=True, followlinks=False, onerror=on_walk_error
    ):
        safe_directories = []
        for name in directories:
            path = Path(current) / name
            try:
                if is_reparse_point(path):
                    reparse_points_skipped += 1
                    continue
            except OSError:
                stat_errors += 1
                continue
            safe_directories.append(name)
        directories[:] = safe_directories

        for name in files:
            path = Path(current) / name
            try:
                if is_reparse_point(path):
                    reparse_points_skipped += 1
                    continue
                file_stat = path.stat()
                if not S_ISREG(file_stat.st_mode):
                    continue
                size = int(file_stat.st_size)
                relative = path.relative_to(root)
            except (OSError, ValueError):
                stat_errors += 1
                continue
            bucket = buckets[_bucket_for(relative)]
            bucket["files"] += 1
            bucket["bytes"] += size
            total["files"] += 1
            total["bytes"] += size
    return total, buckets, reparse_points_skipped, stat_errors


def inventory_candidate(workspace: CandidateWorkspace) -> Dict[str, Any]:
    """Return a stat-only storage inventory for one candidate workspace."""
    total, buckets, skipped, errors = _inventory_tree(workspace.path.resolve())
    return {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "measurement": "regular-file bytes from filesystem stat; content was not read",
        "total": total,
        "buckets": buckets,
        "reparse_points_skipped": skipped,
        "stat_errors": errors,
    }


def _sum_measurements(inventories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate candidate inventories without re-reading the filesystem."""
    total = _empty_measurement()
    buckets = {name: _empty_measurement() for name in STORAGE_BUCKETS}
    skipped = 0
    errors = 0
    for inventory in inventories:
        measured = inventory["total"]
        total["files"] += measured["files"]
        total["bytes"] += measured["bytes"]
        for name in STORAGE_BUCKETS:
            buckets[name]["files"] += inventory["buckets"][name]["files"]
            buckets[name]["bytes"] += inventory["buckets"][name]["bytes"]
        skipped += inventory["reparse_points_skipped"]
        errors += inventory["stat_errors"]
    return {
        "total": total,
        "buckets": buckets,
        "reparse_points_skipped": skipped,
        "stat_errors": errors,
    }


def build_storage_report(repository_root: Path, candidate_id: Optional[str] = None) -> Dict[str, Any]:
    """Build a read-only storage report for one or all valid candidates."""
    repository_root = Path(repository_root).resolve()
    incomplete: List[Dict[str, str]] = []
    if candidate_id is not None:
        candidates = [load_candidate(repository_root, candidate_id)]
        scope = "candidate"
    else:
        candidates, incomplete = discover_candidates_with_outcomes(repository_root)
        scope = "all-candidates"
    inventories = [inventory_candidate(candidate) for candidate in candidates]
    aggregate = _sum_measurements(inventories)
    return {
        "schema_version": 1,
        "scope": scope,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "measurement": "regular-file bytes from filesystem stat; content was not read",
        "candidates": inventories,
        "incomplete": incomplete,
        **aggregate,
    }
