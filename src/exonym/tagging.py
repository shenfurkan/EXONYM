"""Candidate tag management and metadata query engine.

Tags are stored in ``candidate/<id>/candidate.json`` under
``identifiers.tags`` and can be filtered with ``exonym list --tag``.
"""

from __future__ import annotations

import json
from typing import Iterable, List, Optional, Sequence

from .workspace import (
    CandidateWorkspace,
    METADATA_FILENAME,
    discover_candidates,
    validate_metadata,
)


def _tags(metadata: dict) -> List[str]:
    return list(metadata.get("identifiers", {}).get("tags", []))


def add_tags(workspace: CandidateWorkspace, tags: Sequence[str]) -> List[str]:
    """Append tags to a candidate record (deduplicated, case preserved)."""
    metadata = dict(workspace.metadata)
    current = _tags(metadata)
    seen = set(current)
    additions = [tag for tag in tags if tag and tag not in seen and not (seen.add(tag) or False)]
    if not additions:
        return current
    identifiers = dict(metadata["identifiers"])
    identifiers["tags"] = current + additions
    metadata["identifiers"] = identifiers
    validate_metadata(metadata, workspace.candidate_id)
    path = workspace.path / METADATA_FILENAME
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return identifiers["tags"]


def has_tag(workspace: CandidateWorkspace, tag: str) -> bool:
    return tag in _tags(workspace.metadata)


def filter_candidates(
    candidates: Iterable[CandidateWorkspace],
    tag: Optional[str] = None,
    phase: Optional[str] = None,
    mission: Optional[str] = None,
) -> List[CandidateWorkspace]:
    """Filter candidates by tag, workflow phase, and/or mission."""
    filtered = []
    for candidate in candidates:
        if tag is not None and not has_tag(candidate, tag):
            continue
        if phase is not None and candidate.metadata["workflow"]["phase"] != phase:
            continue
        if mission is not None and candidate.metadata.get("identifiers", {}).get("mission") != mission:
            continue
        filtered.append(candidate)
    return filtered


def evaluate_habitable_zone_tag(insolation_earth: float) -> Optional[str]:
    """Return 'HabitableZoneCandidate' tag if insolation falls within Kopparapu HZ limits [0.32, 1.11]."""
    if insolation_earth <= 0:
        raise ValueError("insolation_earth must be positive")
    if 0.32 <= insolation_earth <= 1.11:
        return "HabitableZoneCandidate"
    return None
