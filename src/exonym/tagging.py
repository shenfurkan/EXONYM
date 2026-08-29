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
    validate_metadata,
)


def _tags(metadata: dict) -> List[str]:
    return list(metadata.get("identifiers", {}).get("tags", []))


def add_tags(workspace: CandidateWorkspace, tags: Sequence[str]) -> List[str]:
    """Append distinct, nonempty tags to one candidate record.

    Tag spelling is preserved and existing tags remain in their original
    order. The updated metadata is validated before it is written.

    Args:
        workspace: Candidate workspace whose metadata owns the tags.
        tags: Proposed tags; empty values and already-present values are
            ignored.

    Returns:
        The complete ordered tag list after any additions.

    Raises:
        ValueError: If the resulting candidate metadata is invalid.
    """
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
    """Return whether a candidate has one exact metadata tag.

    Args:
        workspace: Candidate workspace to inspect.
        tag: Case-sensitive tag to look up.

    Returns:
        ``True`` when ``tag`` is present in ``identifiers.tags``.
    """
    return tag in _tags(workspace.metadata)


def filter_candidates(
    candidates: Iterable[CandidateWorkspace],
    tag: Optional[str] = None,
    phase: Optional[str] = None,
    mission: Optional[str] = None,
) -> List[CandidateWorkspace]:
    """Filter candidate workspaces by optional metadata fields.

    Every supplied criterion must match. Omitting a criterion leaves that
    field unrestricted.

    Args:
        candidates: Candidate workspaces to consider.
        tag: Optional exact metadata tag.
        phase: Optional workflow phase.
        mission: Optional originating mission identifier.

    Returns:
        Candidate workspaces that satisfy all supplied criteria, preserving
        input order.
    """
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
    """Return the metadata tag for the configured insolation interval.

    The interval is used only to assign a candidate-management tag. It is not
    a physical characterization or a habitability claim.

    Args:
        insolation_earth: Positive incident flux in Earth-insolation units.

    Returns:
        ``"HabitableZoneCandidate"`` for values in the configured inclusive
        interval, otherwise ``None``.

    Raises:
        ValueError: If ``insolation_earth`` is not positive.
    """
    if insolation_earth <= 0:
        raise ValueError("insolation_earth must be positive")
    if 0.32 <= insolation_earth <= 1.11:
        return "HabitableZoneCandidate"
    return None
