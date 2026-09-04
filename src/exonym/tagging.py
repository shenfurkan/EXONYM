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
    disposition: Optional[str] = None,
    publication: Optional[str] = None,
    lifecycle: Optional[str] = None,
    review_status: Optional[str] = None,
    retention_class: Optional[str] = None,
) -> List[CandidateWorkspace]:
    """Filter candidate workspaces by optional metadata fields.

    Every supplied criterion must match. Omitting a criterion leaves that
    field unrestricted.

    Args:
        candidates: Candidate workspaces to consider.
        tag: Optional exact metadata tag.
        phase: Optional workflow phase.
        mission: Optional originating mission identifier.
        disposition: Optional scientific disposition.
        publication: Optional publication state.
        lifecycle: Optional lifecycle state.
        review_status: Optional human-review status.
        retention_class: Optional operational storage label.

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
        if disposition is not None and candidate.metadata.get("scientific_disposition") != disposition:
            continue
        if publication is not None and candidate.metadata.get("publication") != publication:
            continue
        if lifecycle is not None and candidate.metadata.get("lifecycle", {}).get("state") != lifecycle:
            continue
        if review_status is not None and candidate.metadata.get("review_status", "unreviewed") != review_status:
            continue
        if retention_class is not None and candidate.metadata.get("retention_class", "hot") != retention_class:
            continue
        filtered.append(candidate)
    return filtered


def evaluate_habitable_zone_tag(
    insolation_earth: float,
    inner_flux_earth: float,
    outer_flux_earth: float,
) -> Optional[str]:
    """Apply an evidence-supplied habitable-zone tag boundary.

    This administrative helper deliberately does not calculate stellar
    habitable-zone limits.  Such limits depend on the candidate's stellar
    effective temperature and the selected Kopparapu et al. (2013, ApJ 765,
    131; DOI: 10.1088/0004-637X/765/2/131) climate boundary.  Callers must
    therefore supply both limits from a candidate-owned, reviewed calculation
    and retain its provenance separately.

    Args:
        insolation_earth: Candidate incident flux in Earth-insolation units.
        inner_flux_earth: Evidence-derived inner flux boundary in the same
            units.
        outer_flux_earth: Evidence-derived outer flux boundary in the same
            units.

    Returns:
        ``"HabitableZoneCandidate"`` only when the supplied flux lies within
        the supplied inclusive interval, otherwise ``None``.  The tag is
        administrative and is not a habitability characterization or claim.

    Raises:
        ValueError: If any supplied flux is non-finite, non-positive, or the
            interval is inverted.
    """
    values = (insolation_earth, inner_flux_earth, outer_flux_earth)
    try:
        insolation, inner, outer = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("insolation and habitable-zone boundaries must be finite positive numbers") from exc
    if not all(value > 0.0 for value in (insolation, inner, outer)):
        raise ValueError("insolation and habitable-zone boundaries must be positive")
    if inner > outer:
        raise ValueError("inner_flux_earth must not exceed outer_flux_earth")
    if inner <= insolation <= outer:
        return "HabitableZoneCandidate"
    return None
