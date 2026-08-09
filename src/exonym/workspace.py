"""Candidate workspace registration, template mirroring, and layout helpers.

Target-specific research is only permitted under ``candidate/<candidate-id>/``.
New workspaces are provisioned by cloning the global ``templates/`` tree and
binding placeholder identifiers. This module contains no target constants.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .resources import iter_template_texts


CANDIDATE_DIRECTORY = "candidate"
METADATA_FILENAME = "candidate.json"
SCHEMA_VERSION = 2
WORKSPACE_DIRECTORIES = (
    "config",
    "data/raw",
    "data/external",
    "data/interim",
    "data/processed",
    "protocols",
    "runs",
    "gates",
    "claims",
    "decisions",
    "provenance",
    "outputs",
    "figures",
    "literature",
    "manuscripts",
    "releases",
    "scripts",
    "tests",
    "docs",
    "tracking",
    "scratch",
)

LIFECYCLE_STATES = ("active", "paused", "stopped", "published", "archived")
WORKFLOW_PHASES = (
    "intake",
    "feasibility",
    "acquisition",
    "vetting",
    "followup",
    "analysis",
    "review",
)
SCIENTIFIC_DISPOSITIONS = (
    "unknown",
    "candidate",
    "unvalidated_candidate",
    "false_positive",
    "validated",
    "confirmed",
    "inconclusive",
)
PUBLICATION_STATES = ("none", "draft", "submitted", "published")
MISSIONS = ("tess", "kepler", "k2", "plato", "cheops")

_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass(frozen=True)
class CandidateWorkspace:
    """A registered candidate and its workspace metadata."""

    repository_root: Path
    candidate_id: str
    path: Path
    metadata: Dict[str, Any]


def validate_candidate_id(candidate_id: str) -> str:
    """Normalize and validate a directory-safe candidate identifier."""
    normalized = candidate_id.strip().lower()
    if not _CANDIDATE_ID.fullmatch(normalized):
        raise ValueError(
            "candidate_id must use lowercase letters, numbers, dots, hyphens, or underscores"
        )
    if normalized.endswith((".", " ")) or normalized.split(".")[0].upper() in _RESERVED_WINDOWS_NAMES:
        raise ValueError("candidate_id is not a safe directory name")
    return normalized


def _candidate_path(repository_root: Path, candidate_id: str) -> Path:
    return repository_root.resolve() / CANDIDATE_DIRECTORY / validate_candidate_id(candidate_id)


def _created_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _placeholder_bindings(metadata: Dict[str, Any]) -> Dict[str, str]:
    identifiers = metadata["identifiers"]
    return {
        "{{CANDIDATE_ID}}": metadata["candidate_id"],
        "{{TOI}}": identifiers.get("toi") or "TBD",
        "{{TIC}}": identifiers.get("tic") or "TBD",
        "{{TIMESTAMP}}": metadata["created_at"],
        "{{STATUS}}": metadata["lifecycle"]["state"],
        "{{PHASE}}": metadata["workflow"]["phase"],
    }


def mirror_templates(
    repository_root: Path,
    workspace: CandidateWorkspace,
    template_texts: Optional[Sequence[Tuple[Path, str]]] = None,
) -> List[Path]:
    """Clone the global template tree into a candidate workspace.

    Template files are copied into their target directories (docs/, protocols/,
    decisions/, tracking/) and placeholder tokens are bound to the candidate
    identity record. Existing files are never overwritten.  Source checkouts
    use their editable root ``templates/`` directory; installed wheels use the
    bundled target-neutral template copy.
    """
    if template_texts is None:
        template_texts = list(iter_template_texts(repository_root))
    bindings = _placeholder_bindings(workspace.metadata)
    written: List[Path] = []
    for relative, content in template_texts:
        destination = workspace.path / relative
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        for token, value in bindings.items():
            content = content.replace(token, value)
        destination.write_text(content, encoding="utf-8")
        written.append(destination)
    return written


def _candidate_readme(metadata: Dict[str, Any]) -> str:
    identifiers = metadata["identifiers"]
    toi = identifiers.get("toi") or "pending verification"
    tic = identifiers.get("tic") or "pending verification"
    return """# {candidate_id}

## State

- Lifecycle: `{lifecycle}`
- Workflow phase: `{workflow}`
- Scientific disposition: `{disposition}`
- Publication: `{publication}`

## Identity

- Candidate workspace: `{candidate_id}`
- TOI: `{toi}`
- TIC: `{tic}`

## First Pass

1. Verify the canonical TOI/TIC metadata from a primary catalog and record the
   source, retrieval date, and ephemeris in `docs/`.
2. Complete the phase documents cloned from `templates/`.
3. Write a feasibility decision before a full data download.
4. Freeze target-specific decisions in `protocols/` before producing outputs.

Run `exonym track {candidate_id}` to view gate progress and
`exonym advance {candidate_id}` to promote phases after gate sign-off.
""".format(
        candidate_id=metadata["candidate_id"],
        lifecycle=metadata["lifecycle"]["state"],
        workflow=metadata["workflow"]["phase"],
        disposition=metadata["scientific_disposition"],
        publication=metadata["publication"],
        toi=toi,
        tic=tic,
    )


def new_candidate_metadata(
    candidate_id: str,
    toi: Optional[str] = None,
    tic: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    mission: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the standard schema v2 identity record for a new candidate."""
    if toi is not None and not re.fullmatch(r"\d{1,7}(\.\d{1,2})?", str(toi)):
        raise ValueError("toi must look like a TOI number, e.g. 1234.01")
    if tic is not None and not re.fullmatch(r"[1-9]\d{0,19}", str(tic)):
        raise ValueError("tic must be a positive integer string")
    if mission is not None and mission not in MISSIONS:
        raise ValueError("mission must be one of: {0}".format(", ".join(MISSIONS)))
    identifiers: Dict[str, Any] = {
        "toi": toi,
        "tic": tic,
        "aliases": [candidate_id],
    }
    if mission is not None:
        identifiers["mission"] = mission
    if tags:
        identifiers["tags"] = list(tags)
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "identifiers": identifiers,
        "lifecycle": {
            "state": "active",
            "state_since": _created_at(),
            "reason": "Initial intake",
        },
        "workflow": {"phase": "intake"},
        "scientific_disposition": "unknown",
        "publication": "none",
        "created_at": _created_at(),
        "notes": "Verify canonical catalog metadata before beginning analysis.",
    }


def validate_metadata(metadata: Dict[str, Any], candidate_id: str) -> None:
    """Minimally validate an identity record without external schema files."""
    if not isinstance(metadata, dict):
        raise ValueError("candidate metadata must be a JSON object")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported candidate schema_version")
    if metadata.get("candidate_id") != candidate_id:
        raise ValueError("candidate metadata ID does not match its directory")
    identifiers = metadata.get("identifiers")
    if not isinstance(identifiers, dict):
        raise ValueError("candidate metadata requires an identifiers object")
    tags = identifiers.get("tags")
    if tags is not None and not (
        isinstance(tags, list) and all(isinstance(tag, str) and tag for tag in tags)
    ):
        raise ValueError("identifiers.tags must be a list of non-empty strings")
    mission = identifiers.get("mission")
    if mission is not None and mission not in MISSIONS:
        raise ValueError("invalid mission identifier")
    lifecycle = metadata.get("lifecycle")
    if not isinstance(lifecycle, dict) or lifecycle.get("state") not in LIFECYCLE_STATES:
        raise ValueError("invalid lifecycle state")
    workflow = metadata.get("workflow")
    if not isinstance(workflow, dict) or workflow.get("phase") not in WORKFLOW_PHASES:
        raise ValueError("invalid workflow phase")
    if metadata.get("scientific_disposition") not in SCIENTIFIC_DISPOSITIONS:
        raise ValueError("invalid scientific disposition")
    if metadata.get("publication") not in PUBLICATION_STATES:
        raise ValueError("invalid publication state")


def create_candidate(
    repository_root: Path,
    candidate_id: str,
    toi: Optional[str] = None,
    tic: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    mission: Optional[str] = None,
) -> CandidateWorkspace:
    """Create a registered candidate workspace without overwriting existing work."""
    repository_root = repository_root.resolve()
    normalized_id = validate_candidate_id(candidate_id)
    path = _candidate_path(repository_root, normalized_id)
    if path.exists():
        raise FileExistsError("candidate workspace already exists: {0}".format(path))
    existing = [candidate.candidate_id for candidate in discover_candidates(repository_root)]
    if any(other.casefold() == normalized_id.casefold() for other in existing):
        raise FileExistsError("candidate ID collides with an existing workspace")

    # Resolve templates before mutating the candidate tree.  A missing or empty
    # template source must not leave behind a partial workspace.
    template_texts = list(iter_template_texts(repository_root))

    metadata = new_candidate_metadata(normalized_id, toi=toi, tic=tic, tags=tags, mission=mission)
    path.mkdir(parents=True)
    for relative_path in WORKSPACE_DIRECTORIES:
        (path / relative_path).mkdir(parents=True)
    (path / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (path / "README.md").write_text(_candidate_readme(metadata), encoding="utf-8")
    workspace = CandidateWorkspace(repository_root, normalized_id, path, metadata)
    mirror_templates(repository_root, workspace, template_texts=template_texts)
    return workspace


def load_candidate(repository_root: Path, candidate_id: str) -> CandidateWorkspace:
    """Load and validate a registered candidate workspace."""
    repository_root = repository_root.resolve()
    normalized_id = validate_candidate_id(candidate_id)
    path = _candidate_path(repository_root, normalized_id)
    metadata_path = path / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError("candidate metadata not found: {0}".format(metadata_path))

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid candidate metadata: {0}".format(metadata_path)) from exc
    validate_metadata(metadata, normalized_id)
    return CandidateWorkspace(repository_root, normalized_id, path, metadata)


def discover_candidates(repository_root: Path) -> List[CandidateWorkspace]:
    """Return registered candidate workspaces ordered by identifier."""
    candidate_root = repository_root.resolve() / CANDIDATE_DIRECTORY
    if not candidate_root.is_dir():
        return []
    candidates = []
    for path in sorted(candidate_root.iterdir(), key=lambda item: item.name):
        if path.is_dir() and (path / METADATA_FILENAME).is_file():
            candidates.append(load_candidate(repository_root, path.name))
    return candidates


def workspace_layout(candidate: CandidateWorkspace) -> Dict[str, Path]:
    """Return named standard paths for a candidate workspace."""
    paths = {"workspace": candidate.path, "metadata": candidate.path / METADATA_FILENAME}
    for relative_path in WORKSPACE_DIRECTORIES:
        paths[relative_path] = candidate.path / relative_path
    return paths
