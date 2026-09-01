"""Candidate workspace registration, template mirroring, and layout helpers.

Directory Ownership Model
--------------------------
The repository enforces a strict two-zone isolation boundary (see
``exonym-core-architecture``).  Target-specific data --- photometry, ephemerides,
pipeline outputs, decisions, and claims --- may *only* reside under
``candidate/<candidate-id>/``.  Every other subdirectory (``src/``, ``schemas/``,
``templates/``, ``tests/``, ``docs/``, ``policy/``) is the protected neutral
zone.  This module defines the 23 standard subdirectories that every candidate
workspace provisions and the metadata schema (v2) that binds each workspace to
the lifecycle and workflow state machine.

Template Mirroring Invariant
-----------------------------
New workspaces are provisioned by cloning the global ``templates/`` tree and
substituting ``{{CANDIDATE_ID}}`` and related identity placeholders.  The
template source is resolved in priority order: (1) a project-local editable
``templates/`` directory, (2) the bundled immutable copy inside the installed
wheel.  A missing *or empty* template tree is a hard error --- it would leave a
workspace without its mandatory protocol, decision, and tracking skeleton files,
preventing the workspace from ever passing its workflow gates.

Schema v2 Metadata Contract
----------------------------
Every workspace is registered by ``candidate.json`` at schema version 2.  The
record carries:

* ``candidate_id`` --- normalized lowercase directory-safe identifier.
* ``identifiers`` --- namespace of survey-alert and catalog handles (aliases,
  mission tag, optional per-signal forwarder).
* ``lifecycle`` --- one of ``active``, ``paused``, ``stopped``, ``published``,
  ``archived``, with a state-transition timestamp and reason.
* ``workflow`` --- current seven-phase position (``intake`` through ``review``).
* ``scientific_disposition`` --- ``unknown``, ``candidate``,
  ``unvalidated_candidate``, ``false_positive``, ``validated``, ``confirmed``,
  or ``inconclusive``.
* ``publication`` --- ``none``, ``draft``, ``submitted``, or ``published``.
* ``review_status`` --- ``unreviewed``, ``triaged``, ``reviewed``, or
  ``adjudicated``.
* ``retention_class`` --- ``hot``, ``warm``, ``cold``, or ``hold``; an
  operational storage label independent of lifecycle and science.

Validation enforces:

* Schema version match (exactly 2).
* Candidate ID matches its enclosing directory name.
* Lifecycle state, workflow phase, disposition, publication, review status, and
  retention values drawn from the closed enumerations declared in this module.
* Non-finite JSON number constants (``Infinity``, ``NaN``) are rejected at the
  parse level; duplicate object keys produce an immediate error.
* Workspace paths and metadata files must never be symlinks, junctions, or
  reparse points (validated by ``exonym.isolation.is_reparse_point``).

Relationship to Feasibility and Triage
---------------------------------------
The workspace lifecycle is defined by ``methods/candidate-feasibility.md``
(12 stop conditions; go/no-go assessment before committing analysis time) and
``methods/engine-execution-and-triage.md`` (routing rule triage =
max(s_screen, s_archive, s_localization, s_activity, s_dilution); FPP sum
formula; DS9 coordinate handling).  The ``analysis`` gate is intentionally
always blocked until calibrated scene-model integration exists.

This module contains no target constants, sector numbers, ephemeris values, or
registered candidate aliases.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .resources import iter_template_texts


# ---------------------------------------------------------------------------
# Directory ownership: every candidate workspace lives under this top-level
# collection directory.  The name is part of the isolation boundary ---
# target-specific data may NOT appear outside it.
# ---------------------------------------------------------------------------
CANDIDATE_DIRECTORY = "candidate"

# The schema-v2 identity record stored in every workspace root.
METADATA_FILENAME = "candidate.json"

# Increment only when the metadata record format changes incompatibly.
# The ``validate_metadata`` gate rejects any other version.
SCHEMA_VERSION = 2
# ---------------------------------------------------------------------------
# 23 standard directories provisioned in every candidate workspace.
#
# config/         Candidate-local YAML/JSON configuration overrides.
# data/raw/       Immutable ingested photometry, pixel-level FITS, and
#                 provenance sidecars keyed by product stem.
# data/external/  Star catalogs, stellar-parameter tables, literature
#                 values ingested from third-party sources.
# data/interim/   Intermediate pipeline products (e.g. extracted light
#                 curves before detrending).
# data/processed/ Detrended light curves, normalized time series, and
#                 other analysis-ready artifacts.
# protocols/      Frozen analysis protocols and methodological decisions
#                 recorded before producing outputs.
# runs/           Per-engine run directories with SHA-256 input/output
#                 manifests and status records.
# gates/          Phase-gate sign-off artifacts produced by
#                 ``exonym.gatekeeper`` (mandatory-item checklists).
# claims/         Candidate-local claim evidence (FPP reports, validation
#                 summaries).  The analysis gate intentionally blocks
#                 writing ``fpp_claim.json``.
# decisions/      Recorded go/no-go, rejection, and hold decisions with
#                 supporting evidence references.
# provenance/     SHA-256 manifests for reproducibility bundles.
# outputs/        Numerical outputs, tables, CSV exports from pipeline
#                 stages.
# figures/        Publication-quality and diagnostic plots.
# literature/     PDFs, references, and annotated bibliography files.
# manuscripts/    Draft manuscripts, figure compositions, and LaTeX
#                 sources.
# releases/       Frozen release bundles produced by ``exonym freeze``.
# scripts/        Candidate-specific analysis and plotting scripts.
# tests/          Candidate-local validation and regression tests.
# docs/           Narrative documentation, feasibility reports, and
#                 technical notes.
# tracking/       Gate-progress telemetry (Markdown checklists parsed by
#                 ``exonym.tracking``).
# scratch/        Temporary files not tracked by provenance or releases.
# ---------------------------------------------------------------------------
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

# Closed enumeration of lifecycle states.  Transitions are recorded with a
# UTC timestamp and free-text reason in the metadata record.
LIFECYCLE_STATES = ("active", "paused", "stopped", "published", "archived")
# Ordered seven-phase workflow defined by ``methods/candidate-feasibility.md``
# and ``methods/engine-execution-and-triage.md``.  Phase advancement requires
# checked ``[MANDATORY]`` tracking items and a passing phase-specific gate.
# The ``analysis`` gate is intentionally always blocked.
WORKFLOW_PHASES = (
    "intake",
    "feasibility",
    "acquisition",
    "vetting",
    "followup",
    "analysis",
    "review",
)
# Scientific disposition taxonomy.  ``unvalidated_candidate`` is the
# machine-produced label written by ``exonym vet``; ``validated`` and
# ``confirmed`` require external review evidence not produced by the
# pipeline alone.  ``false_positive`` may be assigned by a decisive
# rejection record or an external adjudication.
SCIENTIFIC_DISPOSITIONS = (
    "unknown",
    "candidate",
    "unvalidated_candidate",
    "false_positive",
    "validated",
    "confirmed",
    "inconclusive",
)
# Publication lifecycle, independent of the scientific disposition.
PUBLICATION_STATES = ("none", "draft", "submitted", "published")
# Human-review progress is independent of scientific disposition.  A triage
# result may route work, but only an explicit review record reaches
# ``adjudicated``.
REVIEW_STATUSES = ("unreviewed", "triaged", "reviewed", "adjudicated")
# Storage labels are operational hints only.  ``cold`` never authorizes local
# deletion; an independently verified archive is required before any purge.
RETENTION_CLASSES = ("hot", "warm", "cold", "hold")

# Recognized survey missions for metadata tagging and ingestion routing.
MISSIONS = ("tess", "kepler", "k2", "plato", "cheops")

# NUMERICAL_GUARD: pattern enforces directory-safe candidate IDs.
# Must start with alphanumeric, contain only lowercase letters, digits,
# dots, hyphens, or underscores.  This prevents path traversal, shell
# metacharacter injection, and directory separator injection.
_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

# Per-signal fixed-width suffix: exactly a dot followed by two digits.
# This deterministic length prevents glob ambiguity in per-signal paths.
_SIGNAL_SUFFIX = re.compile(r"^\.\d{2}$")

# NUMERICAL_GUARD: Windows filesystem reserves these names regardless
# of extension.  Rejecting them at validation time prevents a candidate
# workspace from being created at a path that Windows cannot access
# reliably.
_RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass(frozen=True)
class CandidateWorkspace:
    """A registered candidate and its workspace metadata.

    This immutable record binds a candidate identifier to its filesystem
    location and schema-v2 identity record.  It is produced by
    ``create_candidate`` or ``load_candidate`` and consumed by every
    downstream module that needs access to candidate-owned artifacts.

    Parameters
    ----------
    repository_root : pathlib.Path
        Absolute resolved path to the repository root.
    candidate_id : str
        Normalized lowercase directory-safe identifier.
    path : pathlib.Path
        Absolute path to the workspace directory
        (``<repository_root>/candidate/<candidate_id>``).
    metadata : dict
        Schema-v2 identity record loaded from ``candidate.json``,
        validated against the closed enumerations in this module.
    """

    repository_root: Path
    candidate_id: str
    path: Path
    metadata: Dict[str, Any]


def validate_candidate_id(candidate_id: str) -> str:
    """Normalize and validate a directory-safe candidate identifier.

    Applies three layers of defense:

    1. **Normalization**: strips leading/trailing whitespace and lowercases
       the entire string so that identifier lookup is case-insensitive
       while the filesystem entry is always lowercase.
    2. **Character-class gate**: rejects identifiers that contain characters
       outside ``[a-z0-9._-]`` or that start with a non-alphanumeric
       character.  This prevents path traversal via ``..``, shell
       metacharacter injection, and directory separator embedding.
    3. **Windows reserved-name check**: rejects identifiers whose first
       dot-separated component collides with a reserved Windows device
       name (``CON``, ``PRN``, ``AUX``, ``NUL``, ``COM1``-``COM9``,
       ``LPT1``-``LPT9``) or that end with a dot or space --- both
       illegal on Windows NTFS.

    Parameters
    ----------
    candidate_id : str
        Raw identifier string, possibly with mixed case or whitespace.

    Returns
    -------
    str
        Normalized lowercase identifier safe for use as a directory name.

    Raises
    ------
    ValueError
        If the identifier contains forbidden characters, collides with a
        reserved name, or has an illegal terminal character.
    """
    normalized = candidate_id.strip().lower()
    if not _CANDIDATE_ID.fullmatch(normalized):
        raise ValueError(
            "candidate_id must use lowercase letters, numbers, dots, hyphens, or underscores"
        )
    if normalized.endswith((".", " ")) or normalized.split(".")[0].upper() in _RESERVED_WINDOWS_NAMES:
        raise ValueError("candidate_id is not a safe directory name")
    return normalized


def validate_signal_suffix(signal: Optional[str]) -> Optional[str]:
    """Validate one fixed-width per-signal suffix before it is used in a path.

    Per-signal artifact directories and filenames use a deterministic
    ``.NN`` suffix (dot followed by exactly two digits, e.g. ``.01``,
    ``.02``).  This fixed-width convention:

    * Prevents lexicographic sorting artefacts (``.2`` would sort after
      ``.10`` in a naive string sort).
    * Makes glob patterns unambiguous: ``*.01.*`` uniquely selects the
      first signal's artifacts.
    * Avoids collision with the reserved Windows device-name rule applied
      to candidate IDs (``.01`` is never a reserved component).

    Parameters
    ----------
    signal : str or None
        The raw signal suffix (e.g. ``".01"``) or ``None`` to indicate
        the top-level (primary) signal.

    Returns
    -------
    str or None
        The validated suffix if non-``None``, otherwise ``None``.

    Raises
    ------
    ValueError
        If ``signal`` is not a string matching the ``.NN`` format.
    """
    if signal is None:
        return None
    if not isinstance(signal, str) or _SIGNAL_SUFFIX.fullmatch(signal) is None:
        raise ValueError("signal must use the .NN format")
    return signal


def _candidate_path(repository_root: Path, candidate_id: str) -> Path:
    """Return the canonical flat workspace path for a candidate ID.

    This enforces the directory ownership invariant: a candidate's
    workspace is *always* ``<repository_root>/candidate/<candidate_id>/``.
    No other location is valid for candidate-owned data.  The path is
    resolved before return to eliminate any symlink indirection.

    New workspaces are provisioned under a lifecycle group (``active/``)
    via :func:`_candidate_group_path`; this flat-path helper remains for
    collision checks and legacy compat.
    """
    return repository_root.resolve() / CANDIDATE_DIRECTORY / validate_candidate_id(candidate_id)


def _candidate_group_path(repository_root: Path, candidate_id: str, group: str) -> Path:
    """Return the workspace path nested under a lifecycle-group directory."""
    return repository_root.resolve() / CANDIDATE_DIRECTORY / group / validate_candidate_id(candidate_id)


def _resolve_candidate_path(repository_root: Path, candidate_id: str) -> Path:
    """Find a candidate workspace at its actual filesystem location.

    Resolution order:
    1. Flat legacy path ``candidate/<id>/``.
    2. Lifecycle-group paths ``candidate/<group>/<id>/`` for every
       recognised group (``active``, ``paused``, …).

    Returns the first match whose ``candidate.json`` exists.  When no
    match is found, the flat path is returned so that the caller can
    issue a descriptive ``FileNotFoundError`` pointing at the canonical
    location.
    """
    normalized = validate_candidate_id(candidate_id)
    repo = repository_root.resolve()
    flat = repo / CANDIDATE_DIRECTORY / normalized
    if (flat / METADATA_FILENAME).is_file():
        return flat
    for group in LIFECYCLE_STATES:
        group_path = repo / CANDIDATE_DIRECTORY / group / normalized
        if (group_path / METADATA_FILENAME).is_file():
            return group_path
    return flat


def _created_at() -> str:
    """Return the current UTC time as a compact ISO-8601 string with ``Z`` suffix.

    The microsecond field is truncated so that timestamps are human-readable
    and stable across round-trips.  The ``Z`` suffix (rather than ``+00:00``)
    is chosen for terseness in metadata records.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _placeholder_bindings(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Build the token-replacement map for template mirroring.

    Every ``{{TOKEN}}`` in a template file is replaced with its bound value
    before the file is written to the candidate workspace.  Tokens that
    resolve to ``None`` (e.g. an unset survey-alert identifier) are replaced
    with the literal string ``"TBD"`` so that the resulting file remains
    syntactically valid and the operator can see what needs to be filled in.

    The binding map is:

    * ``{{CANDIDATE_ID}}`` → ``metadata["candidate_id"]``
    * ``{{TOI}}`` → ``identifiers["toi"]`` or ``"TBD"``
    * ``{{TIC}}`` → ``identifiers["tic"]`` or ``"TBD"``
    * ``{{TIMESTAMP}}`` → ``metadata["created_at"]`` (UTC ISO-8601)
    * ``{{STATUS}}`` → ``metadata["lifecycle"]["state"]``
    * ``{{PHASE}}`` → ``metadata["workflow"]["phase"]``

    Parameters
    ----------
    metadata : dict
        Schema-v2 identity record with ``candidate_id``, ``identifiers``,
        ``lifecycle``, ``workflow``, and ``created_at`` keys.

    Returns
    -------
    dict
        Mapping from template token string to replacement string.
    """
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

    Template files are copied into their target directories (``docs/``,
    ``protocols/``, ``decisions/``, ``tracking/``) and every ``{{TOKEN}}``
    placeholder is bound to the candidate identity record.  Existing files
    are *never* overwritten --- this protects operator-authored content from
    accidental clobbering during re-provisioning.

    Template resolution follows the priority chain implemented by
    ``exonym.resources.iter_template_texts``:

    1. A project-local editable ``templates/`` directory at the repository
       root, if present and non-empty.
    2. The bundled immutable template copy inside the installed wheel
       (``exonym._resources/templates/``).

    A missing or empty template source raises an error *before* the
    candidate workspace tree is mutated, preventing a partial workspace
    that can never pass its workflow gates.

    Parameters
    ----------
    repository_root : pathlib.Path
        Resolved repository root used to locate the ``templates/`` source.
    workspace : CandidateWorkspace
        Target workspace whose ``path`` receives the mirrored files.
    template_texts : sequence of (Path, str) or None
        Pre-fetched template payloads.  When ``None``, templates are
        resolved via ``iter_template_texts(repository_root)``.

    Returns
    -------
    list of pathlib.Path
        Absolute paths of newly written template files (existing files
        are excluded from this list).
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
    """Generate the workspace README.md from schema-v2 identity metadata.

    The rendered README is a human-facing dashboard that displays the
    current lifecycle state, workflow phase, scientific disposition,
    publication status, and identifier bindings.  It is regenerated on
    ``create_candidate`` but never overwritten automatically afterward ---
    operators are expected to maintain it alongside their workspace.

    Parameters
    ----------
    metadata : dict
        Schema-v2 identity record.

    Returns
    -------
    str
        Markdown text for the workspace root README.
    """
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
    """Return the standard schema-v2 identity record for a new candidate.

    The returned dictionary is the authoritative bootstrap record written
    as ``candidate.json`` during ``create_candidate``.  It initialises:

    * ``schema_version`` to 2 (the only version accepted by
      ``validate_metadata``).
    * ``lifecycle.state`` to ``"active"`` with an ISO-8601 UTC timestamp
      and reason ``"Initial intake"``.
    * ``workflow.phase`` to ``"intake"`` (the first of seven phases).
    * ``scientific_disposition`` to ``"unknown"``.
    * ``publication`` to ``"none"``.
    * ``review_status`` to ``"unreviewed"``.
    * ``retention_class`` to ``"hot"``.
    * ``identifiers.aliases`` seeded with ``[candidate_id]``.

    Parameters
    ----------
    candidate_id : str
        Normalized lowercase directory-safe identifier (already validated).
    toi : str or None
        Optional survey-alert identifier; validated against the pattern
        ``\\d{1,7}(\\.\\d{1,2})?`` (e.g. ``"1234.01"``).
    tic : str or None
        Optional catalog identifier; must be a positive integer string
        matching ``[1-9]\\d{0,19}``.
    tags : sequence of str or None
        Optional metadata tags attached to ``identifiers.tags``.
    mission : str or None
        Survey mission tag; must be one of ``MISSIONS`` if provided.

    Returns
    -------
    dict
        Schema-v2 identity record ready for JSON serialization.

    Raises
    ------
    ValueError
        If ``toi``, ``tic``, or ``mission`` fail their format validations.
    """
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
        "review_status": "unreviewed",
        "retention_class": "hot",
        "created_at": _created_at(),
        "notes": "Verify canonical catalog metadata before beginning analysis.",
    }


def validate_metadata(metadata: Dict[str, Any], candidate_id: str) -> None:
    """Minimally validate a schema-v2 identity record without external schema files.

    This inline validator enforces the core schema contract that every
    downstream consumer depends on.  It checks:

    1. **Schema version** --- must be exactly 2.  A mismatch indicates a
       record from a different (possibly incompatible) EXONYM version.
    2. **Candidate ID binding** --- ``metadata["candidate_id"]`` must match
       the directory name, preventing identity/directory drift.
    3. **Closed enumerations** --- ``lifecycle.state``, ``workflow.phase``,
       ``scientific_disposition``, and ``publication`` must each be drawn
       from the respective module-level tuple.  Unknown values are rejected.
    4. **Identifiers shape** --- ``identifiers`` must be a dict, optional
       ``tags`` must be a list of non-empty strings, and optional
       ``mission`` must be a recognized value.

    This function is intentionally self-contained (no JSON Schema library
    dependency) so that it can gate workspace loading before any optional
    dependencies are verified.

    Parameters
    ----------
    metadata : dict
        Parsed ``candidate.json`` contents.
    candidate_id : str
        Normalized identifier expected to match the enclosing directory.

    Raises
    ------
    ValueError
        If any validation check fails, with a message naming the violated
        constraint.
    """
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
    if metadata.get("review_status", "unreviewed") not in REVIEW_STATUSES:
        raise ValueError("invalid review status")
    if metadata.get("retention_class", "hot") not in RETENTION_CLASSES:
        raise ValueError("invalid retention class")


def _parse_finite_float(value: str) -> float:
    """Parse one metadata number without allowing an infinite value.

    # NUMERICAL_GUARD: JSON's ``Infinity`` and ``-Infinity`` are not
    # finite numeric values and would corrupt downstream arithmetic.
    # Rejecting them at the parse level prevents silent propagation.
    """
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_nonfinite_json_constant(value: str) -> object:
    """Reject non-standard JSON constants in candidate metadata.

    # NUMERICAL_GUARD: JSON's ``NaN`` and signed ``Infinity`` constants
    # are not valid numbers for metadata records.  They cannot be
    # serialized losslessly, compared reliably, or used in downstream
    # calculations.
    """
    raise ValueError("non-finite JSON constant: {0}".format(value))


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    """Reject duplicate metadata keys rather than applying last-key-wins semantics.

    Python's default ``json.loads`` silently keeps the last value when a
    key appears more than once.  For candidate metadata this is dangerous:
    a typo that duplicates a key could mask a stale or incorrect value.
    Rejecting duplicates at parse time makes the error immediately visible.
    """
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {0}".format(key))
        result[key] = value
    return result


def create_candidate(
    repository_root: Path,
    candidate_id: str,
    toi: Optional[str] = None,
    tic: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    mission: Optional[str] = None,
) -> CandidateWorkspace:
    """Create a registered candidate workspace without overwriting existing work.

    This is the sole entry point for provisioning a new candidate workspace.
    It executes the following ordered steps, each gated by the preceding one:

    1. **Validate and normalize** the candidate ID (``validate_candidate_id``).
    2. **Reject collision** --- if a workspace or case-insensitive ID match
       already exists, raise ``FileExistsError``.
    3. **Resolve template source** via ``iter_template_texts``.  This call
       fails fast if the template tree is missing or empty, preventing
       mutation of ``candidate/`` before the workspace's skeleton is known
       to be complete.
    4. **Build schema-v2 metadata** (``new_candidate_metadata``) and write
       ``candidate.json`` with sorted keys and ``parse_constant``-safe
       serialization.
    5. **Provision the 23 standard subdirectories** defined by
       ``WORKSPACE_DIRECTORIES``.
    6. **Write README.md** from the metadata template.
    7. **Mirror templates** --- clone every file from the resolved template
       source, substituting ``{{TOKEN}}`` placeholders.  Existing files are
       skipped.

    The returned ``CandidateWorkspace`` is an immutable snapshot that can be
    passed to every downstream consumer.

    Parameters
    ----------
    repository_root : pathlib.Path
        Absolute or relative path to the repository root (resolved before use).
    candidate_id : str
        Raw identifier; will be normalized and validated.
    toi : str or None
        Optional survey-alert identifier.
    tic : str or None
        Optional catalog identifier.
    tags : sequence of str or None
        Optional metadata tags.
    mission : str or None
        Survey mission key (e.g. ``"tess"``).

    Returns
    -------
    CandidateWorkspace
        The newly created workspace record.

    Raises
    ------
    FileExistsError
        If the workspace directory already exists or a case-insensitive
        ID collision is detected.
    FileNotFoundError
        If the template source is missing or empty.
    ValueError
        If the candidate ID or any metadata field fails validation.
    """
    repository_root = repository_root.resolve()
    normalized_id = validate_candidate_id(candidate_id)
    path = _candidate_path(repository_root, normalized_id)
    if path.exists():
        raise FileExistsError("candidate workspace already exists: {0}".format(path))
    # Case-insensitive collision check across all discovered workspaces
    # (both flat legacy and grouped).
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
    """Load and validate a registered candidate workspace.

    This is the primary entry point for opening an existing workspace.
    It performs a sequence of safety checks before returning:

    1. **Resolve and normalize** the repository root and candidate ID.
    2. **Reject reparse points** --- symlinks, junctions, and mount points
       on the workspace directory or its metadata file are disallowed.
       This prevents workspace identity spoofing and cross-filesystem
       boundary violations (see ``exonym.isolation.is_reparse_point``).
    3. **Require metadata file** --- ``candidate.json`` must exist as a
       regular file.
    4. **Parse with guards** --- the JSON decoder uses
       ``_parse_finite_float``, ``_reject_nonfinite_json_constant``, and
       ``_reject_duplicate_json_keys`` to reject malformed or dangerous
       input at the parse level.
    5. **Validate** the parsed record against the closed enumerations and
       candidate-ID/directory binding (``validate_metadata``).

    Any failure in the chain is reported as a ``ValueError`` (parse,
    validation, or reparse-point issues) or ``FileNotFoundError`` (missing
    metadata file), with a message that includes the affected path.

    Parameters
    ----------
    repository_root : pathlib.Path
        Path to the repository root (resolved to absolute before use).
    candidate_id : str
        Raw identifier; normalized and validated internally.

    Returns
    -------
    CandidateWorkspace
        Immutable workspace record with parsed and validated metadata.

    Raises
    ------
    FileNotFoundError
        If ``candidate.json`` does not exist at the expected path.
    ValueError
        If the workspace is a reparse point, the metadata fails to parse,
        or validation against the schema contract fails.
    """
    repository_root = repository_root.resolve()
    normalized_id = validate_candidate_id(candidate_id)
    path = _resolve_candidate_path(repository_root, normalized_id)
    metadata_path = path / METADATA_FILENAME
    from .isolation import is_reparse_point

    # NUMERICAL_GUARD: reparse points (symlinks, junctions, mount points)
    # could alias a workspace to a different filesystem location, breaking
    # the ownership invariant that candidate data lives only under
    # candidate/<id>/.
    if (path.exists() and is_reparse_point(path)) or (
        metadata_path.exists() and is_reparse_point(metadata_path)
    ):
        raise ValueError("candidate workspace contains a symlink or reparse point")
    if not metadata_path.is_file():
        raise FileNotFoundError("candidate metadata not found: {0}".format(metadata_path))

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8"),
            parse_float=_parse_finite_float,
            parse_constant=_reject_nonfinite_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "invalid candidate metadata: {0}: {1}".format(metadata_path, exc)
        ) from exc
    validate_metadata(metadata, normalized_id)
    return CandidateWorkspace(repository_root, normalized_id, path, metadata)


def discover_candidates(repository_root: Path) -> List[CandidateWorkspace]:
    """Return candidate workspaces from flat and grouped locations.

    Candidate workspaces may live directly under ``candidate/`` (legacy
    flat layout) or under lifecycle-group subdirectories
    (``candidate/active/``, ``candidate/paused/``, …).  Collection
    directories beginning with an underscore are reserved for
    candidate-local cohorts (e.g. ``_surveys/``) and are never
    individual candidates.

    Discovery walks ``candidate/``, classifies each top-level entry, and
    collects workspaces from both flat and grouped layouts.  Returns are
    sorted lexicographically by candidate ID so that CLI listing and
    batch operations see a stable order.

    Parameters
    ----------
    repository_root : pathlib.Path
        Repository root resolved before scanning.

    Returns
    -------
    list of CandidateWorkspace
        Loaded workspaces, ordered by identifier.
    """
    candidate_root = repository_root.resolve() / CANDIDATE_DIRECTORY
    if not candidate_root.is_dir():
        return []
    candidates = []
    for path in sorted(candidate_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        if (path / METADATA_FILENAME).is_file():
            # Legacy flat candidate.
            cid = path.name.lower()
            try:
                candidates.append(load_candidate(repository_root, cid))
            except (FileNotFoundError, ValueError):
                continue
        elif path.name in LIFECYCLE_STATES:
            # Lifecycle-group directory: descend and collect.
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                if not child.is_dir() or child.name.startswith("_"):
                    continue
                if not (child / METADATA_FILENAME).is_file():
                    continue
                try:
                    candidates.append(load_candidate(repository_root, child.name))
                except (FileNotFoundError, ValueError):
                    continue
    return candidates


def discover_candidates_with_outcomes(
    repository_root: Path,
) -> Tuple[List[CandidateWorkspace], List[Dict[str, str]]]:
    """Discover valid workspaces and retain invalid direct entries as outcomes.

    Batch automation must not silently omit a workspace merely because
    its metadata is incomplete or invalid. The normal discovery API keeps its
    valid-workspace-only behavior; this companion supplies an operator-visible
    incomplete outcome for batch callers.

    Workspaces are collected from both the legacy flat layout and any
    lifecycle-group directories (``active/``, ``paused/``, …).
    """
    candidate_root = repository_root.resolve() / CANDIDATE_DIRECTORY
    if not candidate_root.is_dir():
        return [], []
    candidates: List[CandidateWorkspace] = []
    incomplete: List[Dict[str, str]] = []

    def _collect(path: Path) -> None:
        if not (path / METADATA_FILENAME).is_file():
            incomplete.append(
                {
                    "candidate_id": path.name,
                    "status": "incomplete",
                    "reason": "Candidate workspace has no candidate metadata.",
                }
            )
            return
        try:
            candidate = load_candidate(repository_root, path.name)
        except (OSError, FileNotFoundError, ValueError) as exc:
            incomplete.append(
                {
                    "candidate_id": path.name,
                    "status": "incomplete",
                    "reason": "Candidate workspace could not be loaded: {0}".format(exc),
                }
            )
        else:
            candidates.append(candidate)

    for path in sorted(candidate_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        if path.name in LIFECYCLE_STATES:
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                if child.is_dir() and not child.name.startswith("_"):
                    _collect(child)
        else:
            _collect(path)
    return candidates, incomplete


def workspace_layout(candidate: CandidateWorkspace) -> Dict[str, Path]:
    """Return named standard paths for a candidate workspace.

    Produces a flat dictionary mapping logical names to absolute paths.
    The dictionary always includes:

    * ``"workspace"`` → workspace root.
    * ``"metadata"`` → ``candidate.json`` path.
    * One entry per item in ``WORKSPACE_DIRECTORIES``, keyed by the
      relative path string (e.g. ``"data/raw"``, ``"figures"``).

    Callers can use this dictionary to locate standard directories without
    hardcoding the ``candidate/<id>/`` prefix.

    Parameters
    ----------
    candidate : CandidateWorkspace
        A loaded workspace record.

    Returns
    -------
    dict
        Mapping from logical name (str) to absolute ``pathlib.Path``.
    """
    paths = {"workspace": candidate.path, "metadata": candidate.path / METADATA_FILENAME}
    for relative_path in WORKSPACE_DIRECTORIES:
        paths[relative_path] = candidate.path / relative_path
    return paths


LIFECYCLE_GROUP_MAP = {
    "active": "active",
    "paused": "paused",
    "stopped": "stopped",
    "published": "published",
    "archived": "archived",
}


def move_candidate(workspace: CandidateWorkspace, group: str) -> CandidateWorkspace:
    """Atomically relocate a workspace to a lifecycle-group directory.

    The rename happens within ``candidate/``, so it is a same-filesystem
    atomic operation on all major platforms.  The returned
    ``CandidateWorkspace`` has an updated ``path`` pointing at the new
    location.
    """
    if group not in LIFECYCLE_STATES:
        raise ValueError("invalid lifecycle group: {0}".format(group))
    target_dir = _candidate_group_path(
        workspace.repository_root, workspace.candidate_id, group
    )
    if workspace.path.resolve() == target_dir.resolve():
        return workspace
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    workspace.path.rename(target_dir)
    return CandidateWorkspace(
        workspace.repository_root,
        workspace.candidate_id,
        target_dir,
        dict(workspace.metadata),
    )


def organize_candidates(
    repository_root: Path,
    *,
    candidate_id: Optional[str] = None,
    by: str = "lifecycle",
    apply: bool = False,
) -> Dict[str, Any]:
    """Move candidates into lifecycle-group directories (dry-run when ``apply=False``)."""
    repository_root = Path(repository_root).resolve()
    incomplete: List[Dict[str, str]] = []
    if candidate_id is not None:
        try:
            candidates = [load_candidate(repository_root, candidate_id)]
            scope = "candidate"
        except (FileNotFoundError, ValueError) as exc:
            return {
                "schema_version": 1,
                "scope": "candidate",
                "by": by,
                "apply": apply,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": {"total": 0, "moved": 0, "unchanged": 0, "errors": 1},
                "incomplete": [{"candidate_id": candidate_id, "reason": str(exc)}],
                "candidates": [],
            }
    else:
        candidates, incomplete = discover_candidates_with_outcomes(repository_root)
        scope = "all-candidates"

    total = 0
    moved = 0
    unchanged = 0
    errors = 0
    items: List[Dict[str, Any]] = []

    for ws in candidates:
        total += 1
        lifecycle = ws.metadata.get("lifecycle", {}).get("state")
        if lifecycle not in LIFECYCLE_GROUP_MAP:
            errors += 1
            items.append({"candidate_id": ws.candidate_id, "status": "error",
                          "reason": "unknown lifecycle state"})
            continue
        target_group = LIFECYCLE_GROUP_MAP[lifecycle]
        current_parent = ws.path.resolve().parent.name
        if current_parent == target_group:
            unchanged += 1
            items.append({"candidate_id": ws.candidate_id, "status": "unchanged",
                          "from_group": current_parent, "to_group": target_group})
            continue
        if apply:
            try:
                moved_ws = move_candidate(ws, target_group)
                moved += 1
                items.append({
                    "candidate_id": ws.candidate_id,
                    "from": ws.path.relative_to(repository_root).as_posix(),
                    "to": moved_ws.path.relative_to(repository_root).as_posix(),
                    "from_group": current_parent, "to_group": target_group,
                    "status": "moved",
                })
            except OSError as exc:
                errors += 1
                items.append({"candidate_id": ws.candidate_id, "status": "error",
                              "reason": str(exc)})
        else:
            moved += 1
            target_path = _candidate_group_path(
                repository_root, ws.candidate_id, target_group
            )
            items.append({
                "candidate_id": ws.candidate_id,
                "from": ws.path.relative_to(repository_root).as_posix(),
                "to": target_path.relative_to(repository_root).as_posix(),
                "from_group": current_parent, "to_group": target_group,
                "status": "proposed",
            })

    return {
        "schema_version": 1,
        "scope": scope,
        "by": by,
        "apply": apply,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": total,
            "moved": moved if apply else 0,
            "unchanged": unchanged,
            "proposed": moved if not apply else 0,
            "errors": errors,
        },
        "incomplete": incomplete,
        "candidates": items,
    }
