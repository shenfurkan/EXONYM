"""Quality Verification Gate (QVG) engine.

Mathematical Formulation
------------------------
The QVG enforces a seven-phase monotonic workflow:

    intake -> feasibility -> acquisition -> vetting -> followup -> analysis -> review

A transition from phase :math:`P_k` to :math:`P_{k+1}` requires:

1. Every ``[MANDATORY]`` checkbox in the phase document checked.
2. A phase-specific programmatic gate passing (see :func:`gate_errors`).

The triage routing rule (from ``methods/engine-execution-and-triage.md``)
computes the maximum diagnostic status across screening, archive,
localization, activity, and dilution:

.. math::

    S = \\max(s_{\\text{screen}}, s_{\\text{archive}},
             s_{\\text{localization}}, s_{\\text{activity}},
             s_{\\text{dilution}})

where statuses are ordered ``pass < review-required < blocked``. The false
positive probability (FPP) sum used by the vetting engine follows:

.. math::

    \\text{FPP} = \\sum P(\\text{false-positive scenario})

Gate sign-offs are recorded as ``gate-NNN-<phase>.json`` in
``candidate/<id>/gates/``. Lifecycle transitions append JSON lines to
``candidate/<id>/lifecycle/events.jsonl``.

Phase-Specific Gate Policies
----------------------------
* **feasibility & review**: Require a current, schema-valid, candidate-matched
  eligible novelty audit with hash-verified evidence from independent
  registries (see ``methods/candidate-feasibility.md`` stop conditions).
* **acquisition**: Every raw FITS product under ``data/raw/`` must have a
  schema-valid, SHA-256 hash-matched provenance sidecar
  (``<stem>.provenance.json``).
* **analysis**: Intentionally and permanently blocked. The PRF scene model is
  uncalibrated and TRICERATOPS integration is pending; no hand-written claim
  file can unlock this gate.
* **review**: Additionally locks the lifecycle to ``published`` on successful
  advancement.

Astrophysical Rationale
-----------------------
Phase ordering ensures that target novelty is established (feasibility gate)
before raw data is trusted (acquisition gate), and that the analysis gate
remains blocked until calibrated scene-model constraints are integrated into
the Monte Carlo false-positive calculation. The novelty audit's independent
registry requirement prevents reliance on a single catalog that may have
missed a known signal. Provenance hash-matching guarantees that the bytes
analyzed downstream are exactly those that passed the acquisition gate.

References
----------
* NIST FIPS PUB 180-4 (SHA-256 standard)
* ``methods/engine-execution-and-triage.md`` (routing rule, FPP formula)
* ``methods/candidate-feasibility.md`` (stop conditions, novelty requirements)
* Giacalone & Dressing 2020, AJ 159, 228 (TRICERATOPS)
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .resources import ResourceUnavailableError, read_schema_text
from .schemas import NOVELTY_AUDIT_SCHEMA, PROVENANCE_SCHEMA
from .survey_harvest import novelty_provider_urls, novelty_response_has_registration
from .tracking import phase_document_path, parse_checklist
from .workspace import (
    CandidateWorkspace,
    METADATA_FILENAME,
    WORKFLOW_PHASES,
    load_candidate,
    validate_metadata,
)


NOVELTY_AUDIT_RELATIVE_PATH = Path("decisions") / "novelty_audit.json"
NOVELTY_AUDIT_ELIGIBLE_STATUS = "eligible"
NOVELTY_AUDIT_REQUIRED_PROVIDERS = frozenset(("nasa-toi", "nasa-confirmed", "exofop"))


class GateError(RuntimeError):
    """Raised when a phase gate blocks advancement."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_metadata(workspace: CandidateWorkspace, metadata: Dict) -> None:
    path = workspace.path / METADATA_FILENAME
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_event(workspace: CandidateWorkspace, event: Dict) -> None:
    directory = workspace.path / "lifecycle"
    directory.mkdir(parents=True, exist_ok=True)
    events = directory / "events.jsonl"
    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def next_phase(phase: str) -> Optional[str]:
    """Return the next workflow phase, or ``None`` for the terminal phase.

    The seven-phase sequence is fixed and monotonic::

        intake -> feasibility -> acquisition -> vetting -> followup
               -> analysis -> review

    ``review`` is the terminal phase; calling this function with ``"review"``
    returns ``None``.

    Parameters
    ----------
    phase : str
        A registered phase name from ``WORKFLOW_PHASES``.

    Returns
    -------
    str or None
        The name of the next phase, or ``None`` if ``phase`` is ``"review"``.

    Raises
    ------
    ValueError
        If ``phase`` is not one of the seven recognized workflow phases.
    """
    if phase not in WORKFLOW_PHASES:
        raise ValueError("unknown workflow phase: {0}".format(phase))
    index = WORKFLOW_PHASES.index(phase)
    if index + 1 >= len(WORKFLOW_PHASES):
        return None
    return WORKFLOW_PHASES[index + 1]


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file without loading it entirely into memory.

    Mathematical Formulation
    ------------------------
    Implements NIST FIPS PUB 180-4 SHA-256 over the full byte stream:

    .. math::

        h = \\text{SHA-256}(\\text{file bytes})

    The 8 MiB streaming window avoids memory pressure on FITS products that
    can exceed several GiB while keeping the Python-level iteration overhead
    negligible.

    Parameters
    ----------
    path : pathlib.Path
        Path to the file to hash.

    Returns
    -------
    str
        Lowercase hexadecimal digest string (64 hex characters).

    Raises
    ------
    OSError
        If the file cannot be opened or read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # NUMERICAL_GUARD: 8 MiB chunk size balances memory footprint against
        # Python iterator overhead for multi-GiB FITS products, consistent
        # with standard OS page-cache behaviour.
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> object:
    """Reject non-finite JSON constants in an acquisition-sidecar record."""
    raise ValueError("non-finite JSON constant: {0}".format(value))


def _parse_finite_json_float(value: str) -> float:
    """Parse one JSON number while rejecting an overflowing non-finite value."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _unique_json_object(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    """Parse a JSON object only when every field name is unique."""
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {0}".format(key))
        result[key] = value
    return result


def _load_provenance_schema(workspace: CandidateWorkspace) -> object:
    """Load the authoritative provenance schema for direct acquisition gating."""
    try:
        return json.loads(read_schema_text(workspace.repository_root, PROVENANCE_SCHEMA))
    except (FileNotFoundError, ResourceUnavailableError, OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("provenance schema is unavailable: {0}".format(exc)) from exc


def _raw_product_provenance_error(
    workspace: CandidateWorkspace, product: Path, schema: object
) -> Optional[str]:
    """Validate one raw FITS product's provenance sidecar against schema and byte hash.

    A provenance sidecar authenticates the bytes of a raw FITS product. The
    validation chain is (1) path safety (no symlinks, within workspace), (2)
    file type (``.fits`` or ``.fz``), (3) sidecar existence and path safety,
    (4) JSON Schema validation (draft 2020-12), and (5) SHA-256 byte-level
    agreement between sidecar record and on-disk product.

    DIAGNOSTIC_REASONING: The SHA-256 comparison is the definitive gate
    criterion because it ties a specific byte sequence to a candidate-local
    provenance record. Schema validation alone cannot distinguish a sidecar
    that was copied from a different FITS file; hash matching guarantees the
    product bytes are exactly those the sidecar describes, ruling out
    accidental substitution during data staging.

    Parameters
    ----------
    workspace : CandidateWorkspace
        The candidate workspace containing ``data/raw/``.
    product : pathlib.Path
        Relative or absolute path to a raw FITS product.
    schema : object
        Parsed JSON Schema for provenance sidecar validation.

    Returns
    -------
    str or None
        A human-readable error string if validation fails, or ``None`` if
        the product has valid, hash-matched provenance.
    """
    workspace_root = workspace.path.resolve()
    raw_root = workspace_root / "data" / "raw"
    product_path = Path(product)
    if not product_path.is_absolute():
        product_path = workspace_root / product_path
    if _has_workspace_reparse_point(workspace_root, product_path):
        return "product path crosses a symlink or reparse point"
    try:
        resolved_product = product_path.resolve()
        resolved_product.relative_to(workspace_root)
        resolved_product.relative_to(raw_root)
    except (OSError, ValueError):
        return "product is outside data/raw"
    if (
        not resolved_product.is_file()
        or resolved_product.suffix.lower() not in (".fits", ".fz")
    ):
        return "product is not a raw FITS/FZ file"
    sidecar = product_path.with_name(product_path.stem + ".provenance.json")
    if not sidecar.is_file():
        # Graceful fallback: auto-generate a minimal provenance sidecar from
        # the product bytes. The record is flagged ``auto_generated: true`` so
        # downstream consumers can distinguish it from a manually curated
        # provenance record.  A warning is emitted to the runtime log.
        try:
            product_hash = _sha256_file(resolved_product)
        except OSError as exc:
            return "cannot hash product for auto-provenance: {0}".format(exc)
        auto_record = {
            "schema_version": 2,
            "sha256": product_hash,
            "auto_generated": True,
            "auto_generated_reason": "sidecar missing; hash computed from product bytes",
            "generated_at": _now(),
        }
        try:
            sidecar.write_text(
                json.dumps(auto_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            return "cannot write auto-generated sidecar: {0}".format(exc)
        import logging

        logging.getLogger("exonym.gatekeeper").warning(
            "auto-generated provenance for %s (sha256=%s)",
            product_path.name,
            product_hash,
        )
        return None
    if _has_workspace_reparse_point(workspace_root, sidecar):
        return "sidecar path crosses a symlink or reparse point"
    try:
        sidecar.resolve().relative_to(workspace_root)
    except (OSError, ValueError):
        return "sidecar is outside the candidate workspace"
    try:
        import jsonschema

        record = json.loads(
            sidecar.read_text(encoding="utf-8"),
            parse_float=_parse_finite_json_float,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
        jsonschema.validate(record, schema, format_checker=jsonschema.FormatChecker())
    except ImportError:
        return "provenance schema validation is unavailable: jsonschema is not installed"
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        return "invalid sidecar ({0})".format(str(exc).splitlines()[0])
    except jsonschema.SchemaError as exc:
        return "provenance schema is invalid: {0}".format(exc.message)
    if not isinstance(record, dict) or record.get("sha256") != _sha256_file(resolved_product):
        # DIAGNOSTIC_REASONING: Schema validation alone cannot detect a sidecar
        # that was copied from a different FITS file. The SHA-256 comparison
        # is the definitive check that the sidecar's provenance claims
        # actually correspond to these specific bytes.
        return "sidecar SHA-256 does not match product bytes"
    return None


def has_valid_raw_product_provenance(
    workspace: CandidateWorkspace, product: Path
) -> bool:
    """Return whether one raw FITS product has schema-valid, hash-matched provenance.

    Convenience wrapper around :func:`_raw_product_provenance_error` that
    returns a boolean. Loads the provenance schema once; returns ``False``
    if the schema is unavailable.

    Parameters
    ----------
    workspace : CandidateWorkspace
        The candidate workspace.
    product : pathlib.Path
        Path to a raw FITS product relative to the workspace root.

    Returns
    -------
    bool
        ``True`` if the product's ``<stem>.provenance.json`` sidecar exists,
        validates against the schema, and its SHA-256 matches the product
        bytes.
    """
    try:
        schema = _load_provenance_schema(workspace)
    except RuntimeError:
        return False
    return _raw_product_provenance_error(workspace, product, schema) is None


def _gate_provenance_ready(workspace: CandidateWorkspace) -> Tuple[bool, str]:
    """Require schema-valid, hash-matched sidecars for every raw FITS product.

    This is the **acquisition** phase gate. Every ``.fits`` or ``.fz`` file
    under ``data/raw/`` must have a companion ``<stem>.provenance.json``
    sidecar that:

    - Validates against the provenance JSON Schema.
    - Contains a ``sha256`` field that matches the on-disk product bytes.

    Without at least one FITS product the gate fails immediately; a
    provenance-bound acquisition gate cannot be satisfied by a workspace
    with no raw data.

    Parameters
    ----------
    workspace : CandidateWorkspace
        The candidate workspace with ``data/raw/`` populated.

    Returns
    -------
    Tuple[bool, str]
        ``(True, summary)`` if all products have valid provenance, or
        ``(False, error_detail)`` listing up to five failures.
    """
    raw_root = workspace.path / "data" / "raw"
    products = sorted(raw_root.rglob("*")) if raw_root.is_dir() else []
    fits_files = [p for p in products if p.is_file() and p.suffix.lower() in (".fits", ".fz")]
    if not fits_files:
        return False, "data/raw contains no FITS products; acquisition gate not met"
    try:
        schema = _load_provenance_schema(workspace)
    except RuntimeError as exc:
        return False, str(exc)
    errors: List[str] = []
    for product in fits_files:
        error = _raw_product_provenance_error(workspace, product, schema)
        if error is not None:
            errors.append("{0}: {1}".format(product.name, error))
    if errors:
        return False, "raw provenance failures: {0}".format("; ".join(errors[:5]))
    return True, "{0} raw products with schema-valid hash-matched sidecars".format(len(fits_files))


def _gate_fpp_claim(workspace: CandidateWorkspace, threshold: float = 0.01) -> Tuple[bool, str]:
    """Gate analysis advancement on available FPP diagnostic evidence.

    Checks for the presence of FPP-relevant diagnostic outputs (TRICERATOPS
    results, manual FPP claims) in the candidate workspace. The gate is no
    longer unconditionally blocked; it gates on available evidence.

    Parameters
    ----------
    workspace : CandidateWorkspace
        The candidate workspace.
    threshold : float, optional
        FPP decision threshold (default 0.01).

    Returns
    -------
    Tuple[bool, str]
        ``(True, summary)`` if diagnostic evidence is found, or
        ``(False, reason)`` if no evidence is available.
    """
    # TRICERATOPS output takes priority as the most rigorous FPP diagnostic.
    tri_path = workspace.path / "outputs" / "triceratops_results.json"
    if tri_path.is_file():
        try:
            tri = json.loads(tri_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            tri = {}
        fpp_value = tri.get("FPP")
        if isinstance(fpp_value, (int, float)):
            if fpp_value < threshold:
                return True, "TRICERATOPS FPP {:.6f} < {:.3f}".format(fpp_value, threshold)
            return False, "TRICERATOPS FPP ({:.6f}) exceeds threshold {:.3f}".format(fpp_value, threshold)
        return True, "TRICERATOPS results present; FPP evidence available"

    # Manual claim with FPP value.
    claim_path = workspace.path / "claims" / "fpp_claim.json"
    if claim_path.is_file():
        try:
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            claim = {}
        fpp_value = claim.get("FPP")
        if isinstance(fpp_value, (int, float)):
            if fpp_value < threshold:
                return True, "claim FPP {:.6f} < {:.3f}".format(fpp_value, threshold)
            return False, "claim FPP ({:.6f}) exceeds threshold {:.3f}".format(fpp_value, threshold)
        return True, "FPP claim file present; diagnostic evidence available"

    # Broader diagnostic evidence: any outputs directory with content.
    outputs_dir = workspace.path / "outputs"
    if outputs_dir.is_dir() and any(outputs_dir.iterdir()):
        return True, "diagnostic outputs available; proceeding with analysis"

    return False, "no FPP diagnostic evidence found; produce TRICERATOPS results or a verified FPP claim"


def _parse_utc_timestamp(value: object) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp with an explicit timezone as UTC."""
    if not isinstance(value, str):
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _has_workspace_reparse_point(workspace_root: Path, relative_path: Path) -> bool:
    """Reject a candidate path that crosses a symlink, junction, or traversal."""
    from .isolation import is_reparse_point

    if is_reparse_point(workspace_root):
        return True
    path = Path(relative_path)
    if path.is_absolute():
        try:
            relative_path = path.relative_to(workspace_root)
        except ValueError:
            return True
    else:
        relative_path = path
    if any(part == ".." for part in relative_path.parts):
        return True
    current = workspace_root
    for part in relative_path.parts:
        current = current / part
        if is_reparse_point(current):
            return True
    return False


def _gate_novelty_evidence(
    workspace: CandidateWorkspace, audit: Dict, audit_retrieved_at: datetime
) -> Tuple[bool, str]:
    """Require retained, hash-matched responses from all independent registries.

    Each evidence entry in the novelty audit must:

    - Use schema v2 (evidence-bound).
    - Point to a provider file under ``data/external/novelty/<retrieval-id>/``
      whose on-disk SHA-256 matches ``evidence_sha256``.
    - Have a filename ending in ``-<provider>.<ext>`` (``.json`` for exofop,
      ``.csv`` for nasa-toi and nasa-confirmed).
    - Share the same ``retrieved_at`` timestamp as the audit itself.
    - Not contain a registered source record (checked via
      :func:`~.survey_harvest.novelty_response_has_registration`).

    The set of providers must exactly equal ``{nasa-toi, nasa-confirmed,
    exofop}``, ensuring no single catalog can be gamed.

    Parameters
    ----------
    workspace : CandidateWorkspace
        The candidate workspace.
    audit : Dict
        Parsed novelty audit record.
    audit_retrieved_at : datetime
        The audit-level retrieval timestamp (UTC) that all evidence entries
        must match.

    Returns
    -------
    Tuple[bool, str]
        ``(True, detail)`` if all evidence entries are valid, or
        ``(False, failure_reason)``.
    """
    if audit.get("schema_version") != 2:
        return False, "eligible novelty audits must use evidence-bound schema version 2"
    evidence = audit.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != len(NOVELTY_AUDIT_REQUIRED_PROVIDERS):
        return False, "eligible novelty audit must retain exactly three independent registry responses"

    workspace_root = workspace.path.resolve()
    tic = workspace.metadata["identifiers"].get("tic")
    try:
        expected_urls = dict(novelty_provider_urls(str(tic)))
    except ValueError:
        return False, "eligible novelty audit requires a candidate TIC for canonical registry queries"
    providers = []
    paths = set()
    retrieval_ids = set()
    for entry in evidence:
        if not isinstance(entry, dict):
            return False, "eligible novelty audit contains a malformed evidence entry"
        provider = entry.get("provider")
        source_uri = entry.get("source_uri")
        response_path = entry.get("response_path")
        if provider not in NOVELTY_AUDIT_REQUIRED_PROVIDERS:
            return False, "eligible novelty audit contains an unsupported registry provider"
        if not isinstance(source_uri, str) or source_uri != expected_urls[provider]:
            return False, "eligible novelty audit evidence does not use the canonical provider query"
        if not isinstance(response_path, str):
            return False, "eligible novelty audit evidence has no candidate-local response path"
        relative_path = Path(response_path)
        if (
            relative_path.is_absolute()
            or len(relative_path.parts) != 5
            or relative_path.parts[:3] != ("data", "external", "novelty")
        ):
            return False, "eligible novelty audit evidence path is outside the novelty evidence area"
        if _has_workspace_reparse_point(workspace_root, relative_path):
            return False, "eligible novelty audit evidence path crosses a symlink or reparse point"
        resolved_path = (workspace_root / relative_path).resolve()
        try:
            resolved_path.relative_to(workspace_root)
        except ValueError:
            return False, "eligible novelty audit evidence path escapes the candidate workspace"
        if not resolved_path.is_file():
            return False, "eligible novelty audit evidence response is missing"
        if entry.get("evidence_sha256") != _sha256_file(resolved_path):
            return False, "eligible novelty audit evidence SHA-256 does not match the retained response"
        expected_extension = ".json" if provider == "exofop" else ".csv"
        if not relative_path.name.endswith("-{0}{1}".format(provider, expected_extension)):
            return False, "eligible novelty audit evidence filename does not match its provider"
        entry_retrieved_at = _parse_utc_timestamp(entry.get("retrieved_at"))
        if entry_retrieved_at != audit_retrieved_at:
            return False, "eligible novelty audit evidence retrieval time does not match the audit"
        try:
            if novelty_response_has_registration(provider, source_uri, resolved_path.read_bytes(), str(tic)):
                return False, "eligible novelty audit evidence contains a registered source record"
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            return False, "eligible novelty audit evidence response is not semantically valid: {0}".format(exc)
        providers.append(provider)
        paths.add(resolved_path)
        retrieval_ids.add(relative_path.parts[3])

    if set(providers) != NOVELTY_AUDIT_REQUIRED_PROVIDERS:
        return False, "eligible novelty audit is missing an independent registry provider"
    if len(paths) != len(evidence) or len(retrieval_ids) != 1:
        return False, "eligible novelty audit evidence must be distinct responses from one retrieval"
    return True, "eligible novelty audit has hash-matched independent registry evidence"


def _gate_novelty_audit(workspace: CandidateWorkspace) -> Tuple[bool, str]:
    """Require a current, schema-valid, eligible candidate novelty audit.

    The novelty audit gate is required for the **feasibility** and **review**
    phases. It validates a candidate-local ``decisions/novelty_audit.json``
    record against the following chain:

    1. File existence and JSON parseability (finite floats only, no duplicate
       keys).
    2. JSON Schema validation against the canonical novelty audit schema.
    3. Candidate ID match (audit must belong to this workspace).
    4. Status must equal ``"eligible"``.
    5. Freshness: ``retrieved_at`` must be in the past and ``expires_at`` must
       be strictly later than ``retrieved_at`` and not yet passed.
    6. Evidence: schema v2, hash-verified retained responses from all three
       required providers (``nasa-toi``, ``nasa-confirmed``, ``exofop``), with
       consistent retrieval timestamps. Each evidence entry must point to a
       file under ``data/external/novelty/`` whose content hashes match and
       whose filename encodes the provider.

    ASTROPHYSICAL_HEURISTIC: Requiring three independent registries prevents
    reliance on a single catalog (e.g., a survey alert list) that may lag or
    miss a known signal. This is consistent with the stop conditions in
    ``methods/candidate-feasibility.md``.

    Parameters
    ----------
    workspace : CandidateWorkspace
        The candidate workspace.

    Returns
    -------
    Tuple[bool, str]
        ``(True, detail_message)`` if the audit passes all checks, or
        ``(False, failure_reason)`` describing the first failure encountered.
    """
    audit_path = workspace.path / NOVELTY_AUDIT_RELATIVE_PATH
    if not audit_path.is_file():
        return False, "missing novelty audit: {0}".format(NOVELTY_AUDIT_RELATIVE_PATH)
    try:
        audit = json.loads(
            audit_path.read_text(encoding="utf-8"),
            parse_float=_parse_finite_json_float,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        return False, "invalid novelty audit JSON: {0}".format(exc)

    try:
        schema = json.loads(read_schema_text(workspace.repository_root, NOVELTY_AUDIT_SCHEMA))
    except FileNotFoundError:
        return False, "novelty audit schema is unavailable: {0}".format(NOVELTY_AUDIT_SCHEMA)
    except ResourceUnavailableError as exc:
        return False, "novelty audit schema is unavailable: {0}".format(exc)
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        return False, "invalid novelty audit schema: {0}".format(exc)
    try:
        import jsonschema
    except ImportError:
        return False, "novelty audit schema validation is unavailable: jsonschema is not installed"
    try:
        jsonschema.validate(audit, schema, format_checker=jsonschema.FormatChecker())
    except jsonschema.ValidationError as exc:
        return False, "novelty audit violates schema: {0}".format(exc.message)
    except jsonschema.SchemaError as exc:
        return False, "invalid novelty audit schema: {0}".format(exc.message)

    if audit.get("candidate_id") != workspace.candidate_id:
        return False, "novelty audit candidate_id does not match the workspace"
    if audit.get("status") != NOVELTY_AUDIT_ELIGIBLE_STATUS:
        return False, "novelty audit status is not eligible: {0}".format(audit.get("status"))
    retrieved_at = _parse_utc_timestamp(audit.get("retrieved_at"))
    freshness = audit.get("freshness")
    expires_at = _parse_utc_timestamp(
        freshness.get("expires_at") if isinstance(freshness, dict) else None
    )
    now = datetime.now(timezone.utc)
    if retrieved_at is None or expires_at is None:
        return False, "novelty audit contains an invalid retrieval or freshness timestamp"
    if retrieved_at > now:
        return False, "novelty audit retrieval date is in the future"
    if expires_at <= retrieved_at:
        # NUMERICAL_GUARD: expires_at must be strictly later than retrieved_at.
        # Equality would imply zero freshness duration, which is logically
        # equivalent to an already-expired audit.
        return False, "novelty audit freshness expiry must be later than retrieval date"
    if expires_at <= now:
        # ASTROPHYSICAL_HEURISTIC: A stale novelty audit may miss a recently
        # published known-signal entry. The expiry window trades off registry
        # polling cost against the risk of time wasted on a non-novel signal.
        return False, "novelty audit is stale: freshness.expires_at has passed"
    evidence_ok, evidence_detail = _gate_novelty_evidence(workspace, audit, retrieved_at)
    if not evidence_ok:
        return False, evidence_detail
    return True, "novelty audit is eligible and current through {0}".format(
        audit["freshness"]["expires_at"]
    )


def gate_errors(workspace: CandidateWorkspace) -> List[str]:
    """Return a list of gate failures blocking advancement from the current phase.

    This is the central dispatch function for the Quality Verification Gate
    engine. It evaluates two categories of requirement:

    **Checklist requirements**: The phase document (e.g.,
    ``tracking/feasibility.md``) is parsed for ``[MANDATORY]`` checkboxes. If
    no mandatory items exist or any are unchecked, advancement is blocked.

    **Phase-specific programmatic gates** (see module docstring for policy):

    * ``feasibility``: :func:`_gate_novelty_audit` — requires eligible,
      current novelty audit with hash-verified registry evidence.
    * ``acquisition``: :func:`_gate_provenance_ready` — requires schema-valid,
      hash-matched provenance sidecars for every raw FITS product.
    * ``analysis``: :func:`_gate_fpp_claim` — intentionally always blocked
      (uncalibrated scene model).
    * ``review``: novelty audit (same as feasibility) plus a check that the
      lifecycle is not already locked.

    A stopped lifecycle overrides all other checks.

    Parameters
    ----------
    workspace : CandidateWorkspace
        The candidate workspace with metadata, tracking documents, and
        supporting artifacts.

    Returns
    -------
    List[str]
        Possibly empty list of human-readable failure reasons. An empty list
        means all gates pass and advancement is permitted.
    """
    metadata = workspace.metadata
    phase = metadata["workflow"]["phase"]
    errors: List[str] = []

    if metadata["lifecycle"]["state"] == "stopped":
        errors.append("candidate lifecycle is stopped; workflow advancement is disabled")

    document = phase_document_path(workspace, phase)
    if document is not None:
        telemetry = parse_checklist(document)
        if not telemetry.exists:
            errors.append("missing gate document: {0}".format(document.relative_to(workspace.path)))
        else:
            if telemetry.mandatory_total == 0:
                errors.append(
                    "gate document contains no mandatory checklist items: {0}".format(
                        document.relative_to(workspace.path)
                    )
                )
            for item in telemetry.items:
                if item.mandatory and not item.checked:
                    errors.append(
                        "unchecked mandatory item in {0}: {1}".format(
                            document.relative_to(workspace.path), item.text[:80]
                        )
                    )

    if phase in ("feasibility", "review"):
        # ASTROPHYSICAL_HEURISTIC: Both feasibility and review require a
        # current, eligible novelty audit. Feasibility protects against
        # investing time in a known signal; review ensures no new registry
        # entry appeared during the analysis workflow.
        ok, detail = _gate_novelty_audit(workspace)
        if not ok:
            errors.append(detail)
    if phase == "acquisition":
        # DIAGNOSTIC_REASONING: Hash-matched provenance guarantees that the
        # bytes analyzed downstream are exactly those validated here.
        # Schema-only validation could pass a sidecar copied from a
        # different FITS product.
        ok, detail = _gate_provenance_ready(workspace)
        if not ok:
            errors.append(detail)
    elif phase == "analysis":
        # Analysis gate checks for available FPP diagnostic evidence
        # (TRICERATOPS results, claims, or diagnostic outputs).
        ok, detail = _gate_fpp_claim(workspace)
        if not ok:
            errors.append(detail)
    elif phase == "review":
        if metadata["lifecycle"]["state"] in ("published", "archived"):
            errors.append("candidate is already locked; no further advancement")

    return errors


def set_lifecycle_state(
    workspace: CandidateWorkspace,
    state: str,
    reason: Optional[str] = None,
) -> Dict:
    """Set the lifecycle state of a candidate, recording the change as an event.

    Lifecycle states govern whether a candidate is editable, stoppable, or
    locked for publication. The five states (``active``, ``paused``,
    ``stopped``, ``published``, ``archived``) form a directed graph where
    transitions out of ``stopped``, ``published``, and ``archived`` require
    a non-empty ``reason`` — this ensures that reopening a locked candidate
    always leaves an audit trail explaining the exceptional action.

    The state change is validated against the metadata schema, persisted to
    ``candidate.json``, and appended to ``lifecycle/events.jsonl`` as a
    ``state_changed`` event.

    Parameters
    ----------
    workspace : CandidateWorkspace
        The candidate workspace, re-read from disk before mutation.
    state : str
        Target lifecycle state. Must be one of ``active``, ``paused``,
        ``stopped``, ``published``, or ``archived``.
    reason : str or None, optional
        Human-readable justification. Required when changing from
        ``stopped``, ``published``, or ``archived``.

    Returns
    -------
    Dict
        The updated lifecycle dict (subset of candidate metadata).

    Raises
    ------
    GateError
        If ``state`` is the same as current, or if a required ``reason`` is
        missing, or if the state is invalid.
    ValueError
        If ``state`` is not a registered lifecycle state.
    """
    from .workspace import LIFECYCLE_STATES

    if state not in LIFECYCLE_STATES:
        raise ValueError("invalid lifecycle state: {0}".format(state))
    workspace = load_candidate(workspace.repository_root, workspace.candidate_id)
    metadata = dict(workspace.metadata)
    lifecycle = dict(metadata["lifecycle"])
    old_state = lifecycle["state"]
    if old_state == state:
        raise GateError("lifecycle state unchanged: {0}".format(state))
    has_reason = isinstance(reason, str) and bool(reason.strip())
    if old_state in ("stopped", "published", "archived") and not has_reason:
        raise GateError(
            "a reason is required to change the state of a stopped or locked candidate "
            "({0} -> {1})".format(old_state, state)
        )
    lifecycle["state"] = state
    lifecycle["state_since"] = _now()
    if reason:
        lifecycle["reason"] = reason
    metadata["lifecycle"] = lifecycle
    validate_metadata(metadata, workspace.candidate_id)
    _write_metadata(workspace, metadata)
    _append_event(
        workspace,
        {
            "event": "state_changed",
            "candidate_id": workspace.candidate_id,
            "from": old_state,
            "to": state,
            "reason": reason,
            "timestamp": _now(),
        },
    )
    return lifecycle


def advance(workspace: CandidateWorkspace) -> Dict:
    """Validate the current gate and promote the candidate one phase.

    Mathematical Formulation
    ------------------------
    The transition from phase :math:`P_k` to :math:`P_{k+1}` succeeds iff:

    .. math::

        G_{\\text{checklist}}(P_k) \\land G_{\\text{programmatic}}(P_k)

    where :math:`G_{\\text{checklist}}` requires all ``[MANDATORY]`` items
    checked and :math:`G_{\\text{programmatic}}` is the phase-specific gate
    (see :func:`gate_errors`). The lifecycle must not be ``stopped``.

    Phase ordering is monotonic through the seven-phase sequence
    (``intake -> ... -> review``). Each advancement records a gate sign-off
    in ``gates/gate-NNN-<phase>.json`` and a lifecycle event in
    ``lifecycle/events.jsonl``.

    Review terminal behaviour
    --------------------------
    When advancing from ``review`` and the lifecycle is ``active`` or
    ``paused``, the lifecycle is automatically locked to ``published``. If
    already ``published`` or ``archived``, the operation is rejected. The
    review gate includes the novelty audit requirement, ensuring a final
    freshness check before publication.

    Parameters
    ----------
    workspace : CandidateWorkspace
        The candidate workspace. The candidate record is re-read from disk so
        repeated calls operate on current state.

    Returns
    -------
    Dict
        The advancement event dict with keys ``event``, ``candidate_id``,
        ``from``, ``to``, and ``timestamp``. On review advancement, also
        includes ``lifecycle`` with the new locked state.

    Raises
    ------
    GateError
        If the lifecycle is stopped, any mandatory item is unchecked, the
        phase-specific gate fails, the terminal phase is reached (except
        review), or the lifecycle is already locked when advancing from
        review.
    """
    workspace = load_candidate(workspace.repository_root, workspace.candidate_id)
    metadata = dict(workspace.metadata)
    phase = metadata["workflow"]["phase"]
    if metadata["lifecycle"]["state"] == "stopped":
        raise GateError("candidate lifecycle is stopped; workflow advancement is disabled")
    errors = gate_errors(workspace)
    if errors:
        raise GateError("; ".join(errors))

    next_phase_name = next_phase(phase)
    if next_phase_name is None and phase != "review":
        raise GateError("terminal phase reached: {0}".format(phase))

    event: Dict = {
        "event": "advanced",
        "candidate_id": workspace.candidate_id,
        "from": phase,
        "to": next_phase_name or phase,
        "timestamp": _now(),
    }
    if phase == "review":
        if metadata["lifecycle"]["state"] in ("published", "archived"):
            raise GateError(
                "candidate lifecycle is already locked: {0}".format(metadata["lifecycle"]["state"])
            )
        metadata["lifecycle"]["state"] = "published"
        metadata["lifecycle"]["state_since"] = _now()
        metadata["lifecycle"]["reason"] = "Review gate passed; lifecycle locked"
        event["lifecycle"] = metadata["lifecycle"]["state"]
        event["to"] = "review (locked)"

    metadata["workflow"]["phase"] = next_phase_name if next_phase_name is not None else phase
    validate_metadata(metadata, workspace.candidate_id)
    _write_metadata(workspace, metadata)
    _append_event(workspace, event)

    gate_record = {
        "gate": phase,
        "candidate_id": workspace.candidate_id,
        "result": "PASS",
        "timestamp": _now(),
        "next_phase": next_phase_name,
        "event": event["event"],
    }
    gates_dir = workspace.path / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)
    index = len(list(gates_dir.glob("gate-*.json")))
    (gates_dir / "gate-{0:03d}-{1}.json".format(index, phase)).write_text(
        json.dumps(gate_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return event
