"""Conservative administrative batch classification for candidate workspaces.

This module assigns operational retention and review labels to existing
candidate workspaces based on their lifecycle state and retained machine
triage evidence.  It does not perform scientific classification, assign a
planetary disposition, or change publication state.

Scientific boundary:
    The classification is strictly adminstrative policy bookkeeping.  It never
    evaluates astrophysical evidence, writes a validation claim, calibrates a
    false-positive rate, or replaces human review.  Candidates with decisive
    rejections, triage passes, or statistical-vetting evidence are tagged for
    review; all others follow lifecycle-based retention defaults.

Unit and claim-invariant contract
---------------------------------
All inputs are boolean file-existence checks and string lifecycle labels;
no astrophysical units or relations appear in this module.  Output retention
labels ("hot", "warm", "cold") are operational storage hints.  A missing
triage record, missing lifecycle state, or missing candidate directory does
not proceed and cannot set ``claim_eligible``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .isolation import is_reparse_point
from .review import (
    DEFAULT_RETENTION_CLASS,
    DEFAULT_REVIEW_STATUS,
    apply_classification_review,
)
from .workspace import CandidateWorkspace, discover_candidates_with_outcomes, load_candidate


POLICY_ID = "conservative-administrative-classification"
POLICY_VERSION = "1.0.0"
POLICY_REVIEWER = "exonym-policy"
POLICY_REASON = (
    "Administrative classification policy: retention follows lifecycle state; "
    "review status reflects retained machine triage evidence; scientific "
    "disposition and publication state are unchanged."
)


def _has_review_records(workspace: CandidateWorkspace) -> bool:
    """Return whether an explicit classification review already exists."""
    reviews = workspace.path / "decisions" / "reviews"
    return any(path.is_file() for path in reviews.glob("*.json")) if reviews.is_dir() else False


def _has_triage_evidence(workspace: CandidateWorkspace) -> bool:
    """Return whether candidate-local machine triage evidence is retained."""
    return any(
        (workspace.path / relative).is_file()
        for relative in (
            Path("decisions") / "decisive_rejection.json",
            Path("decisions") / "automated_triage.json",
            Path("outputs") / "statistical_vetting_evidence.json",
        )
    )


def _basis_path(workspace: CandidateWorkspace) -> Optional[str]:
    """Choose a stable candidate-local file for the policy review hash."""
    for relative in (
        Path("decisions") / "decisive_rejection.json",
        Path("decisions") / "automated_triage.json",
        Path("outputs") / "statistical_vetting_evidence.json",
        Path("README.md"),
        Path("docs") / "01_intake_manifest.md",
    ):
        if (workspace.path / relative).is_file():
            return relative.as_posix()
    return None


def _retention_for_lifecycle(state: str) -> str:
    """Map lifecycle to a conservative operational storage label."""
    if state == "active":
        return "hot"
    if state in {"published", "archived"}:
        return "hold"
    return "warm"


def _current_summary(workspace: CandidateWorkspace) -> Dict[str, str]:
    """Normalize legacy metadata to the current classification summary."""
    metadata = workspace.metadata
    return {
        "scientific_disposition": str(metadata.get("scientific_disposition", "unknown")),
        "publication": str(metadata.get("publication", "none")),
        "review_status": str(metadata.get("review_status", DEFAULT_REVIEW_STATUS)),
        "retention_class": str(metadata.get("retention_class", DEFAULT_RETENTION_CLASS)),
    }


def propose_classification(workspace: CandidateWorkspace) -> Dict[str, Any]:
    """Return one deterministic, non-scientific classification proposal."""
    current = _current_summary(workspace)
    explicit_review = _has_review_records(workspace)
    proposed = dict(current)
    if not explicit_review:
        proposed["retention_class"] = _retention_for_lifecycle(
            workspace.metadata["lifecycle"]["state"]
        )
        proposed["review_status"] = "triaged" if _has_triage_evidence(workspace) else "unreviewed"
    basis = _basis_path(workspace)
    fields_materialized = any(name not in workspace.metadata for name in proposed)
    changed = proposed != current or fields_materialized
    return {
        "candidate_id": workspace.candidate_id,
        "status": "proposed" if changed else "unchanged",
        "lifecycle": workspace.metadata["lifecycle"]["state"],
        "phase": workspace.metadata["workflow"]["phase"],
        "current": current,
        "proposed": proposed,
        "basis": basis,
        "reason": POLICY_REASON,
    }


def _count_values(proposals: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    """Count one normalized classification value across proposals."""
    counts: Dict[str, int] = {}
    for item in proposals:
        value = item[key]
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _summary(proposals: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Count batch outcomes and report the resulting administrative facets."""
    return {
        "total": len(proposals),
        "proposed": sum(item["status"] == "proposed" for item in proposals),
        "unchanged": sum(item["status"] == "unchanged" for item in proposals),
        "missing_basis": sum(item["basis"] is None for item in proposals),
        "by_lifecycle": _count_values(proposals, "lifecycle"),
        "by_phase": _count_values(proposals, "phase"),
        "by_scientific_disposition": _count_values(
            [
                {"scientific_disposition": item["proposed"]["scientific_disposition"]}
                for item in proposals
            ],
            "scientific_disposition",
        ),
        "by_review_status": _count_values(
            [{"review_status": item["proposed"]["review_status"]} for item in proposals],
            "review_status",
        ),
        "by_retention_class": _count_values(
            [{"retention_class": item["proposed"]["retention_class"]} for item in proposals],
            "retention_class",
        ),
        "by_publication": _count_values(
            [{"publication": item["proposed"]["publication"]} for item in proposals],
            "publication",
        ),
    }


def batch_classify(
    repository_root: Path,
    *,
    candidate_id: Optional[str] = None,
    apply: bool = False,
) -> Dict[str, Any]:
    """Propose or apply conservative administrative classification to candidates."""
    repository_root = Path(repository_root).resolve()
    incomplete: List[Dict[str, str]] = []
    if candidate_id is None:
        candidates, incomplete = discover_candidates_with_outcomes(repository_root)
        scope = "all-candidates"
    else:
        candidates = [load_candidate(repository_root, candidate_id)]
        scope = "candidate"
    proposals = [propose_classification(candidate) for candidate in candidates]
    summary = _summary(proposals)
    applied: List[str] = []
    if apply:
        missing = [item["candidate_id"] for item in proposals if item["status"] == "proposed" and item["basis"] is None]
        if missing:
            raise ValueError(
                "classification requires candidate-local evidence; missing basis for: "
                + ", ".join(missing)
            )
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        for item in proposals:
            if item["status"] != "proposed":
                continue
            proposed = item["proposed"]
            review_path = apply_classification_review(
                by_id[item["candidate_id"]],
                reviewer=POLICY_REVIEWER,
                reason=POLICY_REASON,
                evidence_paths=[item["basis"]],
                scientific_disposition=proposed["scientific_disposition"],
                publication=proposed["publication"],
                review_status=proposed["review_status"],
                retention_class=proposed["retention_class"],
            )
            applied.append(review_path.relative_to(repository_root).as_posix())
        for item in proposals:
            if item["status"] == "proposed":
                item["status"] = "applied"
    return {
        "schema_version": 1,
        "policy": {"id": POLICY_ID, "version": POLICY_VERSION},
        "scope": scope,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "apply": apply,
        "summary": summary,
        "applied_reviews": applied,
        "incomplete": incomplete,
        "candidates": proposals,
    }


def _reject_json_constant(value: str) -> object:
    """Reject non-standard JSON constants in classification records."""
    raise ValueError("non-finite JSON number: {0}".format(value))


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, object]]) -> Dict[str, Any]:
    """Reject ambiguous duplicate keys in a classification record."""
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {0}".format(key))
        result[key] = value
    return result


def _sha256(path: Path) -> str:
    """Hash one classification evidence file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_classification_records(
    repository_root: Path, *, candidate_id: Optional[str] = None
) -> Dict[str, Any]:
    """Verify classification-record JSON and evidence hashes without raw-data traversal."""
    repository_root = Path(repository_root).resolve()
    incomplete: List[Dict[str, str]] = []
    if candidate_id is None:
        candidates, incomplete = discover_candidates_with_outcomes(repository_root)
        scope = "all-candidates"
    else:
        candidates = [load_candidate(repository_root, candidate_id)]
        scope = "candidate"

    findings: List[Dict[str, str]] = []
    review_files = 0
    valid_reviews = 0
    for workspace in candidates:
        reviews_dir = workspace.path / "decisions" / "reviews"
        for review_path in sorted(reviews_dir.glob("*.json")):
            review_files += 1
            relative_review = review_path.relative_to(repository_root).as_posix()
            try:
                payload = json.loads(
                    review_path.read_text(encoding="utf-8"),
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_reject_duplicate_keys,
                )
                if not isinstance(payload, dict):
                    raise ValueError("classification review must be a JSON object")
                if payload.get("candidate_id") != workspace.candidate_id:
                    raise ValueError("classification review candidate_id does not match its workspace")
                evidence = payload.get("evidence")
                if not isinstance(evidence, list) or not evidence:
                    raise ValueError("classification review has no evidence")
                seen = set()
                for item in evidence:
                    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                        raise ValueError("classification review evidence entry is invalid")
                    path = workspace.path / item["path"]
                    resolved = path.resolve()
                    resolved.relative_to(workspace.path.resolve())
                    if item["path"] in seen:
                        raise ValueError("classification review evidence path is duplicated")
                    seen.add(item["path"])
                    current = workspace.path.resolve()
                    for part in Path(item["path"]).parts:
                        current = current / part
                        if current.exists() and is_reparse_point(current):
                            raise ValueError("classification review evidence uses a reparse point")
                    if not resolved.is_file() or item.get("sha256") != _sha256(resolved):
                        raise ValueError(
                            "classification review evidence is missing or hash-mismatched"
                        )
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                findings.append({"path": relative_review, "reason": str(exc)})
            else:
                valid_reviews += 1
    return {
        "schema_version": 1,
        "scope": scope,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not findings and not incomplete else "fail",
        "review_files": review_files,
        "valid_reviews": valid_reviews,
        "invalid_reviews": len(findings),
        "findings": findings,
        "incomplete": incomplete,
    }
