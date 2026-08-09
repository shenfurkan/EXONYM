"""Markdown telemetry parser and ANSI progress dashboard.

Phase progress is derived deterministically from task checkboxes
(``- [ ]`` vs ``- [x]``) in candidate-local documents. Mandatory gate items
are tagged ``[MANDATORY]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .workspace import CandidateWorkspace

CHECKBOX = re.compile(r"-\s*\[([ xX])\]\s*(.*)")
MANDATORY_TAG = re.compile(r"\[MANDATORY\]", re.IGNORECASE)

# workflow phase -> required gate document (relative to the workspace)
PHASE_DOCUMENTS: Dict[str, Optional[str]] = {
    "intake": "docs/01_intake_manifest.md",
    "feasibility": "docs/02_feasibility_report.md",
    "acquisition": None,  # gate is data/raw provenance, handled by gatekeeper
    "vetting": "docs/03_spoc_dv_vetting.md",
    "followup": "docs/04_tfop_sg_followup.md",
    "analysis": None,  # gate is an FPP claim, handled by gatekeeper
    "review": "decisions/review_gate.md",
}

DOCUMENT_ORDER = (
    "docs/01_intake_manifest.md",
    "docs/02_feasibility_report.md",
    "docs/03_spoc_dv_vetting.md",
    "docs/04_tfop_sg_followup.md",
    "decisions/review_gate.md",
)


@dataclass(frozen=True)
class ChecklistItem:
    text: str
    checked: bool
    mandatory: bool


@dataclass
class DocumentTelemetry:
    path: str
    exists: bool
    items: List[ChecklistItem] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def checked(self) -> int:
        return sum(item.checked for item in self.items)

    @property
    def mandatory_total(self) -> int:
        return sum(item.mandatory for item in self.items)

    @property
    def mandatory_checked(self) -> int:
        return sum(item.checked and item.mandatory for item in self.items)

    @property
    def completion(self) -> float:
        if not self.total:
            return 0.0
        return self.checked / self.total * 100.0

    @property
    def gate_pass(self) -> bool:
        return bool(self.exists) and self.mandatory_checked == self.mandatory_total


def parse_checklist(path: Path) -> DocumentTelemetry:
    """Parse a Markdown document into checkbox telemetry."""
    telemetry = DocumentTelemetry(path.name if path else "?", path.exists() if path else False)
    if not path or not path.is_file():
        return telemetry
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CHECKBOX.match(line.strip())
        if not match:
            continue
        checked = match.group(1).lower() == "x"
        text = match.group(2).strip()
        telemetry.items.append(ChecklistItem(text, checked, bool(MANDATORY_TAG.search(text))))
    return telemetry


def phase_document_path(workspace: CandidateWorkspace, phase: str) -> Optional[Path]:
    relative = PHASE_DOCUMENTS.get(phase)
    if not relative:
        return None
    return workspace.path / relative


def candidate_telemetry(workspace: CandidateWorkspace) -> Dict[str, DocumentTelemetry]:
    """Return per-document telemetry for every phase gate document."""
    telemetry: Dict[str, DocumentTelemetry] = {}
    for relative in DOCUMENT_ORDER:
        telemetry[relative] = parse_checklist(workspace.path / relative)
    return telemetry


def overall_progress(telemetry: Sequence[DocumentTelemetry]) -> Tuple[int, int, float]:
    """Return (mandatory_checked, mandatory_total, completion fraction)."""
    checked = sum(doc.mandatory_checked for doc in telemetry)
    total = sum(doc.mandatory_total for doc in telemetry)
    fraction = (checked / total) if total else 0.0
    return checked, total, fraction


def _progress_bar(fraction: float, width: int = 40) -> str:
    filled = int(round(fraction * width))
    return "[" + "=" * filled + "-" * (width - filled) + "]"


def format_dashboard(workspace: CandidateWorkspace, telemetry: Dict[str, DocumentTelemetry]) -> str:
    """Render the ANSI candidate telemetry dashboard."""
    metadata = workspace.metadata
    phase = metadata["workflow"]["phase"]
    phase_index = list(PHASE_DOCUMENTS).index(phase)
    _, _, fraction = overall_progress(list(telemetry.values()))
    width = 66

    def row(content: str) -> str:
        return "| " + content.ljust(width - 3) + "|"

    lines = [
        "+" + "-" * (width - 2) + "+",
        row("EXONYM CANDIDATE TELEMETRY DASHBOARD :: Target: {0}".format(workspace.candidate_id)),
        "+" + "-" * (width - 2) + "+",
        row("Lifecycle State    : {0}".format(metadata["lifecycle"]["state"].upper())),
        row(
            "Workflow Phase     : {0} (Phase {1} of {2})".format(
                phase.upper(), phase_index + 1, len(PHASE_DOCUMENTS)
            )
        ),
        row("Scientific Disp.   : {0}".format(metadata["scientific_disposition"].upper())),
        row(
            "Progress           : {0} {1:5.1f}%".format(
                _progress_bar(fraction, 26), fraction * 100.0
            )
        ),
        "+" + "-" * (width - 2) + "+",
        row("DOCUMENT MANIFEST:"),
    ]
    for relative, doc in telemetry.items():
        status = "[X]" if doc.gate_pass else ("[!]" if doc.exists else "[ ]")
        lines.append(row("{0} {1}".format(status, relative)))
        lines.append(
            row(
                "    ({0:3.0f}% - {1}/{2} tasks completed)".format(
                    doc.completion, doc.checked, doc.total
                )
            )
        )
        for item in doc.items:
            if item.mandatory and not item.checked:
                lines.append(row("    [ ] {0}".format(item.text[:52])))
    lines.append("+" + "-" * (width - 2) + "+")
    return "\n".join(lines)
