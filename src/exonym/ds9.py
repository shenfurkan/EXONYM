"""Export candidate-local archival sources as SAOImage DS9 regions.

The exporter never derives coordinates from a transit centroid or candidate
metadata.  Every DS9 point is copied from a validated archival report; PRF
localization can only add source-ID annotations to those existing points.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from .workspace import CandidateWorkspace


ARCHIVAL_REPORT_RELATIVE_PATH = Path("outputs") / "archival_vetting_report.json"
LOCALIZATION_REPORT_RELATIVE_PATH = Path("outputs") / "prf_localization_results.json"
DS9_REGION_RELATIVE_PATH = Path("figures") / "ds9_sources.reg"


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    """Return a JSON object from ``path``, or ``None`` when it is unusable."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def _finite_float(value: Any) -> Optional[float]:
    """Return a finite numeric value, or ``None`` for an unusable coordinate."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _localized_source_ids(report: Optional[Dict[str, Any]]) -> Set[str]:
    """Return IDs named as the fitted PRF-dominant source in completed sectors."""
    if report is None:
        return set()
    sectors = report.get("sector_results")
    if not isinstance(sectors, list):
        return set()
    source_ids = set()
    for sector in sectors:
        if not isinstance(sector, dict) or sector.get("skipped") is True:
            continue
        source_id = sector.get("fit_dominant_source_id")
        if isinstance(source_id, str) and source_id.strip():
            source_ids.add(source_id)
    return source_ids


def _ds9_label(parts: Iterable[str]) -> str:
    """Build a single-line DS9 label without allowing region syntax injection."""
    return "; ".join(
        str(part).replace("{", "[").replace("}", "]").replace("\n", " ")
        for part in parts
    )


def export_ds9_regions(
    workspace: CandidateWorkspace, output_path: Optional[Path] = None
) -> Path:
    """Write DS9 FK5 points from validated candidate-local archival sources.

    ``archival_vetting_report.json`` must contain a validated Gaia source list.
    Sources without finite RA and Dec are omitted rather than approximated. If a
    PRF localization result exists, only its fitted source IDs are included as
    labels; its centroid offsets are never converted into sky coordinates.
    """
    archival_path = workspace.path / ARCHIVAL_REPORT_RELATIVE_PATH
    archival = _load_json_object(archival_path)
    gaia = archival.get("gaia_astrometry") if archival else None
    if not isinstance(gaia, dict) or gaia.get("validated") is not True:
        raise ValueError("a validated candidate-local archival report is required")
    sources = gaia.get("sources")
    if not isinstance(sources, list):
        raise ValueError("validated archival report has no source list")

    localization = _load_json_object(workspace.path / LOCALIZATION_REPORT_RELATIVE_PATH)
    localized_ids = _localized_source_ids(localization)
    target_source_id = str(gaia.get("target_source_id")) if gaia.get("target_source_id") is not None else None

    lines = [
        "# Region file format: DS9 version 4.1",
        "global color=green width=1 font=\"helvetica 10 normal\"",
        "fk5",
    ]
    exported = 0
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            continue
        ra_deg = _finite_float(source.get("ra_deg"))
        dec_deg = _finite_float(source.get("dec_deg"))
        if ra_deg is None or dec_deg is None:
            continue

        label_parts = [source_id]
        is_target = source_id == target_source_id
        if is_target:
            label_parts.append("archive-target")
        if source_id in localized_ids:
            label_parts.append("prf-fit-dominant")
        color = "green" if is_target else "yellow"
        lines.append(
            "point({0:.8f},{1:.8f}) # point=circle color={2} text={{{3}}}".format(
                ra_deg, dec_deg, color, _ds9_label(label_parts)
            )
        )
        exported += 1

    if not exported:
        raise ValueError("validated archival report has no sources with finite sky coordinates")

    destination = output_path or workspace.path / DS9_REGION_RELATIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
