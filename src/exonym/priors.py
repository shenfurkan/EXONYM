"""Preserve catalog transit-prior retrievals as candidate-owned evidence.

The module stores the unmodified provider response, retrieval metadata, a
content digest, and a normalized signal configuration.  Normalization changes
only representation, including an explicitly recorded source-time conversion;
it does not create missing ephemeris values or convert a catalog record into a
scientific claim.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .workspace import CandidateWorkspace


EXOFOP_TOI_CSV_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
EXOFOP_PRIOR_PROVIDER = "exofop-priors"
PRIOR_MANIFEST_FILENAME = "exofop-prior-manifest.json"


def _timestamp() -> str:
    """Return an explicit UTC timestamp without fractional seconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _retrieval_id() -> str:
    """Return a collision-resistant, append-only prior retrieval identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    return "{0}-{1}".format(timestamp, uuid.uuid4().hex[:12])


def _sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest of one raw response."""
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one candidate-local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically write one evidence file after its parent exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically serialize one UTF-8 JSON evidence record."""
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _signal_suffix(toi: object) -> str:
    """Return the ExoFOP signal suffix, retaining a safe default for malformed rows."""
    text = str(toi or "").strip()
    if "." not in text:
        return ".01"
    suffix = text.rsplit(".", 1)[1]
    return ".{0}".format(suffix) if suffix.isdigit() else ".01"


def _normalized_rows(raw_csv: str, tic_id: object) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract valid TIC-matched signal rows while retaining skipped-row reasons."""
    signals: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    expected_tic = str(tic_id).strip()
    seen_suffixes = set()
    for source_row_number, row in enumerate(csv.DictReader(io.StringIO(raw_csv)), start=2):
        if str(row.get("TIC ID", "")).strip() != expected_tic:
            continue
        try:
            period = float(row.get("Period (days)", 0.0))
            epoch = float(row.get("Epoch (BJD)", 0.0))
            depth = float(row.get("Depth (ppm)", 0.0))
            duration = float(row.get("Duration (hours)", 0.0))
        except (TypeError, ValueError):
            skipped_rows.append(
                {"source_row_number": source_row_number, "reason": "non-numeric transit field"}
            )
            continue
        if period <= 0.0:
            skipped_rows.append(
                {"source_row_number": source_row_number, "reason": "non-positive orbital period"}
            )
            continue
        signal_suffix = _signal_suffix(row.get("TOI"))
        if signal_suffix in seen_suffixes:
            skipped_rows.append(
                {"source_row_number": source_row_number, "reason": "duplicate signal suffix"}
            )
            continue
        seen_suffixes.add(signal_suffix)
        signals.append(
            {
                "toi": str(row.get("TOI", "")).strip(),
                "signal_suffix": signal_suffix,
                "source_row_number": source_row_number,
                "period_days": float(period),
                "epoch_btjd": float(epoch - 2457000.0 if epoch > 2450000.0 else epoch),
                "depth_ppm": float(depth),
                "duration_hours": float(duration),
                "source_time_system": "BJD value converted to BTJD only when greater than 2450000",
            }
        )
    return signals, skipped_rows


def fetch_exofop_priors(workspace: CandidateWorkspace) -> List[Path]:
    """Fetch ExoFOP priors with raw-response and normalization provenance.

    The provider CSV is retained byte-for-byte before matching rows to the
    workspace catalog identifier.  Finite source period, epoch, depth, and
    duration fields are copied to candidate-local signal configurations with
    source row numbers and SHA-256-linked retrieval metadata.

    Args:
        workspace (CandidateWorkspace): Workspace whose metadata supplies the
            catalog identifier used to select provider rows.

    Returns:
        List[Path]: Paths to normalized candidate-local signal configuration
        files.  The raw CSV, response metadata, and manifest remain retained
        even if a later retrieval replaces a normalized configuration.

    Raises:
        ValueError: The workspace lacks the required catalog identifier.
        RuntimeError: The provider returns a non-success HTTP status.
        OSError: The network request or candidate-local evidence write fails.

    Note:
        Provider values are externally supplied priors.  Downstream workflows
        must evaluate their own candidate-local evidence rather than treating
        this retrieval as discovery or validation.
    """
    tic_id = workspace.metadata.get("identifiers", {}).get("tic")
    if not tic_id:
        raise ValueError("candidate lacks a TIC identifier")

    request = urllib.request.Request(EXOFOP_TOI_CSV_URL, headers={"User-Agent": "exonym/1.2.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("ExoFOP returned status {0}".format(response.status))
        raw_response = response.read()
    raw_csv = raw_response.decode("utf-8", errors="replace")

    retrieval_id = _retrieval_id()
    retrieved_at = _timestamp()
    raw_dir = workspace.path / "data" / "external" / "catalog" / EXOFOP_PRIOR_PROVIDER / retrieval_id
    run_dir = workspace.path / "runs" / "catalog" / EXOFOP_PRIOR_PROVIDER / retrieval_id
    response_path = raw_dir / "response.csv"
    metadata_path = raw_dir / "prior-response-metadata.json"
    manifest_path = run_dir / PRIOR_MANIFEST_FILENAME
    response_hash = _sha256_bytes(raw_response)
    _atomic_write_bytes(response_path, raw_response)
    _atomic_write_json(
        metadata_path,
        {
            "schema_version": 1,
            "candidate_id": workspace.candidate_id,
            "provider": EXOFOP_PRIOR_PROVIDER,
            "retrieval_id": retrieval_id,
            "source_uri": EXOFOP_TOI_CSV_URL,
            "retrieved_at": retrieved_at,
            "http_status": 200,
            "content_encoding": "raw response bytes; parsed as UTF-8 with replacement for normalization",
            "response_sha256": response_hash,
        },
    )

    # SCIENTIFIC_BOUNDARY: Persisted raw-response provenance keeps normalized
    # provider values auditable without elevating them to a scientific claim.
    signals, skipped_rows = _normalized_rows(raw_csv, tic_id)
    manifest = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "provider": EXOFOP_PRIOR_PROVIDER,
        "retrieval_id": retrieval_id,
        "source_uri": EXOFOP_TOI_CSV_URL,
        "retrieved_at": retrieved_at,
        "requested_tic_id": str(tic_id),
        "raw_response": {
            "path": response_path.relative_to(workspace.path).as_posix(),
            "sha256": response_hash,
        },
        "raw_metadata": {
            "path": metadata_path.relative_to(workspace.path).as_posix(),
            "sha256": _sha256_file(metadata_path),
        },
        "signals": signals,
        "skipped_rows": skipped_rows,
        "normalization_policy": "Normalized transit fields preserve source row numbers and retain the full raw CSV; no default ephemeris values are invented.",
    }
    _atomic_write_json(manifest_path, manifest)
    manifest_hash = _sha256_file(manifest_path)

    signals_dir = workspace.path / "config" / "signals"
    written_paths: List[Path] = []
    for signal in signals:
        config_path = signals_dir / "transit_config{0}.json".format(signal["signal_suffix"])
        _atomic_write_json(
            config_path,
            {
                "transit": {
                    "period_days": signal["period_days"],
                    "epoch_btjd": signal["epoch_btjd"],
                    "depth_ppm": signal["depth_ppm"],
                    "duration_hours": signal["duration_hours"],
                    "source": "nasa-exofop",
                },
                "provenance": {
                    "provider": EXOFOP_PRIOR_PROVIDER,
                    "retrieval_id": retrieval_id,
                    "retrieved_at": retrieved_at,
                    "source_toi": signal["toi"],
                    "source_row_number": signal["source_row_number"],
                    "source_time_system": signal["source_time_system"],
                    "manifest_path": manifest_path.relative_to(workspace.path).as_posix(),
                    "manifest_sha256": manifest_hash,
                    "raw_response_path": response_path.relative_to(workspace.path).as_posix(),
                    "raw_response_sha256": response_hash,
                },
            },
        )
        written_paths.append(config_path)

    return written_paths
