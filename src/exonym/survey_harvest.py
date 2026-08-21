"""Streaming TCE harvesting and conservative live novelty screening.

The TCE source is supplied by the operator so its release, columns, and
selection are retained with the candidate-local evidence rather than embedded
in shared source. A successful lookup only means that the queried registries
returned no matching record at retrieval time; it is not a scientific claim.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .survey import SurveyWorkspace, register_survey_target
from .workspace import CandidateWorkspace, create_candidate


DEFAULT_HTTP_TIMEOUT_SECONDS = 20.0
_TIC_PATTERN = re.compile(r"^[1-9][0-9]{0,19}$")
Transport = Callable[[str, float], bytes]


@dataclass(frozen=True)
class TceFilters:
    """Operator-selected numeric bounds for a reproducible TCE prefilter."""

    minimum_snr: float
    period_min_days: float
    period_max_days: float
    depth_min_ppm: float
    depth_max_ppm: float
    radius_min_earth: float
    radius_max_earth: float
    stellar_radius_max_solar: float
    tmag_max: float

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("TCE filter bounds must be finite")
        if (
            self.minimum_snr <= 0.0
            or self.period_min_days <= 0.0
            or self.period_max_days < self.period_min_days
            or self.depth_min_ppm <= 0.0
            or self.depth_max_ppm < self.depth_min_ppm
            or self.radius_min_earth <= 0.0
            or self.radius_max_earth < self.radius_min_earth
            or self.stellar_radius_max_solar <= 0.0
        ):
            raise ValueError("TCE filter bounds are not physically ordered")

    def matches(self, row: Mapping[str, Any]) -> bool:
        """Return whether one source row supplies and satisfies every bound."""
        values = {
            "snr": _number(row, "snr", "tce_snr", "mes", "tce_mes"),
            "period": _number(row, "period", "period_days", "tce_period"),
            "depth": _number(row, "depth_ppm", "depth", "tce_depth_ppm"),
            "radius": _number(row, "planet_radius_earth", "planet_radius", "planet_radius_rearth"),
            "stellar_radius": _number(row, "stellar_radius_solar", "stellar_radius", "rstar"),
            "tmag": _number(row, "tmag", "tessmag", "tess_mag"),
        }
        if any(value is None for value in values.values()):
            return False
        return bool(
            values["snr"] >= self.minimum_snr
            and self.period_min_days <= values["period"] <= self.period_max_days
            and self.depth_min_ppm <= values["depth"] <= self.depth_max_ppm
            and self.radius_min_earth <= values["radius"] <= self.radius_max_earth
            and values["stellar_radius"] <= self.stellar_radius_max_solar
            and values["tmag"] <= self.tmag_max
        )


@dataclass(frozen=True)
class NoveltyResult:
    """Raw, source-addressable result of live registry lookups."""

    status: str
    reason: str
    responses: Tuple[Tuple[str, str, bytes], ...]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _timestamp(value: Optional[datetime] = None) -> str:
    return (value or _utc_now()).isoformat().replace("+00:00", "Z")


def _number(row: Mapping[str, Any], *names: str) -> Optional[float]:
    normalized = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = normalized.get(name)
        if value is None or isinstance(value, bool):
            continue
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _tic(row: Mapping[str, Any]) -> Optional[str]:
    normalized = {str(key).strip().lower(): value for key, value in row.items()}
    for name in ("tic_id", "tic", "ticid", "target_id"):
        value = normalized.get(name)
        if value is not None:
            candidate = str(value).strip().replace("TIC", "").strip()
            if _TIC_PATTERN.fullmatch(candidate):
                return candidate
    return None


def _https_bytes(url: str, timeout: float) -> bytes:
    """Fetch one HTTPS resource with certificate validation and a hard timeout."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("live catalog sources must use an HTTPS URL")
    request = urllib.request.Request(
        url,
        headers={"Accept": "text/csv, application/json;q=0.9, text/plain;q=0.8", "User-Agent": "exonym-survey/1.2.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 -- HTTPS is enforced above
        if getattr(response, "status", 200) != 200:
            raise RuntimeError("catalog request returned HTTP {0}".format(response.status))
        return response.read()


def _nea_urls(tic: str) -> Tuple[Tuple[str, str], ...]:
    """Build reviewed NASA Archive TAP lookups for one candidate-local TIC."""
    base = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    queries = (
        ("nasa-toi", "select toi,tic_id from toi where tic_id={0}".format(tic)),
        ("nasa-confirmed", "select pl_name,tic_id from pscomppars where tic_id={0}".format(tic)),
    )
    return tuple(
        (name, base + "?" + urllib.parse.urlencode({"query": query, "format": "csv"}))
        for name, query in queries
    )


def _csv_has_records(payload: bytes) -> bool:
    try:
        rows = csv.DictReader(io.StringIO(payload.decode("utf-8-sig", errors="replace")))
        return any(any(str(value).strip() for value in row.values()) for row in rows)
    except csv.Error as exc:
        raise ValueError("NASA Archive response is not readable CSV") from exc


def _exofop_has_registration(value: object, key: str = "") -> bool:
    """Recognize populated TOI/cTOI/planet registration fields conservatively."""
    normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
    registration_key = any(token in normalized_key for token in ("toi", "ctoi", "planet"))
    if isinstance(value, Mapping):
        return any(_exofop_has_registration(child, str(name)) for name, child in value.items())
    if isinstance(value, list):
        return any(_exofop_has_registration(child, key) for child in value)
    if registration_key and value not in (None, "", 0, False, [], {}):
        return True
    return False


def evaluate_live_novelty(
    tic: str,
    timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    transport: Transport = _https_bytes,
) -> NoveltyResult:
    """Query independent registries without treating a no-match as proof.

    Any unavailable or malformed source is inconclusive. Eligibility therefore
    requires completed NASA TOI, NASA confirmed-planet, and ExoFOP checks.
    """
    if not _TIC_PATTERN.fullmatch(str(tic)):
        raise ValueError("TIC identifier must be a positive integer string")
    if not math.isfinite(float(timeout)) or float(timeout) <= 0.0:
        raise ValueError("timeout must be positive and finite")
    responses: List[Tuple[str, str, bytes]] = []
    try:
        for source_name, source_url in _nea_urls(str(tic)):
            payload = transport(source_url, float(timeout))
            responses.append((source_name, source_url, payload))
            if _csv_has_records(payload):
                return NoveltyResult("ineligible", "NASA Exoplanet Archive returned a registered record.", tuple(responses))

        exofop_url = "https://exofop.ipac.caltech.edu/tess/target.php?" + urllib.parse.urlencode(
            {"id": str(tic), "json": ""}
        )
        exofop_payload = transport(exofop_url, float(timeout))
        responses.append(("exofop", exofop_url, exofop_payload))
        exofop_data = json.loads(exofop_payload.decode("utf-8"))
        if not isinstance(exofop_data, Mapping):
            raise ValueError("ExoFOP response is not a JSON object")
        if _exofop_has_registration(exofop_data):
            return NoveltyResult("ineligible", "ExoFOP returned a TOI, cTOI, or planet registration.", tuple(responses))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        return NoveltyResult("unavailable", "Live novelty check was incomplete: {0}".format(exc), tuple(responses))
    return NoveltyResult(
        "eligible",
        "Queried NASA TOI, NASA confirmed-planet, and ExoFOP registries returned no registration at retrieval time.",
        tuple(responses),
    )


def write_novelty_audit(
    candidate: CandidateWorkspace,
    result: NoveltyResult,
    freshness_hours: float,
) -> Path:
    """Retain raw responses and write the schema-v1 candidate novelty audit."""
    if not math.isfinite(float(freshness_hours)) or float(freshness_hours) <= 0.0:
        raise ValueError("freshness_hours must be positive and finite")
    retrieved = _utc_now()
    retrieval_id = uuid.uuid4().hex
    evidence_dir = candidate.path / "data" / "external" / "novelty" / retrieval_id
    evidence_dir.mkdir(parents=True, exist_ok=False)
    evidence = []
    for index, (source_name, source_uri, payload) in enumerate(result.responses):
        extension = "json" if source_name == "exofop" else "csv"
        raw_path = evidence_dir / "{0}-{1}.{2}".format(index, source_name, extension)
        raw_path.write_bytes(payload)
        evidence.append(
            {
                "source_uri": source_uri,
                "retrieved_at": _timestamp(retrieved),
                "finding": "{0}: {1}".format(source_name, result.reason),
                "evidence_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not evidence:
        raise RuntimeError("a novelty audit requires at least one retained source response")
    audit = {
        "schema_version": 1,
        "candidate_id": candidate.candidate_id,
        "retrieved_at": _timestamp(retrieved),
        "freshness": {"expires_at": _timestamp(retrieved + timedelta(hours=float(freshness_hours)))},
        "status": result.status,
        "decision_basis": result.reason,
        "evidence": evidence,
    }
    path = candidate.path / "decisions" / "novelty_audit.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def stream_tce_rows(source: str, timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS) -> Iterator[Dict[str, str]]:
    """Yield CSV rows without loading a TCE release into process memory."""
    source_path = Path(source)
    if source_path.is_file():
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                yield dict(row)
        return
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("TCE stream source must be a local file or HTTPS URL")
    request = urllib.request.Request(
        source,
        headers={"Accept": "text/csv", "User-Agent": "exonym-survey/1.2.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 -- HTTPS is enforced above
        if getattr(response, "status", 200) != 200:
            raise RuntimeError("TCE stream returned HTTP {0}".format(response.status))
        with io.TextIOWrapper(response, encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                yield dict(row)


def harvest_tces(
    survey: SurveyWorkspace,
    source: str,
    filters: TceFilters,
    max_candidates: int,
    novelty_timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    freshness_hours: float = 24.0,
    transport: Transport = _https_bytes,
) -> List[Dict[str, str]]:
    """Filter a streamed release and provision only currently eligible workspaces."""
    if isinstance(max_candidates, bool) or int(max_candidates) < 1:
        raise ValueError("max_candidates must be at least one")
    outcomes: List[Dict[str, str]] = []
    for row in stream_tce_rows(source, timeout=novelty_timeout):
        if len(outcomes) >= int(max_candidates):
            break
        if not filters.matches(row):
            continue
        tic = _tic(row)
        if tic is None:
            continue
        result = evaluate_live_novelty(tic, timeout=novelty_timeout, transport=transport)
        if result.status != "eligible":
            outcomes.append({"status": result.status, "reason": result.reason})
            continue
        candidate_id = "tce-" + tic
        try:
            candidate = create_candidate(
                survey.repository_root,
                candidate_id,
                tic=tic,
                mission=survey.metadata["mission"],
                tags=["survey-harvest"],
            )
        except FileExistsError:
            outcomes.append({"candidate_id": candidate_id, "status": "already-provisioned"})
            continue
        write_novelty_audit(candidate, result, freshness_hours=freshness_hours)
        register_survey_target(survey, candidate)
        outcomes.append({"candidate_id": candidate.candidate_id, "status": "registered"})
    return outcomes
