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
import shutil
import time
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError

from .survey import SurveyWorkspace, register_survey_target
from .workspace import CandidateWorkspace, create_candidate


DEFAULT_HTTP_TIMEOUT_SECONDS = 20.0
DEFAULT_HTTP_MAX_ATTEMPTS = 3
DEFAULT_HTTP_RETRY_BACKOFF_SECONDS = 0.5
_TIC_PATTERN = re.compile(r"^[1-9][0-9]{0,19}$")
_NOVELTY_PROVIDERS = ("nasa-toi", "nasa-confirmed", "exofop")
_NASA_RESPONSE_COLUMNS = {
    "nasa-toi": frozenset(("toi", "tid")),
    "nasa-confirmed": frozenset(("pl_name", "tic_id")),
}
_EXOFOP_REGISTRATION_FIELDS = ("tois", "ctois", "planet_parameters")
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
            "snr": _number(row, "snr", "tce_snr", "mes", "tce_mes", "tce_model_snr"),
            "period": _number(row, "period", "period_days", "tce_period", "tcet_period"),
            "depth": _number(row, "depth_ppm", "depth", "tce_depth_ppm", "tce_depth", "tcet_depth"),
            "radius": _number(row, "planet_radius_earth", "planet_radius", "planet_radius_rearth", "tce_prad", "prad"),
            "stellar_radius": _number(row, "stellar_radius_solar", "stellar_radius", "rstar", "tce_sradius", "tce_srad"),
            "tmag": _number(row, "tmag", "tessmag", "tess_mag", "tce_tmag"),
        }
        if any(value is None for key, value in values.items() if key != "tmag"):
            return False
        return bool(
            values["snr"] >= self.minimum_snr
            and self.period_min_days <= values["period"] <= self.period_max_days
            and self.depth_min_ppm <= values["depth"] <= self.depth_max_ppm
            and self.radius_min_earth <= values["radius"] <= self.radius_max_earth
            and values["stellar_radius"] <= self.stellar_radius_max_solar
            and (values["tmag"] is None or values["tmag"] <= self.tmag_max)
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


def _validate_timeout(timeout: float) -> float:
    """Return one finite positive request timeout."""
    try:
        value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be positive and finite") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("timeout must be positive and finite")
    return value


@contextmanager
def _https_response(url: str, timeout: float, accept: str) -> Iterator[Any]:
    """Open an HTTPS response with bounded retry while retaining TLS checks."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("live catalog sources must use an HTTPS URL")
    timeout = _validate_timeout(timeout)
    request = urllib.request.Request(
        url,
        headers={"Accept": accept, "User-Agent": "exonym-survey/1.2.0"},
    )
    last_error: Optional[BaseException] = None
    for attempt in range(DEFAULT_HTTP_MAX_ATTEMPTS):
        try:
            response = urllib.request.urlopen(request, timeout=timeout)  # nosec B310 -- HTTPS is enforced above
        except HTTPError as exc:
            # Authentication and malformed-request responses cannot recover by retrying.
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError("catalog request returned HTTP {0}".format(exc.code)) from exc
            last_error = exc
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
        else:
            with response:
                if getattr(response, "status", 200) != 200:
                    raise RuntimeError("catalog request returned HTTP {0}".format(response.status))
                yield response
            return
        if attempt + 1 < DEFAULT_HTTP_MAX_ATTEMPTS:
            time.sleep(DEFAULT_HTTP_RETRY_BACKOFF_SECONDS * (2 ** attempt))
    raise RuntimeError(
        "catalog request failed after {0} attempts: {1}".format(
            DEFAULT_HTTP_MAX_ATTEMPTS, last_error
        )
    ) from last_error


def _https_bytes(url: str, timeout: float) -> bytes:
    """Fetch one retained catalog response with bounded retry and TLS checks."""
    with _https_response(url, timeout, "text/csv, application/json;q=0.9, text/plain;q=0.8") as response:
        return response.read()


def _nea_urls(tic: str) -> Tuple[Tuple[str, str], ...]:
    """Build reviewed NASA Archive TAP lookups for one candidate-local TIC."""
    base = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    queries = (
        ("nasa-toi", "select toi,tid from toi where tid={0}".format(tic)),
        ("nasa-confirmed", "select pl_name,tic_id from ps where tic_id='TIC {0}'".format(tic)),
    )
    return tuple(
        (name, base + "?" + urllib.parse.urlencode({"query": query, "format": "csv"}))
        for name, query in queries
    )


def novelty_provider_urls(tic: str) -> Tuple[Tuple[str, str], ...]:
    """Return the canonical three-registry novelty queries for one TIC."""
    normalized_tic = str(tic)
    if not _TIC_PATTERN.fullmatch(normalized_tic):
        raise ValueError("TIC identifier must be a positive integer string")
    exofop_url = "https://exofop.ipac.caltech.edu/tess/target.php?" + urllib.parse.urlencode(
        {"id": normalized_tic, "json": ""}
    )
    return _nea_urls(normalized_tic) + (("exofop", exofop_url),)


def _csv_has_records(payload: bytes, expected_columns: Sequence[str]) -> bool:
    """Return whether a schema-valid NASA CSV response has data rows."""
    try:
        rows = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), strict=True)
        field_names = rows.fieldnames
        if field_names is None:
            raise ValueError("NASA Archive response has no CSV header")
        available = {name.strip().lower() for name in field_names if isinstance(name, str)}
        required = {name.strip().lower() for name in expected_columns}
        if not required.issubset(available):
            raise ValueError("NASA Archive response does not match the requested column contract")
        return any(
            any(value is not None and str(value).strip() for value in row.values())
            for row in rows
        )
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("NASA Archive response is not readable UTF-8 CSV") from exc


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


def _exofop_response_matches_tic(payload: Mapping[str, object], tic: str) -> bool:
    """Require ExoFOP's structured target identity before accepting a no-match."""
    basic_info = payload.get("basic_info")
    if not isinstance(basic_info, Mapping):
        return False
    normalized = {re.sub(r"[^a-z0-9]", "", str(key).lower()): value for key, value in basic_info.items()}
    value = normalized.get("ticid")
    if value is None:
        return False
    observed = str(value).upper().replace("TIC", "").strip()
    return observed == str(tic)


def _validate_exofop_response(payload: Mapping[str, object], tic: str) -> None:
    """Reject HTML/error JSON and incomplete result shapes as unavailable."""
    if not _exofop_response_matches_tic(payload, tic):
        raise ValueError("ExoFOP response does not identify the requested TIC")
    for field in _EXOFOP_REGISTRATION_FIELDS:
        value = payload.get(field)
        if not isinstance(value, list):
            raise ValueError("ExoFOP response is missing the {0} registration field".format(field))


def _strict_json_object(payload: bytes) -> Mapping[str, object]:
    """Parse an ExoFOP response without accepting duplicate keys or non-finite values."""
    def unique_object(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: {0}".format(key))
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError("non-finite JSON constant: {0}".format(value))

    data = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(data, Mapping):
        raise ValueError("ExoFOP response is not a JSON object")
    return data


def novelty_response_has_registration(
    provider: str, source_uri: str, payload: bytes, tic: str
) -> bool:
    """Validate one retained response and report whether it contains a registry record."""
    expected_urls = dict(novelty_provider_urls(tic))
    if provider not in expected_urls or source_uri != expected_urls[provider]:
        raise ValueError("novelty response does not use the canonical provider query")
    if provider in _NASA_RESPONSE_COLUMNS:
        return _csv_has_records(payload, _NASA_RESPONSE_COLUMNS[provider])
    exofop_data = _strict_json_object(payload)
    _validate_exofop_response(exofop_data, str(tic))
    return _exofop_has_registration(exofop_data)


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
    timeout = _validate_timeout(timeout)
    responses: List[Tuple[str, str, bytes]] = []
    try:
        for source_name, source_url in novelty_provider_urls(str(tic)):
            payload = transport(source_url, timeout)
            responses.append((source_name, source_url, payload))
            if novelty_response_has_registration(source_name, source_url, payload, str(tic)):
                reason = (
                    "NASA Exoplanet Archive returned a registered record."
                    if source_name.startswith("nasa-")
                    else "ExoFOP returned a TOI, cTOI, or planet registration."
                )
                return NoveltyResult("ineligible", reason, tuple(responses))
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
    """Retain raw responses and write a hash-bound schema-v2 novelty audit."""
    try:
        freshness = float(freshness_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError("freshness_hours must be positive and finite") from exc
    if not math.isfinite(freshness) or freshness <= 0.0:
        raise ValueError("freshness_hours must be positive and finite")
    providers = [source_name for source_name, _source_uri, _payload in result.responses]
    if any(provider not in _NOVELTY_PROVIDERS for provider in providers):
        raise ValueError("novelty audit received an unsupported registry provider")
    if result.status == "eligible" and (
        len(providers) != len(_NOVELTY_PROVIDERS)
        or set(providers) != set(_NOVELTY_PROVIDERS)
    ):
        raise RuntimeError("an eligible novelty audit requires all independent registry responses")
    if result.status == "eligible":
        tic = candidate.metadata["identifiers"].get("tic")
        try:
            for provider, source_uri, payload in result.responses:
                if novelty_response_has_registration(provider, source_uri, payload, str(tic)):
                    raise RuntimeError("an eligible novelty audit cannot retain a registered source response")
        except ValueError as exc:
            raise RuntimeError("an eligible novelty audit requires canonical valid registry responses") from exc
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
                "provider": source_name,
                "response_path": raw_path.relative_to(candidate.path).as_posix(),
            }
        )
    if not evidence:
        raise RuntimeError("a novelty audit requires at least one retained source response")
    audit = {
        "schema_version": 2,
        "candidate_id": candidate.candidate_id,
        "retrieved_at": _timestamp(retrieved),
        "freshness": {"expires_at": _timestamp(retrieved + timedelta(hours=freshness))},
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
            non_comments = (line for line in handle if not line.lstrip().startswith("#") and line.strip())
            for row in csv.DictReader(non_comments):
                yield dict(row)
        return
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("TCE stream source must be a local file or HTTPS URL")
    with _https_response(source, timeout, "text/csv") as response:
        with io.TextIOWrapper(response, encoding="utf-8-sig", errors="replace", newline="") as handle:
            non_comments = (line for line in handle if not line.lstrip().startswith("#") and line.strip())
            for row in csv.DictReader(non_comments):
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
    try:
        freshness = float(freshness_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError("freshness_hours must be positive and finite") from exc
    if not math.isfinite(freshness) or freshness <= 0.0:
        raise ValueError("freshness_hours must be positive and finite")
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
        survey_target_dir = survey.path / "targets" / candidate.candidate_id
        survey_target_existed = survey_target_dir.exists()
        try:
            write_novelty_audit(candidate, result, freshness_hours=freshness)
            register_survey_target(survey, candidate)
        except Exception:
            shutil.rmtree(candidate.path, ignore_errors=True)
            if not survey_target_existed:
                shutil.rmtree(survey_target_dir, ignore_errors=True)
            raise
        outcomes.append({"candidate_id": candidate.candidate_id, "status": "registered"})
    return outcomes
