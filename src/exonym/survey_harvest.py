"""Streaming TCE harvesting and conservative live novelty screening.

The operator supplies the TCE release so its version, columns, and selection
are retained with candidate-local evidence rather than embedded in shared
source. Numeric prefilters bound work, while independent registry responses
are stored with a freshness window for later review.

Scientific Boundary:
    A no-registration response means only that the queried registries returned
    no matching record at retrieval time. It is neither proof of novelty nor a
    scientific claim.
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

from .survey import SurveyWorkspace, _load_target_record, register_survey_target
from .workspace import CandidateWorkspace, create_candidate, load_candidate


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
_TERMINAL_HARVEST_STATUSES = frozenset(
    ("registered", "ineligible", "already-provisioned", "rollback-leftover")
)
Transport = Callable[[str, float], bytes]


@dataclass(frozen=True)
class TceFilters:
    """Operator-selected numeric bounds for a reproducible TCE prefilter.

    Attributes:
        minimum_snr: Positive source-reported ranking-statistic lower bound.
        period_min_days: Inclusive positive lower orbital-period bound in days.
        period_max_days: Inclusive orbital-period upper bound in days.
        depth_min_ppm: Inclusive positive transit-depth lower bound in ppm.
        depth_max_ppm: Inclusive transit-depth upper bound in ppm.
        radius_min_earth: Inclusive positive source-reported radius lower
            bound in Earth radii.
        radius_max_earth: Inclusive source-reported radius upper bound in
            Earth radii.
        stellar_radius_max_solar: Inclusive positive stellar-radius upper
            bound in solar radii.
        tmag_max: Optional-magnitude acceptance upper bound.
    """

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
        """Return whether one source row supplies and satisfies every bound.

        Args:
            row: TCE source row with normalized or recognized alternate column
                names for the declared filter quantities.

        Returns:
            True only when every required quantity is finite and within the
            configured inclusive bounds. A missing optional magnitude does not
            exclude an otherwise complete row.
        """
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
    """Raw, source-addressable result of live registry lookups.

    Attributes:
        status: Eligible, ineligible, or unavailable retrieval outcome.
        reason: Human-readable retrieval rationale, not a scientific claim.
        responses: Ordered provider name, canonical URL, and raw response
            bytes retained for a later candidate-local audit.
    """

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
            # HTTPS URL scheme is strictly enforced above.
            response = urllib.request.urlopen(request, timeout=timeout)  # nosec B310
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
    """Construct the canonical independent-registry queries for one identifier.

    Args:
        tic: Positive mission-catalog identifier as a decimal string.

    Returns:
        Ordered provider-name and canonical-HTTPS-URL pairs used for a single
        live novelty retrieval.

    Raises:
        ValueError: If tic is not a positive decimal identifier string.
    """
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
        if field_names is None or not all(isinstance(name, str) for name in field_names):
            raise ValueError("NASA Archive response has no CSV header")
        normalized_headers = [name.strip().lower() for name in field_names]
        if not all(normalized_headers) or len(normalized_headers) != len(set(normalized_headers)):
            raise ValueError("NASA Archive response has ambiguous CSV headers")
        available = set(normalized_headers)
        required = {name.strip().lower() for name in expected_columns}
        if available != required:
            raise ValueError("NASA Archive response does not match the requested column contract")
        for row in rows:
            if None in row:
                raise ValueError("NASA Archive response has a row with unexpected columns")
            if any(value is not None and str(value).strip() for value in row.values()):
                return True
        return False
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("NASA Archive response is not readable UTF-8 CSV") from exc


def _exofop_response_matches_tic(payload: Mapping[str, object], tic: str) -> bool:
    """Require ExoFOP's structured target identity before accepting a no-match."""
    basic_info = payload.get("basic_info")
    if not isinstance(basic_info, Mapping):
        return False
    normalized: Dict[str, object] = {}
    for key, value in basic_info.items():
        normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
        if normalized_key in normalized:
            raise ValueError("ExoFOP response has ambiguous basic_info keys")
        normalized[normalized_key] = value
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

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON number")
        return parsed

    data = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
        parse_float=parse_finite_float,
    )
    if not isinstance(data, Mapping):
        raise ValueError("ExoFOP response is not a JSON object")
    return data


def novelty_response_has_registration(
    provider: str, source_uri: str, payload: bytes, tic: str
) -> bool:
    """Validate a retained registry response and test for registration content.

    Args:
        provider: Supported provider label associated with the response.
        source_uri: Canonical query URL expected for provider and tic.
        payload: Raw provider response bytes retained by the caller.
        tic: Positive mission-catalog identifier used for the query.

    Returns:
        True when a schema-valid response contains a registration record.

    Raises:
        ValueError: If provider identity, query URL, response shape, or
            encoding does not satisfy the provider contract.
    """
    expected_urls = dict(novelty_provider_urls(tic))
    if provider not in expected_urls or source_uri != expected_urls[provider]:
        raise ValueError("novelty response does not use the canonical provider query")
    if provider in _NASA_RESPONSE_COLUMNS:
        return _csv_has_records(payload, _NASA_RESPONSE_COLUMNS[provider])
    exofop_data = _strict_json_object(payload)
    _validate_exofop_response(exofop_data, str(tic))
    return any(bool(exofop_data[field]) for field in _EXOFOP_REGISTRATION_FIELDS)


def evaluate_live_novelty(
    tic: str,
    timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    transport: Transport = _https_bytes,
) -> NoveltyResult:
    """Query independent registries without treating a no-match as proof.

    Any unavailable, malformed, or noncanonical source is inconclusive.
    Eligibility therefore requires completed responses from every configured
    provider, while a detected registration stops the lookup as ineligible.

    Args:
        tic: Positive mission-catalog identifier as a decimal string.
        timeout: Positive network timeout in seconds for each request attempt.
        transport: Injectable HTTPS byte transport used for live retrieval.

    Returns:
        Raw provider responses and an eligible, ineligible, or unavailable
        retrieval state.

    Raises:
        ValueError: If tic or timeout is invalid before network retrieval.

    Notes:
        An eligible result is a time-bounded retrieval observation, not proof
        of novelty or a scientific validation claim.
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
    """Retain raw responses and write a hash-bound candidate-local audit.

    Args:
        candidate: Candidate workspace that owns raw responses and audit JSON.
        result: Source-addressable live novelty retrieval result.
        freshness_hours: Positive duration before the audit becomes stale.

    Returns:
        Path to the written candidate-local novelty audit.

    Raises:
        ValueError: If freshness or provider identity is invalid.
        RuntimeError: If an eligible result lacks every valid independent
            response or contains a detected registration.
    """
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
    """Yield non-comment CSV rows from a local release or HTTPS endpoint.

    Args:
        source: Existing local CSV path or HTTPS URL supplied by the operator.
        timeout: Positive request timeout in seconds for an HTTPS stream.

    Yields:
        String-keyed CSV row mappings in source order.

    Raises:
        ValueError: If source is neither a readable local file nor an HTTPS
            URL accepted by the transport policy.
    """
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


def _is_terminal_harvest_status(status: str) -> bool:
    """Return whether a harvest result consumes one bounded candidate slot.

    Live-registry outages are explicitly non-terminal: they leave novelty
    unresolved and must not prevent later source rows from being assessed.
    Every other emitted state is a definitive result for this invocation and
    therefore consumes the operator's requested cap.
    """
    return status in _TERMINAL_HARVEST_STATUSES


def _existing_harvest_outcome(
    survey: SurveyWorkspace, candidate_id: str, tic: str
) -> Dict[str, str]:
    """Validate an existing workspace before reporting it as provisioned.

    ``create_candidate`` deliberately raises on any path collision.  A
    collision is not evidence of a completed earlier harvest: an interrupted
    rollback can leave a valid-looking ``candidate.json`` without the survey
    denominator record that commits the harvest transaction.  Only a loaded,
    identity-matched candidate with a valid matching target record is
    idempotently reported as already provisioned.
    """
    try:
        candidate = load_candidate(survey.repository_root, candidate_id)
    except (OSError, ValueError):
        return {
            "candidate_id": candidate_id,
            "status": "rollback-leftover",
            "reason": (
                "Existing candidate workspace could not be validated; it is not "
                "treated as already provisioned."
            ),
        }
    identifiers = candidate.metadata.get("identifiers", {})
    if (
        identifiers.get("tic") != tic
        or identifiers.get("mission") != survey.metadata["mission"]
    ):
        return {
            "candidate_id": candidate_id,
            "status": "rollback-leftover",
            "reason": (
                "Existing candidate workspace does not match this survey's TIC "
                "and mission identity."
            ),
        }
    try:
        _load_target_record(survey, candidate_id)
    except (OSError, ValueError):
        return {
            "candidate_id": candidate_id,
            "status": "rollback-leftover",
            "reason": (
                "Existing candidate workspace has no valid matching survey target "
                "record and requires operator inspection."
            ),
        }
    return {"candidate_id": candidate_id, "status": "already-provisioned"}


def _rollback_harvest_provisioning(
    candidate_path: Path,
    survey_target_dir: Path,
    survey_target_existed: bool,
) -> List[Path]:
    """Remove paths created by one failed harvest and return any leftovers.

    Pre-existing survey target records are never deleted.  The caller uses the
    returned paths to make an interrupted rollback explicit instead of hiding
    it behind ``ignore_errors=True`` and misreporting it on a later retry.
    """
    paths = [candidate_path]
    if not survey_target_existed:
        paths.append(survey_target_dir)
    leftovers: List[Path] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            # A post-cleanup existence check below distinguishes an error that
            # still left recoverable debris from one that completed removal.
            pass
        if path.exists():
            leftovers.append(path)
    return leftovers


def harvest_tces(
    survey: SurveyWorkspace,
    source: str,
    filters: TceFilters,
    max_candidates: int,
    novelty_timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    freshness_hours: float = 24.0,
    transport: Transport = _https_bytes,
) -> List[Dict[str, str]]:
    """Filter a streamed release and provision candidate-local eligible records.

    Rows first pass operator-selected numerical bounds, then conservative live
    novelty retrieval. Registered workspaces retain raw response evidence
    before becoming part of the survey denominator. The returned outcomes
    retain unavailable, ineligible, and already-provisioned states rather than
    silently treating them as candidates.

    Args:
        survey: Survey that owns newly registered target records.
        source: Local CSV path or HTTPS release stream.
        filters: Reproducible numeric prefilter contract.
        max_candidates: Positive cap on terminal candidate outcomes. Registry
            outages remain visible in the result list but do not consume it.
        novelty_timeout: Positive timeout in seconds for novelty requests.
        freshness_hours: Positive audit freshness duration in hours.
        transport: Injectable HTTPS byte transport used by novelty retrieval.

    Returns:
        Ordered, JSON-safe per-row harvesting outcomes.

    Raises:
        ValueError: If maximum count or freshness is invalid.
        RuntimeError: If evidence needed to register an eligible workspace
            cannot be written or validated.

    Notes:
        Provisioning and no-match retrieval are operational triage steps, not
        scientific claims of novelty or planetary status.
    """
    if isinstance(max_candidates, bool) or int(max_candidates) < 1:
        raise ValueError("max_candidates must be at least one")
    try:
        freshness = float(freshness_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError("freshness_hours must be positive and finite") from exc
    if not math.isfinite(freshness) or freshness <= 0.0:
        raise ValueError("freshness_hours must be positive and finite")
    candidate_limit = int(max_candidates)
    terminal_outcomes = 0
    outcomes: List[Dict[str, str]] = []
    for row in stream_tce_rows(source, timeout=novelty_timeout):
        if terminal_outcomes >= candidate_limit:
            break
        if not filters.matches(row):
            continue
        tic = _tic(row)
        if tic is None:
            continue
        # SCIENTIFIC_BOUNDARY: A completed no-match lookup may permit
        # candidate-local triage, but it cannot establish scientific novelty.
        result = evaluate_live_novelty(tic, timeout=novelty_timeout, transport=transport)
        if result.status != "eligible":
            outcome = {"status": result.status, "reason": result.reason}
            outcomes.append(outcome)
            if _is_terminal_harvest_status(outcome["status"]):
                terminal_outcomes += 1
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
            outcome = _existing_harvest_outcome(survey, candidate_id, tic)
            outcomes.append(outcome)
            terminal_outcomes += 1
            continue
        survey_target_dir = survey.path / "targets" / candidate.candidate_id
        survey_target_existed = survey_target_dir.exists()
        try:
            write_novelty_audit(candidate, result, freshness_hours=freshness)
            register_survey_target(survey, candidate)
        except Exception as exc:
            leftovers = _rollback_harvest_provisioning(
                candidate.path,
                survey_target_dir,
                survey_target_existed,
            )
            if leftovers:
                raise RuntimeError(
                    "harvest provisioning failed and rollback left incomplete paths: {0}".format(
                        ", ".join(path.as_posix() for path in leftovers)
                    )
                ) from exc
            raise
        outcomes.append({"candidate_id": candidate.candidate_id, "status": "registered"})
        terminal_outcomes += 1
    return outcomes
