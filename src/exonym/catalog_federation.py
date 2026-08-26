"""Candidate-local, append-only retrievals from reviewed public catalogs.

Each retrieval keeps the response bytes, request context, timestamp, and
digest together so later consumers can distinguish retained evidence from a
live query that may have changed. Provider URLs and query templates are fixed
in ``PROVIDERS``; callers can select only an allowlisted provider name.

Scientific boundary:
    Catalog fields are provider-declared measurements with provider-specific
    units and selection effects. This module preserves them for review; it does
    not derive priors, claims, workflow state, or survey-search inputs.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .archive import load_validated_archival_report
from .workspace import CandidateWorkspace

CATALOG_STATUSES = (
    "available",
    "empty",
    "unavailable",
    "ambiguous",
    "requires-authentication",
)
RETRIEVAL_TTL = timedelta(days=30)
RETRYABLE_HTTP_STATUSES = (429, 500, 502, 503, 504)
KNOWN_SIGNAL_REQUIRED_COLUMNS = {
    "nasa-exoplanet-archive": (
        "pl_orbper", "pl_tranmid", "pl_trandur", "pl_tranmid_systemref",
    ),
    "nasa-exoplanet-archive-toi": (
        "toi", "tid", "pl_orbper", "pl_tranmid", "pl_trandurh",
    ),
}


@dataclass(frozen=True)
class ProviderSpec:
    """A reviewed provider endpoint and its fixed, candidate-safe template."""

    name: str
    official_source_uri: str
    release: str
    citation: str
    template_id: str
    expected_format: str
    native_time_scale: Optional[str]
    units_note: str
    identity_query: bool
    requires_coordinates: bool = False
    normalized_record_types: Tuple[str, ...] = ()
    access_policy_uri: Optional[str] = None
    rate_limit_note: str = "No client-side rate limit is assumed; retry only documented transient failures."
    failure_policy: str = "Retain the response and mark the retrieval unavailable; do not substitute another provider."
    archive_collections: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogRequest:
    """One reviewed provider request ready for a transport implementation."""

    method: str
    source_uri: str
    headers: Mapping[str, str]
    body: Optional[bytes]
    parameters: Mapping[str, str]


@dataclass(frozen=True)
class TransportResponse:
    """Transport-independent HTTP response retained as raw evidence."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes


PROVIDERS: Dict[str, ProviderSpec] = {
    "mast": ProviderSpec(
        "mast",
        "https://mast.stsci.edu/",
        "MAST API v0",
        "MAST: Mikulski Archive for Space Telescopes, STScI.",
        "mast-tic-by-tic-v1",
        "json",
        None,
        "Source-native catalog units are retained in raw records.",
        True,
    ),
    "gaia": ProviderSpec(
        "gaia",
        "https://gea.esac.esa.int/archive/",
        "Gaia Archive DR3",
        "Gaia Collaboration, Gaia Data Release 3.",
        "gaia-dr3-source-by-source-id-v1",
        "csv",
        "TCB",
        "Gaia DR3 source-native units and reference epochs are retained.",
        True,
    ),
    "simbad": ProviderSpec(
        "simbad",
        "https://simbad.cds.unistra.fr/",
        "SIMBAD current service",
        "Wenger et al. 2000, SIMBAD astronomical database.",
        "simbad-identifier-votable-v1",
        "votable",
        None,
        "SIMBAD identifiers and measurements retain their source-native units.",
        True,
    ),
    "vizier": ProviderSpec(
        "vizier",
        "https://vizier.cds.unistra.fr/",
        "VizieR current service",
        "Ochsenbein et al. 2000, VizieR catalogue service.",
        "vizier-identifier-search-votable-v1",
        "votable",
        None,
        "VizieR catalog units and quality flags are retained per source table.",
        True,
    ),
    "nasa-exoplanet-archive": ProviderSpec(
        "nasa-exoplanet-archive",
        "https://exoplanetarchive.ipac.caltech.edu/",
        "NASA Exoplanet Archive TAP current service",
        "NASA Exoplanet Archive, Caltech/IPAC.",
        "nea-pscomppars-by-tic-v2",
        "csv",
        "BJD_TDB where supplied by the source table",
        "NASA Exoplanet Archive source-native units are retained.",
        False,
    ),
    "nasa-exoplanet-archive-toi": ProviderSpec(
        "nasa-exoplanet-archive-toi",
        "https://exoplanetarchive.ipac.caltech.edu/",
        "NASA Exoplanet Archive TOI table current service",
        "NASA Exoplanet Archive, Caltech/IPAC.",
        "nea-toi-by-tic-v2",
        "csv",
        "BJD declared by the source table; no BJD_TDB conversion is inferred",
        "NASA Exoplanet Archive TOI period and duration remain source-native. Epochs are retained but require an explicit time-standard contract before epoch matching.",
        False,
    ),
    "irsa": ProviderSpec(
        "irsa",
        "https://irsa.ipac.caltech.edu/",
        "IRSA Gator current service",
        "NASA/IPAC Infrared Science Archive (IRSA).",
        "irsa-allwise-by-identifier-v1",
        "ipac-table",
        None,
        "2MASS, AllWISE, and NEOWISE photometric systems remain separate.",
        False,
    ),
    "ztf": ProviderSpec(
        "ztf",
        "https://irsa.ipac.caltech.edu/Missions/ztf.html",
        "ZTF public data service",
        "Bellm et al. 2019 and Masci et al. 2019, ZTF public data releases.",
        "ztf-public-lightcurve-by-identifier-v1",
        "ipac-table",
        "JD UTC where supplied by the source service",
        "ZTF source-native filters, flags, and time scale are retained.",
        False,
    ),
    "exofop": ProviderSpec(
        "exofop",
        "https://exofop.ipac.caltech.edu/tess/",
        "ExoFOP-TESS current service",
        "ExoFOP-TESS, NASA Exoplanet Follow-up Observing Program.",
        "exofop-tess-target-by-tic-v1",
        "html",
        None,
        "ExoFOP values are preserved as retrieval-time context, not priors.",
        False,
    ),
    "lamost-dr11": ProviderSpec(
        "lamost-dr11",
        "https://www.lamost.org/dr11/v2.0/",
        "LAMOST DR11 v2.0",
        "LAMOST DR11: Guoshoujing Telescope data release 11.",
        "lamost-dr11-lrs-cone-v1",
        "csv",
        "MJD where supplied by the source table",
        "LAMOST labels, uncertainties, spectra identifiers, and quality flags remain source-native.",
        False,
        True,
        ("stellar-parameters",),
        "http://www.lamost.org/lmusers/cms/article/view?id=1",
        "Use the release query service conservatively; no bulk or spectrum download is implemented.",
    ),
    "smoka": ProviderSpec(
        "smoka",
        "https://smoka.nao.ac.jp/",
        "SMOKA v3.7",
        "SMOKA Science Archive, Astronomical Data Center, NAOJ.",
        "smoka-archive-discovery-cone-v1",
        "html",
        "UTC where supplied by the archive listing",
        "SMOKA instrument, filter, exposure, and time metadata remain source-native; this adapter never requests FITS files.",
        False,
        True,
        ("archive-discovery",),
        "https://smoka.nao.ac.jp/help/howto_search.jsp",
        "Discovery requests are bounded to the reviewed cone form. FITS retrieval requires user registration and is intentionally unsupported.",
    ),
    "mast-hubble-jwst": ProviderSpec(
        "mast-hubble-jwst",
        "https://mast.stsci.edu/api/v0/",
        "MAST API v0 Hubble/JWST discovery",
        "MAST: Mikulski Archive for Space Telescopes, STScI.",
        "mast-hubble-jwst-cone-v1",
        "json",
        "Source-native mission time standards are retained per product record",
        "Hubble and JWST product metadata, filters, and calibration levels remain source-native; this adapter never downloads products.",
        False,
        True,
        ("archive-discovery",),
        "https://mast.stsci.edu/api/v0/",
        "The fixed cone request is bounded and retrieves metadata only; proprietary products may return requires-authentication.",
        archive_collections=("HST", "JWST"),
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError("non-finite JSON number: {0}".format(value))


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _json_safe(value: Any) -> Any:
    """Reject non-finite provider values before normalized evidence is written."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("provider response contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _safe_identifier(candidate: CandidateWorkspace) -> str:
    tic = candidate.metadata.get("identifiers", {}).get("tic")
    if not isinstance(tic, str) or not tic.isdigit():
        raise ValueError("catalog retrieval requires a candidate TIC identifier")
    return "TIC {0}".format(tic)


def _gaia_source_id(candidate: CandidateWorkspace) -> str:
    aliases = candidate.metadata.get("identifiers", {}).get("aliases", [])
    for alias in aliases if isinstance(aliases, list) else []:
        match = re.fullmatch(r"Gaia(?: DR3)?\s+(\d+)", str(alias), flags=re.IGNORECASE)
        if match:
            return match.group(1)
    raise ValueError("Gaia retrieval requires a recorded Gaia DR3 source identifier alias")


def _candidate_coordinates(candidate: CandidateWorkspace) -> Mapping[str, str]:
    """Read finite candidate-local coordinates without inventing an astrometric prior."""
    report = load_validated_archival_report(candidate)
    if report is None:
        raise ValueError(
            "catalog retrieval requires an owned, validated successful archival report"
        )
    try:
        coordinates = report.get("target_coordinates")
        ra_value = float(coordinates["ra_deg"])
        dec_value = float(coordinates["dec_deg"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "{0} retrieval requires finite coordinates in outputs/archival_vetting_report.json"
            .format("catalog")
        )
    if not math.isfinite(ra_value) or not math.isfinite(dec_value) or not 0.0 <= ra_value < 360.0 or not -90.0 <= dec_value <= 90.0:
        raise ValueError("catalog retrieval requires finite ICRS coordinates in archival_vetting_report.json")
    return {"ra_deg": "{0:.8f}".format(ra_value), "dec_deg": "{0:.8f}".format(dec_value)}


def _request_for(spec: ProviderSpec, candidate: CandidateWorkspace) -> CatalogRequest:
    """Build one fixed endpoint request without accepting URLs or query language."""
    identifier = _safe_identifier(candidate)
    coordinates = _candidate_coordinates(candidate) if spec.requires_coordinates else None
    if spec.name == "mast":
        tic = identifier.split()[1]
        payload = {
            "service": "Mast.Catalogs.Filtered.Tic",
            "params": {"columns": "*", "filters": [{"paramName": "ID", "values": [tic]}]},
            "format": "json",
        }
        return CatalogRequest(
            "POST", "https://mast.stsci.edu/api/v0/invoke",
            {"Content-Type": "application/json", "Accept": "application/json"},
            json.dumps(payload, sort_keys=True).encode("utf-8"), {"tic": tic},
        )
    if spec.name == "gaia":
        source_id = _gaia_source_id(candidate)
        query = "SELECT * FROM gaiadr3.gaia_source WHERE source_id = {0}".format(source_id)
        params = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
        return CatalogRequest(
            "GET", "https://gea.esac.esa.int/tap-server/tap/sync?" + urlencode(params),
            {"Accept": "text/csv"}, None, {"gaia_source_id": source_id},
        )
    if spec.name == "simbad":
        return CatalogRequest(
            "GET", "https://simbad.cds.unistra.fr/simbad/sim-id?" + urlencode(
                {"Ident": identifier, "output.format": "VOTable"}
            ), {"Accept": "application/x-votable+xml"}, None, {"identifier": identifier},
        )
    if spec.name == "vizier":
        return CatalogRequest(
            "GET", "https://vizier.cds.unistra.fr/viz-bin/votable?" + urlencode(
                {"-words": identifier, "-out.all": ""}
            ), {"Accept": "application/x-votable+xml"}, None, {"identifier": identifier},
        )
    if spec.name == "nasa-exoplanet-archive":
        tic = identifier.split()[1]
        query = (
            "SELECT pl_name,pl_orbper,pl_tranmid,pl_trandur,pl_tranmid_systemref "
            "FROM pscomppars WHERE tic_id = 'TIC {0}'"
        ).format(tic)
        params = {"query": query, "format": "csv"}
        return CatalogRequest(
            "GET", "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?" + urlencode(params),
            {"Accept": "text/csv"}, None, {"tic": tic},
        )
    if spec.name == "nasa-exoplanet-archive-toi":
        tic = identifier.split()[1]
        query = (
            "SELECT toi,tid,pl_orbper,pl_tranmid,pl_trandurh "
            "FROM toi WHERE tid = {0}"
        ).format(tic)
        params = {"query": query, "format": "csv"}
        return CatalogRequest(
            "GET", "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?" + urlencode(params),
            {"Accept": "text/csv"}, None, {"tic": tic},
        )
    if spec.name == "irsa":
        return CatalogRequest(
            "GET", "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?" + urlencode(
                {"catalog": "allwise_p3as_psd", "objstr": identifier, "outfmt": "3"}
            ), {"Accept": "text/plain"}, None, {"identifier": identifier, "catalog": "allwise_p3as_psd"},
        )
    if spec.name == "ztf":
        return CatalogRequest(
            "GET", "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?" + urlencode(
                {"ID": identifier}
            ), {"Accept": "text/plain"}, None, {"identifier": identifier},
        )
    if spec.name == "exofop":
        tic = identifier.split()[1]
        return CatalogRequest(
            "GET", "https://exofop.ipac.caltech.edu/tess/target.php?" + urlencode({"id": tic}),
            {"Accept": "text/html"}, None, {"tic": tic},
        )
    if spec.name == "lamost-dr11":
        parameters = {
            "pos.type": "cone", "pos.racenter": coordinates["ra_deg"],
            "pos.deccenter": coordinates["dec_deg"], "pos.radius": "5",
            "output.fmt": "csv", "showcol": "combined.obsid",
        }
        return CatalogRequest(
            "POST", "https://www.lamost.org/dr11/v2.0/table/combined/q",
            {"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/csv"},
            urlencode(parameters).encode("ascii"), parameters,
        )
    if spec.name == "smoka":
        parameters = {
            "action": "Search", "coordsys": "Equatorial", "equinox": "J2000",
            "RadOrRec": "radius", "longitudeC": coordinates["ra_deg"],
            "latitudeC": coordinates["dec_deg"], "radius": "5", "asciitable": "Table",
            "frameorshot": "Frame", "data_typ": "OBJECT", "obs_mod": "SPEC",
            "diff": "100", "output_equinox": "J2000",
        }
        return CatalogRequest(
            "POST", "https://smoka.nao.ac.jp/fssearch",
            {"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/html"},
            urlencode(parameters).encode("ascii"), parameters,
        )
    if spec.name == "mast-hubble-jwst":
        payload = {
            "service": "Mast.Caom.Cone",
            "params": {"ra": float(coordinates["ra_deg"]), "dec": float(coordinates["dec_deg"]), "radius": 0.05},
            "format": "json", "pagesize": 1000,
        }
        return CatalogRequest(
            "POST", "https://mast.stsci.edu/api/v0/invoke",
            {"Content-Type": "application/json", "Accept": "application/json"},
            json.dumps(payload, sort_keys=True).encode("utf-8"), coordinates,
        )
    raise ValueError("unsupported catalog provider: {0}".format(spec.name))


def _default_transport(request: CatalogRequest) -> TransportResponse:
    request_object = Request(request.source_uri, data=request.body, headers=dict(request.headers), method=request.method)
    try:
        # Request URI is restricted to fixed, reviewed catalog provider endpoints.
        with urlopen(request_object, timeout=30) as response:  # nosec B310
            return TransportResponse(response.getcode(), dict(response.headers.items()), response.read())
    except HTTPError as exc:
        return TransportResponse(exc.code, dict(exc.headers.items()) if exc.headers else {}, exc.read())
    except URLError as exc:
        raise RuntimeError("network unavailable: {0}".format(exc.reason)) from exc


def _parse_rows(content: bytes, expected_format: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Parse a provider-aware response before it can be marked available."""
    if not content.strip():
        return [], None
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return [], "response is not UTF-8; retained as binary raw evidence"
    try:
        if expected_format == "json":
            parsed = json.loads(
                text, parse_constant=_reject_nonfinite_constant, parse_float=_parse_finite_float
            )
            if isinstance(parsed, list):
                rows = parsed
            elif isinstance(parsed, dict):
                rows = parsed.get("data", parsed.get("rows", parsed.get("results", [])))
            else:
                return [], "JSON response is neither an object nor an array"
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                return [], "JSON response does not contain a list of object records"
            return [_json_safe(dict(row)) for row in rows], None
        if expected_format == "csv":
            return [dict(row) for row in csv.DictReader(io.StringIO(text))], None
        if expected_format in ("votable", "ipac-table"):
            from astropy.table import Table

            format_name = "votable" if expected_format == "votable" else "ascii.ipac"
            table = Table.read(io.BytesIO(content), format=format_name)
            return [_json_safe(dict(zip(table.colnames, row))) for row in table], None
        if expected_format == "html":
            lowered = text.lower()
            if "<html" not in lowered or any(marker in lowered for marker in ("error", "not found", "login")):
                return [], "HTML response does not contain a usable reviewed provider document"
            return [{"document": "html-response"}], None
    except (ValueError, csv.Error, OSError, TypeError) as exc:
        return [], "malformed {0} response: {1}".format(expected_format, exc)
    return [], "unsupported provider response format: {0}".format(expected_format)


def _validate_known_signal_columns(spec: ProviderSpec, rows: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Reject populated known-signal snapshots missing their reviewed field contract."""
    required = KNOWN_SIGNAL_REQUIRED_COLUMNS.get(spec.name)
    if required is None or not rows:
        return None
    available = {str(key).lower() for key in rows[0]}
    missing = [name for name in required if name.lower() not in available]
    if missing:
        return "known-signal field contract missing columns: {0}".format(", ".join(missing))
    return None


def _response_status(response: TransportResponse, rows: Sequence[Mapping[str, Any]], parse_error: Optional[str], spec: ProviderSpec) -> str:
    if response.status_code in (401, 403):
        return "requires-authentication"
    if response.status_code < 200 or response.status_code >= 300:
        return "unavailable"
    if parse_error:
        return "unavailable"
    if spec.identity_query and len(rows) > 1:
        return "ambiguous"
    if not rows and spec.expected_format in ("json", "csv"):
        return "empty"
    return "available"


def _finite_record_value(record: Mapping[str, Any], *names: str) -> Optional[float]:
    """Read one finite source-native numeric field without supplying a default."""
    lowered = {str(key).lower(): value for key, value in record.items()}
    for name in names:
        try:
            value = float(lowered[name.lower()])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _record_astrometry(record: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """Extract explicitly supplied ICRS astrometry from common catalog field names."""
    ra_value = _finite_record_value(record, "ra", "ra_deg", "ra2000", "raj2000")
    dec_value = _finite_record_value(record, "dec", "dec_deg", "dec2000", "dej2000")
    if ra_value is None or dec_value is None or not 0.0 <= ra_value < 360.0 or not -90.0 <= dec_value <= 90.0:
        return None
    values = {"ra_deg": ra_value, "dec_deg": dec_value}
    aliases = {
        "pmra_mas_per_year": ("pmra", "pmra_mas_per_year"),
        "pmdec_mas_per_year": ("pmdec", "pmdec_mas_per_year"),
        "reference_epoch_jyear": ("ref_epoch", "reference_epoch_jyear", "epoch"),
        "ra_uncertainty_mas": ("ra_error", "ra_uncertainty_mas", "e_ra"),
        "dec_uncertainty_mas": ("dec_error", "dec_uncertainty_mas", "e_dec"),
        "pmra_uncertainty_mas_per_year": ("pmra_error", "e_pmra"),
        "pmdec_uncertainty_mas_per_year": ("pmdec_error", "e_pmdec"),
    }
    for key, names in aliases.items():
        value = _finite_record_value(record, *names)
        if value is not None:
            values[key] = value
    return values


def _propagate_astrometry(values: Mapping[str, float], comparison_jyear: float) -> Optional[Dict[str, float]]:
    """Propagate ICRS position to a common Julian year using catalog proper motion.

    ``pmra`` is interpreted as mu_alpha* (including cos(delta)). Position and
    proper-motion uncertainties are propagated independently in quadrature; no
    parallax, radial-velocity, time-scale, or passband conversion is implied.
    """
    reference_jyear = values.get("reference_epoch_jyear")
    if reference_jyear is None:
        return None
    elapsed_years = comparison_jyear - reference_jyear
    ra_value = values["ra_deg"]
    dec_value = values["dec_deg"]
    pmra_value = values.get("pmra_mas_per_year", 0.0)
    pmdec_value = values.get("pmdec_mas_per_year", 0.0)
    cosine_dec = math.cos(math.radians(dec_value))
    if abs(cosine_dec) < 1e-12:
        return None
    propagated_ra = (ra_value + pmra_value * elapsed_years / (1000.0 * 3600.0 * cosine_dec)) % 360.0
    propagated_dec = dec_value + pmdec_value * elapsed_years / (1000.0 * 3600.0)
    ra_error = values.get("ra_uncertainty_mas")
    dec_error = values.get("dec_uncertainty_mas")
    if ra_error is None or dec_error is None:
        return None
    propagated_uncertainty = math.sqrt(
        ra_error ** 2 + dec_error ** 2
        + (values.get("pmra_uncertainty_mas_per_year", 0.0) * elapsed_years) ** 2
        + (values.get("pmdec_uncertainty_mas_per_year", 0.0) * elapsed_years) ** 2
    )
    return {
        "ra_deg": propagated_ra, "dec_deg": propagated_dec,
        "uncertainty_mas": propagated_uncertainty,
    }


def _separation_mas(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    """Return small-angle ICRS separation in milliarcseconds at one common epoch."""
    delta_ra = (first["ra_deg"] - second["ra_deg"] + 180.0) % 360.0 - 180.0
    delta_dec = first["dec_deg"] - second["dec_deg"]
    return 3600.0 * 1000.0 * math.hypot(
        delta_ra * math.cos(math.radians(second["dec_deg"])), delta_dec
    )


def normalize_cross_matches(
    candidate_astrometry: Mapping[str, Any], source_records: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Normalize an uncertainty-aware cross-match without selecting ambiguous rows.

    Both positions are propagated to the candidate reference Julian year. A
    record is plausible only when its separation is within five combined
    one-sigma positional uncertainties. Missing coordinates, epochs, or
    uncertainties produce ``insufficient-astrometry`` rather than a positional
    match. Every plausible record is retained; more than one remains
    ``ambiguous`` for human review.
    """
    target = _record_astrometry(candidate_astrometry)
    if target is None or "reference_epoch_jyear" not in target:
        return {
            "method": "icrs-proper-motion-uncertainty-v1",
            "status": "insufficient-astrometry", "comparison_epoch_jyear": None,
            "matches": [],
        }
    comparison_jyear = target["reference_epoch_jyear"]
    propagated_target = _propagate_astrometry(target, comparison_jyear)
    if propagated_target is None:
        return {
            "method": "icrs-proper-motion-uncertainty-v1",
            "status": "insufficient-astrometry", "comparison_epoch_jyear": comparison_jyear,
            "matches": [],
        }
    matches = []
    incomplete = False
    for index, source_record in enumerate(source_records):
        source = _record_astrometry(source_record)
        if source is None:
            continue
        propagated_source = _propagate_astrometry(source, comparison_jyear)
        if propagated_source is None:
            incomplete = True
            continue
        separation = _separation_mas(propagated_target, propagated_source)
        combined_uncertainty = math.hypot(
            propagated_target["uncertainty_mas"], propagated_source["uncertainty_mas"]
        )
        if combined_uncertainty > 0.0 and separation <= 5.0 * combined_uncertainty:
            matches.append({
                "source_record_index": index,
                "separation_mas": separation,
                "combined_uncertainty_mas": combined_uncertainty,
                "sigma_distance": separation / combined_uncertainty,
            })
    status = "ambiguous" if len(matches) > 1 else "matched" if matches else (
        "insufficient-astrometry" if incomplete else "no-plausible-match"
    )
    return {
        "method": "icrs-proper-motion-uncertainty-v1", "status": status,
        "comparison_epoch_jyear": comparison_jyear, "matches": matches,
    }


def _candidate_crossmatch_astrometry(candidate: CandidateWorkspace) -> Mapping[str, Any]:
    """Use only candidate-local archival context; never synthesize uncertainties."""
    report_path = candidate.path / "outputs" / "archival_vetting_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    if not isinstance(report, dict):
        return {}
    coordinates = report.get("target_coordinates")
    if not isinstance(coordinates, dict):
        return {}
    return dict(coordinates)


def _is_demo_record(record: Mapping[str, Any]) -> bool:
    """Keep raw evidence but reject declared demonstration values from normalization."""
    markers = ("source", "provenance", "origin", "data_source")
    return any(
        "demo" in str(record.get(marker, "")).lower()
        or "synthetic" in str(record.get(marker, "")).lower()
        for marker in markers
    )


def _normalized_records(rows: Sequence[Mapping[str, Any]], archive_collections: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Link normalized records to source rows without inferring stellar values."""
    return [
        {"source_record_index": index, "source_native_fields": dict(row)}
        for index, row in enumerate(rows)
        if not _is_demo_record(row)
        and (
            not archive_collections
            or str(row.get("obs_collection", row.get("collection", ""))).upper() in archive_collections
        )
    ]


def _retry_transport(request: CatalogRequest, transport: Callable[[CatalogRequest], TransportResponse]) -> Tuple[TransportResponse, int, Optional[str]]:
    attempts = 0
    failure = None
    while attempts < 3:
        attempts += 1
        try:
            response = transport(request)
        except RuntimeError as exc:
            failure = str(exc)
            continue
        if response.status_code not in RETRYABLE_HTTP_STATUSES or attempts == 3:
            return response, attempts, failure
    return TransportResponse(0, {}, b""), attempts, failure or "network unavailable"


def _artifact(path: Path, candidate: CandidateWorkspace, role: str) -> Dict[str, str]:
    return {"path": path.relative_to(candidate.path).as_posix(), "sha256": _sha256_file(path), "role": role}


def _retrieval_paths(candidate: CandidateWorkspace, provider: str, retrieval_id: str) -> Tuple[Path, Path]:
    raw_dir = candidate.path / "data" / "external" / "catalog" / provider / retrieval_id
    run_dir = candidate.path / "runs" / "catalog" / provider / retrieval_id
    return raw_dir, run_dir


def _write_context(candidate: CandidateWorkspace) -> Path:
    """Summarize retrieval references only, never duplicate catalog measurements."""
    entries = []
    for manifest_path in sorted((candidate.path / "runs" / "catalog").glob("*/*/query-manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        entries.append({
            "provider": manifest.get("provider"), "retrieval_id": manifest.get("retrieval_id"),
            "status": manifest.get("status"), "retrieved_at": manifest.get("retrieved_at"),
            "expires_at": manifest.get("expires_at"), "citation": manifest.get("citation"),
            "manifest_path": manifest_path.relative_to(candidate.path).as_posix(),
            "manifest_sha256": _sha256_file(manifest_path),
        })
    output = candidate.path / "outputs" / "catalog_context.json"
    _write_json(output, {"schema_version": 1, "candidate_id": candidate.candidate_id, "generated_at": _now(), "retrievals": entries})
    return output


def fetch_catalog(
    candidate: CandidateWorkspace,
    provider_names: Sequence[str],
    transport: Optional[Callable[[CatalogRequest], TransportResponse]] = None,
) -> List[Path]:
    """Fetch reviewed provider templates into new append-only retrieval directories."""
    names = list(provider_names)
    if not names:
        raise ValueError("at least one allowlisted catalog provider is required")
    unknown = sorted(set(names).difference(PROVIDERS))
    if unknown:
        raise ValueError("unsupported catalog provider: {0}".format(", ".join(unknown)))
    transport = transport or _default_transport
    manifests = []
    for provider_name in names:
        spec = PROVIDERS[provider_name]
        retrieval_id = "{0}-{1}".format(datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz"), uuid.uuid4().hex[:12])
        raw_dir, run_dir = _retrieval_paths(candidate, provider_name, retrieval_id)
        raw_dir.mkdir(parents=True, exist_ok=False)
        run_dir.mkdir(parents=True, exist_ok=False)
        retrieved_at = _now()
        try:
            request = _request_for(spec, candidate)
            response, attempts, transport_error = _retry_transport(request, transport)
        except ValueError as exc:
            request = CatalogRequest("GET", spec.official_source_uri, {}, None, {})
            response, attempts, transport_error = TransportResponse(0, {}, b""), 0, str(exc)
        response_path = raw_dir / "response.bin"
        response_path.write_bytes(response.body)
        rows, parse_error = _parse_rows(response.body, spec.expected_format)
        if parse_error is None:
            parse_error = _validate_known_signal_columns(spec, rows)
        status = "unavailable" if transport_error else _response_status(response, rows, parse_error, spec)
        expires_at = (datetime.now(timezone.utc) + RETRIEVAL_TTL).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        raw_metadata_path = raw_dir / "response-metadata.json"
        _write_json(raw_metadata_path, {
            "schema_version": 1, "candidate_id": candidate.candidate_id, "provider": provider_name,
            "retrieval_id": retrieval_id, "source_uri": request.source_uri, "release": spec.release,
            "query_template_id": spec.template_id, "request_method": request.method, "request_parameters": dict(request.parameters),
            "retrieved_at": retrieved_at, "status": status, "http_status": response.status_code,
            "http_cache": {key.lower(): value for key, value in response.headers.items() if key.lower() in ("etag", "last-modified", "cache-control")},
            "response_sha256": _sha256_file(response_path), "citation": spec.citation,
            "units_note": spec.units_note, "native_time_scale": spec.native_time_scale,
            "access_policy_uri": spec.access_policy_uri, "rate_limit_note": spec.rate_limit_note,
            "failure_policy": spec.failure_policy,
        })
        parser_log_path = run_dir / "parser-log.json"
        _write_json(parser_log_path, {"schema_version": 1, "provider": provider_name, "retrieval_id": retrieval_id, "parser": {"name": "exonym-catalog", "version": "1"}, "known_signal_required_columns": list(KNOWN_SIGNAL_REQUIRED_COLUMNS.get(provider_name, ())), "record_count": len(rows), "message": transport_error or parse_error or "parsed"})
        snapshot_path = run_dir / "snapshot.json"
        _write_json(snapshot_path, {
            "schema_version": 1, "candidate_id": candidate.candidate_id, "provider": provider_name,
            "retrieval_id": retrieval_id, "status": status, "source_uri": request.source_uri,
            "release": spec.release, "query_template_id": spec.template_id, "retrieved_at": retrieved_at,
            "raw_response": _artifact(response_path, candidate, "raw-response"), "citation": spec.citation,
            "units_note": spec.units_note, "native_time_scale": spec.native_time_scale, "records": rows,
            "normalization_policy": "Source-native values, time scales, passbands, and quality flags are retained without conversion or inferred solar defaults.",
        })
        cross_match_path = run_dir / "cross-match.json"
        _write_json(cross_match_path, {
            "schema_version": 1, "candidate_id": candidate.candidate_id, "provider": provider_name,
            "retrieval_id": retrieval_id, "snapshot_path": snapshot_path.relative_to(candidate.path).as_posix(),
            "snapshot_sha256": _sha256_file(snapshot_path),
            "cross_match": normalize_cross_matches(_candidate_crossmatch_astrometry(candidate), rows),
        })
        # These source-specific containers retain source-native fields only.
        for filename, record_type in (
            ("stellar-parameters.json", "stellar-parameters"), ("stellar-photometry.json", "stellar-photometry"),
            ("archive-discovery.json", "archive-discovery"), ("contrast-curves.json", "contrast-curves"),
        ):
            _write_json(run_dir / filename, {
                "schema_version": 1, "candidate_id": candidate.candidate_id, "provider": provider_name,
                "retrieval_id": retrieval_id, "status": status, "snapshot_path": snapshot_path.relative_to(candidate.path).as_posix(),
                "snapshot_sha256": _sha256_file(snapshot_path), "record_type": record_type,
                "records": (
                    _normalized_records(rows, spec.archive_collections)
                    if record_type in spec.normalized_record_types else []
                ),
            })
        manifest_path = run_dir / "query-manifest.json"
        _write_json(manifest_path, {
            "schema_version": 1, "candidate_id": candidate.candidate_id, "provider": provider_name,
            "retrieval_id": retrieval_id, "source_uri": request.source_uri, "release": spec.release,
            "query_template_id": spec.template_id, "request_method": request.method, "request_parameters": dict(request.parameters),
            "retrieved_at": retrieved_at, "expires_at": expires_at, "status": status, "attempts": attempts,
            "citation": spec.citation, "artifacts": [
                _artifact(response_path, candidate, "raw-response"), _artifact(raw_metadata_path, candidate, "raw-metadata"),
                _artifact(snapshot_path, candidate, "normalized-snapshot"), _artifact(parser_log_path, candidate, "parser-log"),
                _artifact(cross_match_path, candidate, "cross-match"),
            ],
        })
        manifests.append(manifest_path)
    _write_context(candidate)
    return manifests


def refresh_catalog(candidate: CandidateWorkspace, transport: Optional[Callable[[CatalogRequest], TransportResponse]] = None) -> List[Path]:
    """Create fresh retrievals only for providers whose latest snapshot is expired."""
    latest: Dict[str, Mapping[str, Any]] = {}
    for path in sorted((candidate.path / "runs" / "catalog").glob("*/*/query-manifest.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        if isinstance(record, dict) and record.get("provider") in PROVIDERS:
            provider = str(record["provider"])
            if provider not in latest or str(record.get("retrieved_at", "")) > str(latest[provider].get("retrieved_at", "")):
                latest[provider] = record
    now = datetime.now(timezone.utc)
    expired = []
    for provider, record in latest.items():
        try:
            expiry = datetime.fromisoformat(str(record["expires_at"]).replace("Z", "+00:00"))
        except (KeyError, ValueError):
            expired.append(provider)
            continue
        if expiry <= now:
            expired.append(provider)
    return fetch_catalog(candidate, expired, transport) if expired else []


def catalog_report(candidate: CandidateWorkspace) -> Dict[str, Any]:
    """Return retrieval status, stale evidence, citations, and ambiguity for human review."""
    now = datetime.now(timezone.utc)
    records = []
    for path in sorted((candidate.path / "runs" / "catalog").glob("*/*/query-manifest.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            expiry = datetime.fromisoformat(str(record["expires_at"]).replace("Z", "+00:00"))
        except (OSError, UnicodeError, ValueError, KeyError):
            continue
        records.append({
            "provider": record.get("provider"), "retrieval_id": record.get("retrieval_id"),
            "status": record.get("status"), "stale": expiry <= now, "citation": record.get("citation"),
            "manifest_path": path.relative_to(candidate.path).as_posix(),
        })
    return {"candidate_id": candidate.candidate_id, "retrievals": records, "note": "Catalog context is retrieval-time evidence only; it creates no claim or workflow state."}
