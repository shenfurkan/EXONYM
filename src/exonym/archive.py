"""Target-neutral multi-archive evidence collection for vetting.

The service retrieves candidate-owned archival context such as astrometric
quality, nearby-source geometry, and publicly reported follow-up metadata.
Coordinates, identifiers, and query radii enter through the workspace or CLI;
the shared implementation does not embed target-specific records.

Astrophysical rationale:
    Astrometric quality and nearby-source evidence can expose blends or an
    unresolved companion, but neither is a direct measurement of transit
    origin. Angular separations use a wrapped right-ascension difference so a
    query close to the coordinate discontinuity retains local geometry.

Scientific boundary:
    Archive availability, crowding summaries, and quality flags are screening
    evidence only. They neither calibrate a photometric scene model nor create
    a novelty, disposition, or validation claim.

Units, retained reference, and fail-closed boundary
----------------------------------------------------
Archive positions are ICRS degrees; separations are arcsec or mas as named;
proper motions are mas yr^-1; parallax is mas; catalog magnitudes/errors are
mag; and provider time values retain their declared scale rather than being
silently converted.  The exact Pogson error propagation used here is supported
by Evans et al. (2018), ADS ``2018A&A...616A...4E``, DOI
``10.1051/0004-6361/201832756``.  Provider availability, response completeness,
and selection functions are external evidence limitations, not formulas this
module resolves.  Network failure, malformed response, missing target context,
or a search radius below the stated crowding boundary is recorded as unavailable
or blocked, never as an empty sky or a claim-eligible result.
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import ARCSECONDS_PER_DEGREE, MILLIARCSECONDS_PER_ARCSECOND
from .inputs import load_stellar_parameters, load_tpf_cubes
from .workspace import CandidateWorkspace


ARCHIVAL_REPORT_RELATIVE_PATH = Path("outputs") / "archival_vetting_report.json"
DEFAULT_HTTP_TIMEOUT_SECONDS = 8.0
DEFAULT_HTTP_MAX_RETRIES = 2
# ASTROPHYSICAL_HEURISTIC: A cone smaller than a detector pixel cannot support
# a negative crowding assessment for the photometric aperture.
DEFAULT_ARCHIVE_SEARCH_RADIUS_ARCSEC = 60.0
MINIMUM_CROWDING_SEARCH_RADIUS_ARCSEC = 21.0
# Exact Pogson magnitude-error propagation factor:
#     sigma_m = |dm/dF| * sigma_F,   m = -2.5 * log10(F) + ZP
#   => sigma_m = (2.5 / ln 10) * (sigma_F / F)
# (Evans et al. 2018, A&A, 616, A4, doi:10.1051/0004-6361/201832756, flux-error
# propagation of Gaia photometry).  Tier 3 analytic identity: the factor is
# evaluated from math constants only; truncated decimals such as 1.086 or
# 1.0857 are prohibited (AGENTS.md Rule 13).
POGSON_MAGNITUDE_FACTOR = 2.5 / math.log(10.0)


def _utc_timestamp() -> str:
    """Return a compact, timezone-aware timestamp for archival evidence."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class ArchivalVettingService:
    """Query archive providers and retain candidate-local screening evidence.

    Args:
        timeout: Positive network timeout in seconds for one request attempt.
        max_retries: Positive number of bounded transport attempts.
        retry_backoff_factor: Non-negative delay multiplier between attempts.

    The class records provider outcomes instead of converting an unavailable
    remote service into an absence-of-evidence conclusion.
    """

    ESA_GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
    AIP_GAIA_TAP_URL = "https://gaia.aip.de/tap/sync"
    MIRROR_GAIA_TAP_URL = "https://gaia.gec.asiaa.sinica.edu.tw/tap-server/tap/sync"
    ESA_GAIA_TABLE = "gaiadr3.gaia_source"
    ESA_GAIA_FLAME_TABLE = "gaiadr3.astrophysical_parameters"
    MIRROR_GAIA_TABLE = "gaiadr3.gaia_source"
    VIZIER_GAIA_CATALOG = "I/355/gaiadr3"
    TARGET_PRESENCE_ARCSEC = 2.0
    EXOFOP_TARGET_EPOCH_JYEAR = 2000.0
    MAX_JSON_RESPONSE_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_HTTP_MAX_RETRIES,
        retry_backoff_factor: float = 0.5,
    ) -> None:
        timeout_value = _finite_float(timeout)
        backoff_value = _finite_float(retry_backoff_factor)
        if timeout_value is None or timeout_value <= 0.0:
            raise ValueError("timeout must be positive and finite")
        if int(max_retries) < 1:
            raise ValueError("max_retries must be at least one")
        if backoff_value is None or backoff_value < 0.0:
            raise ValueError("retry_backoff_factor must be finite and non-negative")
        self.timeout = timeout_value
        self.max_retries = int(max_retries)
        self.retry_backoff_factor = backoff_value

    @staticmethod
    def _angular_separation_arcsec(
        first_ra_deg: float,
        first_dec_deg: float,
        second_ra_deg: float,
        second_dec_deg: float,
    ) -> Optional[float]:
        """Return a small-angle separation in arcsec for finite ICRS positions."""
        values = (
            _finite_float(first_ra_deg),
            _finite_float(first_dec_deg),
            _finite_float(second_ra_deg),
            _finite_float(second_dec_deg),
        )
        if any(value is None for value in values):
            return None
        first_ra, first_dec, second_ra, second_dec = values
        assert first_ra is not None
        assert first_dec is not None
        assert second_ra is not None
        assert second_dec is not None
        if not (-90.0 <= first_dec <= 90.0 and -90.0 <= second_dec <= 90.0):
            return None
        delta_ra_deg = (first_ra - second_ra + 180.0) % 360.0 - 180.0
        separation = math.hypot(
            delta_ra_deg * math.cos(math.radians(second_dec)) * ARCSECONDS_PER_DEGREE,
            (first_dec - second_dec) * ARCSECONDS_PER_DEGREE,
        )
        return separation if math.isfinite(separation) else None

    @classmethod
    def _propagated_j2000_separation_arcsec(
        cls,
        source: Dict[str, Any],
        target_ra_deg: float,
        target_dec_deg: float,
    ) -> Optional[float]:
        """Compare a Gaia position propagated to J2000 with an ExoFOP position.

        ExoFOP target coordinates are conventionally supplied at J2000.  A
        direct Gaia DR3 match can therefore fail for a high-proper-motion star.
        Propagation is used only when both proper-motion components and a Gaia
        reference epoch are available; it never widens the positional radius by
        itself.
        """
        source_ra = _finite_float(source.get("ra_deg"))
        source_dec = _finite_float(source.get("dec_deg"))
        pmra = _finite_float(source.get("pmra_mas_per_year"))
        pmdec = _finite_float(source.get("pmdec_mas_per_year"))
        reference_epoch = _finite_float(source.get("reference_epoch_jyear"))
        if (
            source_ra is None
            or source_dec is None
            or pmra is None
            or pmdec is None
            or reference_epoch is None
        ):
            return None
        cosine_dec = math.cos(math.radians(source_dec))
        if not math.isfinite(cosine_dec) or abs(cosine_dec) < 1e-8:
            return None
        elapsed_years = cls.EXOFOP_TARGET_EPOCH_JYEAR - reference_epoch
        propagated_ra = (
            source_ra
            + pmra
            * elapsed_years
            / (MILLIARCSECONDS_PER_ARCSECOND * ARCSECONDS_PER_DEGREE * cosine_dec)
        ) % 360.0
        propagated_dec = source_dec + pmdec * elapsed_years / (
            MILLIARCSECONDS_PER_ARCSECOND * ARCSECONDS_PER_DEGREE
        )
        if not (-90.0 <= propagated_dec <= 90.0):
            return None
        return cls._angular_separation_arcsec(
            propagated_ra,
            propagated_dec,
            target_ra_deg,
            target_dec_deg,
        )

    @classmethod
    def _mark_target_matches(
        cls,
        sources: List[Dict[str, Any]],
        target_ra_deg: float,
        target_dec_deg: float,
    ) -> None:
        """Mark direct or proper-motion-propagated target matches in-place."""
        for source in sources:
            native_separation = _finite_float(source.get("separation_arcsec"))
            propagated_separation = cls._propagated_j2000_separation_arcsec(
                source,
                target_ra_deg,
                target_dec_deg,
            )
            source["j2000_separation_arcsec"] = (
                float(propagated_separation)
                if propagated_separation is not None
                else None
            )
            source["target_match_method"] = None
            source["target_match_separation_arcsec"] = None
            if (
                native_separation is not None
                and native_separation <= cls.TARGET_PRESENCE_ARCSEC
            ):
                source["target_match_method"] = "native_position"
                source["target_match_separation_arcsec"] = native_separation
            elif (
                propagated_separation is not None
                and propagated_separation <= cls.TARGET_PRESENCE_ARCSEC
            ):
                source["target_match_method"] = "proper_motion_to_j2000"
                source["target_match_separation_arcsec"] = propagated_separation

    def _http_get_json(self, url: str) -> Optional[Any]:
        """Perform an HTTP GET request with retry logic and timeout handling."""
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "exonym-archive/1.2.0 (astronomy-research-framework)",
                        "Accept": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    if response.status == 200:
                        raw_bytes = response.read(self.MAX_JSON_RESPONSE_BYTES + 1)
                        if len(raw_bytes) > self.MAX_JSON_RESPONSE_BYTES:
                            return None
                        raw_data = raw_bytes.decode("utf-8", errors="replace")
                        try:
                            return json.loads(raw_data)
                        except json.JSONDecodeError:
                            return None
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff_factor * (2**attempt))
        return None

    def _gaia_sources_tap(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float,
        base_url: str,
        table_name: str,
    ) -> List[Dict[str, Any]]:
        """Cone search via a TAP sync endpoint returning JSON row data.

        The selected columns end with the Gaia mean-flux and flux-error
        columns so magnitude uncertainties propagate with the exact Pogson
        factor.  Legacy rows without those columns still parse; their
        magnitude errors remain unmeasured (``None``) instead of fabricated.
        """
        source_alias = "gs"
        flame_alias = "flame"
        source_columns = (
            f"{source_alias}.source_id, {source_alias}.ra, {source_alias}.dec, "
            f"{source_alias}.phot_g_mean_mag, {source_alias}.phot_bp_mean_mag, "
            f"{source_alias}.phot_rp_mean_mag, {source_alias}.ruwe, {source_alias}.pmra, "
            f"{source_alias}.pmdec, {source_alias}.ref_epoch, "
            f"DISTANCE(POINT('ICRS', {source_alias}.ra, {source_alias}.dec), "
            f"POINT('ICRS', {ra}, {dec}))*{ARCSECONDS_PER_DEGREE} AS sep_arcsec, "
            f"{source_alias}.phot_g_mean_flux, {source_alias}.phot_g_mean_flux_error, "
            f"{source_alias}.phot_bp_mean_flux, {source_alias}.phot_bp_mean_flux_error, "
            f"{source_alias}.phot_rp_mean_flux, {source_alias}.phot_rp_mean_flux_error"
        )
        flame_columns = (
            f", {flame_alias}.mass_flame, {flame_alias}.radius_flame, "
            f"{flame_alias}.evolstage_flame, {flame_alias}.flags_flame, "
            f"{flame_alias}.age_flame, {flame_alias}.mass_flame_lower, "
            f"{flame_alias}.mass_flame_upper, {flame_alias}.radius_flame_lower, "
            f"{flame_alias}.radius_flame_upper, {flame_alias}.age_flame_lower, "
            f"{flame_alias}.age_flame_upper"
        )
        if base_url == self.ESA_GAIA_TAP_URL and table_name == self.ESA_GAIA_TABLE:
            source_clause = (
                f"FROM {table_name} AS {source_alias} LEFT OUTER JOIN "
                f"{self.ESA_GAIA_FLAME_TABLE} AS {flame_alias} ON "
                f"{source_alias}.source_id={flame_alias}.source_id "
            )
            selected_columns = source_columns + flame_columns
        else:
            source_clause = f"FROM {table_name} AS {source_alias} "
            selected_columns = source_columns
        tap_query = (
            f"SELECT {selected_columns} {source_clause}"
            f"WHERE 1=CONTAINS(POINT('ICRS', {source_alias}.ra, {source_alias}.dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec / ARCSECONDS_PER_DEGREE})) "
            f"ORDER BY sep_arcsec ASC"
        )
        url = (
            f"{base_url}?REQUEST=doQuery&LANG=ADQL&FORMAT=json&QUERY={urllib.parse.quote(tap_query)}"
        )
        data = self._http_get_json(url)
        if not (isinstance(data, dict) and data.get("data")):
            return []
        sources: List[Dict[str, Any]] = []
        for row in data["data"]:
            sid = str(row[0]) if len(row) > 0 else "unknown"
            ra_val = _finite_float(row[1]) if len(row) > 1 else None
            dec_val = _finite_float(row[2]) if len(row) > 2 else None
            gmag = _finite_float(row[3]) if len(row) > 3 else None
            if len(row) >= 11:
                bp_mag = _finite_float(row[4])
                rp_mag = _finite_float(row[5])
                ruwe_val = _finite_float(row[6])
                pmra_val = _finite_float(row[7])
                pmdec_val = _finite_float(row[8])
                reference_epoch = _finite_float(row[9])
                sep_val = _finite_float(row[10])
            else:
                bp_mag = None
                rp_mag = None
                ruwe_val = _finite_float(row[4]) if len(row) > 4 else None
                pmra_val = _finite_float(row[5]) if len(row) > 5 else None
                pmdec_val = _finite_float(row[6]) if len(row) > 6 else None
                reference_epoch = _finite_float(row[7]) if len(row) > 7 else None
                sep_val = _finite_float(row[8]) if len(row) > 8 else None
            # Backward-compatible handling of fixtures and cached responses
            # produced before proper-motion columns were requested.
            if sep_val is None and len(row) == 6:
                sep_val = _finite_float(row[5])
                pmra_val = None
            if sep_val is None or sep_val < 0.0:
                continue
            if len(row) >= 17:
                g_flux = _finite_float(row[11])
                g_flux_err = _finite_float(row[12])
                bp_flux = _finite_float(row[13])
                bp_flux_err = _finite_float(row[14])
                rp_flux = _finite_float(row[15])
                rp_flux_err = _finite_float(row[16])
            else:
                g_flux = g_flux_err = None
                bp_flux = bp_flux_err = None
                rp_flux = rp_flux_err = None
            if len(row) >= 28:
                mass_flame = _finite_float(row[17])
                radius_flame = _finite_float(row[18])
                evolstage_flame = _finite_float(row[19])
                flags_flame = str(row[20]) if row[20] is not None else None
                age_flame = _finite_float(row[21])
                mass_flame_lower = _finite_float(row[22])
                mass_flame_upper = _finite_float(row[23])
                radius_flame_lower = _finite_float(row[24])
                radius_flame_upper = _finite_float(row[25])
                age_flame_lower = _finite_float(row[26])
                age_flame_upper = _finite_float(row[27])
            else:
                mass_flame = radius_flame = evolstage_flame = flags_flame = age_flame = None
                mass_flame_lower = mass_flame_upper = None
                radius_flame_lower = radius_flame_upper = None
                age_flame_lower = age_flame_upper = None

            sources.append(
                {
                    "source_id": sid,
                    "ra_deg": float(ra_val) if ra_val is not None else None,
                    "dec_deg": float(dec_val) if dec_val is not None else None,
                    "separation_arcsec": float(sep_val),
                    "ruwe": float(ruwe_val) if ruwe_val is not None else None,
                    "phot_g_mean_mag": float(gmag) if gmag is not None else None,
                    "phot_g_mean_mag_error": self._pogson_magnitude_error(g_flux, g_flux_err),

                    "phot_bp_mean_mag": float(bp_mag) if bp_mag is not None else None,
                    "phot_bp_mean_mag_error": self._pogson_magnitude_error(bp_flux, bp_flux_err),

                    "phot_rp_mean_mag": float(rp_mag) if rp_mag is not None else None,
                    "phot_rp_mean_mag_error": self._pogson_magnitude_error(rp_flux, rp_flux_err),

                    "pmra_mas_per_year": float(pmra_val)
                    if pmra_val is not None
                    else None,
                    "pmdec_mas_per_year": float(pmdec_val)
                    if pmdec_val is not None
                    else None,
                    "reference_epoch_jyear": float(reference_epoch)
                    if reference_epoch is not None
                    else None,
                    "mass_flame": float(mass_flame) if mass_flame is not None else None,
                    "radius_flame": float(radius_flame) if radius_flame is not None else None,
                    "evolstage_flame": int(evolstage_flame)
                    if evolstage_flame is not None and evolstage_flame.is_integer()
                    else None,
                    "flags_flame": flags_flame,
                    "age_flame_gyr": float(age_flame) if age_flame is not None else None,
                    "mass_flame_lower_solar": float(mass_flame_lower)
                    if mass_flame_lower is not None
                    else None,
                    "mass_flame_upper_solar": float(mass_flame_upper)
                    if mass_flame_upper is not None
                    else None,
                    "radius_flame_lower_solar": float(radius_flame_lower)
                    if radius_flame_lower is not None
                    else None,
                    "radius_flame_upper_solar": float(radius_flame_upper)
                    if radius_flame_upper is not None
                    else None,
                    "age_flame_lower_gyr": float(age_flame_lower)
                    if age_flame_lower is not None
                    else None,
                    "age_flame_upper_gyr": float(age_flame_upper)
                    if age_flame_upper is not None
                    else None,
                }
            )
        sources.sort(key=lambda item: item["separation_arcsec"])
        return sources

    @staticmethod
    def _pogson_magnitude_error(flux: Any, flux_error: Any) -> Optional[float]:
        """Return ``sigma_m = POGSON_MAGNITUDE_FACTOR * sigma_F / F`` when valid.

        The mean flux and its error must both be finite and strictly positive;
        any other input yields ``None`` so an unmeasured magnitude error is
        never substituted with a fabricated value.
        """
        flux_value = _finite_float(flux)
        error_value = _finite_float(flux_error)
        if (
            flux_value is None
            or error_value is None
            or flux_value <= 0.0
            or error_value <= 0.0
        ):
            return None
        return POGSON_MAGNITUDE_FACTOR * (error_value / flux_value)

    def _gaia_sources_vizier(
        self, ra: float, dec: float, radius_arcsec: float
    ) -> List[Dict[str, Any]]:
        """Cone search via VizieR Gaia DR3 (independent of Gaia TAP outages)."""
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier

        vizier = Vizier(row_limit=-1, timeout=self.timeout)
        coordinate = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        result = vizier.query_region(
            coordinate, radius=radius_arcsec * u.arcsec, catalog=self.VIZIER_GAIA_CATALOG
        )
        if not result or len(result) == 0:
            return []
        sources: List[Dict[str, Any]] = []
        for row in result[0]:
            try:
                row_ra = _finite_float(row["RA_ICRS"])
                row_dec = _finite_float(row["DE_ICRS"])
            except KeyError:
                row_ra = None
                row_dec = None
            if row_ra is None or row_dec is None:
                continue
            # NUMERICAL_GUARD: reuse the wrapped local-sky separation used by
            # TAP responses so the VizieR fallback handles the RA discontinuity.
            sep_arcsec = self._angular_separation_arcsec(row_ra, row_dec, ra, dec)
            if sep_arcsec is None:
                continue
            ruwe_val: Optional[float] = None
            try:
                ruwe_val = _finite_float(row["RUWE"])
            except KeyError:
                ruwe_val = None
            g_mag: Optional[float] = None
            try:
                g_mag = _finite_float(row["Gmag"])
            except KeyError:
                g_mag = None
            bp_mag: Optional[float] = None
            rp_mag: Optional[float] = None
            for column, destination in (("BPmag", "bp"), ("RPmag", "rp")):
                try:
                    value = _finite_float(row[column])
                except KeyError:
                    value = None
                if destination == "bp":
                    bp_mag = value
                else:
                    rp_mag = value
            pmra_val: Optional[float] = None
            for column in ("pmRA", "pmra"):
                try:
                    pmra_val = _finite_float(row[column])
                except KeyError:
                    continue
                if pmra_val is not None:
                    break
            pmdec_val: Optional[float] = None
            for column in ("pmDE", "pmdec"):
                try:
                    pmdec_val = _finite_float(row[column])
                except KeyError:
                    continue
                if pmdec_val is not None:
                    break
            reference_epoch: Optional[float] = None
            for column in ("Epoch", "RefEpoch", "ref_epoch"):
                try:
                    reference_epoch = _finite_float(row[column])
                except KeyError:
                    continue
                if reference_epoch is not None:
                    break
            flux_values: Dict[str, Optional[float]] = {}
            for column in ("FG", "e_FG", "FBP", "e_FBP", "FRP", "e_FRP"):
                try:
                    flux_values[column] = _finite_float(row[column])
                except KeyError:
                    flux_values[column] = None

            sources.append(
                {
                    "source_id": str(row["Source"]) if "Source" in row.colnames else "unknown",
                    "ra_deg": float(row_ra),
                    "dec_deg": float(row_dec),
                    "separation_arcsec": float(sep_arcsec),
                    "ruwe": float(ruwe_val) if ruwe_val is not None else None,
                    "phot_g_mean_mag": float(g_mag) if g_mag is not None else None,
                    "phot_g_mean_mag_error": self._pogson_magnitude_error(flux_values["FG"], flux_values["e_FG"]),

                    "phot_bp_mean_mag": float(bp_mag) if bp_mag is not None else None,
                    "phot_bp_mean_mag_error": self._pogson_magnitude_error(flux_values["FBP"], flux_values["e_FBP"]),

                    "phot_rp_mean_mag": float(rp_mag) if rp_mag is not None else None,
                    "phot_rp_mean_mag_error": self._pogson_magnitude_error(flux_values["FRP"], flux_values["e_FRP"]),

                    "pmra_mas_per_year": float(pmra_val)
                    if pmra_val is not None
                    else None,
                    "pmdec_mas_per_year": float(pmdec_val)
                    if pmdec_val is not None
                    else None,
                    "reference_epoch_jyear": float(reference_epoch)
                    if reference_epoch is not None
                    else None,
                }
            )
        sources.sort(key=lambda item: item["separation_arcsec"])
        return sources

    def query_gaia_astrometry(
        self, ra: float, dec: float, radius_arcsec: float = DEFAULT_ARCHIVE_SEARCH_RADIUS_ARCSEC
    ) -> Dict[str, Any]:
        """Cone search Gaia DR3 for celestial sources around target coordinates.

        Queries bounded backends in order (ESA TAP sync gaia_source, CDS
        VizieR I/355/gaiadr3, AIP Leibniz TAP gaia_source, ASIAA mirror) with
        8-second timeouts and
        adopts the first result validated by a source inside
        ``TARGET_PRESENCE_ARCSEC`` of the target. For high-proper-motion
        targets, a Gaia source may instead be propagated from its catalog
        reference epoch to the J2000 ExoFOP coordinate; missing proper-motion
        metadata never widens the match radius. Extracts RUWE for the matched
        source and flags suspected_binary if RUWE > 1.4.
        """
        ra_value = _finite_float(ra)
        dec_value = _finite_float(dec)
        radius_value = _finite_float(radius_arcsec)
        results: Dict[str, Any] = {
            "target_ra_deg": ra_value,
            "target_dec_deg": dec_value,
            "target_coordinate_epoch_assumption": "J2000.0",
            "search_radius_arcsec": radius_value,
            "target_match_max_arcsec": self.TARGET_PRESENCE_ARCSEC,
            "target_source_id": None,
            "target_separation_arcsec": None,
            "target_native_separation_arcsec": None,
            "target_match_method": None,
            "target_phot_g_mean_mag": None,
            "ruwe": None,
            "suspected_binary": None,
            "nearby_sources_count": 0,
            "sources": [],
            "source": "gaia-dr3",
            "backend": None,
            "validated": False,
            "query_status": "unavailable",
            "query_errors": [],
        }

        if ra_value is None or dec_value is None:
            results["query_errors"].append("target coordinates are not finite")
            return results
        if radius_value is None or radius_value <= 0.0:
            results["query_errors"].append("search radius must be positive and finite")
            return results

        backends = (
            (
                "esa-tap",
                lambda: self._gaia_sources_tap(
                    ra_value, dec_value, radius_value, self.ESA_GAIA_TAP_URL, self.ESA_GAIA_TABLE
                ),
            ),
            (
                "vizier-dr3",
                lambda: self._gaia_sources_vizier(ra_value, dec_value, radius_value),
            ),
            (
                "gaia-aip",
                lambda: self._gaia_sources_tap(
                    ra_value, dec_value, radius_value, self.AIP_GAIA_TAP_URL, self.MIRROR_GAIA_TABLE
                ),
            ),
            (


                "gaia-mirror",
                lambda: self._gaia_sources_tap(
                    ra_value, dec_value, radius_value, self.MIRROR_GAIA_TAP_URL, self.MIRROR_GAIA_TABLE
                ),
            ),
        )

        fallback: Optional[Tuple[str, List[Dict[str, Any]]]] = None
        for backend_name, fetch in backends:
            try:
                sources = fetch()
            except Exception as exc:
                logging.warning(
                    "gaia backend %s failed: %s: %s",
                    backend_name, type(exc).__name__, exc,
                )
                results["query_errors"].append(
                    "{0}: {1}".format(backend_name, type(exc).__name__)
                )
                sources = []
            if not sources:
                continue
            self._mark_target_matches(sources, ra_value, dec_value)
            if fallback is None:
                fallback = (backend_name, sources)
            validated = any(item.get("target_match_method") is not None for item in sources)
            if validated:
                self._apply_gaia_sources(results, sources, backend_name, validated=True)
                results["query_status"] = "ok"
                return results

        if fallback is not None:
            self._apply_gaia_sources(results, fallback[1], fallback[0], validated=False)
            results["query_status"] = "unvalidated"
        return results

    @staticmethod
    def _apply_gaia_sources(
        results: Dict[str, Any],
        sources: List[Dict[str, Any]],
        backend_name: str,
        validated: bool,
    ) -> None:
        """Populate the astrometry result dict from a resolved source list."""
        results["nearby_sources_count"] = len(sources)
        results["sources"] = sources
        results["backend"] = backend_name
        results["validated"] = bool(validated)
        target_sources = [
            source for source in sources if source.get("target_match_method") is not None
        ]
        target_sources.sort(
            key=lambda source: (
                0 if source.get("target_match_method") == "native_position" else 1,
                _finite_float(source.get("target_match_separation_arcsec"))
                if _finite_float(source.get("target_match_separation_arcsec")) is not None
                else float("inf"),
            )
        )
        target_source = target_sources[0] if target_sources else None
        if target_source is not None:
            results["target_source_id"] = target_source["source_id"]
            results["target_separation_arcsec"] = target_source.get(
                "target_match_separation_arcsec"
            )
            results["target_native_separation_arcsec"] = target_source.get(
                "separation_arcsec"
            )
            results["target_match_method"] = target_source.get("target_match_method")
            results["target_phot_g_mean_mag"] = target_source["phot_g_mean_mag"]
        target_ruwe = _finite_float(target_source.get("ruwe")) if target_source is not None else None
        if target_ruwe is not None and target_ruwe > 0.0:
            results["ruwe"] = target_ruwe
            results["suspected_binary"] = target_ruwe > 1.4

    def query_exofop_metadata(self, tic_id: str) -> Dict[str, Any]:
        """Query NASA ExoFOP JSON API for imaging and spectroscopy records for a target TIC.

        API endpoints:
        - https://exofop.ipac.caltech.edu/tess/target.php?id=<TIC_ID>&json
        - https://exofop.ipac.caltech.edu/tess/api.php?target=<TIC_ID>&json
        """
        tic_clean = str(tic_id).strip().lstrip("TIC").strip()
        results: Dict[str, Any] = {
            "tic_id": tic_clean,
            "has_imaging": None,
            "has_spectroscopy": None,
            "imaging_records_count": 0,
            "spectroscopy_records_count": 0,
            "imaging_types": [],
            "spectroscopy_types": [],
            "target_coordinates": None,
            "source": "nasa-exofop",
            "query_status": "not_requested",
            "queried_endpoints": [],
            "retrieved_at_utc": None,
        }

        if not tic_clean:
            return results

        # A failed or malformed archive query must remain distinct from a
        # successful query with no follow-up records.
        url_target = f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic_clean}&json"
        results["queried_endpoints"].append(url_target)
        payload = self._http_get_json(url_target)

        if not payload or not isinstance(payload, dict):
            url_api = f"https://exofop.ipac.caltech.edu/tess/api.php?target={tic_clean}&json"
            results["queried_endpoints"].append(url_api)
            payload = self._http_get_json(url_api)

        if payload and isinstance(payload, dict):
            results["retrieved_at_utc"] = _utc_timestamp()
            coords = payload.get("coordinates") or payload.get("target_coordinates")
            if isinstance(coords, dict):
                ra_val = coords.get("ra") or coords.get("ra_deg")
                dec_val = coords.get("dec") or coords.get("dec_deg")
                if ra_val is not None and dec_val is not None:
                    try:
                        results["target_coordinates"] = {
                            "ra_deg": float(ra_val),
                            "dec_deg": float(dec_val),
                        }
                    except (TypeError, ValueError):
                        pass

            imaging_present = "imaging" in payload or "high_res_imaging" in payload
            imaging = payload.get("imaging") or payload.get("high_res_imaging") or []
            if isinstance(imaging, list):
                results["imaging_records_count"] = len(imaging)
                if imaging_present:
                    results["has_imaging"] = len(imaging) > 0
                types = set()
                for rec in imaging:
                    if isinstance(rec, dict):
                        itype = (
                            rec.get("itype")
                            or rec.get("type")
                            or rec.get("technique")
                            or rec.get("iinst")
                            or rec.get("instrument")
                        )
                        if itype:
                            types.add(str(itype).strip())
                results["imaging_types"] = sorted(list(types))

            spectroscopy_present = any(
                key in payload
                for key in ("spectroscopy", "high_res_spectroscopy", "spectra")
            )
            spectroscopy = (
                payload.get("spectroscopy")
                or payload.get("high_res_spectroscopy")
                or payload.get("spectra")
                or []
            )
            if isinstance(spectroscopy, list):
                results["spectroscopy_records_count"] = len(spectroscopy)
                if spectroscopy_present:
                    results["has_spectroscopy"] = len(spectroscopy) > 0
                stypes = set()
                for rec in spectroscopy:
                    if isinstance(rec, dict):
                        stype = (
                            rec.get("stype")
                            or rec.get("type")
                            or rec.get("technique")
                            or rec.get("sinst")
                            or rec.get("instrument")
                            or rec.get("observation_type")
                        )
                        if stype:
                            stypes.add(str(stype).strip())
                results["spectroscopy_types"] = sorted(list(stypes))

            results["query_status"] = (
                "ok" if imaging_present or spectroscopy_present else "incomplete"
            )
        else:
            results["query_status"] = "unavailable"

        return results

    def synthesize_archival_report(
        self,
        workspace: CandidateWorkspace,
        radius_arcsec: float = DEFAULT_ARCHIVE_SEARCH_RADIUS_ARCSEC,
    ) -> Dict[str, Any]:
        """Synthesize Gaia astrometry and ExoFOP metadata for a candidate workspace."""
        identifiers = workspace.metadata.get("identifiers", {})
        tic_id = identifiers.get("tic")
        toi_id = identifiers.get("toi")
        candidate_id = workspace.candidate_id
        radius_value = _finite_float(radius_arcsec)
        if radius_value is None or radius_value <= 0.0:
            raise ValueError("radius_arcsec must be positive and finite")

        exofop_data: Dict[str, Any] = {}
        if tic_id:
            exofop_data = self.query_exofop_metadata(str(tic_id))

        ra_deg: Optional[float] = None
        dec_deg: Optional[float] = None

        params = load_stellar_parameters(workspace)
        if "ra_deg" in params and "dec_deg" in params:
            ra_deg = float(params["ra_deg"])
            dec_deg = float(params["dec_deg"])

        if ra_deg is None or dec_deg is None:
            cubes = load_tpf_cubes(workspace, require_raw_provenance=True)
            for cube in cubes:
                header = cube.get("header", {})
                if "RA_OBJ" in header and "DEC_OBJ" in header:
                    try:
                        ra_deg = float(header["RA_OBJ"])
                        dec_deg = float(header["DEC_OBJ"])
                        break
                    except (TypeError, ValueError):
                        pass

        if (ra_deg is None or dec_deg is None) and exofop_data.get("target_coordinates"):
            coords = exofop_data["target_coordinates"]
            ra_deg = coords.get("ra_deg")
            dec_deg = coords.get("dec_deg")

        if ra_deg is None or dec_deg is None:
            gaia_data: Dict[str, Any] = {
                "target_ra_deg": None,
                "target_dec_deg": None,
                "search_radius_arcsec": radius_value,
                "ruwe": None,
                "suspected_binary": None,
                "nearby_sources_count": None,
                "sources": [],
                "source": "gaia-dr3",
                "backend": None,
                "validated": False,
                "query_status": "unavailable",
                "query_errors": ["target coordinates unavailable"],
            }
        else:
            gaia_data = self.query_gaia_astrometry(
                ra_deg, dec_deg, radius_arcsec=radius_value
            )

        ruwe_val = gaia_data.get("ruwe")
        gaia_status = gaia_data.get("query_status", "ok")
        exofop_status = exofop_data.get("query_status", "ok") if exofop_data else "not_requested"
        gaia_available = gaia_status == "ok" and gaia_data.get("validated") is True
        exofop_available = exofop_status == "ok"
        ruwe = _finite_float(ruwe_val)
        is_hidden_binary = (
            gaia_data.get("suspected_binary")
            if gaia_available
            and ruwe is not None
            and ruwe > 0.0
            and isinstance(gaia_data.get("suspected_binary"), bool)
            else None
        )
        nearby_count = gaia_data.get("nearby_sources_count")
        crowding_radius_sufficient = radius_value >= MINIMUM_CROWDING_SEARCH_RADIUS_ARCSEC
        has_nearby_contaminants = (
            bool(nearby_count > 1)
            if gaia_available and crowding_radius_sufficient and isinstance(nearby_count, int)
            else None
        )
        has_imaging = exofop_data.get("has_imaging") if exofop_available else None
        has_spectroscopy = exofop_data.get("has_spectroscopy") if exofop_available else None
        has_ground_based_followup = (
            bool(has_imaging or has_spectroscopy) if exofop_available else None
        )

        ruwe_str = f"{ruwe:.4f}" if ruwe is not None else "N/A"
        if is_hidden_binary is None:
            evidence_binary = "Gaia astrometry unavailable or unvalidated; binarity status is unknown"
        elif is_hidden_binary:
            evidence_binary = (
                f"Gaia RUWE ({ruwe_str}) > 1.4 flags possible unresolved multiplicity or astrometric mismatch"
            )
        else:
            evidence_binary = (
                f"Gaia RUWE ({ruwe_str}) does not exceed 1.4; this does not exclude companions"
            )

        if not crowding_radius_sufficient:
            evidence_crowding = (
                "Gaia search radius is below one TESS pixel; crowding status is unknown"
            )
        elif has_nearby_contaminants is None:
            evidence_crowding = "Gaia astrometry unavailable or unvalidated; crowding status is unknown"
        elif has_nearby_contaminants:
            evidence_crowding = (
                f"{nearby_count} celestial sources detected within {radius_value}\" radius"
            )
        else:
            evidence_crowding = (
                f"No additional Gaia sources were detected within {radius_value}\" radius"
            )

        imaging_types_str = ", ".join(exofop_data.get("imaging_types", [])) or "Registered"
        spectroscopy_types_str = ", ".join(exofop_data.get("spectroscopy_types", [])) or "Registered"
        followup_parts = []
        if has_imaging:
            followup_parts.append(f"High-res imaging ({imaging_types_str})")
        if has_spectroscopy:
            followup_parts.append(f"Spectroscopy ({spectroscopy_types_str})")
        if has_ground_based_followup is None:
            evidence_followup = (
                "ExoFOP follow-up query was unavailable or incomplete; follow-up status is unknown"
            )
        elif has_ground_based_followup:
            evidence_followup = "Ground-based follow-up on ExoFOP: " + "; ".join(followup_parts)
        else:
            evidence_followup = "No high-resolution follow-up records were returned by ExoFOP"

        return {
            "candidate_id": candidate_id,
            "tic_id": str(tic_id) if tic_id else None,
            "toi_id": str(toi_id) if toi_id else None,
            "target_coordinates": {
                "ra_deg": float(ra_deg) if ra_deg is not None else None,
                "dec_deg": float(dec_deg) if dec_deg is not None else None,
            },
            "scientific_assessment": {
                "1_is_hidden_binary": {
                    "answer": is_hidden_binary,
                    "ruwe": ruwe,
                    "threshold": 1.4,
                    "availability": gaia_status,
                    "evidence": evidence_binary,
                },
                "2_has_nearby_contaminants": {
                    "answer": has_nearby_contaminants,
                    "search_radius_arcsec": radius_value,
                    "search_radius_sufficient_for_crowding": crowding_radius_sufficient,
                    "nearby_sources_count": nearby_count,
                    "availability": gaia_status,
                    "evidence": evidence_crowding,
                },
                "3_has_ground_based_followup": {
                    "answer": has_ground_based_followup,
                    "has_high_res_imaging": has_imaging,
                    "has_spectroscopy": has_spectroscopy,
                    "availability": exofop_status,
                    "evidence": evidence_followup,
                },
            },
            "gaia_astrometry": gaia_data,
            "exofop_metadata": exofop_data,
            "timestamp_utc": _utc_timestamp(),
        }


def run_archival_vetting(
    workspace: CandidateWorkspace,
    radius_arcsec: float = DEFAULT_ARCHIVE_SEARCH_RADIUS_ARCSEC,
    service: Optional[ArchivalVettingService] = None,
) -> Path:
    """Run multi-archive vetting on candidate workspace and write output report JSON.

    Returns the absolute path to outputs/archival_vetting_report.json.
    """
    if service is None:
        service = ArchivalVettingService()

    report = service.synthesize_archival_report(workspace, radius_arcsec=radius_arcsec)
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    report_path = outputs_dir / ARCHIVAL_REPORT_RELATIVE_PATH.name
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path


def _finite_float(value: Any) -> Optional[float]:
    """Return a finite numeric value, or ``None`` for invalid report fields."""
    if np.ma.is_masked(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _load_archival_report(workspace: CandidateWorkspace) -> Optional[Dict[str, Any]]:
    """Load one archival report without accepting ambiguous JSON records."""
    path = workspace.path / ARCHIVAL_REPORT_RELATIVE_PATH
    if not path.is_file():
        return None

    def reject_constant(value: str) -> object:
        raise ValueError("non-finite JSON constant: {0}".format(value))

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON number")
        return parsed

    def unique_object(pairs: List[Tuple[str, object]]) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: {0}".format(key))
            result[key] = value
        return result

    try:
        report = json.loads(
            path.read_text(encoding="utf-8"),
            parse_float=parse_finite_float,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (json.JSONDecodeError, OSError, UnicodeError, ValueError):
        return None
    return report if isinstance(report, dict) else None


def load_validated_archival_report(
    workspace: CandidateWorkspace,
) -> Optional[Dict[str, Any]]:
    """Return an owned, successful, validated Gaia archival report only."""
    report = _load_archival_report(workspace)
    if report is None or report.get("candidate_id") != workspace.candidate_id:
        return None
    gaia = report.get("gaia_astrometry")
    if (
        not isinstance(gaia, dict)
        or gaia.get("validated") is not True
        or gaia.get("query_status") != "ok"
    ):
        return None
    return report


def load_validated_archival_gaia_sources(
    workspace: CandidateWorkspace,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Load a validated target source and neighbors from an archival report.

    Reports created by current EXONYM versions name the Gaia target source.
    Older reports are accepted only as a sensitivity fallback, using the
    closest source inside their recorded target-match radius.
    """
    metadata: Dict[str, Any] = {
        "availability": "unavailable",
        "source": str(ARCHIVAL_REPORT_RELATIVE_PATH).replace("\\", "/"),
    }
    report = load_validated_archival_report(workspace)
    if report is None:
        return None, [], metadata
    gaia = report["gaia_astrometry"]
    sources = gaia.get("sources")
    if not isinstance(sources, list):
        return None, [], metadata

    target_index: Optional[int] = None
    target_source_id = gaia.get("target_source_id")
    if target_source_id is not None:
        target_source_id = str(target_source_id)
        for index, source in enumerate(sources):
            if isinstance(source, dict) and str(source.get("source_id")) == target_source_id:
                target_index = index
                metadata["target_selection"] = "reported-target-source-id"
                break

    if target_index is None:
        match_max_arcsec = _finite_float(gaia.get("target_match_max_arcsec"))
        if match_max_arcsec is None or match_max_arcsec <= 0.0:
            match_max_arcsec = ArchivalVettingService.TARGET_PRESENCE_ARCSEC
        candidates = [
            (index, _finite_float(source.get("separation_arcsec")))
            for index, source in enumerate(sources)
            if isinstance(source, dict)
        ]
        candidates = [
            (index, separation_arcsec)
            for index, separation_arcsec in candidates
            if separation_arcsec is not None and separation_arcsec <= match_max_arcsec
        ]
        if not candidates:
            return None, [], metadata
        target_index = min(candidates, key=lambda item: item[1])[0]
        metadata["target_selection"] = "closest-source-within-validated-radius"

    target = sources[target_index]
    if not isinstance(target, dict):
        return None, [], metadata
    metadata["availability"] = "available"
    metadata["target_match_max_arcsec"] = _finite_float(
        gaia.get("target_match_max_arcsec")
    ) or ArchivalVettingService.TARGET_PRESENCE_ARCSEC
    search_radius_arcsec = _finite_float(gaia.get("search_radius_arcsec"))
    if search_radius_arcsec is not None:
        metadata["archive_search_radius_arcsec"] = search_radius_arcsec
    return dict(target), [
        dict(source)
        for index, source in enumerate(sources)
        if index != target_index and isinstance(source, dict)
    ], metadata
