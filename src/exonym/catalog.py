"""Catalog identifier parsing, provenance sidecars, and feasibility helpers.

Identifier parsing and provenance writing are deterministic, target-neutral
utilities used by ingestion. The numerical helpers estimate radial-velocity,
astrometric, atmospheric-scale-height, and transmission observables from their
explicit physical inputs; they are useful for planning, not fitted evidence.

Astrophysical rationale:
    The Doppler and astrometric helpers are unit-explicit two-body planning
    scalings. Input and return units are encoded in their public function names
    and docstrings so callers cannot silently mix catalog conventions.

Scientific boundary:
    These estimates do not consume candidate observations, uncertainty models,
    or a scene model. They must not be interpreted as a detection or validation
    result.

Units, provenance gap, and failure boundary
-------------------------------------------
The helper names encode all inputs and returns: Earth/solar masses and radii,
AU, pc, K, km, m s^-1, microarcsec, and ppm.  IAU 2015 nominal values and
CODATA 2018 constants are verified in `literature/README.md` as ADS
``2016AJ....152...41P`` / DOI ``10.3847/0004-6256/152/2/41`` and ADS
``2021RvMP...93b5010T`` / DOI ``10.1103/RevModPhys.93.025010``.  The current
repository does not retain a primary source for every convenience planning
scaling in this module; they therefore remain fail-closed positive-input
calculators, not calibrated characterisation relations.  A nonpositive physical
input raises, and no output can set ``claim_eligible``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .constants import (
    BOLTZMANN_CONSTANT_J_K,
    EARTH_MASS_ONE_JULIAN_YEAR_RV_SEMI_AMPLITUDE_M_PER_S,
    EARTH_TO_SOLAR_MASS_PARAMETER_RATIO,
    JULIAN_YEAR_DAYS,
    NOMINAL_EARTH_EQUATORIAL_RADIUS_M,
    NOMINAL_EARTH_MASS_PARAMETER_M3_S2,
    NOMINAL_SOLAR_RADIUS_KM,
    PROTON_MASS_KG,
)

MISSIONS = ("tess", "kepler", "k2", "plato", "cheops")
INGEST_FETCHER = "exonym-ingest/1.2.0"

_PATTERNS = (
    ("toi", "tess", re.compile(r"^(?:TOI)[\s._-]*(\d{1,7}(?:\.\d{1,2})?)$", re.IGNORECASE)),
    ("tic", "tess", re.compile(r"^(?:TIC)[\s._:-]*(\d{5,12})$", re.IGNORECASE)),
    ("koi", "kepler", re.compile(r"^(?:K)[\s_-]*(\d{1,7})\.(\d{1,2})$", re.IGNORECASE)),
    ("epic", "k2", re.compile(r"^(?:EPIC)[\s._-]*(\d{6,10})$", re.IGNORECASE)),
    ("pic", "plato", re.compile(r"^(?:PIC)[\s._-]*(\d{4,10})$", re.IGNORECASE)),
    ("cheops", "cheops", re.compile(r"^(?:CHEOPS)[\s._-]*([A-Za-z0-9._-]+)$", re.IGNORECASE)),
)


class IdentifierError(ValueError):
    """Raised when a catalog identifier cannot be parsed."""


def parse_identifier(identifier: str) -> Dict[str, str]:
    """Parse a mission-aware catalog identifier.

    Returns a dict with ``kind`` (toi/tic/koi/epic/pic/cheops), ``mission``,
    and ``value``. Raises :class:`IdentifierError` for unrecognized input.
    """
    text = str(identifier).strip()
    if not text:
        raise IdentifierError("empty identifier")
    for kind, mission, pattern in _PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        value = match.group(1)
        if kind == "koi":
            value = "{0}.{1}".format(match.group(1), match.group(2))
        return {"kind": kind, "mission": mission, "value": value}
    raise IdentifierError("unrecognized catalog identifier: {0}".format(identifier))


def mission_for_identifier(identifier: str) -> str:
    """Return the mission for an identifier, raising on unknown input."""
    return parse_identifier(identifier)["mission"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_provenance(
    product_path: Path,
    source_uri: str,
    fetched_by: str = INGEST_FETCHER,
    download_timestamp_utc: Optional[str] = None,
    sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a provenance sidecar record for a downloaded product (pure).

    Args:
        product_path: Path to the product file on disk.
        source_uri: Canonical HTTPS or MAST URI from which the file was retrieved.
        fetched_by: Retrieval-agent label recorded in the sidecar.
        download_timestamp_utc: ISO-8601 UTC timestamp; defaults to now.
        sha256: Pre-computed hex SHA-256 digest.  When supplied the file is not
            re-read; when ``None`` the digest is computed from *product_path*.

    Returns:
        Dictionary suitable for JSON serialisation as a provenance sidecar.
    """
    if download_timestamp_utc is None:
        download_timestamp_utc = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
    digest = sha256 if sha256 is not None else _sha256(Path(product_path))
    return {
        "source_uri": source_uri,
        "download_timestamp_utc": download_timestamp_utc,
        "sha256": digest,
        "fetched_by": fetched_by,
    }


def write_provenance_sidecar(
    product_path: Path,
    source_uri: str,
    fetched_by: str = INGEST_FETCHER,
    download_timestamp_utc: Optional[str] = None,
    sha256: Optional[str] = None,
) -> Path:
    """Write ``<product>.provenance.json`` next to the product and return it.

    The sidecar naming matches the acquisition gate convention in
    ``exonym.gatekeeper`` (``<stem>.provenance.json``).

    Args:
        product_path: Path to the ingested product file.
        source_uri: Canonical HTTPS or MAST URI for the provenance record.
        fetched_by: Retrieval-agent label.
        download_timestamp_utc: ISO-8601 UTC timestamp; defaults to now.
        sha256: Pre-computed hex SHA-256 digest.  When supplied the file is not
            re-read; when ``None`` the digest is computed from *product_path*.

    Returns:
        Path to the written provenance sidecar.
    """
    product_path = Path(product_path)
    sidecar = product_path.with_name(product_path.stem + ".provenance.json")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + sidecar.name + ".", suffix=".tmp", dir=str(sidecar.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    make_provenance(
                        product_path, source_uri, fetched_by, download_timestamp_utc,
                        sha256=sha256,
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        temporary.replace(sidecar)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sidecar


def calculate_radial_velocity_semi_amplitude(
    m_planet_earth: float,
    m_star_solar: float,
    period_days: float,
    inclination_deg: float = 90.0,
    eccentricity: float = 0.0,
) -> float:
    """Return host star Doppler reflex velocity semi-amplitude K in m/s."""
    if m_planet_earth <= 0 or m_star_solar <= 0 or period_days <= 0:
        raise ValueError("physical parameters must be positive")
    if not (0.0 <= eccentricity < 1.0):
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    sin_i = math.sin(math.radians(inclination_deg))
    return (
        EARTH_MASS_ONE_JULIAN_YEAR_RV_SEMI_AMPLITUDE_M_PER_S
        * (m_planet_earth * sin_i)
        * (m_star_solar ** (-2.0 / 3.0))
        * ((period_days / JULIAN_YEAR_DAYS) ** (-1.0 / 3.0))
        / math.sqrt(1.0 - eccentricity**2)
    )


def calculate_astrometric_wobble_microarcsec(
    m_planet_earth: float,
    m_star_solar: float,
    semi_major_axis_au: float,
    distance_pc: float,
) -> float:
    """Return astrometric signature alpha_* in microarcseconds (uas)."""
    if (
        m_planet_earth <= 0
        or m_star_solar <= 0
        or semi_major_axis_au <= 0
        or distance_pc <= 0
    ):
        raise ValueError("physical parameters must be positive")
    m_earth_solar = EARTH_TO_SOLAR_MASS_PARAMETER_RATIO
    m_p_solar = m_planet_earth * m_earth_solar
    alpha_arcsec = (m_p_solar / m_star_solar) * semi_major_axis_au / distance_pc
    return alpha_arcsec * 1.0e6


def calculate_atmospheric_scale_height_km(
    t_eq_kelvin: float,
    m_planet_earth: float,
    r_planet_earth: float,
    mean_molecular_weight: float = 2.3,
) -> float:
    """Return planetary atmospheric pressure scale height H in kilometers (km)."""
    if (
        t_eq_kelvin <= 0
        or m_planet_earth <= 0
        or r_planet_earth <= 0
        or mean_molecular_weight <= 0
    ):
        raise ValueError("physical parameters must be positive")
    g_p = (
        NOMINAL_EARTH_MASS_PARAMETER_M3_S2 * m_planet_earth
    ) / ((r_planet_earth * NOMINAL_EARTH_EQUATORIAL_RADIUS_M) ** 2)
    mu_kg = mean_molecular_weight * PROTON_MASS_KG
    h_meters = (BOLTZMANN_CONSTANT_J_K * t_eq_kelvin) / (mu_kg * g_p)
    return h_meters / 1000.0


def calculate_transmission_signal_ppm(
    r_star_solar: float,
    r_planet_earth: float,
    scale_height_km: float,
    n_scale_heights: float = 5.0,
) -> float:
    """Return transmission spectrum modulation depth in ppm."""
    if (
        r_star_solar <= 0
        or r_planet_earth <= 0
        or scale_height_km <= 0
        or n_scale_heights <= 0
    ):
        raise ValueError("physical parameters must be positive")
    r_star_km = r_star_solar * NOMINAL_SOLAR_RADIUS_KM
    r_planet_km = r_planet_earth * NOMINAL_EARTH_EQUATORIAL_RADIUS_M / 1.0e3
    delta_trans = (2.0 * r_planet_km * n_scale_heights * scale_height_km) / (r_star_km**2)
    return delta_trans * 1.0e6
