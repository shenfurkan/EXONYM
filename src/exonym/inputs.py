"""Target-neutral input loading for scientific analysis modules.

Every loader probes candidate workspace files and metadata only. Ephemerides,
stellar parameters, photometry, light curves, and target pixel files are read
dynamically; generic demonstration values are used only when no candidate data
exists and are always labelled ``synthetic-demo``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .workspace import CandidateWorkspace, validate_signal_suffix

# Generic demonstration ephemeris used only when no candidate ephemeris source
# exists. These are placeholder values, never target data.
DEMO_PERIOD_DAYS = 3.5
DEMO_EPOCH_BTJD = 2.0
DEMO_DURATION_DAYS = 0.12
DEMO_DEPTH_PPM = 1200.0

# Generic demonstration stellar parameters (solar reference values).
DEMO_TEFF_K = 5772.0
DEMO_LOGG_CGS = 4.438
DEMO_FEH = 0.0
DEMO_MASS_SOLAR = 1.0
DEMO_RADIUS_SOLAR = 1.0
DEMO_PARALLAX_MAS = 10.0
BTJD_REFERENCE_BJD = 2457000.0
BTJD_TIME_SYSTEM = "BTJD_TDB"
# This is a candidate-selection threshold, not a calibrated false-alarm rate.
MINIMUM_BLS_CANDIDATE_SNR = 7.1

EPHEMERIS_CONFIG_NAMES = (
    "transit_config.json",
    "ephemeris.json",
    "candidate_ephemeris.json",
)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_float=_parse_finite_float,
            parse_constant=_reject_nonfinite_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_finite_float(value: str) -> float:
    """Parse a JSON number without permitting infinities through overflow."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_nonfinite_json_constant(value: str) -> object:
    """Reject non-standard JSON numeric constants."""
    raise ValueError("non-finite JSON constant: {0}".format(value))


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    """Reject ambiguous JSON objects instead of silently keeping the final key."""
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {0}".format(key))
        result[key] = value
    return result


def _first_number(payload: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = float(value)
            if np.isfinite(parsed):
                return parsed
    return None


def _has_complete_candidate_ephemeris(result: Dict[str, Any], source_prefix: str) -> bool:
    field_sources = result["field_sources"]
    return all(
        str(field_sources.get(field, "")).startswith(source_prefix)
        for field in ("period_days", "epoch_btjd", "duration_days", "depth_ppm")
    )


def _epoch_btjd_from_transit_config(transit: Dict[str, Any]) -> Optional[float]:
    declared_system = str(
        transit.get("epoch_time_system", transit.get("time_system", ""))
    ).strip().upper()
    if declared_system and declared_system not in ("BTJD", BTJD_TIME_SYSTEM):
        return None
    explicit = _first_number(transit, ("epoch_btjd", "t0_btjd"))
    if explicit is not None:
        return explicit
    if declared_system not in ("BTJD", BTJD_TIME_SYSTEM):
        return None
    return _first_number(transit, ("t0", "epoch"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_candidate_photometry_input(
    workspace: CandidateWorkspace, candidate_root: Path, record: Dict[str, Any]
) -> bool:
    """Confirm that a manifest input is an unchanged raw, provenanced FITS product."""
    if not isinstance(record.get("path"), str):
        return False
    try:
        product_path = (candidate_root / record["path"]).resolve()
        relative = product_path.relative_to(candidate_root)
    except (OSError, ValueError):
        return False
    if (
        len(relative.parts) < 3
        or relative.parts[:2] != ("data", "raw")
        or not product_path.name.lower().endswith((".fits", ".fits.fz", ".fz"))
        or not product_path.is_file()
        or record.get("sha256") != _sha256(product_path)
    ):
        return False
    sidecar_path = product_path.with_name(product_path.stem + ".provenance.json")
    try:
        sidecar_relative = sidecar_path.resolve().relative_to(candidate_root).as_posix()
    except (OSError, ValueError):
        return False
    if (
        not sidecar_path.is_file()
        or record.get("provenance_path") != sidecar_relative
        or record.get("provenance_sha256") != _sha256(sidecar_path)
    ):
        return False
    from .gatekeeper import has_valid_raw_product_provenance

    return has_valid_raw_product_provenance(workspace, product_path)


def is_manifest_bound_bls_result(
    workspace: CandidateWorkspace,
    result_path: Path,
    payload: Dict[str, Any],
    signal: Optional[str],
) -> bool:
    suffix = signal or ""
    manifest_path = workspace.path / "outputs" / ("bls_search_manifest" + suffix + ".json")
    try:
        result_relative = result_path.resolve().relative_to(workspace.path.resolve()).as_posix()
    except (OSError, ValueError):
        return False
    if not result_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    if manifest is None:
        return False
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        return False
    period = _first_number(payload, ("best_period",))
    epoch = _first_number(payload, ("best_epoch",))
    duration_hours = _first_number(payload, ("best_duration_hours",))
    depth_ppm = _first_number(payload, ("best_depth_ppm",))
    event_count = payload.get("n_distinct_transit_events")
    if (
        manifest.get("schema") != "exonym-bls-search-manifest-1"
        or manifest.get("candidate_id") != workspace.candidate_id
        or manifest.get("source") != "candidate-data"
        or manifest.get("detection_status") != "detected"
        or manifest.get("result_path") != result_relative
        or manifest.get("result_sha256") != _sha256(result_path)
        or configuration.get("engine") != "bls"
        or configuration.get("signal") != signal
        or configuration.get("time_system") != BTJD_TIME_SYSTEM
        or configuration.get("detection_threshold_snr") != MINIMUM_BLS_CANDIDATE_SNR
        or payload.get("detection_status") != "detected"
        or payload.get("time_system") != BTJD_TIME_SYSTEM
        or payload.get("detection_threshold_snr") != MINIMUM_BLS_CANDIDATE_SNR
        or (_first_number(payload, ("snr",)) or 0.0) < MINIMUM_BLS_CANDIDATE_SNR
        or period is None
        or period <= 0.0
        or epoch is None
        or duration_hours is None
        or duration_hours <= 0.0
        or duration_hours / 24.0 >= period
        or depth_ppm is None
        or depth_ppm <= 0.0
        or isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 2
    ):
        return False
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        return False
    candidate_root = workspace.path.resolve()
    for record in inputs:
        if not isinstance(record, dict) or not _is_candidate_photometry_input(
            workspace, candidate_root, record
        ):
            return False
    return True


def is_bls_bound_transit_config(
    workspace: CandidateWorkspace,
    config_path: Path,
    payload: Dict[str, Any],
    signal: Optional[str] = None,
) -> bool:
    """Confirm a BLS-derived transit config still names the current BLS evidence."""
    if payload.get("source") != "candidate-data-bls":
        return False
    try:
        config_path.resolve().relative_to(workspace.path.resolve())
    except (OSError, ValueError):
        return False
    provenance = payload.get("bls_provenance")
    transit = payload.get("transit")
    if not isinstance(provenance, dict) or not isinstance(transit, dict):
        return False
    result_record = provenance.get("result")
    manifest_record = provenance.get("manifest")
    if not isinstance(result_record, dict) or not isinstance(manifest_record, dict):
        return False
    suffix = signal or ""
    expected_result = workspace.path / "outputs" / ("bls_search_results" + suffix + ".json")
    expected_manifest = workspace.path / "outputs" / ("bls_search_manifest" + suffix + ".json")
    if (
        not expected_result.is_file()
        or not expected_manifest.is_file()
        or result_record.get("path")
        != expected_result.relative_to(workspace.path).as_posix()
        or result_record.get("sha256") != _sha256(expected_result)
        or manifest_record.get("path")
        != expected_manifest.relative_to(workspace.path).as_posix()
        or manifest_record.get("sha256") != _sha256(expected_manifest)
    ):
        return False
    result = _read_json(expected_result)
    if result is None or not is_manifest_bound_bls_result(workspace, expected_result, result, signal):
        return False
    values = (
        (_first_number(transit, ("period_days", "period")), _first_number(result, ("best_period",))),
        (_epoch_btjd_from_transit_config(transit), _first_number(result, ("best_epoch",))),
        (
            _first_number(transit, ("duration_days",)),
            (
                _first_number(result, ("best_duration_hours",)) / 24.0
                if _first_number(result, ("best_duration_hours",)) is not None
                else None
            ),
        ),
        (_first_number(transit, ("depth_ppm", "depth")), _first_number(result, ("best_depth_ppm",))),
    )
    return all(
        configured is not None
        and measured is not None
        and math.isclose(configured, measured, rel_tol=1e-12, abs_tol=1e-12)
        for configured, measured in values
    )


def load_transit_ephemeris(
    workspace: CandidateWorkspace, signal: Optional[str] = None
) -> Dict[str, Any]:
    """Return the best-known transit ephemeris for a candidate workspace.

    When ``signal`` is given (e.g. ``.01``) the per-signal config
    ``config/signals/transit_config<signal>.json`` takes precedence. Otherwise
    probes ``config/`` JSON files (``transit`` or top-level keys) first, then
    ``outputs/bls_search_results.json``. Falls back to a generic demonstration
    ephemeris labelled ``synthetic-demo`` when nothing readable exists.
    """
    signal = validate_signal_suffix(signal)
    result: Dict[str, Any] = {
        "period_days": DEMO_PERIOD_DAYS,
        "epoch_btjd": DEMO_EPOCH_BTJD,
        "duration_days": DEMO_DURATION_DAYS,
        "depth_ppm": DEMO_DEPTH_PPM,
        "time_system": "synthetic-demo",
        "source": "synthetic-demo",
        "field_sources": {
            "period_days": "synthetic-demo",
            "epoch_btjd": "synthetic-demo",
            "duration_days": "synthetic-demo",
            "depth_ppm": "synthetic-demo",
        },
    }

    if signal is not None:
        config_path = (
            workspace.path / "config" / "signals" / ("transit_config" + signal + ".json")
        )
        payload = _read_json(config_path)
        if payload is not None and payload.get("source") == "candidate-data-bls" and not is_bls_bound_transit_config(
            workspace, config_path, payload, signal
        ):
            payload = None
        if payload is not None:
            transit = payload.get("transit")
            if not isinstance(transit, dict):
                transit = payload
            source_prefix = (
                "candidate-data-bls"
                if payload.get("source") == "candidate-data-bls"
                else "candidate-config-signal"
            )
            period_value = _first_number(transit, ("period", "period_days", "p"))
            epoch_value = _epoch_btjd_from_transit_config(transit)
            duration_hours_value = _first_number(
                transit, ("duration_hrs", "duration_hours", "duration_h")
            )
            duration_days_value = _first_number(transit, ("duration_days",))
            depth_value = _first_number(transit, ("depth_ppm", "depth"))
            found = False
            if period_value is not None and period_value > 0:
                result["period_days"] = period_value
                result["field_sources"]["period_days"] = source_prefix
                found = True
            if epoch_value is not None:
                result["epoch_btjd"] = epoch_value
                result["field_sources"]["epoch_btjd"] = source_prefix
                result["time_system"] = BTJD_TIME_SYSTEM
                found = True
            if duration_hours_value is not None and duration_hours_value > 0:
                result["duration_days"] = duration_hours_value / 24.0
                result["field_sources"]["duration_days"] = source_prefix
                found = True
            if duration_days_value is not None and duration_days_value > 0:
                result["duration_days"] = duration_days_value
                result["field_sources"]["duration_days"] = source_prefix
                found = True
            if depth_value is not None and depth_value >= 0:
                result["depth_ppm"] = depth_value
                result["field_sources"]["depth_ppm"] = source_prefix
                found = True
            if found:
                result["source"] = (
                    source_prefix
                    if _has_complete_candidate_ephemeris(result, source_prefix)
                    else "partial-" + source_prefix
                )
                return result

    for config_name in EPHEMERIS_CONFIG_NAMES:
        config_path = workspace.path / "config" / config_name
        payload = _read_json(config_path)
        if payload is not None and payload.get("source") == "candidate-data-bls" and not is_bls_bound_transit_config(
            workspace, config_path, payload, None
        ):
            payload = None
        if payload is None:
            continue
        transit = payload.get("transit")
        if not isinstance(transit, dict):
            transit = payload
        source_prefix = (
            "candidate-data-bls"
            if payload.get("source") == "candidate-data-bls"
            else "candidate-config"
        )
        period_value = _first_number(transit, ("period", "period_days", "p"))
        epoch_value = _epoch_btjd_from_transit_config(transit)
        duration_hours_value = _first_number(
            transit, ("duration_hrs", "duration_hours", "duration_h")
        )
        duration_days_value = _first_number(transit, ("duration_days",))
        depth_value = _first_number(transit, ("depth_ppm", "depth"))
        found = False
        if period_value is not None and period_value > 0:
            result["period_days"] = period_value
            result["field_sources"]["period_days"] = source_prefix
            found = True
        if epoch_value is not None:
            result["epoch_btjd"] = epoch_value
            result["field_sources"]["epoch_btjd"] = source_prefix
            result["time_system"] = BTJD_TIME_SYSTEM
            found = True
        if duration_hours_value is not None and duration_hours_value > 0:
            result["duration_days"] = duration_hours_value / 24.0
            result["field_sources"]["duration_days"] = source_prefix
            found = True
        if duration_days_value is not None and duration_days_value > 0:
            result["duration_days"] = duration_days_value
            result["field_sources"]["duration_days"] = source_prefix
            found = True
        if depth_value is not None and depth_value >= 0:
            result["depth_ppm"] = depth_value
            result["field_sources"]["depth_ppm"] = source_prefix
            found = True
        if found:
            result["source"] = (
                source_prefix
                if _has_complete_candidate_ephemeris(result, source_prefix)
                else "partial-" + source_prefix
            )
            break

    if result["source"] == "synthetic-demo":
        suffix = signal if signal is not None else ""
        bls_path = workspace.path / "outputs" / ("bls_search_results" + suffix + ".json")
        payload = _read_json(bls_path)
        if (
            payload is not None
            and payload.get("source") == "candidate-data"
            and payload.get("detection_status") == "detected"
            and payload.get("time_system") == BTJD_TIME_SYSTEM
            and is_manifest_bound_bls_result(workspace, bls_path, payload, signal)
        ):
            period_value = _first_number(payload, ("best_period",))
            epoch_value = _first_number(payload, ("best_epoch",))
            duration_hours_value = _first_number(payload, ("best_duration_hours",))
            depth_value = _first_number(payload, ("best_depth_ppm",))
            event_count = _first_number(payload, ("n_distinct_transit_events",))
            if period_value is not None and period_value > 0:
                result["period_days"] = period_value
                result["field_sources"]["period_days"] = "bls-search"
            if epoch_value is not None:
                result["epoch_btjd"] = epoch_value
                result["field_sources"]["epoch_btjd"] = "bls-search"
                result["time_system"] = BTJD_TIME_SYSTEM
            if duration_hours_value is not None and duration_hours_value > 0:
                result["duration_days"] = duration_hours_value / 24.0
                result["field_sources"]["duration_days"] = "bls-search"
            if depth_value is not None and depth_value >= 0:
                result["depth_ppm"] = depth_value
                result["field_sources"]["depth_ppm"] = "bls-search"
            if (
                period_value is not None
                and period_value > 0
                and epoch_value is not None
                and duration_hours_value is not None
                and duration_hours_value > 0
                and depth_value is not None
                and depth_value > 0
                and event_count is not None
                and event_count >= 2
            ):
                result["source"] = "bls-search"

    if result["period_days"] <= 0 or result["duration_days"] <= 0:
        result["period_days"] = DEMO_PERIOD_DAYS
        result["duration_days"] = DEMO_DURATION_DAYS
        result["source"] = "synthetic-demo"
    return result


def load_stellar_parameters(workspace: CandidateWorkspace) -> Dict[str, Any]:
    """Return stellar parameters read from ``data/external/stellar_params.json``.

    Falls back to generic solar demonstration values labelled ``synthetic-demo``
    when no file is present.  When the file exists but only partially populates
    the required physics fields (``teff_k``, ``logg_cgs``, ``feh``,
    ``mass_solar``, ``radius_solar``), the source is set to
    ``"partial-candidate-data"`` and any missing physics field retains its
    generic demonstration value.  Callers that require fully candidate-owned
    stellar physics should reject ``source != "candidate-data"``.

    ``ra_deg``, ``dec_deg``, and ``parallax_mas`` are optional positional
    fields; their presence or absence does not affect the source label.  The
    optional ``mass_solar_err`` and ``radius_solar_err`` fields retain
    candidate-supplied symmetric one-sigma uncertainties for inference modules
    that must propagate stellar-density uncertainty.
    """
    _PHYSICS_FIELDS = ("teff_k", "logg_cgs", "feh", "mass_solar", "radius_solar")
    result: Dict[str, Any] = {
        "teff_k": DEMO_TEFF_K,
        "logg_cgs": DEMO_LOGG_CGS,
        "feh": DEMO_FEH,
        "mass_solar": DEMO_MASS_SOLAR,
        "radius_solar": DEMO_RADIUS_SOLAR,
        "parallax_mas": DEMO_PARALLAX_MAS,
        "source": "synthetic-demo",
    }
    params_path = workspace.path / "data" / "external" / "stellar_params.json"
    payload = _read_json(params_path)
    if payload is None:
        return result
    values = {
        "ra_deg": _first_number(payload, ("ra_deg", "ra", "right_ascension")),
        "dec_deg": _first_number(payload, ("dec_deg", "dec", "declination")),
        "teff_k": _first_number(payload, ("teff_k", "teff", "temperature_k")),
        "logg_cgs": _first_number(payload, ("logg_cgs", "logg", "log_g")),
        "feh": _first_number(payload, ("feh", "metallicity")),
        "mass_solar": _first_number(payload, ("mass_solar", "mass_msun", "mass")),
        "mass_solar_err": _first_number(
            payload, ("mass_solar_err", "mass_msun_err", "mass_err", "mass_error")
        ),
        "radius_solar": _first_number(
            payload, ("radius_solar", "radius_rsun", "radius")
        ),
        "radius_solar_err": _first_number(
            payload, ("radius_solar_err", "radius_rsun_err", "radius_err", "radius_error")
        ),
        "parallax_mas": _first_number(
            payload, ("parallax_mas", "parallax", "plx")
        ),
        "parallax_mas_err": _first_number(
            payload, ("parallax_mas_err", "parallax_err", "parallax_error", "plx_error")
        ),
    }
    for name, value in values.items():
        if value is not None:
            result[name] = value
    physics_present = sum(1 for f in _PHYSICS_FIELDS if values.get(f) is not None)
    if physics_present == len(_PHYSICS_FIELDS):
        result["source"] = "candidate-data"
    elif physics_present > 0:
        result["source"] = "partial-candidate-data"
    # else: no physics field found -> source stays "synthetic-demo"
    return result


def load_photometry(workspace: CandidateWorkspace) -> Optional[Dict[str, Any]]:
    """Return broadband photometry from ``data/external/stellar_photometry.json``.

    Expected generic shape: ``{"2MASS": {"J": {"mag":.., "error":..}, ...},
    "AllWISE": {"W1": ...}, "gaia": {"parallax_mas": .., "g_mag": ..}}``.
    Returns None when no readable photometry file exists.
    """
    path = workspace.path / "data" / "external" / "stellar_photometry.json"
    return _read_json(path)


def _mad_flux_error(flux: np.ndarray) -> float:
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median)))
    if not np.isfinite(mad) or mad <= 0:
        return 1.0
    return float(1.4826 * mad)


def _median_bin(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    sector_values: np.ndarray,
    n_bins: int = 4000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median-bin a time-sorted table down to at most n_bins rows."""
    if time.size <= n_bins:
        return time, flux, flux_err, sector_values
    order = np.argsort(time)
    time_sorted = time[order]
    flux_sorted = flux[order]
    err_sorted = flux_err[order]
    sector_sorted = sector_values[order]
    edges = np.linspace(0, time_sorted.size, n_bins + 1).astype(int)
    bin_times = np.empty(n_bins, dtype=float)
    bin_flux = np.empty(n_bins, dtype=float)
    bin_err = np.empty(n_bins, dtype=float)
    bin_sector = np.empty(n_bins, dtype=int)
    for index in range(n_bins):
        start, stop = edges[index], edges[index + 1]
        if stop <= start:
            bin_times[index] = np.nan
            bin_flux[index] = np.nan
            bin_err[index] = np.nan
            bin_sector[index] = 0
            continue
        bin_times[index] = float(np.mean(time_sorted[start:stop]))
        bin_flux[index] = float(np.median(flux_sorted[start:stop]))
        bin_err[index] = float(np.median(err_sorted[start:stop]))
        bin_sector[index] = int(
            np.median(sector_sorted[start:stop].astype(float))
        )
    valid = np.isfinite(bin_times) & np.isfinite(bin_flux)
    return (
        bin_times[valid],
        bin_flux[valid],
        bin_err[valid],
        bin_sector[valid],
    )


def _sector_from_canonical_filename(path: Path) -> Optional[int]:
    """Return a TESS sector from a canonical product filename, if present."""
    match = re.search(r"(?:^|[_-])s(?P<sector>\d{1,4})(?=[_.-])", path.name, re.IGNORECASE)
    if match is None:
        return None
    sector_value = int(match.group("sector"))
    return sector_value if sector_value > 0 else None


def _fits_time_header(path: Path, extension_index: int = 1) -> Dict[str, Any]:
    """Return the primary and time-extension FITS headers for time normalization."""
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        header = dict(hdul[0].header)
        if len(hdul) > extension_index:
            header.update(dict(hdul[extension_index].header))
    return header


def _header_number(header: Dict[str, Any], name: str) -> Optional[float]:
    value = header.get(name)
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _time_values_to_btjd_tdb(
    values: np.ndarray,
    header: Dict[str, Any],
    declared_format: Optional[str] = None,
    declared_scale: Optional[str] = None,
) -> np.ndarray:
    """Normalize a declared TDB FITS time vector to BTJD (BJD_TDB - 2457000)."""
    time = np.asarray(values, dtype=float)
    finite = time[np.isfinite(time)]
    if finite.size == 0:
        raise ValueError("time vector contains no finite values")

    header_scale = str(header.get("TIMESYS", "")).strip().upper()
    normalized_scale = str(declared_scale or "").strip().upper()
    if header_scale and header_scale not in ("TDB", "BJD_TDB"):
        raise ValueError("FITS TIMESYS must be TDB for BTJD ephemerides")
    if normalized_scale and normalized_scale != "TDB":
        raise ValueError("light-curve time scale must be TDB for BTJD ephemerides")
    if not header_scale and normalized_scale != "TDB":
        raise ValueError("time scale is not declared as TDB")

    time_unit = str(header.get("TIMEUNIT", "")).strip().lower()
    if time_unit and time_unit not in ("d", "day", "days"):
        raise ValueError("FITS TIMEUNIT must be days for BTJD ephemerides")

    median_time = float(np.median(finite))
    time_format = str(declared_format or "").strip().lower()
    if not time_unit and time_format not in ("btjd", "mjd", "jd"):
        raise ValueError("time unit is not declared as days")
    bjd_reference = _header_number(header, "BJDREFI")
    bjd_fraction = _header_number(header, "BJDREFF")
    if bjd_reference is not None:
        bjd_reference += bjd_fraction or 0.0
    mjd_reference = _header_number(header, "MJDREFI")
    mjd_fraction = _header_number(header, "MJDREFF")
    if mjd_reference is not None:
        mjd_reference += mjd_fraction or 0.0

    if time_format == "mjd" or (mjd_reference is not None and bjd_reference is None):
        return time + (mjd_reference or 0.0) + 2400000.5 - BTJD_REFERENCE_BJD
    if median_time > 2000000.0:
        return time - BTJD_REFERENCE_BJD
    if bjd_reference is not None:
        return time + bjd_reference - BTJD_REFERENCE_BJD
    if time_format == "btjd":
        return time
    raise ValueError("time origin is not declared as BTJD or BJD_TDB")


def load_light_curve_table(
    workspace: CandidateWorkspace,
    max_points: Optional[int] = 4000,
    sectors: Optional[Sequence[int]] = None,
    raw_only: bool = False,
    require_raw_provenance: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return a light curve table from candidate FITS products, or None.

    The returned dict has ``time``, ``flux`` (normalized), ``flux_err``,
    ``sector`` (int array), and ``input_files`` (the accepted products).
    Products are read from ``data/processed/`` first, then ``data/raw/``.
    When ``sectors`` is supplied, only products whose resolved TESS sector is
    in that sequence are returned. Multiple products in one selected sector
    are deduplicated by sorted filename, with the first product retained.
    ``max_points`` is a per-product cap; accepted sectors are not globally
    re-binned after concatenation because that would make effective cadence
    depend on the number of observed sectors. Returns None when no readable
    light curve exists after filtering.
    """
    if require_raw_provenance:
        raw_only = True
        from .gatekeeper import has_valid_raw_product_provenance

    roots = (
        (workspace.path / "data" / "raw",)
        if raw_only
        else (
            workspace.path / "data" / "processed",
            workspace.path / "data" / "raw",
        )
    )
    fits_files: List[Path] = []
    processed_names: set = set()
    for root in roots:
        if not root.is_dir():
            continue
        hits: List[Path] = []
        for suffix in (".fits", ".fits.fz", ".fz"):
            hits.extend(root.rglob("*" + suffix))
        if root.name == "processed":
            processed_names = {h.name for h in hits}
        else:
            hits = [h for h in hits if h.name not in processed_names]
        fits_files.extend(hits)
    fits_files = [path for path in fits_files if "tp" not in path.stem.lower()]
    fits_files.sort()
    if not fits_files:
        return None

    try:
        import lightkurve as lk
    except ImportError:  # pragma: no cover - optional dependency
        return None

    import warnings as _warnings

    requested_sectors = set(sectors) if sectors is not None else None
    tables: List[Dict[str, Any]] = []
    seen_sectors: set = set()
    for path in fits_files:
        try:
            if require_raw_provenance and not has_valid_raw_product_provenance(workspace, path):
                _warnings.warn(
                    "skipped {0}: raw provenance sidecar is missing, invalid, or stale".format(
                        path.name
                    ),
                    stacklevel=2,
                )
                continue
            input_sha256 = _sha256(path)
            _lc = lk.read(path)
            # Apply the TESS quality bitmask so that momentum-dump, scattered-
            # light, and other flagged cadences are excluded before any
            # analysis. A malformed quality column makes the product
            # scientifically unusable; it must never be treated as clean.
            if hasattr(_lc, "quality"):
                try:
                    quality = np.asarray(getattr(_lc.quality, "value", _lc.quality))
                    cadence_count = np.asarray(_lc.time.value).size
                    if quality.ndim != 1 or quality.size != cadence_count:
                        raise ValueError("quality cadence count does not match the light curve")
                    _lc = _lc[quality == 0]
                except (AttributeError, TypeError, ValueError, IndexError) as exc:
                    _warnings.warn(
                        "skipped {0}: unusable quality column: {1!r}".format(path.name, exc),
                        stacklevel=2,
                    )
                    continue
            light_curve = _lc.remove_nans().normalize()
            flux = np.asarray(light_curve.flux.value, dtype=float)
            sector_value = None
            try:
                sector_value = int(light_curve.meta.get("SECTOR", 0))
            except (TypeError, ValueError):
                sector_value = None
            if not sector_value or sector_value <= 0:
                sector_value = _sector_from_canonical_filename(path)
            if not sector_value or sector_value <= 0:
                _warnings.warn(
                    "skipped {0}: TESS sector cannot be verified from metadata or canonical filename".format(
                        path.name
                    ),
                    stacklevel=2,
                )
                continue
            if requested_sectors is not None and sector_value not in requested_sectors:
                continue
            time = _time_values_to_btjd_tdb(
                np.asarray(light_curve.time.value, dtype=float),
                _fits_time_header(path),
                declared_format=getattr(light_curve.time, "format", None),
                declared_scale=getattr(light_curve.time, "scale", None),
            )
            if time.size < 50 or time.size != flux.size:
                continue
            flux_err = None
            flux_err_source = "reported"
            try:
                flux_err = np.asarray(light_curve.flux_err.value, dtype=float)
                if flux_err.shape != flux.shape:
                    flux_err = None
            except (AttributeError, TypeError, ValueError) as exc:
                _warnings.warn(
                    "flux_err unavailable for {0}: {1!r} — using MAD estimate".format(
                        path.name, exc
                    ),
                    stacklevel=2,
                )
                flux_err = None
            if flux_err is not None and not np.all(np.isfinite(flux_err) & (flux_err > 0)):
                _warnings.warn(
                    "flux_err contains non-finite or non-positive values for {0} â€” using MAD estimate".format(
                        path.name
                    ),
                    stacklevel=2,
                )
                flux_err = None
                flux_err_source = "mad-estimate-invalid-reported"
            if flux_err is None:
                flux_err = np.full_like(flux, _mad_flux_error(flux))
                if flux_err_source == "reported":
                    flux_err_source = "mad-estimate"
            if sector_value in seen_sectors:
                # Different products from the same TESS sector (e.g. SPOC and
                # QLP copies) would otherwise double-count one sector and make
                # the combined design matrix singular. The first product in
                # sorted order wins (SPOC 2-min sorts before QLP for s30).
                continue
            seen_sectors.add(sector_value)
            sector_values = np.full(time.size, sector_value, dtype=int)
            binned = (
                _median_bin(time, flux, flux_err, sector_values, n_bins=max_points)
                if max_points is not None
                else (time, flux, flux_err, sector_values)
            )
            if input_sha256 != _sha256(path):
                _warnings.warn(
                    "skipped {0}: product bytes changed while loading".format(path.name),
                    stacklevel=2,
                )
                continue
            if binned[0].size >= 50:
                tables.append(
                    {
                        "time": binned[0],
                        "flux": binned[1],
                        "flux_err": binned[2],
                        "flux_err_source": flux_err_source,
                        "sector": binned[3],
                        "path": path,
                        "sha256": input_sha256,
                        "time_system": BTJD_TIME_SYSTEM,
                    }
                )
        except Exception as exc:
            _warnings.warn(
                "skipped {0}: {1!r}".format(path.name, exc),
                stacklevel=2,
            )
            continue
    if not tables:
        return None

    time = np.concatenate([table["time"] for table in tables])
    flux = np.concatenate([table["flux"] for table in tables])
    flux_err = np.concatenate([table["flux_err"] for table in tables])
    sector_values = np.concatenate([table["sector"] for table in tables])
    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "flux_err_sources": sorted({table["flux_err_source"] for table in tables}),
        "sector": sector_values.astype(int),
        "input_files": [table["path"] for table in tables],
        "input_sha256s": [table["sha256"] for table in tables],
        "time_system": BTJD_TIME_SYSTEM,
    }


def load_tpf_cubes(
    workspace: CandidateWorkspace,
    raw_only: bool = False,
    require_raw_provenance: bool = False,
) -> List[Dict[str, Any]]:
    """Return TPF pixel cubes from candidate data, or an empty list.

    Each entry has ``path``, ``sector``, ``time``, ``quality``, ``flux``
    (n_time x n_y x n_x), ``aperture`` and ``header`` (primary header dict).
    TPFs without a positive sector in their primary header or canonical file
    name are skipped rather than assigned an inferred sector number.
    """
    if require_raw_provenance:
        raw_only = True
        from .gatekeeper import has_valid_raw_product_provenance

    roots = (
        (workspace.path / "data" / "raw",)
        if raw_only
        else (
            workspace.path / "data" / "processed",
            workspace.path / "data" / "raw",
        )
    )
    fits_files: List[Path] = []
    processed_names: set = set()
    for root in roots:
        if not root.is_dir():
            continue
        hits: List[Path] = []
        for suffix in (".fits", ".fits.fz", ".fz"):
            hits.extend(root.rglob("*" + suffix))
        if root.name == "processed":
            processed_names = {h.name for h in hits}
        else:
            hits = [h for h in hits if h.name not in processed_names]
        fits_files.extend(hits)
    fits_files = [path for path in fits_files if "tp" in path.stem.lower()]
    fits_files.sort()
    if not fits_files:
        return []

    try:
        from astropy.io import fits
    except ImportError:  # pragma: no cover - optional dependency
        return []

    import warnings as _warnings

    cubes: List[Dict[str, Any]] = []
    for path in fits_files:
        try:
            if require_raw_provenance and not has_valid_raw_product_provenance(workspace, path):
                _warnings.warn(
                    "skipped {0}: raw provenance sidecar is missing, invalid, or stale".format(
                        path.name
                    ),
                    stacklevel=2,
                )
                continue
            with fits.open(path, memmap=False) as hdul:
                if len(hdul) < 3:
                    continue
                pix_hdu, ap_hdu = hdul[1], hdul[2]
                header = dict(hdul[0].header)
                header.update(dict(pix_hdu.header))
                quality = np.asarray(pix_hdu.data["QUALITY"], dtype=np.int64)
                flux = np.asarray(pix_hdu.data["FLUX"], dtype=float)
                aperture = np.asarray(ap_hdu.data)
                sector_value = None
                for key in ("SECTOR", "SECTOR_NUM"):
                    if key in header:
                        try:
                            sector_value = int(header[key])
                        except (TypeError, ValueError):
                            sector_value = None
                        if sector_value:
                            break
                if not sector_value or sector_value <= 0:
                    sector_value = _sector_from_canonical_filename(path)
                if not sector_value or sector_value <= 0:
                    _warnings.warn(
                        "skipped {0}: TESS sector cannot be verified from metadata or canonical filename".format(
                            path.name
                        ),
                        stacklevel=2,
                    )
                    continue
                time = _time_values_to_btjd_tdb(
                    np.asarray(pix_hdu.data["TIME"], dtype=float),
                    header,
                )
                if flux.shape[0] == time.size and time.size >= 50:
                    cubes.append(
                        {
                            "path": path,
                            "sector": int(sector_value),
                            "time": time,
                            "quality": quality,
                            "flux": flux,
                            "aperture": aperture,
                            "header": header,
                        }
                    )
        except Exception:
            continue
    return cubes
