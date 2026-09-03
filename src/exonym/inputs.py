"""Target-neutral, candidate-local input loading for scientific modules.

Every loader probes candidate workspace files and metadata only. Ephemerides,
stellar parameters, photometry, light curves, and target-pixel products are
read dynamically from candidate-owned records. Missing or invalid evidence is
reported explicitly and is never substituted with demonstration values.

Provenance-aware loaders reject stale or unbound BLS and detrending evidence
before a downstream workflow treats it as candidate-derived. Light-curve and
pixel loaders also retain mission time-system and sector provenance rather than
inferring unavailable values.

Scientific Boundary:
    Loader output indicates availability and binding of input evidence. It does
    not establish a detection, source association, or scientific claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zipfile import BadZipFile

import numpy as np

from .workspace import CandidateWorkspace, validate_signal_suffix

BTJD_REFERENCE_BJD = 2457000.0
BTJD_TIME_SYSTEM = "BTJD_TDB"
# This is a candidate-selection threshold, not a calibrated false-alarm rate.
MINIMUM_BLS_CANDIDATE_SNR = 7.1
PIPELINE_NORMALIZATION = {"kind": "pipeline-normalization"}

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


def _has_complete_candidate_ephemeris(
    result: Dict[str, Any], source_prefix: str, *, require_duration: bool = True, require_depth: bool = True
) -> bool:
    """Return whether all transit fields form one usable candidate-owned record."""
    field_sources = result.get("field_sources")
    if not isinstance(field_sources, dict):
        return False
    required_fields = ("period_days", "epoch_btjd")
    if require_duration:
        required_fields += ("duration_days",)
    if require_depth:
        required_fields += ("depth_ppm",)
    if not all(
        str(field_sources.get(field, "")).startswith(source_prefix)
        for field in required_fields
    ):
        return False
    try:
        period_days = float(result["period_days"])
        epoch_btjd = float(result["epoch_btjd"])
        duration_days = float(result["duration_days"]) if require_duration else None
        depth_ppm = float(result["depth_ppm"]) if require_depth else None
    except (KeyError, TypeError, ValueError):
        return False
    return (
        # ``epoch_btjd`` is an explicit BTJD coordinate. The candidate loader
        # records BTJD_TDB for it; accept an omitted redundant label from an
        # in-memory caller, but never accept a conflicting declared system.
        result.get("time_system") in (None, BTJD_TIME_SYSTEM)
        and all(
            math.isfinite(value)
            for value in (period_days, epoch_btjd, duration_days)
            if value is not None
        )
        and period_days > 0.0
        and (
            duration_days is None
            or (duration_days > 0.0 and duration_days < period_days)
        )
        and (depth_ppm is None or (math.isfinite(depth_ppm) and depth_ppm > 0.0))
    )


def is_complete_candidate_ephemeris(
    ephemeris: Dict[str, Any], *, require_duration: bool = True, require_depth: bool = True
) -> bool:
    """Return whether an ephemeris is complete, physical, and candidate-derived.

    This is the shared boundary for scientific callers.  A parsed configuration
    can be candidate-owned yet partial; it must not be passed to inference until
    period, epoch, the BTJD_TDB declaration, and field provenance are present
    and mutually consistent. Duration and depth are required by default;
    callers that do not consume one or both fields may set the corresponding
    requirement to false.
    """
    if not isinstance(ephemeris, dict):
        return False
    source = ephemeris.get("source")
    source_prefixes = {
        "candidate-config",
        "candidate-config-signal",
        "candidate-data-bls",
        "bls-search",
    }
    if not isinstance(source, str):
        return False
    source_prefix = source.removeprefix("partial-")
    if source_prefix not in source_prefixes:
        return False
    return _has_complete_candidate_ephemeris(
        ephemeris,
        source_prefix,
        require_duration=require_duration,
        require_depth=require_depth,
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


def _is_bound_preprocessing(
    workspace: CandidateWorkspace, record: object, require_transit_mask: bool = True
) -> bool:
    """Confirm a preprocessing record names a current derived product."""
    if record == PIPELINE_NORMALIZATION:
        return True
    if not isinstance(record, dict) or record.get("kind") != "candidate-detrending":
        return False
    method = record.get("method")
    if method not in ("running-median", "wotan", "celerite"):
        return False
    manifest_record = record.get("manifest")
    artifact_record = record.get("artifact")
    if not isinstance(manifest_record, dict) or not isinstance(artifact_record, dict):
        return False
    manifest_path = workspace.path / "outputs" / "detrending_manifest.{0}.json".format(method)
    artifact_path = workspace.path / "data" / "processed" / "detrended-{0}.npz".format(method)
    expected_manifest_path = manifest_path.relative_to(workspace.path).as_posix()
    expected_artifact_path = artifact_path.relative_to(workspace.path).as_posix()
    if (
        manifest_record.get("path") != expected_manifest_path
        or artifact_record.get("path") != expected_artifact_path
        or not manifest_path.is_file()
        or not artifact_path.is_file()
        or manifest_record.get("sha256") != _sha256(manifest_path)
        or artifact_record.get("sha256") != _sha256(artifact_path)
    ):
        return False
    manifest = _read_json(manifest_path)
    artifact = manifest.get("artifact") if isinstance(manifest, dict) else None
    configuration = manifest.get("configuration") if isinstance(manifest, dict) else None
    # SCIENTIFIC_BOUNDARY: A preprocessing label alone is not evidence; the
    # artifact and raw inputs must still bind. Targeted consumers also require
    # an ephemeris-bound protected-transit mask.
    if not (
        isinstance(artifact, dict)
        and isinstance(configuration, dict)
        and manifest.get("schema_version") == 2
        and manifest.get("candidate_id") == workspace.candidate_id
        and manifest.get("method") == method
        and artifact.get("path") == expected_artifact_path
        and artifact.get("sha256") == artifact_record.get("sha256")
        and artifact.get("data_sha256") == artifact_record.get("data_sha256")
    ):
        return False
    transit_mask_applied = configuration.get("transit_mask_applied")
    transit_mask_provenance = configuration.get("transit_mask_provenance")
    if not isinstance(transit_mask_applied, bool):
        return False
    if not transit_mask_applied:
        return not require_transit_mask and transit_mask_provenance is None
    if not isinstance(transit_mask_provenance, dict):
        return False
    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            time = np.asarray(archive["time_btjd"], dtype=float)
        from .detrending import validate_transit_mask_provenance

        provenance = transit_mask_provenance
        validate_transit_mask_provenance(time, provenance, provenance.get("ephemeris"))
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def is_manifest_bound_bls_result(
    workspace: CandidateWorkspace,
    result_path: Path,
    payload: Dict[str, Any],
    signal: Optional[str],
) -> bool:
    """Check whether a detected BLS result remains bound to its manifest.

    The check verifies candidate ownership, result and manifest digests,
    declared BTJD time system, detected ephemeris fields, raw-product inputs,
    and any mask-bound preprocessing record. It returns false rather than
    accepting incomplete or stale evidence.

    Args:
        workspace: Candidate workspace that owns result and manifest paths.
        result_path: Candidate-local BLS result JSON path.
        payload: Parsed BLS result mapping.
        signal: Optional validated per-signal suffix expected by the manifest.

    Returns:
        True only when the result is detected, physically usable, and
        provenance-bound to current candidate-local evidence.
    """
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
    manifest_preprocessing = configuration.get("preprocessing", PIPELINE_NORMALIZATION)
    result_preprocessing = payload.get("preprocessing", PIPELINE_NORMALIZATION)
    if (
        manifest_preprocessing != result_preprocessing
        or not _is_bound_preprocessing(
            workspace, manifest_preprocessing, require_transit_mask=signal is not None
        )
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
    """Check that a BLS-derived transit configuration names current evidence.

    Args:
        workspace: Candidate workspace that owns configuration and BLS outputs.
        config_path: Candidate-local transit-configuration path.
        payload: Parsed configuration mapping to validate.
        signal: Optional per-signal suffix expected by all linked records.

    Returns:
        True only when configuration values, result digest, manifest digest,
        and current BLS evidence agree exactly at the recorded precision.
    """
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
    """Load the best available ephemeris and field-level provenance.

    A validated per-signal configuration has precedence when requested.
    Otherwise, candidate configuration is considered before provenance-bound
    BLS outputs. When neither source is readable, every transit field remains
    unavailable rather than receiving an in-band demonstration substitute.

    Args:
        workspace: Candidate workspace whose configuration and outputs are
            inspected.
        signal: Optional validated per-signal suffix.

    Returns:
        Period in days, epoch in BTJD_TDB days, duration in days, depth in ppm,
        time-system label, source label, and per-field provenance mapping.

    Raises:
        ValueError: If signal suffix syntax is invalid.

    Notes:
        Callers that require observed candidate evidence must require
        :func:`is_complete_candidate_ephemeris` rather than treating a partial
        configuration as scientifically usable.
    """
    signal = validate_signal_suffix(signal)
    result: Dict[str, Any] = {
        "period_days": None,
        "epoch_btjd": None,
        "duration_days": None,
        "depth_ppm": None,
        "time_system": None,
        "source": "unavailable",
        "field_sources": {
            "period_days": None,
            "epoch_btjd": None,
            "duration_days": None,
            "depth_ppm": None,
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

    if result["source"] == "unavailable":
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

    if result["period_days"] is not None and (result["period_days"] <= 0 or (result["duration_days"] is not None and result["duration_days"] <= 0)):
        result["period_days"] = None
        result["duration_days"] = None
        result["source"] = "unavailable"
    return result


def load_stellar_parameters(workspace: CandidateWorkspace) -> Dict[str, Any]:
    """Load stellar parameters with field availability and source labels.

    Missing files yield unavailable source labels with None fields.
    A partly populated candidate file yields partial-candidate-data; callers
    that require complete observed stellar physics must require the
    candidate-data source label.

    Args:
        workspace: Candidate workspace containing optional external stellar
            parameter data.

    Returns:
        Effective temperature in K, surface gravity in log10(cgs), metallicity
        in dex, mass and radius in solar units, optional astrometry and
        uncertainties, plus a source label. A candidate-owned nested
        ``dnu_correction`` record is retained for the asteroseismic diagnostic
        to validate before use.

    Notes:
        Optional position and parallax fields do not determine the stellar
        physics source label. Optional mass and radius errors represent
        candidate-supplied symmetric one-sigma uncertainties.
    """
    _PHYSICS_FIELDS = ("teff_k", "logg_cgs", "feh", "mass_solar", "radius_solar")
    result: Dict[str, Any] = {
        "teff_k": None,
        "logg_cgs": None,
        "feh": None,
        "mass_solar": None,
        "radius_solar": None,
        "parallax_mas": None,
        "source": "unavailable",
    }
    params_path = workspace.path / "data" / "external" / "stellar_params.json"
    payload = _read_json(params_path)
    if payload is None:
        return result
    values = {
        "ra_deg": _first_number(payload, ("ra_deg", "ra", "right_ascension")),
        "dec_deg": _first_number(payload, ("dec_deg", "dec", "declination")),
        "teff_k": _first_number(payload, ("teff_k", "teff", "temperature_k")),
        "teff_k_err": _first_number(
            payload, ("teff_k_err", "teff_err_k", "teff_error_k", "teff_uncertainty_k")
        ),
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
    dnu_correction = payload.get("dnu_correction")
    if isinstance(dnu_correction, dict):
        # SCIENTIFIC_BOUNDARY: This loader preserves candidate-owned evidence;
        # asteroseismology validates its factor and applicability before use.
        result["dnu_correction"] = dict(dnu_correction)
    physics_present = sum(1 for f in _PHYSICS_FIELDS if values.get(f) is not None)
    if physics_present == len(_PHYSICS_FIELDS):
        result["source"] = "candidate-data"
    elif physics_present > 0:
        result["source"] = "partial-candidate-data"
    return result


def load_photometry(workspace: CandidateWorkspace) -> Optional[Dict[str, Any]]:
    """Load optional candidate-local broadband stellar photometry.

    Args:
        workspace: Candidate workspace containing an optional external
            photometry JSON record.

    Returns:
        Parsed photometry mapping when readable, otherwise None. Band entries
        are passed through for consumers that apply their own validation.

    Notes:
        Presence of a mapping does not establish calibration, extinction
        treatment, or suitability for a particular inference model.
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


def _load_detrended_light_curve_table(
    workspace: CandidateWorkspace,
    method: str,
    max_points: Optional[int],
    sectors: Optional[Sequence[int]],
    require_raw_provenance: bool,
    require_transit_mask: bool,
) -> Optional[Dict[str, Any]]:
    """Load one hash-bound candidate detrending product without FITS conversion."""
    normalized_method = method.strip().lower()
    if normalized_method not in ("running-median", "wotan", "celerite"):
        raise ValueError("detrending_method must be running-median, wotan, or celerite")
    manifest_path = workspace.path / "outputs" / "detrending_manifest.{0}.json".format(normalized_method)
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("detrended input requires a readable detrending manifest")
    if manifest.get("schema_version") != 2:
        raise ValueError(
            "detrended input uses a legacy manifest; regenerate with `exonym detrend`"
        )
    artifact = manifest.get("artifact")
    expected_path = "data/processed/detrended-{0}.npz".format(normalized_method)
    if (
        manifest.get("candidate_id") != workspace.candidate_id
        or manifest.get("method") != normalized_method
        or not isinstance(artifact, dict)
        or artifact.get("path") != expected_path
    ):
        raise ValueError("detrending manifest does not match its candidate, method, or artifact path")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(
            "detrended input lacks a mask-bound configuration; regenerate with `exonym detrend`"
        )
    transit_mask_applied = configuration.get("transit_mask_applied")
    transit_mask_provenance = configuration.get("transit_mask_provenance")
    if not isinstance(transit_mask_applied, bool):
        raise ValueError(
            "detrended input lacks a mask-bound configuration; regenerate with `exonym detrend`"
        )
    if transit_mask_applied and not isinstance(transit_mask_provenance, dict):
        raise ValueError(
            "detrended input has no transit mask provenance; regenerate with `exonym detrend`"
        )
    if not transit_mask_applied and transit_mask_provenance is not None:
        raise ValueError("unmasked detrended input must not declare transit mask provenance")
    if require_transit_mask and not transit_mask_applied:
        raise ValueError(
            "detrended input is not transit-mask-bound; regenerate with `exonym detrend`"
        )
    artifact_path = workspace.path / expected_path
    if not artifact_path.is_file() or artifact.get("sha256") != _sha256(artifact_path):
        raise ValueError("detrended input artifact is missing or does not match its manifest digest")
    data_sha256 = artifact.get("data_sha256")
    if not isinstance(data_sha256, str):
        raise ValueError("detrended input artifact has no numerical content digest")
    from .remediation import numerical_npz_sha256

    try:
        numerical_digest = numerical_npz_sha256(artifact_path)
    except (BadZipFile, EOFError, OSError, ValueError) as exc:
        raise ValueError("detrended input artifact is unreadable") from exc
    if numerical_digest != data_sha256:
        raise ValueError("detrended input artifact numerical content does not match its manifest")

    product_records = manifest.get("input_products")
    input_files: List[Path] = []
    input_sha256s: List[str] = []
    if require_raw_provenance:
        from .gatekeeper import has_valid_raw_product_provenance

        if not isinstance(product_records, list) or not product_records:
            raise ValueError("detrended input requires hash-bound raw input products")
        for record in product_records:
            if not isinstance(record, dict):
                raise ValueError("detrending manifest has a malformed raw input record")
            relative = record.get("path")
            expected_digest = record.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected_digest, str):
                raise ValueError("detrending manifest raw input record is incomplete")
            product_path = (workspace.path / relative).resolve()
            try:
                product_path.relative_to((workspace.path / "data" / "raw").resolve())
            except ValueError as exc:
                raise ValueError("detrending manifest references a non-raw input product") from exc
            if (
                not product_path.is_file()
                or _sha256(product_path) != expected_digest
                or not has_valid_raw_product_provenance(workspace, product_path)
            ):
                raise ValueError("detrended input raw provenance is missing, stale, or mismatched")
            input_files.append(product_path)
            input_sha256s.append(expected_digest)

    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            required = {"time_btjd", "detrended_flux", "sector"}
            if not required.issubset(archive.files):
                raise ValueError("detrended artifact must retain time_btjd, detrended_flux, and sector arrays")
            time = np.asarray(archive["time_btjd"], dtype=float)
            flux = np.asarray(archive["detrended_flux"], dtype=float)
            sector_values = np.asarray(archive["sector"], dtype=int)
            flux_err = (
                np.asarray(archive["detrended_flux_err"], dtype=float)
                if "detrended_flux_err" in archive.files
                else None
            )
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("detrended artifact is unreadable") from exc
    if (
        time.ndim != 1
        or flux.ndim != 1
        or sector_values.ndim != 1
        or time.shape != flux.shape
        or time.shape != sector_values.shape
        or np.any(sector_values <= 0)
    ):
        raise ValueError("detrended artifact arrays have incompatible shapes or invalid sectors")
    if flux_err is not None and (flux_err.ndim != 1 or flux_err.shape != flux.shape):
        raise ValueError("detrended artifact uncertainty array does not match its flux")
    if transit_mask_applied:
        from .detrending import validate_transit_mask_provenance

        try:
            # The processed artifact carries the exact canonical ephemeris used to
            # protect its transit mask. A stale BLS-derived config can become
            # unavailable before a fresh blind BLS search rebinds it. In that
            # narrow case validate the immutable mask provenance against its
            # recorded ephemeris. A complete current candidate ephemeris still
            # has to match exactly.
            mask_ephemeris = transit_mask_provenance.get("ephemeris")
            current_ephemeris = load_transit_ephemeris(workspace)
            ephemeris_for_validation = (
                mask_ephemeris
                if not is_complete_candidate_ephemeris(current_ephemeris)
                else current_ephemeris
            )
            validate_transit_mask_provenance(
                time, transit_mask_provenance, ephemeris_for_validation
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "detrended input transit mask provenance is stale or mismatched"
            ) from exc
    valid = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        valid &= np.isfinite(flux_err) & (flux_err > 0)
    time, flux, sector_values = time[valid], flux[valid], sector_values[valid]
    if flux_err is None:
        flux_err = np.full_like(flux, _mad_flux_error(flux))
        flux_err_source = "detrended-mad-estimate"
    else:
        flux_err = flux_err[valid]
        flux_err_source = "detrended-reported"
    if sectors is not None:
        selected = np.isin(sector_values, np.asarray(sectors, dtype=int))
        time, flux, flux_err, sector_values = (
            time[selected], flux[selected], flux_err[selected], sector_values[selected]
        )
    if time.size < 50:
        return None
    tables: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for sector_value in sorted(int(value) for value in np.unique(sector_values)):
        mask = sector_values == sector_value
        binned = (
            _median_bin(time[mask], flux[mask], flux_err[mask], sector_values[mask], n_bins=max_points)
            if max_points is not None
            else (time[mask], flux[mask], flux_err[mask], sector_values[mask])
        )
        if binned[0].size:
            tables.append(binned)
    if not tables:
        return None
    return {
        "time": np.concatenate([table[0] for table in tables]),
        "flux": np.concatenate([table[1] for table in tables]),
        "flux_err": np.concatenate([table[2] for table in tables]),
        "flux_err_sources": [flux_err_source],
        "sector": np.concatenate([table[3] for table in tables]).astype(int),
        "input_files": input_files,
        "input_sha256s": input_sha256s,
        "time_system": BTJD_TIME_SYSTEM,
        "detrending": {
            "kind": "candidate-detrending",
            "method": normalized_method,
            "manifest": {
                "path": manifest_path.relative_to(workspace.path).as_posix(),
                "sha256": _sha256(manifest_path),
            },
            "artifact": {
                "path": expected_path,
                "sha256": artifact["sha256"],
                "data_sha256": data_sha256,
            },
        },
        "sampling": {
            "mode": "median-binned" if max_points is not None else "native",
            "max_points": int(max_points) if max_points is not None else None,
        },
    }


def load_light_curve_table(
    workspace: CandidateWorkspace,
    max_points: Optional[int] = 4000,
    sectors: Optional[Sequence[int]] = None,
    raw_only: bool = False,
    require_raw_provenance: bool = False,
    detrending_method: Optional[str] = None,
    require_transit_mask: bool = True,
) -> Optional[Dict[str, Any]]:
    """Load normalized candidate light curves with sector and input provenance.

    Processed FITS products take precedence over same-named raw products unless
    raw_only is requested. A named detrending method instead loads a
        hash-bound processed array and validates its raw-input provenance before
        returning it. Transit-mask provenance is required by default.

    Args:
        workspace: Candidate workspace that owns FITS or detrended inputs.
        max_points: Optional per-product cadence cap after product-local
            normalization and quality filtering.
        sectors: Optional positive sector selection; products without a
            resolved selected sector are excluded.
        raw_only: Restrict loading to data/raw products.
        require_raw_provenance: Require valid provenance sidecars and therefore
            raw products.
        detrending_method: Optional named mask-bound detrending product to
            consume instead of FITS data.
        require_transit_mask: Require a transit mask bound to the current
            ephemeris. Blind searches may explicitly disable this to avoid
            masking a previous candidate signal.

    Returns:
        A mapping with BTJD_TDB time, normalized flux, uncertainties, per-
        cadence sectors, and accepted input records, or None when no readable
        product remains after filtering.

    Raises:
        ValueError: If detrending options conflict or a named processed product
            is stale, unbound, malformed, or inconsistent with current inputs.

    Notes:
        Accepted sectors are concatenated without global rebinning so effective
        cadence does not depend on the number of observed sectors.
    """
    if detrending_method is not None:
        if raw_only:
            raise ValueError("raw_only cannot be combined with a detrending_method")
        return _load_detrended_light_curve_table(
            workspace,
            detrending_method,
            max_points,
            sectors,
            require_raw_provenance,
            require_transit_mask,
        )
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
            else:
                _warnings.warn(
                    "Product {0} has no QUALITY column; skipping to avoid unmasked cadences "
                    "entering science path.".format(path.name),
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
        "sampling": {
            "mode": "median-binned" if max_points is not None else "native",
            "max_points": int(max_points) if max_points is not None else None,
        },
    }


def load_tpf_cubes(
    workspace: CandidateWorkspace,
    raw_only: bool = False,
    require_raw_provenance: bool = False,
    skipped_products: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Load candidate target-pixel cubes with fail-closed quality filtering.

    Every retained cube has a candidate-local path, resolved positive sector,
    BTJD_TDB time, integral quality flags, pixel-flux cube, aperture mask, and
    primary-header mapping. Products lacking quality information or unambiguous
    sector provenance are skipped rather than guessed.

    Args:
        workspace: Candidate workspace that owns target-pixel products.
        raw_only: Restrict loading to data/raw products.
        require_raw_provenance: Require valid provenance sidecars and restrict
            loading to raw products.
        skipped_products: Optional list extended with candidate-relative paths
            and reasons for products rejected before pixel analysis.

    Returns:
        Retained target-pixel cube mappings, or an empty list when no usable
        products or optional FITS dependency are available.

    Notes:
        QUALITY flags support rejection of unsuitable cadences; they do not
        calibrate pixel-level systematics or source localization.
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

    def record_skipped_product(path: Path, reason: str) -> None:
        if skipped_products is not None:
            skipped_products.append(
                {
                    "path": path.relative_to(workspace.path).as_posix(),
                    "reason": reason,
                }
            )

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
                try:
                    raw_quality = np.asarray(pix_hdu.data["QUALITY"])
                except (AttributeError, KeyError):
                    _warnings.warn(
                        "skipped {0}: no QUALITY column; refusing unmasked TPF cadences".format(
                            path.name
                        ),
                        stacklevel=2,
                    )
                    record_skipped_product(path, "missing-quality-column")
                    continue
                except (TypeError, ValueError) as exc:
                    _warnings.warn(
                        "skipped {0}: unusable QUALITY column: {1!r}".format(path.name, exc),
                        stacklevel=2,
                    )
                    record_skipped_product(path, "unusable-quality-column")
                    continue
                # NUMERICAL_GUARD: Do not coerce ambiguous QUALITY shapes or
                # values; a pixel product is safer to skip than reinterpret.
                if raw_quality.ndim != 1:
                    _warnings.warn(
                        "skipped {0}: unusable QUALITY column; values must be one-dimensional integers".format(
                            path.name
                        ),
                        stacklevel=2,
                    )
                    record_skipped_product(path, "unusable-quality-column")
                    continue
                int64_limits = np.iinfo(np.int64)
                if np.issubdtype(raw_quality.dtype, np.integer):
                    if (
                        np.issubdtype(raw_quality.dtype, np.unsignedinteger)
                        and np.any(raw_quality > int64_limits.max)
                    ):
                        _warnings.warn(
                            "skipped {0}: unusable QUALITY column; values exceed int64 flags".format(
                                path.name
                            ),
                            stacklevel=2,
                        )
                        record_skipped_product(path, "unusable-quality-column")
                        continue
                    quality = raw_quality.astype(np.int64)
                elif np.issubdtype(raw_quality.dtype, np.floating):
                    quality_values = np.asarray(raw_quality, dtype=np.float64)
                    if (
                        not np.all(np.isfinite(quality_values))
                        or not np.all(quality_values == np.floor(quality_values))
                        or np.any(quality_values < int64_limits.min)
                        # ``float64`` cannot exactly represent int64.max, so
                        # reject its rounded boundary before the narrowing cast.
                        or np.any(quality_values >= float(int64_limits.max))
                    ):
                        _warnings.warn(
                            "skipped {0}: unusable QUALITY column; values must be finite int64 flags".format(
                                path.name
                            ),
                            stacklevel=2,
                        )
                        record_skipped_product(path, "unusable-quality-column")
                        continue
                    quality = quality_values.astype(np.int64)
                else:
                    _warnings.warn(
                        "skipped {0}: unusable QUALITY column; values must be numeric flags".format(
                            path.name
                        ),
                        stacklevel=2,
                    )
                    record_skipped_product(path, "unusable-quality-column")
                    continue
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
                time_values = np.asarray(pix_hdu.data["TIME"], dtype=float)
                if quality.ndim != 1 or quality.size != time_values.size:
                    _warnings.warn(
                        "skipped {0}: unusable QUALITY cadence count".format(path.name),
                        stacklevel=2,
                    )
                    record_skipped_product(path, "unusable-quality-column")
                    continue
                time = _time_values_to_btjd_tdb(time_values, header)
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
