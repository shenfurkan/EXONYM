"""Candidate-local LDTk quadratic limb-darkening prior generation.

This module deliberately reads stellar parameters directly from the candidate
workspace rather than using demonstration fallbacks.  A prior report is only
written after LDTk has successfully produced finite coefficients.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .workspace import CandidateWorkspace


STELLAR_PARAMETERS_PATH = Path("data") / "external" / "stellar_params.json"
OUTPUT_FILENAME = "ldtk_quadratic_limb_darkening_prior.json"

_PARAMETERS: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...], str], ...] = (
    ("teff_k", ("teff_k", "teff", "temperature_k"), ("teff_err_k", "teff_error_k", "teff_uncertainty_k", "teff_err"), "K"),
    ("logg_cgs", ("logg_cgs", "logg", "log_g"), ("logg_err_cgs", "logg_error_cgs", "logg_uncertainty_cgs", "logg_err"), "log10(cm s^-2)"),
    ("feh", ("feh", "metallicity"), ("feh_err", "feh_error", "feh_uncertainty", "metallicity_err"), "dex"),
)


def _first_value(payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _finite_float(value: Any, label: str, positive: bool = False) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{0} must be a finite number".format(label)) from exc
    if not math.isfinite(numeric):
        raise ValueError("{0} must be a finite number".format(label))
    if positive and numeric <= 0:
        raise ValueError("{0} must be positive".format(label))
    return numeric


def _load_stellar_parameters(workspace: CandidateWorkspace) -> Dict[str, Tuple[float, float, str]]:
    """Load finite candidate-owned atmospheric parameters and uncertainties."""
    path = Path(workspace.path) / STELLAR_PARAMETERS_PATH
    if not path.is_file():
        raise FileNotFoundError("missing candidate stellar parameters: {0}".format(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("candidate stellar parameters are not valid JSON: {0}".format(path)) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("candidate stellar parameters must be a JSON object")

    parameters: Dict[str, Tuple[float, float, str]] = {}
    for canonical, value_names, uncertainty_names, unit in _PARAMETERS:
        value = _finite_float(_first_value(payload, value_names), canonical, positive=canonical == "teff_k")
        uncertainty = _finite_float(
            _first_value(payload, uncertainty_names), "{0} uncertainty".format(canonical), positive=True
        )
        parameters[canonical] = (value, uncertainty, unit)
    return parameters


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filter_label(filter_definition: Any, index: int) -> str:
    label = getattr(filter_definition, "name", None)
    if label is None:
        label = str(filter_definition)
    label = str(label).strip()
    if not label:
        raise ValueError("filter at index {0} has no usable name".format(index))
    return label


def _quadratic_rows(
    filters: Sequence[Any], coefficients: Any, uncertainties: Any
) -> List[Dict[str, Any]]:
    """Validate LDTk's per-filter quadratic coefficients for JSON output."""
    try:
        coefficient_count = len(coefficients)
        uncertainty_count = len(uncertainties)
    except TypeError as exc:
        raise ValueError("LDTk returned non-sequence quadratic coefficients") from exc
    if coefficient_count != len(filters) or uncertainty_count != len(filters):
        raise ValueError("LDTk returned coefficients for an unexpected number of filters")

    rows = []
    for index, filter_definition in enumerate(filters):
        try:
            u1, u2 = coefficients[index]
            u1_err, u2_err = uncertainties[index]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("LDTk returned malformed quadratic coefficients") from exc
        rows.append(
            {
                "filter": _filter_label(filter_definition, index),
                "u1": _finite_float(u1, "u1"),
                "u1_err": _finite_float(u1_err, "u1 uncertainty", positive=True),
                "u2": _finite_float(u2, "u2"),
                "u2_err": _finite_float(u2_err, "u2 uncertainty", positive=True),
                "unit": "dimensionless",
            }
        )
    return rows


def generate_ldtk_quadratic_prior(
    workspace: CandidateWorkspace, filters: Sequence[Any]
) -> Path:
    """Generate LDTk quadratic limb-darkening priors for a candidate workspace.

    Args:
        workspace: Candidate workspace that owns ``data/external/stellar_params.json``.
        filters: Non-empty sequence of LDTk filter definitions for the observed bandpasses.

    Returns:
        Path to ``outputs/ldtk_quadratic_limb_darkening_prior.json``.

    Raises:
        FileNotFoundError: Candidate stellar parameters are absent.
        ValueError: Parameters, uncertainties, filters, or LDTk output are invalid.
        RuntimeError: LDTk cannot be imported.
    """
    if isinstance(filters, (str, bytes)) or not filters:
        raise ValueError("filters must be a non-empty sequence of LDTk filter definitions")
    filters = list(filters)
    parameters = _load_stellar_parameters(workspace)

    try:
        import ldtk
        from ldtk import LDPSetCreator
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("LDTk dependency is unavailable; no prior artifact was written") from exc

    creator = LDPSetCreator(
        teff=parameters["teff_k"][:2],
        logg=parameters["logg_cgs"][:2],
        z=parameters["feh"][:2],
        filters=filters,
    )
    profiles = creator.create_profiles()
    coefficients, uncertainties = profiles.coeffs_qd(do_mc=True)
    quadratic_coefficients = _quadratic_rows(filters, coefficients, uncertainties)

    parameters_path = Path(workspace.path) / STELLAR_PARAMETERS_PATH
    generated_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload = {
        "schema_version": "1.0",
        "work_package": "LDTK_QUADRATIC_LIMB_DARKENING_PRIOR",
        "generated_utc": generated_utc,
        "candidate_id": workspace.candidate_id,
        "method": "LDTk quadratic coefficients from stellar-atmosphere profiles",
        "ldtk": {
            "version": str(getattr(ldtk, "__version__", "unknown")),
            "coefficient_method": "coeffs_qd",
            "monte_carlo": True,
        },
        "input_provenance": {
            "stellar_parameters_path": STELLAR_PARAMETERS_PATH.as_posix(),
            "stellar_parameters_sha256": _sha256(parameters_path),
        },
        "stellar_parameters": {
            name: {"value": value, "uncertainty": uncertainty, "unit": unit}
            for name, (value, uncertainty, unit) in parameters.items()
        },
        "quadratic_coefficients": quadratic_coefficients,
    }
    encoded_payload = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path = Path(workspace.path) / "outputs" / OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded_payload, encoding="utf-8")
    return output_path
