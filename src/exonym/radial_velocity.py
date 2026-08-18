"""Candidate-local radial-velocity ingestion and Keplerian model comparison.

Observation times are BJD_TDB days. Velocities and their uncertainties are
metres per second. The fitted Keplerian is descriptive dynamical evidence: it
does not create a claim or change a candidate disposition.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .resources import read_schema_text
from .workspace import CandidateWorkspace


RV_OBSERVATIONS_FILENAME = "radial_velocity_observations.json"
RV_OBSERVATIONS_SCHEMA = "radial-velocity-observations.schema.json"
RV_FIT_FILENAME = "rv_keplerian_fit.json"
RV_FIT_SCHEMA = "rv-keplerian-fit.schema.json"
RV_ENGINE_NAME = "rv-keplerian"
MAX_INGEST_BYTES = 10 * 1024 * 1024
_TAU = 2.0 * math.pi


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_object_keys(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    """Reject ambiguous JSON objects rather than retaining a last duplicate key."""
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {0}".format(key))
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError("non-finite JSON number: {0}".format(value))


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _read_safe_json(path: Path) -> object:
    """Load bounded UTF-8 JSON with duplicate and non-finite values rejected."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("RV input must be a regular JSON file")
    if path.stat().st_size > MAX_INGEST_BYTES:
        raise ValueError("RV input exceeds the maximum supported JSON size")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("RV input is not valid finite UTF-8 JSON: {0}".format(exc)) from exc


def _validate_observation_record(workspace: CandidateWorkspace, record: object) -> Dict[str, Any]:
    """Validate one candidate-owned observation payload against its JSON Schema."""
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema is required for RV observation ingestion") from exc
    schema = json.loads(read_schema_text(workspace.repository_root, RV_OBSERVATIONS_SCHEMA))
    try:
        jsonschema.validate(record, schema, format_checker=jsonschema.FormatChecker())
    except jsonschema.ValidationError as exc:
        raise ValueError("RV observation schema violation: {0}".format(exc.message)) from exc
    if not isinstance(record, dict):
        raise ValueError("RV observation record must be a JSON object")
    if record.get("candidate_id") != workspace.candidate_id:
        raise ValueError("RV observation candidate_id does not match the workspace")
    observations = record.get("observations")
    if not isinstance(observations, list):
        raise ValueError("RV observations must be an array")
    seen = set()
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("RV observation is not an object")
        time_value = observation["observation_time"]["value"]
        key = (observation["instrument"], float(time_value))
        if key in seen:
            raise ValueError("RV observations must not duplicate an instrument and time")
        seen.add(key)
    return record


def _observation_path(workspace: CandidateWorkspace) -> Path:
    return workspace.path / "data" / "external" / RV_OBSERVATIONS_FILENAME


def ingest_radial_velocity_observations(workspace: CandidateWorkspace, source_path: Path) -> Path:
    """Validate and atomically canonicalize an RV observation JSON file below a workspace."""
    record = _validate_observation_record(workspace, _read_safe_json(Path(source_path)))
    destination = _observation_path(workspace)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_radial_velocity_observations(workspace: CandidateWorkspace) -> Dict[str, Any]:
    """Load the one validated candidate-local RV observation record."""
    path = _observation_path(workspace)
    if not path.is_file():
        raise FileNotFoundError("no candidate-local RV observations have been ingested")
    return _validate_observation_record(workspace, _read_safe_json(path))


def keplerian_velocity_m_per_s(
    time_bjd_tdb: Sequence[float],
    semi_amplitude_m_per_s: float,
    mean_anomaly_reference_rad: float,
    eccentricity: float,
    argument_periastron_rad: float,
    reference_time_bjd_tdb: float,
    period_days: float,
) -> np.ndarray:
    """Evaluate a Keplerian radial-velocity curve in m/s at BJD_TDB days."""
    if period_days <= 0 or not 0 <= eccentricity < 1 or semi_amplitude_m_per_s < 0:
        raise ValueError("Keplerian parameters are outside their physical bounds")
    mean_anomaly = (
        mean_anomaly_reference_rad
        + _TAU * (np.asarray(time_bjd_tdb, dtype=float) - reference_time_bjd_tdb) / period_days
    )
    eccentric_anomaly = np.mod(mean_anomaly, _TAU)
    for _ in range(16):
        numerator = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
        denominator = 1.0 - eccentricity * np.cos(eccentric_anomaly)
        eccentric_anomaly -= numerator / denominator
    true_anomaly = 2.0 * np.arctan2(
        math.sqrt(1.0 + eccentricity) * np.sin(eccentric_anomaly / 2.0),
        math.sqrt(1.0 - eccentricity) * np.cos(eccentric_anomaly / 2.0),
    )
    return semi_amplitude_m_per_s * (
        np.cos(true_anomaly + argument_periastron_rad)
        + eccentricity * math.cos(argument_periastron_rad)
    )


def _observation_arrays(record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    observations = record["observations"]
    instruments = sorted({str(item["instrument"]) for item in observations})
    positions = {instrument: index for index, instrument in enumerate(instruments)}
    time = np.asarray([item["observation_time"]["value"] for item in observations], dtype=float)
    velocity = np.asarray([item["velocity"]["value"] for item in observations], dtype=float)
    uncertainty = np.asarray([item["uncertainty"]["value"] for item in observations], dtype=float)
    labels = np.asarray([positions[str(item["instrument"])] for item in observations], dtype=int)
    if not (np.all(np.isfinite(time)) and np.all(np.isfinite(velocity)) and np.all(np.isfinite(uncertainty))):
        raise ValueError("RV observations must be finite")
    if np.any(uncertainty <= 0):
        raise ValueError("RV observation uncertainties must be positive")
    return time, velocity, uncertainty, labels, instruments


def _instrument_design(labels: np.ndarray, instrument_count: int) -> np.ndarray:
    return np.column_stack([(labels == index).astype(float) for index in range(instrument_count)])


def _constant_model(
    velocity: np.ndarray, uncertainty: np.ndarray, labels: np.ndarray, instrument_count: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = _instrument_design(labels, instrument_count)
    weights = 1.0 / uncertainty**2
    normal = design.T @ (weights[:, None] * design)
    covariance = np.linalg.pinv(normal)
    coefficients = covariance @ (design.T @ (weights * velocity))
    return design @ coefficients, coefficients, covariance


def _parameter(value: Optional[float], uncertainty: Optional[float], unit: str) -> Dict[str, Optional[float]]:
    return {"value": value, "uncertainty": uncertainty, "unit": unit, "source": "weighted RV fit"}


def _model_statistics(residual: np.ndarray, uncertainty: np.ndarray, parameter_count: int) -> Dict[str, float]:
    chi_squared = float(np.sum((residual / uncertainty) ** 2))
    log_likelihood = float(
        -0.5 * np.sum((residual / uncertainty) ** 2 + np.log(_TAU * uncertainty**2))
    )
    count = int(residual.size)
    return {
        "chi_squared": chi_squared,
        "log_likelihood": log_likelihood,
        "aic": float(2 * parameter_count - 2.0 * log_likelihood),
        "bic": float(math.log(count) * parameter_count - 2.0 * log_likelihood),
        "parameter_count": parameter_count,
    }


def _eccentricity_components(theta: np.ndarray, component_start: int) -> Tuple[float, float, float, float]:
    first = float(theta[component_start])
    second = float(theta[component_start + 1])
    norm_squared = first * first + second * second
    eccentricity = 0.95 * norm_squared / (1.0 + norm_squared)
    argument = math.atan2(second, first) if norm_squared > 0 else 0.0
    return eccentricity, argument, first, second


def _finite_uncertainty(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) and value >= 0 else None


def fit_radial_velocity(
    workspace: CandidateWorkspace,
    period_days: float,
    period_uncertainty_days: Optional[float] = None,
) -> Path:
    """Fit constant and eccentric Keplerian RV models at a fixed period in days.

    The period is an explicit user-supplied dynamical input, not a claim. The
    report retains the observation hash, units, covariance-derived uncertainties,
    and information-criterion comparison for later human review.
    """
    from scipy.optimize import least_squares

    period_days = float(period_days)
    if not math.isfinite(period_days) or period_days <= 0:
        raise ValueError("period_days must be finite and positive")
    if period_uncertainty_days is not None:
        period_uncertainty_days = float(period_uncertainty_days)
        if not math.isfinite(period_uncertainty_days) or period_uncertainty_days <= 0:
            raise ValueError("period_uncertainty_days must be finite and positive when provided")

    record = load_radial_velocity_observations(workspace)
    input_path = _observation_path(workspace)
    input_hash = _sha256(input_path)
    time, velocity, uncertainty, labels, instruments = _observation_arrays(record)
    instrument_count = len(instruments)
    keplerian_parameter_count = instrument_count + 4
    if time.size <= keplerian_parameter_count:
        raise ValueError("RV fit requires more observations than Keplerian free parameters")

    reference_time_bjd_tdb = float(np.median(time))
    constant_prediction, constant_offsets, constant_covariance = _constant_model(
        velocity, uncertainty, labels, instrument_count
    )
    constant_residual = velocity - constant_prediction
    constant_statistics = _model_statistics(constant_residual, uncertainty, instrument_count)
    design = _instrument_design(labels, instrument_count)
    velocity_span = float(np.ptp(velocity))
    initial_amplitude = max(float(np.std(velocity) * math.sqrt(2.0)), float(np.min(uncertainty)))
    lower_log_amplitude = math.log(float(np.min(uncertainty)) * 1e-3)
    upper_log_amplitude = math.log(max(velocity_span * 100.0, float(np.max(uncertainty)) * 100.0))
    start = np.concatenate(
        (
            constant_offsets,
            np.asarray([math.log(initial_amplitude), 0.0, 0.0, 0.0]),
        )
    )
    lower = np.concatenate(
        (
            np.full(instrument_count, -np.inf),
            np.asarray([lower_log_amplitude, -_TAU, -np.inf, -np.inf]),
        )
    )
    upper = np.concatenate(
        (
            np.full(instrument_count, np.inf),
            np.asarray([upper_log_amplitude, _TAU, np.inf, np.inf]),
        )
    )

    def residuals(theta: np.ndarray) -> np.ndarray:
        amplitude = math.exp(float(theta[instrument_count]))
        mean_anomaly = float(theta[instrument_count + 1])
        eccentricity, argument, _, _ = _eccentricity_components(theta, instrument_count + 2)
        model = design @ theta[:instrument_count] + keplerian_velocity_m_per_s(
            time,
            amplitude,
            mean_anomaly,
            eccentricity,
            argument,
            reference_time_bjd_tdb,
            period_days,
        )
        return (velocity - model) / uncertainty

    result = None
    for mean_anomaly_start in np.linspace(-math.pi, math.pi, 9):
        trial_start = start.copy()
        trial_start[instrument_count + 1] = mean_anomaly_start
        trial = least_squares(residuals, trial_start, bounds=(lower, upper), method="trf")
        if result is None or trial.cost < result.cost:
            result = trial
    if result is None:
        raise RuntimeError("Keplerian RV optimization did not produce a result")
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError("Keplerian RV optimization did not converge: {0}".format(result.message))
    fitted_residual = residuals(result.x)
    fitted_prediction = velocity - fitted_residual * uncertainty
    keplerian_statistics = _model_statistics(
        velocity - fitted_prediction, uncertainty, keplerian_parameter_count
    )
    degrees_of_freedom = int(time.size - keplerian_parameter_count)
    covariance = np.linalg.pinv(result.jac.T @ result.jac)
    covariance *= keplerian_statistics["chi_squared"] / degrees_of_freedom
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    amplitude = math.exp(float(result.x[instrument_count]))
    amplitude_uncertainty = _finite_uncertainty(amplitude * standard_errors[instrument_count])
    eccentricity, argument, first_component, second_component = _eccentricity_components(
        result.x, instrument_count + 2
    )
    norm_squared = first_component * first_component + second_component * second_component
    eccentricity_gradient = np.zeros(keplerian_parameter_count)
    eccentricity_gradient[instrument_count + 2] = 1.9 * first_component / (1.0 + norm_squared) ** 2
    eccentricity_gradient[instrument_count + 3] = 1.9 * second_component / (1.0 + norm_squared) ** 2
    eccentricity_uncertainty = _finite_uncertainty(
        math.sqrt(max(float(eccentricity_gradient @ covariance @ eccentricity_gradient), 0.0))
    )
    argument_uncertainty: Optional[float] = None
    if norm_squared > 1e-12:
        argument_gradient = np.zeros(keplerian_parameter_count)
        argument_gradient[instrument_count + 2] = -second_component / norm_squared
        argument_gradient[instrument_count + 3] = first_component / norm_squared
        argument_uncertainty = _finite_uncertainty(
            math.degrees(math.sqrt(max(float(argument_gradient @ covariance @ argument_gradient), 0.0)))
        )
    instrument_parameters = []
    for index, instrument in enumerate(instruments):
        instrument_parameters.append(
            {
                "instrument": instrument,
                "systemic_velocity": _parameter(
                    float(result.x[index]), _finite_uncertainty(standard_errors[index]), "m/s"
                ),
            }
        )
    input_artifact = {
        "path": input_path.relative_to(workspace.path).as_posix(),
        "sha256": input_hash,
        "role": "radial-velocity-observations",
    }
    report = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": "keplerian-rv-model-comparison",
        "observation_time_standard": "BJD_TDB",
        "velocity_unit": "m/s",
        "input_artifacts": [input_artifact],
        "fixed_orbital_inputs": {
            "period": _parameter(period_days, period_uncertainty_days, "days"),
            "reference_time": _parameter(reference_time_bjd_tdb, None, "BJD_TDB"),
        },
        "models": {
            "constant": {
                "model": "instrument-specific constant systemic velocities",
                **constant_statistics,
            },
            "keplerian": {
                "model": "single-companion eccentric Keplerian with instrument systemic velocities",
                **keplerian_statistics,
                "parameters": {
                    "semi_amplitude": _parameter(amplitude, amplitude_uncertainty, "m/s"),
                    "eccentricity": _parameter(eccentricity, eccentricity_uncertainty, "dimensionless"),
                    "argument_periastron": _parameter(math.degrees(argument), argument_uncertainty, "deg"),
                    "mean_anomaly_reference": _parameter(
                        float(result.x[instrument_count + 1]),
                        _finite_uncertainty(standard_errors[instrument_count + 1]),
                        "rad",
                    ),
                    "systemic_velocities": instrument_parameters,
                },
            },
        },
        "model_comparison": {
            "reference_model": "constant",
            "alternative_model": "keplerian",
            "delta_bic_constant_minus_keplerian": float(
                constant_statistics["bic"] - keplerian_statistics["bic"]
            ),
            "delta_aic_constant_minus_keplerian": float(
                constant_statistics["aic"] - keplerian_statistics["aic"]
            ),
            "interpretation": "Information-criterion differences are descriptive model-comparison evidence, not validation probabilities.",
        },
        "diagnostics": {
            "observation_count": int(time.size),
            "instrument_count": instrument_count,
            "degrees_of_freedom": degrees_of_freedom,
            "optimizer": "scipy.optimize.least_squares method=trf",
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
        },
        "caveat": "This candidate-local RV fit is non-claim evidence and does not determine scientific disposition or lifecycle state.",
    }
    output_path = workspace.path / "outputs" / RV_FIT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_digest = hashlib.sha256(
        (input_hash + json.dumps(report["fixed_orbital_inputs"], sort_keys=True)).encode("utf-8")
    ).hexdigest()
    run_id = "fit-" + run_digest[:16]
    run_dir = workspace.path / "runs" / RV_ENGINE_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    completed_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "engine": RV_ENGINE_NAME,
        "run_id": run_id,
        "status": "succeeded",
        "started_at": completed_at,
        "completed_at": completed_at,
        "runtime": {"kind": "direct", "version": __version__, "executable": "scipy"},
        "inputs": [input_artifact],
        "outputs": [
            {
                "path": output_path.relative_to(workspace.path).as_posix(),
                "sha256": _sha256(output_path),
                "role": "keplerian-rv-model-comparison",
            }
        ],
    }
    (run_dir / "engine-run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path
