"""Preserve candidate-local radial velocities and compare fixed-period models.

Observation times use BJD_TDB days; velocities and quoted uncertainties use
metres per second.  The Keplerian component follows
``v(t) = gamma + K [cos(nu(t) + omega) + e cos(omega)]`` after solving
Kepler's equation.  The fitting path compares that component with a
nuisance-controlled constant model that shares instrument offsets, jitter,
linear trend, and an optional homogeneous activity indicator.

Information-criterion differences are descriptive model-comparison evidence.
They are not false-positive probabilities, a companion-mass inference, a
planet claim, or a lifecycle decision.
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
KEPLER_SOLVER_TOLERANCE_RAD = 1e-12
KEPLER_SOLVER_MAX_ITERATIONS = 64
MAXIMUM_FIT_ECCENTRICITY = 0.95
RV_MODEL_CONFIGURATION = "instrument-jitter-linear-trend-optional-activity-v1"


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
    """Validate and atomically canonicalize one candidate-owned RV record.

    Args:
        workspace (CandidateWorkspace): Registered workspace that will own the
            canonical observation record.
        source_path (Path): Regular UTF-8 JSON file to validate against the RV
            observation schema before copying it into the workspace.

    Returns:
        Path: Canonical candidate-local RV observation path.  It is consumed by
        :func:`fit_radial_velocity` and retains the schema-normalized units.

    Raises:
        ValueError: The source is not a bounded regular JSON file, has duplicate
            keys or non-finite numbers, violates the schema, or belongs to a
            different workspace.
        RuntimeError: JSON Schema validation support is unavailable.
        OSError: The validated record cannot be atomically written.
    """
    record = _validate_observation_record(workspace, _read_safe_json(Path(source_path)))
    destination = _observation_path(workspace)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_radial_velocity_observations(workspace: CandidateWorkspace) -> Dict[str, Any]:
    """Load and revalidate the canonical candidate-local RV record.

    Args:
        workspace (CandidateWorkspace): Workspace that owns the canonical
            BJD_TDB and metres-per-second observation record.

    Returns:
        Dict[str, Any]: Schema-valid observation mapping, including candidate
        identity, instrument labels, times, velocities, and uncertainties.

    Raises:
        FileNotFoundError: No RV observation record has been ingested.
        ValueError: The stored JSON is malformed, non-finite, schema-invalid,
            or no longer matches the workspace identity.
        RuntimeError: JSON Schema validation support is unavailable.
    """
    path = _observation_path(workspace)
    if not path.is_file():
        raise FileNotFoundError("no candidate-local RV observations have been ingested")
    return _validate_observation_record(workspace, _read_safe_json(path))


def _solve_kepler_equation(
    mean_anomaly_rad: np.ndarray,
    eccentricity: float,
    tolerance_rad: float = KEPLER_SOLVER_TOLERANCE_RAD,
    max_iterations: int = KEPLER_SOLVER_MAX_ITERATIONS,
) -> np.ndarray:
    """Solve ``E - e sin(E) = M`` to a checked residual tolerance.

    A fixed number of Newton iterations can silently return a non-solution for
    high eccentricity or an unfortunate starting phase.  This vectorized
    solver uses a Danby (1988) starter and Halley third-order iterations,
    rejects non-finite inputs, and raises if every requested cadence has
    not converged to the declared angular residual tolerance.
    """
    mean_anomaly = np.asarray(mean_anomaly_rad, dtype=float)
    if (
        not 0 <= eccentricity < 1
        or not math.isfinite(eccentricity)
        or not math.isfinite(tolerance_rad)
        or tolerance_rad <= 0
        or isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
        or not np.all(np.isfinite(mean_anomaly))
    ):
        raise ValueError("Kepler equation inputs are outside their physical or numerical bounds")
    if mean_anomaly.size == 0:
        return mean_anomaly.copy()

    reduced_mean_anomaly = np.mod(mean_anomaly, _TAU)
    # Danby (1988) cubic starter: improves first-order convergence for high e
    # and near-periastron phases relative to the naive E0 = M seed.
    eccentric_anomaly = reduced_mean_anomaly + 0.85 * eccentricity * np.sign(np.sin(reduced_mean_anomaly))
    for _ in range(max_iterations):
        sin_e = np.sin(eccentric_anomaly)
        cos_e = np.cos(eccentric_anomaly)
        residual = eccentric_anomaly - eccentricity * sin_e - reduced_mean_anomaly
        derivative = 1.0 - eccentricity * cos_e
        # Halley's third-order Householder correction for cubic convergence.
        correction = residual / (derivative - 0.5 * eccentricity * sin_e * residual / derivative)
        eccentric_anomaly -= correction
        if np.all(np.isfinite(correction)) and float(np.max(np.abs(correction))) <= tolerance_rad:
            break
    final_residual = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - reduced_mean_anomaly
    if not np.all(np.isfinite(final_residual)) or float(np.max(np.abs(final_residual))) > tolerance_rad:
        raise RuntimeError(
            "Kepler equation did not converge within {0} iterations to {1:.1e} rad".format(
                max_iterations, tolerance_rad
            )
        )
    return eccentric_anomaly


def keplerian_velocity_m_per_s(
    time_bjd_tdb: Sequence[float],
    semi_amplitude_m_per_s: float,
    mean_anomaly_reference_rad: float,
    eccentricity: float,
    argument_periastron_rad: float,
    reference_time_bjd_tdb: float,
    period_days: float,
) -> np.ndarray:
    """Evaluate the stellar Keplerian radial-velocity component in m/s.

    Mathematical Formulation:
        Mean anomaly advances as ``M(t) = M_ref + 2 pi (t - t_ref) / P``.
        The solver obtains eccentric anomaly from ``M = E - e sin(E)`` and
        evaluates ``K [cos(nu + omega) + e cos(omega)]``.  This is the
        fixed-period eccentric model documented in the RV comparison method.

    Args:
        time_bjd_tdb (Sequence[float]): Finite observation times in BJD_TDB
            days.
        semi_amplitude_m_per_s (float): Non-negative velocity semi-amplitude
            in metres per second.
        mean_anomaly_reference_rad (float): Mean anomaly at the reference time
            in radians.
        eccentricity (float): Dimensionless eccentricity in the bound-orbit
            interval from zero up to, but excluding, one.
        argument_periastron_rad (float): Stellar-orbit periastron argument in
            radians.
        reference_time_bjd_tdb (float): Epoch of the reference mean anomaly in
            BJD_TDB days.
        period_days (float): Positive fixed orbital period in days.

    Returns:
        np.ndarray: Model radial velocities in metres per second, aligned with
        the supplied observation times.

    Raises:
        ValueError: Times or scalar parameters are non-finite, the period is
            not positive, or the eccentricity is outside the bound-orbit range.
        RuntimeError: Newton iteration does not meet its declared angular
            residual tolerance at every cadence.
    """
    scalar_values = (
        semi_amplitude_m_per_s,
        mean_anomaly_reference_rad,
        eccentricity,
        argument_periastron_rad,
        reference_time_bjd_tdb,
        period_days,
    )
    if (
        not all(math.isfinite(float(value)) for value in scalar_values)
        or period_days <= 0
        or not 0 <= eccentricity < 1
        or semi_amplitude_m_per_s < 0
    ):
        raise ValueError("Keplerian parameters are outside their physical bounds")
    time = np.asarray(time_bjd_tdb, dtype=float)
    if not np.all(np.isfinite(time)):
        raise ValueError("Keplerian observation times must be finite")
    mean_anomaly = (
        mean_anomaly_reference_rad
        + _TAU * (time - reference_time_bjd_tdb) / period_days
    )
    eccentric_anomaly = _solve_kepler_equation(mean_anomaly, eccentricity)
    true_anomaly = 2.0 * np.arctan2(
        math.sqrt(1.0 + eccentricity) * np.sin(eccentric_anomaly / 2.0),
        math.sqrt(1.0 - eccentricity) * np.cos(eccentric_anomaly / 2.0),
    )
    return semi_amplitude_m_per_s * (
        np.cos(true_anomaly + argument_periastron_rad)
        + eccentricity * math.cos(argument_periastron_rad)
    )


def _observation_arrays(
    record: Dict[str, Any]
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Optional[np.ndarray],
    Optional[str],
]:
    """Return validated numeric RV arrays and an optional common activity index."""
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
    activity_records = [item.get("activity_indicator") for item in observations]
    if any(item is not None for item in activity_records) and not all(
        item is not None for item in activity_records
    ):
        raise ValueError("RV activity indicators must be supplied for every observation or none")
    if not activity_records or activity_records[0] is None:
        return time, velocity, uncertainty, labels, instruments, None, None

    activity_values = np.asarray([item["value"] for item in activity_records], dtype=float)
    activity_units = {str(item["unit"]) for item in activity_records}
    if not np.all(np.isfinite(activity_values)) or len(activity_units) != 1:
        raise ValueError("RV activity indicators must be finite and use one common unit")
    activity_scale = float(np.std(activity_values))
    if activity_scale <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(activity_values)))):
        raise ValueError("RV activity indicators must vary to support a joint regression")
    return time, velocity, uncertainty, labels, instruments, activity_values, activity_units.pop()


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


def _parameter(value: Optional[float], uncertainty: Optional[float], unit: str) -> Dict[str, Any]:
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


def _effective_uncertainty(
    uncertainty: np.ndarray, labels: np.ndarray, log_jitters: np.ndarray
) -> np.ndarray:
    """Combine quoted uncertainties with non-negative per-instrument jitter."""
    jitter = np.exp(np.asarray(log_jitters, dtype=float))
    if not np.all(np.isfinite(jitter)):
        raise ValueError("RV jitter parameters are non-finite")
    return np.hypot(uncertainty, jitter[labels])


def _negative_log_likelihood(residual: np.ndarray, effective_uncertainty: np.ndarray) -> float:
    """Return the Gaussian RV negative log likelihood with its normalization."""
    if np.any(effective_uncertainty <= 0) or not np.all(np.isfinite(effective_uncertainty)):
        return math.inf
    standardized = residual / effective_uncertainty
    value = 0.5 * np.sum(standardized**2 + np.log(_TAU * effective_uncertainty**2))
    return float(value) if math.isfinite(float(value)) else math.inf


def _inverse_hessian_covariance(result: Any, parameter_count: int) -> np.ndarray:
    """Extract a finite local inverse-Hessian covariance approximation."""
    inverse_hessian = getattr(result, "hess_inv", None)
    if inverse_hessian is None:
        return np.full((parameter_count, parameter_count), np.nan)
    dense = inverse_hessian.todense() if hasattr(inverse_hessian, "todense") else inverse_hessian
    covariance = np.asarray(dense, dtype=float)
    if covariance.shape != (parameter_count, parameter_count) or not np.all(np.isfinite(covariance)):
        return np.full((parameter_count, parameter_count), np.nan)
    return 0.5 * (covariance + covariance.T)


def _eccentricity_components(theta: np.ndarray, component_start: int) -> Tuple[float, float, float, float]:
    first = float(theta[component_start])
    second = float(theta[component_start + 1])
    norm_squared = first * first + second * second
    # NUMERICAL_GUARD: This bounded Cartesian parameterization keeps optimizer
    # proposals inside the elliptic-orbit domain required by Kepler's equation.
    eccentricity = MAXIMUM_FIT_ECCENTRICITY * norm_squared / (1.0 + norm_squared)
    argument = math.atan2(second, first) if norm_squared > 0 else 0.0
    return eccentricity, argument, first, second


def _finite_uncertainty(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) and value >= 0 else None


def fit_radial_velocity(
    workspace: CandidateWorkspace,
    period_days: float,
    period_uncertainty_days: Optional[float] = None,
) -> Path:
    """Compare constant and eccentric Keplerian RV models at a fixed period.

    Mathematical Formulation:
        Each model uses ``sigma_eff**2 = sigma_quoted**2 + jitter**2`` in an
        independent Gaussian likelihood.  The Keplerian adds the eccentric
        velocity component while retaining the constant model's instrument
        offsets, linear trend, and optional activity regression.  The report
        computes AIC and BIC from those likelihoods, including
        ``Delta BIC = BIC_constant - BIC_keplerian``.

    Astrophysical Rationale:
        Shared nuisance terms reduce the chance that a Keplerian merely absorbs
        an instrument offset, drift, contemporaneous activity correlation, or
        underestimated quoted uncertainty.

    Args:
        workspace (CandidateWorkspace): Workspace containing validated
            candidate-local RV observations.
        period_days (float): Positive fixed period in days.  This function does
            not search or marginalize over a period grid.
        period_uncertainty_days (Optional[float]): Positive reported input
            uncertainty in days, retained as provenance when available.

    Returns:
        Path: Candidate-local ``outputs/rv_keplerian_fit.json`` with input
        hashes, formal local covariance summaries, model statistics, and a run
        manifest.

    Raises:
        ValueError: The period, observations, activity indicators, or model
            dimensionality are invalid.
        FileNotFoundError: Validated candidate-local RV observations are absent.
        RuntimeError: The Keplerian solver, JSON Schema support, or optimizer
            cannot produce a valid fit.

    Note:
        The fit uses a local inverse-Hessian uncertainty approximation and
        independent residual likelihood.  It does not model correlated noise,
        additional companions, or establish a validation probability.
    """
    from scipy.optimize import minimize

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
    (
        time,
        velocity,
        uncertainty,
        labels,
        instruments,
        activity_values,
        activity_unit,
    ) = _observation_arrays(record)
    instrument_count = len(instruments)
    activity_parameter_count = 1 if activity_values is not None else 0
    nuisance_parameter_count = instrument_count + 1 + activity_parameter_count
    constant_parameter_count = nuisance_parameter_count + instrument_count
    keplerian_parameter_count = nuisance_parameter_count + 4 + instrument_count
    if time.size <= keplerian_parameter_count:
        raise ValueError("RV fit requires more observations than Keplerian free parameters")

    reference_time_bjd_tdb = float(np.median(time))
    centered_time_days = time - reference_time_bjd_tdb
    constant_prediction, constant_offsets, constant_covariance = _constant_model(
        velocity, uncertainty, labels, instrument_count
    )
    design = _instrument_design(labels, instrument_count)
    del constant_prediction, constant_covariance
    if activity_values is None:
        standardized_activity = None
        activity_center = None
        activity_scale = None
    else:
        activity_center = float(np.median(activity_values))
        activity_scale = float(np.std(activity_values))
        standardized_activity = (activity_values - activity_center) / activity_scale
    velocity_span = float(np.ptp(velocity))
    initial_amplitude = max(float(np.std(velocity) * math.sqrt(2.0)), float(np.min(uncertainty)))
    lower_log_amplitude = math.log(float(np.min(uncertainty)) * 1e-3)
    upper_log_amplitude = math.log(max(velocity_span * 100.0, float(np.max(uncertainty)) * 100.0))
    lower_log_jitter = math.log(float(np.min(uncertainty)) * 1e-6)
    upper_log_jitter = math.log(max(velocity_span * 100.0, float(np.max(uncertainty)) * 100.0))
    initial_log_jitters = np.full(instrument_count, math.log(float(np.min(uncertainty)) * 1e-3))
    common_start = np.concatenate(
        (constant_offsets, np.zeros(1 + activity_parameter_count), initial_log_jitters)
    )
    common_bounds = [(None, None)] * (instrument_count + 1 + activity_parameter_count) + [
        (lower_log_jitter, upper_log_jitter)
    ] * instrument_count

    def constant_components(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        trend = float(theta[instrument_count])
        activity_coefficient = (
            float(theta[instrument_count + 1]) if standardized_activity is not None else 0.0
        )
        jitter_start = nuisance_parameter_count
        prediction = design @ theta[:instrument_count] + trend * centered_time_days
        if standardized_activity is not None:
            prediction = prediction + activity_coefficient * standardized_activity
        effective_uncertainty = _effective_uncertainty(
            uncertainty, labels, theta[jitter_start : jitter_start + instrument_count]
        )
        return prediction, effective_uncertainty, jitter_start

    def constant_negative_log_likelihood(theta: np.ndarray) -> float:
        prediction, effective_uncertainty, _ = constant_components(theta)
        return _negative_log_likelihood(velocity - prediction, effective_uncertainty)

    keplerian_start = np.concatenate(
        (
            common_start[:nuisance_parameter_count],
            np.asarray([math.log(initial_amplitude), 0.0, 0.0, 0.0]),
            common_start[nuisance_parameter_count:],
        )
    )
    keplerian_bounds = (
        common_bounds[:nuisance_parameter_count]
        + [(lower_log_amplitude, upper_log_amplitude), (-_TAU, _TAU), (None, None), (None, None)]
        + common_bounds[nuisance_parameter_count:]
    )

    def keplerian_components(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        amplitude_index = nuisance_parameter_count
        amplitude = math.exp(float(theta[amplitude_index]))
        mean_anomaly = float(theta[amplitude_index + 1])
        eccentricity, argument, _, _ = _eccentricity_components(theta, amplitude_index + 2)
        trend = float(theta[instrument_count])
        activity_coefficient = (
            float(theta[instrument_count + 1]) if standardized_activity is not None else 0.0
        )
        jitter_start = amplitude_index + 4
        prediction = (
            design @ theta[:instrument_count]
            + trend * centered_time_days
            + keplerian_velocity_m_per_s(
                time,
                amplitude,
                mean_anomaly,
                eccentricity,
                argument,
                reference_time_bjd_tdb,
                period_days,
            )
        )
        if standardized_activity is not None:
            prediction = prediction + activity_coefficient * standardized_activity
        effective_uncertainty = _effective_uncertainty(
            uncertainty, labels, theta[jitter_start : jitter_start + instrument_count]
        )
        return prediction, effective_uncertainty, jitter_start

    def keplerian_negative_log_likelihood(theta: np.ndarray) -> float:
        prediction, effective_uncertainty, _ = keplerian_components(theta)
        return _negative_log_likelihood(velocity - prediction, effective_uncertainty)

    def optimize_model(
        objective: Any,
        start_values: np.ndarray,
        bounds: Sequence[Tuple[Optional[float], Optional[float]]],
        phase_index: Optional[int],
    ) -> Any:
        """Run deterministic phase starts and return the best converged likelihood fit."""
        best_result = None
        phase_starts = np.linspace(-math.pi, math.pi, 9) if phase_index is not None else [0.0]
        for phase_start in phase_starts:
            trial_start = start_values.copy()
            if phase_index is not None:
                trial_start[phase_index] = phase_start
            trial = minimize(
                objective,
                trial_start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if (
                trial.success
                and np.all(np.isfinite(trial.x))
                and math.isfinite(float(trial.fun))
                and (best_result is None or float(trial.fun) < float(best_result.fun))
            ):
                best_result = trial
        if best_result is None:
            raise RuntimeError("Keplerian RV optimization did not converge from any deterministic phase start")
        return best_result

    constant_result = optimize_model(constant_negative_log_likelihood, common_start, common_bounds, None)
    result = optimize_model(
        keplerian_negative_log_likelihood,
        keplerian_start,
        keplerian_bounds,
        nuisance_parameter_count + 1,
    )
    constant_prediction, constant_effective_uncertainty, constant_jitter_start = constant_components(
        constant_result.x
    )
    fitted_prediction, fitted_effective_uncertainty, keplerian_jitter_start = keplerian_components(result.x)
    constant_statistics = _model_statistics(
        velocity - constant_prediction, constant_effective_uncertainty, constant_parameter_count
    )
    keplerian_statistics = _model_statistics(
        velocity - fitted_prediction, fitted_effective_uncertainty, keplerian_parameter_count
    )
    degrees_of_freedom = int(time.size - keplerian_parameter_count)
    covariance = _inverse_hessian_covariance(result, keplerian_parameter_count)
    constant_covariance = _inverse_hessian_covariance(constant_result, constant_parameter_count)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    constant_standard_errors = np.sqrt(np.clip(np.diag(constant_covariance), 0.0, np.inf))
    amplitude_index = nuisance_parameter_count
    amplitude = math.exp(float(result.x[amplitude_index]))
    amplitude_uncertainty = _finite_uncertainty(amplitude * standard_errors[amplitude_index])
    eccentricity, argument, first_component, second_component = _eccentricity_components(
        result.x, amplitude_index + 2
    )
    norm_squared = first_component * first_component + second_component * second_component
    eccentricity_gradient = np.zeros(keplerian_parameter_count)
    eccentricity_gradient[amplitude_index + 2] = 1.9 * first_component / (1.0 + norm_squared) ** 2
    eccentricity_gradient[amplitude_index + 3] = 1.9 * second_component / (1.0 + norm_squared) ** 2
    eccentricity_uncertainty = _finite_uncertainty(
        math.sqrt(max(float(eccentricity_gradient @ covariance @ eccentricity_gradient), 0.0))
    )
    argument_uncertainty: Optional[float] = None
    if norm_squared > 1e-12:
        argument_gradient = np.zeros(keplerian_parameter_count)
        argument_gradient[amplitude_index + 2] = -second_component / norm_squared
        argument_gradient[amplitude_index + 3] = first_component / norm_squared
        argument_uncertainty = _finite_uncertainty(
            math.degrees(math.sqrt(max(float(argument_gradient @ covariance @ argument_gradient), 0.0)))
        )

    def nuisance_parameters(
        theta: np.ndarray,
        errors: np.ndarray,
        jitter_start: int,
    ) -> Dict[str, Any]:
        """Serialize shared offsets, trend, optional activity, and jitter parameters."""
        systemic_velocities = []
        instrument_jitters = []
        for index, instrument in enumerate(instruments):
            systemic_velocities.append(
                {
                    "instrument": instrument,
                    "systemic_velocity": _parameter(
                        float(theta[index]), _finite_uncertainty(errors[index]), "m/s"
                    ),
                }
            )
            jitter = math.exp(float(theta[jitter_start + index]))
            instrument_jitters.append(
                {
                    "instrument": instrument,
                    "jitter": _parameter(
                        jitter,
                        _finite_uncertainty(jitter * errors[jitter_start + index]),
                        "m/s",
                    ),
                }
            )
        if activity_values is None:
            activity_parameter = _parameter(None, None, "m/s per unavailable activity index")
            activity_parameter["source"] = "not fitted because no activity indicator was supplied"
        else:
            activity_index = instrument_count + 1
            activity_parameter = _parameter(
                float(theta[activity_index] / activity_scale),
                _finite_uncertainty(errors[activity_index] / activity_scale),
                "m/s per {0}".format(activity_unit),
            )
        return {
            "systemic_velocities": systemic_velocities,
            "linear_trend": _parameter(
                float(theta[instrument_count]),
                _finite_uncertainty(errors[instrument_count]),
                "m/s/day",
            ),
            "activity_coefficient": activity_parameter,
            "instrument_jitters": instrument_jitters,
        }

    constant_nuisance_parameters = nuisance_parameters(
        constant_result.x, constant_standard_errors, constant_jitter_start
    )
    keplerian_nuisance_parameters = nuisance_parameters(
        result.x, standard_errors, keplerian_jitter_start
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
                "model": "instrument systemic velocities with jitter, linear trend, and optional activity regression",
                **constant_statistics,
                "parameters": constant_nuisance_parameters,
            },
            "keplerian": {
                "model": "single-companion eccentric Keplerian with shared jitter, linear trend, and optional activity regression",
                **keplerian_statistics,
                "parameters": {
                    "semi_amplitude": _parameter(amplitude, amplitude_uncertainty, "m/s"),
                    "eccentricity": _parameter(eccentricity, eccentricity_uncertainty, "dimensionless"),
                    "argument_periastron": _parameter(math.degrees(argument), argument_uncertainty, "deg"),
                    "mean_anomaly_reference": _parameter(
                        float(result.x[amplitude_index + 1]),
                        _finite_uncertainty(standard_errors[amplitude_index + 1]),
                        "rad",
                    ),
                    **keplerian_nuisance_parameters,
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
            "optimizer": "scipy.optimize.minimize method=L-BFGS-B with deterministic mean-anomaly starts",
            "model_configuration": RV_MODEL_CONFIGURATION,
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "constant_optimizer_status": int(constant_result.status),
            "constant_optimizer_message": str(constant_result.message),
            "uncertainty_estimation": "local inverse-Hessian approximation to the full Gaussian likelihood; it does not include model-selection or activity-model uncertainty",
            "noise_model": "quoted per-observation uncertainty combined in quadrature with fitted per-instrument jitter",
            "activity_regression": {
                "status": "jointly-fitted" if activity_values is not None else "not-provided",
                "unit": activity_unit,
                "standardization": (
                    "median-centered and standard-deviation scaled before fitting"
                    if activity_values is not None
                    else "not applicable"
                ),
            },
            "kepler_equation_solver": {
                "method": "Danby starter + Halley third-order iteration with residual convergence check",
                "tolerance_rad": KEPLER_SOLVER_TOLERANCE_RAD,
                "max_iterations": KEPLER_SOLVER_MAX_ITERATIONS,
            },
            "eccentricity_parameterization": {
                "coordinates": "unbounded Cartesian optimizer coordinates (x, y)",
                "mapping": "e = 0.95 * (x^2 + y^2) / (1 + x^2 + y^2)",
                "maximum_eccentricity_exclusive": MAXIMUM_FIT_ECCENTRICITY,
                "numerical_rationale": (
                    "The bounded mapping keeps all optimizer proposals away from the "
                    "parabolic e=1 boundary, where Kepler solving and local curvature "
                    "estimates become poorly conditioned."
                ),
                "scientific_limitation": (
                    "This is a numerical support restriction, not an astrophysical "
                    "eccentricity prior; fits requiring e near or above 0.95 need a "
                    "separate model and review."
                ),
            },
        },
        "caveat": "This candidate-local RV fit is non-claim evidence and does not determine scientific disposition or lifecycle state. The optional activity term tests only a contemporaneous linear correlation in the supplied indicator; it is not an activity model, and the jitter/trend terms do not establish a companion.",
    }
    output_path = workspace.path / "outputs" / RV_FIT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    run_digest = hashlib.sha256(
        (
            input_hash
            + RV_MODEL_CONFIGURATION
            + json.dumps(report["fixed_orbital_inputs"], sort_keys=True, allow_nan=False)
        ).encode("utf-8")
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
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path
