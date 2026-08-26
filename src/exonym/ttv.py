"""Target-neutral transit timing variation (TTV / O-C) engine.

Measures per-transit timing deviations (O - C) from a refitted linear ephemeris
for diagnostic investigation of possible gravitational perturbations from
non-transiting companions and resonant multi-planet systems
(Agol et al. 2005, Holman & Murray 2005):

1. Linear Ephemeris Reference:
   T_calc(N) = T_0 + N * P_orb
   where N is the integer transit epoch cycle.

2. Observed-minus-Calculated (O - C) Residuals:
   (O - C)_N = (T_obs(N) - T_calc(N)) * 1440.0   [minutes]
   derived via template cross-correlation with a Mandel & Agol (2002) transit profile.

3. First-Order Mean Motion Resonance (MMR) Super-Periods (Lithwick, Xie & Wu 2012):
   When two planets orbit near a j:(j-1) resonance, gravitational interactions induce
   sinusoidal TTVs with super-period:
       P_super = 1 / | j / P_outer - (j - 1) / P_inner |   [days]

Contains zero candidate-specific identifiers; all timing fits operate dynamically
on the candidate's light curve table and declared ephemeris priors.

Scientific Boundary:
    Per-epoch timings and resonance super-periods are exploratory diagnostics.
    They do not establish a TTV detection, a perturber, a planet claim, or a
    candidate lifecycle transition.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from scipy.optimize import least_squares

from .constants import JULIAN_YEAR_DAYS, SECONDS_PER_DAY
from .inputs import load_light_curve_table, load_stellar_parameters, load_transit_ephemeris
from .lightcurve import kipping_to_quadratic_limb_darkening
from .search import calculate_ttv_super_period
from .transit_fit import stellar_density_a_rs
from .workspace import CandidateWorkspace, validate_signal_suffix

MIN_POINTS_PER_TRANSIT = 30     # Minimum photometric cadences required to fit a single transit epoch
WINDOW_DAYS = 0.35              # Half-width of individual transit isolation window (days)
GRID_HALF_WINDOW_DAYS = 0.02    # Local grid search range around calculated transit time (days)
GRID_STEP_DAYS = 0.001          # Fine grid resolution for initial transit center search (days)
MIN_EPOCH_DEPTH_SNR = 3.0       # Formal local-template detection threshold
MIN_TIMING_SIGMA_DAYS = 0.0005  # Lower reporting bound for a finite timing error
MAX_TIMING_SIGMA_DAYS = 0.05    # Upper reporting bound for a finite timing error
MIN_ORBITAL_DECAY_TRANSITS = 4  # Leaves one quadratic-fit residual degree of freedom
TTV_TEMPLATE_WORK_PACKAGES = ("MCMC_TRANSIT_FIT", "NESTED_TRANSIT_FIT")
_ECCENTRIC_TEMPLATE_PARAMETERS = frozenset(("sqe_cosw", "sqe_sinw"))


def _parse_finite_json_float(value: str) -> float:
    """Parse a JSON float while rejecting overflowed non-finite values."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_nonfinite_json_constant(value: str) -> object:
    """Reject non-standard JSON numeric constants such as NaN and Infinity."""
    raise ValueError("non-finite JSON constant: {0}".format(value))


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    """Reject ambiguous JSON objects instead of retaining the last duplicate."""
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {0}".format(key))
        result[key] = value
    return result


def _posterior_median(
    posterior: Dict[str, Any], parameter: str, artifact_path: str
) -> float:
    """Return one required finite numeric posterior median or fail explicitly."""
    summary = posterior.get(parameter)
    if not isinstance(summary, dict) or "median" not in summary:
        raise ValueError(
            "{0} is missing posterior.{1}.median".format(artifact_path, parameter)
        )
    value = summary["median"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "{0} has a non-numeric posterior.{1}.median".format(artifact_path, parameter)
        )
    median = float(value)
    if not math.isfinite(median):
        raise ValueError(
            "{0} has a non-finite posterior.{1}.median".format(artifact_path, parameter)
        )
    return median


def transit_template_parameters(
    ephemeris: Dict[str, Any],
    a_rs: float,
    impact_parameter: float,
    q1: float,
    q2: float,
) -> Dict[str, Any]:
    """Build the fixed circular transit template for local epoch fitting.

    Mathematical Formulation:
        The radius ratio is initialized from ``rp_rs = sqrt(depth_ppm * 1e-6)``
        from the candidate ephemeris.  The impact parameter and quadratic
        limb-darkening coefficients are transformed from candidate-local
        transit-fit posterior medians.  Each epoch search varies only the
        local transit center.

    Args:
        ephemeris (Dict[str, Any]): Candidate-derived period in days, epoch in
            BTJD, and transit depth in ppm.
        a_rs (float): Dimensionless stellar-density-derived semimajor-axis to
            stellar-radius ratio.
        impact_parameter (float): Candidate-local transit-fit posterior median
            for the circular conjunction impact parameter.
        q1, q2 (float): Candidate-local transit-fit posterior medians in the
            Kipping (2013) quadratic limb-darkening parameterization.

    Returns:
        Dict[str, Any]: Fixed template parameters used by
        :func:`fit_transit_epoch`; limb-darkening coefficients and geometry are
        dimensionless except for the period in days.

    Raises:
        KeyError: Required ephemeris fields are absent.
        ValueError: Required values cannot be converted to a physical, finite
            fixed-template geometry.

    Note:
        This is a timing template, not a full per-epoch transit inference.
    """
    period_days = float(ephemeris["period_days"])
    depth_ppm = float(ephemeris["depth_ppm"])
    a_rs_value = float(a_rs)
    impact_parameter_value = float(impact_parameter)
    q1_value = float(q1)
    q2_value = float(q2)
    if not math.isfinite(period_days) or period_days <= 0.0:
        raise ValueError("template period_days must be finite and positive")
    if not math.isfinite(depth_ppm) or depth_ppm <= 0.0:
        raise ValueError("template depth_ppm must be finite and positive")
    if not math.isfinite(a_rs_value) or a_rs_value <= 0.0:
        raise ValueError("template a_rs must be finite and positive")
    if (
        not math.isfinite(impact_parameter_value)
        or impact_parameter_value < 0.0
        or impact_parameter_value >= a_rs_value
    ):
        raise ValueError("template impact_parameter is outside the circular inclination domain")
    if not (math.isfinite(q1_value) and math.isfinite(q2_value)):
        raise ValueError("template q1 and q2 must be finite")
    if not (0.0 <= q1_value <= 1.0 and 0.0 <= q2_value <= 1.0):
        raise ValueError("template q1 and q2 must be in [0, 1]")

    rp_rs = math.sqrt(depth_ppm * 1e-6)
    u1, u2 = kipping_to_quadratic_limb_darkening(q1_value, q2_value)
    return {
        "period_days": period_days,
        "rp_rs": rp_rs,
        "a_rs": a_rs_value,
        "impact_parameter": impact_parameter_value,
        "q1": q1_value,
        "q2": q2_value,
        "u1": u1,
        "u2": u2,
    }


def _load_transit_fit_template(
    workspace: CandidateWorkspace,
    signal: Optional[str],
    ephemeris: Dict[str, Any],
    a_rs: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load a signal-bound posterior and build a provenance-bound TTV template.

    TTV timing fits must not silently invent geometry or limb darkening.  The
    required candidate-local transit-fit artifact is parsed as strict, finite
    JSON and supplies the fixed impact parameter and Kipping limb-darkening
    medians used for every local timing fit.
    """
    suffix = ".{0}".format(signal.lstrip(".")) if signal else ""
    relative_path = "outputs/mcmc_transit_fit{0}.json".format(suffix)
    artifact_path = workspace.path / relative_path
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise RuntimeError(
            "TTV template requires a candidate-local transit-fit artifact at {0}".format(
                relative_path
            )
        )
    try:
        artifact_bytes = artifact_path.read_bytes()
        payload = json.loads(
            artifact_bytes.decode("utf-8"),
            parse_float=_parse_finite_json_float,
            parse_constant=_reject_nonfinite_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "TTV template requires valid finite JSON in {0}: {1}".format(relative_path, exc)
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("TTV template artifact {0} must contain a JSON object".format(relative_path))
    work_package = payload.get("work_package")
    if work_package not in TTV_TEMPLATE_WORK_PACKAGES:
        raise RuntimeError(
            "TTV template artifact {0} is not a supported transit-fit output".format(relative_path)
        )
    if payload.get("source") != "candidate-data":
        raise RuntimeError(
            "TTV template artifact {0} is not candidate-derived".format(relative_path)
        )
    if payload.get("signal") != signal:
        raise RuntimeError(
            "TTV template artifact {0} is not bound to signal {1!r}".format(relative_path, signal)
        )
    parameter_names = payload.get("parameter_names")
    if not isinstance(parameter_names, list) or not all(
        isinstance(name, str) for name in parameter_names
    ):
        raise RuntimeError(
            "TTV template artifact {0} has no valid parameter_names contract".format(relative_path)
        )
    if _ECCENTRIC_TEMPLATE_PARAMETERS.intersection(parameter_names):
        raise RuntimeError(
            "TTV template artifact {0} uses an eccentric fit, which the circular timing template cannot use".format(
                relative_path
            )
        )
    fitted_ephemeris = payload.get("ephemeris")
    if not isinstance(fitted_ephemeris, dict):
        raise RuntimeError(
            "TTV template artifact {0} has no fitted ephemeris contract".format(relative_path)
        )
    for field in ("period_days", "epoch_btjd"):
        try:
            fitted_value = float(fitted_ephemeris[field])
            current_value = float(ephemeris[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "TTV template artifact {0} has an invalid fitted {1}".format(
                    relative_path, field
                )
            ) from exc
        if not (
            math.isfinite(fitted_value)
            and math.isfinite(current_value)
            and math.isclose(fitted_value, current_value, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise RuntimeError(
                "TTV template artifact {0} has a stale fitted {1}".format(relative_path, field)
            )
    posterior = payload.get("posterior")
    if not isinstance(posterior, dict):
        raise RuntimeError("TTV template artifact {0} has no posterior object".format(relative_path))
    try:
        impact_parameter = _posterior_median(posterior, "impact_parameter", relative_path)
        q1 = _posterior_median(posterior, "q1", relative_path)
        q2 = _posterior_median(posterior, "q2", relative_path)
        template = transit_template_parameters(
            ephemeris,
            a_rs,
            impact_parameter=impact_parameter,
            q1=q1,
            q2=q2,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "TTV template cannot use posterior medians from {0}: {1}".format(relative_path, exc)
        ) from exc

    scientific_status = payload.get("scientific_status")
    validation_eligible = payload.get("validation_eligible")
    return template, {
        "kind": "candidate-local-transit-fit-posterior-median",
        "artifact": {
            "path": relative_path,
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "work_package": work_package,
            "source": "candidate-data",
            "signal": signal,
            "scientific_status": scientific_status if isinstance(scientific_status, str) else None,
            "validation_eligible": (
                validation_eligible if isinstance(validation_eligible, bool) else None
            ),
        },
        "parameters": {
            "impact_parameter": template["impact_parameter"],
            "a_rs": {
                "value": template["a_rs"],
                "source": "candidate-stellar-mass-radius-and-ephemeris-period",
            },
            "rp_rs": {
                "value": template["rp_rs"],
                "source": "candidate-ephemeris-depth-approximation",
            },
            "limb_darkening": {
                "parameterization": "Kipping-2013-q1-q2",
                "q1": template["q1"],
                "q2": template["q2"],
                "u1": template["u1"],
                "u2": template["u2"],
            },
        },
        "model_assumptions": {
            "orbit": "circular",
            "per_epoch_free_parameter": "transit-center-only",
        },
        "limitations": [
            "Impact parameter and limb darkening are fixed at posterior medians; their uncertainty and covariance are not propagated into per-epoch timing errors.",
            "The timing template is circular and uses an ephemeris-depth radius-ratio approximation with stellar-density-derived a_rs.",
            "This template provenance does not convert exploratory timing residuals into a TTV detection or companion inference.",
        ],
    }


def _template_flux(
    template: Dict[str, Any], time: np.ndarray, t0_value: float
) -> Optional[Union[np.ndarray, Dict[str, Any]]]:
    """Evaluate the batman template with the transit center shifted to t0.

    Returns the model light curve array, a per-epoch failure record dict
    with ``status: "failed"`` when the template cannot be built, or None
    when batman is not available.
    """
    try:
        import batman

        params = batman.TransitParams()
        params.t0 = float(t0_value)
        params.per = template["period_days"]
        params.rp = template["rp_rs"]
        params.a = template["a_rs"]
        params.inc = math.degrees(
            math.acos(template["impact_parameter"] / template["a_rs"])
        )
        params.ecc = 0.0
        params.w = 90.0
        params.u = [template["u1"], template["u2"]]
        params.limb_dark = "quadratic"
        model = batman.TransitModel(params, np.asarray(time, dtype=float))
        return np.asarray(model.light_curve(params), dtype=float)
    except Exception as exc:
        logging.warning(
            "TTV template failed for epoch t0=%.6f: %s", float(t0_value), exc
        )
        return {"status": "failed", "reason": str(exc)}


def _fit_local_template_depth(
    flux: np.ndarray,
    flux_err: np.ndarray,
    model: np.ndarray,
) -> Optional[Dict[str, float]]:
    """Fit a local baseline and template-depth scale under independent errors."""
    flux_arr = np.asarray(flux, dtype=float)
    error_arr = np.asarray(flux_err, dtype=float)
    model_arr = np.asarray(model, dtype=float)
    if flux_arr.shape != error_arr.shape or flux_arr.shape != model_arr.shape:
        return None
    template_deficit = 1.0 - model_arr
    valid = (
        np.isfinite(flux_arr)
        & np.isfinite(error_arr)
        & (error_arr > 0)
        & np.isfinite(template_deficit)
    )
    if int(np.count_nonzero(valid)) < 3:
        return None
    flux_valid = flux_arr[valid]
    error_valid = error_arr[valid]
    deficit_valid = template_deficit[valid]
    template_depth_ppm = float(np.max(deficit_valid) * 1e6)
    if not math.isfinite(template_depth_ppm) or template_depth_ppm <= 0:
        return None
    design = np.column_stack((np.ones(deficit_valid.size), -deficit_valid))
    weights = 1.0 / error_valid**2
    normal_matrix = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(normal_matrix)
    except np.linalg.LinAlgError:
        return None
    coefficients = covariance @ (design.T @ (weights * flux_valid))
    depth_scale = float(coefficients[1])
    depth_scale_variance = float(covariance[1, 1])
    if (
        not math.isfinite(depth_scale)
        or not math.isfinite(depth_scale_variance)
        or depth_scale_variance <= 0
    ):
        return None
    depth_scale_uncertainty = float(math.sqrt(depth_scale_variance))
    depth_ppm = float(depth_scale * template_depth_ppm)
    depth_uncertainty_ppm = float(depth_scale_uncertainty * template_depth_ppm)
    if not math.isfinite(depth_ppm) or not math.isfinite(depth_uncertainty_ppm):
        return None
    return {
        "depth_ppm": depth_ppm,
        "depth_uncertainty_ppm": depth_uncertainty_ppm,
        "depth_snr": float(depth_ppm / depth_uncertainty_ppm),
    }


def _rejected_epoch_fit(
    reason: str,
    t0_fit: Optional[float] = None,
    depth_fit: Optional[Dict[str, float]] = None,
    at_search_boundary: bool = False,
) -> Dict[str, Any]:
    """Return a JSON-safe no-timing-fit record with any local depth evidence."""
    result: Dict[str, Any] = {
        "t0_fit": t0_fit,
        "sigma_t0": None,
        "sigma_t0_raw": None,
        "depth_ppm": None,
        "depth_uncertainty_ppm": None,
        "depth_snr": None,
        "excluded_no_detection": True,
        "rejection_reason": reason,
        "at_search_boundary": at_search_boundary,
        "sigma_t0_clipped": False,
    }
    if depth_fit is not None:
        result.update(depth_fit)
    return result


def fit_transit_epoch(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    template: Dict[str, Any],
    t0_expected: float,
    window_days: float = WINDOW_DAYS,
    grid_half_window_days: float = GRID_HALF_WINDOW_DAYS,
    grid_step_days: float = GRID_STEP_DAYS,
) -> Dict[str, Any]:
    """Fit one transit epoch by grid search plus parabolic refinement.

    Mathematical Formulation:
        Each grid point evaluates a fixed-shape transit template.  A weighted
        local baseline plus template-depth scale estimates depth and its formal
        uncertainty; local timing curvature gives the reported formal timing
        uncertainty when the curvature is positive.

    Args:
        time (np.ndarray): Cadence times in BTJD days.
        flux (np.ndarray): Normalized relative flux paired with ``time``.
        flux_err (np.ndarray): Positive normalized-flux uncertainties paired
            with ``time``.
        template (Dict[str, Any]): Fixed circular transit-template parameters.
        t0_expected (float): Expected transit center in BTJD days.
        window_days (float): Half-width of the local fitting window in days.
        grid_half_window_days (float): Half-width of the trial-center grid in
            days.
        grid_step_days (float): Trial-center spacing in days.

    Returns:
        Dict[str, Any]: JSON-safe accepted timing fields or a rejection record
        with local depth evidence, boundary flags, and an explicit reason.

    Note:
        A local depth recovery is required before timing is retained, but its
        formal signal-to-noise threshold is not a calibrated detection
        probability.

    Returns a JSON-safe fit/rejection record. A timed epoch must have a
    positive local template depth with formal depth SNR at least
    ``MIN_EPOCH_DEPTH_SNR``. The reported timing uncertainty is unavailable
    (``None``) when the local timing curvature is non-positive.
    """
    mask = (time > t0_expected - window_days) & (time < t0_expected + window_days)
    t_window = time[mask]
    f_window = flux[mask]
    e_window = flux_err[mask]
    if t_window.size < MIN_POINTS_PER_TRANSIT:
        return _rejected_epoch_fit("insufficient-cadences")
    if (
        not np.all(np.isfinite(t_window))
        or not np.all(np.isfinite(f_window))
        or not np.all(np.isfinite(e_window))
        or np.any(e_window <= 0)
    ):
        return _rejected_epoch_fit("invalid-window-photometry")

    def chi2(t0_trial: float) -> float:
        model = _template_flux(template, t_window, t0_trial)
        if (
            not isinstance(model, np.ndarray)
            or model.shape != f_window.shape
            or not np.all(np.isfinite(model))
        ):
            return 1e100
        return float(np.sum(((f_window - model) / e_window) ** 2))

    trials = np.arange(
        t0_expected - grid_half_window_days,
        t0_expected + grid_half_window_days + grid_step_days,
        grid_step_days,
    )
    values = np.array([chi2(trial) for trial in trials])
    best_index = int(np.argmin(values))
    t0_fit = float(trials[best_index])
    if values[best_index] >= 1e99:
        return _rejected_epoch_fit("template-evaluation-failed", t0_fit=t0_fit)
    at_search_boundary = bool(best_index == 0 or best_index == len(trials) - 1)

    eps = grid_step_days
    if 0 < best_index < len(trials) - 1:
        a = values[best_index - 1]
        b = values[best_index]
        c = values[best_index + 1]
        denominator = a - 2.0 * b + c
        if abs(denominator) > 1e-12:
            t0_fit = t0_fit + 0.5 * eps * (a - c) / denominator
    best_model = _template_flux(template, t_window, t0_fit)
    if not isinstance(best_model, np.ndarray) or best_model.shape != f_window.shape:
        return _rejected_epoch_fit(
            "template-evaluation-failed",
            t0_fit=t0_fit,
            at_search_boundary=at_search_boundary,
        )
    depth_fit = _fit_local_template_depth(f_window, e_window, best_model)
    if depth_fit is None:
        return _rejected_epoch_fit(
            "invalid-local-depth-fit",
            t0_fit=t0_fit,
            at_search_boundary=at_search_boundary,
        )
    # DIAGNOSTIC_REASONING: A timing offset is only interpretable after the
    # local window contains a positive transit-shaped depth recovery.
    if depth_fit["depth_ppm"] <= 0:
        return _rejected_epoch_fit(
            "non-positive-local-depth",
            t0_fit=t0_fit,
            depth_fit=depth_fit,
            at_search_boundary=at_search_boundary,
        )
    if depth_fit["depth_snr"] < MIN_EPOCH_DEPTH_SNR:
        return _rejected_epoch_fit(
            "low-local-depth-snr",
            t0_fit=t0_fit,
            depth_fit=depth_fit,
            at_search_boundary=at_search_boundary,
        )
    # NUMERICAL_GUARD: Non-positive local curvature cannot provide a finite
    # quadratic timing uncertainty, so preserve a rejection record instead.
    curvature = (chi2(t0_fit + eps) - 2.0 * chi2(t0_fit) + chi2(t0_fit - eps)) / (eps**2)
    if not math.isfinite(curvature) or curvature <= 0:
        return _rejected_epoch_fit(
            "non-positive-timing-curvature",
            t0_fit=t0_fit,
            depth_fit=depth_fit,
            at_search_boundary=at_search_boundary,
        )
    sigma_t0 = math.sqrt(2.0 / curvature)
    if not math.isfinite(sigma_t0):
        return _rejected_epoch_fit(
            "non-finite-timing-uncertainty",
            t0_fit=t0_fit,
            depth_fit=depth_fit,
            at_search_boundary=at_search_boundary,
        )
    sigma_raw = float(sigma_t0)
    sigma_t0 = float(np.clip(sigma_t0, MIN_TIMING_SIGMA_DAYS, MAX_TIMING_SIGMA_DAYS))
    return {
        "t0_fit": t0_fit,
        "sigma_t0": sigma_t0,
        "sigma_t0_raw": sigma_raw,
        **depth_fit,
        "excluded_no_detection": False,
        "at_search_boundary": at_search_boundary,
        "sigma_t0_clipped": bool(sigma_t0 != sigma_raw),
        "rejection_reason": None,
    }


def fit_weighted_linear_ephemeris(
    epochs: np.ndarray,
    observed_btjd: np.ndarray,
    timing_errors_days: np.ndarray,
) -> Dict[str, Any]:
    """Fit a formal weighted linear ephemeris to measured transit centers.

    Mathematical Formulation:
        The fitted centers follow ``t(N) = t_ref + (N - N_ref) P``.  With
        independent timing errors, weighted least squares uses weights
        ``1 / sigma_t0**2`` and returns the formal covariance of ``t_ref`` and
        ``P``.

    Args:
        epochs (np.ndarray): Integer transit-cycle labels.
        observed_btjd (np.ndarray): Measured transit centers in BTJD days.
        timing_errors_days (np.ndarray): Positive formal timing errors in days.

    Returns:
        Dict[str, Any]: Fit status, reference epoch and period in days, formal
        covariance, chi-square diagnostics, or an explicit non-fit status.

    Note:
        The covariance excludes correlated photometric noise, template mismatch,
        and uncertainty in assigning an epoch cycle.

    The timing errors are treated as independent Gaussian measurement errors.
    The resulting covariance is therefore formal only: it does not model
    correlated photometric noise, template mismatch, or uncertain epoch-cycle
    assignment.
    """
    valid = (
        np.isfinite(epochs)
        & np.isfinite(observed_btjd)
        & np.isfinite(timing_errors_days)
        & (timing_errors_days > 0)
    )
    epochs = np.asarray(epochs[valid], dtype=int)
    observed_btjd = np.asarray(observed_btjd[valid], dtype=float)
    timing_errors_days = np.asarray(timing_errors_days[valid], dtype=float)
    if epochs.size < 2:
        return {
            "status": "not-fit-insufficient-transits",
            "method": "weighted-linear-least-squares",
            "n_transits_used": int(epochs.size),
            "reference_epoch": None,
            "reference_epoch_btjd": None,
            "reference_epoch_uncertainty_days": None,
            "period_days": None,
            "period_uncertainty_days": None,
            "covariance_reference_epoch_period_days2": None,
            "chi_square": None,
            "degrees_of_freedom": None,
            "reduced_chi_square": None,
            "uncertainty_interpretation": "not available without at least two timed events",
        }

    reference_epoch = int(np.rint(np.median(epochs)))
    centered_epochs = epochs.astype(float) - float(reference_epoch)
    design = np.column_stack((np.ones(epochs.size), centered_epochs))
    weights = 1.0 / timing_errors_days**2
    normal_matrix = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(normal_matrix)
        coefficients = covariance @ (design.T @ (weights * observed_btjd))
    except np.linalg.LinAlgError:
        return {
            "status": "not-fit-singular-normal-matrix",
            "method": "weighted-linear-least-squares",
            "n_transits_used": int(epochs.size),
            "reference_epoch": reference_epoch,
            "reference_epoch_btjd": None,
            "reference_epoch_uncertainty_days": None,
            "period_days": None,
            "period_uncertainty_days": None,
            "covariance_reference_epoch_period_days2": None,
            "chi_square": None,
            "degrees_of_freedom": None,
            "reduced_chi_square": None,
                "uncertainty_interpretation": "not available because the weighted fit was singular",
        }

    coefficient_variances = np.diag(covariance)
    if (
        not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(coefficient_variances))
        or np.any(coefficient_variances < 0.0)
    ):
        return {
            "status": "not-fit-invalid-covariance",
            "method": "weighted-linear-least-squares",
            "n_transits_used": int(epochs.size),
            "reference_epoch": reference_epoch,
            "reference_epoch_btjd": None,
            "reference_epoch_uncertainty_days": None,
            "period_days": None,
            "period_uncertainty_days": None,
            "covariance_reference_epoch_period_days2": None,
            "chi_square": None,
            "degrees_of_freedom": None,
            "reduced_chi_square": None,
            "uncertainty_interpretation": "not available because the formal covariance was invalid",
        }

    fitted_btjd = design @ coefficients
    residuals = observed_btjd - fitted_btjd
    chi_square = float(np.sum((residuals / timing_errors_days) ** 2))
    degrees_of_freedom = int(epochs.size - 2)
    return {
        "status": "fit",
        "method": "weighted-linear-least-squares",
        "n_transits_used": int(epochs.size),
        "reference_epoch": reference_epoch,
        "reference_epoch_btjd": float(coefficients[0]),
        "reference_epoch_uncertainty_days": float(math.sqrt(covariance[0, 0])),
        "period_days": float(coefficients[1]),
        "period_uncertainty_days": float(math.sqrt(covariance[1, 1])),
        "covariance_reference_epoch_period_days2": float(covariance[0, 1]),
        "covariance_matrix_days2": covariance.tolist(),
        "covariance_parameter_order": ["reference_epoch_btjd", "period_days"],
        "chi_square": chi_square,
        "degrees_of_freedom": degrees_of_freedom,
        "reduced_chi_square": (
            float(chi_square / degrees_of_freedom) if degrees_of_freedom > 0 else None
        ),
        "uncertainty_interpretation": (
            "formal independent-timing-error covariance; correlated noise, template mismatch, "
            "and cycle-count uncertainty are not included"
        ),
    }


def fit_weighted_quadratic_ephemeris(
    epochs: np.ndarray,
    observed_btjd: np.ndarray,
    timing_errors_days: np.ndarray,
) -> Dict[str, Any]:
    """Fit a formal quadratic ephemeris to accepted timing centers.

    The fitted form is ``t(N) = t_ref + P E + q E**2``, where
    ``E = N - N_ref``.  The reported period change per epoch is ``2q``.
    This is a descriptive alternative to the linear ephemeris, not a
    dynamical model, an apsidal-motion inference, or a TTV detection.
    """
    valid = (
        np.isfinite(epochs)
        & np.isfinite(observed_btjd)
        & np.isfinite(timing_errors_days)
        & (timing_errors_days > 0)
    )
    epochs = np.asarray(epochs[valid], dtype=int)
    observed_btjd = np.asarray(observed_btjd[valid], dtype=float)
    timing_errors_days = np.asarray(timing_errors_days[valid], dtype=float)
    if epochs.size < 3:
        return {
            "status": "not-fit-insufficient-transits",
            "method": "weighted-quadratic-least-squares",
            "n_transits_used": int(epochs.size),
            "reference_epoch": None,
            "reference_epoch_btjd": None,
            "reference_epoch_uncertainty_days": None,
            "period_days": None,
            "period_uncertainty_days": None,
            "quadratic_coefficient_days_per_epoch2": None,
            "quadratic_coefficient_uncertainty_days_per_epoch2": None,
            "period_change_per_epoch_days": None,
            "period_change_per_epoch_uncertainty_days": None,
            "covariance_matrix_days2": None,
            "covariance_parameter_order": None,
            "chi_square": None,
            "degrees_of_freedom": None,
            "reduced_chi_square": None,
            "uncertainty_interpretation": "not available without at least three timed events",
        }

    reference_epoch = int(np.rint(np.median(epochs)))
    centered_epochs = epochs.astype(float) - float(reference_epoch)
    design = np.column_stack((np.ones(epochs.size), centered_epochs, centered_epochs**2))
    weights = 1.0 / timing_errors_days**2
    normal_matrix = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(normal_matrix)
        coefficients = covariance @ (design.T @ (weights * observed_btjd))
    except np.linalg.LinAlgError:
        return {
            "status": "not-fit-singular-normal-matrix",
            "method": "weighted-quadratic-least-squares",
            "n_transits_used": int(epochs.size),
            "reference_epoch": reference_epoch,
            "reference_epoch_btjd": None,
            "reference_epoch_uncertainty_days": None,
            "period_days": None,
            "period_uncertainty_days": None,
            "quadratic_coefficient_days_per_epoch2": None,
            "quadratic_coefficient_uncertainty_days_per_epoch2": None,
            "period_change_per_epoch_days": None,
            "period_change_per_epoch_uncertainty_days": None,
            "covariance_matrix_days2": None,
            "covariance_parameter_order": None,
            "chi_square": None,
            "degrees_of_freedom": None,
            "reduced_chi_square": None,
                "uncertainty_interpretation": "not available because the weighted fit was singular",
        }

    coefficient_variances = np.diag(covariance)
    if (
        not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(coefficient_variances))
        or np.any(coefficient_variances < 0.0)
    ):
        return {
            "status": "not-fit-invalid-covariance",
            "method": "weighted-quadratic-least-squares",
            "n_transits_used": int(epochs.size),
            "reference_epoch": reference_epoch,
            "reference_epoch_btjd": None,
            "reference_epoch_uncertainty_days": None,
            "period_days": None,
            "period_uncertainty_days": None,
            "quadratic_coefficient_days_per_epoch2": None,
            "quadratic_coefficient_uncertainty_days_per_epoch2": None,
            "period_change_per_epoch_days": None,
            "period_change_per_epoch_uncertainty_days": None,
            "covariance_matrix_days2": None,
            "covariance_parameter_order": None,
            "chi_square": None,
            "degrees_of_freedom": None,
            "reduced_chi_square": None,
            "uncertainty_interpretation": "not available because the formal covariance was invalid",
        }

    fitted_btjd = design @ coefficients
    residuals = observed_btjd - fitted_btjd
    chi_square = float(np.sum((residuals / timing_errors_days) ** 2))
    degrees_of_freedom = int(epochs.size - 3)
    quadratic_error = float(math.sqrt(covariance[2, 2]))
    return {
        "status": "fit",
        "method": "weighted-quadratic-least-squares",
        "n_transits_used": int(epochs.size),
        "reference_epoch": reference_epoch,
        "reference_epoch_btjd": float(coefficients[0]),
        "reference_epoch_uncertainty_days": float(math.sqrt(covariance[0, 0])),
        "period_days": float(coefficients[1]),
        "period_uncertainty_days": float(math.sqrt(covariance[1, 1])),
        "quadratic_coefficient_days_per_epoch2": float(coefficients[2]),
        "quadratic_coefficient_uncertainty_days_per_epoch2": quadratic_error,
        "period_change_per_epoch_days": float(2.0 * coefficients[2]),
        "period_change_per_epoch_uncertainty_days": float(2.0 * quadratic_error),
        "covariance_matrix_days2": covariance.tolist(),
        "covariance_parameter_order": [
            "reference_epoch_btjd",
            "period_days",
            "quadratic_coefficient_days_per_epoch2",
        ],
        "chi_square": chi_square,
        "degrees_of_freedom": degrees_of_freedom,
        "reduced_chi_square": (
            float(chi_square / degrees_of_freedom) if degrees_of_freedom > 0 else None
        ),
        "uncertainty_interpretation": (
            "formal independent-timing-error covariance; correlated noise, template mismatch, "
            "and cycle-count uncertainty are not included"
        ),
    }


def compare_ephemeris_models(
    linear_ephemeris: Dict[str, Any], quadratic_ephemeris: Dict[str, Any]
) -> Dict[str, Any]:
    """Return a descriptive BIC comparison of linear and quadratic timing fits."""
    if (
        linear_ephemeris.get("status") != "fit"
        or quadratic_ephemeris.get("status") != "fit"
    ):
        return {
            "status": "not-compared-insufficient-or-singular-fit",
            "criterion": "BIC = chi_square + parameter_count * ln(n_transits)",
            "linear_bic": None,
            "quadratic_bic": None,
            "delta_bic_linear_minus_quadratic": None,
            "preferred_model": None,
            "interpretation": "Both formal ephemeris fits are required before a descriptive BIC comparison.",
        }
    n_transits = int(linear_ephemeris["n_transits_used"])
    if n_transits < 3 or n_transits != int(quadratic_ephemeris["n_transits_used"]):
        return {
            "status": "not-compared-inconsistent-sample-count",
            "criterion": "BIC = chi_square + parameter_count * ln(n_transits)",
            "linear_bic": None,
            "quadratic_bic": None,
            "delta_bic_linear_minus_quadratic": None,
            "preferred_model": None,
            "interpretation": "The linear and quadratic fits do not share a sufficient timing sample.",
        }
    linear_chi_square = float(linear_ephemeris["chi_square"])
    quadratic_chi_square = float(quadratic_ephemeris["chi_square"])
    if not math.isfinite(linear_chi_square) or not math.isfinite(quadratic_chi_square):
        return {
            "status": "not-compared-nonfinite-fit-statistic",
            "criterion": "BIC = chi_square + parameter_count * ln(n_transits)",
            "linear_bic": None,
            "quadratic_bic": None,
            "delta_bic_linear_minus_quadratic": None,
            "preferred_model": None,
            "interpretation": "A formal fit statistic is non-finite.",
        }
    linear_bic = linear_chi_square + 2.0 * math.log(n_transits)
    quadratic_bic = quadratic_chi_square + 3.0 * math.log(n_transits)
    delta_bic = linear_bic - quadratic_bic
    return {
        "status": "compared",
        "criterion": "BIC = chi_square + parameter_count * ln(n_transits)",
        "linear_bic": float(linear_bic),
        "quadratic_bic": float(quadratic_bic),
        "delta_bic_linear_minus_quadratic": float(delta_bic),
        "preferred_model": "quadratic" if delta_bic > 0.0 else "linear",
        "interpretation": (
            "This formal information-criterion comparison is descriptive and does not identify "
            "a dynamical TTV mechanism, apsidal motion, or a companion."
        ),
    }


def _solve_kepler_ltt(mean_anom: np.ndarray, ecc: float) -> np.ndarray:
    """Solve Kepler's equation E - e*sin(E) = M for Rømer LTT calculation."""
    ecc = float(np.clip(ecc, 0.0, 0.95))
    m = np.mod(mean_anom, 2.0 * np.pi)
    e_anom = m.copy()
    for _ in range(8):
        f = e_anom - ecc * np.sin(e_anom) - m
        f_prime = 1.0 - ecc * np.cos(e_anom)
        e_anom -= f / np.maximum(f_prime, 1e-12)
    return e_anom


def fit_secular_timing_models(
    epochs: np.ndarray,
    transit_times: np.ndarray,
    transit_errors: np.ndarray,
) -> Dict[str, Any]:
    """Fit and compare four competing secular orbital timing ephemeris models.

    Models:
    1. Linear Ephemeris (k=2):
       T_lin(N) = T0 + P0 * N
    2. Tidal Orbital Decay / Quadratic Ephemeris (k=3):
       T_decay(N) = T0 + P0 * N + 0.5 * P0 * P_dot * N^2 = T0 + P0 * N + q * N^2
       where characteristic decay timescale tau_decay = P0 / |P_dot| = P0^2 / (2 * |q|).
    3. Analytical Apsidal Precession (k=5):
       T_tra(N) = T0 + P_s * N - (e * P_a / pi) * cos(omega_0 + omega_dot * N)
       where P_a = P_s / (1 - omega_dot / (2 * pi)).
    4. Rømer Light-Travel Time (LTT) Delay (k=6):
       T_LTT(N) = T0 + P0 * N + A_LTT * [(1 - e_b^2)/(1 + e_b * cos(nu_b(N))) * sin(nu_b(N) + omega_b)]
       where nu_b(N) is the companion true anomaly.

    Evaluates chi2, reduced chi2, AIC, BIC, Delta_AIC, Delta_BIC, and flags
    observational baseline degeneracy between P_dot and omega_dot.
    """
    valid = (
        np.isfinite(epochs)
        & np.isfinite(transit_times)
        & np.isfinite(transit_errors)
        & (transit_errors > 0)
    )
    epochs_clean = np.asarray(epochs[valid], dtype=float)
    times_clean = np.asarray(transit_times[valid], dtype=float)
    errors_clean = np.asarray(transit_errors[valid], dtype=float)
    n_transits = int(epochs_clean.size)

    base_result: Dict[str, Any] = {
        "status": "not-fit-insufficient-transits",
        "n_transits": n_transits,
        "models": {},
        "preferred_model_bic": None,
        "preferred_model_aic": None,
        "interpretation": "At least 2 valid transit timings are required for secular timing model comparison.",
    }

    if n_transits < 2:
        return base_result

    # 1. Model 0: Linear Ephemeris (k=2)
    weights = 1.0 / errors_clean**2
    design_lin = np.column_stack((np.ones(n_transits), epochs_clean))
    normal_lin = design_lin.T @ (weights[:, None] * design_lin)
    try:
        cov_lin = np.linalg.inv(normal_lin)
        coeff_lin = cov_lin @ (design_lin.T @ (weights * times_clean))
        t0_lin = float(coeff_lin[0])
        p0_lin = float(coeff_lin[1])
        model_lin = design_lin @ coeff_lin
        chi2_lin = float(np.sum(((times_clean - model_lin) / errors_clean) ** 2))
        dof_lin = max(1, n_transits - 2)
        red_chi2_lin = float(chi2_lin / dof_lin)
        aic_lin = float(chi2_lin + 2.0 * 2)
        bic_lin = float(chi2_lin + 2.0 * math.log(n_transits))
        models_dict: Dict[str, Any] = {
            "linear": {
                "name": "constant_linear_ephemeris",
                "k_parameters": 2,
                "status": "fit",
                "parameters": {
                    "t0_btjd": t0_lin,
                    "t0_uncertainty_days": float(math.sqrt(cov_lin[0, 0])),
                    "period_days": p0_lin,
                    "period_uncertainty_days": float(math.sqrt(cov_lin[1, 1])),
                },
                "chi_square": chi2_lin,
                "reduced_chi_square": red_chi2_lin,
                "aic": aic_lin,
                "bic": bic_lin,
                "delta_aic": 0.0,
                "delta_bic": 0.0,
            }
        }
    except (np.linalg.LinAlgError, ValueError):
        return base_result

    # 2. Model 1: Quadratic / Tidal Orbital Decay (k=3)
    if n_transits >= 3:
        design_quad = np.column_stack((np.ones(n_transits), epochs_clean, epochs_clean**2))
        normal_quad = design_quad.T @ (weights[:, None] * design_quad)
        try:
            cov_quad = np.linalg.inv(normal_quad)
            coeff_quad = cov_quad @ (design_quad.T @ (weights * times_clean))
            t0_quad = float(coeff_quad[0])
            p0_quad = float(coeff_quad[1])
            q_quad = float(coeff_quad[2])
            model_quad = design_quad @ coeff_quad
            chi2_quad = float(np.sum(((times_clean - model_quad) / errors_clean) ** 2))
            dof_quad = max(1, n_transits - 3)
            red_chi2_quad = float(chi2_quad / dof_quad)
            aic_quad = float(chi2_quad + 2.0 * 3)
            bic_quad = float(chi2_quad + 3.0 * math.log(n_transits))
            p_dot = 2.0 * q_quad / p0_quad if p0_quad > 0 else 0.0
            tau_decay_days = (p0_quad**2 / (2.0 * abs(q_quad))) if abs(q_quad) > 1e-15 else None
            tau_decay_years = (tau_decay_days / JULIAN_YEAR_DAYS) if tau_decay_days is not None else None
            models_dict["quadratic_decay"] = {
                "name": "tidal_orbital_decay",
                "k_parameters": 3,
                "status": "fit",
                "parameters": {
                    "t0_btjd": t0_quad,
                    "period_days": p0_quad,
                    "quadratic_coefficient_q": q_quad,
                    "period_derivative_p_dot": p_dot,
                    "decay_timescale_days": tau_decay_days,
                    "decay_timescale_years": tau_decay_years,
                },
                "chi_square": chi2_quad,
                "reduced_chi_square": red_chi2_quad,
                "aic": aic_quad,
                "bic": bic_quad,
                "delta_aic": float(aic_quad - aic_lin),
                "delta_bic": float(bic_quad - bic_lin),
            }
        except (np.linalg.LinAlgError, ValueError):
            pass

    # 3. Model 2: Analytical Apsidal Precession (k=5)
    n_span = float(np.max(epochs_clean) - np.min(epochs_clean))
    if n_transits >= 5:
        def _res_apsidal(p: np.ndarray) -> np.ndarray:
            t0_v, ps_v, e_v, w0_v, wdot_v = p
            pa_denom = 1.0 - wdot_v / (2.0 * math.pi)
            if pa_denom <= 1e-5:
                return np.full_like(times_clean, 1e6)
            pa_v = ps_v / pa_denom
            m_v = t0_v + ps_v * epochs_clean - (e_v * pa_v / math.pi) * np.cos(w0_v + wdot_v * epochs_clean)
            return (times_clean - m_v) / errors_clean

        x0_aps = [t0_lin, p0_lin, 0.01, 0.0, 0.0]
        bounds_aps = (
            [t0_lin - 10.0 * p0_lin, 0.1 * p0_lin, 0.0, -math.pi, -0.5],
            [t0_lin + 10.0 * p0_lin, 10.0 * p0_lin, 0.95, math.pi, 0.5],
        )
        try:
            opt_aps = least_squares(_res_apsidal, x0_aps, bounds=bounds_aps, max_nfev=2000)
            if opt_aps.success:
                t0_aps, ps_aps, e_aps, w0_aps, wdot_aps = [float(v) for v in opt_aps.x]
                pa_denom = 1.0 - wdot_aps / (2.0 * math.pi)
                pa_aps = ps_aps / max(pa_denom, 1e-5)
                chi2_aps = float(np.sum(opt_aps.fun**2))
                dof_aps = max(1, n_transits - 5)
                red_chi2_aps = float(chi2_aps / dof_aps)
                aic_aps = float(chi2_aps + 2.0 * 5)
                bic_aps = float(chi2_aps + 5.0 * math.log(n_transits))
                delta_phi = float(n_span * abs(wdot_aps))
                degenerate = bool(delta_phi < math.pi / 2.0)
                warning = (
                    "Observational baseline span (N_span={0:.0f}) covers only {1:.3f} rad ({2:.2f} cycles) of apsidal precession; insufficient to break mathematical degeneracy with quadratic orbital decay (P_dot).".format(
                        n_span, delta_phi, delta_phi / (2.0 * math.pi)
                    )
                    if degenerate
                    else None
                )
                precession_period_epochs = (2.0 * math.pi / abs(wdot_aps)) if abs(wdot_aps) > 1e-9 else None
                models_dict["apsidal_precession"] = {
                    "name": "apsidal_precession",
                    "k_parameters": 5,
                    "status": "fit",
                    "parameters": {
                        "t0_btjd": t0_aps,
                        "sidereal_period_days": ps_aps,
                        "anomalistic_period_days": pa_aps,
                        "eccentricity": e_aps,
                        "omega_0_rad": w0_aps,
                        "omega_dot_rad_per_epoch": wdot_aps,
                        "precession_period_epochs": precession_period_epochs,
                    },
                    "chi_square": chi2_aps,
                    "reduced_chi_square": red_chi2_aps,
                    "aic": aic_aps,
                    "bic": bic_aps,
                    "delta_aic": float(aic_aps - aic_lin),
                    "delta_bic": float(bic_aps - bic_lin),
                    "degenerate_with_orbital_decay": degenerate,
                    "baseline_coverage_warning": warning,
                }
        except Exception:
            pass

    # 4. Model 3: Rømer Light-Travel Time Delay (k=6)
    if n_transits >= 6:
        def _res_ltt(p: np.ndarray) -> np.ndarray:
            t0_v, p0_v, a_ltt_v, pb_v, eb_v, wb_v = p
            if pb_v <= 0 or p0_v <= 0:
                return np.full_like(times_clean, 1e6)
            m_b = 2.0 * math.pi * (p0_v / pb_v) * epochs_clean
            e_anom = _solve_kepler_ltt(m_b, eb_v)
            cos_e = np.cos(e_anom)
            sin_e = np.sin(e_anom)
            denom = 1.0 - eb_v * cos_e
            denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
            cos_nu = (cos_e - eb_v) / denom
            sin_nu = (math.sqrt(max(0.0, 1.0 - eb_v**2)) * sin_e) / denom
            nu_b = np.arctan2(sin_nu, cos_nu)
            f_denom = 1.0 + eb_v * np.cos(nu_b)
            f_denom = np.where(np.abs(f_denom) < 1e-9, 1e-9, f_denom)
            factor = ((1.0 - eb_v**2) / f_denom) * np.sin(nu_b + wb_v)
            m_v = t0_v + p0_v * epochs_clean + a_ltt_v * factor
            return (times_clean - m_v) / errors_clean

        pb_init = max(5.0 * p0_lin, (n_span + 1.0) * p0_lin)
        x0_ltt = [t0_lin, p0_lin, 0.001, pb_init, 0.01, 0.0]
        bounds_ltt = (
            [t0_lin - 10.0 * p0_lin, 0.1 * p0_lin, 0.0, 1.5 * p0_lin, 0.0, -math.pi],
            [t0_lin + 10.0 * p0_lin, 10.0 * p0_lin, 1.0, 1000.0 * p0_lin, 0.95, math.pi],
        )
        try:
            opt_ltt = least_squares(_res_ltt, x0_ltt, bounds=bounds_ltt, max_nfev=2000)
            if opt_ltt.success:
                t0_ltt, p0_ltt, a_ltt, pb_ltt, eb_ltt, wb_ltt = [float(v) for v in opt_ltt.x]
                chi2_ltt = float(np.sum(opt_ltt.fun**2))
                dof_ltt = max(1, n_transits - 6)
                red_chi2_ltt = float(chi2_ltt / dof_ltt)
                aic_ltt = float(chi2_ltt + 2.0 * 6)
                bic_ltt = float(chi2_ltt + 6.0 * math.log(n_transits))
                models_dict["roemer_ltt"] = {
                    "name": "roemer_light_travel_time",
                    "k_parameters": 6,
                    "status": "fit",
                    "parameters": {
                        "t0_btjd": t0_ltt,
                        "period_days": p0_ltt,
                        "amplitude_ltt_days": a_ltt,
                        "amplitude_ltt_minutes": a_ltt * 1440.0,
                        "companion_period_days": pb_ltt,
                        "eccentricity": eb_ltt,
                        "omega_rad": wb_ltt,
                    },
                    "chi_square": chi2_ltt,
                    "reduced_chi_square": red_chi2_ltt,
                    "aic": aic_ltt,
                    "bic": bic_ltt,
                    "delta_aic": float(aic_ltt - aic_lin),
                    "delta_bic": float(bic_ltt - bic_lin),
                }
        except Exception:
            pass

    preferred_bic = min(models_dict.keys(), key=lambda m: models_dict[m]["bic"])
    preferred_aic = min(models_dict.keys(), key=lambda m: models_dict[m]["aic"])

    return {
        "status": "compared",
        "n_transits": n_transits,
        "models": models_dict,
        "preferred_model_bic": preferred_bic,
        "preferred_model_aic": preferred_aic,
        "interpretation": (
            "Objective information-criterion comparison across linear, tidal decay, apsidal precession, "
            "and light-travel time models. Model selection is descriptive and does not constitute a "
            "definitive physical detection without external confirmation."
        ),
    }


def fit_orbital_decay(
    epochs: np.ndarray,
    observed_btjd: np.ndarray,
    timing_errors_days: np.ndarray,
) -> Dict[str, Any]:
    """Describe a quadratic timing trend as a formal orbital-period derivative.

    The quadratic ephemeris is ``t(E) = t_ref + P E + q E**2`` around the
    fitted reference epoch. It implies ``dP/dE = 2q`` and, at that reference
    epoch, ``dP/dt = 2q / P`` in days per day. This transformation does not
    distinguish orbital decay from apsidal motion, light-travel-time effects,
    additional companions, correlated timing noise, or a cycle-count error.
    """
    linear_ephemeris = fit_weighted_linear_ephemeris(
        epochs, observed_btjd, timing_errors_days
    )
    quadratic_ephemeris = fit_weighted_quadratic_ephemeris(
        epochs, observed_btjd, timing_errors_days
    )
    comparison = compare_ephemeris_models(linear_ephemeris, quadratic_ephemeris)
    n_transits = int(quadratic_ephemeris["n_transits_used"])
    result: Dict[str, Any] = {
        "status": "not-fit-insufficient-transits",
        "method": "weighted-quadratic-ephemeris-period-derivative",
        "n_transits_used": n_transits,
        "reference_epoch": quadratic_ephemeris.get("reference_epoch"),
        "reference_epoch_btjd": quadratic_ephemeris.get("reference_epoch_btjd"),
        "quadratic_coefficient_days_per_epoch2": None,
        "quadratic_coefficient_uncertainty_days_per_epoch2": None,
        "period_change_per_epoch_days": None,
        "period_change_per_epoch_uncertainty_days": None,
        "period_derivative_days_per_day": None,
        "period_derivative_uncertainty_days_per_day": None,
        "period_derivative_ms_per_julian_year": None,
        "period_derivative_uncertainty_ms_per_julian_year": None,
        "covariance_matrix": None,
        "covariance_parameter_order": None,
        "covariance_parameter_units": None,
        "ephemeris_model_comparison": comparison,
        "validation_eligible": False,
        "claim_eligible": False,
        "interpretation": (
            "A quadratic timing trend requires at least four accepted timings and remains "
            "a formal descriptive diagnostic, not evidence for orbital decay."
        ),
    }
    if n_transits < MIN_ORBITAL_DECAY_TRANSITS:
        return result
    if linear_ephemeris.get("status") != "fit" or quadratic_ephemeris.get("status") != "fit":
        result["status"] = "not-fit-singular-or-invalid-ephemeris"
        result["interpretation"] = (
            "The accepted timings did not support both formal ephemeris fits; no period "
            "derivative was calculated."
        )
        return result

    try:
        period_days = float(quadratic_ephemeris["period_days"])
        coefficient = float(quadratic_ephemeris["quadratic_coefficient_days_per_epoch2"])
        covariance = np.asarray(quadratic_ephemeris["covariance_matrix_days2"], dtype=float)
    except (KeyError, TypeError, ValueError):
        result["status"] = "not-fit-invalid-quadratic-solution"
        return result
    if (
        not math.isfinite(period_days)
        or period_days <= 0.0
        or not math.isfinite(coefficient)
        or covariance.shape != (3, 3)
        or not np.all(np.isfinite(covariance))
    ):
        result["status"] = "not-fit-invalid-quadratic-solution"
        return result
    coefficient_variance = float(covariance[2, 2])
    if not math.isfinite(coefficient_variance) or coefficient_variance < 0.0:
        result["status"] = "not-fit-invalid-quadratic-solution"
        return result

    derivative_days_per_day = 2.0 * coefficient / period_days
    jacobian = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -2.0 * coefficient / period_days**2, 2.0 / period_days],
        ],
        dtype=float,
    )
    transformed_covariance = jacobian @ covariance @ jacobian.T
    transformed_covariance = 0.5 * (transformed_covariance + transformed_covariance.T)
    variances = np.diag(transformed_covariance)
    if not np.all(np.isfinite(variances)) or np.any(variances < 0.0):
        result["status"] = "not-fit-invalid-transformed-covariance"
        return result
    conversion = JULIAN_YEAR_DAYS * SECONDS_PER_DAY * 1000.0
    coefficient_uncertainty = math.sqrt(coefficient_variance)
    derivative_uncertainty = math.sqrt(float(variances[2]))
    result.update(
        {
            "status": "fit",
            "quadratic_coefficient_days_per_epoch2": coefficient,
            "quadratic_coefficient_uncertainty_days_per_epoch2": coefficient_uncertainty,
            "period_change_per_epoch_days": 2.0 * coefficient,
            "period_change_per_epoch_uncertainty_days": 2.0 * coefficient_uncertainty,
            "period_derivative_days_per_day": derivative_days_per_day,
            "period_derivative_uncertainty_days_per_day": derivative_uncertainty,
            "period_derivative_ms_per_julian_year": derivative_days_per_day * conversion,
            "period_derivative_uncertainty_ms_per_julian_year": derivative_uncertainty * conversion,
            "covariance_matrix": transformed_covariance.tolist(),
            "covariance_parameter_order": [
                "reference_epoch_btjd",
                "period_days",
                "period_derivative_days_per_day",
            ],
            "covariance_parameter_units": ["days", "days", "days_per_day"],
            "interpretation": (
                "This formal quadratic-ephemeris period derivative is descriptive only. "
                "A quadratic preference does not identify orbital decay over apsidal motion, "
                "light-travel-time effects, additional companions, correlated timing noise, "
                "template evolution, timing-error clipping, or a cycle-count error."
            ),
        }
    )
    return result


def _ephemeris_calculated_times(
    fit: Dict[str, Any], epochs: np.ndarray, model: str
) -> Optional[np.ndarray]:
    """Evaluate a fitted linear or quadratic ephemeris on epoch labels."""
    if fit.get("status") != "fit":
        return None
    try:
        centered_epochs = epochs.astype(float) - float(fit["reference_epoch"])
        calculated = float(fit["reference_epoch_btjd"]) + centered_epochs * float(
            fit["period_days"]
        )
        if model == "quadratic":
            calculated = calculated + centered_epochs**2 * float(
                fit["quadratic_coefficient_days_per_epoch2"]
            )
    except (KeyError, TypeError, ValueError):
        return None
    return calculated if np.all(np.isfinite(calculated)) else None


def transit_timing_analysis(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    ephemeris: Dict[str, Any],
    template: Dict[str, Any],
    window_days: float = WINDOW_DAYS,
    ephemeris_model: str = "linear",
    include_orbital_decay: bool = False,
) -> Dict[str, Any]:
    """Fit measurable epochs and construct a provenance-ready O-C report.

    Mathematical Formulation:
        For accepted epochs, observed-minus-calculated residuals are
        ``O-C = (t_obs - t_calc)`` and are reported in minutes.  A formal
        weighted linear ephemeris is refit when sufficient timing measurements
        are available.

    Args:
        time (np.ndarray): Cadence times in BTJD days.
        flux (np.ndarray): Normalized relative flux values.
        flux_err (np.ndarray): Positive normalized-flux uncertainties.
        ephemeris (Dict[str, Any]): Candidate-derived period, epoch, depth, and
            duration values with their expected units.
        template (Dict[str, Any]): Fixed template geometry and quadratic
            limb-darkening coefficients.  Production callers must build it
            from a candidate-local transit-fit posterior.
        window_days (float): Half-width for each local timing fit in days.
        ephemeris_model (str): O-C reference model, ``"linear"`` by default
            or ``"quadratic"`` when the optional formal quadratic fit is
            available. Both fits and their BIC comparison are always reported.
        include_orbital_decay (bool): Include the opt-in formal period-
            derivative diagnostic without changing the selected O-C reference.

    Returns:
        Dict[str, Any]: Accepted and rejected epoch records, O-C values in
        minutes, formal timing errors, and linear/quadratic ephemeris diagnostics.

    Note:
        Rejected epochs remain explicit evidence.  The report does not fit a
        dynamical multi-body model or determine a companion mass.
    """
    if ephemeris_model not in ("linear", "quadratic"):
        raise ValueError("ephemeris_model must be one of: linear, quadratic")
    if not isinstance(include_orbital_decay, bool):
        raise ValueError("include_orbital_decay must be a Boolean")
    period_days = float(ephemeris["period_days"])
    t0_reference = float(ephemeris["epoch_btjd"])
    n_min = int(np.floor((np.min(time) - t0_reference) / period_days))
    n_max = int(np.ceil((np.max(time) - t0_reference) / period_days))

    epochs: List[int] = []
    t_observed: List[float] = []
    t_calculated: List[float] = []
    t_errors: List[float] = []
    per_epoch_records: List[Dict[str, Any]] = []
    rejected_epochs: List[Dict[str, Any]] = []
    uncertainty_clipped_epochs: List[Dict[str, Any]] = []
    search_boundary_epochs: List[Dict[str, Any]] = []
    n_excluded_no_detection = 0
    n_template_failures = 0
    for epoch in range(n_min, n_max + 1):
        t_expected = t0_reference + epoch * period_days
        fit = fit_transit_epoch(
            time, flux, flux_err, template, t_expected, window_days=window_days
        )
        if fit is None:
            fit = _rejected_epoch_fit("no-measurable-timing-fit")
        if fit["excluded_no_detection"]:
            record = {
                "epoch": epoch,
                "t0_fit_btjd": fit["t0_fit"],
                "t_expected_btjd": t_expected,
                "sigma_t0_days": None,
                "sigma_t0_raw_days": None,
                "local_depth_ppm": fit["depth_ppm"],
                "local_depth_uncertainty_ppm": fit["depth_uncertainty_ppm"],
                "local_depth_snr": fit["depth_snr"],
                "excluded_no_detection": True,
                "rejection_reason": fit["rejection_reason"],
                "at_search_boundary": fit["at_search_boundary"],
                "sigma_t0_clipped": False,
            }
            per_epoch_records.append(record)
            rejected_epochs.append(record)
            n_excluded_no_detection += 1
            if record["at_search_boundary"]:
                search_boundary_epochs.append(record)
            if fit["rejection_reason"] == "template-evaluation-failed":
                n_template_failures += 1
            continue
        t0_fit = fit["t0_fit"]
        sigma_t0 = fit["sigma_t0"]
        epochs.append(epoch)
        t_observed.append(t0_fit)
        t_calculated.append(t_expected)
        t_errors.append(sigma_t0)
        record = {
            "epoch": epoch,
            "t0_fit_btjd": t0_fit,
            "t_expected_btjd": t_expected,
            "sigma_t0_days": sigma_t0,
            "sigma_t0_raw_days": fit["sigma_t0_raw"],
            "local_depth_ppm": fit["depth_ppm"],
            "local_depth_uncertainty_ppm": fit["depth_uncertainty_ppm"],
            "local_depth_snr": fit["depth_snr"],
            "at_search_boundary": fit["at_search_boundary"],
            "sigma_t0_clipped": fit["sigma_t0_clipped"],
            "excluded_no_detection": False,
            "rejection_reason": None,
        }
        per_epoch_records.append(record)
        if record["at_search_boundary"]:
            search_boundary_epochs.append(record)
        if record["sigma_t0_clipped"]:
            uncertainty_clipped_epochs.append(record)

    epochs_arr = np.asarray(epochs, dtype=int)
    t_observed_arr = np.asarray(t_observed, dtype=float)
    t_errors_arr = np.asarray(t_errors, dtype=float)
    input_t_calculated_arr = np.asarray(t_calculated, dtype=float)
    linear_ephemeris = fit_weighted_linear_ephemeris(
        epochs_arr, t_observed_arr, t_errors_arr
    )
    quadratic_ephemeris = fit_weighted_quadratic_ephemeris(
        epochs_arr, t_observed_arr, t_errors_arr
    )
    ephemeris_comparison = compare_ephemeris_models(linear_ephemeris, quadratic_ephemeris)
    selected_fit = quadratic_ephemeris if ephemeris_model == "quadratic" else linear_ephemeris
    calculated_from_selected = _ephemeris_calculated_times(
        selected_fit, epochs_arr, ephemeris_model
    )
    if calculated_from_selected is None and ephemeris_model == "quadratic":
        calculated_from_selected = _ephemeris_calculated_times(
            linear_ephemeris, epochs_arr, "linear"
        )
        ephemeris_model_used = "linear-fallback-quadratic-unavailable"
    elif calculated_from_selected is None:
        ephemeris_model_used = "input-ephemeris-fallback"
    else:
        ephemeris_model_used = ephemeris_model
    t_calculated_arr = (
        calculated_from_selected
        if calculated_from_selected is not None
        else input_t_calculated_arr
    )
    oc_minutes = (t_observed_arr - t_calculated_arr) * 1440.0
    input_ephemeris_oc_minutes = (t_observed_arr - input_t_calculated_arr) * 1440.0
    oc_errors_minutes = t_errors_arr * 1440.0
    rms_oc = float(np.sqrt(np.mean(oc_minutes**2))) if oc_minutes.size else None
    mean_uncertainty = float(np.mean(oc_errors_minutes)) if oc_errors_minutes.size else None
    ephemeris_models_comparison = fit_secular_timing_models(
        epochs_arr, t_observed_arr, t_errors_arr
    )
    result = {
        "epochs": [int(epoch) for epoch in epochs_arr],
        "t_observed_btjd": [float(value) for value in t_observed_arr],
        "t_calculated_btjd": [float(value) for value in t_calculated_arr],
        "oc_minutes": [float(value) for value in oc_minutes],
        "input_ephemeris_oc_minutes": [float(value) for value in input_ephemeris_oc_minutes],
        "oc_error_minutes": [float(value) for value in oc_errors_minutes],
        "oc_rms_minutes": rms_oc,
        "mean_uncertainty_minutes": mean_uncertainty,
        "n_transits_fit": int(epochs_arr.size),
        "n_excluded_no_detection": n_excluded_no_detection,
        "n_rejected_epochs": len(rejected_epochs),
        "n_template_failures": n_template_failures,
        "rejected_epochs": rejected_epochs,
        "uncertainty_clipped_epochs": uncertainty_clipped_epochs,
        "search_boundary_epochs": search_boundary_epochs,
        "per_epoch": per_epoch_records,
        "epoch_acceptance": {
            "requires_positive_local_depth": True,
            "minimum_local_depth_snr": MIN_EPOCH_DEPTH_SNR,
            "local_depth_method": "weighted baseline plus fixed-template depth scale",
            "local_depth_uncertainty": "formal independent-flux-error covariance",
        },
        "linear_ephemeris": linear_ephemeris,
        "quadratic_ephemeris": quadratic_ephemeris,
        "ephemeris_model_requested": ephemeris_model,
        "ephemeris_model_used": ephemeris_model_used,
        "ephemeris_model_comparison": ephemeris_comparison,
        "ephemeris_models_comparison": ephemeris_models_comparison,
    }
    if include_orbital_decay:
        result["orbital_decay_fit"] = fit_orbital_decay(
            epochs_arr, t_observed_arr, t_errors_arr
        )
    return result


def _is_timing_number(value: object) -> bool:
    """Return whether a value is a non-Boolean JSON-compatible number."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _finite_timing_series(analysis: Dict[str, Any], field: str, *, positive: bool = False) -> np.ndarray:
    """Read one finite timing series without coercing strings or Boolean values."""
    values = analysis.get(field)
    if not isinstance(values, list):
        raise ValueError("timing {0} must be a list".format(field))
    parsed: List[float] = []
    for index, value in enumerate(values):
        if not _is_timing_number(value):
            raise ValueError("timing {0}[{1}] must be numeric".format(field, index))
        number = float(value)
        if not math.isfinite(number) or (positive and number <= 0.0):
            qualifier = "finite and positive" if positive else "finite"
            raise ValueError(
                "timing {0}[{1}] must be {2}".format(field, index, qualifier)
            )
        parsed.append(number)
    return np.asarray(parsed, dtype=float)


def _integer_timing_series(analysis: Dict[str, Any], field: str) -> np.ndarray:
    """Read integer epoch labels without silently truncating a numeric value."""
    values = analysis.get(field)
    if not isinstance(values, list):
        raise ValueError("timing {0} must be a list".format(field))
    parsed: List[int] = []
    for index, value in enumerate(values):
        if not _is_timing_number(value):
            raise ValueError("timing {0}[{1}] must be an integer".format(field, index))
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError("timing {0}[{1}] must be a finite integer".format(field, index))
        parsed.append(int(number))
    return np.asarray(parsed, dtype=int)


def _timing_numbers_agree(reported: object, expected: float, field: str) -> None:
    """Require a finite reported scalar to agree with its recomputation."""
    if not _is_timing_number(reported):
        raise ValueError("timing {0} must be numeric".format(field))
    value = float(reported)
    if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("timing {0} does not match its recomputed value".format(field))


def _timing_values_agree(reported: object, expected: object) -> bool:
    """Compare finite JSON-like timing values with tolerance for float replay."""
    if _is_timing_number(reported) and _is_timing_number(expected):
        left = float(reported)
        right = float(expected)
        return math.isfinite(left) and math.isfinite(right) and math.isclose(
            left, right, rel_tol=1e-10, abs_tol=1e-10
        )
    if reported is None or expected is None:
        return reported is expected
    if isinstance(reported, bool) or isinstance(expected, bool):
        return isinstance(reported, bool) and isinstance(expected, bool) and reported == expected
    if isinstance(reported, list) and isinstance(expected, list):
        return len(reported) == len(expected) and all(
            _timing_values_agree(left, right) for left, right in zip(reported, expected)
        )
    if isinstance(reported, dict) and isinstance(expected, dict):
        return set(reported) == set(expected) and all(
            _timing_values_agree(reported[key], expected[key]) for key in reported
        )
    return reported == expected


def _record_timing_float(record: Dict[str, Any], field: str, label: str) -> float:
    """Read a required finite numeric per-epoch field."""
    value = record.get(field)
    if not _is_timing_number(value):
        raise ValueError("{0}.{1} must be numeric".format(label, field))
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("{0}.{1} must be finite".format(label, field))
    return number


def _recompute_and_validate_timing_summary(
    analysis: Dict[str, Any],
    ephemeris: Dict[str, Any],
    include_orbital_decay: bool = False,
) -> Dict[str, Any]:
    """Recompute TTV aggregates from timing arrays before writing an artifact.

    This is intentionally a narrow producer-side consistency check, modelled on
    the survey sensitivity recovery validator.  It does not introduce a generic
    schema: it rebuilds only the TTV fields that are mathematically derived from
    accepted epoch arrays or per-epoch records, and rejects a mismatch instead
    of serialising a tampered summary.
    """
    if not isinstance(analysis, dict):
        raise ValueError("TTV timing analysis must be an object")
    try:
        period_days = float(ephemeris["period_days"])
        reference_epoch_btjd = float(ephemeris["epoch_btjd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("TTV ephemeris lacks finite timing inputs") from exc
    if not math.isfinite(period_days) or period_days <= 0.0 or not math.isfinite(reference_epoch_btjd):
        raise ValueError("TTV ephemeris has invalid timing inputs")

    epochs = _integer_timing_series(analysis, "epochs")
    observed = _finite_timing_series(analysis, "t_observed_btjd")
    reported_calculated = _finite_timing_series(analysis, "t_calculated_btjd")
    reported_oc = _finite_timing_series(analysis, "oc_minutes")
    reported_input_oc = _finite_timing_series(analysis, "input_ephemeris_oc_minutes")
    error_minutes = _finite_timing_series(analysis, "oc_error_minutes", positive=True)
    expected_length = epochs.size
    if any(
        values.size != expected_length
        for values in (observed, reported_calculated, reported_oc, reported_input_oc, error_minutes)
    ):
        raise ValueError("TTV accepted timing arrays have inconsistent lengths")

    per_epoch = analysis.get("per_epoch")
    if not isinstance(per_epoch, list):
        raise ValueError("timing per_epoch must be a list")
    accepted_records: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []
    clipped_records: List[Dict[str, Any]] = []
    boundary_records: List[Dict[str, Any]] = []
    seen_record_epochs = set()
    for index, record in enumerate(per_epoch):
        label = "timing per_epoch[{0}]".format(index)
        if not isinstance(record, dict):
            raise ValueError("{0} must be an object".format(label))
        epoch_value = record.get("epoch")
        if not _is_timing_number(epoch_value) or not float(epoch_value).is_integer():
            raise ValueError("{0}.epoch must be an integer".format(label))
        epoch = int(epoch_value)
        if epoch in seen_record_epochs:
            raise ValueError("timing per_epoch contains duplicate epochs")
        seen_record_epochs.add(epoch)
        expected_btjd = reference_epoch_btjd + epoch * period_days
        if not math.isclose(
            _record_timing_float(record, "t_expected_btjd", label),
            expected_btjd,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError("{0}.t_expected_btjd does not match the input ephemeris".format(label))
        excluded = record.get("excluded_no_detection")
        boundary = record.get("at_search_boundary")
        clipped = record.get("sigma_t0_clipped")
        if not isinstance(excluded, bool) or not isinstance(boundary, bool) or not isinstance(clipped, bool):
            raise ValueError("{0} has non-Boolean epoch status flags".format(label))
        if boundary:
            boundary_records.append(record)
        if excluded:
            if record.get("sigma_t0_days") is not None or record.get("sigma_t0_raw_days") is not None:
                raise ValueError("{0} rejected epoch has a timing uncertainty".format(label))
            if clipped or not isinstance(record.get("rejection_reason"), str) or not record["rejection_reason"]:
                raise ValueError("{0} rejected epoch has inconsistent status fields".format(label))
            rejected_records.append(record)
            continue
        if record.get("rejection_reason") is not None:
            raise ValueError("{0} accepted epoch has a rejection reason".format(label))
        _record_timing_float(record, "t0_fit_btjd", label)
        if _record_timing_float(record, "sigma_t0_days", label) <= 0.0:
            raise ValueError("{0}.sigma_t0_days must be positive".format(label))
        if _record_timing_float(record, "sigma_t0_raw_days", label) <= 0.0:
            raise ValueError("{0}.sigma_t0_raw_days must be positive".format(label))
        accepted_records.append(record)
        if clipped:
            clipped_records.append(record)

    if len(accepted_records) != expected_length:
        raise ValueError("TTV per_epoch accepted records do not match timing arrays")
    for index, record in enumerate(accepted_records):
        label = "timing per_epoch[{0}]".format(index)
        if int(record["epoch"]) != int(epochs[index]):
            raise ValueError("TTV accepted epoch order does not match timing arrays")
        if not math.isclose(
            _record_timing_float(record, "t0_fit_btjd", label), observed[index], rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError("TTV accepted epoch centers do not match timing arrays")
        if not math.isclose(
            _record_timing_float(record, "sigma_t0_days", label) * 1440.0,
            error_minutes[index],
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError("TTV accepted epoch uncertainties do not match timing arrays")

    timing_errors_days = error_minutes / 1440.0
    recomputed_linear = fit_weighted_linear_ephemeris(epochs, observed, timing_errors_days)
    recomputed_quadratic = fit_weighted_quadratic_ephemeris(
        epochs, observed, timing_errors_days
    )
    recomputed_comparison = compare_ephemeris_models(
        recomputed_linear, recomputed_quadratic
    )
    recomputed_orbital_decay = (
        fit_orbital_decay(epochs, observed, timing_errors_days)
        if include_orbital_decay
        else None
    )
    requested_model = analysis.get("ephemeris_model_requested", "linear")
    if requested_model not in ("linear", "quadratic"):
        raise ValueError("timing ephemeris_model_requested must be linear or quadratic")
    input_calculated = reference_epoch_btjd + epochs.astype(float) * period_days
    selected_fit = (
        recomputed_quadratic if requested_model == "quadratic" else recomputed_linear
    )
    calculated_from_selected = _ephemeris_calculated_times(
        selected_fit, epochs, requested_model
    )
    if calculated_from_selected is None and requested_model == "quadratic":
        calculated_from_selected = _ephemeris_calculated_times(
            recomputed_linear, epochs, "linear"
        )
        used_model = "linear-fallback-quadratic-unavailable"
    elif calculated_from_selected is None:
        used_model = "input-ephemeris-fallback"
    else:
        used_model = requested_model
    calculated = (
        calculated_from_selected
        if calculated_from_selected is not None
        else input_calculated
    )
    oc_minutes = (observed - calculated) * 1440.0
    input_oc_minutes = (observed - input_calculated) * 1440.0
    rms_oc = float(np.sqrt(np.mean(oc_minutes**2))) if expected_length else None
    mean_uncertainty = float(np.mean(error_minutes)) if expected_length else None

    for field, reported, expected in (
        ("t_calculated_btjd", reported_calculated, calculated),
        ("oc_minutes", reported_oc, oc_minutes),
        ("input_ephemeris_oc_minutes", reported_input_oc, input_oc_minutes),
    ):
        if not np.allclose(reported, expected, rtol=1e-10, atol=1e-10):
            raise ValueError("timing {0} does not match its recomputed value".format(field))
    for field, expected in (
        ("n_transits_fit", int(expected_length)),
        ("n_excluded_no_detection", len(rejected_records)),
        ("n_rejected_epochs", len(rejected_records)),
        (
            "n_template_failures",
            sum(record["rejection_reason"] == "template-evaluation-failed" for record in rejected_records),
        ),
    ):
        if analysis.get(field) != expected:
            raise ValueError("timing {0} does not match its recomputed value".format(field))
    for field, expected in (
        ("rejected_epochs", rejected_records),
        ("uncertainty_clipped_epochs", clipped_records),
        ("search_boundary_epochs", boundary_records),
        ("linear_ephemeris", recomputed_linear),
    ):
        if not _timing_values_agree(analysis.get(field), expected):
            raise ValueError("timing {0} does not match its recomputed value".format(field))
    recomputed_models_comparison = fit_secular_timing_models(
        epochs, observed, timing_errors_days
    )
    for field, expected in (
        ("quadratic_ephemeris", recomputed_quadratic),
        ("ephemeris_model_comparison", recomputed_comparison),
        ("ephemeris_models_comparison", recomputed_models_comparison),
    ):
        if field in analysis and not _timing_values_agree(analysis.get(field), expected):
            raise ValueError("timing {0} does not match its recomputed value".format(field))
    if include_orbital_decay:
        if not _timing_values_agree(analysis.get("orbital_decay_fit"), recomputed_orbital_decay):
            raise ValueError("timing orbital_decay_fit does not match its recomputed value")
    elif "orbital_decay_fit" in analysis:
        raise ValueError("timing orbital_decay_fit requires an explicit opt-in request")
    if "ephemeris_model_requested" in analysis and analysis.get("ephemeris_model_requested") != requested_model:
        raise ValueError("timing ephemeris_model_requested does not match its recomputed value")
    if "ephemeris_model_used" in analysis and analysis.get("ephemeris_model_used") != used_model:
        raise ValueError("timing ephemeris_model_used does not match its recomputed value")
    if rms_oc is None:
        if analysis.get("oc_rms_minutes") is not None:
            raise ValueError("timing oc_rms_minutes does not match its recomputed value")
    else:
        _timing_numbers_agree(analysis.get("oc_rms_minutes"), rms_oc, "oc_rms_minutes")
    if mean_uncertainty is None:
        if analysis.get("mean_uncertainty_minutes") is not None:
            raise ValueError("timing mean_uncertainty_minutes does not match its recomputed value")
    else:
        _timing_numbers_agree(
            analysis.get("mean_uncertainty_minutes"), mean_uncertainty, "mean_uncertainty_minutes"
        )
    if not isinstance(analysis.get("epoch_acceptance"), dict):
        raise ValueError("timing epoch_acceptance must be an object")

    result = {
        "n_transits_fit": int(expected_length),
        "n_rejected_epochs": len(rejected_records),
        "n_template_failures": sum(
            record["rejection_reason"] == "template-evaluation-failed" for record in rejected_records
        ),
        "oc_rms_minutes": rms_oc,
        "mean_uncertainty_minutes": mean_uncertainty,
        "epochs": [int(value) for value in epochs],
        "t_observed_btjd": [float(value) for value in observed],
        "t_calculated_btjd": [float(value) for value in calculated],
        "oc_minutes": [float(value) for value in oc_minutes],
        "input_ephemeris_oc_minutes": [float(value) for value in input_oc_minutes],
        "oc_error_minutes": [float(value) for value in error_minutes],
        "rejected_epochs": rejected_records,
        "uncertainty_clipped_epochs": clipped_records,
        "search_boundary_epochs": boundary_records,
        "per_epoch": per_epoch,
        "epoch_acceptance": analysis["epoch_acceptance"],
        "linear_ephemeris": recomputed_linear,
        "quadratic_ephemeris": recomputed_quadratic,
        "ephemeris_model_requested": requested_model,
        "ephemeris_model_used": used_model,
        "ephemeris_model_comparison": recomputed_comparison,
        "ephemeris_models_comparison": recomputed_models_comparison,
    }
    if recomputed_orbital_decay is not None:
        result["orbital_decay_fit"] = recomputed_orbital_decay
    return result


def plot_timing_diagram(
    epochs: Sequence[int],
    oc_minutes: Sequence[float],
    oc_errors_minutes: Sequence[float],
    output_path: Path,
) -> Path:
    """Render accepted O-C measurements and formal errors to a PNG diagram.

    Args:
        epochs (Sequence[int]): Transit-cycle labels.
        oc_minutes (Sequence[float]): Observed-minus-calculated values in
            minutes.
        oc_errors_minutes (Sequence[float]): Formal timing errors in minutes.
        output_path (Path): Candidate-local destination for the PNG image.

    Returns:
        Path: Written diagram path.

    Raises:
        OSError: The output directory or image cannot be written.

    Note:
        The plot visualizes the supplied formal measurements; it is not a
        statistical TTV-detection test.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        list(epochs),
        list(oc_minutes),
        yerr=list(oc_errors_minutes),
        fmt="o",
        color="gray",
        markeredgecolor="black",
        capsize=3,
        alpha=0.6,
    )
    ax.axhline(0.0, color="red", linestyle="--")
    ax.set_xlabel("Transit Epoch N")
    ax.set_ylabel("O-C (minutes)")
    ax.set_title("Transit Timing Diagram")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _companion_periods(workspace: CandidateWorkspace) -> List[float]:
    """Return declared companion orbital periods from the candidate config."""
    periods: List[float] = []
    for config_name in ("transit_config.json", "ephemeris.json"):
        config_path = workspace.path / "config" / config_name
        if not config_path.is_file():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        declared = payload.get("ttv_companion_period_days")
        if isinstance(declared, (int, float)) and not isinstance(declared, bool):
            if float(declared) > 0:
                periods.append(float(declared))
        companions = payload.get("companions")
        if isinstance(companions, list):
            for companion in companions:
                if not isinstance(companion, dict):
                    continue
                for key in ("period_days", "period"):
                    value = companion.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if float(value) > 0:
                            periods.append(float(value))
                        break
    return sorted(set(round(value, 6) for value in periods))


def enumerate_companion_super_periods(
    candidate_period_days: float, companion_periods_days: Sequence[float]
) -> List[Dict[str, Any]]:
    """Enumerate first-order resonance context on either side of a signal.

    The beat-frequency formula is defined for an ordered inner/outer pair, but
    an observed signal may itself be the inner or outer member. This helper
    orders each finite companion pair for the calculation while retaining the
    companion's relation to the observed signal in the diagnostic record.

    Exact resonances have an unbounded analytic super-period. They are recorded
    explicitly as such rather than serialised as a non-finite JSON number.
    """
    candidate_period = float(candidate_period_days)
    if not math.isfinite(candidate_period) or candidate_period <= 0.0:
        raise ValueError("candidate_period_days must be finite and positive")
    records: List[Dict[str, Any]] = []
    for companion_value in companion_periods_days:
        try:
            companion_period = float(companion_value)
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(companion_period)
            or companion_period <= 0.0
            or companion_period == candidate_period
        ):
            continue
        inner_period, outer_period = sorted((candidate_period, companion_period))
        relation = (
            "inner-companion" if companion_period < candidate_period else "outer-companion"
        )
        for resonance in (2, 3, 4):
            try:
                super_period = calculate_ttv_super_period(
                    inner_period, outer_period, j_resonance=resonance
                )
            except ValueError:
                continue
            records.append(
                {
                    "companion_period_days": round(companion_period, 6),
                    "companion_orbital_relation": relation,
                    "resonance_j": int(resonance),
                    "super_period_days": round(super_period, 4)
                    if math.isfinite(super_period)
                    else None,
                    "super_period_status": (
                        "finite" if math.isfinite(super_period) else "exact-resonance-unbounded"
                    ),
                }
            )
    return records


def _synthetic_timing_table(
    ttv_amplitude_minutes: float = 0.0,
    rng_seed: int = 17,
) -> Dict[str, np.ndarray]:
    """Deterministic demonstration light curve with injected transits.

    Transits are generated with the same batman template used by the fitter,
    optionally with a sinusoidal per-epoch TTV shift.
    """
    rng = np.random.default_rng(seed=rng_seed)
    demo_period_days = 3.5
    demo_epoch_btjd = 2.0
    depth_ppm = 2500.0
    ttv_cycles = 6
    cadence_days = 20.0 / 1440.0
    time = np.arange(0.0, 35.0, cadence_days)
    rho_solar = 1.0
    a_rs = stellar_density_a_rs(rho_solar, demo_period_days)
    _demo_dur = 0.12
    ephemeris = {
        "period_days": demo_period_days,
        "epoch_btjd": demo_epoch_btjd,
        "duration_days": _demo_dur,
        "depth_ppm": depth_ppm,
    }
    # TEST_FIXTURE: synthetic data needs an explicit, declared geometry; the
    # production runner instead loads these values from a candidate-local fit.
    template = transit_template_parameters(
        ephemeris,
        a_rs,
        impact_parameter=0.3,
        q1=0.3,
        q2=0.3,
    )
    ttv_amplitude_days = ttv_amplitude_minutes / 1440.0
    n_min = int(np.floor((np.min(time) - demo_epoch_btjd) / demo_period_days))
    n_max = int(np.ceil((np.max(time) - demo_epoch_btjd) / demo_period_days))
    flux = np.ones_like(time)
    for epoch in range(n_min, n_max + 1):
        t0_epoch = demo_epoch_btjd + epoch * demo_period_days
        shift = ttv_amplitude_days * math.sin(2.0 * np.pi * epoch / ttv_cycles)
        mask = (time > t0_epoch - 0.35) & (time < t0_epoch + 0.35)
        model = _template_flux(template, time[mask], t0_epoch + shift)
        if isinstance(model, np.ndarray):
            flux[mask] = model
    flux = flux + rng.normal(0.0, 250e-6, size=time.shape)
    flux_err = np.full_like(flux, 250e-6)
    sector_values = np.ones(time.size, dtype=int)
    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "sector": sector_values,
        "_period_days": demo_period_days,
        "_epoch_btjd": demo_epoch_btjd,
        "_duration_days": _demo_dur,
        "_depth_ppm": depth_ppm,
    }


def run_ttv_analysis(
    workspace: CandidateWorkspace,
    signal: Optional[str] = None,
    ephemeris_model: str = "linear",
    fit_orbital_decay: bool = False,
) -> Path:
    """Run candidate-local exploratory timing analysis for one signal.

    The runner requires provenance-bound photometry, a complete
    candidate-derived ephemeris, candidate-derived stellar parameters, and a
    signal-bound candidate-local transit-fit posterior.  It records accepted
    and rejected epochs, a formal refitted ephemeris, optional resonance
    super-period context, and a timing diagram when timings exist.

    Args:
        workspace (CandidateWorkspace): Workspace that owns required inputs and
            receives timing artifacts.
        signal (Optional[str]): Optional validated signal suffix.  ``None``
            selects the workspace's primary signal.
        ephemeris_model (str): O-C reference ephemeris, ``"linear"`` by
            default or optional ``"quadratic"`` when the formal fit is
            available. Both model records and their BIC comparison are kept.
        fit_orbital_decay (bool): Include the opt-in formal period-derivative
            diagnostic. The default keeps this result absent.

    Returns:
        Path: Candidate-local ``outputs/ttv_analysis_results`` JSON path for
        the selected signal.

    Raises:
        RuntimeError: Provenance-bound photometry, a candidate-derived
            ephemeris, candidate-derived stellar parameters, or a valid
            signal-bound transit-fit posterior are unavailable.
        ValueError: The selected signal suffix or timing inputs are invalid.
        OSError: Output artifacts cannot be written.

    Note:
        The output is an exploratory timing diagnostic.  It does not assert a
        TTV detection, infer a companion, or validate a planet.
    """
    signal = validate_signal_suffix(signal)
    if ephemeris_model not in ("linear", "quadratic"):
        raise ValueError("ephemeris_model must be one of: linear, quadratic")
    if not isinstance(fit_orbital_decay, bool):
        raise ValueError("fit_orbital_decay must be a Boolean")
    outputs_dir = workspace.path / "outputs"
    figures_dir = workspace.path / "figures"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = load_light_curve_table(workspace, require_raw_provenance=True)
    if table is None:
        raise RuntimeError("TTV analysis requires observed candidate photometry")
    source = "candidate-data"
    ephemeris = load_transit_ephemeris(workspace, signal=signal)
    if ephemeris["source"] == "synthetic-demo" or any(
        value == "synthetic-demo" for value in ephemeris.get("field_sources", {}).values()
    ):
        raise RuntimeError("TTV analysis requires a complete candidate-derived transit ephemeris")

    stellar = load_stellar_parameters(workspace)
    if stellar["source"] != "candidate-data":
        raise RuntimeError("TTV analysis requires complete candidate-derived stellar parameters")
    rho_prior_solar = float(stellar["mass_solar"]) / float(stellar["radius_solar"]) ** 3
    a_rs = stellar_density_a_rs(rho_prior_solar, ephemeris["period_days"])
    suffix = f".{signal.lstrip('.')}" if signal else ""
    template, template_provenance = _load_transit_fit_template(
        workspace, signal, ephemeris, a_rs
    )

    analysis_kwargs: Dict[str, Any] = {"ephemeris_model": ephemeris_model}
    if fit_orbital_decay:
        analysis_kwargs["include_orbital_decay"] = True
    analysis = transit_timing_analysis(
        table["time"], table["flux"], table["flux_err"], ephemeris, template, **analysis_kwargs
    )
    try:
        timing = _recompute_and_validate_timing_summary(
            analysis, ephemeris, include_orbital_decay=fit_orbital_decay
        )
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise RuntimeError(
            "TTV timing summary failed recompute-and-validate: {0}".format(exc)
        ) from exc
    timing_diagram = figures_dir / f"ttv_timing_diagram{suffix}.png"
    if timing["n_transits_fit"] > 0:
        plot_timing_diagram(
            timing["epochs"],
            timing["oc_minutes"],
            timing["oc_error_minutes"],
            timing_diagram,
        )
    else:
        timing_diagram = None

    companion_periods = _companion_periods(workspace)
    super_periods = enumerate_companion_super_periods(
        ephemeris["period_days"], companion_periods
    )

    payload = {
        "schema_version": "1.0",
        "candidate_id": workspace.candidate_id,
        "work_package": "TTV_ANALYSIS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "signal": signal,
        "ephemeris": {
            "period_days": ephemeris["period_days"],
            "epoch_btjd": ephemeris["epoch_btjd"],
            "source": ephemeris["source"],
            "field_sources": ephemeris.get("field_sources", {}),
        },
        "input_provenance": {
            "photometry_source": source,
            "photometry_files": [
                str(Path(path).relative_to(workspace.path)).replace("\\", "/")
                for path in table.get("input_files", [])
                if Path(path).is_relative_to(workspace.path)
            ],
            "stellar_parameters_source": stellar["source"],
            "transit_fit_artifact": template_provenance["artifact"],
        },
        "timing": {
            "template": template_provenance,
            **timing,
        },
        "companion_super_periods": super_periods,
        "timing_diagram": (
            str(timing_diagram.relative_to(workspace.path)).replace("\\", "/")
            if timing_diagram is not None
            else None
        ),
        "caveat": (
            "Per-transit timing is exploratory. Epochs require a positive local fixed-template "
            "depth with formal SNR at least 3, but this does not calibrate detection probability. "
            "Linear and optional quadratic ephemeris covariances/BIC are formal descriptive fits "
            "only. An optional period derivative is not evidence for orbital decay, and no TTV "
            "detection or companion inference is claimed by this artifact."
        ),
    }
    output_path = outputs_dir / f"ttv_analysis_results{suffix}.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_path
