"""Target-neutral phase curve and secondary eclipse engine.

Decomposes out-of-transit photometric time-series into physical orbital harmonic
components based on the BEER (Beaming, Ellipsoidal, and Reflection/emission) model
(Faigler & Mazeh 2011, Shporer 2017):
1. Reflection / Thermal Day-Night Emission: -A_refl * cos(phi)
   Circular-orbit basis term; a hotspot offset or eccentric orbit requires a
   dedicated brightness-map model rather than this diagnostic basis.
2. Doppler Beaming / Boosting: A_beam * sin(phi)
   BEER-basis modulation proportional to radial velocity K_RV / c.
3. Tidal Ellipsoidal Variations: -A_ellip * cos(2*phi)
   Tidal distortion of the host star by the companion, producing double-frequency modulation
   with maxima at quadrature (phi = 0.25, 0.75) and minima at conjunctions.
4. Second Harmonic Sine Control: A_sin2 * sin(2*phi)
   Astrophysically forbidden symmetric component used as a systematic / stellar activity null control.
5. Secondary Eclipse Box: a circular phase-0.5 control unless a compatible
   candidate-local eccentric transit posterior supplies a marginalized
   eclipse phase and duration template.

Uncertainties and covariances are estimated with a finite-sample-corrected
cluster-sandwich estimator grouped into day-based time blocks to expose some
within-block residual correlation.

Scientific Boundary:
    This is an exploratory circular-harmonic regression with a secondary-eclipse
    control.  Its amplitudes and formal significances are not calibrated
    detection probabilities, physical amplitude measurements, or validation
    evidence on their own.

Verified sources, units, and failure boundary
----------------------------------------------
The retained BEER source is Faigler & Mazeh (2011), ADS
``2011MNRAS.415.3921F``, DOI ``10.1111/j.1365-2966.2011.19011.x``; the
ellipsoidal context is Morris (1985), ADS ``1985ApJ...295..143M``, DOI
``10.1086/163359``; and secondary/phase-curve interpretation is Shporer (2017),
ADS ``2017PASP..129g2001S``, DOI ``10.1088/1538-3873/aa7112``.  Input time is
``BTJD_TDB`` days; flux/error are dimensionless normalized relative flux;
orbital phase and eccentricity are dimensionless; component amplitudes/errors
are ppm; covariance is ppm^2 after output scaling; duration/block widths are
days; and posterior angles are radians where named.  The regression is circular
unless a hash-matched eccentric posterior supplies only its secondary-control
template.  Missing coverage, invalid ephemeris, nonpositive error, or
mismatched posterior template fails; the result is never calibrated source
assignment, detection probability, or ``claim_eligible`` evidence.

The retained local corpus does not yet contain a primary source for the exact
cluster-sandwich implementation or its block construction.  Its covariance is
therefore a robust regression diagnostic, not a calibrated red-noise model.
"""

from __future__ import annotations

import json
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .constants import PARTS_PER_MILLION
from .inputs import is_complete_candidate_ephemeris, load_light_curve_table, load_transit_ephemeris
from .lightcurve import phase_hours
from .workspace import CandidateWorkspace

# ASTROPHYSICAL_PROVENANCE:
# 1. Primary transit masking width: half duration (0.5 * T14) expanded by a 30% baseline
# buffer (0.5 * 1.3 = 0.65) to ensure ingress/egress wings are completely excluded without
# encroaching into out-of-transit planetary reflection/emission. This is a
# declared diagnostic mask policy, not a separately calibrated physical relation.
TRANSIT_MASK_BUFFER_FRACTION = 0.30
PRIMARY_MASK_HALF_DURATIONS = 0.5 * (1.0 + TRANSIT_MASK_BUFFER_FRACTION)

# 2. Block size for cluster-robust Huber-White sandwich standard errors (Cameron et al. 2011).
# Grouping cadences into blocks accounts for red-noise temporal autocorrelation.
DEFAULT_SANDWICH_BLOCK_DAYS = 0.5
BLOCK_DAYS = DEFAULT_SANDWICH_BLOCK_DAYS
SECONDARY_TEMPLATE_MAX_SAMPLES = 512
SECONDARY_TEMPLATE_MAX_CELLS = 1000000

PHYSICAL_COMPONENTS = (
    "reflection_semiamplitude",
    "beaming_semiamplitude",
    "ellipsoidal_semiamplitude",
    "second_harmonic_sine_control",
    "secondary_eclipse_depth",
)


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant is not permitted: {0}".format(value))


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number is not permitted")
    return parsed


def _unique_json_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key is not permitted: {0}".format(key))
        result[key] = value
    return result


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_finite_json_float,
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("cannot read candidate-local transit-fit report: {0}".format(path)) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("candidate-local transit-fit report must be a JSON object: {0}".format(path))
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _circular_phase_summary(phases: np.ndarray) -> Dict[str, Any]:
    """Summarize orbital phases while retaining an interval that can cross zero."""
    values = np.asarray(phases, dtype=float)
    values = values[np.isfinite(values)] % 1.0
    if values.size == 0:
        raise ValueError("secondary-eclipse phase posterior has no finite samples")
    mean_angle = math.atan2(
        float(np.mean(np.sin(2.0 * np.pi * values))),
        float(np.mean(np.cos(2.0 * np.pi * values))),
    )
    reference = (mean_angle / (2.0 * np.pi)) % 1.0
    offsets = ((values - reference + 0.5) % 1.0) - 0.5
    lower, median, upper = (float(item) for item in np.quantile(offsets, [0.16, 0.5, 0.84]))
    lower_phase = (reference + lower) % 1.0
    median_phase = (reference + median) % 1.0
    upper_phase = (reference + upper) % 1.0
    return {
        "median": _phase_fraction(median_phase),
        "p16": _phase_fraction(lower_phase),
        "p84": _phase_fraction(upper_phase),
        "credible_interval_wraps_phase_zero": bool(lower_phase > upper_phase),
        "half_width_phase": float(0.5 * (upper - lower)),
    }


def _phase_fraction(phase_fraction: float) -> float:
    """Serialize a native-precision circular phase in the half-open interval [0, 1)."""
    return float(phase_fraction) % 1.0


def _true_anomaly_from_mean_anomaly(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    """Solve Kepler's equation and convert eccentric anomaly to true anomaly."""
    eccentric_anomaly = np.mod(np.asarray(mean_anomaly, dtype=float), 2.0 * np.pi)
    for _ in range(32):
        residual = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - np.mod(
            mean_anomaly, 2.0 * np.pi
        )
        step = residual / (1.0 - eccentricity * np.cos(eccentric_anomaly))
        eccentric_anomaly -= step
        if np.max(np.abs(step)) < 1e-13:
            break
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + eccentricity) * np.sin(0.5 * eccentric_anomaly),
        np.sqrt(1.0 - eccentricity) * np.cos(0.5 * eccentric_anomaly),
    )


def _orbital_harmonic_angle(
    phase_days: np.ndarray,
    period_days: float,
    eccentricity: float,
    argument_periastron_radians: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return true-anomaly offset from transit and the Keplerian beaming basis."""
    if not math.isfinite(eccentricity) or not 0.0 <= eccentricity < 1.0:
        raise ValueError("orbital eccentricity must be finite and in [0, 1)")
    if not math.isfinite(argument_periastron_radians):
        raise ValueError("argument of periastron must be finite")
    transit_true_anomaly = 0.5 * np.pi - argument_periastron_radians
    transit_eccentric_anomaly = 2.0 * np.arctan2(
        math.sqrt(1.0 - eccentricity) * math.sin(0.5 * transit_true_anomaly),
        math.sqrt(1.0 + eccentricity) * math.cos(0.5 * transit_true_anomaly),
    )
    transit_mean_anomaly = transit_eccentric_anomaly - eccentricity * math.sin(transit_eccentric_anomaly)
    true_anomaly = _true_anomaly_from_mean_anomaly(
        transit_mean_anomaly + 2.0 * np.pi * phase_days / period_days, eccentricity
    )
    offset = true_anomaly - transit_true_anomaly
    return offset, -(
        np.cos(true_anomaly + argument_periastron_radians)
        + eccentricity * math.cos(argument_periastron_radians)
    )


def _secondary_eclipse_geometry_samples(
    eccentricity: np.ndarray,
    omega_radians: np.ndarray,
    rp_rs: np.ndarray,
    impact_parameter: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return eclipse phase, duration ratio, and occultation mask per posterior draw.

    The conjunction locations use the edge-on Keplerian approximation. The
    duration ratio additionally uses the small-angle chord/velocity relation;
    it is intentionally a regression-template control, not an occultation fit.
    """
    eccentricity = np.asarray(eccentricity, dtype=float)
    omega_radians = np.asarray(omega_radians, dtype=float)
    rp_rs = np.asarray(rp_rs, dtype=float)
    impact_parameter = np.asarray(impact_parameter, dtype=float)
    if not (
        eccentricity.shape == omega_radians.shape == rp_rs.shape == impact_parameter.shape
    ):
        raise ValueError("eccentric posterior arrays must share one shape")

    finite = (
        np.isfinite(eccentricity)
        & np.isfinite(omega_radians)
        & np.isfinite(rp_rs)
        & np.isfinite(impact_parameter)
        & (eccentricity >= 0.0)
        & (eccentricity < 1.0)
        & (rp_rs > 0.0)
        & (impact_parameter >= 0.0)
    )
    phase = np.full(eccentricity.shape, np.nan, dtype=float)
    duration_ratio = np.full(eccentricity.shape, np.nan, dtype=float)
    if not np.any(finite):
        return phase, duration_ratio, np.zeros(eccentricity.shape, dtype=bool)

    e = eccentricity[finite]
    omega = omega_radians[finite]
    transit_true_anomaly = 0.5 * np.pi - omega
    occultation_true_anomaly = 1.5 * np.pi - omega

    def mean_anomaly(true_anomaly: np.ndarray) -> np.ndarray:
        eccentric_anomaly = 2.0 * np.arctan2(
            np.sqrt(1.0 - e) * np.sin(0.5 * true_anomaly),
            np.sqrt(1.0 + e) * np.cos(0.5 * true_anomaly),
        )
        return eccentric_anomaly - e * np.sin(eccentric_anomaly)

    phase[finite] = (
        np.mod(mean_anomaly(occultation_true_anomaly) - mean_anomaly(transit_true_anomaly), 2.0 * np.pi)
        / (2.0 * np.pi)
    )

    transit_speed_factor = 1.0 + e * np.sin(omega)
    occultation_speed_factor = 1.0 - e * np.sin(omega)
    ratio = transit_speed_factor / occultation_speed_factor
    b_transit = impact_parameter[finite]
    b_occultation = b_transit * ratio
    radius_sum = 1.0 + rp_rs[finite]
    transit_chord_squared = radius_sum**2 - b_transit**2
    occultation_chord_squared = radius_sum**2 - b_occultation**2
    occulting = (
        (transit_speed_factor > 0.0)
        & (occultation_speed_factor > 0.0)
        & (transit_chord_squared > 0.0)
        & (occultation_chord_squared > 0.0)
    )
    local_duration_ratio = np.full(e.shape, np.nan, dtype=float)
    local_duration_ratio[occulting] = (
        ratio[occulting]
        * np.sqrt(occultation_chord_squared[occulting] / transit_chord_squared[occulting])
    )
    duration_ratio[finite] = local_duration_ratio
    occultation_mask = finite & np.isfinite(phase) & np.isfinite(duration_ratio) & (duration_ratio > 0.0)
    return phase, duration_ratio, occultation_mask


def _posterior_secondary_eclipse_template(
    phase_days: np.ndarray,
    period_days: float,
    phase_samples: np.ndarray,
    duration_samples_days: np.ndarray,
    total_samples: int,
) -> np.ndarray:
    """Marginalize a box-eclipse control over posterior phase/duration draws."""
    phase_days = np.asarray(phase_days, dtype=float)
    phase_samples = np.asarray(phase_samples, dtype=float)
    duration_samples_days = np.asarray(duration_samples_days, dtype=float)
    if isinstance(total_samples, bool) or not isinstance(total_samples, (int, np.integer)):
        raise ValueError("secondary-eclipse posterior template sample count must be an integer")
    if (
        not math.isfinite(period_days)
        or period_days <= 0.0
        or total_samples <= 0
        or total_samples < phase_samples.size
        or phase_samples.size == 0
        or phase_samples.shape != duration_samples_days.shape
    ):
        raise ValueError("secondary-eclipse posterior template has invalid dimensions")
    if not (
        np.all(np.isfinite(phase_samples))
        and np.all((phase_samples >= 0.0) & (phase_samples < 1.0))
        and np.all(np.isfinite(duration_samples_days))
        and np.all(duration_samples_days > 0.0)
    ):
        raise ValueError("secondary-eclipse posterior template has invalid samples")

    template = np.zeros(phase_days.size, dtype=float)
    sample_chunk = max(1, SECONDARY_TEMPLATE_MAX_CELLS // max(1, phase_days.size))
    phase_fraction = phase_days / period_days
    for start in range(0, phase_samples.size, sample_chunk):
        stop = min(phase_samples.size, start + sample_chunk)
        phase_distance = np.abs(
            ((phase_fraction[:, None] - phase_samples[None, start:stop] + 0.5) % 1.0) - 0.5
        )
        half_duration_phase = 0.5 * duration_samples_days[None, start:stop] / period_days
        template += np.count_nonzero(phase_distance < half_duration_phase, axis=1)
    return template / float(total_samples)


def resolve_secondary_eclipse_control(
    workspace: CandidateWorkspace, ephemeris: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Select a circular or posterior-marginalized secondary-eclipse control.

    A compatible eccentric chain is candidate-local, candidate-derived, and
    explicitly tied to the ephemeris used for the phase-curve regression.

    Args:
        workspace (CandidateWorkspace): Workspace containing an optional
            candidate-local transit-fit report and numeric posterior chain.
        ephemeris (Dict[str, Any]): Current candidate-derived period, epoch,
            and primary-transit duration in days.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Regression arguments for
        :func:`fit_phase_curve_components` and a provenance-rich control report.
        The report names either a circular box control or an
        eccentric-posterior-marginalized box control.

    Raises:
        RuntimeError: A present transit-fit report or posterior chain is
            malformed, non-candidate-data, incompatible with the current
            ephemeris, or cannot provide a usable eccentric control.
        ValueError: Required duration or posterior values are invalid.

    Note:
        The eccentric control approximates conjunction geometry for a regression
        template.  It is not a complete occultation or brightness-map model.
    """
    duration_days = float(ephemeris["duration_days"])
    circular_arguments = {
        "secondary_eclipse_phase": 0.5,
        "secondary_eclipse_duration_days": duration_days,
        "secondary_eclipse_phase_samples": None,
        "secondary_eclipse_duration_samples_days": None,
        "secondary_eclipse_template_total_samples": None,
        "orbital_eccentricity": 0.0,
        "argument_periastron_radians": 0.5 * math.pi,
    }
    circular_report = {
        "mode": "circular-ephemeris-box-control",
        "phase": {"median": 0.5, "p16": 0.5, "p84": 0.5, "credible_interval_wraps_phase_zero": False, "half_width_phase": 0.0},
        "duration_hours": {
            "median": float(duration_days * 24.0),
            "p16": float(duration_days * 24.0),
            "p84": float(duration_days * 24.0),
        },
        "caveat": "No compatible candidate-local eccentric posterior was available; the secondary box is a circular-orbit control.",
    }
    report_path = workspace.path / "outputs" / "mcmc_transit_fit.json"
    if not report_path.exists():
        # SCIENTIFIC_BOUNDARY: An absent compatible eccentric posterior yields a
        # explicitly labeled circular control, never an unlabelled assumption.
        return circular_arguments, circular_report

    report = _read_json_object(report_path)
    if report.get("source") != "candidate-data":
        raise RuntimeError("candidate-local transit-fit report is not based on candidate data")
    fit_ephemeris = report.get("ephemeris")
    if not isinstance(fit_ephemeris, dict):
        raise RuntimeError("candidate-local transit-fit report has no ephemeris object")
    for field in ("period_days", "epoch_btjd"):
        try:
            current_value = float(ephemeris[field])
            fitted_value = float(fit_ephemeris[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("candidate-local transit-fit ephemeris is incomplete") from exc
        if not (math.isfinite(current_value) and math.isfinite(fitted_value)) or not math.isclose(
            current_value, fitted_value, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise RuntimeError("candidate-local eccentric fit is not compatible with the current transit ephemeris")
    model = report.get("model")
    if not isinstance(model, str):
        raise RuntimeError("candidate-local transit-fit report has no model description")
    if "eccentric orbit" not in model:
        circular_report["caveat"] = "The compatible candidate-local transit fit is circular; the secondary box is a circular-orbit control."
        circular_report["transit_fit"] = {
            "path": "outputs/mcmc_transit_fit.json",
            "sha256": _sha256_file(report_path),
        }
        return circular_arguments, circular_report

    chain_path = workspace.path / "outputs" / "mcmc_transit_fit_chain.npy"
    try:
        chain = np.asarray(np.load(str(chain_path), allow_pickle=False), dtype=float)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError("compatible eccentric transit fit requires its numeric posterior chain") from exc
    if chain.ndim != 2 or chain.shape[0] == 0 or chain.shape[1] < 9 or not np.all(np.isfinite(chain)):
        raise RuntimeError("candidate-local eccentric transit posterior chain has invalid dimensions or values")

    parameter_names = report.get("parameter_names")
    if parameter_names is None:
        # Backward-compatible handling for pre-contract artifacts. The
        # current producer records names; this fallback retains readability of
        # legacy candidate evidence while the test below pins its historical
        # final-coordinate layout.
        from .transit_fit import PARAMETER_NAMES_ECCENTRIC

        if tuple(PARAMETER_NAMES_ECCENTRIC[-2:]) != ("sqe_cosw", "sqe_sinw"):
            raise RuntimeError("legacy eccentric transit-fit chain layout is not recognized")
        rp_rs_index, impact_parameter_index = 0, 2
        sqe_cosw_index, sqe_sinw_index = -2, -1
    else:
        required_names = ("rp_rs", "impact_parameter", "sqe_cosw", "sqe_sinw")
        if (
            not isinstance(parameter_names, list)
            or not all(isinstance(name, str) for name in parameter_names)
            or len(parameter_names) != chain.shape[1]
            or any(parameter_names.count(name) != 1 for name in required_names)
        ):
            raise RuntimeError("candidate-local eccentric transit-fit parameter_names contract is invalid")
        rp_rs_index = parameter_names.index("rp_rs")
        impact_parameter_index = parameter_names.index("impact_parameter")
        sqe_cosw_index = parameter_names.index("sqe_cosw")
        sqe_sinw_index = parameter_names.index("sqe_sinw")

    sample_indices = np.linspace(
        0, chain.shape[0] - 1, min(chain.shape[0], SECONDARY_TEMPLATE_MAX_SAMPLES), dtype=int
    )
    sampled_chain = chain[sample_indices]
    eccentricity = (
        sampled_chain[:, sqe_cosw_index] ** 2 + sampled_chain[:, sqe_sinw_index] ** 2
    )
    omega_radians = np.arctan2(
        sampled_chain[:, sqe_sinw_index], sampled_chain[:, sqe_cosw_index]
    )
    phase_samples, duration_ratios, occulting = _secondary_eclipse_geometry_samples(
        eccentricity,
        omega_radians,
        sampled_chain[:, rp_rs_index],
        sampled_chain[:, impact_parameter_index],
    )
    if not np.any(occulting):
        raise RuntimeError("candidate-local eccentric posterior predicts no secondary occultation")
    valid_phase_samples = phase_samples[occulting]
    valid_duration_samples = duration_days * duration_ratios[occulting]
    valid_eccentricity = eccentricity[occulting]
    valid_omega_radians = omega_radians[occulting]
    duration_hours = valid_duration_samples * 24.0
    phase_summary = _circular_phase_summary(valid_phase_samples)
    mean_omega_radians = math.atan2(
        float(np.mean(np.sin(valid_omega_radians))),
        float(np.mean(np.cos(valid_omega_radians))),
    )
    return (
        {
            "secondary_eclipse_phase": float(phase_summary["median"]),
            "secondary_eclipse_duration_days": float(np.median(valid_duration_samples)),
            "secondary_eclipse_phase_samples": valid_phase_samples,
            "secondary_eclipse_duration_samples_days": valid_duration_samples,
            "secondary_eclipse_template_total_samples": int(sampled_chain.shape[0]),
            "orbital_eccentricity": float(np.median(valid_eccentricity)),
            "argument_periastron_radians": mean_omega_radians,
        },
        {
            "mode": "eccentric-posterior-marginalized-box-control",
            "phase": phase_summary,
            "duration_hours": {
                "median": float(np.median(duration_hours)),
                "p16": float(np.quantile(duration_hours, 0.16)),
                "p84": float(np.quantile(duration_hours, 0.84)),
            },
            "occultation_probability_from_sampled_posterior": float(
                float(np.count_nonzero(occulting)) / float(sampled_chain.shape[0])
            ),
            "posterior_template_samples": int(sampled_chain.shape[0]),
            "transit_fit": {
                "path": "outputs/mcmc_transit_fit.json",
                "sha256": _sha256_file(report_path),
                "chain_path": "outputs/mcmc_transit_fit_chain.npy",
                "chain_sha256": _sha256_file(chain_path),
            },
            "geometry": "edge-on Keplerian conjunction phase with small-angle chord/velocity duration ratio",
            "caveat": (
                "The secondary-box template is marginalized over retained eccentric posterior draws, "
                "but does not propagate ephemeris uncertainty, brightness-map/hotspot offsets, or "
                "correlated-noise false-alarm calibration."
            ),
        },
    )


def cluster_sandwich_covariance(
    design: np.ndarray,
    residual: np.ndarray,
    sigma: np.ndarray,
    cluster: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Calculate finite-sample-corrected cluster-sandwich covariance (Huber-White / Liang-Zeger).

    Computes:
        V = (X^T W X)^-1 [ sum_g (X_g^T W_g r_g r_g^T W_g X_g) ] (X^T W X)^-1
    with finite-sample degree-of-freedom correction:
        c = (G / (G - 1)) * ((N - 1) / (N - P))

    The 'bread' matrix uses a Moore-Penrose pseudo-inverse (np.linalg.pinv) to safely
    handle collinear design columns (e.g. duplicate baseline offsets across overlapping sectors).

    Args:
        design (np.ndarray): Regression design matrix with one row per retained
            cadence and one column per fitted parameter.
        residual (np.ndarray): Residual normalized relative flux values.
        sigma (np.ndarray): Positive normalized-flux uncertainties.
        cluster (np.ndarray): Integer cadence-cluster labels for robust
            covariance aggregation.

    Returns:
        Tuple[np.ndarray, int]: Parameter covariance matrix and the number of
        distinct covariance clusters.

    Note:
        This robust covariance addresses within-cluster correlation only.  It
        does not calibrate a false-alarm probability for any phase component.
    """
    weighted_design = design / sigma[:, None]
    weighted_residual = residual / sigma
    bread = np.linalg.pinv(weighted_design.T @ weighted_design)
    meat = np.zeros((design.shape[1], design.shape[1]))
    groups = np.unique(cluster)
    for group in groups:
        mask = cluster == group
        score = weighted_design[mask].T @ weighted_residual[mask]
        meat += np.outer(score, score)
    n_points, n_params = design.shape
    n_groups = len(groups)
    if n_groups < 2 or n_points <= n_params:
        # A cluster-robust covariance needs at least two clusters and positive
        # residual degrees of freedom. Zero is an explicit undefined-error
        # sentinel consumed by fit_phase_curve_components.
        return np.zeros((n_params, n_params), dtype=float), n_groups
    correction = n_groups / (n_groups - 1.0) * (n_points - 1.0) / (n_points - n_params)
    # NUMERICAL_GUARD: floating-point asymmetry in the triple product can
    # leave tiny negative values on the covariance diagonal; explicit
    # symmetrization keeps sqrt(diag(...)) free of NaN warnings.
    raw_cov = correction * (bread @ meat @ bread)
    return 0.5 * (raw_cov + raw_cov.T), n_groups


def build_design_matrix(
    time: np.ndarray,
    phase_days: np.ndarray,
    period_days: float,
    duration_days: float,
    sector_values: np.ndarray,
    block_days: float = BLOCK_DAYS,
    secondary_eclipse_phase: float = 0.5,
    secondary_eclipse_duration_days: float = None,
    secondary_eclipse_template: np.ndarray = None,
    orbital_eccentricity: float = 0.0,
    argument_periastron_radians: float = 0.5 * math.pi,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Build the phase-curve regression design matrix.

    Columns are per-sector offsets and linear slopes, the four orbital harmonic
    components, and a secondary-eclipse control. The control is either a fixed
    box or a posterior-marginalized candidate-local eccentric template. Returns
    (design, names, cluster) with cluster grouping 0.5-day blocks per sector.

    Mathematical Formulation:
        The circular basis contains reflection/emission proportional to
        ``-cos(phi)``, beaming proportional to ``sin(phi)``, ellipsoidal
        variation proportional to ``-cos(2 phi)``, and a second-harmonic sine
        null control.  The secondary column is a fixed or marginalized box
        template rather than an occultation light-curve model.

    Args:
        time (np.ndarray): Retained cadence times in days.
        phase_days (np.ndarray): Transit-centered orbital phase offsets in days.
        period_days (float): Positive orbital period in days.
        duration_days (float): Primary transit duration in days.
        sector_values (np.ndarray): Cadence-aligned integer segment labels.
        block_days (float): Time width in days for robust covariance clusters.
        secondary_eclipse_phase (float): Secondary control center as an orbital
            phase fraction in the half-open unit interval.
        secondary_eclipse_duration_days (float): Optional positive duration in
            days; the primary duration is used when omitted.
        secondary_eclipse_template (np.ndarray): Optional cadence-aligned,
            non-negative marginalized control template.

    Returns:
        Tuple[np.ndarray, List[str], np.ndarray]: Design matrix, matching
        column names, and covariance-cluster labels.

    Raises:
        ValueError: Secondary-control phase, duration, or template alignment is
            invalid.
    """
    try:
        period_days = float(period_days)
        duration_days = float(duration_days)
        block_days = float(block_days)
        secondary_eclipse_phase = float(secondary_eclipse_phase)
    except (TypeError, ValueError) as exc:
        raise ValueError("phase-curve periods, durations, blocks, and phases must be finite") from exc
    if not math.isfinite(period_days) or period_days <= 0.0:
        raise ValueError("orbital period must be finite and positive")
    if not math.isfinite(duration_days) or duration_days <= 0.0 or duration_days >= period_days:
        raise ValueError("primary transit duration must be finite, positive, and shorter than the orbital period")
    if not math.isfinite(block_days) or block_days <= 0.0:
        raise ValueError("block_days must be finite and positive")
    if not math.isfinite(secondary_eclipse_phase) or not 0.0 <= secondary_eclipse_phase < 1.0:
        raise ValueError("secondary eclipse phase must be finite and in [0, 1)")
    if secondary_eclipse_duration_days is None:
        secondary_eclipse_duration_days = duration_days
    try:
        secondary_eclipse_duration_days = float(secondary_eclipse_duration_days)
    except (TypeError, ValueError) as exc:
        raise ValueError("secondary eclipse duration must be finite and positive") from exc
    if (
        not math.isfinite(secondary_eclipse_duration_days)
        or secondary_eclipse_duration_days <= 0.0
        or secondary_eclipse_duration_days >= period_days
    ):
        raise ValueError("secondary eclipse duration must be finite, positive, and shorter than the orbital period")
    unique_sectors = sorted(int(value) for value in np.unique(sector_values))
    columns: List[np.ndarray] = []
    names: List[str] = []
    cluster = np.empty(len(time), dtype=int)
    group_offset = 0
    for sector_value in unique_sectors:
        in_sector = sector_values == sector_value
        columns.append(in_sector.astype(float))
        names.append(f"sector_{sector_value}_offset")
        centered_time = np.zeros(len(time))
        centered_time[in_sector] = time[in_sector] - np.median(time[in_sector])
        columns.append(centered_time)
        names.append(f"sector_{sector_value}_slope")
        local_block = np.floor(
            (time[in_sector] - np.min(time[in_sector])) / block_days
        ).astype(int)
        cluster[in_sector] = group_offset + local_block
        group_offset += int(np.max(local_block)) + 1

    true_anomaly_offset, beaming_basis = _orbital_harmonic_angle(
        phase_days,
        period_days,
        float(orbital_eccentricity),
        float(argument_periastron_radians),
    )
    if secondary_eclipse_template is None:
        phase_fraction = phase_days / period_days
        secondary_phase_distance = np.abs(
            ((phase_fraction - secondary_eclipse_phase + 0.5) % 1.0) - 0.5
        )
        eclipse = (
            secondary_phase_distance < 0.5 * secondary_eclipse_duration_days / period_days
        ).astype(float)
    else:
        eclipse = np.asarray(secondary_eclipse_template, dtype=float)
        if (
            eclipse.shape != phase_days.shape
            or not np.all(np.isfinite(eclipse))
            or np.any(eclipse < 0.0)
            or np.any(eclipse > 1.0)
        ):
            raise ValueError("secondary eclipse template must be finite, in [0, 1], and cadence-aligned")
    physical_values = {
        "reflection_semiamplitude": -np.cos(true_anomaly_offset),
        "beaming_semiamplitude": beaming_basis,
        "ellipsoidal_semiamplitude": -np.cos(2.0 * true_anomaly_offset),
        "second_harmonic_sine_control": np.sin(2.0 * true_anomaly_offset),
        "secondary_eclipse_depth": -eclipse,
    }
    for name, values in physical_values.items():
        columns.append(values)
        names.append(name)
    return np.column_stack(columns), names, cluster


def fit_phase_curve_components(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    sector_values: np.ndarray,
    ephemeris: Dict[str, Any],
    block_days: float = BLOCK_DAYS,
    primary_mask_half_durations: float = PRIMARY_MASK_HALF_DURATIONS,
    secondary_eclipse_phase: float = 0.5,
    secondary_eclipse_duration_days: float = None,
    secondary_eclipse_phase_samples: np.ndarray = None,
    secondary_eclipse_duration_samples_days: np.ndarray = None,
    secondary_eclipse_template_total_samples: int = None,
    orbital_eccentricity: float = 0.0,
    argument_periastron_radians: float = 0.5 * math.pi,
) -> Dict[str, Any]:
    """Fit circular harmonic and secondary-control components to retained flux.

    The primary-transit window is excluded, then weighted least squares fits
    per-segment baselines, orbital harmonics, and a secondary control together.
    Cluster-sandwich covariance supplies the reported block-robust component
    errors in ppm.

    Args:
        time (np.ndarray): Cadence times in BTJD days.
        flux (np.ndarray): Normalized relative flux.
        flux_err (np.ndarray): Positive normalized-flux uncertainties.
        sector_values (np.ndarray): Cadence-aligned observation-segment labels.
        ephemeris (Dict[str, Any]): Candidate-derived period, epoch, and
            duration in days.
        block_days (float): Robust covariance-cluster width in days.
        primary_mask_half_durations (float): Number of transit half-durations
            excluded around primary transit.
        secondary_eclipse_phase (float): Secondary-control phase fraction.
        secondary_eclipse_duration_days (float): Optional secondary duration in
            days.
        secondary_eclipse_phase_samples (np.ndarray): Optional posterior phase
            samples for a marginalized eccentric control.
        secondary_eclipse_duration_samples_days (np.ndarray): Optional paired
            posterior duration samples in days.
        secondary_eclipse_template_total_samples (int): Total posterior draws
            represented by the marginalized control template.

    Returns:
        Dict[str, Any]: Component amplitudes and robust errors in ppm, routing
        status, retained-coverage counts, and secondary-control metadata.

    Raises:
        ValueError: Coverage, eclipse control, or posterior-template inputs are
            insufficient or misaligned.

    Note:
        Formal component significance is a regression diagnostic, not a
        correlated-noise-calibrated detection significance.
    """
    try:
        period_days = float(ephemeris["period_days"])
        epoch_btjd = float(ephemeris["epoch_btjd"])
        duration_days = float(ephemeris["duration_days"])
        block_days = float(block_days)
        primary_mask_half_durations = float(primary_mask_half_durations)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("phase-curve ephemeris and mask settings must be finite physical values") from exc
    if (
        not math.isfinite(period_days)
        or period_days <= 0.0
        or not math.isfinite(epoch_btjd)
        or not math.isfinite(duration_days)
        or duration_days <= 0.0
        or duration_days >= period_days
    ):
        raise ValueError("phase-curve ephemeris requires 0 < duration_days < period_days")
    if not math.isfinite(block_days) or block_days <= 0.0:
        raise ValueError("block_days must be finite and positive")
    if not math.isfinite(primary_mask_half_durations) or primary_mask_half_durations <= 0.0:
        raise ValueError("primary_mask_half_durations must be finite and positive")

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    sector_values = np.asarray(sector_values)
    if (
        time.ndim != 1
        or flux.ndim != 1
        or flux_err.ndim != 1
        or sector_values.ndim != 1
        or not (time.shape == flux.shape == flux_err.shape == sector_values.shape)
    ):
        raise ValueError("phase-curve time, flux, uncertainty, and sector arrays must share one dimension")
    phase_days = phase_hours(time, period_days, epoch_btjd) / 24.0
    keep = (
        np.abs(phase_days) > primary_mask_half_durations * duration_days
    ) & np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    time = time[keep]
    phase_days = phase_days[keep]
    flux = flux[keep]
    flux_err = flux_err[keep]
    sector_values = sector_values[keep]
    if time.size < 100:
        raise ValueError("insufficient out-of-transit coverage for phase curve analysis")

    secondary_template = None
    if secondary_eclipse_phase_samples is not None or secondary_eclipse_duration_samples_days is not None:
        if secondary_eclipse_phase_samples is None or secondary_eclipse_duration_samples_days is None:
            raise ValueError("secondary eclipse posterior phase and duration samples must be provided together")
        if secondary_eclipse_template_total_samples is None:
            raise ValueError("secondary eclipse posterior template sample count is required")
        secondary_template = _posterior_secondary_eclipse_template(
            phase_days,
            period_days,
            secondary_eclipse_phase_samples,
            secondary_eclipse_duration_samples_days,
            secondary_eclipse_template_total_samples,
        )
        secondary_template_method = "posterior-marginalized-eccentric-box"
    else:
        secondary_template_method = "fixed-secondary-box"

    design, names, cluster = build_design_matrix(
        time,
        phase_days,
        period_days,
        duration_days,
        sector_values,
        block_days=block_days,
        secondary_eclipse_phase=secondary_eclipse_phase,
        secondary_eclipse_duration_days=secondary_eclipse_duration_days,
        secondary_eclipse_template=secondary_template,
        orbital_eccentricity=orbital_eccentricity,
        argument_periastron_radians=argument_periastron_radians,
    )
    sigma = np.asarray(flux_err, dtype=float)
    weighted_design = design / sigma[:, None]
    if weighted_design.shape[0] <= weighted_design.shape[1]:
        raise ValueError("phase-curve regression has insufficient residual degrees of freedom")
    if np.linalg.matrix_rank(weighted_design) != weighted_design.shape[1]:
        raise ValueError("phase-curve regression design is rank deficient")
    coefficients = np.linalg.lstsq(weighted_design, flux / sigma, rcond=None)[0]
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("phase-curve regression returned non-finite coefficients")
    model = design @ coefficients
    residual = flux - model
    covariance, n_clusters = cluster_sandwich_covariance(design, residual, sigma, cluster)
    covariance_diagonal = np.diag(covariance)
    safe_covariance_diagonal = np.where(
        np.isfinite(covariance_diagonal) & (covariance_diagonal > 0.0),
        covariance_diagonal,
        0.0,
    )
    errors = np.sqrt(safe_covariance_diagonal)

    components: Dict[str, Dict[str, Any]] = {}
    for name in PHYSICAL_COMPONENTS:
        index = names.index(name)
        value_ppm = float(coefficients[index] * PARTS_PER_MILLION)
        error_ppm = float(errors[index] * PARTS_PER_MILLION)
        has_defined_error = math.isfinite(error_ppm) and error_ppm > 0.0
        components[name] = {
            "value_ppm": float(value_ppm),
            "block_robust_error_ppm": float(error_ppm),
            # NUMERICAL_GUARD: a zero or invalid covariance error does not
            # represent a null detection significance or a finite upper bound.
            "significance_sigma": (
                float(value_ppm / error_ppm) if has_defined_error else None
            ),
            "three_sigma_absolute_upper_bound_ppm": (
                float(abs(value_ppm) + 3.0 * error_ppm)
                if has_defined_error
                else None
            ),
        }

    finite_significances = [
        abs(float(item["significance_sigma"]))
        for item in components.values()
        if isinstance(item["significance_sigma"], (int, float))
        and math.isfinite(float(item["significance_sigma"]))
    ]
    max_significance = max(finite_significances) if finite_significances else None
    reflection = components["reflection_semiamplitude"]
    # DIAGNOSTIC_REASONING: A significant negative reflection-basis amplitude
    # flags a circular-harmonic interpretation that is likely systematics-limited.
    unphysical_reflection = (
        reflection["value_ppm"] < 0.0
        and isinstance(reflection["significance_sigma"], (int, float))
        and abs(float(reflection["significance_sigma"])) >= 3.0
    )
    if unphysical_reflection:
        status = "unphysical_phase_harmonic_detected_systematics_limited"
    elif max_significance is None:
        status = "undefined_component_significance"
    elif max_significance < 3.0:
        status = "no_significant_phase_curve_component"
    else:
        status = "component_above_three_sigma_requires_followup"

    return {
        "status": status,
        "period_days": float(period_days),
        "epoch_btjd": float(epoch_btjd),
        "n_points_after_primary_transit_mask": int(time.size),
        "n_sectors": int(len(np.unique(sector_values))),
        "n_covariance_clusters": int(n_clusters),
        "primary_mask_half_width_hours": float(primary_mask_half_durations * duration_days * 24.0),
        "secondary_box_phase": _phase_fraction(secondary_eclipse_phase),
        "secondary_box_duration_hours": float(
            float(secondary_eclipse_duration_days or duration_days) * 24.0
        ),
        "secondary_box_template_method": secondary_template_method,
        "orbital_geometry": {
            "eccentricity": float(orbital_eccentricity),
            "argument_periastron_radians": float(argument_periastron_radians),
            "harmonic_basis": "Keplerian true-anomaly conjunction geometry",
        },
        "components": components,
        "maximum_absolute_significance_sigma": (
            float(max_significance) if max_significance is not None else None
        ),
    }


def run_phase_curve_search(workspace: CandidateWorkspace) -> Path:
    """Run the candidate-local exploratory phase-curve regression.

    Args:
        workspace (CandidateWorkspace): Workspace that owns provenance-bound
            photometry, a candidate-derived ephemeris, and output artifacts.

    Returns:
        Path: Candidate-local ``outputs/phase_curve_results.json`` with the
        component report, secondary-control provenance, and calibration limits.

    Raises:
        RuntimeError: Required photometry or a complete candidate-derived
            ephemeris is unavailable, or the secondary control is unusable.
        ValueError: Retained regression data or control inputs are invalid.
        OSError: The result artifact cannot be written.

    Note:
        The artifact is an exploratory phase-curve diagnostic.  It cannot
        establish a secondary eclipse, a physical harmonic amplitude, or a
        validation result.
    """
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    table = load_light_curve_table(workspace, require_raw_provenance=True)
    if table is None:
        raise RuntimeError("phase-curve analysis requires observed candidate photometry")
    ephemeris = load_transit_ephemeris(workspace)
    if not is_complete_candidate_ephemeris(ephemeris, require_depth=False):
        raise RuntimeError("phase-curve analysis requires a complete candidate-derived transit ephemeris")
    source = "candidate-data"
    secondary_arguments, secondary_control = resolve_secondary_eclipse_control(workspace, ephemeris)

    result = fit_phase_curve_components(
        table["time"],
        table["flux"],
        table["flux_err"],
        table["sector"],
        ephemeris,
        **secondary_arguments
    )
    payload = {
        "schema_version": "1.2",
        "work_package": "PHASE_CURVE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "scientific_status": "exploratory-phase-curve-diagnostic",
        "validation_eligible": False,
        "validation_reason": (
            "This harmonic regression has no calibrated false-alarm or eclipse "
            "detection probability under correlated photometric noise."
        ),
        "method": (
            "weighted simultaneous Keplerian-harmonic and secondary-box-control regression with "
            "sector offsets/slopes and 0.5-day cluster-sandwich covariance"
        ),
        "status": result["status"],
        "period_days": result["period_days"],
        "epoch_btjd": result["epoch_btjd"],
        "n_points_after_primary_transit_mask": result["n_points_after_primary_transit_mask"],
        "n_sectors": result["n_sectors"],
        "n_covariance_clusters": result["n_covariance_clusters"],
        "primary_mask_half_width_hours": result["primary_mask_half_width_hours"],
        "secondary_eclipse_control": secondary_control,
        "secondary_box_phase": result["secondary_box_phase"],
        "secondary_box_duration_hours": result["secondary_box_duration_hours"],
        "secondary_box_template_method": result["secondary_box_template_method"],
        "orbital_geometry": result["orbital_geometry"],
        "components": result["components"],
        "maximum_absolute_significance_sigma": result["maximum_absolute_significance_sigma"],
        "input_error_binning_convention": (
            "When the shared light-curve loader downsamples a sector, it retains "
            "the median reported per-cadence uncertainty in each bin; this is not "
            "the standard error of a binned median. The artifact does not infer "
            "whether a particular input required downsampling."
        ),
        "interpretation": (
            "Exploratory photometric diagnostic; the harmonic terms use the available Keplerian "
            "geometry and no physical amplitude or secondary-eclipse detection is claimed."
        ),
    }
    output_path = outputs_dir / "phase_curve_results.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_path
