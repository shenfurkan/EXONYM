"""Target-neutral phase curve and secondary eclipse engine.

Decomposes out-of-transit photometric time-series into physical orbital harmonic
components based on the BEER (Beaming, Ellipsoidal, and Reflection/emission) model
(Faigler & Mazeh 2011, Shporer 2017):
1. Reflection / Thermal Day-Night Emission: A_refl * cos(phi)
   Circular-orbit basis term; a hotspot offset or eccentric orbit requires a
   dedicated brightness-map model rather than this diagnostic basis.
2. Doppler Beaming / Boosting: A_beam * sin(phi)
   Relativistic modulation proportional to radial velocity K_RV / c (Loeb & Gaudi 2003).
3. Tidal Ellipsoidal Variations: -A_ellip * cos(2*phi)
   Tidal distortion of the host star by the companion, producing double-frequency modulation
   with minima at quadrature (phi = 0.25, 0.75) and maxima at conjunctions.
4. Second Harmonic Sine Control: A_sin2 * sin(2*phi)
   Astrophysically forbidden symmetric component used as a systematic / stellar activity null control.
5. Secondary Eclipse Box: a circular phase-0.5 control unless a compatible
   candidate-local eccentric transit posterior supplies a marginalized
   eclipse phase and duration template.

Uncertainties and covariances are estimated via a Generalized Estimating Equations (GEE)
Huber-White cluster-sandwich covariance estimator (Liang & Zeger 1986) grouped into 0.5-day
time blocks to remain robust against correlated red noise and stellar granulation.
"""

from __future__ import annotations

import json
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .inputs import load_light_curve_table, load_transit_ephemeris
from .lightcurve import phase_hours
from .workspace import CandidateWorkspace

PRIMARY_MASK_HALF_DURATIONS = 0.65  # Masking width around primary transit (fraction of duration)
BLOCK_DAYS = 0.5                    # Time cluster block size for sandwich covariance (days)
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
        "median": round(median_phase, 8),
        "p16": round(lower_phase, 8),
        "p84": round(upper_phase, 8),
        "credible_interval_wraps_phase_zero": bool(lower_phase > upper_phase),
        "half_width_phase": round(0.5 * (upper - lower), 8),
    }


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
    if (
        period_days <= 0.0
        or total_samples <= 0
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
    """
    duration_days = float(ephemeris["duration_days"])
    circular_arguments = {
        "secondary_eclipse_phase": 0.5,
        "secondary_eclipse_duration_days": duration_days,
        "secondary_eclipse_phase_samples": None,
        "secondary_eclipse_duration_samples_days": None,
        "secondary_eclipse_template_total_samples": None,
    }
    circular_report = {
        "mode": "circular-ephemeris-box-control",
        "phase": {"median": 0.5, "p16": 0.5, "p84": 0.5, "credible_interval_wraps_phase_zero": False, "half_width_phase": 0.0},
        "duration_hours": {"median": round(duration_days * 24.0, 8), "p16": round(duration_days * 24.0, 8), "p84": round(duration_days * 24.0, 8)},
        "caveat": "No compatible candidate-local eccentric posterior was available; the secondary box is a circular-orbit control.",
    }
    report_path = workspace.path / "outputs" / "mcmc_transit_fit.json"
    if not report_path.exists():
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

    sample_indices = np.linspace(
        0, chain.shape[0] - 1, min(chain.shape[0], SECONDARY_TEMPLATE_MAX_SAMPLES), dtype=int
    )
    sampled_chain = chain[sample_indices]
    eccentricity = sampled_chain[:, -2] ** 2 + sampled_chain[:, -1] ** 2
    omega_radians = np.arctan2(sampled_chain[:, -1], sampled_chain[:, -2])
    phase_samples, duration_ratios, occulting = _secondary_eclipse_geometry_samples(
        eccentricity, omega_radians, sampled_chain[:, 0], sampled_chain[:, 2]
    )
    if not np.any(occulting):
        raise RuntimeError("candidate-local eccentric posterior predicts no secondary occultation")
    valid_phase_samples = phase_samples[occulting]
    valid_duration_samples = duration_days * duration_ratios[occulting]
    duration_hours = valid_duration_samples * 24.0
    return (
        {
            "secondary_eclipse_phase": float(np.median(valid_phase_samples)),
            "secondary_eclipse_duration_days": float(np.median(valid_duration_samples)),
            "secondary_eclipse_phase_samples": valid_phase_samples,
            "secondary_eclipse_duration_samples_days": valid_duration_samples,
            "secondary_eclipse_template_total_samples": int(sampled_chain.shape[0]),
        },
        {
            "mode": "eccentric-posterior-marginalized-box-control",
            "phase": _circular_phase_summary(valid_phase_samples),
            "duration_hours": {
                "median": round(float(np.median(duration_hours)), 8),
                "p16": round(float(np.quantile(duration_hours, 0.16)), 8),
                "p84": round(float(np.quantile(duration_hours, 0.84)), 8),
            },
            "occultation_probability_from_sampled_posterior": round(
                float(np.count_nonzero(occulting)) / float(sampled_chain.shape[0]), 8
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
        correction = float(n_points) / max(n_points - n_params, 1)
    else:
        correction = n_groups / (n_groups - 1.0) * (n_points - 1.0) / (n_points - n_params)
    return correction * bread @ meat @ bread, n_groups


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
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Build the phase-curve regression design matrix.

    Columns are per-sector offsets and linear slopes, the four orbital harmonic
    components, and a secondary-eclipse control. The control is either a fixed
    box or a posterior-marginalized candidate-local eccentric template. Returns
    (design, names, cluster) with cluster grouping 0.5-day blocks per sector.
    """
    if not math.isfinite(secondary_eclipse_phase) or not 0.0 <= secondary_eclipse_phase < 1.0:
        raise ValueError("secondary eclipse phase must be finite and in [0, 1)")
    if secondary_eclipse_duration_days is None:
        secondary_eclipse_duration_days = duration_days
    if not math.isfinite(secondary_eclipse_duration_days) or secondary_eclipse_duration_days <= 0.0:
        raise ValueError("secondary eclipse duration must be finite and positive")
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

    angle = 2.0 * np.pi * phase_days / period_days
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
        if eclipse.shape != phase_days.shape or not np.all(np.isfinite(eclipse)) or np.any(eclipse < 0.0):
            raise ValueError("secondary eclipse template must be finite, non-negative, and cadence-aligned")
    physical_values = {
        "reflection_semiamplitude": -np.cos(angle),
        "beaming_semiamplitude": np.sin(angle),
        "ellipsoidal_semiamplitude": -np.cos(2.0 * angle),
        "second_harmonic_sine_control": np.sin(2.0 * angle),
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
) -> Dict[str, Any]:
    """Fit harmonic + eclipse components and return the component report."""
    period_days = ephemeris["period_days"]
    epoch_btjd = ephemeris["epoch_btjd"]
    duration_days = ephemeris["duration_days"]
    phase_days = phase_hours(time, period_days, epoch_btjd) / 24.0
    keep = (
        np.abs(phase_days) > primary_mask_half_durations * duration_days
    ) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
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
    )
    sigma = np.asarray(flux_err, dtype=float)
    weighted_design = design / sigma[:, None]
    coefficients = np.linalg.lstsq(weighted_design, flux / sigma, rcond=None)[0]
    model = design @ coefficients
    residual = flux - model
    covariance, n_clusters = cluster_sandwich_covariance(design, residual, sigma, cluster)
    errors = np.sqrt(np.diag(covariance))

    components: Dict[str, Dict[str, float]] = {}
    for name in PHYSICAL_COMPONENTS:
        index = names.index(name)
        value_ppm = float(coefficients[index] * 1e6)
        error_ppm = float(errors[index] * 1e6)
        components[name] = {
            "value_ppm": round(value_ppm, 3),
            "block_robust_error_ppm": round(error_ppm, 3),
            "significance_sigma": round(value_ppm / error_ppm if error_ppm > 0 else 0.0, 2),
            "three_sigma_absolute_upper_bound_ppm": round(abs(value_ppm) + 3.0 * error_ppm, 3),
        }

    max_significance = max(
        abs(item["significance_sigma"]) for item in components.values()
    )
    reflection = components["reflection_semiamplitude"]
    unphysical_reflection = (
        reflection["value_ppm"] < 0.0
        and abs(reflection["significance_sigma"]) >= 3.0
    )
    if unphysical_reflection:
        status = "unphysical_phase_harmonic_detected_systematics_limited"
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
        "primary_mask_half_width_hours": round(primary_mask_half_durations * duration_days * 24.0, 3),
        "secondary_box_phase": round(float(secondary_eclipse_phase), 8),
        "secondary_box_duration_hours": round(float(secondary_eclipse_duration_days or duration_days) * 24.0, 3),
        "secondary_box_template_method": secondary_template_method,
        "components": components,
        "maximum_absolute_significance_sigma": round(max_significance, 2),
    }


def _synthetic_phase_curve_table() -> Dict[str, np.ndarray]:
    """Deterministic test-only light curve with an injected reflection signal."""
    rng = np.random.default_rng(seed=13)
    demo_period_days = 3.5
    demo_epoch_btjd = 2.0
    demo_duration_days = 0.12
    cadence_days = 120.0 / 86400.0
    time = np.arange(0.0, 27.0, cadence_days)
    phase_days = (
        (time - demo_epoch_btjd + 0.5 * demo_period_days) % demo_period_days
    ) - 0.5 * demo_period_days
    angle = 2.0 * np.pi * phase_days / demo_period_days
    reflection = 150e-6 * (-np.cos(angle))
    flux = 1.0 + reflection + rng.normal(0.0, 400e-6, size=time.shape)
    flux_err = np.full_like(flux, 400e-6)
    sector_values = np.ones(time.size, dtype=int)
    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "sector": sector_values,
        "_duration_days": demo_duration_days,
        "_epoch_btjd": demo_epoch_btjd,
        "_period_days": demo_period_days,
    }


def run_phase_curve_search(workspace: CandidateWorkspace) -> Path:
    """Run the phase curve search and write outputs/phase_curve_results.json."""
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    table = load_light_curve_table(workspace, require_raw_provenance=True)
    if table is None:
        raise RuntimeError("phase-curve analysis requires observed candidate photometry")
    ephemeris = load_transit_ephemeris(workspace)
    required_fields = ("period_days", "epoch_btjd", "duration_days")
    if ephemeris.get("source") == "synthetic-demo" or any(
        ephemeris.get("field_sources", {}).get(field) == "synthetic-demo"
        for field in required_fields
    ):
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
        "schema_version": "1.1",
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
            "weighted simultaneous circular-harmonic and secondary-box-control regression with "
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
        "components": result["components"],
        "maximum_absolute_significance_sigma": result["maximum_absolute_significance_sigma"],
        "interpretation": (
            "Exploratory photometric diagnostic; the harmonic terms retain a circular-orbit "
            "basis and no physical amplitude or secondary-eclipse detection is claimed."
        ),
    }
    output_path = outputs_dir / "phase_curve_results.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_path
