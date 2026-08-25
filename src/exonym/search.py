"""Target-neutral transit search engine supporting BLS and native TLS.

Implements blind and targeted periodic transit detection algorithms across photometric
time-series without hardcoded candidate designations or ephemerides.

.. admonition:: Literature references
    - Kovács, Zucker & Mazeh 2002 (BLS algorithm)
    - Hippke & Heller 2019 (TLS algorithm, SDE ranking statistic)
    - ``methods/tls_search.md`` — SDE field definition and use-threads rationale
    - ``methods/detrending-and-transit-inference.md`` — density-duration relation,
      Mandel & Agol 2002 limb darkening
    - Ivezić et al. 2014 (Statistics, Data Mining, and Machine Learning in Astronomy) —
      BLS SNR, MAD/biweight estimators, 1.4826 normalisation factor
    - Perryman 2018 (The Exoplanet Handbook), §2 — transit contact points T₁–T₄,
      circular-orbit duration relation

1. Box Least Squares (BLS) Search (Kovács, Zucker & Mazeh 2002)
   -------------------------------------------------------------
   Uses Astropy's weighted ``BoxLeastSquares`` implementation to fit periodic
   step functions (top-hat boxes) defined by:

   - Trial period *P* in [P_min, P_max] days.
   - Fractional transit duration *q = T₁₄ / P*.
   - Transit epoch / centre time *T₀* (BTJD).
   - Transit depth *δ = ⟨y_out⟩ − ⟨y_in⟩*.

   The reported ``snr`` is the fitted depth divided by its formal uncertainty.
   **It is a ranking statistic only** — not a calibrated false-alarm probability,
   detection reliability, or population completeness measure.

   **Frequency grid** — To prevent periodogram under-resolution across a
   multi-sector baseline, Astropy's ``autopower`` method spaces trial
   frequencies by :math:`\\Delta f \\propto q / T_{\\rm baseline}^2` (duration
   over baseline-squared).  The natural period resolution near *P* ≈ *P_max*
   is :math:`\\Delta P \\approx 2\\,q\\,P^2 / T_{\\rm baseline}`.
   ``n_periods`` acts as a *floor*, never reducing the number of trials below
   the baseline-duration minimum.

   **BLS SNR** — Following Ivezić §10.3.2, the BLS signal-to-noise is

   .. math::

       {\\rm SNR}_{\\rm BLS}
       = \\frac{\\delta}{\\sigma_{\\rm out}} \\sqrt{n_{\\rm eff}},
       \\qquad
       n_{\\rm eff} = \\frac{n_{\\rm in}\\,n_{\\rm out}}{n_{\\rm in} + n_{\\rm out}}

   where δ is the transit depth, σ_out the out-of-transit RMS, and n_in, n_out
   the number of in/out-of-transit cadences.  The Astropy weighted fit reports
   a depth uncertainty whose ratio with the fitted depth is the field ``snr``.

2. Grid Resolution
   ----------------
   - Astropy's baseline-and-duration-aware frequency grid prevents a requested
     sparse scan from under-resolving a multi-sector light curve.
   - The ``frequency_factor`` cap (≤ 1.0) ensures the grid can only be
     *oversampled* relative to the Astropy default, never coarser.

3. Optional Transit Least Squares (TLS) (Hippke & Heller 2019)
   ------------------------------------------------------------
   Integrates realistic physical limb-darkened transit shapes
   (Mandel & Agol 2002) with ingress/egress morphology, yielding higher
   sensitivity for shallow small-planet transits.  The ranking statistic is the
   **Signal Detection Efficiency** (SDE), defined as

   .. math::

       {\\rm SDE} = \\frac{{\\rm SR}_{\\rm peak} - \\langle{\\rm SR}\\rangle}
                         {\\sigma_{\\rm SR}}

   where SR is the TLS signal-residue statistic, ⟨SR⟩ its arithmetic mean
   across the searched period range, and σ_SR its standard deviation.

4. Harmonic alias checks
   ----------------------
   Downstream consumers (e.g. screening, activity analysis) test the dominant
   peak against *P*/2, 2*P*, and subharmonic aliases to discriminate the true
   orbital period from common observing-window harmonics.  The search engine
   itself returns only the strongest peak in the requested range.

5. Density-duration relation
   ---------------------------
   When asteroseismic or archival stellar density ρ_* is available, a physical
   duration grid can be derived from the circular-orbit transit duration:

   .. math::

       T_{14} = \\left(\\frac{3P}{\\pi^2 G\\rho_*}\\right)^{1/3}
                \\frac{\\sqrt{(1+k)^2 - b^2}}{\\sin i}

   where *k* = R_p/R_*, *b* = a cos i / R_* (impact parameter), and *i* is the
   orbital inclination (Perryman §2).

"""

from __future__ import annotations

import json
import logging
import re
import hashlib
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .inputs import MINIMUM_BLS_CANDIDATE_SNR, PIPELINE_NORMALIZATION
from .lightcurve import phase_hours
from .workspace import CandidateWorkspace, validate_signal_suffix


@dataclass
class BLSSearchResult:
    """Standardized result container for periodic transit searches.

    .. note::
        The ``snr`` field is the BLS fitted depth divided by its formal
        uncertainty — a **ranking statistic**, not a calibrated false-alarm
        probability (see Ivezić §10.3.2 for the BLS SNR derivation).  The
        ``detection_status`` field is a pipeline-level label reflecting whether
        the peak crossed the candidate-selection threshold; it is not a claim
        of planetary origin.

    Attributes
    ----------
    best_period : Optional[float]
        Optimal orbital period in **days**.  Derived from the peak of the BLS
        or TLS periodogram.  ``None`` when no peak crosses the threshold.
    best_epoch : Optional[float]
        Transit epoch (centre time T₀) in **BTJD** (BJD_TDB − 2_457_000).
    best_depth_ppm : Optional[float]
        Fitted transit depth in **parts per million**.  For BLS this is the
        box-car depth :math:`\\langle y_{\\rm out}\\rangle - \\langle y_{\\rm
        in}\\rangle`; for TLS it is ``1 - min(flux_model)``.
    best_duration_hours : Optional[float]
        Total transit duration T₁₄ in **hours** (first to fourth contact).
    snr : Optional[float]
        BLS fitted depth divided by formal depth uncertainty; TLS returns
        ``None`` here (its SDE occupies a separate field in the raw result dict).
    n_distinct_transit_events : int
        Number of unique transit epochs (integer-rounded event numbers)
        observed within the dataset for the best period and epoch.
    n_period_trials : int
        Total number of period samples evaluated in the periodogram grid.
    detection_status : str
        Pipeline label: ``"detected"`` when the peak exceeds
        ``MINIMUM_BLS_CANDIDATE_SNR`` and has ≥ 2 observed events, else
        ``"no-detection"``.
    best_depth_uncertainty_ppm : Optional[float]
        Formal uncertainty of the fitted depth in **ppm** from the weighted
        BLS fit.  Propagated from ``periodogram.depth_err``.
    """

    best_period: Optional[float]
    best_epoch: Optional[float]
    best_depth_ppm: Optional[float]
    best_duration_hours: Optional[float]
    snr: Optional[float]
    n_distinct_transit_events: int
    n_period_trials: int = 0
    detection_status: str = "detected"
    best_depth_uncertainty_ppm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_period": float(self.best_period) if self.best_period is not None else None,
            "best_epoch": float(self.best_epoch) if self.best_epoch is not None else None,
            "best_depth_ppm": float(self.best_depth_ppm) if self.best_depth_ppm is not None else None,
            "best_duration_hours": float(self.best_duration_hours) if self.best_duration_hours is not None else None,
            "snr": float(self.snr) if self.snr is not None else None,
            "n_distinct_transit_events": int(self.n_distinct_transit_events),
            "n_period_trials": int(self.n_period_trials),
            "detection_status": self.detection_status,
            "best_depth_uncertainty_ppm": (
                float(self.best_depth_uncertainty_ppm)
                if self.best_depth_uncertainty_ppm is not None
                else None
            ),
        }


def _frequency_period_grid(period_min: float, period_max: float, n_periods: int) -> np.ndarray:
    """Return trial periods uniformly sampled in orbital frequency.

    Mathematical Formulation
    ------------------------
    A uniform period grid :math:`P_k = P_{\\rm min} + k\\,\\Delta P` under-resolves
    long-period signals on a multi-sector baseline because a fixed period step
    corresponds to an ever-narrowing frequency step as period increases:

    .. math::

        \\Delta f \\approx \\frac{\\Delta P}{P^2}.

    By instead sampling *frequency* uniformly,

    .. math::

        f_k = f_{\\rm min} + k\\,\\frac{f_{\\rm max} - f_{\\rm min}}{N-1},
        \\qquad
        P_k = 1 / f_k,

    the periodogram maintains approximately constant phase-drift resolution
    at all trial periods.  This is equivalent to requiring
    :math:`\\Delta f \\lesssim 1/T_{\\rm baseline}` (one cycle across the
    observational baseline) at the longest periods searched.

    Parameters
    ----------
    period_min : float
        Shortest trial period in days (> 0).
    period_max : float
        Longest trial period in days (> period_min).
    n_periods : int
        Number of frequency samples (≥ 2).

    Returns
    -------
    np.ndarray
        Periods in **decreasing** order (from ``period_max`` to ``period_min``).
        Callers must not infer a ranking from the ordering.
    """
    if isinstance(n_periods, bool) or not isinstance(n_periods, (int, np.integer)):
        raise ValueError("n_periods must be an integer of at least two")
    if n_periods < 2:
        raise ValueError("n_periods must be an integer of at least two")
    frequencies = np.linspace(1.0 / period_max, 1.0 / period_min, int(n_periods))
    return 1.0 / frequencies


def _uncertainties_for_bls(
    values: np.ndarray, flux_err: Optional[Sequence[float]]
) -> np.ndarray:
    """Return finite positive per-cadence uncertainties for weighted BLS.

    Candidate-product loading normally supplies reported uncertainties. Public
    array callers may omit them, in which case a robust constant scatter is
    used **solely to retain a dimensionless ranking statistic**; the
    candidate-facing runner records that fallback in the search provenance.

    Mathematical Formulation
    ------------------------
    When uncertainties are absent, a robust scatter estimate is computed from
    the **median absolute deviation** (MAD), rescaled to approximate the
    standard deviation of a Gaussian distribution (Ivezić §3.4.2):

    .. math::

        \\sigma_{\\rm robust}
        = 1.4826 \\times {\\rm MAD},
        \\qquad
        {\\rm MAD} = {\\rm median}\\bigl(|y_i - {\\rm median}(y)|\\bigr).

    The factor 1.4826 = 1 / Φ⁻¹(0.75) is the asymptotic normalisation such
    that, for Gaussian noise, σ_robust → σ.

    Parameters
    ----------
    values : np.ndarray
        Flux values used for the scatter estimate when errors are absent.
    flux_err : Optional[Sequence[float]]
        Reported per-cadence uncertainties or ``None``.

    Returns
    -------
    np.ndarray
        Strictly positive finite uncertainties matching the shape of ``values``.

    Raises
    ------
    ValueError
        If ``flux_err`` is provided but contains non-finite or non-positive
        entries, or its shape mismatches ``values``.
    """
    if flux_err is not None:
        errors = np.asarray(flux_err, dtype=float)
        if errors.shape != values.shape:
            raise ValueError("flux_err must match the time and flux shapes")
        if not np.all(np.isfinite(errors) & (errors > 0)):
            raise ValueError("flux_err must contain only positive finite values")
        return errors

    median = float(np.median(values))
    # NUMERICAL_GUARD: 1.4826 * MAD ≈ σ for Gaussian noise (Ivezić Eq. 3.37);
    # the factor is the inverse of the normal CDF at 0.75.
    mad = float(np.median(np.abs(values - median)))
    scatter = 1.4826 * mad
    # NUMERICAL_GUARD: non-finite or negative scatter from degenerate data
    # (e.g. all values identical) → fall back to a tiny constant.
    if not np.isfinite(scatter) or scatter <= 0:
        scatter = 1e-4
    return np.full(values.size, scatter, dtype=float)


def _baseline_aware_frequency_factor(
    time: np.ndarray,
    period_min: float,
    period_max: float,
    duration_days: float,
    requested_minimum_trials: int,
) -> float:
    """Select an Astropy BLS frequency factor without under-resolving a grid.

    ``BoxLeastSquares.autopower`` uses a duration/baseline-squared frequency
    scale. A caller can request *more* samples through ``n_periods`` but never
    a coarser-than-standard scan, preventing a fixed sparse grid from missing
    narrow, long-baseline signals.

    Mathematical Formulation
    ------------------------
    Astropy's default frequency step for a duration *q* (in days) across a
    baseline *T* is

    .. math::

        \\Delta f_{\\rm natural} = \\frac{q}{T^2}.

    The number of natural trials over the frequency span
    :math:`f_{\\rm max} - f_{\\rm min}` is therefore

    .. math::

        N_{\\rm natural}
        = \\frac{1/P_{\\rm min} - 1/P_{\\rm max}}{\\Delta f_{\\rm natural}}.

    The returned ``frequency_factor`` is the ratio
    :math:`N_{\\rm natural} / N_{\\rm requested}`, capped at 1.0 so that the
    grid is never coarser than the Astropy default.

    Parameters
    ----------
    time : np.ndarray
        Observation times in BTJD.
    period_min : float
        Shortest searched period in days.
    period_max : float
        Longest searched period in days.
    duration_days : float
        Trial transit duration in days.
    requested_minimum_trials : int
        The ``n_periods`` value passed by the caller.

    Returns
    -------
    float
        Frequency factor in (0, 1.0]; 1.0 preserves full Astropy resolution.
    """
    # NUMERICAL_GUARD: baseline must be finite and positive.
    baseline_days = float(np.ptp(time))
    if not np.isfinite(baseline_days) or baseline_days <= 0:
        raise ValueError("BLS requires observations spanning a positive time baseline")
    # Astropy natural frequency step: Δf = q / T².
    natural_step = duration_days / (baseline_days * baseline_days)
    frequency_span = 1.0 / period_min - 1.0 / period_max
    # NUMERICAL_GUARD: at least 2 trials to define a frequency span.
    natural_trials = max(2, int(np.ceil(frequency_span / natural_step)) + 1)
    # Scale frequency_factor so the evaluated grid size matches requested_minimum_trials.
    return max(1e-4, natural_trials / float(requested_minimum_trials))


def _distinct_transit_events(
    time: np.ndarray, period: float, epoch: float, duration_hours: float
) -> int:
    """Count observed event windows containing at least one cadence.

    Each transit event is identified by its integer-rounded epoch number
    :math:`n = {\\rm round}((t_i - T_0) / P)`.  Cadences whose phase falls
    within ±½ T₁₄ of the nominal epoch centre are tagged as in-transit and
    binned by their epoch number.  The returned count is the number of
    **unique** epoch numbers represented.

    This integer-rounding approach is robust against finite cadence sampling:
    a transit that spans two consecutive integer epoch bins still contributes
    one distinct event per epoch, while the same epoch observed across
    multiple cadences (e.g. 30-minute sampling of a 3-hour transit) is not
    double-counted.

    Parameters
    ----------
    time : np.ndarray
        Observation times in BTJD.
    period : float
        Trial period in days.
    epoch : float
        Transit epoch (T₀) in BTJD.
    duration_hours : float
        Total transit duration T₁₄ in hours.

    Returns
    -------
    int
        Number of unique transit event epochs with at least one in-transit
        cadence.  Returns 0 if no cadences lie within any transit window.
    """
    # Identify cadences within ±½ T₁₄ of the transit centre.
    in_transit = np.abs(phase_hours(time, period, epoch)) <= 0.5 * duration_hours
    if not np.any(in_transit):
        return 0
    # Integer-round epoch number: n_i = round((t_i − T₀) / P).
    event_numbers = np.rint((time[in_transit] - epoch) / period).astype(int)
    return int(np.unique(event_numbers).size)


def find_transits(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    period_min: float = 0.5,
    period_max: float = 15.0,
    n_periods: int = 2000,
    duration_hours: float = 3.0,
    flux_err: Optional[Sequence[float]] = None,
) -> BLSSearchResult:
    """Run a target-neutral BLS periodogram search over a light curve.

    Implements the Kovács, Zucker & Mazeh (2002) Box Least Squares algorithm
    via Astropy's weighted ``BoxLeastSquares.autopower``.  A periodic top-hat
    (box-car) transit with fractional duration :math:`q = T_{14} / P` is
    fitted at each trial frequency; the resulting depth and formal uncertainty
    yield the ``snr`` field.

    Mathematical Formulation
    ------------------------
    The BLS statistic at each trial period and epoch is the fitted depth
    :math:`\\delta = \\langle y_{\\rm out}\\rangle - \\langle y_{\\rm
    in}\\rangle` divided by its formal uncertainty.  For weighted data with
    per-cadence errors σᵢ, the BLS SNR reduces to (Ivezić §10.3.2):

    .. math::

        {\\rm SNR}_{\\rm BLS}
        = \\frac{\\delta}{\\sigma_{\\rm out}} \\sqrt{n_{\\rm eff}},

    where n_eff = (n_in · n_out) / (n_in + n_out) accounts for the effective
    number of independent measurements.

    Astrophysical Rationale
    -----------------------
    - The trial grid is generated by Astropy using the observed time baseline
      and the requested transit duration.  ``n_periods`` is a **minimum**
      requested density: the ``frequency_factor`` adaptor ensures the grid can
      be oversampled but never made coarser than the Astropy default.
    - A selected peak must contain at least **two** observed distinct transit
      events and cross ``MINIMUM_BLS_CANDIDATE_SNR``.  That threshold is an
      empirical cut, not a calibrated false-alarm probability.
    - ``snr`` is retained as a compatibility field name only; it is not a
      calibrated detection significance, false-alarm probability, or
      reliability estimate.

    Parameters
    ----------
    time_btjd : Sequence[float]
        Observation times in BTJD (BJD_TDB − 2_457_000).
    flux : Sequence[float]
        Normalised flux values (median ≈ 1 in the out-of-transit baseline).
    period_min : float
        Shortest searched orbital period in days.  Default 0.5 d.
    period_max : float
        Longest searched orbital period in days.  Default 15.0 d.
    n_periods : int
        Minimum number of trial periods (acts as a floor; Astropy may use
        more).  Default 2000.
    duration_hours : float
        Fixed trial transit duration T₁₄ in hours.
    flux_err : Optional[Sequence[float]]
        Per-cadence normalised flux uncertainties.  When ``None``, a robust
        constant scatter is used; the candidate-facing runner records this
        fallback in the provenance.

    Returns
    -------
    BLSSearchResult
        Standardised container with optimal period, epoch, depth, duration,
        SNR, and detection status.

    Raises
    ------
    ValueError
        If input arrays are incompatible, too few points, or the period /
        duration bounds are invalid.
    RuntimeError
        If the ``astropy`` core dependency is absent.
    """
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)

    if time.shape != values.shape:
        raise ValueError("time and flux must have matching shapes")
    # Filter to finite time, flux, and (when provided) finite positive errors.
    finite = np.isfinite(time) & np.isfinite(values)
    if flux_err is not None:
        raw_errors = np.asarray(flux_err, dtype=float)
        if raw_errors.shape != values.shape:
            raise ValueError("flux_err must match the time and flux shapes")
        finite &= np.isfinite(raw_errors) & (raw_errors > 0)
        raw_errors = raw_errors[finite]
    else:
        raw_errors = None
    time = time[finite]
    values = values[finite]

    # NUMERICAL_GUARD: Astropy's BLS requires a minimum of 2 transits × a
    # handful of cadences each; 50 points is a pragmatic floor.
    if time.size < 50:
        raise ValueError("insufficient data points for BLS transit search")
    if period_min <= 0 or period_max <= period_min:
        raise ValueError("invalid period search bounds")
    # Validate the frequency grid without using its return value (side-effect check).
    _frequency_period_grid(period_min, period_max, n_periods)

    try:
        from astropy.timeseries import BoxLeastSquares
    except ImportError as exc:  # pragma: no cover - declared core dependency
        raise RuntimeError("BLS search requires the core astropy dependency") from exc

    duration_days = float(duration_hours) / 24.0
    if not np.isfinite(duration_days) or duration_days <= 0:
        raise ValueError("duration_hours must be positive and finite")
    errors = _uncertainties_for_bls(values, raw_errors)
    frequency_factor = _baseline_aware_frequency_factor(
        time, period_min, period_max, duration_days, n_periods
    )
    # Build weighted BLS periodogram via Astropy's C-optimised "fast" solver.
    # ``objective="likelihood"`` uses the chi-squared-based statistic for which
    # the formal depth uncertainty is well-defined (see Astropy BLS docs).
    # ``minimum_n_transit=2`` guards against single-event false positives.
    periodogram = BoxLeastSquares(time, values, dy=errors).autopower(
        duration_days,
        objective="likelihood",
        method="fast",
        minimum_n_transit=2,
        minimum_period=period_min,
        maximum_period=period_max,
        frequency_factor=frequency_factor,
    )
    # NUMERICAL_GUARD: require positive finite depth and depth_err; negative
    # depths are unphysical and infinite errors indicate degenerate fits.
    valid = (
        np.isfinite(periodogram.power)
        & np.isfinite(periodogram.period)
        & np.isfinite(periodogram.transit_time)
        & np.isfinite(periodogram.depth)
        & np.isfinite(periodogram.depth_err)
        & (periodogram.depth > 0)
        & (periodogram.depth_err > 0)
    )
    # Walk peaks in descending periodogram power until one satisfies all
    # quality gates (finite geometry, ≥ 2 observed events, finite SNR).
    best: Optional[Dict[str, float]] = None
    for index in np.argsort(periodogram.power)[::-1]:
        if not valid[index]:
            continue
        period = float(periodogram.period[index])
        epoch = float(periodogram.transit_time[index])
        n_events = _distinct_transit_events(time, period, epoch, duration_hours)
        if n_events < 2:
            continue
        depth = float(periodogram.depth[index])
        depth_err = float(periodogram.depth_err[index])
        # SCIENTIFIC_BOUNDARY: this SNR is a ranking statistic only — see
        # module docstring §1 for the BLS SNR derivation.
        snr = depth / depth_err
        if not np.isfinite(snr):
            continue
        best = {
            "period": period,
            "epoch": epoch,
            "depth": depth,
            "depth_err": depth_err,
            "snr": snr,
            "n_events": n_events,
        }
        break
    if best is None:
        return BLSSearchResult(
            best_period=None,
            best_epoch=None,
            best_depth_ppm=None,
            best_duration_hours=None,
            snr=None,
            n_distinct_transit_events=0,
            n_period_trials=int(periodogram.period.size),
            detection_status="no-detection",
        )
    # ASTROPHYSICAL_HEURISTIC: MINIMUM_BLS_CANDIDATE_SNR is a candidate-
    # selection threshold, not a calibrated FAP (see inputs.py constant).
    if best["snr"] < MINIMUM_BLS_CANDIDATE_SNR:
        return BLSSearchResult(
            best_period=None,
            best_epoch=None,
            best_depth_ppm=None,
            best_duration_hours=None,
            snr=best["snr"],
            n_distinct_transit_events=int(best["n_events"]),
            n_period_trials=int(periodogram.period.size),
            detection_status="no-detection",
            best_depth_uncertainty_ppm=best["depth_err"] * 1e6,
        )

    # NUMERICAL_GUARD: clamp non-negative snr; the gate above ensures it is
    # at least the threshold, but floating-point drift is harmless.
    return BLSSearchResult(
        best_period=best["period"],
        best_epoch=best["epoch"],
        best_depth_ppm=best["depth"] * 1e6,
        best_duration_hours=duration_hours,
        snr=max(best["snr"], 0.0),
        n_distinct_transit_events=int(best["n_events"]),
        n_period_trials=int(periodogram.period.size),
        best_depth_uncertainty_ppm=best["depth_err"] * 1e6,
    )


def find_transits_duration_grid(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    duration_grid_hours: Sequence[float],
    period_min: float = 0.5,
    period_max: float = 15.0,
    n_periods: int = 2000,
    flux_err: Optional[Sequence[float]] = None,
) -> Tuple[BLSSearchResult, List[Dict[str, Any]]]:
    """Run the declared BLS duration grid and retain every trial result.

    The returned best result is selected by the same uncalibrated ranking
    statistic as an individual BLS search.  This is a deterministic model
    selection step, not an additional significance calibration.

    Astrophysical Rationale
    -----------------------
    A single fixed duration may misrepresent transits whose T₁₄ differs
    from the trial value: durations that are too short under-sample in-transit
    cadences and inflate scatter; durations that are too long dilute the
    box-car depth relative to the true signal.  Scanning a physically
    motivated grid — typically derived from the circular-orbit density
    relation (module docstring §5) or a broad prior such as
    [1.5, 3.0, 6.0, 12.0] hours — identifies the highest-SNR match.

    Parameters
    ----------
    time_btjd : Sequence[float]
        Observation times in BTJD.
    flux : Sequence[float]
        Normalised flux values.
    duration_grid_hours : Sequence[float]
        Non-empty, duplicate-free list of positive trial durations in hours.
    period_min, period_max : float
        Period search bounds in days.
    n_periods : int
        Minimum trial period count per duration.
    flux_err : Optional[Sequence[float]]
        Per-cadence uncertainties or ``None``.

    Returns
    -------
    Tuple[BLSSearchResult, List[Dict]]
        The best result across all durations (by SNR) and the full list of
        per-duration result payloads for provenance.
    """
    durations = [float(value) for value in duration_grid_hours]
    if not durations or any(not np.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("duration_grid_hours must contain positive finite values")
    if len(set(durations)) != len(durations):
        raise ValueError("duration_grid_hours must not contain duplicates")

    results: List[Tuple[BLSSearchResult, Dict[str, Any]]] = []
    for duration_hours in durations:
        result = find_transits(
            time_btjd,
            flux,
            flux_err=flux_err,
            period_min=period_min,
            period_max=period_max,
            n_periods=n_periods,
            duration_hours=duration_hours,
        )
        results.append((result, result.to_dict()))
    best, _ = max(
        results,
        key=lambda item: float(item[0].snr) if item[0].snr is not None else float("-inf"),
    )
    return best, [payload for _, payload in results]


def find_transits_tls(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    flux_err: Sequence[float],
    period_min: float = 0.5,
    period_max: float = 15.0,
    use_threads: int = 1,
) -> Dict[str, float]:
    """Run a weighted, native-cadence Transit Least Squares search.

    Implements Hippke & Heller (2019) via the ``transitleastsquares`` package
    (``transitleastsquares`` function).  Unlike BLS, TLS convolves the
    light curve with a **physically realistic limb-darkened transit model**
    (Mandel & Agol 2002), including ingress/egress morphology and quadratic
    limb-darkening coefficients.  This yields higher sensitivity to shallow,
    small-planet transits and native-cadence resolution.

    Mathematical Formulation
    ------------------------
    TLS ranks trial periods by the **Signal Detection Efficiency** (SDE),
    defined as the z-score of the signal-residue (SR) statistic across the
    searched period range (Hippke & Heller 2019, Eq. 8):

    .. math::

        {\\rm SDE} = \\frac{{\\rm SR}_{\\rm peak} - \\mu_{\\rm SR}}
                         {\\sigma_{\\rm SR}},

    where μ_SR is the arithmetic mean and σ_SR the standard deviation of SR
    over all trial periods.  The SDE is a ranking statistic; its relationship
    to a false-alarm probability depends on the noise properties of the
    specific light curve, detrending, and search bounds (see
    ``methods/tls_search.md``).

    Parameters
    ----------
    time_btjd : Sequence[float]
        Observation times in BTJD (BJD_TDB − 2_457_000).
    flux : Sequence[float]
        Normalised flux values (median ≈ 1).
    flux_err : Sequence[float]
        Per-cadence normalised flux uncertainties (must be finite, positive).
    period_min : float
        Shortest searched orbital period in days.  Default 0.5 d.
    period_max : float
        Longest searched orbital period in days.  Default 15.0 d.
    use_threads : int
        TLS worker count.  The default of 1 avoids TLS's multiprocessing path,
        which is unreliable in constrained Windows shells
        (``methods/tls_search.md``).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys ``best_period``, ``best_epoch``, ``best_depth_ppm``,
        ``best_duration_hours``, and ``sde``.  **This is a discovery ranking
        statistic, not a planetary-validation result.**

    Raises
    ------
    RuntimeError
        If the ``transitleastsquares`` package is absent (requires the
        ``[discovery]`` optional dependency group) or if TLS returns a
        non-physical solution.
    ValueError
        If input arrays are incompatible, too few points, or period/thread
        bounds are invalid.
    """
    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(flux_err, dtype=float)
    # Filter to finite time, flux, and finite positive errors.
    finite = np.isfinite(time) & np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    time = time[finite]
    values = values[finite]
    errors = errors[finite]
    # NUMERICAL_GUARD: same 50-point floor as BLS.
    if time.size < 50:
        raise ValueError("insufficient data points for TLS transit search")
    if period_min <= 0 or period_max <= period_min:
        raise ValueError("invalid period search bounds")
    if use_threads < 1:
        raise ValueError("TLS use_threads must be at least one")

    try:
        from transitleastsquares import transitleastsquares
    except ImportError as exc:
        raise RuntimeError(
            "TLS search requires the optional 'discovery' dependency group"
        ) from exc

    # Run TLS with verbose=False and no progress bar (headless operation).
    # use_threads=1 avoids multiprocessing failures on Windows.
    result = transitleastsquares(time, values, errors, verbose=False).power(
        period_min=period_min,
        period_max=period_max,
        show_progress_bar=False,
        use_threads=use_threads,
    )
    # TLS reports the bottom-of-transit flux (minimum of the best-fit
    # limb-darkened model).  Convert to relative depth in ppm.
    bottom_flux = float(result.depth)
    depth_relative = 1.0 - bottom_flux
    # NUMERICAL_GUARD: reject non-finite or unphysical solutions (depth
    # outside (0, 1) in relative flux units).
    values_to_check = (
        float(result.period),
        float(result.T0),
        float(result.duration),
        float(result.SDE),
        bottom_flux,
    )
    if not np.all(np.isfinite(values_to_check)) or not 0.0 < depth_relative < 1.0:
        raise RuntimeError("TLS did not return a physical transit solution")
    return {
        "best_period": float(result.period),
        "best_epoch": float(result.T0),
        "best_depth_ppm": depth_relative * 1e6,
        "best_duration_hours": float(result.duration) * 24.0,
        "sde": float(result.SDE),
    }


def _median_bin(time: np.ndarray, flux: np.ndarray, n_bins: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """Median-bin a time-sorted light curve down to at most ``n_bins`` samples.

    Each bin contains approximately ``N_total / n_bins`` consecutive cadences
    after time-sorting.  The bin centre is the **mean** time and the bin
    value is the **median** flux.  The median is more robust against outlier
    cadences than simple averaging and does not require iterative sigma-
    clipping.

    This is the binning step used by the candidate-facing BLS runner to
    cap the matrix at ``max_points`` (default 4000).  TLS searches operate
    at native cadence and are not binned.

    Parameters
    ----------
    time : np.ndarray
        1-D array of observation times (any absolute system).
    flux : np.ndarray
        Matching 1-D flux array.
    n_bins : int
        Maximum number of output points.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Binned (time, flux) arrays of length ≤ ``n_bins``, or the original
        arrays if they are already smaller.
    """
    if time.size <= n_bins:
        return time, flux
    order = np.argsort(time)
    time_sorted = time[order]
    flux_sorted = flux[order]
    # Divide sorted indices into n_bins nearly equal partitions.
    edges = np.linspace(0, time_sorted.size, n_bins + 1).astype(int)
    bin_times = np.empty(n_bins, dtype=float)
    bin_flux = np.empty(n_bins, dtype=float)
    for index in range(n_bins):
        start, stop = edges[index], edges[index + 1]
        if stop > start:
            bin_times[index] = np.mean(time_sorted[start:stop])
            bin_flux[index] = np.median(flux_sorted[start:stop])
    return bin_times, bin_flux


def load_candidate_light_curve(
    workspace: CandidateWorkspace,
    max_points: int = 4000,
    sectors: Optional[Sequence[int]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (time_btjd, normalized_flux) from candidate FITS data, or None.

    Products are read from ``data/processed/`` first, then ``data/raw/``.
    Multiple products are concatenated (per-sector binning) so multi-sector
    baselines are searched jointly.  Returns ``None`` when no readable FITS
    light curve with at least 50 points exists after quality filtering.

    The ``max_points`` cap triggers ``_median_bin`` inside the underlying
    ``load_light_curve_table`` call, ensuring that the BLS matrix stays
    within computational bounds for long-baseline, high-cadence data.

    Parameters
    ----------
    workspace : CandidateWorkspace
        The candidate's workspace directory.
    max_points : int
        Maximum number of points after per-product median binning.
    sectors : Optional[Sequence[int]]
        Optional list of integer sector numbers to scope; all sectors are
        used when ``None``.

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray]]
        (time_btjd, flux) arrays or ``None``.
    """
    from .inputs import load_light_curve_table

    table = load_light_curve_table(workspace, max_points=max_points, sectors=sectors)
    if table is None:
        return None
    time = np.asarray(table["time"], dtype=float)
    flux = np.asarray(table["flux"], dtype=float)
    if time.size < 50 or time.size != flux.size:
        return None
    return time, flux


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for a candidate-local input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bls_runtime_provenance() -> Dict[str, str]:
    """Return the exact core BLS implementation and installed package version."""
    try:
        version = importlib.metadata.version("astropy")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - core dependency
        version = "unknown"
    return {
        "implementation": "astropy.timeseries.BoxLeastSquares",
        "package": "astropy",
        "version": version,
    }


def _input_manifest_records(
    workspace: CandidateWorkspace,
    input_files: Sequence[Path],
    input_sha256s: Sequence[str],
) -> List[Dict[str, Any]]:
    """Describe raw products only while they retain their acquisition provenance.

    Each raw FITS product must have a hash-matched ``.provenance.json``
    sidecar and pass the gatekeeper's ``has_valid_raw_product_provenance``
    check.  If *any* product fails, the whole list is rejected (returned
    as empty), ensuring that no search output claims provenance that has
    been invalidated by later product changes.

    This is a **hard gate**: search outputs are only evidence-bearing when
    every contributing raw product has a valid acquisition provenance record
    (see ``methods/detrending-and-transit-inference.md``).

    Parameters
    ----------
    workspace : CandidateWorkspace
    input_files : Sequence[Path]
        Absolute or candidate-relative paths to FITS products.
    input_sha256s : Sequence[str]
        Expected SHA-256 digests for each product, in matching order.

    Returns
    -------
    List[Dict[str, Any]]
        List of manifest-ready provenance records, or an empty list if any
        product fails validation.
    """
    if len(input_files) != len(input_sha256s) or not input_files:
        return []
    from .gatekeeper import has_valid_raw_product_provenance

    records: List[Dict[str, Any]] = []
    candidate_root = workspace.path.resolve()
    for path, expected_sha256 in zip(input_files, input_sha256s):
        product_path = Path(path)
        sidecar_path = product_path.with_name(product_path.stem + ".provenance.json")
        try:
            relative_path = product_path.resolve().relative_to(candidate_root).as_posix()
            provenance_path = sidecar_path.resolve().relative_to(candidate_root).as_posix()
        except (OSError, ValueError):
            return []
        if (
            not product_path.is_file()
            or _sha256(product_path) != expected_sha256
            or not sidecar_path.is_file()
            or not has_valid_raw_product_provenance(workspace, product_path)
        ):
            return []
        records.append(
            {
                "path": relative_path,
                "sha256": expected_sha256,
                "provenance_path": provenance_path,
                "provenance_sha256": _sha256(sidecar_path),
            }
        )
    return records


def run_bls_on_candidate(
    workspace: CandidateWorkspace,
    period_min: Optional[float] = None,
    period_max: Optional[float] = None,
    n_periods: int = 2000,
    signal: Optional[str] = None,
    engine: str = "bls",
    sectors: Optional[Sequence[int]] = None,
    result_suffix: Optional[str] = None,
    duration_grid_hours: Optional[Sequence[float]] = None,
    detrending_method: Optional[str] = None,
) -> Path:
    """Run BLS or TLS transit search on candidate data and save JSON summary to ``outputs/``.

    This is the **candidate-facing orchestrator** — it loads photometry,
    validates provenance, selects the engine, and writes a result JSON
    and a signed manifest.  It replaces a previously decentralised collection
    of scripts with a single, reproducible pipeline entry point.

    Signal-targeted mode
    --------------------
    When ``signal`` is provided (e.g. ``""`` for the primary, ``".1"`` for a
    secondary), the search reads the matching per-signal prior from
    ``config/signals/transit_config<signal>.json``, uses its duration, and
    restricts the period grid to ±0.1 days around the prior period. Explicit
    period bounds must agree with that effective window; conflicting bounds
    fail rather than being silently replaced.
    Targeted runs write to ``outputs/<engine>_search_results<signal>.json``
    so independent signals cannot overwrite one another.

    Blind search mode
    -----------------
    Without ``signal``, the full ``[period_min, period_max]`` range is scanned
    using ``n_periods`` as a minimum trial density.  BLS uses per-product
    median binning capped at 4000 points; TLS operates at native cadence.

    Provenance gates
    ----------------
    Both engines require **schema-valid, hash-matched raw provenance
    sidecars** for every input FITS product.  If a product's provenance has
    been invalidated (missing sidecar, hash mismatch, or failed gatekeeper
    check), the search fails with a ``ValueError`` rather than producing
    unattributable output.

    Parameters
    ----------
    workspace : CandidateWorkspace
    period_min, period_max : float or None
        Blind-search period bounds in days. When omitted, blind searches use
        0.5 to 15.0 days. In targeted mode, supplied values must match the
        prior-defined effective window.
    n_periods : int
        Minimum trial period density (BLS only; acts as floor).
    signal : Optional[str]
        Per-signal suffix (``""`` or ``".1"``, etc.) for targeted mode.
    engine : str
        ``"bls"`` or ``"tls"``.
    sectors : Optional[Sequence[int]]
        Optional sector filter.
    result_suffix : Optional[str]
        ``.label`` suffix for blind search output disambiguation; mutually
        exclusive with ``signal``.
    duration_grid_hours : Optional[Sequence[float]]
        BLS-only multi-duration scan.
    detrending_method : Optional[str]
        Name of the detrending method whose ``data/processed/`` product
        should be consumed.  ``None`` uses raw photometry.

    Returns
    -------
    Path
        Absolute path to the written result JSON.

    Raises
    ------
    ValueError
        If engine, signal, suffix, or input data are inconsistent or invalid.
    RuntimeError
        If provenance sidecars change during the search (integrity violation).
    """
    from .inputs import BTJD_TIME_SYSTEM

    if engine not in ("bls", "tls"):
        raise ValueError("search engine must be 'bls' or 'tls'")
    if duration_grid_hours is not None and engine != "bls":
        raise ValueError("duration_grid_hours is supported only by BLS searches")
    duration_grid: Optional[List[float]] = None
    if duration_grid_hours is not None:
        duration_grid = [float(value) for value in duration_grid_hours]
        if not duration_grid or any(not np.isfinite(value) or value <= 0 for value in duration_grid):
            raise ValueError("duration_grid_hours must contain positive finite values")
        if len(set(duration_grid)) != len(duration_grid):
            raise ValueError("duration_grid_hours must not contain duplicates")

    duration_hours: Optional[float] = None
    signal_provenance: Optional[Dict[str, Any]] = None
    if signal is not None:
        validate_signal_suffix(signal)
        if duration_grid is not None:
            raise ValueError("duration_grid_hours cannot be combined with a targeted signal search")

        from .inputs import load_transit_ephemeris

        ephem = load_transit_ephemeris(workspace, signal=signal)
        field_sources = ephem.get("field_sources", {})
        if (
            not isinstance(field_sources, dict)
            or field_sources.get("period_days") != "candidate-config-signal"
            or field_sources.get("duration_days") != "candidate-config-signal"
        ):
            raise ValueError(
                "no readable signal prior with candidate period and duration at config/signals/transit_config{0}.json".format(
                    signal
                )
            )

        prior_p = float(ephem["period_days"])
        duration_hours = float(ephem["duration_days"]) * 24.0
        if not np.isfinite(prior_p) or prior_p <= 0:
            raise ValueError("signal prior period_days must be positive and finite")
        if not np.isfinite(duration_hours) or duration_hours <= 0:
            raise ValueError("signal prior duration must be positive and finite")

        # ASTROPHYSICAL_HEURISTIC: ±0.1 d window around the prior period.
        # This is wide enough to capture the expected precision of a catalog
        # ephemeris on a multi-sector baseline while excluding unrelated peaks.
        targeted_period_min = max(0.5, prior_p - 0.1)
        targeted_period_max = prior_p + 0.1
        if targeted_period_max <= targeted_period_min:
            raise ValueError("signal prior period is below the supported BLS range")
        conflicting_bounds = []
        for name, requested, effective in (
            ("period_min", period_min, targeted_period_min),
            ("period_max", period_max, targeted_period_max),
        ):
            if requested is None:
                continue
            try:
                agrees = bool(
                    np.isfinite(float(requested))
                    and np.isclose(float(requested), effective, rtol=0.0, atol=1e-12)
                )
            except (TypeError, ValueError):
                agrees = False
            if not agrees:
                conflicting_bounds.append(name)
        if conflicting_bounds:
            raise ValueError(
                "targeted signal search bounds conflict with candidate prior window "
                "[{0:.12g}, {1:.12g}] days ({2}); omit explicit bounds or use the "
                "matching prior-defined window".format(
                    targeted_period_min,
                    targeted_period_max,
                    ", ".join(conflicting_bounds),
                )
            )
        period_min = targeted_period_min
        period_max = targeted_period_max
        signal_provenance: Dict[str, Any] = {
            "mode": "targeted-prior",
            "signal": signal,
            "prior_path": "config/signals/transit_config{0}.json".format(signal),
            "prior_source": ephem["source"],
            "prior_period_days": prior_p,
            "prior_duration_hours": duration_hours,
            "period_min_days": period_min,
            "period_max_days": period_max,
        }
        if field_sources.get("epoch_btjd") == "candidate-config-signal":
            prior_epoch = float(ephem["epoch_btjd"])
            if np.isfinite(prior_epoch):
                signal_provenance["prior_epoch_btjd"] = prior_epoch
    else:
        period_min = 0.5 if period_min is None else period_min
        period_max = 15.0 if period_max is None else period_max

    # Validate the effective grid before any photometry I/O. In targeted mode
    # this is the candidate-prior window after explicit-bound reconciliation.
    if engine == "bls":
        _frequency_period_grid(period_min, period_max, n_periods)

    if result_suffix is not None:
        if signal is not None:
            raise ValueError("result_suffix cannot be combined with a signal search")
        if not re.fullmatch(r"\.[a-z0-9][a-z0-9-]*", result_suffix):
            raise ValueError("result_suffix must use the .label format")

    tls_errors: Optional[np.ndarray] = None
    bls_errors: Optional[np.ndarray] = None
    bls_error_sources: Optional[List[str]] = None
    input_records: List[Dict[str, Any]] = []
    input_files: List[Path] = []
    input_sha256s: List[str] = []
    preprocessing: Dict[str, Any] = dict(PIPELINE_NORMALIZATION)
    if engine == "tls":
        # TLS loads at native cadence (max_points=None) and requires
        # raw provenance sidecars for every contributing product.
        from .inputs import load_light_curve_table

        native_table = load_light_curve_table(
            workspace,
            max_points=None,
            sectors=sectors,
            require_raw_provenance=True,
            detrending_method=detrending_method,
        )
        loaded = None
        if native_table is not None:
            input_files = [Path(path) for path in native_table.get("input_files", [])]
            input_sha256s = list(native_table.get("input_sha256s", []))
            input_records = _input_manifest_records(
                workspace, input_files, input_sha256s
            )
            if len(input_records) != len(input_files):
                raise ValueError(
                    "TLS transit search requires schema-valid, hash-matched raw provenance sidecars"
                )
            loaded = (
                np.asarray(native_table["time"], dtype=float),
                np.asarray(native_table["flux"], dtype=float),
            )
            tls_errors = np.asarray(native_table["flux_err"], dtype=float)
            preprocessing = dict(native_table.get("detrending", PIPELINE_NORMALIZATION))
    else:
        from .inputs import load_light_curve_table

        # A requested detrending method has a candidate-local derivation
        # manifest that retains raw input hashes and acquisition sidecars.
        bls_table = load_light_curve_table(
            workspace,
            sectors=sectors,
            raw_only=detrending_method is None,
            require_raw_provenance=True,
            detrending_method=detrending_method,
        )
        loaded = None
        if bls_table is not None:
            input_files = [Path(path) for path in bls_table.get("input_files", [])]
            input_sha256s = list(bls_table.get("input_sha256s", []))
            input_records = _input_manifest_records(
                workspace, input_files, input_sha256s
            )
            if len(input_records) != len(input_files):
                raise ValueError(
                    "BLS transit search requires schema-valid, hash-matched raw provenance sidecars"
                )
            loaded = (
                np.asarray(bls_table["time"], dtype=float),
                np.asarray(bls_table["flux"], dtype=float),
            )
            bls_errors = np.asarray(bls_table["flux_err"], dtype=float)
            bls_error_sources = list(bls_table.get("flux_err_sources", []))
            preprocessing = dict(bls_table.get("detrending", PIPELINE_NORMALIZATION))
    if loaded is None:
        raise ValueError("no readable candidate light-curve photometry available for BLS transit search")
    time, flux = loaded
    source = "candidate-data"

    if engine == "tls":
        if tls_errors is None:
            raise ValueError("TLS transit search requires per-cadence flux uncertainties")
        payload = find_transits_tls(
            time,
            flux,
            tls_errors,
            period_min=period_min,
            period_max=period_max,
        )
    else:
        search_kwargs: Dict[str, Any] = {
            "period_min": period_min,
            "period_max": period_max,
            "n_periods": n_periods,
        }
        duration_trials: Optional[List[Dict[str, Any]]] = None
        if duration_grid is not None:
            result, duration_trials = find_transits_duration_grid(
                time,
                flux,
                duration_grid,
                period_min=period_min,
                period_max=period_max,
                n_periods=n_periods,
                flux_err=bls_errors,
            )
            payload = result.to_dict()
        else:
            if duration_hours is not None:
                search_kwargs["duration_hours"] = duration_hours
            payload = find_transits(time, flux, flux_err=bls_errors, **search_kwargs).to_dict()
        if duration_trials is not None:
            payload["duration_grid_trials"] = duration_trials
    payload["source"] = source
    payload["time_system"] = BTJD_TIME_SYSTEM
    payload["n_points"] = int(time.size)
    payload["preprocessing"] = preprocessing
    # SCIENTIFIC_BOUNDARY: the statistic payload explicitly records that
    # neither SNR nor SDE is a calibrated FAP.  Downstream consumers
    # (screening, vetting) must treat them as ranking scores only.
    payload["statistic"] = {
        "name": (
            "weighted BLS fitted-depth signal-to-noise"
            if engine == "bls"
            else "TLS signal detection efficiency"
        ),
        "value_field": "snr" if engine == "bls" else "sde",
        "calibrated_false_alarm_probability": None,
        "population_detection_reliability": None,
        "scientific_use": "ranking and human-review triage only",
    }
    if engine == "bls":
        payload["detection_threshold_snr"] = MINIMUM_BLS_CANDIDATE_SNR
        payload["statistic"]["uncertainty_source"] = (
            bls_error_sources if bls_errors is not None else ["robust-scatter-fallback"]
        )
    if signal_provenance is not None:
        payload["signal"] = signal
        payload["search_provenance"] = signal_provenance

    if engine in ("bls", "tls"):
        # Hard gate: re-validate provenance AFTER writing the result so we
        # detect sidecar changes that occurred during the search window.
        current_input_records = _input_manifest_records(
            workspace, input_files, input_sha256s
        )
        if current_input_records != input_records:
            raise RuntimeError("search input products or provenance changed during the search")

    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_name = "{0}_search_results{1}.json".format(engine, signal or result_suffix or "")
    output_path = outputs_dir / output_name
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    manifest: Dict[str, Any] = {
        "schema": "exonym-{0}-search-manifest-1".format(engine),
        "candidate_id": workspace.candidate_id,
        "result_path": output_path.relative_to(workspace.path).as_posix(),
        "result_sha256": _sha256(output_path),
        "result_semantic_sha256": None,
        "source": source,
        "detection_status": payload.get("detection_status"),
        "inputs": input_records,
        "configuration": {
            "period_min_days": period_min,
            "period_max_days": period_max,
            "duration_hours": duration_hours if duration_grid is None else None,
            "duration_grid_hours": duration_grid,
            "n_periods": n_periods if engine == "bls" else None,
            "n_periods_role": (
                "minimum requested trial density; Astropy baseline-duration grid may use more"
                if engine == "bls"
                else None
            ),
            "period_grid": "astropy-autopower-baseline-duration-resolved" if engine == "bls" else None,
            "max_points": 4000 if engine == "bls" else None,
            "quality_filter": "quality == 0 when available",
            "normalization": "lightkurve.remove_nans().normalize()",
            "binning": (
                "per-product median binning; no global rebinning"
                if engine == "bls"
                else "none; native cadence"
            ),
            "signal": signal,
            "engine": engine,
            "cadence": "native" if engine == "tls" else "median-binned",
            "use_threads": 1 if engine == "tls" else None,
            "uncertainty_source": bls_error_sources if engine == "bls" else None,
            "detection_threshold_snr": MINIMUM_BLS_CANDIDATE_SNR if engine == "bls" else None,
            "sectors": list(sectors) if sectors is not None else None,
            "time_system": BTJD_TIME_SYSTEM,
            "detrending_method": detrending_method,
            "preprocessing": preprocessing,
        },
        "search_statistic": payload["statistic"],
        "runtime": _bls_runtime_provenance() if engine == "bls" else None,
    }
    from .remediation import semantic_json_sha256

    manifest["result_semantic_sha256"] = semantic_json_sha256(output_path)
    if signal_provenance is not None:
        prior_path = workspace.path / signal_provenance["prior_path"]
        manifest["targeted_prior"] = {
            "path": signal_provenance["prior_path"],
            "sha256": _sha256(prior_path),
            "search_provenance": signal_provenance,
        }
    manifest_path = outputs_dir / "{0}_search_manifest{1}.json".format(
        engine, signal or result_suffix or ""
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_path


def calculate_ttv_super_period(
    period_inner_days: float,
    period_outer_days: float,
    j_resonance: int = 2,
) -> float:
    """Return TTV super-period :math:`P_{\\rm TTV}` in days for a j:j−1 resonance.

    Mathematical Formulation
    ------------------------
    For two planets in (or near) a first-order mean-motion resonance
    *j* : *j*−1, the libration (super) period of the transit-timing
    variations is the inverse of the beat frequency between the resonant
    outer frequency and inner frequency (Lithwick et al. 2012, Fabrycky
    2010):

    .. math::

        P_{\\rm TTV}
        = \\frac{1}{|f_{\\rm inner} - f_{\\rm outer}|},
        \\qquad
        f_{\\rm inner} = \\frac{j}{P_{\\rm outer}},
        \\quad
        f_{\\rm outer} = \\frac{j-1}{P_{\\rm inner}}.

    When :math:`P_{\\rm outer}/P_{\\rm inner} = j/(j-1)`, the frequencies
    are equal, :math:`P_{\\rm TTV}\\to\\infty` (exact resonance), and the
    analytic formula returns ``float('inf')``.

    Parameters
    ----------
    period_inner_days : float
        Orbital period of the inner planet in days (> 0).
    period_outer_days : float
        Orbital period of the outer planet in days (> period_inner).
    j_resonance : int
        Integer *j* for a j:j−1 resonance (≥ 2).  Default 2 (2:1).

    Returns
    -------
    float
        TTV super-period in days, or ``float('inf')`` for exact resonance.
    """
    if period_inner_days <= 0 or period_outer_days <= period_inner_days:
        raise ValueError("periods must satisfy 0 < P_inner < P_outer")
    if j_resonance <= 1:
        raise ValueError("j_resonance must be an integer >= 2")
    # Beat frequency: |j/P_outer − (j−1)/P_inner|.
    freq_inner = j_resonance / period_outer_days
    freq_outer = (j_resonance - 1) / period_inner_days
    delta_freq = abs(freq_inner - freq_outer)
    # NUMERICAL_GUARD: exact resonance yields Δf = 0 → infinite super-period.
    if delta_freq == 0:
        return float("inf")
    return 1.0 / delta_freq


def compute_linear_ephemeris_residuals(
    transit_times_btjd: Sequence[float],
    period_days: float,
    epoch_btjd: float,
) -> Dict[str, Any]:
    """Compute observed-minus-calculated (O−C) residuals for a linear ephemeris.

    For each observed mid-time :math:`t_{\\rm obs}`, the nearest epoch
    number is

    .. math::

        n = {\\rm round}\\left(\\frac{t_{\\rm obs} - T_0}{P}\\right),

    and the calculated time is :math:`t_{\\rm calc} = T_0 + n P`.  The O−C
    residual in minutes is :math:`(t_{\\rm obs} - t_{\\rm calc}) \\times
    1440` (rounded to 4 decimal places).

    This linear formulation assumes no TTV, orbital decay, apsidal precession,
    or instrument-system clock differences.  Non-finite mid-times are skipped
    and tallied separately.

    Parameters
    ----------
    transit_times_btjd : Sequence[float]
        Observed transit mid-times in BTJD.
    period_days : float
        Linear ephemeris period in days (> 0).
    epoch_btjd : float
        Reference transit epoch T₀ in BTJD.

    Returns
    -------
    Dict[str, Any]
        ``residuals_minutes`` (list of float, rounded to 4 d.p.) and
        ``n_nonfinite_midtimes`` (int count of skipped non-finite entries).
    """
    if period_days <= 0:
        raise ValueError("period_days must be positive")
    residuals_min = []
    n_nonfinite_midtimes = 0
    for t_obs in transit_times_btjd:
        t_obs_float = float(t_obs)
        if not np.isfinite(t_obs_float):
            logging.warning(
                "Skipping non-finite mid-time %.6f in O-C residual computation", t_obs_float
            )
            n_nonfinite_midtimes += 1
            continue
        # Nearest integer epoch: n = round((t_obs − T₀) / P).
        n_epoch = round((t_obs_float - float(epoch_btjd)) / float(period_days))
        t_calc = float(epoch_btjd) + n_epoch * float(period_days)
        # O−C in days, converted to minutes (×1440), rounded to 0.0001 min.
        omc_days = t_obs_float - t_calc
        residuals_min.append(round(omc_days * 1440.0, 4))
    return {"residuals_minutes": residuals_min, "n_nonfinite_midtimes": n_nonfinite_midtimes}
