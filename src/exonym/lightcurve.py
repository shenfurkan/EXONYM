"""Target-independent helpers for phase-folded transit light curves.

Scientific foundations
----------------------
* **Transit geometry** — Perryman (2018, The Exoplanet Handbook, §2):
  contact points T₁–T₄, circular-orbit duration equations, impact
  parameter *b* = a cos *i* / R_*, and the V-shape statistic for
  distinguishing U-shaped planetary transits from V-shaped grazing
  eclipsing binaries.
* **Robust depth estimation** — Ivezić et al. (2019, Statistics, Data
  Mining, and Machine Learning in Astronomy): asymptotic standard error
  of the median :math:`\\sigma_{\\text{med}} = \\sqrt{\\pi/(2N)}\\,\\sigma
  \\approx 1.253\\,\\sigma / \\sqrt{N}`, median-binned phase-folded light
  curves, and quadrature propagation of independent errors.
* **Limb-darkening parametrisation** — Kipping (2013, MNRAS 435, 2152):
  triangular sampling transform between quadratic (*u*₁, *u*₂) and
  hyper-cube (*q*₁, *q*₂) coordinates.

Units convention
----------------
* Time: BTJD (BJD_TDB − 2457000) in days; phase-hours in signed hours
  relative to the nearest transit centre.
* Flux: normalised relative flux (dimensionless); depths and
  uncertainties reported in parts-per-million (ppm).
* Stellar / planetary radii: solar and Earth units respectively, converted
  through the IAU 2015 nominal-radius ratio in :mod:`exonym.constants`.
* Density: g cm⁻³ (CGS); the solar normalization is reproducibly derived
  from IAU nominal solar constants and CODATA G.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    EARTH_TO_SOLAR_RADIUS_RATIO,
    GRAVITATIONAL_CONSTANT_CGS,
    SECONDS_PER_DAY,
    SOLAR_MEAN_DENSITY_G_CM3,
)


def parse_tess_sector(mission: object) -> Optional[int]:
    """Return the TESS sector number in a MAST mission label, if present.

    Parameters
    ----------
    mission : object
        Any object whose string representation may contain a ``Sector N``
        substring (e.g. a FITS header ``TESS-SPOC`` keyword value).

    Returns
    -------
    Optional[int]
        The extracted sector number, or ``None`` when no match is found.
    """
    match = re.search(r"\bSector\s+(\d+)\b", str(mission), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def phase_hours(
    time_btjd: Sequence[float], period_days: float, epoch_btjd: float
) -> np.ndarray:
    """Return signed hours from the nearest transit centre for each BTJD time.

    Mathematical formulation
    ------------------------
    For a linear ephemeris with period :math:`P` (days) and epoch
    :math:`T_0` (BTJD), the phase of time :math:`t` is

    .. math::

        \\phi(t) = \\big( (t - T_0 + P/2) \\bmod P \\big) - P/2.

    The additive :math:`P/2` shift places the transit centre at
    :math:`\\phi = 0`, the domain centred on zero at
    :math:`[-P/2,\\,+P/2]`.  Multiplying by 24 converts fractional days
    to signed hours.

    Parameters
    ----------
    time_btjd : Sequence[float]
        BTJD timestamps.
    period_days : float
        Orbital period in days (must be > 0).
    epoch_btjd : float
        Reference transit epoch in BTJD.

    Returns
    -------
    np.ndarray
        Signed phase-hours for each input timestamp.  Values near zero
        correspond to transit centre; negative (positive) values precede
        (follow) the nearest centre.

    Raises
    ------
    ValueError
        If ``period_days`` is not positive.
    """
    if period_days <= 0:
        raise ValueError("period_days must be positive")
    time = np.asarray(time_btjd, dtype=float)
    return (
        (time - float(epoch_btjd) + 0.5 * float(period_days)) % float(period_days)
        - 0.5 * float(period_days)
    ) * 24.0


def robust_transit_depth(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
) -> Tuple[float, float, int, int]:
    """Estimate median in/out transit depth and robust uncertainty in ppm.

    Mathematical formulation
    ------------------------
    The transit depth :math:`\\delta` (fractional) is the difference of
    out-of-transit and in-transit median fluxes:

    .. math::

        \\delta = \\operatorname{median}(\\mathbf{f}_{\\text{out}})
                - \\operatorname{median}(\\mathbf{f}_{\\text{in}}).

    The standard error of each median uses the asymptotic factor
    :math:`\\sqrt{\\pi/(2N)} \\approx 1.253` (Ivezić et al. 2019,
    Eq. 3.37), giving :math:`\\sigma_{\\text{med}} = 1.253\\,\\sigma / \\sqrt{N}`.
    The quadrature-sum of the in- and out-of-transit median errors yields
    the combined depth uncertainty.

    Window definitions
    ------------------
    * **in-transit**: :math:`|\\phi| < T_{\\text{dur}}/2`
    * **out-of-transit**: :math:`1.2\\,T_{\\text{dur}} < |\\phi| < 2.5\\,T_{\\text{dur}}`

    The 1.2× buffer guards against ingress/egress contamination in the
    baseline window (Perryman §4.3.1).  The 2.5× outer cap limits
    contamination from phase-curve variability and adjacent transits.

    Parameters
    ----------
    time_btjd : Sequence[float]
        BTJD timestamps.
    flux : Sequence[float]
        Normalised relative flux values (same shape as ``time_btjd``).
    period_days : float
        Orbital period in days.
    epoch_btjd : float
        Reference transit epoch in BTJD.
    duration_hours : float
        Catalog transit duration in hours (T₁₄, must be > 0).

    Returns
    -------
    depth_ppm : float
        Median depth in parts-per-million.
    uncertainty_ppm : float
        Robust 1‑σ depth uncertainty in ppm (asymptotic median error
        propagation).
    n_in : int
        Number of finite in-transit samples used.
    n_out : int
        Number of finite out-of-transit samples used.

    Raises
    ------
    ValueError
        If ``duration_hours`` is not positive, shapes mismatch, or
        either window contains no finite samples.

    Notes
    -----
    The factor 1.253 is the three-significant-figure truncation of the
    exact asymptotic value :math:`\\sqrt{\\pi/2} \\approx 1.253314`.
    This is a **robust** depth estimate; it does not account for limb
    darkening or dilution.
    """
    # NUMERICAL_GUARD: non-positive duration is physically meaningless.
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")

    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    if time.shape != values.shape:
        raise ValueError("time_btjd and flux must have identical shapes")

    hours = phase_hours(time, period_days, epoch_btjd)
    finite = np.isfinite(hours) & np.isfinite(values)
    in_transit = finite & (np.abs(hours) < 0.5 * duration_hours)
    # ASTROPHYSICAL_HEURISTIC: 1.2× buffer avoids ingress/egress bleed
    # into the out-of-transit baseline (Perryman §4.3.1).
    out_of_transit = finite & (np.abs(hours) > 1.2 * duration_hours) & (
        np.abs(hours) < 2.5 * duration_hours
    )
    in_values = values[in_transit]
    out_values = values[out_of_transit]
    if not in_values.size or not out_values.size:
        raise ValueError("insufficient finite in-transit or out-of-transit coverage")

    depth = float(np.median(out_values) - np.median(in_values))
    # NUMERICAL_GUARD: asymptotic median-error factor 1.253 ≈ √(π/2)
    # (Ivezić et al. 2019, Eq. 3.37).  Standard errors are propagated in
    # quadrature, assuming independent in/out samples.
    uncertainty = float(
        np.sqrt(
            (1.253 * np.std(in_values) / np.sqrt(in_values.size)) ** 2
            + (1.253 * np.std(out_values) / np.sqrt(out_values.size)) ** 2
        )
    )
    return depth * 1e6, uncertainty * 1e6, int(in_values.size), int(out_values.size)


def bin_phase_folded_flux(
    time_btjd: Sequence[float],
    flux: Sequence[float],
    period_days: float,
    epoch_btjd: float,
    limit_hours: float = 14.0,
    bin_minutes: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase-fold and median-bin a light curve for a diagnostic plot.

    The phase range :math:`[-L,\\,+L]` hours (``limit_hours``) is divided
    into equal-width bins of ``bin_minutes`` / 60 hours each.  Within
    each bin the median flux and its asymptotic standard error
    (Ivezić et al. 2019, factor 1.253) are computed.  Bins with fewer
    than three samples are left as ``NaN``.

    The final (rightmost) bin uses an inclusive upper edge
    (``hours <= edge[-1]``) so that the boundary sample is not dropped.

    Parameters
    ----------
    time_btjd : Sequence[float]
        BTJD timestamps.
    flux : Sequence[float]
        Normalised relative flux values.
    period_days : float
        Orbital period in days.
    epoch_btjd : float
        Reference transit epoch in BTJD.
    limit_hours : float
        Half-width of the phase window in hours (default 14.0).
    bin_minutes : float
        Bin width in minutes (default 8.0).

    Returns
    -------
    centers : np.ndarray
        Bin-centre phase-hours.
    median : np.ndarray
        Median flux per bin (``NaN`` where the bin has < 3 samples).
    error : np.ndarray
        Asymptotic standard error of the median per bin (``NaN`` where
        the bin has < 3 samples).

    Raises
    ------
    ValueError
        If ``limit_hours`` or ``bin_minutes`` is non-positive, or shapes
        mismatch.
    """
    # NUMERICAL_GUARD: enforce positive window and bin width.
    if limit_hours <= 0:
        raise ValueError("limit_hours must be positive")
    if bin_minutes <= 0:
        raise ValueError("bin_minutes must be positive")

    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    if time.shape != values.shape:
        raise ValueError("time_btjd and flux must have identical shapes")

    hours = phase_hours(time, period_days, epoch_btjd)
    valid = np.isfinite(hours) & np.isfinite(values) & (np.abs(hours) <= limit_hours)
    width_hours = bin_minutes / 60.0
    bin_count = int(np.ceil(2.0 * limit_hours / width_hours))
    edges = np.linspace(-limit_hours, limit_hours, bin_count + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    median = np.full(bin_count, np.nan, dtype=float)
    error = np.full(bin_count, np.nan, dtype=float)

    for index in range(bin_count):
        # Include the rightmost edge point in the last bin; all other
        # bins use the half-open convention [left, right).
        if index == bin_count - 1:
            mask = valid & (hours >= edges[index]) & (hours <= edges[index + 1])
        else:
            mask = valid & (hours >= edges[index]) & (hours < edges[index + 1])
        samples = values[mask]
        # NUMERICAL_GUARD: require ≥ 3 samples so the standard-deviation
        # estimate is minimally constrained.
        if samples.size >= 3:
            median[index] = np.median(samples)
            # Asymptotic standard error of the median (Ivezić et al. 2019, Eq. 3.37).
            error[index] = 1.253 * np.std(samples) / np.sqrt(samples.size)

    return centers, median, error


def calculate_contact_durations(
    period_days: float,
    r_star_solar: float,
    m_star_solar: float,
    r_planet_earth: float,
    impact_parameter_b: float,
    eccentricity: float = 0.0,
    omega_deg: float = 90.0,
) -> Dict[str, object]:
    """Return dict of contact durations T₁₄, T₂₃, T₁₂ in hours and grazing status.

    Mathematical formulation
    ------------------------
    **Radius ratio.**  Planet-to-star radius ratio from mixed units:

    .. math::

        k = \\frac{R_p}{R_*} = \\frac{R_{p,\\oplus} \\times (R_\\oplus/R_\\odot)}
                                        {R_{*,\\odot}},

    where the radius conversion uses IAU 2015 nominal equatorial-Earth and
    solar radii.

    **Scaled semi-major axis.**  From Kepler's Third Law and the
    definition of mean stellar density :math:`\\rho_*` (Perryman §2;
    Seager & Mallén-Ornelas 2003):

    .. math::

        \\left(\\frac{a}{R_*}\\right)^3
        = \\frac{G\\,P^2\\,\\rho_*}{3\\pi},

    with :math:`P` in seconds, :math:`\\rho_*` in g cm⁻³, and the CODATA
    gravitational constant. The stellar density is scaled from the
    reproducible nominal-solar normalization in :mod:`exonym.constants`.

    **Eccentricity factor.**  For an orbit with eccentricity *e* and
    argument of periastron :math:`\\omega` (Perryman §2, Eq. 2.26):

    .. math::

        f_e = \\frac{\\sqrt{1 - e^2}}{1 + e \\sin\\omega}.

    **Total duration T₁₄.**  The time between first and fourth contact
    for a general eccentric orbit:

    .. math::

        T_{14} = \\frac{P}{\\pi}\\,\\frac{1}{a/R_*}
                 \\sqrt{\\max(0,\\,(1 + k)^2 - b^2)}
                 \\, f_e \\quad [\\text{seconds}].

    **Full-transit duration T₂₃.**  Defined only for non-grazing
    geometries (:math:`b \\le 1 - k`):

    .. math::

        T_{23} = \\frac{P}{\\pi}\\,\\frac{1}{a/R_*}
                 \\sqrt{\\max(0,\\,(1 - k)^2 - b^2)}
                 \\, f_e \\quad [\\text{seconds}].

    **Ingress/egress duration T₁₂.**  Assuming symmetric ingress and
    egress:

    .. math::

        T_{12} = T_{34} = \\tfrac{1}{2}(T_{14} - T_{23}).

    **V-shape statistic.**  Dimensionless morphology diagnostic
    (Perryman §2):

    .. math::

        V_{\\text{stat}} = \\frac{T_{12} + T_{34}}{T_{14}}
                         = \\frac{2\\,T_{12}}{T_{14}}.

    Astrophysical rationale
    -----------------------
    * **Grazing condition** (:math:`b > 1 - k`): the planet disk never
      fully enters the stellar disk; T₂₃ → 0, producing a V-shaped
      light curve.
    * **No-contact condition** (:math:`b \\ge 1 + k`): the planet disk
      does not overlap the stellar disk at any point; no transit occurs.
    * **V_stat interpretation**:
      - :math:`V_{\\text{stat}} \\ll 0.80` — U-shaped, consistent with a
        planetary transit.
      - :math:`V_{\\text{stat}} \\to 1.00` — V-shaped, consistent with a
        grazing eclipsing binary (ASTROPHYSICAL_HEURISTIC).

    Parameters
    ----------
    period_days : float
        Orbital period in days (> 0).
    r_star_solar : float
        Stellar radius in solar units (> 0).
    m_star_solar : float
        Stellar mass in solar units (> 0).
    r_planet_earth : float
        Planet radius in Earth units (> 0).
    impact_parameter_b : float
        Dimensionless impact parameter (≥ 0).
    eccentricity : float
        Orbital eccentricity (0 ≤ *e* < 1).  Default 0.0 (circular).
    omega_deg : float
        Argument of periastron in degrees.  Default 90.0.

    Returns
    -------
    Dict[str, object]
        Includes ``geometry_status`` as ``"full-transit"``, ``"grazing"``,
        or ``"non-transiting"``. The no-contact status has ``v_stat = None``.
        Keys:
        - ``"T14_hr"`` — total duration T₁₄ in hours (4 decimal places).
        - ``"T23_hr"`` — full-transit duration T₂₃ in hours (0.0 when grazing).
        - ``"T12_hr"`` — ingress/egress duration T₁₂ in hours.
        - ``"grazing"`` — 1.0 for a contact geometry with
          :math:`1-k < b < 1+k`, else 0.0.
        - ``"v_stat"`` — V-shape statistic for transiting geometries, or
          ``None`` when there is no contact.

    Raises
    ------
    ValueError
        If physical parameters are non-positive, eccentricity out of
        range, or impact parameter is negative.

    Notes
    -----
    The ``b >= 1 + k`` no-contact case uses
    ``geometry_status = "non-transiting"``, ``grazing = 0.0``, and
    ``v_stat = None``. It must remain distinct from a contact geometry that
    produces a genuinely V-shaped grazing event.
    """
    if period_days <= 0 or r_star_solar <= 0 or m_star_solar <= 0 or r_planet_earth <= 0:
        raise ValueError("physical parameters must be positive")
    if not (0.0 <= eccentricity < 1.0):
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    if impact_parameter_b < 0:
        raise ValueError("impact_parameter_b must be non-negative")

    # Radius ratio uses IAU 2015 nominal equatorial-Earth / solar radii.
    k = (r_planet_earth * EARTH_TO_SOLAR_RADIUS_RATIO) / r_star_solar
    # SCIENTIFIC_BOUNDARY: b ≥ 1+k means no disc overlap at any orbital
    # phase. It has no measurable V-shape morphology and is not a grazing
    # transit.
    if impact_parameter_b >= 1.0 + k:
        return {
            "T14_hr": 0.0,
            "T23_hr": 0.0,
            "T12_hr": 0.0,
            "grazing": 0.0,
            "v_stat": None,
            "geometry_status": "non-transiting",
        }

    # IAU-nominal/CODATA-derived solar density for scaling stellar density.
    rho_solar_gcm3 = SOLAR_MEAN_DENSITY_G_CM3
    rho_star_gcm3 = m_star_solar / (r_star_solar**3) * rho_solar_gcm3
    period_sec = period_days * SECONDS_PER_DAY
    # a / R_* from Kepler's Third Law (Perryman §2; Seager & Mallén-Ornelas 2003).
    a_over_r = (
        (GRAVITATIONAL_CONSTANT_CGS * (period_sec**2) * rho_star_gcm3)
        / (3.0 * math.pi)
    ) ** (1.0 / 3.0)

    # Eccentricity correction factor (Perryman §2, Eq. 2.26).
    ecc_factor = math.sqrt(1.0 - eccentricity**2) / (
        1.0 + eccentricity * math.sin(math.radians(omega_deg))
    )

    # NUMERICAL_GUARD: max(0, ...) protects against floating-point
    # round-off when (1+k)² ≈ b² due to finite-precision arithmetic.
    t14_sec = (
        (period_sec / math.pi)
        * (1.0 / a_over_r)
        * math.sqrt(max(0.0, (1.0 + k) ** 2 - impact_parameter_b**2))
        * ecc_factor
    )
    # ASTROPHYSICAL_HEURISTIC: grazing when planet never fully enters
    # the stellar disc (b > 1 − k).
    grazing = impact_parameter_b > (1.0 - k)
    t23_sec = 0.0
    if not grazing:
        t23_sec = (
            (period_sec / math.pi)
            * (1.0 / a_over_r)
            # NUMERICAL_GUARD: same floating-point protection as T₁₄.
            * math.sqrt(max(0.0, (1.0 - k) ** 2 - impact_parameter_b**2))
            * ecc_factor
        )

    # T₁₂ = T₃₄ = ½ (T₁₄ − T₂₃) assuming symmetric ingress/egress.
    t12_sec = 0.5 * (t14_sec - t23_sec)
    # V_stat ≡ (T₁₂ + T₃₄) / T₁₄ = 2 T₁₂ / T₁₄.
    # DIAGNOSTIC_REASONING: values near 1 indicate a continuously
    # V-shaped light curve (grazing EB); values well below 0.8 indicate
    # a flat-bottomed U-shape characteristic of planetary transits.
    v_stat = (2.0 * t12_sec / t14_sec) if t14_sec > 0 else 1.0

    return {
        "T14_hr": round(t14_sec / 3600.0, 4),
        "T23_hr": round(t23_sec / 3600.0, 4),
        "T12_hr": round(t12_sec / 3600.0, 4),
        "grazing": 1.0 if grazing else 0.0,
        "v_stat": round(v_stat, 4),
        "geometry_status": "grazing" if grazing else "full-transit",
    }


def kipping_to_quadratic_limb_darkening(q1: float, q2: float) -> Tuple[float, float]:
    """Convert Kipping (2013) hyper-cube parameters (*q*₁, *q*₂) to quadratic (*u*₁, *u*₂).

    Mathematical formulation
    ------------------------
    Kipping (2013, MNRAS 435, 2152) introduced the triangular sampling
    transform that maps the uninformative hyper-cube :math:`(q_1, q_2)
    \\in [0,1]^2` to physically-allowed quadratic limb-darkening
    coefficients :math:`(u_1, u_2)` for the Mandel & Agol (2002) law:

    .. math::

        I(\\mu)/I(1) = 1 - u_1(1 - \\mu) - u_2(1 - \\mu)^2,

    where :math:`\\mu = \\cos\\theta` is the foreshortening angle.

    The forward transform is

    .. math::

        u_1 &= 2\\sqrt{q_1}\\,q_2, \\\\
        u_2 &= \\sqrt{q_1}\\,(1 - 2q_2).

    This preserves uniform sampling over the physically-allowed region
    :math:`u_1 > 0`, :math:`u_1 + u_2 < 1` of the quadratic coefficient
    plane.

    Parameters
    ----------
    q1 : float
        Kipping hyper-cube parameter (0 ≤ *q*₁ ≤ 1).
    q2 : float
        Kipping hyper-cube parameter (0 ≤ *q*₂ ≤ 1).

    Returns
    -------
    u1 : float
        Quadratic limb-darkening coefficient *u*₁.
    u2 : float
        Quadratic limb-darkening coefficient *u*₂.

    Raises
    ------
    ValueError
        If *q*₁ or *q*₂ lies outside [0, 1].
    """
    if not (0.0 <= q1 <= 1.0 and 0.0 <= q2 <= 1.0):
        raise ValueError("q1 and q2 must be in [0, 1]")
    sqrt_q1 = math.sqrt(q1)
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    return u1, u2


def quadratic_to_kipping_limb_darkening(u1: float, u2: float) -> Tuple[float, float]:
    """Convert quadratic limb darkening parameters (*u*₁, *u*₂) to Kipping (*q*₁, *q*₂).

    Mathematical formulation
    ------------------------
    The inverse of the Kipping (2013, MNRAS 435, 2152) triangular-sampling
    transform:

    .. math::

        q_1 &= (u_1 + u_2)^2, \\\\
        q_2 &= \\frac{u_1}{2\\,(u_1 + u_2)}.

    When :math:`u_1 + u_2 = 0` (uniform-disk limit), both :math:`q_1`
    and :math:`q_2` are returned as 0.0.

    Parameters
    ----------
    u1 : float
        Quadratic limb-darkening coefficient *u*₁.
    u2 : float
        Quadratic limb-darkening coefficient *u*₂.

    Returns
    -------
    q1 : float
        Kipping hyper-cube parameter *q*₁.
    q2 : float
        Kipping hyper-cube parameter *q*₂.
    """
    try:
        u1 = float(u1)
        u2 = float(u2)
    except (TypeError, ValueError) as exc:
        raise ValueError("quadratic limb-darkening coefficients must be finite numbers") from exc
    if not (math.isfinite(u1) and math.isfinite(u2)):
        raise ValueError("quadratic limb-darkening coefficients must be finite numbers")
    # NUMERICAL_GUARD: only the physical uniform disk has a zero denominator.
    if u1 == 0.0 and u2 == 0.0:
        return 0.0, 0.0
    coefficient_sum = u1 + u2
    if coefficient_sum <= 0.0 or u1 < 0.0 or u1 + 2.0 * u2 < 0.0 or coefficient_sum > 1.0:
        raise ValueError("quadratic limb-darkening coefficients are outside the physical region")
    q1 = coefficient_sum**2
    q2 = u1 / (2.0 * coefficient_sum)
    return q1, q2
