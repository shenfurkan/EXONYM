"""Vectorized prior distributions for the TREX statistical vetting engine.

Implements the astrophysical priors described in Giacalone et al. (2021):

* Planet radius: Fressin et al. (2013) broken power law.
* Orbital inclination: isotropic (uniform in cos i).
* Eccentricity: Kipping (2013) Beta distribution for planets;
  Moe & Di Stefano (2017) power law for binaries.
* Mass ratio: Moe & Di Stefano (2017) short-period binary distribution.
* Bound companion occurrence: Raghavan et al. (2010) + Moe & Di Stefano (2017).
* Background star occurrence: TRILEGAL-based.

All sampling functions operate on arrays of uniform random draws in [0, 1)
and return the corresponding parameter samples.  Log-prior functions compute
the (log) prior probability density at given parameter values.

Verified sources, units, and limits
-----------------------------------
The retained sources are Fressin et al. (2013), ADS ``2013ApJ...766...81F``,
DOI ``10.1088/0004-637X/766/2/81``; Moe & Di Stefano (2017), ADS
``2017ApJS..230...15M``, DOI ``10.3847/1538-4365/aa6fb6``; Raghavan et al.
(2010), ADS ``2010ApJS..190....1R``, DOI ``10.1088/0067-0049/190/1/1``; and
Winters et al. (2019), ADS ``2019AJ....157..216W``, DOI
``10.3847/1538-3881/ab05dc``.  Radius draws are Earth radii, primary mass is
solar mass, periods are days, inclinations/arguments are degrees, separations
are arcsec or AU as named, parallax is mas, and log priors are dimensionless.
The Winters relation is limited to its documented M-dwarf range; out-of-range
or nonfinite draws must be rejected rather than extrapolated.  These priors are
conditional scenario assumptions, not population validation, and cannot set
``claim_eligible``.
"""

from __future__ import annotations

import fractions
import math
from typing import Optional, Tuple

import numpy as np
from scipy.special import ndtr as _standard_normal_cdf
from scipy.stats import beta as _beta_dist
from scipy.stats import powerlaw as _powerlaw_dist

from ...constants import (
    FULL_TURN_DEGREES,
    MILLIARCSECONDS_PER_ARCSECOND,
    SQUARE_ARCSECONDS_PER_SQUARE_DEGREE,
)
from .constants import Msun, Rsun, au, G, pi, SECONDS_PER_DAY
from .funcs import separation_at_contrast

# Winters et al. (2019), AJ, 157, 216, DOI:10.3847/1538-3881/ab05dc,
# arXiv:1901.06364, §5.3 and Figure 22.  The retained source PDF is
# literature/winters_2019_m_dwarf_multiplicity.pdf.  The survey measures a
# 26.8% corrected M-dwarf multiplicity rate and fits the distribution of
# log10(projected separation / AU) with a Gaussian centered at 20 AU and a
# 1.16-dex standard deviation.  Its reported primary-mass sub-samples span
# 0.075--0.60 solar masses; no extrapolation beyond that lower boundary is
# scientifically supported by this relation.
WINTERS_2019_M_DWARF_MINIMUM_MASS_SOLAR = 0.075
WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR = 0.60
WINTERS_2019_M_DWARF_MULTIPLICITY_FRACTION = 0.268
WINTERS_2019_LOG10_SEPARATION_MEAN_AU = math.log10(20.0)
WINTERS_2019_LOG10_SEPARATION_STANDARD_DEVIATION_DEX = 1.16

# Moe & Di Stefano (2017), ApJS, 230, 15, ADS ``2017ApJS..230...15M``, DOI
# ``10.3847/1538-4365/aa6fb6``, Section 4 analytical integrals.  The mass-ratio
# prior integrates ((u - 2.0) / 2.1) du over log10(P/day) in [3.4, m] and
# exp(-0.3 * (u - 5.5)) du over [5.5, m]; both are evaluated exactly so no
# rounded decimal placeholder remains and the piecewise log-prior stays C^0
# continuous at log10(P/day) = 5.5.
#
#     \int_{3.4}^m ((u - 2.0) / 2.1) du = (5/21) m^2 - (20/21) m + (17/35)
#     \int_{5.5}^m exp(-0.3 (u - 5.5)) du = (10/3) (1 - exp(-0.3 (m - 5.5)))
MOE2017_LOGP_QUAD_C2 = float(fractions.Fraction(5, 21))
MOE2017_LOGP_QUAD_C1 = float(fractions.Fraction(20, 21))
MOE2017_LOGP_QUAD_C0 = float(fractions.Fraction(17, 35))
MOE2017_EXP_SCALE = float(fractions.Fraction(10, 3))

# Moe & Di Stefano (2017), ApJS, 230, 15, Section 5 (broken power-law mass-ratio
# distribution) and Section 6 (companion fraction vs. primary mass).  The
# retained parameters are:
#   * q distribution slopes p1 (0.1 <= q < 0.3) and p2 (0.3 <= q < 0.95),
#   * the twin fraction F_twin for q >= 0.95,
#   * the companion-fraction polynomial coefficients (f1, f2, f3) as functions
#     of log10(primary mass / Msun), with their power-law slope alpha and
#     log-period bin width dlogP.
# These are empirical population parameters from a peer-reviewed survey and are
# kept as named constants so no anonymous scalar enters the integrals.
MOE2017_Q_SLOPE_LOW = 0.3
MOE2017_Q_SLOPE_HIGH = -0.5
MOE2017_TWIN_FRACTION = 0.30
MOE2017_Q_BREAK_LOW = 0.1
MOE2017_Q_BREAK_MID = 0.3
MOE2017_Q_TWIN_THRESHOLD = 0.95
MOE2017_COMPANION_FRACTION_SLOPE = 0.018
MOE2017_COMPANION_LOGPERIOD_BIN_DEX = 0.7
# Companion-fraction polynomial coefficients vs. log10(primary mass / Msun)
# from Moe & Di Stefano (2017, ApJS, 230, 15), Section 6, Table 2.  Each tuple
# is (constant, linear, quadratic) in x = log10(M_s / Msun).
MOE2017_COMPANION_FRACTION_F1_COEFFS = (0.020, 0.04, 0.07)
MOE2017_COMPANION_FRACTION_F2_COEFFS = (0.039, 0.07, 0.01)
MOE2017_COMPANION_FRACTION_F3_COEFFS = (0.078, -0.05, 0.04)

# Kipping (2013), MNRAS, 434, L51, ADS ``2013MNRAS.434L..51K``, DOI
# ``10.1093/mnrasl/slt075``: Beta-distribution eccentricity prior parameters
# for transiting planets.  Moe & Di Stefano (2017) eccentricity power-law
# indices for binary companions, with the retained period break at 10 days.
KIPPING_2013_ECCENTRICITY_BETA_A = 0.867
KIPPING_2013_ECCENTRICITY_BETA_B = 3.030
MOE2017_ECCENTRICITY_POWERLAW_SHORT = 0.2
MOE2017_ECCENTRICITY_POWERLAW_LONG = 0.6
MOE2017_ECCENTRICITY_PERIOD_BREAK_DAYS = 10.0

# ---------------------------------------------------------------------------
# Planet radius prior  (Fressin et al. 2013, broken power law)
# ---------------------------------------------------------------------------

# Pre-computed normalisation constants for the Fressin et al. (2013, ApJ, 766,
# 81, ADS ``2013ApJ...766...81F``, DOI ``10.1088/0004-637X/766/2/81``) broken
# power-law planet-radius prior.  Break radii are Earth radii; power-law indices
# switch at a 0.45 Msun host-mass boundary as reported in that paper.
_R_BREAK1 = 3.0    # R_earth
_R_BREAK2 = 6.0    # R_earth
_R_MIN = 0.5
_R_MAX = 20.0

# Power-law indices for M_s > 0.45 Msun
_P1_HI, _P2_HI, _P3_HI = 0.0, -4.0, -0.5
# Power-law indices for M_s <= 0.45 Msun
_P1_LO, _P2_LO, _P3_LO = 0.0, -7.0, -0.5

_A1 = _R_BREAK1 ** _P1_HI / _R_BREAK1 ** _P2_HI
_A2 = _R_BREAK2 ** _P2_HI / _R_BREAK2 ** _P3_HI
_I1 = (_R_BREAK1 ** (_P1_HI + 1) - _R_MIN ** (_P1_HI + 1)) / (_P1_HI + 1)
_I2 = _A1 * (_R_BREAK2 ** (_P2_HI + 1) - _R_BREAK1 ** (_P2_HI + 1)) / (_P2_HI + 1)
_I3 = _A2 * _A1 * (_R_MAX ** (_P3_HI + 1) - _R_BREAK2 ** (_P3_HI + 1)) / (_P3_HI + 1)
_NORM1 = 1.0 / (_I1 + _I2 + _I3)

_A3 = _R_BREAK1 ** _P1_LO / _R_BREAK1 ** _P2_LO
_A4 = _R_BREAK2 ** _P2_LO / _R_BREAK2 ** _P3_LO
_I4 = (_R_BREAK1 ** (_P1_LO + 1) - _R_MIN ** (_P1_LO + 1)) / (_P1_LO + 1)
_I5 = _A3 * (_R_BREAK2 ** (_P2_LO + 1) - _R_BREAK1 ** (_P2_LO + 1)) / (_P2_LO + 1)
_I6 = _A4 * _A3 * (_R_MAX ** (_P3_LO + 1) - _R_BREAK2 ** (_P3_LO + 1)) / (_P3_LO + 1)
_NORM2 = 1.0 / (_I4 + _I5 + _I6)


def sample_rp(x: np.ndarray, M_s: np.ndarray, flatpriors: bool = False) -> np.ndarray:
    """Sample planet radii from the Fressin+ (2013) broken power law.

    Args:
        x: Uniform random draws in [0, 1), shape (N,).
        M_s: Host star masses [Msun], shape (N,).
        flatpriors: If True, sample uniformly in [0.5, 20] R_earth.

    Returns:
        Planet radii [R_earth], shape (N,).
    """
    x = np.asarray(x, dtype=float).copy()
    M_s = np.asarray(M_s, dtype=float)

    if flatpriors:
        return x * (_R_MAX - _R_MIN) + _R_MIN

    hi_mass = M_s > 0.45
    lo_mass = ~hi_mass

    # High-mass branch
    if np.any(hi_mass):
        m1 = hi_mass & (x <= _NORM1 * _I1)
        m2 = hi_mass & (x > _NORM1 * _I1) & (x <= _NORM1 * (_I1 + _I2))
        m3 = hi_mass & (x > _NORM1 * (_I1 + _I2)) & (x <= _NORM1 * (_I1 + _I2 + _I3))
        x[m1] = (x[m1] / _NORM1 * (_P1_HI + 1) + _R_MIN ** (_P1_HI + 1)) ** (1.0 / (_P1_HI + 1))
        x[m2] = ((x[m2] / _NORM1 - _I1) * (_P2_HI + 1) / _A1 + _R_BREAK1 ** (_P2_HI + 1)) ** (1.0 / (_P2_HI + 1))
        xx = x[m3]
        x[m3] = ((xx / _NORM1 - _I1 - _I2) * (_P3_HI + 1) / (_A1 * _A2) + _R_BREAK2 ** (_P3_HI + 1)) ** (1.0 / (_P3_HI + 1))

    # Low-mass branch
    if np.any(lo_mass):
        m4 = lo_mass & (x <= _NORM2 * _I4)
        m5 = lo_mass & (x > _NORM2 * _I4) & (x <= _NORM2 * (_I4 + _I5))
        m6 = lo_mass & (x > _NORM2 * (_I4 + _I5)) & (x <= _NORM2 * (_I4 + _I5 + _I6))
        x[m4] = (x[m4] / _NORM2 * (_P1_LO + 1) + _R_MIN ** (_P1_LO + 1)) ** (1.0 / (_P1_LO + 1))
        x[m5] = ((x[m5] / _NORM2 - _I4) * (_P2_LO + 1) / _A3 + _R_BREAK1 ** (_P2_LO + 1)) ** (1.0 / (_P2_LO + 1))
        xx = x[m6]
        x[m6] = ((xx / _NORM2 - _I4 - _I5) * (_P3_LO + 1) / (_A3 * _A4) + _R_BREAK2 ** (_P3_LO + 1)) ** (1.0 / (_P3_LO + 1))

    return x


# ---------------------------------------------------------------------------
# Inclination prior  (isotropic: uniform in cos i)
# ---------------------------------------------------------------------------

def sample_inc(x: np.ndarray, lower: float = 0.0, upper: float = 90.0) -> np.ndarray:
    """Sample inclinations from an isotropic distribution.

    Args:
        x: Uniform random draws in [0, 1), shape (N,).
        lower: Lower bound [deg].
        upper: Upper bound [deg].

    Returns:
        Inclinations [deg], shape (N,).
    """
    low_rad = math.radians(lower)
    up_rad = math.radians(upper)
    norm = 1.0 / (math.cos(low_rad) - math.cos(up_rad))
    return np.degrees(np.arccos(np.cos(low_rad) - np.asarray(x, dtype=float) / norm))


# ---------------------------------------------------------------------------
# Eccentricity prior
# ---------------------------------------------------------------------------

def sample_ecc(x: np.ndarray, planet: bool, P_orb: float) -> np.ndarray:
    """Sample orbital eccentricities.

    Planets: Kipping (2013) Beta(0.867, 3.030).
    Binaries, P <= 10 d: Moe & Di Stefano (2017) power-law nu=0.2.
    Binaries, P > 10 d: Moe & Di Stefano (2017) power-law nu=0.6.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.any((x < 0.0) | (x >= 1.0)):
        raise ValueError("eccentricity draws must be finite values in [0, 1)")
    if planet:
        return _beta_dist.ppf(x, KIPPING_2013_ECCENTRICITY_BETA_A, KIPPING_2013_ECCENTRICITY_BETA_B)
    elif P_orb <= MOE2017_ECCENTRICITY_PERIOD_BREAK_DAYS:
        return _powerlaw_dist.ppf(x, MOE2017_ECCENTRICITY_POWERLAW_SHORT)
    else:
        return _powerlaw_dist.ppf(x, MOE2017_ECCENTRICITY_POWERLAW_LONG)


# ---------------------------------------------------------------------------
# Argument of periastron  (uniform)
# ---------------------------------------------------------------------------

def sample_w(x: np.ndarray) -> np.ndarray:
    """Sample argument of periastron uniformly in [0, 360) deg."""
    return np.asarray(x, dtype=float) * FULL_TURN_DEGREES


# ---------------------------------------------------------------------------
# Mass-ratio prior for short-period binaries  (Moe & Di Stefano 2017)
# ---------------------------------------------------------------------------

def sample_q(x: np.ndarray, M_s: float) -> np.ndarray:
    """Sample binary mass ratios q = M_sec / M_prim.

    Broken power-law from Moe & Di Stefano (2017) with twin fraction
    ``MOE2017_TWIN_FRACTION`` for q >= ``MOE2017_Q_TWIN_THRESHOLD``.  All
    population parameters are the named ``MOE2017_*`` constants.
    """
    x = np.asarray(x, dtype=float).copy()
    p1 = MOE2017_Q_SLOPE_LOW
    p2 = MOE2017_Q_SLOPE_HIGH
    q_low = MOE2017_Q_BREAK_LOW
    q_break = MOE2017_Q_BREAK_MID
    q_twin = MOE2017_Q_TWIN_THRESHOLD
    F_twin = MOE2017_TWIN_FRACTION
    A1 = q_break ** p1 / q_break ** p2
    A2 = 1.0 + (F_twin / (1.0 - F_twin)) * (
        (1.0 ** (p2 + 1) - q_break ** (p2 + 1)) / (p2 + 1)
    ) / ((1.0 ** (p2 + 1) - q_twin ** (p2 + 1)) / (p2 + 1))
    if M_s >= 1.0:
        I1 = (q_break ** (p1 + 1) - q_low ** (p1 + 1)) / (p1 + 1)
        I2 = A1 * (q_twin ** (p2 + 1) - q_break ** (p2 + 1)) / (p2 + 1)
        I3 = A2 * A1 * (1.0 ** (p2 + 1) - q_twin ** (p2 + 1)) / (p2 + 1)
        Norm = 1.0 / (I1 + I2 + I3)
        m1 = x <= Norm * I1
        m2 = (x > Norm * I1) & (x <= Norm * (I1 + I2))
        m3 = (x > Norm * (I1 + I2)) & (x <= Norm * (I1 + I2 + I3))
        x[m1] = (x[m1] / Norm * (p1 + 1) + q_low ** (p1 + 1)) ** (1.0 / (p1 + 1))
        x[m2] = ((x[m2] / Norm - I1) * (p2 + 1) / A1 + q_break ** (p2 + 1)) ** (1.0 / (p2 + 1))
        xx = x[m3]
        x[m3] = ((xx / Norm - I1 - I2) * (p2 + 1) / (A1 * A2) + q_twin ** (p2 + 1)) ** (1.0 / (p2 + 1))
    elif M_s >= 0.3:
        q_min = max(q_low / M_s, q_low)
        I1 = (q_break ** (p1 + 1) - q_min ** (p1 + 1)) / (p1 + 1)
        I2 = A1 * (q_twin ** (p2 + 1) - q_break ** (p2 + 1)) / (p2 + 1)
        I3 = A2 * A1 * (1.0 ** (p2 + 1) - q_twin ** (p2 + 1)) / (p2 + 1)
        Norm = 1.0 / (I1 + I2 + I3)
        m1 = x <= Norm * I1
        m2 = (x > Norm * I1) & (x <= Norm * (I1 + I2))
        m3 = (x > Norm * (I1 + I2)) & (x <= Norm * (I1 + I2 + I3))
        x[m1] = (x[m1] / Norm * (p1 + 1) + q_min ** (p1 + 1)) ** (1.0 / (p1 + 1))
        x[m2] = ((x[m2] / Norm - I1) * (p2 + 1) / A1 + q_break ** (p2 + 1)) ** (1.0 / (p2 + 1))
        xx = x[m3]
        x[m3] = ((xx / Norm - I1 - I2) * (p2 + 1) / (A1 * A2) + q_twin ** (p2 + 1)) ** (1.0 / (p2 + 1))
    else:
        # Physical mass-ratio domain requires q <= 1.0.  The companion-mass
        # lower bound (MOE2017_Q_BREAK_LOW Msun) relative to a sub-0.1 Msun
        # primary would otherwise produce q_min > 1.0 and negative/meaningless
        # integrals; clamp against the hydrogen-burning floor and the twin
        # threshold.
        q_min = min(q_low / max(M_s, WINTERS_2019_M_DWARF_MINIMUM_MASS_SOLAR), q_twin)
        I2 = (q_twin - q_min) - (q_break - q_min)
        I3 = p2 * (q_twin - q_min)
        Norm = 1.0 / (I2 + I3)
        m2 = x <= Norm * I2
        m3 = ~m2
        x[m2] = (x[m2] / Norm) + q_min
        x[m3] = ((x[m3] - Norm * I2) * (-1.0 / Norm) + q_twin)
    return x


# ---------------------------------------------------------------------------
# Log-prior for bound companions  (Raghavan+ 2010; Moe & Di Stefano 2017)
# ---------------------------------------------------------------------------

def _winters2019_m_dwarf_companion_cdf(
    primary_mass_solar: float,
    maximum_separation_au: np.ndarray,
) -> np.ndarray:
    """Return Winters et al. (2019) M-dwarf companion probability.

    The measured M-dwarf system multiplicity normalizes a Gaussian CDF in
    log10 projected separation.  This is applicable only to the observed
    0.075--0.60 solar-mass primary range; it is not an extrapolated
    mass-dependent multiplicity law.
    """
    if not math.isfinite(primary_mass_solar) or (
        primary_mass_solar < WINTERS_2019_M_DWARF_MINIMUM_MASS_SOLAR
    ):
        raise ValueError(
            "Winters et al. (2019) M-dwarf prior requires a primary mass of at least 0.075 solar masses"
        )
    separation_au = np.asarray(maximum_separation_au, dtype=float)
    if not np.all(np.isfinite(separation_au)) or np.any(separation_au <= 0.0):
        raise ValueError("maximum bound-companion separations must be finite and positive")
    standardized_log_separation = (
        np.log10(separation_au) - WINTERS_2019_LOG10_SEPARATION_MEAN_AU
    ) / WINTERS_2019_LOG10_SEPARATION_STANDARD_DEVIATION_DEX
    return WINTERS_2019_M_DWARF_MULTIPLICITY_FRACTION * _standard_normal_cdf(
        standardized_log_separation
    )

def lnprior_bound(
    M_s: float,
    delta_mags: np.ndarray,
    separations: np.ndarray,
    contrasts: np.ndarray,
    plx: float,
) -> np.ndarray:
    """Log-prior probability of a bound companion at given contrasts.

    Implements the occurrence-rate-weighted prior from Winters et al. (2019)
    within its measured M-dwarf domain and from Moe & Di Stefano (2017) for
    more massive hosts.

    Args:
        M_s: Primary mass [Msun].
        delta_mags: Companion contrasts [delta_mag], shape (N,).
        separations: Separation values [arcsec] from contrast curve.
        contrasts: Contrast values [delta_mag] from contrast curve.
        plx: Finite positive parallax [mas].

    Returns:
        ln(prior_bound), shape (N,).
    """
    if not math.isfinite(plx) or plx <= 0.0:
        raise ValueError("bound-companion prior requires a finite positive parallax")
    d_pc = MILLIARCSECONDS_PER_ARCSECOND / plx
    seps_arcsec = separation_at_contrast(
        np.asarray(delta_mags, dtype=float),
        np.asarray(separations, dtype=float),
        np.asarray(contrasts, dtype=float),
    )
    seps_au = d_pc * seps_arcsec
    max_Porbs = ((4.0 * pi ** 2) / (G * M_s * Msun) * (seps_au * au) ** 3) ** 0.5 / SECONDS_PER_DAY

    if M_s <= WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR:
        return np.log(_winters2019_m_dwarf_companion_cdf(M_s, seps_au))

    log10_M_s = math.log10(M_s)
    _c1, _c2, _c3 = MOE2017_COMPANION_FRACTION_F1_COEFFS
    f1 = _c1 + _c2 * log10_M_s + _c3 * log10_M_s ** 2
    _c1, _c2, _c3 = MOE2017_COMPANION_FRACTION_F2_COEFFS
    f2 = _c1 + _c2 * log10_M_s + _c3 * log10_M_s ** 2
    _c1, _c2, _c3 = MOE2017_COMPANION_FRACTION_F3_COEFFS
    f3 = _c1 + _c2 * log10_M_s + _c3 * log10_M_s ** 2

    alpha = MOE2017_COMPANION_FRACTION_SLOPE
    dlogP = MOE2017_COMPANION_LOGPERIOD_BIN_DEX

    t2 = 0.5 * (2.0 - 1.0) * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (2.0 - 1.0))
    t3 = 0.5 * alpha * (3.4 ** 2 - 5.4 * 3.4 + 6.8) + f2 * (3.4 - 2.0)
    t4 = (alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
          + (f3 - f2 - alpha * dlogP) * (
              MOE2017_LOGP_QUAD_C2 * 5.5 ** 2
              - MOE2017_LOGP_QUAD_C1 * 5.5
              + MOE2017_LOGP_QUAD_C0
          ))
    t5 = f3 * MOE2017_EXP_SCALE * (1.0 - math.exp(-0.3 * (8.0 - 5.5)))

    f_comp = np.zeros_like(max_Porbs)
    m = np.log10(max_Porbs)

    r12 = (m >= 1.0) & (m < 2.0)
    r23 = (m >= 2.0) & (m < 3.4)
    r34 = (m >= 3.4) & (m < 5.5)
    r45 = (m >= 5.5) & (m < 8.0)
    r5p = m >= 8.0
    f_comp[r12] = 0.5 * (m[r12] - 1.0) * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (m[r12] - 1.0))
    f_comp[r23] = t2 + 0.5 * alpha * (m[r23] ** 2 - 5.4 * m[r23] + 6.8) + f2 * (m[r23] - 2.0)
    f_comp[r34] = t2 + t3 + alpha * dlogP * (m[r34] - 3.4) + f2 * (m[r34] - 3.4) + (
        f3 - f2 - alpha * dlogP) * (
            MOE2017_LOGP_QUAD_C2 * m[r34] ** 2
            - MOE2017_LOGP_QUAD_C1 * m[r34]
            + MOE2017_LOGP_QUAD_C0
        )
    f_comp[r45] = t2 + t3 + t4 + f3 * MOE2017_EXP_SCALE * (1.0 - np.exp(-0.3 * (m[r45] - 5.5)))
    f_comp[r5p] = t2 + t3 + t4 + t5
    f_comp[f_comp < 0.0] = 0.0
    return np.log(f_comp)


# ---------------------------------------------------------------------------
# Log-prior for background stars  (TRILEGAL-based)
# ---------------------------------------------------------------------------

def lnprior_background(
    N_comp: int,
    delta_mags: np.ndarray,
    separations: np.ndarray,
    contrasts: np.ndarray,
) -> np.ndarray:
    """Log-prior probability of a background star at given contrasts.

    Args:
        N_comp: Number of stars from TRILEGAL simulation.
        delta_mags: Companion contrasts [delta_mag], shape (N,).
        separations: Separation values [arcsec].
        contrasts: Contrast values [delta_mag].

    Returns:
        ln(prior_bg), shape (N,).
    """
    seps = separation_at_contrast(
        np.asarray(delta_mags, dtype=float),
        np.asarray(separations, dtype=float),
        np.asarray(contrasts, dtype=float),
    )
    return np.log((N_comp / 0.1) * seps ** 2 / SQUARE_ARCSECONDS_PER_SQUARE_DEGREE)


__all__ = [
    "sample_rp",
    "sample_inc",
    "sample_ecc",
    "sample_w",
    "sample_q",
    "lnprior_bound",
    "lnprior_background",
]
