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
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.special import ndtr as _standard_normal_cdf
from scipy.stats import beta as _beta_dist
from scipy.stats import powerlaw as _powerlaw_dist

from ...constants import SQUARE_ARCSECONDS_PER_SQUARE_DEGREE
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

# ---------------------------------------------------------------------------
# Planet radius prior  (Fressin et al. 2013, broken power law)
# ---------------------------------------------------------------------------

# Pre-computed normalisation constants
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
        return x * 19.5 + 0.5

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
        return _beta_dist.ppf(x, 0.867, 3.030)
    elif P_orb <= 10.0:
        return _powerlaw_dist.ppf(x, 0.2)
    else:
        return _powerlaw_dist.ppf(x, 0.6)


# ---------------------------------------------------------------------------
# Argument of periastron  (uniform)
# ---------------------------------------------------------------------------

def sample_w(x: np.ndarray) -> np.ndarray:
    """Sample argument of periastron uniformly in [0, 360) deg."""
    return np.asarray(x, dtype=float) * 360.0


# ---------------------------------------------------------------------------
# Mass-ratio prior for short-period binaries  (Moe & Di Stefano 2017)
# ---------------------------------------------------------------------------

def sample_q(x: np.ndarray, M_s: float) -> np.ndarray:
    """Sample binary mass ratios q = M_sec / M_prim.

    Broken power-law from Moe & Di Stefano (2017) with twin fraction
    F_twin = 0.30 for q >= 0.95.
    """
    x = np.asarray(x, dtype=float).copy()
    if M_s >= 1.0:
        p1, p2 = 0.3, -0.5
        A1 = 0.3 ** p1 / 0.3 ** p2
        F_twin = 0.30
        A2 = 1.0 + (F_twin / (1.0 - F_twin)) * (
            (1.0 ** (p2 + 1) - 0.3 ** (p2 + 1)) / (p2 + 1)
        ) / ((1.0 ** (p2 + 1) - 0.95 ** (p2 + 1)) / (p2 + 1))
        I1 = (0.3 ** (p1 + 1) - 0.1 ** (p1 + 1)) / (p1 + 1)
        I2 = A1 * (0.95 ** (p2 + 1) - 0.3 ** (p2 + 1)) / (p2 + 1)
        I3 = A2 * A1 * (1.0 ** (p2 + 1) - 0.95 ** (p2 + 1)) / (p2 + 1)
        Norm = 1.0 / (I1 + I2 + I3)
        m1 = x <= Norm * I1
        m2 = (x > Norm * I1) & (x <= Norm * (I1 + I2))
        m3 = (x > Norm * (I1 + I2)) & (x <= Norm * (I1 + I2 + I3))
        x[m1] = (x[m1] / Norm * (p1 + 1) + 0.1 ** (p1 + 1)) ** (1.0 / (p1 + 1))
        x[m2] = ((x[m2] / Norm - I1) * (p2 + 1) / A1 + 0.3 ** (p2 + 1)) ** (1.0 / (p2 + 1))
        xx = x[m3]
        x[m3] = ((xx / Norm - I1 - I2) * (p2 + 1) / (A1 * A2) + 0.95 ** (p2 + 1)) ** (1.0 / (p2 + 1))
    elif M_s >= 0.3:
        q_min = max(0.1 / M_s, 0.1)
        p1, p2 = 0.3, -0.5
        A1 = 0.3 ** p1 / 0.3 ** p2
        F_twin = 0.30
        A2 = 1.0 + (F_twin / (1.0 - F_twin)) * (
            (1.0 ** (p2 + 1) - 0.3 ** (p2 + 1)) / (p2 + 1)
        ) / ((1.0 ** (p2 + 1) - 0.95 ** (p2 + 1)) / (p2 + 1))
        I1 = (0.3 ** (p1 + 1) - q_min ** (p1 + 1)) / (p1 + 1)
        I2 = A1 * (0.95 ** (p2 + 1) - 0.3 ** (p2 + 1)) / (p2 + 1)
        I3 = A2 * A1 * (1.0 ** (p2 + 1) - 0.95 ** (p2 + 1)) / (p2 + 1)
        Norm = 1.0 / (I1 + I2 + I3)
        m1 = x <= Norm * I1
        m2 = (x > Norm * I1) & (x <= Norm * (I1 + I2))
        m3 = (x > Norm * (I1 + I2)) & (x <= Norm * (I1 + I2 + I3))
        x[m1] = (x[m1] / Norm * (p1 + 1) + q_min ** (p1 + 1)) ** (1.0 / (p1 + 1))
        x[m2] = ((x[m2] / Norm - I1) * (p2 + 1) / A1 + 0.3 ** (p2 + 1)) ** (1.0 / (p2 + 1))
        xx = x[m3]
        x[m3] = ((xx / Norm - I1 - I2) * (p2 + 1) / (A1 * A2) + 0.95 ** (p2 + 1)) ** (1.0 / (p2 + 1))
    else:
        q_min = 0.1 / M_s
        F_twin = 0.30
        I2 = (0.95 - q_min) - (0.3 - q_min)
        I3 = -0.5 * (0.95 - q_min)
        Norm = 1.0 / (I2 + I3)
        m2 = x <= Norm * I2
        m3 = ~m2
        x[m2] = (x[m2] / Norm) + q_min
        x[m3] = ((x[m3] - Norm * I2) * (-1.0 / Norm) + 0.95)
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
    d_pc = 1000.0 / plx
    seps_arcsec = separation_at_contrast(
        np.asarray(delta_mags, dtype=float),
        np.asarray(separations, dtype=float),
        np.asarray(contrasts, dtype=float),
    )
    seps_au = d_pc * seps_arcsec
    max_Porbs = ((4.0 * pi ** 2) / (G * M_s * Msun) * (seps_au * au) ** 3) ** 0.5 / SECONDS_PER_DAY

    if M_s <= WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR:
        return np.log(_winters2019_m_dwarf_companion_cdf(M_s, seps_au))

    f1 = 0.020 + 0.04 * math.log10(M_s) + 0.07 * (math.log10(M_s)) ** 2
    f2 = 0.039 + 0.07 * math.log10(M_s) + 0.01 * (math.log10(M_s)) ** 2
    f3 = 0.078 - 0.05 * math.log10(M_s) + 0.04 * (math.log10(M_s)) ** 2

    alpha = 0.018
    dlogP = 0.7

    t2 = 0.5 * (2.0 - 1.0) * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (2.0 - 1.0))
    t3 = 0.5 * alpha * (3.4 ** 2 - 5.4 * 3.4 + 6.8) + f2 * (3.4 - 2.0)
    t4 = (alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
          + (f3 - f2 - alpha * dlogP) * (0.238095 * 5.5 ** 2 - 0.952381 * 5.5 + 0.485714))
    t5 = f3 * (3.33333 - 17.3566 * math.exp(-0.3 * 8.0))

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
        f3 - f2 - alpha * dlogP) * (0.238095 * m[r34] ** 2 - 0.952381 * m[r34] + 0.485714)
    f_comp[r45] = t2 + t3 + t4 + f3 * (3.33333 - 17.3566 * np.exp(-0.3 * m[r45]))
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
