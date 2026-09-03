"""Physics and utility functions for the TREX statistical vetting engine.

This module provides:

* Stellar mass-radius-Teff relations (Torres + CDWRF).
* TESS magnitude estimation from 2MASS photometry (Stassun et al. 2018).
* Contrast-curve helpers and separation-at-contrast interpolation.
* Flux-to-magnitude conversions and dilution helpers.
* Keplerian orbital mechanics (semi-major axis, impact parameter).

All functions are pure NumPy/SciPy with no side effects.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from astropy.constants import c, h, k_B
from scipy.interpolate import InterpolatedUnivariateSpline

from .constants import GRAVITATIONAL_CONSTANT_CGS, SECONDS_PER_DAY


# TESS's published pivot wavelength is used for the Planck surface-brightness
# ratio; common prefactors cancel when comparing two stellar photospheres.
_TESS_PIVOT_WAVELENGTH_M = 786.5e-9

# ---------------------------------------------------------------------------
# Torres (2010) + Chabrier CDWRF mass-radius-Teff splines
# ---------------------------------------------------------------------------

_Mass_nodes_Torres = np.array(
    [0.26, 0.47, 0.59, 0.69, 0.87, 0.98, 1.085, 1.4, 1.65, 2.0, 2.5, 3.0, 4.4, 15.0, 40.0],
    dtype=float,
)
_Teff_nodes_Torres = np.array(
    [3170, 3520, 3840, 4410, 5150, 5560, 5940, 6650, 7300, 8180, 9790, 11400, 15200, 30000, 42000],
    dtype=float,
)
_Rad_nodes_Torres = np.array(
    [0.28, 0.47, 0.60, 0.72, 0.9, 1.05, 1.2, 1.55, 1.8, 2.1, 2.4, 2.6, 3.0, 6.2, 11.0],
    dtype=float,
)
_Teff_spline_Torres = InterpolatedUnivariateSpline(_Mass_nodes_Torres, _Teff_nodes_Torres)
_Rad_spline_Torres = InterpolatedUnivariateSpline(_Mass_nodes_Torres, _Rad_nodes_Torres)

_Mass_nodes_cdwrf = np.array([0.1, 0.135, 0.2, 0.35, 0.48, 0.58, 0.63], dtype=float)
_Teff_nodes_cdwrf = np.array([2800, 3000, 3200, 3400, 3600, 3800, 4000], dtype=float)
_Rad_nodes_cdwrf = np.array([0.12, 0.165, 0.23, 0.36, 0.48, 0.585, 0.6], dtype=float)
_Teff_spline_cdwrf = InterpolatedUnivariateSpline(_Mass_nodes_cdwrf, _Teff_nodes_cdwrf)
_Rad_spline_cdwrf = InterpolatedUnivariateSpline(_Mass_nodes_cdwrf, _Rad_nodes_cdwrf)


def stellar_relations(
    masses: np.ndarray,
    max_radii: Optional[np.ndarray] = None,
    max_teffs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate radii and effective temperatures from stellar masses.

    Uses Torres et al. (2010) for M > 0.63 Msun and the Chabrier CDWRF
    grid for M <= 0.63 Msun.

    Args:
        masses: Star masses [Solar masses], shape (N,).
        max_radii: Optional per-star maximum radii [Solar radii].
        max_teffs: Optional per-star maximum effective temperatures [K].

    Returns:
        (radii, teffs): Radii [Solar radii], Teffs [K].
    """
    masses = np.asarray(masses, dtype=float)
    radii = np.zeros_like(masses)
    teffs = np.zeros_like(masses)
    hot = masses > 0.63
    cool = ~hot
    if np.any(hot):
        radii[hot] = _Rad_spline_Torres(masses[hot])
        teffs[hot] = _Teff_spline_Torres(masses[hot])
    if np.any(cool):
        radii[cool] = _Rad_spline_cdwrf(masses[cool])
        teffs[cool] = _Teff_spline_cdwrf(masses[cool])
    if max_radii is not None:
        exceed = radii > max_radii
        radii[exceed] = max_radii[exceed]
    if max_teffs is not None:
        exceed = teffs > max_teffs
        teffs[exceed] = max_teffs[exceed]
    radii[radii < 0.1] = 0.1
    teffs[teffs < 2800.0] = 2800.0
    return radii, teffs


# ---------------------------------------------------------------------------
# TESS magnitude from 2MASS J, Ks  (Stassun et al. 2018, Section 2.2.1.1)
# ---------------------------------------------------------------------------

def J_Ks_to_Tmag(J_mags: np.ndarray, Ks_mags: np.ndarray) -> np.ndarray:
    """Convert 2MASS J and Ks to TESS T magnitudes.

    Piecewise polynomial relations from Stassun et al. (2018, AJ, 156, 102).
    """
    J = np.asarray(J_mags, dtype=float)
    Ks = np.asarray(Ks_mags, dtype=float)
    JK = J - Ks
    Tmags = np.full_like(J, np.nan)
    m1 = (-0.1 <= JK) & (JK <= 0.70)
    m2 = (0.7 < JK) & (JK <= 1.0)
    m3 = JK < -0.1
    m4 = JK > 1.0
    first_relation = lambda colour: (
        1.22163 * colour ** 3 - 1.74299 * colour ** 2 + 1.89115 * colour + 0.0563
    )
    second_relation = lambda colour: (
        -269.372 * colour ** 3 + 668.453 * colour ** 2 - 545.64 * colour + 147.811
    )
    second_offset = first_relation(0.70) - second_relation(0.70)
    Tmags[m1] = J[m1] + first_relation(JK[m1])
    Tmags[m2] = J[m2] + second_relation(JK[m2]) + second_offset
    Tmags[m3] = J[m3] + first_relation(-0.1)
    Tmags[m4] = J[m4] + second_relation(1.0) + second_offset
    return Tmags


# ---------------------------------------------------------------------------
# Flux ratio / dilution helpers
# ---------------------------------------------------------------------------

def companion_flux_ratio(flux_frac: np.ndarray) -> np.ndarray:
    """Convert F_comp/(F_comp+F_target) to F_comp/F_target."""
    flux_frac = np.asarray(flux_frac, dtype=float)
    denom = 1.0 - flux_frac
    denom[denom <= 0.0] = np.inf
    return flux_frac / denom


def dilute_flux(
    flux: np.ndarray, companion_fluxratio: float, companion_is_host: bool = False,
) -> np.ndarray:
    """Apply flux dilution from an unresolved companion."""
    if companion_fluxratio <= 0.0:
        return flux
    F_target = 1.0
    F_comp = companion_fluxratio / (1.0 - companion_fluxratio)
    if companion_is_host:
        F_dilute = F_target / F_comp
    else:
        F_dilute = F_comp / F_target
    return (flux + F_dilute) / (1.0 + F_dilute)


# ---------------------------------------------------------------------------
# Contrast curves
# ---------------------------------------------------------------------------

def separation_at_contrast(
    delta_mags: np.ndarray,
    separations: np.ndarray,
    contrasts: np.ndarray,
) -> np.ndarray:
    """Interpolate limiting separation at given contrast values.

    Args:
        delta_mags: Contrast values (delta_mag), shape (N,).
        separations: Separation values [arcsec], shape (M,).
        contrasts: Contrast values [delta_mag], shape (M,).

    Returns:
        Limiting separations [arcsec].
    """
    delta_mags = np.asarray(delta_mags, dtype=float)
    s = np.asarray(separations, dtype=float)
    c = np.asarray(contrasts, dtype=float)
    order = np.argsort(c)
    spline = InterpolatedUnivariateSpline(c[order], s[order], k=1)
    result = spline(delta_mags)
    result = np.clip(result, np.min(s), np.max(s))
    return result


# ---------------------------------------------------------------------------
# Magnitude / flux conversion
# ---------------------------------------------------------------------------

def delta_mag_to_flux_ratio(delta_mag: np.ndarray) -> np.ndarray:
    """Convert delta-magnitude to flux ratio: F2/F1 = 10^(-0.4 * dm)."""
    return np.power(10.0, -0.4 * np.asarray(delta_mag, dtype=float))


def flux_ratio_to_delta_mag(flux_ratio: np.ndarray) -> np.ndarray:
    """Convert flux ratio to delta-magnitude: dm = -2.5 * log10(F2/F1)."""
    flux_ratio = np.asarray(flux_ratio, dtype=float)
    return -2.5 * np.log10(np.maximum(flux_ratio, 1e-30))


def tess_surface_brightness_ratio(
    numerator_teff_k: np.ndarray,
    denominator_teff_k: np.ndarray,
) -> np.ndarray:
    """Return the TESS-band Planck surface-brightness ratio of two stars.

    The ratio is evaluated at the TESS pivot wavelength.  This preserves the
    temperature dependence of the observed band while cancelling the Planck
    prefactor shared by the two photospheres.
    """
    numerator = np.asarray(numerator_teff_k, dtype=float)
    denominator = np.asarray(denominator_teff_k, dtype=float)
    if (
        not np.all(np.isfinite(numerator))
        or not np.all(np.isfinite(denominator))
        or np.any(numerator <= 0.0)
        or np.any(denominator <= 0.0)
    ):
        raise ValueError("effective temperatures must be finite and positive")
    exponent_factor = (h.value * c.value) / (_TESS_PIVOT_WAVELENGTH_M * k_B.value)
    return np.expm1(exponent_factor / denominator) / np.expm1(exponent_factor / numerator)


# ---------------------------------------------------------------------------
# Keplerian orbital mechanics
# ---------------------------------------------------------------------------

def semi_major_axis_cgs(period_days: float, M_total_g: Any) -> Any:
    """Semi-major axis [cm] from Kepler's third law.

    a^3 = G * M_total * P^2 / (4 * pi^2)
    """
    P_s = period_days * SECONDS_PER_DAY
    val = (GRAVITATIONAL_CONSTANT_CGS * np.asarray(M_total_g, dtype=float) * P_s ** 2 / (4.0 * math.pi ** 2)) ** (1.0 / 3.0)
    if isinstance(M_total_g, (np.ndarray, list)):
        return val
    return float(val)


def a_over_Rs(period_days: float, M_s_g: float, R_s_cm: float, M_p_g: float = 0.0) -> float:
    """Scaled semi-major axis a / R_s."""
    a_cm = semi_major_axis_cgs(period_days, M_s_g + M_p_g)
    return a_cm / R_s_cm


def impact_parameter(
    inc_deg: float, a_Rs: float, ecc: float = 0.0, argp_deg: float = 90.0,
) -> float:
    """Impact parameter b for circular or eccentric orbits.

    b = (a/R_s) * cos(i) * (1-e^2) / (1 + e*sin(w))
    """
    cos_i = math.cos(math.radians(inc_deg))
    if ecc == 0.0:
        return a_Rs * abs(cos_i)
    sin_w = math.sin(math.radians(argp_deg))
    return a_Rs * abs(cos_i) * (1.0 - ecc ** 2) / (1.0 + ecc * sin_w)


def secondary_eclipse_phase(ecc: float, argp_deg: float) -> float:
    """Return the secondary-eclipse phase from Keplerian mean anomalies."""
    if not math.isfinite(ecc) or not 0.0 <= ecc < 1.0:
        raise ValueError("eccentricity must be finite and in [0, 1)")
    if not math.isfinite(argp_deg):
        raise ValueError("argument of periastron must be finite")

    argp = math.radians(argp_deg)

    def mean_anomaly(true_anomaly: float) -> float:
        eccentric_anomaly = 2.0 * math.atan2(
            math.sqrt(1.0 - ecc) * math.sin(0.5 * true_anomaly),
            math.sqrt(1.0 + ecc) * math.cos(0.5 * true_anomaly),
        )
        return eccentric_anomaly - ecc * math.sin(eccentric_anomaly)

    transit_mean_anomaly = mean_anomaly(0.5 * math.pi - argp)
    occultation_mean_anomaly = mean_anomaly(1.5 * math.pi - argp)
    return (occultation_mean_anomaly - transit_mean_anomaly) % (2.0 * math.pi) / (2.0 * math.pi)


__all__ = [
    "stellar_relations",
    "J_Ks_to_Tmag",
    "companion_flux_ratio",
    "dilute_flux",
    "separation_at_contrast",
    "delta_mag_to_flux_ratio",
    "flux_ratio_to_delta_mag",
    "tess_surface_brightness_ratio",
    "semi_major_axis_cgs",
    "a_over_Rs",
    "impact_parameter",
    "secondary_eclipse_phase",
]
