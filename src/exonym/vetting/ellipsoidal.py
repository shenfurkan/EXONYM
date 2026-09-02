"""Leading-order ellipsoidal-variation screening diagnostic.

The amplitude helper implements the tidal scaling summarized in the project's
false-positive diagnostic note: companion-to-host mass ratio, stellar radius
relative to separation, and orbital inclination set the leading-order signal.
It provides a physically motivated warning for stellar-mass binary scenarios.

Scientific boundary:
    The leading-order approximation omits detailed stellar response and
    correlated photometric variability. Its gate is a screening input, not a
    companion-mass measurement or a validation conclusion.
"""

from __future__ import annotations

import math
from typing import Tuple

from ..constants import ASTRONOMICAL_UNIT_M, NOMINAL_SOLAR_RADIUS_M

# Single-source-of-truth conversion (IAU nominal values from constants.py).
SOLAR_RADIUS_TO_AU = NOMINAL_SOLAR_RADIUS_M / ASTRONOMICAL_UNIT_M

# ASTROPHYSICAL_HEURISTIC: The retained amplitude threshold is a conservative
# binary-scenario screen, not a calibrated population probability.
ELLIPSOIDAL_THRESHOLD_PPM = 100.0
CONVECTIVE_GRAVITY_DARKENING_EXPONENT = 0.32
RADIATIVE_GRAVITY_DARKENING_EXPONENT = 1.0
RADIATIVE_ENVELOPE_TRANSITION_K = 6500.0


def gravity_darkening_exponent(teff_k: float) -> float:
    """Return the envelope-appropriate gravity-darkening exponent.

    The convective and radiative envelope limits are used in the Morris
    (1985, ApJ 295, 143, doi:10.1086/163359) leading-order relation.
    """
    temperature = float(teff_k)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("teff_k must be positive finite")
    return (
        RADIATIVE_GRAVITY_DARKENING_EXPONENT
        if temperature >= RADIATIVE_ENVELOPE_TRANSITION_K
        else CONVECTIVE_GRAVITY_DARKENING_EXPONENT
    )


def morris_ellipsoidal_coefficient(
    u_linear: float = 0.0, g_darkening: float = 0.0
) -> float:
    """Return the Morris (1985) ellipsoidal-response coefficient."""
    limb_darkening = float(u_linear)
    gravity_darkening = float(g_darkening)
    if not math.isfinite(limb_darkening) or not 0.0 <= limb_darkening < 3.0:
        raise ValueError("u_linear must satisfy 0 <= u < 3")
    if not math.isfinite(gravity_darkening) or gravity_darkening < 0.0:
        raise ValueError("g_darkening must be finite and non-negative")
    return (
        0.15
        * (15.0 + limb_darkening)
        * (1.0 + gravity_darkening)
        / (3.0 - limb_darkening)
    )


def ellipsoidal_variation_amplitude_ppm(
    m_companion_solar: float,
    m_host_solar: float,
    r_host_solar: float,
    semi_major_axis_au: float,
    teff_k: float,
    u_linear: float,
    inclination_deg: float = 90.0,
) -> float:
    """Return the Morris (1985) ellipsoidal variation amplitude in ppm."""
    values = (
        m_companion_solar,
        m_host_solar,
        r_host_solar,
        semi_major_axis_au,
        inclination_deg,
    )
    if (
        not all(math.isfinite(float(value)) for value in values)
        or m_companion_solar <= 0.0
        or m_host_solar <= 0.0
        or r_host_solar <= 0.0
        or semi_major_axis_au <= 0.0
        or not 0.0 <= inclination_deg <= 90.0
    ):
        raise ValueError("physical parameters must be finite, positive, and physically bounded")
    alpha_ellip = morris_ellipsoidal_coefficient(
        u_linear, gravity_darkening_exponent(teff_k)
    )
    sin_i = math.sin(math.radians(inclination_deg))
    r_host_au = r_host_solar * SOLAR_RADIUS_TO_AU
    r_over_a = r_host_au / semi_major_axis_au
    amplitude_fraction = (
        alpha_ellip * (m_companion_solar / m_host_solar) * (r_over_a**3) * (sin_i**2)
    )
    return amplitude_fraction * 1.0e6


def ellipsoidal_gate(
    amplitude_ppm: float,
    threshold_ppm: float = ELLIPSOIDAL_THRESHOLD_PPM,
) -> Tuple[bool, float]:
    """Return (pass, amplitude_ppm). Pass means amplitude is below threshold."""
    if not math.isfinite(amplitude_ppm) or amplitude_ppm < 0:
        raise ValueError("amplitude_ppm must be non-negative")
    if not math.isfinite(threshold_ppm) or threshold_ppm < 0:
        raise ValueError("threshold_ppm must be non-negative")
    return amplitude_ppm < threshold_ppm, amplitude_ppm
