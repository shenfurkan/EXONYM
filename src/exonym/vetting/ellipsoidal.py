"""Out-of-transit ellipsoidal variation amplitude vetting gate.

Evaluates out-of-transit ellipsoidal variation amplitudes to rule out
stellar-mass binary companions.
"""

from __future__ import annotations

import math
from typing import Tuple

ELLIPSOIDAL_THRESHOLD_PPM = 100.0


def ellipsoidal_variation_amplitude_ppm(
    m_companion_solar: float,
    m_host_solar: float,
    r_host_solar: float,
    semi_major_axis_au: float,
    inclination_deg: float = 90.0,
    alpha_ellip: float = 1.2,
) -> float:
    """Return predicted ellipsoidal variation amplitude in ppm."""
    if (
        m_companion_solar <= 0
        or m_host_solar <= 0
        or r_host_solar <= 0
        or semi_major_axis_au <= 0
    ):
        raise ValueError("physical parameters must be positive")
    sin_i = math.sin(math.radians(inclination_deg))
    r_host_au = r_host_solar * 0.00465047
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
    if amplitude_ppm < 0:
        raise ValueError("amplitude_ppm must be non-negative")
    return amplitude_ppm < threshold_ppm, amplitude_ppm
