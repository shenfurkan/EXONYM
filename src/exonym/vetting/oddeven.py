"""Odd-even transit depth asymmetry test.

``Z_odd-even`` rules out a secondary eclipsing binary with twice the assumed
orbital period:

    Z = |depth_odd - depth_even| / sqrt(sigma_odd^2 + sigma_even^2)

Pass (planetary candidate): Z < 3.0 sigma.
"""

from __future__ import annotations

import math
from typing import Tuple

ODD_EVEN_THRESHOLD = 3.0


def odd_even_z(
    depth_odd: float,
    sigma_odd: float,
    depth_even: float,
    sigma_even: float,
) -> float:
    """Return the odd-even asymmetry significance Z in sigma units."""
    if sigma_odd <= 0 or sigma_even <= 0:
        raise ValueError("depth uncertainties must be positive")
    return abs(depth_odd - depth_even) / math.hypot(sigma_odd, sigma_even)


def odd_even_gate(
    depth_odd: float,
    sigma_odd: float,
    depth_even: float,
    sigma_even: float,
    threshold: float = ODD_EVEN_THRESHOLD,
) -> Tuple[bool, float]:
    """Return (pass, Z). Pass means depths are consistent within threshold."""
    z = odd_even_z(depth_odd, sigma_odd, depth_even, sigma_even)
    return z < threshold, z
