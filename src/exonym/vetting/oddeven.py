"""Odd-even transit depth asymmetry test.

``Z_odd-even`` rules out a secondary eclipsing binary with twice the assumed
orbital period:

    Z = |depth_odd - depth_even| / sqrt(sigma_odd^2 + sigma_even^2)

The statistic is the independent-error depth comparison described in the
project's false-positive diagnostic note. It is useful for exposing an
alternating-eclipse interpretation of a periodic signal.

Scientific boundary:
    A small statistic means this particular screen found no resolved odd/even
    discrepancy; it does not establish a planetary origin or a validation
    claim.

Units and fail-closed boundary
------------------------------
Odd/even depths and their errors must share one explicit unit (normally ppm or
dimensionless normalized relative flux); the returned Z score is dimensionless
sigma.  This is elementary independent-error propagation rather than a distinct
retained astrophysical relation, so no paper citation is fabricated.  Nonfinite
or nonpositive errors must be rejected by the caller/guard; a small Z only means
this screen did not resolve an alternating depth and cannot set
``claim_eligible``.
"""

from __future__ import annotations

import math
from typing import Tuple

# ASTROPHYSICAL_HEURISTIC: This routing threshold is a screening convention;
# its output remains candidate-local diagnostic evidence.
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
