"""Difference-image centroid offset significance.

``Z_centroid`` evaluates whether the transit signal originates on the target
star or an offset background source:

    Z = sqrt((d_ra*cos(dec))^2 + (d_dec)^2) / sigma_centroid

All inputs must be finite, and ``sigma_centroid`` must be positive. A missing,
sentinel, or non-finite uncertainty is an unresolved source-location test, not
an on-target result.

The construction follows the difference-image displacement diagnostic in the
project's false-positive-vetting note; the right-ascension component is scaled
by declination before its quadrature combination with declination offset.

Scientific boundary:
    A finite significance is conditional on the supplied difference-image
    uncertainty and reference frame. It is a source-location screen, not a
    calibrated PRF localization result or a validation claim.
"""

from __future__ import annotations

import math
from typing import Tuple

# ASTROPHYSICAL_HEURISTIC: The threshold is a routing convention for a finite
# offset uncertainty, never a replacement for a calibrated scene model.
CENTROID_THRESHOLD = 3.0


def centroid_offset_z(
    ra_offset_arcsec: float,
    dec_offset_arcsec: float,
    dec_deg: float,
    sigma_arcsec: float,
) -> float:
    """Return centroid-offset significance in sigma units.

    Args:
        ra_offset_arcsec: Right-ascension offset in arcseconds.
        dec_offset_arcsec: Declination offset in arcseconds.
        dec_deg: Target declination in degrees.
        sigma_arcsec: One-sigma centroid uncertainty in arcseconds.

    Raises:
        ValueError: If a measurement is non-finite, the declination is outside
            its physical range, or the uncertainty is not strictly positive.
    """
    values = (ra_offset_arcsec, dec_offset_arcsec, dec_deg, sigma_arcsec)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("centroid inputs must be finite")
    if not -90.0 <= dec_deg <= 90.0:
        raise ValueError("dec_deg must be between -90 and 90 degrees")
    if sigma_arcsec <= 0.0:
        raise ValueError("sigma_arcsec must be positive")
    separation = math.hypot(
        ra_offset_arcsec * math.cos(math.radians(dec_deg)), dec_offset_arcsec
    )
    return separation / sigma_arcsec


def centroid_gate(
    ra_offset_arcsec: float,
    dec_offset_arcsec: float,
    dec_deg: float,
    sigma_arcsec: float,
    threshold: float = CENTROID_THRESHOLD,
) -> Tuple[bool, float]:
    """Return ``(passes, z_score)`` for a valid centroid measurement.

    A ``ValueError`` from :func:`centroid_offset_z` means the source-location
    measurement is unresolved and must not be recorded as a passing result.
    """
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")
    z = centroid_offset_z(ra_offset_arcsec, dec_offset_arcsec, dec_deg, sigma_arcsec)
    return z < threshold, z


def centroid_offset_pvalue(z: float) -> float:
    """Return Rayleigh p-value for 2D centroid offset Z score: p = exp(-z^2 / 2)."""
    if not math.isfinite(z) or z < 0:
        raise ValueError("Z score must be finite and non-negative")
    return math.exp(-0.5 * (z**2))
