"""Target-independent helpers for phase-folded transit light curves."""

from __future__ import annotations

import math
import re
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def parse_tess_sector(mission: object) -> Optional[int]:
    """Return the TESS sector number in a MAST mission label, if present."""
    match = re.search(r"\bSector\s+(\d+)\b", str(mission), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def phase_hours(
    time_btjd: Sequence[float], period_days: float, epoch_btjd: float
) -> np.ndarray:
    """Return signed hours from the nearest transit center for each BTJD time."""
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
    """Estimate median in/out transit depth and uncertainty in ppm.

    The in-transit window spans one catalog duration. Symmetric out-of-transit
    windows span 1.2 to 2.5 durations from transit center. Invalid samples are
    excluded before both the depth and its robust standard-error estimate.
    """
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")

    time = np.asarray(time_btjd, dtype=float)
    values = np.asarray(flux, dtype=float)
    if time.shape != values.shape:
        raise ValueError("time_btjd and flux must have identical shapes")

    hours = phase_hours(time, period_days, epoch_btjd)
    finite = np.isfinite(hours) & np.isfinite(values)
    in_transit = finite & (np.abs(hours) < 0.5 * duration_hours)
    out_of_transit = finite & (np.abs(hours) > 1.2 * duration_hours) & (
        np.abs(hours) < 2.5 * duration_hours
    )
    in_values = values[in_transit]
    out_values = values[out_of_transit]
    if not in_values.size or not out_values.size:
        raise ValueError("insufficient finite in-transit or out-of-transit coverage")

    depth = float(np.median(out_values) - np.median(in_values))
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
    """Phase-fold and median-bin a light curve for a diagnostic plot."""
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
        if index == bin_count - 1:
            mask = valid & (hours >= edges[index]) & (hours <= edges[index + 1])
        else:
            mask = valid & (hours >= edges[index]) & (hours < edges[index + 1])
        samples = values[mask]
        if samples.size >= 3:
            median[index] = np.median(samples)
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
) -> Dict[str, float]:
    """Return dict of contact durations T_14, T_23, T_12 in hours and grazing status."""
    if period_days <= 0 or r_star_solar <= 0 or m_star_solar <= 0 or r_planet_earth <= 0:
        raise ValueError("physical parameters must be positive")
    if not (0.0 <= eccentricity < 1.0):
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    if impact_parameter_b < 0:
        raise ValueError("impact_parameter_b must be non-negative")

    k = (r_planet_earth * 0.0091577) / r_star_solar
    if impact_parameter_b >= 1.0 + k:
        return {
            "T14_hr": 0.0,
            "T23_hr": 0.0,
            "T12_hr": 0.0,
            "grazing": 1.0,
            "v_stat": 1.0,
        }

    rho_solar_gcm3 = 1.408
    rho_star_gcm3 = m_star_solar / (r_star_solar**3) * rho_solar_gcm3
    g_cgs = 6.67430e-8
    period_sec = period_days * 86400.0
    a_over_r = ((g_cgs * (period_sec**2) * rho_star_gcm3) / (3.0 * math.pi)) ** (1.0 / 3.0)

    ecc_factor = math.sqrt(1.0 - eccentricity**2) / (
        1.0 + eccentricity * math.sin(math.radians(omega_deg))
    )

    t14_sec = (
        (period_sec / math.pi)
        * (1.0 / a_over_r)
        * math.sqrt(max(0.0, (1.0 + k) ** 2 - impact_parameter_b**2))
        * ecc_factor
    )
    grazing = impact_parameter_b > (1.0 - k)
    t23_sec = 0.0
    if not grazing:
        t23_sec = (
            (period_sec / math.pi)
            * (1.0 / a_over_r)
            * math.sqrt(max(0.0, (1.0 - k) ** 2 - impact_parameter_b**2))
            * ecc_factor
        )

    t12_sec = 0.5 * (t14_sec - t23_sec)
    v_stat = (2.0 * t12_sec / t14_sec) if t14_sec > 0 else 1.0

    return {
        "T14_hr": round(t14_sec / 3600.0, 4),
        "T23_hr": round(t23_sec / 3600.0, 4),
        "T12_hr": round(t12_sec / 3600.0, 4),
        "grazing": 1.0 if grazing else 0.0,
        "v_stat": round(v_stat, 4),
    }


def kipping_to_quadratic_limb_darkening(q1: float, q2: float) -> Tuple[float, float]:
    """Convert Kipping (2013) hyper-cube parameters (q1, q2) to quadratic (u1, u2)."""
    if not (0.0 <= q1 <= 1.0 and 0.0 <= q2 <= 1.0):
        raise ValueError("q1 and q2 must be in [0, 1]")
    sqrt_q1 = math.sqrt(q1)
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    return u1, u2


def quadratic_to_kipping_limb_darkening(u1: float, u2: float) -> Tuple[float, float]:
    """Convert quadratic limb darkening parameters (u1, u2) to Kipping (q1, q2)."""
    q1 = (u1 + u2) ** 2
    if q1 == 0:
        return 0.0, 0.0
    q2 = u1 / (2.0 * (u1 + u2))
    return q1, q2
