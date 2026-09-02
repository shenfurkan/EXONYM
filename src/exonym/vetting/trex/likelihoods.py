"""Mandel-Agol transit light curve models for TREX.

Uses ``batman-package`` (Mandel & Agol 2002, Kreidberg 2015) as the
forward-model engine for quadratic limb-darkened transit and eclipse
light curves.  Supersampling is applied via batman's native ``exptime``
and ``nsamples`` parameters.

All functions operate on phase-folded time arrays relative to transit
midpoint.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import Rsun, Rearth
from .funcs import secondary_eclipse_phase


# ---------------------------------------------------------------------------
# Internal batman wrapper
# ---------------------------------------------------------------------------

def _batman_transit(
    time: np.ndarray,
    period_days: float,
    rp_rs: float,
    a_rs: float,
    inc_deg: float,
    u1: float,
    u2: float,
    exptime_days: float,
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    nsamples: int = 20,
    t0_days: float = 0.0,
) -> np.ndarray:
    """Mandel-Agol quadratic transit flux via batman."""
    import batman

    _validate_exptime_days(exptime_days)

    params = batman.TransitParams()
    params.t0 = float(t0_days)
    params.per = float(period_days)
    params.rp = float(rp_rs)
    params.a = float(a_rs)
    params.inc = float(inc_deg)
    params.ecc = float(ecc)
    params.w = float(argp_deg)
    params.limb_dark = "quadratic"
    params.u = [float(u1), float(u2)]

    model = batman.TransitModel(
        params,
        np.asarray(time, dtype=float),
        supersample_factor=nsamples,
        exp_time=exptime_days,
    )
    return model.light_curve(params)


def _validate_exptime_days(exptime_days: float) -> None:
    if not isinstance(exptime_days, (int, float, np.number)) or not np.isfinite(exptime_days) or exptime_days <= 0.0:
        raise ValueError("exptime_days must be finite and positive")


# ---------------------------------------------------------------------------
# Transit-eclipse simulators
# ---------------------------------------------------------------------------

def simulate_TP(
    time: np.ndarray,
    R_p_earth: float,
    P_orb: float,
    inc_deg: float,
    a_cm: float,
    R_s_solar: float,
    u1: float,
    u2: float,
    exptime_days: float,
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0,
    companion_is_host: bool = False,
    nsamples: int = 20,
) -> np.ndarray:
    """Simulate a transiting planet light curve.

    Args:
        time: Phase-folded times [days from transit midpoint].
        R_p_earth: Planet radius [R_earth].
        P_orb: Orbital period [days].
        inc_deg: Inclination [degrees].
        a_cm: Semi-major axis [cm].
        R_s_solar: Stellar radius [R_sun].
        u1, u2: Quadratic limb-darkening coefficients.
        ecc, argp_deg: Eccentricity, argument of periastron.
        companion_fluxratio: F_comp / (F_comp + F_target).
        companion_is_host: True if transit on unresolved companion.
        exptime_days: Exposure time [days].
        nsamples: Supersampling rate.

    Returns:
        Normalised flux array.
    """
    rp_rs = R_p_earth * Rearth / (R_s_solar * Rsun)
    a_rs = a_cm / (R_s_solar * Rsun)

    _validate_exptime_days(exptime_days)
    flux = _batman_transit(
        time, P_orb, rp_rs, a_rs, inc_deg, u1, u2, exptime_days,
        ecc, argp_deg, nsamples,
    )

    if companion_fluxratio > 0.0:
        F_target = 1.0
        F_comp = companion_fluxratio / (1.0 - companion_fluxratio)
        F_dilute = F_target / F_comp if companion_is_host else F_comp / F_target
        flux = (flux + F_dilute) / (1.0 + F_dilute)

    return flux


def simulate_EB(
    time: np.ndarray,
    R_EB_solar: float,
    EB_fluxratio: float,
    P_orb: float,
    inc_deg: float,
    a_cm: float,
    R_s_solar: float,
    u1: float,
    u2: float,
    exptime_days: float,
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0,
    companion_is_host: bool = False,
    nsamples: int = 20,
) -> Tuple[np.ndarray, float]:
    """Simulate an eclipsing binary light curve.

    Returns the full binary model and the secondary-eclipse phase.
    """
    _validate_exptime_days(exptime_days)
    F_comp = companion_fluxratio / (1.0 - companion_fluxratio) if companion_fluxratio > 0 else 0.0
    F_EB = EB_fluxratio / (1.0 - EB_fluxratio)

    k = R_EB_solar / R_s_solar
    if abs(k - 1.0) < 1e-6:
        k *= 0.999
    a_rs = a_cm / (R_s_solar * Rsun)

    primary_flux = _batman_transit(
        time, P_orb, k, a_rs, inc_deg, u1, u2, exptime_days,
        ecc, argp_deg, nsamples,
    )
    secondary_phase = secondary_eclipse_phase(ecc, argp_deg)
    secondary_flux = _batman_transit(
        time, P_orb, 1.0 / k, a_rs / k, inc_deg, u1, u2, exptime_days,
        ecc, argp_deg + 180.0, nsamples, t0_days=secondary_phase * P_orb,
    )
    flux = (primary_flux + F_EB * secondary_flux) / (1.0 + F_EB)

    # The binary flux is normalized internally.  Only an unresolved third
    # source dilutes it, depending on which source hosts the binary.
    if companion_is_host:
        flux = (F_comp * flux + 1.0) / (1.0 + F_comp)
    elif F_comp > 0.0:
        flux = (flux + F_comp) / (1.0 + F_comp)

    return flux, secondary_phase


# ---------------------------------------------------------------------------
# Log-likelihood functions
# ---------------------------------------------------------------------------

def lnL_TP(
    time: np.ndarray, flux: np.ndarray, sigma: float,
    R_p_earth: float, P_orb: float, inc_deg: float, a_cm: float,
    R_s_solar: float, u1: float, u2: float, exptime_days: float,
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0, companion_is_host: bool = False,
    nsamples: int = 20,
) -> float:
    """Log-likelihood for a transiting planet scenario.

    lnL = -0.5 * sum((flux_obs - flux_model)^2 / sigma^2)
    """
    model = simulate_TP(
        time, R_p_earth, P_orb, inc_deg, a_cm, R_s_solar, u1, u2,
        exptime_days, ecc, argp_deg, companion_fluxratio, companion_is_host,
        nsamples,
    )
    return float(-0.5 * np.sum((flux - model) ** 2 / sigma ** 2))


def lnL_EB(
    time: np.ndarray, flux: np.ndarray, sigma: float,
    R_EB_solar: float, EB_fluxratio: float, P_orb: float,
    inc_deg: float, a_cm: float, R_s_solar: float,
    u1: float, u2: float, exptime_days: float,
    ecc: float = 0.0, argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0, companion_is_host: bool = False,
    nsamples: int = 20,
) -> float:
    """Log-likelihood for an eclipsing binary scenario.

    Both primary and secondary eclipses are evaluated at the observed times.
    """
    model, _ = simulate_EB(
        time, R_EB_solar, EB_fluxratio, P_orb, inc_deg, a_cm, R_s_solar,
        u1, u2, exptime_days, ecc, argp_deg, companion_fluxratio,
        companion_is_host, nsamples,
    )
    return float(-0.5 * np.sum((flux - model) ** 2 / sigma ** 2))


__all__ = [
    "_batman_transit",
    "simulate_TP",
    "simulate_EB",
    "lnL_TP",
    "lnL_EB",
]
