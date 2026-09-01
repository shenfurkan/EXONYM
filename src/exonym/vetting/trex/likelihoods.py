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


# ---------------------------------------------------------------------------
# Internal batman wrapper
# ---------------------------------------------------------------------------

def _batman_transit(
    time: np.ndarray,
    rp_rs: float,
    a_rs: float,
    inc_deg: float,
    u1: float,
    u2: float,
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    exptime_days: float = 0.00139,
    nsamples: int = 20,
) -> np.ndarray:
    """Mandel-Agol quadratic transit flux via batman."""
    import batman

    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = 1.0  # dummy – time is phase-folded
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
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0,
    companion_is_host: bool = False,
    exptime_days: float = 0.00139,
    nsamples: int = 20,
) -> np.ndarray:
    """Simulate a transiting planet light curve.

    Args:
        time: Phase-folded times [days from transit midpoint].
        R_p_earth: Planet radius [R_earth].
        P_orb: Orbital period [days] (unused; for API compatibility).
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

    flux = _batman_transit(
        time, rp_rs, a_rs, inc_deg, u1, u2, ecc, argp_deg,
        exptime_days, nsamples,
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
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0,
    companion_is_host: bool = False,
    exptime_days: float = 0.00139,
    nsamples: int = 20,
) -> Tuple[np.ndarray, float]:
    """Simulate an eclipsing binary light curve.

    Returns (primary-eclipse flux, secondary-eclipse depth).
    """
    F_target = 1.0
    F_comp = companion_fluxratio / (1.0 - companion_fluxratio) if companion_fluxratio > 0 else 0.0
    F_EB = EB_fluxratio / (1.0 - EB_fluxratio)

    k = R_EB_solar / R_s_solar
    if abs(k - 1.0) < 1e-6:
        k *= 0.999
    a_rs = a_cm / (R_s_solar * Rsun)

    # Primary eclipse
    flux = _batman_transit(
        time, k, a_rs, inc_deg, u1, u2, ecc, argp_deg,
        exptime_days, nsamples,
    )

    # Secondary eclipse
    sec_time = np.linspace(-0.05, 0.05, 25)
    sec_flux = _batman_transit(
        sec_time, 1.0 / k, a_rs, inc_deg, u1, u2, ecc, argp_deg + 180.0,
        exptime_days, nsamples=1,
    )
    sec_flux_min = float(np.min(sec_flux))

    # Dilution cascade
    if companion_is_host:
        if F_comp > 0:
            flux = (flux + F_EB / F_comp) / (1.0 + F_EB / F_comp)
            sec_diluted = (sec_flux_min + F_comp / F_EB) / (1.0 + F_comp / F_EB) if F_EB > 0 else sec_flux_min
            F_dilute = F_target / (F_comp + F_EB) if (F_comp + F_EB) > 0 else 0.0
        else:
            sec_diluted = sec_flux_min
            F_dilute = 0.0
        if F_dilute > 0:
            flux = (flux + F_dilute) / (1.0 + F_dilute)
            secdepth = 1.0 - (sec_diluted + F_dilute) / (1.0 + F_dilute)
        else:
            secdepth = 1.0 - sec_diluted
    else:
        flux = (flux + F_EB / F_target) / (1.0 + F_EB / F_target)
        sec_diluted = (sec_flux_min + F_target / F_EB) / (1.0 + F_target / F_EB) if F_EB > 0 else sec_flux_min
        F_dilute = F_comp / (F_target + F_EB) if companion_fluxratio > 0 else 0.0
        if F_dilute > 0:
            flux = (flux + F_dilute) / (1.0 + F_dilute)
            secdepth = 1.0 - (sec_diluted + F_dilute) / (1.0 + F_dilute)
        else:
            secdepth = 1.0 - sec_diluted

    return flux, secdepth


# ---------------------------------------------------------------------------
# Log-likelihood functions
# ---------------------------------------------------------------------------

def lnL_TP(
    time: np.ndarray, flux: np.ndarray, sigma: float,
    R_p_earth: float, P_orb: float, inc_deg: float, a_cm: float,
    R_s_solar: float, u1: float, u2: float, ecc: float = 0.0,
    argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0, companion_is_host: bool = False,
    exptime_days: float = 0.00139, nsamples: int = 20,
) -> float:
    """Log-likelihood for a transiting planet scenario.

    lnL = -0.5 * sum((flux_obs - flux_model)^2 / sigma^2)
    """
    model = simulate_TP(
        time, R_p_earth, P_orb, inc_deg, a_cm, R_s_solar, u1, u2,
        ecc, argp_deg, companion_fluxratio, companion_is_host,
        exptime_days, nsamples,
    )
    return float(-0.5 * np.sum((flux - model) ** 2 / sigma ** 2))


def lnL_EB(
    time: np.ndarray, flux: np.ndarray, sigma: float,
    R_EB_solar: float, EB_fluxratio: float, P_orb: float,
    inc_deg: float, a_cm: float, R_s_solar: float,
    u1: float, u2: float, ecc: float = 0.0, argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0, companion_is_host: bool = False,
    exptime_days: float = 0.00139, nsamples: int = 20,
) -> float:
    """Log-likelihood for an eclipsing binary scenario.

    Vetoes draws where secondary eclipse depth >= 1.5*sigma.
    """
    model, secdepth = simulate_EB(
        time, R_EB_solar, EB_fluxratio, P_orb, inc_deg, a_cm, R_s_solar,
        u1, u2, ecc, argp_deg, companion_fluxratio, companion_is_host,
        exptime_days, nsamples,
    )
    if secdepth >= 1.5 * sigma:
        return -np.inf
    return float(-0.5 * np.sum((flux - model) ** 2 / sigma ** 2))


__all__ = [
    "_batman_transit",
    "simulate_TP",
    "simulate_EB",
    "lnL_TP",
    "lnL_EB",
]