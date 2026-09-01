"""Vectorized Monte Carlo integration for the TREX statistical vetting engine.

Implements log-evidence estimation for all 18 astrophysical scenarios
defined in Giacalone et al. (2021, AJ, 161, 24).

Architecture:
    Each scenario-dedicated function performs the same pattern:
    1. Draw ``n_draws`` samples from astrophysical priors.
    2. Compute geometric transit probability (b, a_Rs, rp_rs).
    3. For transiting draws, evaluate the Mandel-Agol likelihood.
    4. Compute lnZ = log(mean(exp(lnL + lnprior))) via _log_mean_exp.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .constants import Msun, Rsun, Rearth, G, au, pi, SECONDS_PER_DAY
from ._numerics import _log_mean_exp
from .funcs import (
    stellar_relations,
    semi_major_axis_cgs,
    a_over_Rs,
    impact_parameter,
    delta_mag_to_flux_ratio,
    companion_flux_ratio,
)
from .priors import (
    sample_rp,
    sample_inc,
    sample_ecc,
    sample_w,
    sample_q,
    lnprior_bound,
    lnprior_background,
)
from .likelihoods import lnL_TP, lnL_EB


# ---------------------------------------------------------------------------
# Helper: geometric transit probability
# ---------------------------------------------------------------------------

def _geometric_transit_mask(
    b: np.ndarray, rp_rs: np.ndarray, ecc: np.ndarray, argp: np.ndarray, a_rs: np.ndarray,
) -> np.ndarray:
    """Boolean mask for draws that geometrically transit.

    Transit if b <= 1 + rp_rs, accounting for eccentric orbits.
    """
    factor = (1.0 - ecc ** 2) / (1.0 + ecc * np.sin(np.radians(argp)))
    # For circular: factor = 1
    return np.abs(b) <= (1.0 + rp_rs) * factor

# ---------------------------------------------------------------------------
# Scenario identifiers
# ---------------------------------------------------------------------------

SCENARIO_TP: int = 0
SCENARIO_EB: int = 1
SCENARIO_EBX2P: int = 2
SCENARIO_PTP: int = 3
SCENARIO_PEB: int = 4
SCENARIO_PEBX2P: int = 5
SCENARIO_STP: int = 6
SCENARIO_SEB: int = 7
SCENARIO_SEBX2P: int = 8
SCENARIO_DTP: int = 9
SCENARIO_DEB: int = 10
SCENARIO_DEBX2P: int = 11
SCENARIO_BTP: int = 12
SCENARIO_BEB: int = 13
SCENARIO_BEBX2P: int = 14
SCENARIO_NTP_OFFSET: int = 15
SCENARIO_NEB_OFFSET: int = 16
SCENARIO_NEBX2P_OFFSET: int = 17

_SCENARIO_NAMES: Dict[int, str] = {
    0: "TP", 1: "EB", 2: "EBx2P", 3: "PTP", 4: "PEB", 5: "PEBx2P",
    6: "STP", 7: "SEB", 8: "SEBx2P", 9: "DTP", 10: "DEB", 11: "DEBx2P",
    12: "BTP", 13: "BEB", 14: "BEBx2P",
}

N_TARGET_SCENARIOS: int = 15
N_NEIGHBOR_SCENARIOS: int = 3


# ---------------------------------------------------------------------------
# Unified single-scenario evidence estimator
# ---------------------------------------------------------------------------

def _eval_scenario(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    period_days: float,
    M_s_Msun: float,
    R_s_Rsun: float,
    u1: float,
    u2: float,
    n_draws: int,
    is_planet: bool,
    is_EB: bool,
    use_2xP: bool,
    companion_is_host: bool = False,
    companion_fluxratio: float = 0.0,
    lnprior_comp: float = 0.0,
    use_flat_priors: bool = False,
    M_s_neighbor: Optional[float] = None,
    R_s_neighbor: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    exptime_days: float = 0.00139,
    nsamples: int = 20,
) -> float:
    """Estimate lnZ for one astrophysical scenario via Monte Carlo integration.

    Returns:
        Log-evidence lnZ (float; -inf if no draws transit).
    """
    if rng is None:
        rng = np.random.default_rng()

    eff_period = period_days * 2.0 if use_2xP else period_days
    if use_2xP and eff_period > 100.0:
        return -np.inf

    # Use neighbor properties if provided
    M_host = M_s_neighbor if M_s_neighbor is not None else M_s_Msun
    R_host = R_s_neighbor if R_s_neighbor is not None else R_s_Rsun

    # Draw prior samples
    x_rp = rng.random(n_draws)
    x_inc = rng.random(n_draws)
    x_ecc = rng.random(n_draws)
    x_w = rng.random(n_draws)

    M_host_arr = np.full(n_draws, M_host, dtype=float)
    R_p = sample_rp(x_rp, M_host_arr, flatpriors=use_flat_priors)
    inc = sample_inc(x_inc)
    ecc = sample_ecc(x_ecc, planet=is_planet, P_orb=eff_period)
    w = sample_w(x_w)

    # Semi-major axis
    if is_EB and not is_planet:
        x_q = rng.random(n_draws)
        q = sample_q(x_q, M_host)
        M_EB = q * M_host
        M_total = (M_host + M_EB) * Msun
        R_EB = q * R_host
        # Flux ratio from radii
        EB_fluxratio_arr = R_EB ** 2 / (R_host ** 2 + R_EB ** 2)
    else:
        M_total = M_host * Msun
        R_EB = None
        EB_fluxratio_arr = None

    a_cm = semi_major_axis_cgs(eff_period, M_total)
    a_rs = a_cm / (R_host * Rsun)
    rp_rs = R_p * Rearth / (R_host * Rsun)

    b = np.abs(a_rs * np.cos(np.radians(inc)))
    transit_mask = _geometric_transit_mask(b, rp_rs, ecc, w, a_rs)

    lnL = np.full(n_draws, -np.inf)

    if np.any(transit_mask):
        idx = np.where(transit_mask)[0]
        if is_EB:
            for i in idx:
                a_cm_val = float(a_cm[i]) if isinstance(a_cm, np.ndarray) else float(a_cm)
                lnL[i] = lnL_EB(
                    time, flux, sigma,
                    float(R_EB[i]) if R_EB is not None else float(R_host * 0.3),
                    float(EB_fluxratio_arr[i]) if EB_fluxratio_arr is not None else 0.5,
                    eff_period, float(inc[i]), a_cm_val, R_host,
                    u1, u2, float(ecc[i]), float(w[i]),
                    companion_fluxratio, companion_is_host,
                    exptime_days, nsamples,
                )
        else:
            for i in idx:
                a_cm_val = float(a_cm[i]) if isinstance(a_cm, np.ndarray) else float(a_cm)
                lnL[i] = lnL_TP(
                    time, flux, sigma,
                    float(R_p[i]), eff_period, float(inc[i]), a_cm_val,
                    R_host, u1, u2, float(ecc[i]), float(w[i]),
                    companion_fluxratio, companion_is_host,
                    exptime_days, nsamples,
                )

    # Evidence = log(mean(exp(lnL))) + lnprior_comp for companion scenarios
    lnZ = _log_mean_exp(lnL, N_total=n_draws)
    if math.isfinite(lnprior_comp):
        lnZ += lnprior_comp

    return float(lnZ)


# ---------------------------------------------------------------------------
# Top-level evidence calculation across all scenarios
# ---------------------------------------------------------------------------


def calc_target_evidences(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    period_days: float,
    depth_ppm: float,
    M_s_Msun: float,
    R_s_Rsun: float,
    u1: float,
    u2: float,
    n_draws: int = 2000,
    contrast_separations: Optional[np.ndarray] = None,
    contrast_values: Optional[np.ndarray] = None,
    companion_delta_mags: Optional[np.ndarray] = None,
    N_comp_bg: int = 0,
    plx: float = 0.0,
    nearby_stars: Optional[list] = None,
    random_seed: Optional[int] = None,
    progress_callback: Optional[callable] = None,
    exptime_days: float = 0.00139,
    nsamples: int = 20,
) -> Tuple[np.ndarray, str]:
    """Compute log-evidences for all target-star and neighbour scenarios.

    Returns:
        (lnZ, status): lnZ array of shape (n_scenarios,) and status string.
    """
    rng = np.random.default_rng(random_seed)
    n_scenarios = N_TARGET_SCENARIOS
    if nearby_stars:
        n_scenarios += len(nearby_stars) * N_NEIGHBOR_SCENARIOS

    lnZ = np.full(n_scenarios, -np.inf)

    # Companion prior setup
    delta_m = companion_delta_mags[0] if (
        companion_delta_mags is not None and len(companion_delta_mags) > 0
    ) else 0.0
    comp_frac = float(delta_mag_to_flux_ratio(np.array([delta_m]))[0])

    lnp_bound = -np.inf
    if (companion_delta_mags is not None and len(companion_delta_mags) > 0
            and contrast_separations is not None):
        lnp_arr = lnprior_bound(
            M_s_Msun, companion_delta_mags,
            contrast_separations,
            contrast_values if contrast_values is not None else np.array([]),
            plx,
        )
        lnp_bound = float(lnp_arr[0]) if len(lnp_arr) > 0 else -np.inf

    lnp_bg = -np.inf
    if (N_comp_bg > 0 and companion_delta_mags is not None
            and len(companion_delta_mags) > 0 and contrast_separations is not None):
        bg_dm = companion_delta_mags[0]
        lnp_bg_arr = lnprior_background(
            N_comp_bg, np.array([bg_dm]),
            contrast_separations,
            contrast_values if contrast_values is not None else np.array([]),
        )
        lnp_bg = float(lnp_bg_arr[0]) if len(lnp_bg_arr) > 0 else -np.inf

    # (is_planet, is_EB, use_2xP, comp_is_host, comp_frac, flat, lnprior_comp)
    specs = [
        (True, False, False, False, 0.0, False, 0.0),     # TP
        (False, True, False, False, 0.0, False, 0.0),     # EB
        (False, True, True, False, 0.0, False, 0.0),      # EBx2P
        (True, False, False, False, comp_frac, False, lnp_bound),   # PTP
        (False, True, False, False, comp_frac, False, lnp_bound),   # PEB
        (False, True, True, False, comp_frac, False, lnp_bound),    # PEBx2P
        (True, False, False, True, comp_frac, False, lnp_bound),    # STP
        (False, True, False, True, comp_frac, False, lnp_bound),    # SEB
        (False, True, True, True, comp_frac, False, lnp_bound),     # SEBx2P
        (True, False, False, True, 0.0, True, 0.0),       # DTP
        (False, True, False, True, 0.0, True, 0.0),       # DEB
        (False, True, True, True, 0.0, True, 0.0),        # DEBx2P
        (True, False, False, False, 0.0, True, lnp_bg),   # BTP
        (False, True, False, False, 0.0, True, lnp_bg),   # BEB
        (False, True, True, False, 0.0, True, lnp_bg),    # BEBx2P
    ]

    for idx, (is_pl, is_eb, x2p, cih, cfr, flat, lnp_c) in enumerate(specs):
        if progress_callback:
            name = _SCENARIO_NAMES.get(idx, str(idx))
            progress_callback(f"Scenario {name}", done=idx, total=n_scenarios)
        if not math.isfinite(lnp_c) and lnp_c < -1e300:
            continue
        lnZ[idx] = _eval_scenario(
            time, flux, sigma, period_days, M_s_Msun, R_s_Rsun,
            u1, u2, n_draws, is_pl, is_eb, x2p,
            companion_is_host=cih, companion_fluxratio=cfr,
            lnprior_comp=lnp_c, use_flat_priors=flat,
            rng=rng, exptime_days=exptime_days, nsamples=nsamples,
        )

    # Neighbour scenarios
    if nearby_stars:
        for ni, neighbor in enumerate(nearby_stars):
            M_n = float(neighbor.get("M_s", M_s_Msun))
            R_n = float(neighbor.get("R_s", R_s_Rsun))
            base = N_TARGET_SCENARIOS + ni * N_NEIGHBOR_SCENARIOS
            for local in range(N_NEIGHBOR_SCENARIOS):
                is_pl = (local == 0)
                is_eb = not is_pl
                x2p = (local == 2)
                si = base + local
                if progress_callback:
                    progress_callback(f"Neighbor {ni}", done=si, total=n_scenarios)
                lnZ[si] = _eval_scenario(
                    time, flux, sigma, period_days, M_s_Msun, R_s_Rsun,
                    u1, u2, n_draws, is_pl, is_eb, x2p,
                    use_flat_priors=True,
                    M_s_neighbor=M_n, R_s_neighbor=R_n,
                    rng=rng, exptime_days=exptime_days, nsamples=nsamples,
                )

    return lnZ, "ok"


def compute_fpp_nfpp(lnZ: np.ndarray) -> Tuple[float, float, np.ndarray, str]:
    """Compute FPP and NFPP from scenario log-evidences.

    Args:
        lnZ: Log-evidences per scenario (index order above).

    Returns:
        (FPP, NFPP, probs, status): FPP, NFPP, probability vector, status.
    """
    from ._numerics import _normalize_probabilities

    probs, norm_status = _normalize_probabilities(lnZ)

    if norm_status != "ok":
        fpp = 1.0 if norm_status == "all_neginf" else float("nan")
        return fpp, float("nan"), probs, norm_status

    # FPP = 1 - (P_TP + P_PTP + P_DTP)
    n_total = len(probs)
    if n_total > SCENARIO_DTP:
        fpp = 1.0 - (probs[SCENARIO_TP] + probs[SCENARIO_PTP] + probs[SCENARIO_DTP])
    elif n_total > SCENARIO_TP:
        fpp = 1.0 - probs[SCENARIO_TP]
    else:
        fpp = float("nan")
    # NFPP = sum of all neighbour-scenario probabilities
    nfpp = np.sum(probs[N_TARGET_SCENARIOS:]) if n_total > N_TARGET_SCENARIOS else 0.0

    return float(fpp), float(nfpp), probs, norm_status


__all__ = [
    "SCENARIO_TP",
    "SCENARIO_EB",
    "SCENARIO_EBX2P",
    "SCENARIO_PTP",
    "SCENARIO_PEB",
    "SCENARIO_PEBX2P",
    "SCENARIO_STP",
    "SCENARIO_SEB",
    "SCENARIO_SEBX2P",
    "SCENARIO_DTP",
    "SCENARIO_DEB",
    "SCENARIO_DEBX2P",
    "SCENARIO_BTP",
    "SCENARIO_BEB",
    "SCENARIO_BEBX2P",
    "N_TARGET_SCENARIOS",
    "N_NEIGHBOR_SCENARIOS",
    "calc_target_evidences",
    "compute_fpp_nfpp",
    "_eval_scenario",
    "_geometric_transit_mask",
]