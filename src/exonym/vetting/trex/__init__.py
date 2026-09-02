"""TREX -- Native TRiceratops EXonym statistical validation engine.

A native, modernized, vectorized implementation of the Bayesian false-positive
probability framework from Giacalone et al. (2021, AJ, 161, 24).  TREX replaces
the direct runtime dependency on the third-party ``triceratops`` package while
preserving the identical scientific algorithm.

Public API
----------
- ``run_trex_vetting(scene, time, flux, sigma, period_days, ...)``
  Run a complete Monte Carlo false-positive probability calculation
  and return a ``TrexResult``.
- ``TargetScene``
  Stellar population context for one TESS target.
- ``TrexResult``
  Container for FPP, NFPP, scenario probabilities, and diagnostics.

Scientific Guardrail
--------------------
All results carry ``claim_eligible=False``.  The module is a statistical
diagnostic engine only; a discovery claim requires provenance-bound observed
photometry and calibrated scene-model integration.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

import numpy as np

from .target import TargetScene
from .diagnostics import TrexResult, generate_diagnostics
from .marginal_likelihoods import calc_target_evidences
from .licensing import __all__ as _lic_all  # noqa: F401 – ensure copyright visible


def run_trex_vetting(
    scene: TargetScene,
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    period_days: float,
    depth_ppm: float,
    exptime_days: float,
    u1: float = 0.4,
    u2: float = 0.2,
    n_draws: int = 2000,
    random_seed: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    nsamples: int = 20,
) -> TrexResult:
    """Run a complete TREX false-positive probability calculation.

    Args:
        scene: TargetScene with stellar and neighbour properties.
        time: Phase-folded times [days from transit midpoint].
        flux: Normalised flux.
        sigma: Flux uncertainty.
        period_days: Orbital period [days].
        depth_ppm: Transit depth [ppm].
        u1, u2: Quadratic limb-darkening coefficients.
        n_draws: Monte Carlo draws per scenario (default 2000).
        random_seed: RNG seed for reproducibility.
        progress_callback: Optional callable(step, done, total).
        exptime_days: TESS exposure time [days].
        nsamples: Supersampling factor.

    Returns:
        TrexResult with FPP, NFPP, per-scenario probabilities, and diagnostics.
    """
    if not isinstance(exptime_days, (int, float, np.number)) or not math.isfinite(exptime_days) or exptime_days <= 0.0:
        raise ValueError("exptime_days must be finite and positive")
    if not isinstance(scene, TargetScene):
        raise TypeError("scene must be a validated TargetScene")
    scene.verify_background()

    # Build companion data
    companion_delta_mags = scene.companion_delta_mags()
    nearby_stars = scene.neighbor_dicts_for_evidence()

    lnZ, _ = calc_target_evidences(
        time=time,
        flux=flux,
        sigma=sigma,
        period_days=period_days,
        depth_ppm=depth_ppm,
        M_s_Msun=scene.M_s_Msun,
        R_s_Rsun=scene.R_s_Rsun,
        u1=u1,
        u2=u2,
        exptime_days=exptime_days,
        n_draws=n_draws,
        contrast_separations=scene.contrast_separations,
        contrast_values=scene.contrast_values,
        companion_delta_mags=companion_delta_mags if len(companion_delta_mags) > 0 else None,
        N_comp_bg=scene.N_background,
        plx=scene.plx_mas,
        nearby_stars=nearby_stars if nearby_stars else None,
        random_seed=random_seed,
        progress_callback=progress_callback,
        nsamples=nsamples,
    )

    result = generate_diagnostics(lnZ, verbose=True)
    return result


__all__ = [
    "run_trex_vetting",
    "TargetScene",
    "TrexResult",
]
