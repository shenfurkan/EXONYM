"""Target-neutral MCMC transit light curve fitter.

Mathematical Formulation
------------------------
The transit flux is evaluated with batman (Mandel & Agol 2002), which computes
an analytic light curve for a quadratic limb-darkened star occulted by an opaque
spherical planet. The specific intensity profile is

.. math::

    I(\\mu) / I(1) = 1 - u_1 (1 - \\mu) - u_2 (1 - \\mu)^2

where :math:`\\mu = \\cos\\theta` and :math:`\\theta` is the angle between the
line of sight and the local surface normal (Claret 2000).  The free limb-darkening
parameters are sampled in Kipping (2013) coordinates

.. math::

    q_1 = (u_1 + u_2)^2, \\qquad
    q_2 = \\frac{u_1}{2(u_1 + u_2)}

with the inverse transformation

.. math::

    u_1 = 2\\sqrt{q_1}\\,q_2, \\qquad
    u_2 = \\sqrt{q_1}\\,(1 - 2q_2).

The scaled semi-major axis ``a / R_*`` is derived from the mean stellar density
via Kepler's third law (Seager & Mallen-Ornelas 2003; Sozzetti et al. 2007):

.. math::

    (a / R_*)^3 = \\frac{G \\, P^2 \\, \\rho_*}{3\\pi}

where :math:`P` is the orbital period in seconds and :math:`\\rho_*` is the
mean stellar density in cgs.  This density-locking strategy reduces the
degeneracy between ``a / R_*`` and ``R_p / R_*``.

Eccentric orbits are parameterized via :math:`(\\sqrt{e}\\cos\\omega,\\;
\\sqrt{e}\\sin\\omega)` (Eastman et al. 2013) to avoid the coordinate
singularity at :math:`e = 0`.  For circular orbits, :math:`e = 0` and
:math:`\\omega = 90^\\circ` are fixed.

The log-likelihood assumes independent Gaussian photometric errors with a
fitted additive jitter term (Foreman-Mackey et al. 2013):

.. math::

    \\ln L = -\\frac{1}{2} \\sum_{i=1}^N \\left[
        \\frac{(f_i - f_{\\rm model}(t_i))^2}{\\sigma_i^2 + \\sigma_j^2}
        + \\ln\\bigl(2\\pi(\\sigma_i^2 + \\sigma_j^2)\\bigr)
    \\right]

where :math:`\\sigma_j = e^{\\ln\\sigma_j}` is the jitter fitted in
natural-log space to enforce positivity.

Posterior sampling is performed with CUDA NUTS via ``NumPyro`` when its
optional JAX stack is available, the Goodman & Weare (2010) affine-invariant
ensemble sampler via ``emcee`` otherwise, or dynamic nested sampling via
``dynesty`` (Speagle 2020) when explicitly selected. The latter reports the
log evidence :math:`\\ln Z` and its estimated numerical uncertainty; these are
descriptive and are **not** validation probabilities.

Astrophysical Rationale
-----------------------
- The density prior anchors ``a / R_*`` to the independently measured stellar
  parameters, preventing the fit from floating into unphysical regions of
  parameter space where a grazing eclipse of a giant star mimics a planet.
- Kipping coordinates ensure uniform sampling over the physically valid
  region of the quadratic limb-darkening triangle.
- Eccentricity is optional so that circular-orbit fits can serve as a
  sensitivity baseline before incurring the extra degrees of freedom.

Contains no target constants or hardcoded candidate parameters; all stellar priors,
ephemerides, and photometric time-series are loaded dynamically from the candidate workspace.

References
----------
- Mandel & Agol, *ApJL* 580, L171 (2002)
- Kipping, *MNRAS* 435, 2152 (2013)
- Seager & Mallen-Ornelas, *ApJ* 585, 1038 (2003)
- Sozzetti et al., *ApJ* 664, 1190 (2007)
- Eastman et al., *PASP* 125, 83 (2013)
- Goodman & Weare, *Comm. Appl. Math. Comput. Sci.* 5, 65 (2010)
- Speagle, *MNRAS* 493, 3132 (2020)
- Foreman-Mackey et al., *PASP* 125, 306 (2013)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import multiprocessing
import os

import numpy as np

from .constants import (
    GRAVITATIONAL_CONSTANT_CGS,
    SECONDS_PER_DAY,
    SOLAR_MEAN_DENSITY_G_CM3,
)
from .inputs import (
    load_light_curve_table,
    load_stellar_parameters,
    load_transit_ephemeris,
)
from .lightcurve import bin_phase_folded_flux, kipping_to_quadratic_limb_darkening
from .workspace import CandidateWorkspace, validate_signal_suffix

# ASTROPHYSICAL_HEURISTIC: phase-folded crop window twice the nominal TESS
# transit-duration envelope; ensures out-of-transit baseline is captured for
# normalization while keeping far-out-of-eclipse noise from biasing the fit.
WINDOW_HALF_HOURS = 13.0    # Folded light curve crop window half-width (hours)
# ASTROPHYSICAL_HEURISTIC: 8-minute bins for median-binned phase-folded
# visualization only; the likelihood itself operates on native cadence.
BIN_MINUTES = 8.0           # Default phase-binning resolution (minutes)
# NUMERICAL_GUARD: supersampling factor 7 is sufficient for the batman
# analytic transit model with quadratic limb darkening at TESS cadence
# (Kipping 2010 recommends >= 5 for precision better than 10 ppm).
SUPERSAMPLE_FACTOR = 7      # Numerical exposure integration sub-sampling factor
EXPTIME_SECONDS = 120.0     # Nominal TESS 2-minute SPOC cadence (seconds)
# SCIENTIFIC_BOUNDARY: the folded/binned display uses a coarser effective
# integration time; this is not the native-cadence posterior exposure.
FITTED_BIN_EXPOSURE_SECONDS = BIN_MINUTES * 60.0

# Parameter vectors for circular and eccentric orbits
PARAMETER_NAMES_CIRCULAR = (
    "rp_rs",             # Planet-to-star radius ratio (R_p / R_star)
    "log_rho_star",      # log10(rho_star / rho_sun), stellar density proxy
    "impact_parameter",  # Transit impact parameter at inferior conjunction
    "baseline",          # Out-of-transit flux normalization baseline
    "log_jitter",        # natural log(sigma_jitter), added in quadrature to flux uncertainties
    "q1",                # Kipping (2013) limb-darkening parameter 1
    "q2",                # Kipping (2013) limb-darkening parameter 2
)
PARAMETER_NAMES_ECCENTRIC = PARAMETER_NAMES_CIRCULAR + (
    "sqe_cosw",          # sqrt(e) * cos(omega), Lagrangian eccentricity component
    "sqe_sinw",          # sqrt(e) * sin(omega), Lagrangian eccentricity component
)

# A data-independent weakly informative prior on the additive normalized-flux
# jitter. The sampler keeps the same finite support as the historical model,
# but its density is no longer centred on the light curve being fitted.
JITTER_LOG_LOWER = -12.0
JITTER_LOG_UPPER = -2.0
JITTER_HALF_CAUCHY_SCALE = 1.0e-3

# GPU NUTS defaults. These are intentionally independent from the historical
# candidate-workspace emcee arguments so an automatic accelerator selection does
# not silently shorten either sampler's production run.
GPU_NUTS_WARMUP = 1000
GPU_NUTS_SAMPLES = 2000
GPU_NUTS_TARGET_ACCEPT_PROB = 0.9
CPU_EMCEE_WALKERS = 50
CPU_EMCEE_BURN_IN = 1000
CPU_EMCEE_PRODUCTION = 2500


class _GpuBackendUnavailable(RuntimeError):
    """Raised when the optional JAX/NumPyro CUDA stack cannot be used."""


@dataclass(frozen=True)
class _AcceleratedTransitFitData:
    """Validated normalized-flux inputs shared by the accelerator backends."""

    time_days: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    period_days: float
    t0_days: float
    rho_star_g_cm3: float
    rho_star_sigma_g_cm3: float
    period_sigma_days: Optional[float]
    t0_sigma_days: Optional[float]
    exposure_seconds: float
    eccentric: bool


# --- Parallel worker context (module-level globals set once per worker) ---
# These hold the fixed data arrays that every lnprob evaluation needs.
# Setting them via Pool(initializer=...) avoids repeated pickle overhead
# (emcee documentation confirms args/kwargs can cause 3x slowdown).

_worker_context: Optional[Dict[str, Any]] = None


def _init_worker(context: Dict[str, Any]) -> None:
    """Initialise one multiprocessing worker with the full data context."""
    global _worker_context
    _worker_context = context
    # Prevent NumPy MKL/OpenBLAS oversubscription inside workers.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


def _log_prob_worker(theta: np.ndarray) -> float:
    """Negative log-posterior wrapper callable by emcee's pool.map.

    The data arrays are read from the module-level ``_worker_context`` global,
    set once per worker by ``_init_worker``.
    """
    ctx = _worker_context
    if ctx is None:
        raise RuntimeError("_log_prob_worker called without _init_worker context")

    return float(
        -_neg_log_posterior(
            theta,
            ctx["phase_days"],
            ctx["native_flux"],
            ctx["native_error"],
            ctx["ephemeris"],
            ctx["rho_prior_solar"],
            ctx["rho_prior_log10_sigma"],
            ctx["eccentric"],
            ctx["ldtk_prior"],
            sector_index=ctx["sector_index"],
            exposure_seconds_by_sector=ctx["exposure_seconds_by_sector"],
            n_sectors=ctx["n_sectors"],
        )
    )


def stellar_density_a_rs(rho_solar: float, period_days: float) -> float:
    r"""Calculate scaled semimajor axis (a / R_star) from mean stellar density.

    Mathematical Formulation
    ------------------------
    From Kepler's Third Law (Seager & Mallen-Ornelas 2003, eq. 4):

    .. math::

        \\left(\\frac{a}{R_*}\\right)^3
        = \\frac{G \\, P^2 \\, \\rho_*}{3\\pi}

    where :math:`G` is the CODATA gravitational constant, :math:`P` is the
    orbital period in seconds, and :math:`\\rho_*` is the mean stellar density
    in g cm\ :sup:`-3`. The solar-density normalization is reproducibly
    derived from IAU nominal solar constants and CODATA G.

    Astrophysical Rationale
    -----------------------
    This relation removes the strong degeneracy between ``R_p / R_*`` and
    ``a / R_*`` that exists when both are floated freely (Sozzetti et al.
    2007).  By locking ``a / R_*`` to the independently constrained stellar
    density, the transit shape information is channelled into the radius
    ratio and impact parameter.

    Parameters
    ----------
    rho_solar : float
        Mean stellar density in solar units (:math:`\\rho_* / \\rho_\\odot`).
        Must be > 0.
    period_days : float
        Orbital period in days.  Must be > 0.

    Returns
    -------
    float
        Dimensionless scaled semi-major axis ``a / R_*``.

    Raises
    ------
    ValueError
        If either argument is non-finite or non-positive, or the calculation
        cannot produce a finite scaled semi-major axis.
    """
    try:
        rho_solar = float(rho_solar)
        period_days = float(period_days)
    except (TypeError, ValueError) as exc:
        raise ValueError("stellar density and period must be finite and positive") from exc
    if (
        not math.isfinite(rho_solar)
        or not math.isfinite(period_days)
        or rho_solar <= 0.0
        or period_days <= 0.0
    ):
        raise ValueError("stellar density and period must be finite and positive")
    # NUMERICAL_GUARD: convert to CGS (period_days * seconds/day) before
    # evaluating the cube root; the exponent 1/3 is exact.
    rho_gcm3 = rho_solar * SOLAR_MEAN_DENSITY_G_CM3
    period_seconds = period_days * SECONDS_PER_DAY
    a_rs_cubed = (
        (GRAVITATIONAL_CONSTANT_CGS * period_seconds**2 * rho_gcm3) / (3.0 * math.pi)
    )
    if not math.isfinite(a_rs_cubed) or a_rs_cubed <= 0.0:
        raise ValueError("stellar density and period produce a non-finite scaled semi-major axis")
    a_rs = a_rs_cubed ** (1.0 / 3.0)
    if not math.isfinite(a_rs) or a_rs <= 0.0:
        raise ValueError("stellar density and period produce a non-finite scaled semi-major axis")
    return a_rs


def conjunction_distance_a_rs(a_rs: float, eccentricity: float, omega_deg: float) -> Optional[float]:
    """Return planet-star separation at inferior conjunction in stellar radii.

    Mathematical Formulation
    ------------------------
    For a Keplerian orbit with semi-major axis :math:`a`, eccentricity
    :math:`e`, and argument of periastron :math:`\\omega`, the star-planet
    separation at true anomaly :math:`f` is

    .. math::

        r(f) = \\frac{a (1 - e^2)}{1 + e \\cos f}.

    At inferior conjunction (primary transit), :math:`f = \\pi/2 - \\omega`,
    giving (Winn 2010, eq. 20):

    .. math::

        r_{\\rm conj} = \\frac{a (1 - e^2)}{1 + e \\sin \\omega}.

    The returned value is :math:`r_{\\rm conj} / R_*`.

    Parameters
    ----------
    a_rs : float
        Dimensionless semi-major axis ``a / R_*``.  Must be > 0.
    eccentricity : float
        Orbital eccentricity, :math:`0 \\le e < 1`.
    omega_deg : float
        Argument of periastron :math:`\\omega` in degrees.

    Returns
    -------
    float or None
        Conjunction separation in stellar radii, or None if inputs are
        non-finite, non-physical, or the denominator is non-positive.
    """
    # NUMERICAL_GUARD: explicit finiteness and domain checks before division.
    if (
        not math.isfinite(a_rs)
        or not math.isfinite(eccentricity)
        or not math.isfinite(omega_deg)
        or a_rs <= 0
        or not 0.0 <= eccentricity < 1.0
    ):
        return None
    denominator = 1.0 + eccentricity * math.sin(math.radians(omega_deg))
    # NUMERICAL_GUARD: protect against zero or negative denominator which
    # would correspond to a non-physical conjunction geometry.
    if denominator <= 0:
        return None
    return a_rs * (1.0 - eccentricity**2) / denominator


def inclination_deg_from_impact_parameter(
    a_rs: float, impact_parameter: float, eccentricity: float = 0.0, omega_deg: float = 90.0
) -> Optional[float]:
    """Convert conjunction impact parameter to orbital inclination.

    Mathematical Formulation
    ------------------------
    The transit impact parameter :math:`b` is defined as the projected
    sky-plane separation at inferior conjunction in units of :math:`R_*`:

    .. math::

        b = \\frac{r_{\\rm conj}}{R_*} \\cos i,

    where :math:`r_{\\rm conj}` is the orbital separation at conjunction
    (see :func:`conjunction_distance_a_rs`).  Inverting,

    .. math::

        i = \\arccos\\!\\left(\\frac{b}{r_{\\rm conj} / R_*}\\right).

    For circular orbits (:math:`e = 0`), :math:`r_{\\rm conj} / R_* = a / R_*`
    and :math:`b = (a/R_*) \\cos i`.

    Parameters
    ----------
    a_rs : float
        Dimensionless semi-major axis ``a / R_*``.
    impact_parameter : float
        Transit impact parameter :math:`b`, :math:`0 \\le b`.
    eccentricity : float, optional
        Orbital eccentricity (default 0).
    omega_deg : float, optional
        Argument of periastron in degrees (default 90).

    Returns
    -------
    float or None
        Orbital inclination in degrees, or None if the geometry is
        unphysical (e.g., :math:`b` exceeds the conjunction separation).
    """
    conjunction_distance = conjunction_distance_a_rs(a_rs, eccentricity, omega_deg)
    if conjunction_distance is None or not math.isfinite(impact_parameter) or impact_parameter < 0:
        return None
    # NUMERICAL_GUARD: cos(i) > 1 corresponds to non-transiting geometry;
    # the upper edge case cos(i) = 0 (i = 90 deg) is allowed.
    cosine = impact_parameter / conjunction_distance
    if not 0.0 <= cosine < 1.0:
        return None
    return math.degrees(math.acos(cosine))


def _require_batman() -> Any:
    """Import the required batman forward-model dependency."""
    try:
        import batman
    except ImportError as exc:
        raise RuntimeError(
            "batman-package is required for transit fitting; install the pinned core dependency."
        ) from exc
    return batman


def fit_transit_light_curve(
    time_days: Sequence[float],
    flux: Sequence[float],
    flux_err: Sequence[float],
    period_days: float,
    t0_days: float,
    rho_star_g_cm3: float,
    rho_star_sigma_g_cm3: float,
    *,
    device: str = "auto",
    eccentric: bool = False,
    period_sigma_days: Optional[float] = None,
    t0_sigma_days: Optional[float] = None,
    exposure_seconds: float = EXPTIME_SECONDS,
    num_warmup: int = GPU_NUTS_WARMUP,
    num_samples: int = GPU_NUTS_SAMPLES,
    target_accept_prob: float = GPU_NUTS_TARGET_ACCEPT_PROB,
    n_walkers: int = CPU_EMCEE_WALKERS,
    burn_in: int = CPU_EMCEE_BURN_IN,
    production: int = CPU_EMCEE_PRODUCTION,
    seed: int = 5,
    progress: bool = False,
) -> Dict[str, Any]:
    """Fit a normalized transit light curve on CUDA when available.

    ``time_days`` and ``t0_days`` must use the same declared time scale. Flux
    values must be normalized relative flux and ``rho_star_g_cm3`` must be in
    cgs units. Supplying a positive period or epoch uncertainty samples that
    quantity; omitting it keeps the supplied ephemeris fixed while still
    reporting it in the common posterior schema.

    The returned dictionary always has ``p16``, ``median``, and ``p84`` for
    ``rp_rstar``, ``a_rstar``, ``period``, ``t0``, ``inclination_deg``,
    ``q1``, ``q2``, ``u1``, ``u2``, and ``jitter_ppm``. The accelerator path
    is optional: unavailable or incompatible JAX/CUDA dependencies fall back
    to the CPU emcee implementation without importing JAX at module import
    time. A runtime sampling failure on an available GPU is intentionally not
    converted into a CPU fit because that would hide a scientific failure.
    """
    data = _validate_accelerated_transit_fit_data(
        time_days,
        flux,
        flux_err,
        period_days,
        t0_days,
        rho_star_g_cm3,
        rho_star_sigma_g_cm3,
        period_sigma_days=period_sigma_days,
        t0_sigma_days=t0_sigma_days,
        exposure_seconds=exposure_seconds,
        eccentric=eccentric,
    )
    _validate_accelerated_sampler_arguments(
        device=device,
        num_warmup=num_warmup,
        num_samples=num_samples,
        target_accept_prob=target_accept_prob,
        n_walkers=n_walkers,
        burn_in=burn_in,
        production=production,
        seed=seed,
    )

    if device == "cpu":
        return _fit_emcee_cpu(
            data,
            n_walkers=n_walkers,
            burn_in=burn_in,
            production=production,
            seed=seed,
            progress=progress,
            fallback_reason="CPU requested explicitly",
        )

    try:
        stack = _load_jax_gpu_stack()
    except _GpuBackendUnavailable as exc:
        return _fit_emcee_cpu(
            data,
            n_walkers=n_walkers,
            burn_in=burn_in,
            production=production,
            seed=seed,
            progress=progress,
            fallback_reason=str(exc),
        )

    return _fit_numpyro_gpu(
        data,
        stack=stack,
        num_warmup=num_warmup,
        num_samples=num_samples,
        target_accept_prob=target_accept_prob,
        seed=seed,
        progress=progress,
    )


def _validate_accelerated_transit_fit_data(
    time_days: Sequence[float],
    flux: Sequence[float],
    flux_err: Sequence[float],
    period_days: float,
    t0_days: float,
    rho_star_g_cm3: float,
    rho_star_sigma_g_cm3: float,
    *,
    period_sigma_days: Optional[float],
    t0_sigma_days: Optional[float],
    exposure_seconds: float,
    eccentric: bool,
) -> _AcceleratedTransitFitData:
    """Validate one normalized light curve before dispatching a sampler."""
    try:
        time = np.ascontiguousarray(np.asarray(time_days, dtype=np.float64))
        normalized_flux = np.ascontiguousarray(np.asarray(flux, dtype=np.float64))
        uncertainty = np.ascontiguousarray(np.asarray(flux_err, dtype=np.float64))
        period = float(period_days)
        epoch = float(t0_days)
        density = float(rho_star_g_cm3)
        density_sigma = float(rho_star_sigma_g_cm3)
        integration_seconds = float(exposure_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("transit-fit inputs must be numeric finite arrays and scalars") from exc
    if (
        time.ndim != 1
        or normalized_flux.shape != time.shape
        or uncertainty.shape != time.shape
        or time.size < 20
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(normalized_flux))
        or not np.all(np.isfinite(uncertainty))
        or np.any(uncertainty <= 0.0)
    ):
        raise ValueError("transit fitting requires at least 20 finite cadences with positive uncertainties")
    if (
        not math.isfinite(period)
        or period <= 0.0
        or not math.isfinite(epoch)
        or not math.isfinite(density)
        or density <= 0.0
        or not math.isfinite(density_sigma)
        or density_sigma <= 0.0
        or not math.isfinite(integration_seconds)
        or integration_seconds <= 0.0
    ):
        raise ValueError("period, stellar-density prior, and exposure time must be finite and positive")

    def optional_positive(value: Optional[float], name: str) -> Optional[float]:
        if value is None:
            return None
        try:
            converted = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("{0} must be a positive finite number or None".format(name)) from exc
        if not math.isfinite(converted) or converted <= 0.0:
            raise ValueError("{0} must be a positive finite number or None".format(name))
        return converted

    if not isinstance(eccentric, (bool, np.bool_)):
        raise ValueError("eccentric must be a boolean")
    return _AcceleratedTransitFitData(
        time_days=time,
        flux=normalized_flux,
        flux_err=uncertainty,
        period_days=period,
        t0_days=epoch,
        rho_star_g_cm3=density,
        rho_star_sigma_g_cm3=density_sigma,
        period_sigma_days=optional_positive(period_sigma_days, "period_sigma_days"),
        t0_sigma_days=optional_positive(t0_sigma_days, "t0_sigma_days"),
        exposure_seconds=integration_seconds,
        eccentric=bool(eccentric),
    )


def _validate_accelerated_sampler_arguments(
    *,
    device: str,
    num_warmup: int,
    num_samples: int,
    target_accept_prob: float,
    n_walkers: int,
    burn_in: int,
    production: int,
    seed: int,
) -> None:
    """Validate sampler controls before any optional backend initialization."""
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError("device must be one of: auto, cpu, gpu")
    integer_values = {
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "n_walkers": n_walkers,
        "burn_in": burn_in,
        "production": production,
    }
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in integer_values.values()):
        raise ValueError("sampler iteration counts and n_walkers must be positive integers")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(target_accept_prob, (float, int)) or not math.isfinite(float(target_accept_prob)):
        raise ValueError("target_accept_prob must be a finite number")
    if not 0.0 < float(target_accept_prob) < 1.0:
        raise ValueError("target_accept_prob must lie strictly between zero and one")


def _load_jax_gpu_stack() -> Dict[str, Any]:
    """Load a 64-bit JAX/NumPyro CUDA stack or report why it is unavailable."""
    try:
        # This must run before importing JAX arrays or model modules. JAX uses
        # process-global configuration, so a caller that initialized JAX in
        # float32 cannot enter this precision-sensitive transit backend.
        from jax import config as jax_config

        jax_config.update("jax_enable_x64", True)
        import jax

        if not bool(jax_config.read("jax_enable_x64")):
            raise RuntimeError("JAX rejected the required 64-bit floating-point configuration")
        gpu_devices = jax.devices("gpu")
        if not gpu_devices:
            raise RuntimeError("JAX reports no CUDA-compatible GPU devices")
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        from jaxoplanet.orbits import TransitOrbit

        try:
            # Older jaxoplanet releases exposed this at package scope.
            from jaxoplanet.light_curves import limb_dark_light_curve
        except ImportError:
            # Current releases expose the same callable as light_curve.
            from jaxoplanet.light_curves.limb_dark import light_curve as limb_dark_light_curve
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
        raise _GpuBackendUnavailable(
            "JAX GPU backend unavailable: {0}: {1}".format(type(exc).__name__, exc)
        ) from exc
    return {
        "jax": jax,
        "jnp": jnp,
        "numpyro": numpyro,
        "dist": dist,
        "MCMC": MCMC,
        "NUTS": NUTS,
        "TransitOrbit": TransitOrbit,
        "limb_dark_light_curve": limb_dark_light_curve,
        "device": gpu_devices[0],
    }


def _accelerated_parameter_names(data: _AcceleratedTransitFitData) -> List[str]:
    """Return the CPU parameter vector layout for the standalone fitter."""
    names = [
        "rp_rstar",
        "log_rho_star",
        "impact_parameter",
        "baseline",
        "log_jitter",
        "q1",
        "q2",
    ]
    if data.eccentric:
        names.extend(["sqe_cosw", "sqe_sinw"])
    if data.period_sigma_days is not None:
        names.append("period")
    if data.t0_sigma_days is not None:
        names.append("t0")
    return names


def _accelerated_unpack_theta(
    theta: np.ndarray, data: _AcceleratedTransitFitData
) -> Dict[str, float]:
    """Map a CPU walker position to physical transit parameters."""
    names = _accelerated_parameter_names(data)
    values = np.asarray(theta, dtype=float)
    if values.shape != (len(names),):
        raise ValueError("accelerated transit parameter vector has an invalid shape")
    unpacked = {name: float(values[index]) for index, name in enumerate(names)}
    unpacked.setdefault("sqe_cosw", 0.0)
    unpacked.setdefault("sqe_sinw", 0.0)
    unpacked.setdefault("period", data.period_days)
    unpacked.setdefault("t0", data.t0_days)
    return unpacked


def _accelerated_a_rstar_from_cgs(
    rho_star_g_cm3: np.ndarray, period_days: np.ndarray
) -> np.ndarray:
    """Vectorize the density-locked scaled semi-major axis in cgs units."""
    density = np.asarray(rho_star_g_cm3, dtype=float)
    period = np.asarray(period_days, dtype=float)
    value = (
        GRAVITATIONAL_CONSTANT_CGS
        * np.square(period * SECONDS_PER_DAY)
        * density
        / (3.0 * math.pi)
    )
    return np.cbrt(value)


def _summarize_accelerated_samples(
    samples: Dict[str, np.ndarray],
    data: _AcceleratedTransitFitData,
    *,
    backend: str,
    sampler_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert backend-native draws into the public, backend-neutral schema."""
    required = ("rp_rstar", "log_rho_star", "impact_parameter", "log_jitter", "q1", "q2")
    if any(name not in samples for name in required):
        raise RuntimeError("transit sampler did not return all required posterior parameters")
    normalized = {name: np.asarray(value, dtype=float).reshape(-1) for name, value in samples.items()}
    sample_count = normalized["rp_rstar"].size
    if sample_count == 0 or any(values.size != sample_count for values in normalized.values()):
        raise RuntimeError("transit sampler returned posterior arrays with inconsistent shapes")
    if any(not np.all(np.isfinite(normalized[name])) for name in required):
        raise RuntimeError("transit sampler returned non-finite posterior values")

    # NUMERICAL_GUARD: bounded priors keep typical draws physical, but tail
    # draws (wide period priors, grazing b, e near unity) must degrade to the
    # same clipped-edge behaviour as the legacy posterior summaries instead of
    # aborting an otherwise converged fit.
    period = normalized.get("period", np.full(sample_count, data.period_days, dtype=float))
    t0 = normalized.get("t0", np.full(sample_count, data.t0_days, dtype=float))
    rho_star = normalized.get("rho_star_g_cm3", np.exp(normalized["log_rho_star"]))
    sqe_cosw = normalized.get("sqe_cosw", np.zeros(sample_count, dtype=float))
    sqe_sinw = normalized.get("sqe_sinw", np.zeros(sample_count, dtype=float))
    if not np.all(np.isfinite(period)) or not np.all(np.isfinite(t0)):
        raise RuntimeError("transit sampler returned non-finite ephemeris draws")
    safe_period = np.maximum(period, np.finfo(float).tiny)
    safe_rho_star = np.maximum(rho_star, np.finfo(float).tiny)
    eccentricity = np.clip(
        np.square(sqe_cosw) + np.square(sqe_sinw), 0.0, 1.0 - np.finfo(float).eps
    )
    q1 = normalized["q1"]
    q2 = normalized["q2"]
    if np.any(q1 < 0.0) or np.any(q1 > 1.0) or np.any(q2 < 0.0) or np.any(q2 > 1.0):
        raise RuntimeError("transit sampler returned Kipping coordinates outside [0, 1]")

    a_rstar = _accelerated_a_rstar_from_cgs(safe_rho_star, safe_period)
    sqrt_eccentricity = np.sqrt(eccentricity)
    denominator = np.maximum(1.0 + sqrt_eccentricity * sqe_sinw, np.finfo(float).eps)
    conjunction_distance = a_rstar * (1.0 - eccentricity) / denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine_inclination = np.where(
            conjunction_distance > 0.0,
            normalized["impact_parameter"] / conjunction_distance,
            np.nan,
        )
    finite_projection = np.isfinite(cosine_inclination)
    if not np.all(np.isfinite(a_rstar)) or not np.any(finite_projection):
        raise RuntimeError("transit sampler returned no summarizable inclination geometry")
    clipped_projection = finite_projection & (
        (cosine_inclination < 0.0) | (cosine_inclination > 1.0)
    )
    inclination_deg = np.degrees(
        np.arccos(np.clip(cosine_inclination, 0.0, 1.0))
    )
    sqrt_q1 = np.sqrt(q1)
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    jitter_ppm = np.exp(normalized["log_jitter"]) * 1.0e6
    summary_samples = {
        "rp_rstar": normalized["rp_rstar"],
        "a_rstar": a_rstar,
        "period": period,
        "t0": t0,
        "inclination_deg": inclination_deg,
        "q1": q1,
        "q2": q2,
        "u1": u1,
        "u2": u2,
        "jitter_ppm": jitter_ppm,
    }
    result: Dict[str, Any] = {"backend": backend}
    result.update({name: _quantile_summary(values) for name, values in summary_samples.items()})
    # Preserve the legacy grazing-geometry transparency contract.
    inclination_summary = dict(result["inclination_deg"])
    inclination_summary["conjunction_distance_clip_fraction"] = float(
        np.mean(clipped_projection)
    )
    result["inclination_deg"] = inclination_summary
    result["sampler_metadata"] = sampler_metadata
    return result


def _accelerated_cpu_log_probability(
    theta: np.ndarray, data: _AcceleratedTransitFitData
) -> float:
    """Evaluate the standalone emcee posterior without mutable shared state."""
    try:
        values = _accelerated_unpack_theta(theta, data)
    except ValueError:
        return -np.inf
    if not all(math.isfinite(value) for value in values.values()):
        return -np.inf
    rp_rstar = values["rp_rstar"]
    log_rho_star = values["log_rho_star"]
    impact_parameter = values["impact_parameter"]
    baseline = values["baseline"]
    log_jitter = values["log_jitter"]
    q1 = values["q1"]
    q2 = values["q2"]
    period = values["period"]
    t0 = values["t0"]
    sqe_cosw = values["sqe_cosw"]
    sqe_sinw = values["sqe_sinw"]
    eccentricity = sqe_cosw * sqe_cosw + sqe_sinw * sqe_sinw
    if not (
        0.001 < rp_rstar < 0.3
        and 0.0 <= impact_parameter < 1.0 + rp_rstar
        and 0.99 < baseline < 1.01
        and JITTER_LOG_LOWER < log_jitter < JITTER_LOG_UPPER
        and -50.0 < log_rho_star < 50.0
        and 0.0 < q1 < 1.0
        and 0.0 < q2 < 1.0
        and period > 0.0
        and eccentricity < 1.0
    ):
        return -np.inf
    rho_star = math.exp(log_rho_star)
    if not math.isfinite(rho_star) or rho_star <= 0.0:
        return -np.inf
    try:
        a_rstar = float(_accelerated_a_rstar_from_cgs(np.asarray([rho_star]), np.asarray([period]))[0])
    except (FloatingPointError, ValueError, OverflowError):
        return -np.inf
    if not math.isfinite(a_rstar) or a_rstar <= 0.0:
        return -np.inf
    omega_deg = math.degrees(math.atan2(sqe_sinw, sqe_cosw)) if eccentricity > 0.0 else 90.0
    model = batman_transit_flux(
        data.time_days - t0,
        period,
        rp_rstar,
        a_rstar,
        impact_parameter,
        q1,
        q2,
        baseline,
        eccentricity=eccentricity,
        omega_deg=omega_deg,
        exposure_seconds=data.exposure_seconds,
    )
    if model is None:
        return -np.inf
    jitter = math.exp(log_jitter)
    variance = np.square(data.flux_err) + jitter * jitter
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        return -np.inf
    residual = data.flux - model
    log_likelihood = -0.5 * float(
        np.sum(np.square(residual) / variance + np.log(2.0 * math.pi * variance))
    )
    log_density_sigma = data.rho_star_sigma_g_cm3 / data.rho_star_g_cm3
    log_prior = -0.5 * ((log_rho_star - math.log(data.rho_star_g_cm3)) / log_density_sigma) ** 2
    log_prior += _half_cauchy_log_jitter_log_density(log_jitter)
    if data.period_sigma_days is not None:
        log_prior += -0.5 * ((period - data.period_days) / data.period_sigma_days) ** 2
    if data.t0_sigma_days is not None:
        log_prior += -0.5 * ((t0 - data.t0_days) / data.t0_sigma_days) ** 2
    return float(log_likelihood + log_prior) if math.isfinite(log_likelihood + log_prior) else -np.inf


def _accelerated_initial_theta(data: _AcceleratedTransitFitData) -> np.ndarray:
    """Build a finite CPU starting point from normalized flux statistics."""
    observed_depth = max(float(np.median(data.flux) - np.min(data.flux)), 1.0e-8)
    rp_rstar = min(0.2, max(0.005, math.sqrt(observed_depth)))
    scatter = max(float(np.std(data.flux - np.median(data.flux))), float(np.median(data.flux_err)), 1.0e-6)
    values = [
        rp_rstar,
        math.log(data.rho_star_g_cm3),
        0.3,
        float(np.clip(np.median(data.flux), 0.995, 1.005)),
        float(np.clip(math.log(scatter), JITTER_LOG_LOWER + 0.1, JITTER_LOG_UPPER - 0.1)),
        0.5,
        0.5,
    ]
    if data.eccentric:
        values.extend([0.0, 0.0])
    if data.period_sigma_days is not None:
        values.append(data.period_days)
    if data.t0_sigma_days is not None:
        values.append(data.t0_days)
    return np.asarray(values, dtype=float)


def _fit_emcee_cpu(
    data: _AcceleratedTransitFitData,
    *,
    n_walkers: int,
    burn_in: int,
    production: int,
    seed: int,
    progress: bool,
    fallback_reason: Optional[str],
) -> Dict[str, Any]:
    """Run the batman/emcee fallback and emit the normalized output contract."""
    _require_batman()
    try:
        import emcee
    except ImportError as exc:
        raise RuntimeError("emcee is required for CPU transit fitting") from exc

    initial = _accelerated_initial_theta(data)
    ndim = initial.size
    effective_walkers = max(n_walkers, 2 * ndim)
    rng = np.random.default_rng(seed)
    scales = np.full(ndim, 0.01, dtype=float)
    scales[:7] = np.asarray([0.003, 0.03, 0.03, 0.0002, 0.15, 0.03, 0.03])
    cursor = 7
    if data.eccentric:
        scales[cursor:cursor + 2] = 0.01
        cursor += 2
    if data.period_sigma_days is not None:
        scales[cursor] = min(data.period_sigma_days * 0.05, data.period_days * 0.001)
        cursor += 1
    if data.t0_sigma_days is not None:
        scales[cursor] = min(data.t0_sigma_days * 0.05, data.period_days * 0.001)

    p0 = np.empty((effective_walkers, ndim), dtype=float)
    for walker_index in range(effective_walkers):
        for _attempt in range(1000):
            proposal = initial + rng.normal(size=ndim) * scales
            proposal[0] = np.clip(proposal[0], 0.0011, 0.299)
            proposal[2] = np.clip(proposal[2], 0.0, 1.0)
            proposal[3] = np.clip(proposal[3], 0.9901, 1.0099)
            proposal[4] = np.clip(proposal[4], JITTER_LOG_LOWER + 0.01, JITTER_LOG_UPPER - 0.01)
            proposal[5:7] = np.clip(proposal[5:7], 1.0e-5, 1.0 - 1.0e-5)
            if data.eccentric:
                eccentric_radius = float(math.hypot(proposal[7], proposal[8]))
                if eccentric_radius >= 0.99:
                    proposal[7:9] *= 0.99 / eccentric_radius
            if math.isfinite(_accelerated_cpu_log_probability(proposal, data)):
                p0[walker_index] = proposal
                break
        else:
            raise RuntimeError("could not initialize a physically valid CPU transit-fit walker")

    sampler = emcee.EnsembleSampler(
        effective_walkers,
        ndim,
        lambda theta: _accelerated_cpu_log_probability(theta, data),
        moves=emcee.moves.StretchMove(a=1.5),
    )
    sampler.random_state = np.random.RandomState(seed).get_state()
    state = sampler.run_mcmc(p0, burn_in, progress=progress)
    sampler.reset()
    sampler.run_mcmc(state, production, progress=progress)
    chain = np.asarray(sampler.get_chain(flat=True), dtype=float)
    names = _accelerated_parameter_names(data)
    samples = {name: chain[:, index] for index, name in enumerate(names)}
    result = _summarize_accelerated_samples(
        samples,
        data,
        backend="emcee-cpu",
        sampler_metadata={
            "sampler": "emcee.EnsembleSampler",
            "requested_walkers": int(n_walkers),
            "walkers": int(effective_walkers),
            "burn_in": int(burn_in),
            "production": int(production),
            "flat_samples": int(chain.shape[0]),
            "random_seed": int(seed),
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "fallback_reason": fallback_reason,
        },
    )
    return result


def _fit_numpyro_gpu(
    data: _AcceleratedTransitFitData,
    *,
    stack: Dict[str, Any],
    num_warmup: int,
    num_samples: int,
    target_accept_prob: float,
    seed: int,
    progress: bool,
) -> Dict[str, Any]:
    """Run a pure JAX/NumPyro NUTS model on the selected CUDA device."""
    jax = stack["jax"]
    jnp = stack["jnp"]
    numpyro = stack["numpyro"]
    dist = stack["dist"]
    MCMC = stack["MCMC"]
    NUTS = stack["NUTS"]
    TransitOrbit = stack["TransitOrbit"]
    limb_dark_light_curve = stack["limb_dark_light_curve"]
    gpu_device = stack["device"]

    time_days = jax.device_put(jnp.asarray(data.time_days, dtype=jnp.float64), gpu_device)
    flux = jax.device_put(jnp.asarray(data.flux, dtype=jnp.float64), gpu_device)
    flux_err = jax.device_put(jnp.asarray(data.flux_err, dtype=jnp.float64), gpu_device)
    exposure_days = jnp.asarray(data.exposure_seconds / SECONDS_PER_DAY, dtype=jnp.float64)
    sub_sample_offsets = (
        jnp.arange(SUPERSAMPLE_FACTOR, dtype=jnp.float64) - 0.5 * (SUPERSAMPLE_FACTOR - 1)
    ) * (exposure_days / SUPERSAMPLE_FACTOR)
    log_rho_center = math.log(data.rho_star_g_cm3)
    log_rho_sigma = data.rho_star_sigma_g_cm3 / data.rho_star_g_cm3
    machine_epsilon = jnp.finfo(jnp.float64).eps

    def transit_model() -> None:
        """Pure potential-energy model traced once by NumPyro's JIT compiler."""
        rp_rstar = numpyro.sample("rp_rstar", dist.Uniform(0.001, 0.3))
        log_rho_star = numpyro.sample("log_rho_star", dist.Normal(log_rho_center, log_rho_sigma))
        impact_parameter = numpyro.sample("impact_parameter", dist.Uniform(0.0, 1.2))
        baseline = numpyro.sample("baseline", dist.Uniform(0.99, 1.01))
        log_jitter = numpyro.sample("log_jitter", dist.Uniform(JITTER_LOG_LOWER, JITTER_LOG_UPPER))
        q1 = numpyro.sample("q1", dist.Uniform(0.0, 1.0))
        q2 = numpyro.sample("q2", dist.Uniform(0.0, 1.0))
        if data.eccentric:
            sqe_cosw = numpyro.sample("sqe_cosw", dist.Uniform(-1.0, 1.0))
            sqe_sinw = numpyro.sample("sqe_sinw", dist.Uniform(-1.0, 1.0))
        else:
            sqe_cosw = jnp.asarray(0.0, dtype=jnp.float64)
            sqe_sinw = jnp.asarray(0.0, dtype=jnp.float64)

        if data.period_sigma_days is None:
            period = jnp.asarray(data.period_days, dtype=jnp.float64)
            numpyro.deterministic("period", period)
        else:
            period = numpyro.sample("period", dist.Normal(data.period_days, data.period_sigma_days))
        if data.t0_sigma_days is None:
            t0 = jnp.asarray(data.t0_days, dtype=jnp.float64)
            numpyro.deterministic("t0", t0)
        else:
            t0 = numpyro.sample("t0", dist.Normal(data.t0_days, data.t0_sigma_days))

        # Eastman coordinates have a uniform disk prior after the explicit
        # radius guard below. Use safe values for algebra, then assign -inf to
        # proposals outside the physical disk without host-side branching.
        eccentricity = jnp.square(sqe_cosw) + jnp.square(sqe_sinw)
        safe_eccentricity = jnp.minimum(eccentricity, 1.0 - machine_epsilon)
        sqrt_eccentricity = jnp.sqrt(safe_eccentricity)
        conjunction_factor = jnp.maximum(1.0 + sqrt_eccentricity * sqe_sinw, machine_epsilon)
        safe_period = jnp.maximum(period, machine_epsilon)
        safe_log_rho_star = jnp.clip(log_rho_star, -50.0, 50.0)
        rho_star = jnp.exp(safe_log_rho_star)
        a_rstar = jnp.cbrt(
            GRAVITATIONAL_CONSTANT_CGS
            * jnp.square(safe_period * SECONDS_PER_DAY)
            * rho_star
            / (3.0 * math.pi)
        )
        conjunction_distance = a_rstar * (1.0 - safe_eccentricity) / conjunction_factor
        cosine_inclination = impact_parameter / jnp.maximum(conjunction_distance, machine_epsilon)
        sine_inclination = jnp.sqrt(jnp.maximum(1.0 - jnp.square(cosine_inclination), machine_epsilon))
        sky_speed = (
            2.0
            * math.pi
            * a_rstar
            / safe_period
            * conjunction_factor
            / jnp.sqrt(jnp.maximum(1.0 - jnp.square(safe_eccentricity), machine_epsilon))
            * sine_inclination
        )
        valid_geometry = (
            (period > 0.0)
            & (log_rho_star > -50.0)
            & (log_rho_star < 50.0)
            & (eccentricity < 1.0)
            & (impact_parameter < 1.0 + rp_rstar)
            & (cosine_inclination >= 0.0)
            & (cosine_inclination <= 1.0)
            & jnp.isfinite(a_rstar)
            & jnp.isfinite(sky_speed)
            & (sky_speed > 0.0)
        )
        numpyro.factor("physical_geometry", jnp.where(valid_geometry, 0.0, -jnp.inf))
        jitter = jnp.exp(log_jitter)
        half_cauchy_ratio = jitter / JITTER_HALF_CAUCHY_SCALE
        numpyro.factor(
            "log_jitter_prior",
            jnp.log(2.0 / (math.pi * JITTER_HALF_CAUCHY_SCALE))
            - jnp.log1p(jnp.square(half_cauchy_ratio))
            + log_jitter,
        )

        u1 = 2.0 * jnp.sqrt(q1) * q2
        u2 = jnp.sqrt(q1) * (1.0 - 2.0 * q2)
        orbit = TransitOrbit(
            period=safe_period,
            speed=jnp.maximum(sky_speed, machine_epsilon),
            time_transit=t0,
            impact_param=impact_parameter,
            radius_ratio=rp_rstar,
        )
        # Keep exposure integration O(N) in device memory rather than creating
        # an N-by-supersample temporary for every NUTS potential evaluation.
        # The fixed, side-effect-free stencil is unrolled deterministically by
        # JAX during compilation.
        light_curve = limb_dark_light_curve(orbit, u1, u2)
        delta_flux = sum(
            light_curve(time_days + sub_sample_offsets[index])
            for index in range(SUPERSAMPLE_FACTOR)
        ) / SUPERSAMPLE_FACTOR
        model_flux = baseline * (1.0 + delta_flux)
        total_sigma = jnp.sqrt(jnp.square(flux_err) + jnp.square(jitter))
        numpyro.sample("observed_flux", dist.Normal(model_flux, total_sigma).to_event(1), obs=flux)

    kernel = NUTS(transit_model, target_accept_prob=float(target_accept_prob))
    sampler = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=progress,
        jit_model_args=True,
    )
    random_key = jax.device_put(jax.random.PRNGKey(seed), gpu_device)
    sampler.run(random_key, extra_fields=("diverging",))
    raw_samples = sampler.get_samples(group_by_chain=False)
    samples = {name: np.asarray(values, dtype=float).reshape(-1) for name, values in raw_samples.items()}
    if "period" not in samples:
        samples["period"] = np.full(samples["rp_rstar"].size, data.period_days, dtype=float)
    if "t0" not in samples:
        samples["t0"] = np.full(samples["rp_rstar"].size, data.t0_days, dtype=float)
    samples["rho_star_g_cm3"] = np.exp(samples["log_rho_star"])
    extra_fields = sampler.get_extra_fields(group_by_chain=False)
    divergences = int(np.sum(np.asarray(extra_fields.get("diverging", ()), dtype=bool)))
    return _summarize_accelerated_samples(
        samples,
        data,
        backend="jax-gpu",
        sampler_metadata={
            "sampler": "numpyro.NUTS",
            "num_warmup": int(num_warmup),
            "num_samples": int(num_samples),
            "target_accept_prob": float(target_accept_prob),
            "random_seed": int(seed),
            "gpu_device": str(gpu_device),
            "divergences": divergences,
            "jax_enable_x64": True,
        },
    )


def batman_transit_flux(
    phase_days: Sequence[float],
    period_days: float,
    rp_rs: float,
    a_rs: float,
    impact_parameter: float,
    q1: float,
    q2: float,
    baseline: float,
    eccentricity: float = 0.0,
    omega_deg: float = 90.0,
    exposure_seconds: float = FITTED_BIN_EXPOSURE_SECONDS,
) -> Optional[np.ndarray]:
    """Evaluate a batman quadratic limb-darkening transit model.

    This function wraps `batman.TransitModel` (Kreidberg 2015), which
    implements the analytic occultation formalism of Mandel & Agol (2002)
    for a uniform source with quadratic limb darkening.

    The Kipping (2013) coordinates ``(q1, q2)`` are transformed back to
    standard quadratic coefficients ``(u1, u2)`` via
    :func:`kipping_to_quadratic_limb_darkening` before being passed to batman.

    For eccentric orbits, the impact parameter is converted to inclination
    through the conjunction-distance relation (see
    :func:`inclination_deg_from_impact_parameter`), ensuring the Keplerian
    geometry is consistent with the supplied eccentricity and argument of
    periastron.

    Parameters
    ----------
    phase_days : array-like of float
        Transit-relative times in days, centred such that mid-transit is at
        phase = 0.
    period_days : float
        Orbital period in days.
    rp_rs : float
        Planet-to-star radius ratio :math:`R_p / R_*`.
    a_rs : float
        Dimensionless semi-major axis ``a / R_*`` (the Keplerian value
        expected by batman; eccentric geometry is handled via inclination).
    impact_parameter : float
        Transit impact parameter at inferior conjunction.
    q1, q2 : float
        Kipping (2013) limb-darkening parameters, each in ``(0.01, 0.99)``
        in practice.
    baseline : float
        Out-of-transit flux normalization, applied multiplicatively.
    eccentricity : float, optional
        Orbital eccentricity (default 0).
    omega_deg : float, optional
        Argument of periastron in degrees (default 90).  Irrelevant when
        ``eccentricity == 0``.
    exposure_seconds : float, optional
        Effective exposure integration time in seconds.  The ``batman``
        supersampling is performed in units of days, so this value is
        divided by the declared seconds-per-day conversion internally.

    Returns
    -------
    numpy.ndarray or None
        The transit model flux at each phase point, or None if the parameter
        combination produces a non-physical geometry, the inclination cannot
        be computed, or the forward model rejects a proposal.

    Raises
    ------
    RuntimeError
        If the required ``batman-package`` dependency is unavailable.
    """
    # NUMERICAL_GUARD: non-positive exposure time prevents batman's
    # supersampling integrator from running.
    if not math.isfinite(exposure_seconds) or exposure_seconds <= 0:
        return None
    inclination_deg = inclination_deg_from_impact_parameter(
        a_rs, impact_parameter, eccentricity, omega_deg
    )
    if inclination_deg is None:
        return None
    batman = _require_batman()
    try:
        # Kipping (2013) inverse transform to standard quadratic coefficients.
        u1, u2 = kipping_to_quadratic_limb_darkening(q1, q2)
        params = batman.TransitParams()
        params.t0 = 0.0                      # phase-folded: mid-transit at zero
        params.per = period_days
        params.rp = rp_rs
        params.a = a_rs                      # Keplerian semi-major axis
        params.inc = inclination_deg         # derived from b via conjunction distance
        params.ecc = eccentricity
        params.w = omega_deg
        params.u = [u1, u2]
        params.limb_dark = "quadratic"

        # NUMERICAL_GUARD: SUPERSAMPLE_FACTOR = 7 sub-samples the exposure
        # to approximate finite-integration-time smearing (Kipping 2010).
        model = batman.TransitModel(
            params,
            np.asarray(phase_days, dtype=float),
            supersample_factor=SUPERSAMPLE_FACTOR,
            exp_time=float(exposure_seconds) / SECONDS_PER_DAY,
        )

        # Mandel & Agol (2002) analytic flux evaluation.
        flux = np.asarray(model.light_curve(params), dtype=float)
        if not np.all(np.isfinite(flux)):
            return None
        # SCIENTIFIC_BOUNDARY: baseline scaling is a multiplicative
        # out-of-transit normalization, not a calibrated per-sector
        # contamination correction.
        return baseline * flux
    except Exception:
        # The dependency import is intentionally outside this handler. A
        # proposal-level forward-model failure has zero likelihood, while a
        # missing dependency must stop the public fit loudly.
        return None


def _folded_binned_data(
    time: Sequence[float],
    flux: Sequence[float],
    ephemeris: Dict[str, Any],
    window_half_hours: float = WINDOW_HALF_HOURS,
    bin_minutes: float = BIN_MINUTES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase-fold and median-bin a light curve around the transit window.

    This utility delegates to :func:`bin_phase_folded_flux` for the
    phase-folding and median-binning logic, then filters the output to
    retain only finite, positive-uncertainty bins within the specified
    window.

    Parameters
    ----------
    time : array-like of float
        Time stamps (BTJD).
    flux : array-like of float
        Normalized relative flux.
    ephemeris : dict
        Must contain ``period_days`` (float, > 0) and ``epoch_btjd`` (float).
    window_half_hours : float, optional
        Half-width of the transit phase window in hours.
    bin_minutes : float, optional
        Bin width in minutes for median binning.

    Returns
    -------
    centers_days : ndarray
        Median-binned phase centres in days relative to mid-transit.
    binned_flux : ndarray
        Median flux in each bin.
    binned_error : ndarray
        Standard error of the mean in each bin.

    Raises
    ------
    ValueError
        If fewer than 20 valid bins survive the quality filter.
    """
    # SCIENTIFIC_BOUNDARY: bin_phase_folded_flux performs median binning
    # for visualisation and descriptive fit speed; the native-cadence
    # likelihood path uses _native_transit_window_data instead.
    centers_hours, binned_flux, binned_error = bin_phase_folded_flux(
        time,
        flux,
        ephemeris["period_days"],
        ephemeris["epoch_btjd"],
        limit_hours=window_half_hours,
        bin_minutes=bin_minutes,
    )
    # NUMERICAL_GUARD: require finite values and positive uncertainties
    # so that the chi-squared computation downstream is well-defined.
    valid = (
        np.isfinite(centers_hours)
        & np.isfinite(binned_flux)
        & np.isfinite(binned_error)
        & (binned_error > 0)
    )
    # ASTROPHYSICAL_HEURISTIC: 20 bins is a minimal sample for a
    # quadratic limb-darkening fit with 5-7 free parameters; fewer bins
    # risk degenerate posteriors.
    if int(valid.sum()) < 20:
        raise ValueError("insufficient binned transit window coverage")
    return (
        centers_hours[valid] / 24.0,
        binned_flux[valid],
        binned_error[valid],
    )


def _native_transit_window_data(
    table: Dict[str, Any],
    ephemeris: Dict[str, Any],
    window_half_hours: float = WINDOW_HALF_HOURS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], np.ndarray]:
    """Return native-cadence transit windows with sector-specific integrations.

    The light-curve loader supplies normalized, quality-filtered observations.
    This helper retains every finite cadence inside the declared fit window,
    maps each cadence to a compact sector index, and estimates one integration
    time per sector from within-sector sampling intervals.  It does not mix
    cadences across sectors or phase-bin them before inference.

    Mathematical Formulation
    ------------------------
    The transit-relative phase (in days) is computed as

    .. math::

        \\Delta t = \\bigl((t - t_0 + 0.5 P) \\bmod P\\bigr) - 0.5 P

    where :math:`P` is the orbital period and :math:`t_0` is the reference
    mid-transit epoch.  The effective crop window is the minimum of the
    user-supplied ``window_half_hours`` and 2.5 times the ephemeris
    duration to capture the full transit egress plus baseline.

    Parameters
    ----------
    table : dict
        Light curve table with keys ``time``, ``flux``, ``flux_err``,
        ``sector``.
    ephemeris : dict
        Transit ephemeris with ``period_days``, ``epoch_btjd``,
        ``duration_days``.
    window_half_hours : float, optional
        Crop window half-width in hours (default from module constant).

    Returns
    -------
    phase_days : ndarray
        Transit-relative time in days (centred at zero).
    flux : ndarray
        Normalized flux.
    flux_err : ndarray
        Flux uncertainty.
    sector_index : ndarray of int
        Zero-based compact sector index per cadence.
    sector_labels : list of int
        Original sector labels sorted ascending.
    exposure_seconds_by_sector : ndarray
        Median intra-sector cadence in seconds for each sector.

    Raises
    ------
    ValueError
        If the table or ephemeris is malformed, the effective window
        yields fewer than 100 valid cadences, or a sector has no
        measurable cadence.
    """
    try:
        time = np.asarray(table["time"], dtype=float)
        flux = np.asarray(table["flux"], dtype=float)
        flux_err = np.asarray(table["flux_err"], dtype=float)
        sectors = np.asarray(table["sector"], dtype=int)
        period_days = float(ephemeris["period_days"])
        epoch_btjd = float(ephemeris["epoch_btjd"])
        duration_days = float(ephemeris["duration_days"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("candidate photometry or ephemeris is malformed") from exc
    if (
        time.ndim != 1
        or flux.shape != time.shape
        or flux_err.shape != time.shape
        or sectors.shape != time.shape
        or not math.isfinite(period_days)
        or period_days <= 0
        or not math.isfinite(epoch_btjd)
        or not math.isfinite(duration_days)
        or duration_days <= 0
        or not math.isfinite(window_half_hours)
        or window_half_hours <= 0
    ):
        raise ValueError("candidate photometry cannot define a native transit window")

    # ASTROPHYSICAL_HEURISTIC: window capped at 2.5 × duration ensures
    # out-of-transit baseline is captured but very-long-period objects
    # with WINDOW_HALF_HOURS >> duration still have a manageable data
    # volume. The factor 2.5 provides ~1.75 durations of baseline on
    # each side.
    effective_window = min(window_half_hours, max(2.5, duration_days * 24.0 * 2.5))
    phase_days = ((time - epoch_btjd + 0.5 * period_days) % period_days) - 0.5 * period_days
    valid = (
        np.isfinite(time)
        & np.isfinite(phase_days)
        & np.isfinite(flux)
        & np.isfinite(flux_err)
        & (flux_err > 0)
        & (np.abs(phase_days) <= effective_window / 24.0)
    )
    if int(np.count_nonzero(valid)) < 100:
        raise ValueError("insufficient native-cadence transit-window coverage")
    phase_days = phase_days[valid]
    flux = flux[valid]
    flux_err = flux_err[valid]
    time = time[valid]
    sectors = sectors[valid]

    sector_labels = sorted(int(value) for value in np.unique(sectors))
    if not sector_labels:
        raise ValueError("candidate transit window has no sector ownership")
    sector_index = np.empty(sectors.size, dtype=int)
    exposure_seconds = []
    for index, sector_label in enumerate(sector_labels):
        selected = sectors == sector_label
        sector_time = np.sort(time[selected])
        cadence_seconds = np.diff(sector_time) * SECONDS_PER_DAY
        cadence_seconds = cadence_seconds[np.isfinite(cadence_seconds) & (cadence_seconds > 0)]
        if cadence_seconds.size == 0:
            raise ValueError("candidate sector has no measurable integration cadence")
        median_cadence_seconds = float(np.median(cadence_seconds))
        if not math.isfinite(median_cadence_seconds) or median_cadence_seconds <= 0:
            raise ValueError("candidate sector has an invalid integration cadence")
        sector_index[selected] = index
        exposure_seconds.append(median_cadence_seconds)
    return phase_days, flux, flux_err, sector_index, sector_labels, np.asarray(exposure_seconds)


def _parameter_names(
    eccentric: bool,
    n_sectors: int = 1,
    sector_labels: Optional[Sequence[int]] = None,
) -> List[str]:
    """Return the sampled-parameter names for one or more observed sectors.

    For a single sector the layout is ``(rp_rs, log_rho_star,
    impact_parameter, baseline, log_jitter, q1, q2 [, sqe_cosw, sqe_sinw])``.
    When multiple sectors are passed, ``rp_rs``, ``log_rho_star``, and
    ``impact_parameter`` are shared, but each sector receives its own
    baseline parameter ``baseline_sector_<label>`` to absorb per-sector
    normalization offsets.

    Parameters
    ----------
    eccentric : bool
        Whether the eccentricity components are included.
    n_sectors : int, optional
        Number of distinct observation sectors.
    sector_labels : sequence of int, optional
        Original sector labels for the baseline parameter names;
        defaults to ``range(n_sectors)``.

    Returns
    -------
    list of str
        Ordered parameter names matching the theta vector layout.
    """
    if n_sectors <= 0:
        raise ValueError("transit likelihood requires at least one sector")
    if n_sectors == 1:
        return list(PARAMETER_NAMES_ECCENTRIC if eccentric else PARAMETER_NAMES_CIRCULAR)
    labels = list(sector_labels) if sector_labels is not None else list(range(n_sectors))
    if len(labels) != n_sectors:
        raise ValueError("sector labels do not match the transit likelihood")
    names = ["rp_rs", "log_rho_star", "impact_parameter"]
    names.extend("baseline_sector_{0}".format(int(label)) for label in labels)
    names.extend(["log_jitter", "q1", "q2"])
    if eccentric:
        names.extend(["sqe_cosw", "sqe_sinw"])
    return names


def _parameter_count(eccentric: bool, n_sectors: int = 1) -> int:
    """Return the number of sampled parameters for the selected sectors.

    The count is ``6 + n_sectors`` for circular orbits (adding one baseline
    per sector) or ``8 + n_sectors`` for eccentric orbits (adding two
    :math:`(\\sqrt{e}\\cos\\omega, \\sqrt{e}\\sin\\omega)` components).
    """
    return 6 + n_sectors + (2 if eccentric else 0)


def _initial_fit_parameters(
    depth_ppm: float,
    rho_prior_solar: float,
    scatter: float,
    eccentric: bool,
    n_sectors: int = 1,
) -> np.ndarray:
    """Build a finite starting point with explicit logarithm conventions.

    ``log_rho_star`` is base-10 log density in solar units, while
    ``log_jitter`` is the natural log of normalized flux scatter because the
    likelihood recovers jitter as ``exp(log_jitter)``.

    Mathematical Formulation
    ------------------------
    The initial radius ratio is seeded from the transit depth under the
    approximation :math:`\\delta \\approx (R_p / R_*)^2`:

    .. math::

        R_p / R_* \\approx \\sqrt{\\delta} = \\sqrt{\\text{depth\\_ppm} \\times 10^{-6}}.

    Parameters
    ----------
    depth_ppm : float
        Transit depth in parts per million.
    rho_prior_solar : float
        Prior stellar density in solar units.
    scatter : float
        Flux scatter (standard deviation) used to seed log_jitter.
    eccentric : bool
        Whether eccentricity components are initialised.
    n_sectors : int, optional
        Number of observation sectors (default 1).

    Returns
    -------
    ndarray
        Initial parameter vector ordered as in :func:`_parameter_names`.
    """
    if (
        not math.isfinite(depth_ppm)
        or not math.isfinite(rho_prior_solar)
        or not math.isfinite(scatter)
        or depth_ppm < 0
        or rho_prior_solar <= 0
        or scatter < 0
    ):
        raise ValueError("fit initialization requires finite non-negative depth and scatter plus positive density")
    rp_start = min(0.2, max(0.01, math.sqrt(depth_ppm * 1e-6)))
    log_jitter_start = math.log(max(scatter, 1e-6))
    if n_sectors <= 0:
        raise ValueError("transit initialization requires at least one sector")
    common = [rp_start, math.log10(rho_prior_solar), 0.3]
    common.extend([1.0] * n_sectors)
    common.extend([log_jitter_start, 0.35, 0.3])
    if eccentric:
        common.extend([0.0, 0.0])
    return np.asarray(common, dtype=float)


def _jitter_prior_assumption() -> Dict[str, Any]:
    """Describe the data-independent jitter prior shared by both samplers.

    The prior is half-Cauchy in the positive jitter amplitude rather than a
    Gaussian centred on an uncertainty statistic from the fitted light curve.
    The production initializer may still use observed scatter to find a
    practical starting point, but that value is not part of the posterior
    density or the dynesty prior transform.
    """
    return {
        "parameter": "log_jitter",
        "distribution": "half-cauchy-on-jitter",
        "parameterization": "jitter = exp(log_jitter)",
        "scale_normalized_flux": JITTER_HALF_CAUCHY_SCALE,
        "truncation_log_jitter": {
            "lower": JITTER_LOG_LOWER,
            "upper": JITTER_LOG_UPPER,
        },
        "empirical_bayes": False,
        "data_dependent": False,
        "rationale": (
            "A fixed heavy-tailed scale permits excess white noise without using the "
            "same photometry to set the prior centre and then evaluate the likelihood."
        ),
        "limitation": (
            "This is still an independent-white-noise jitter term, not a correlated-noise "
            "or instrument-noise calibration."
        ),
    }


def _half_cauchy_log_jitter_log_density(log_jitter: float) -> float:
    """Return the log density induced on ``log_jitter`` by the fixed prior."""
    jitter = math.exp(log_jitter)
    ratio = jitter / JITTER_HALF_CAUCHY_SCALE
    return float(
        math.log(2.0 / (math.pi * JITTER_HALF_CAUCHY_SCALE))
        - math.log1p(ratio * ratio)
        + log_jitter
    )


def _stellar_density_prior(stellar: Dict[str, Any]) -> Dict[str, float]:
    """Propagate candidate-supplied stellar mass and radius uncertainties.

    Mathematical Formulation
    ------------------------
    Mean stellar density in solar units is

    .. math::

        \\rho_* / \\rho_\\odot = \\frac{M / M_\\odot}{(R / R_\\odot)^3}.

    Assuming independent, symmetric errors :math:`\\sigma_M` and
    :math:`\\sigma_R`, the first-order relative density uncertainty is
    (Carroll & Ostlie, eq. 7.26):

    .. math::

        \\frac{\\sigma_\\rho}{\\rho}
        = \\sqrt{\\left(\\frac{\\sigma_M}{M}\\right)^2
               + \\left(\\frac{3\\sigma_R}{R}\\right)^2}.

    This is converted to :math:`\\log_{10}` space via division by
    :math:`\\ln(10)` to match the ``log_rho`` parameter convention.

    The fit needs symmetric one-sigma ``mass_solar_err`` and
    ``radius_solar_err`` values.  It approximates their errors as independent
    and propagates ``rho_star = M_star / R_star**3`` into base-10 log-density
    space.  No fixed generic density width is substituted when that evidence
    is absent.

    Parameters
    ----------
    stellar : dict
        Must contain ``mass_solar``, ``mass_solar_err``, ``radius_solar``,
        ``radius_solar_err`` — all positive finite floats.

    Returns
    -------
    dict
        Keys: ``rho_solar`` (mean density in solar units), ``log10_sigma``
        (:math:`\\log_{10}` uncertainty), plus the input mass and radius
        values for provenance traceability.

    Raises
    ------
    RuntimeError
        If required keys are missing or produce a non-finite density prior.
    """
    try:
        mass_solar = float(stellar["mass_solar"])
        mass_solar_err = float(stellar["mass_solar_err"])
        radius_solar = float(stellar["radius_solar"])
        radius_solar_err = float(stellar["radius_solar_err"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "transit fitting requires candidate stellar mass_solar_err and radius_solar_err"
        ) from exc
    if not all(
        math.isfinite(value)
        for value in (mass_solar, mass_solar_err, radius_solar, radius_solar_err)
    ) or mass_solar <= 0 or mass_solar_err <= 0 or radius_solar <= 0 or radius_solar_err <= 0:
        raise RuntimeError(
            "transit fitting requires positive finite candidate stellar mass and radius uncertainties"
        )
    rho_solar = mass_solar / radius_solar**3
    relative_density_err = math.sqrt(
        (mass_solar_err / mass_solar) ** 2 + (3.0 * radius_solar_err / radius_solar) ** 2
    )
    log10_sigma = relative_density_err / math.log(10.0)
    if not math.isfinite(rho_solar) or rho_solar <= 0 or not math.isfinite(log10_sigma) or log10_sigma <= 0:
        raise RuntimeError("candidate stellar uncertainties cannot produce a finite density prior")
    return {
        "rho_solar": float(rho_solar),
        "log10_sigma": float(log10_sigma),
        "mass_solar": mass_solar,
        "mass_solar_err": mass_solar_err,
        "radius_solar": radius_solar,
        "radius_solar_err": radius_solar_err,
    }


def _load_ldtk_prior(workspace: CandidateWorkspace) -> Dict[str, Any]:
    """Load one finite candidate-local quadratic LDTk prior for explicit use.

    Reads the output artifact written by ``exonym limb-darkening``, which
    invokes LDTk (Parviainen & Aigrain 2015) to interpolate the Husser et al.
    (2013) PHOENIX stellar atmosphere grid for quadratic limb-darkening
    coefficients ``(u1, u2)`` and their uncertainties, using the candidate's
    :math:`T_{\\rm eff}`, :math:`\\log g`, [Fe/H], and the specified
    passband.

    The prior is used to weight the Kipping-parameter likelihood via Gaussian
    terms centred on the LDTk-predicted ``(u1, u2)`` values (see
    :func:`_log_prior`).

    Parameters
    ----------
    workspace : CandidateWorkspace
        The candidate workspace.

    Returns
    -------
    dict
        Keys: ``u1``, ``u1_err``, ``u2``, ``u2_err`` (all finite floats),
        ``path`` (relative path to the artifact).

    Raises
    ------
    ValueError
        If the artifact is missing, malformed, targets a different
        candidate, or has non-positive uncertainties.
    """
    path = workspace.path / "outputs" / "ldtk_quadratic_limb_darkening_prior.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("--ldtk-prior requires a readable candidate-local LDTk prior artifact") from exc
    if payload.get("candidate_id") != workspace.candidate_id:
        raise ValueError("LDTk prior candidate_id does not match the fit workspace")
    coefficients = payload.get("quadratic_coefficients")
    if not isinstance(coefficients, list) or len(coefficients) != 1:
        raise ValueError("--ldtk-prior requires exactly one recorded passband prior")
    coefficient = coefficients[0]
    if not isinstance(coefficient, dict):
        raise ValueError("LDTk prior coefficient record is invalid")
    values = {}
    for name in ("u1", "u1_err", "u2", "u2_err"):
        value = coefficient.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("LDTk prior {0} must be finite".format(name))
        if name.endswith("_err") and value <= 0:
            raise ValueError("LDTk prior uncertainties must be positive")
        values[name] = float(value)
    values["path"] = str(path.relative_to(workspace.path)).replace("\\", "/")
    return values


def _neg_log_posterior(
    theta: np.ndarray,
    phase_days: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    ephemeris: Dict[str, Any],
    rho_prior_solar: float,
    rho_prior_log10_sigma: float,
    eccentric: bool,
    ldtk_prior: Optional[Dict[str, Any]] = None,
    sector_index: Optional[np.ndarray] = None,
    exposure_seconds_by_sector: Optional[np.ndarray] = None,
    n_sectors: int = 1,
) -> float:
    """Return the negative log posterior for a single parameter vector.

    This function computes

    .. math::

        -\\ln P(\\theta\\,|\\,D) = -\\ln L(D\\,|\\,\\theta) - \\ln P(\\theta)

    and returns ``+inf`` whenever the prior or likelihood is non-finite,
    which is the convention required by ``emcee`` and ``scipy.optimize``
    minimisers.

    Parameters
    ----------
    theta : ndarray
        Parameter vector matching the layout from :func:`_parameter_names`.
    phase_days : ndarray
        Native-cadence transit-relative time in days.
    flux : ndarray
        Native-cadence normalized flux.
    flux_err : ndarray
        Native-cadence flux uncertainty.
    ephemeris : dict
        Transit ephemeris (``period_days``, ``epoch_btjd``).
    rho_prior_solar : float
        Prior mean density in solar units.
    rho_prior_log10_sigma : float
        Prior density width in :math:`\\log_{10}` space.
    eccentric : bool
        Whether the eccentricity components are active.
    ldtk_prior : dict, optional
        LDTk quadratic coefficient prior (see :func:`_load_ldtk_prior`).
    sector_index, exposure_seconds_by_sector, n_sectors
        Native-cadence sector descriptors.

    Returns
    -------
    float
        ``-ln P``, or ``+inf`` for non-finite values.
    """
    log_prior = _log_prior(
        theta,
        rho_prior_solar,
        rho_prior_log10_sigma,
        eccentric,
        ldtk_prior,
        n_sectors=n_sectors,
    )
    if not math.isfinite(log_prior):
        return float("inf")
    log_likelihood = _log_likelihood(
        theta,
        phase_days,
        flux,
        flux_err,
        ephemeris,
        eccentric,
        sector_index=sector_index,
        exposure_seconds_by_sector=exposure_seconds_by_sector,
        n_sectors=n_sectors,
    )
    if not math.isfinite(log_likelihood):
        return float("inf")
    return float(-log_likelihood - log_prior)


def _unpack_theta(
    theta: np.ndarray, eccentric: bool, n_sectors: int = 1
) -> Tuple[Any, ...]:
    """Return physical parameters and one flux normalization per data sector.

    Unpacks the flat parameter vector into named components:

    - ``rp_rs``: planet-to-star radius ratio
    - ``log_rho_star``: :math:`\\log_{10}(\\rho_* / \\rho_\\odot)`
    - ``impact_parameter``: :math:`b` at conjunction
    - ``baselines``: per-sector out-of-transit flux normalizations
    - ``log_jitter``: :math:`\\ln \\sigma_j`, natural-log jitter
    - ``q1, q2``: Kipping (2013) limb-darkening parameters
    - ``sqe_cosw, sqe_sinw``: :math:`(\\sqrt{e}\\cos\\omega, \\sqrt{e}\\sin\\omega)`
      (zero for circular orbits)

    Parameters
    ----------
    theta : ndarray
        Flat parameter vector.
    eccentric : bool
        Whether eccentricity components are present.
    n_sectors : int, optional
        Number of observation sectors.

    Returns
    -------
    tuple
        ``(rp_rs, log_rho_star, impact_parameter, baselines, log_jitter,
        q1, q2, sqe_cosw, sqe_sinw)``.
    """
    expected = _parameter_count(eccentric, n_sectors)
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (expected,):
        raise ValueError("transit parameter vector has an invalid sector layout")
    baseline_stop = 3 + n_sectors
    baselines = theta[3:baseline_stop]
    log_jitter, q1, q2 = (float(value) for value in theta[baseline_stop:baseline_stop + 3])
    if eccentric:
        se_cos, se_sin = (float(value) for value in theta[baseline_stop + 3:baseline_stop + 5])
    else:
        se_cos, se_sin = 0.0, 0.0
    return (
        float(theta[0]),
        float(theta[1]),
        float(theta[2]),
        baselines,
        log_jitter,
        q1,
        q2,
        se_cos,
        se_sin,
    )


def _log_prior(
    theta: np.ndarray,
    rho_prior_solar: float,
    rho_prior_log10_sigma: float,
    eccentric: bool,
    ldtk_prior: Optional[Dict[str, Any]] = None,
    n_sectors: int = 1,
) -> float:
    """Evaluate parameter priors separately from the photometric likelihood.

    The prior encodes:

    - **Kipping limb darkening**: uniform Dirichlet prior on ``(q1, q2)``
      over the triangular support :math:`0 < u_1 + u_2 < 1`, which
      automatically ensures physically valid quadratic limb-darkening
      coefficients (Kipping 2013).
    - **log_rho_star**: Gaussian prior centred on the candidate-local
      stellar density with width propagated from mass and radius
      uncertainties (see :func:`_stellar_density_prior`).
    - **log_jitter**: a fixed half-Cauchy prior on the positive jitter
      amplitude with a 1000-ppm normalized-flux scale. This is intentionally
      independent of the fitted light curve, avoiding empirical-Bayes reuse.
    - **LDTk prior** (optional): Gaussian terms on ``(u1, u2)`` derived
      from the candidate's limb-darkening table, anchored to the
      Husser et al. (2013) PHOENIX grid via Parviainen & Aigrain (2015).
    - **impact_parameter**: unbounded prior with a rectangular guard
      at :math:`b \\le 1.2` allowing for grazing transits slightly
      beyond the stellar limb.
    - **eccentricity**: flat prior on :math:`(\\sqrt{e}\\cos\\omega,
      \\sqrt{e}\\sin\\omega)` with physical bound :math:`e < 1`.

    Returns ``-inf`` for any point outside the joint physical domain.
    """
    if not np.all(np.isfinite(theta)):
        return -np.inf
    try:
        rp, log_rho, b, baselines, log_jitter, q1, q2, se_cos, se_sin = _unpack_theta(
            theta, eccentric, n_sectors
        )
    except ValueError:
        return -np.inf
    if not (
        0.001 < rp < 0.3
        and -2.0 < log_rho < 1.5
        # ASTROPHYSICAL_HEURISTIC: b <= 1.2 admits grazing transits (b slightly > 1).
        # The quadratic limb-darkening model is defined up to the stellar limb;
        # a few percent beyond (partially occulted planet) is allowed because
        # impact-parameter uncertainties from the ephemeris may exceed the
        # stellar radius.  Posteriors with median b > 1.0 should be flagged for
        # manual review as they are degenerate with high-impact-parameter
        # eclipsing binaries.
        # DIAGNOSTIC_REASONING: the bottleneck guard at b < 1.2 prevents the
        # sampler from spending time in the fully non-transiting regime, while
        # the range 1.0 < b < 1.2 allows the posterior to explore grazing
        # geometries that may still produce a detectable transit.
        and 0.0 <= b < 1.2
        and bool(np.all((0.99 < baselines) & (baselines < 1.01)))
        # NUMERICAL_GUARD: log_jitter bounds keep exp(log_jitter) within
        # machine-precision finite range: exp(-12) ≈ 6e-6, exp(-2) ≈ 0.14
        # (normalized flux units).
        and JITTER_LOG_LOWER < log_jitter < JITTER_LOG_UPPER
        and 0.01 < q1 < 0.99
        and 0.01 < q2 < 0.99
    ):
        return -np.inf
    if eccentric and se_cos * se_cos + se_sin * se_sin > 1.0:
        return -np.inf

    if (
        rho_prior_solar <= 0
        or not math.isfinite(rho_prior_solar)
        or rho_prior_log10_sigma <= 0
        or not math.isfinite(rho_prior_log10_sigma)
    ):
        return -np.inf
    log_prior = -0.5 * (
        (log_rho - math.log10(rho_prior_solar)) / rho_prior_log10_sigma
    ) ** 2
    log_prior += _half_cauchy_log_jitter_log_density(log_jitter)
    if ldtk_prior is not None:
        u1, u2 = kipping_to_quadratic_limb_darkening(q1, q2)
        log_prior += -0.5 * ((u1 - ldtk_prior["u1"]) / ldtk_prior["u1_err"]) ** 2
        log_prior += -0.5 * ((u2 - ldtk_prior["u2"]) / ldtk_prior["u2_err"]) ** 2
    return float(log_prior)


def _log_likelihood(
    theta: np.ndarray,
    phase_days: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    ephemeris: Dict[str, Any],
    eccentric: bool,
    sector_index: Optional[np.ndarray] = None,
    exposure_seconds_by_sector: Optional[np.ndarray] = None,
    n_sectors: int = 1,
) -> float:
    """Evaluate a Gaussian native-cadence likelihood with sector baselines.

    The log-likelihood for :math:`N` cadences is (Foreman-Mackey et al. 2013,
    eq. 35):

    .. math::

        \\ln L = -\\frac{1}{2} \\sum_{i=1}^N \\left[
            \\frac{(f_i - m_i)^2}{\\sigma_i^2 + \\sigma_j^2}
            + \\ln\\bigl(2\\pi(\\sigma_i^2 + \\sigma_j^2)\\bigr)
        \\right],

    where :math:`f_i` are the observed fluxes, :math:`m_i` the batman-predicted
    model fluxes, :math:`\\sigma_i` the reported uncertainties, and
    :math:`\\sigma_j = e^{\\ln\\sigma_j}` a global jitter term fitted in
    natural-log space to enforce positivity.

    Each sector may have its own baseline normalization and exposure time,
    but the planet radius ratio, stellar density, impact parameter, limb
    darkening, and jitter are shared across all sectors.

    Returns ``-inf`` if the forward model rejects a proposal or if any sector
    index is out of range. A missing required ``batman-package`` dependency
    propagates as ``RuntimeError`` so a public fit cannot silently continue.
    """
    if not np.all(np.isfinite(theta)):
        return -np.inf
    try:
        rp, log_rho, b, baselines, log_jitter, q1, q2, se_cos, se_sin = _unpack_theta(
            theta, eccentric, n_sectors
        )
    except ValueError:
        return -np.inf
    eccentricity = se_cos * se_cos + se_sin * se_sin if eccentric else 0.0
    if eccentricity >= 1.0:
        return -np.inf
    omega_deg = math.degrees(math.atan2(se_sin, se_cos)) if eccentricity > 0 else 90.0

    try:
        period_days = ephemeris["period_days"]
        rho_solar = 10.0 ** log_rho
        a_rs = stellar_density_a_rs(rho_solar, period_days)
    except (KeyError, OverflowError, ValueError):
        return -np.inf
    if sector_index is None:
        sector_index = np.zeros(np.asarray(phase_days).size, dtype=int)
    else:
        sector_index = np.asarray(sector_index, dtype=int)
    if sector_index.shape != np.asarray(phase_days).shape or np.any(sector_index < 0) or np.any(sector_index >= n_sectors):
        return -np.inf
    if exposure_seconds_by_sector is None:
        exposure_seconds_by_sector = np.full(n_sectors, FITTED_BIN_EXPOSURE_SECONDS)
    else:
        exposure_seconds_by_sector = np.asarray(exposure_seconds_by_sector, dtype=float)
    if (
        exposure_seconds_by_sector.shape != (n_sectors,)
        or not np.all(np.isfinite(exposure_seconds_by_sector))
        or np.any(exposure_seconds_by_sector <= 0)
    ):
        return -np.inf
    model = np.empty(np.asarray(phase_days).shape, dtype=float)
    for sector_number in range(n_sectors):
        selected = sector_index == sector_number
        if not np.any(selected):
            return -np.inf
        sector_model = batman_transit_flux(
            np.asarray(phase_days)[selected],
            period_days,
            rp,
            a_rs,
            b,
            q1,
            q2,
            1.0,
            eccentricity=eccentricity,
            omega_deg=omega_deg,
            exposure_seconds=float(exposure_seconds_by_sector[sector_number]),
        )
        if sector_model is None:
            return -np.inf
        model[selected] = baselines[sector_number] * sector_model

    # NUMERICAL_GUARD: jitter is fitted in log space, so exp(...) is
    # always positive; adding in quadrature with the reported uncertainty
    # ensures the total variance is never smaller than the measurement noise.
    jitter = math.exp(log_jitter)
    ivar = 1.0 / (flux_err**2 + jitter**2)
    residual = flux - model
    chi2 = float(np.sum(residual**2 * ivar))
    logdet = float(np.sum(np.log(2.0 * math.pi / ivar)))
    return float(-0.5 * (chi2 + logdet))


def _map_optimize(
    phase_days: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    ephemeris: Dict[str, Any],
    rho_prior_solar: float,
    rho_prior_log10_sigma: float,
    eccentric: bool,
    start: np.ndarray,
    ldtk_prior: Optional[Dict[str, Any]] = None,
    sector_index: Optional[np.ndarray] = None,
    exposure_seconds_by_sector: Optional[np.ndarray] = None,
    n_sectors: int = 1,
) -> np.ndarray:
    """Find the maximum *a posteriori* point via L-BFGS-B.

    The MAP point is used to initialise the MCMC walkers or nested-sampling
    live points.  The optimisation wraps :func:`_neg_log_posterior` as the
    objective and applies the same bounding boxes as :func:`_log_prior`.

    .. note::

        If the MAP point lies at a boundary or the optimiser fails to
        converge, the initial parameter vector is returned unmodified.
        The downstream sampler is robust to sub-optimal initialisation,
        though convergence may require more steps.

    Parameters
    ----------
    phase_days, flux, flux_err : ndarray
        Native-cadence data.
    ephemeris : dict
        Transit ephemeris.
    rho_prior_solar : float
        Prior mean density (solar units).
    rho_prior_log10_sigma : float
        Prior density width (:math:`\\log_{10}` space).
    eccentric : bool
        Whether eccentricity parameters are active.
    start : ndarray
        Initial guess.
    ldtk_prior : dict, optional
        LDTk prior for limb darkening.
    sector_index, exposure_seconds_by_sector, n_sectors
        Native-cadence sector descriptors.

    Returns
    -------
    ndarray
        MAP parameter vector.
    """
    from scipy.optimize import minimize

    if start.shape != (_parameter_count(eccentric, n_sectors),):
        raise ValueError("transit MAP start has an invalid sector layout")
    bounds = [(0.001, 0.3), (-2.0, 1.5), (0.0, 1.19)]
    bounds.extend([(0.99, 1.01)] * n_sectors)
    bounds.extend([(-12.0, -2.0), (0.01, 0.99), (0.01, 0.99)])
    if eccentric:
        bounds.extend([(-1.0, 1.0), (-1.0, 1.0)])
    offsets = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.01, 0.2, -0.1, 0.05, -0.05, 0.1),
        (-0.01, -0.2, 0.1, -0.05, 0.05, -0.1),
    )
    jitter_starts = np.array([0.0, -2.0, -4.0, -6.0, -8.0])
    best_objective = np.inf
    best_point = start

    def optimizer_objective(candidate_theta: np.ndarray) -> float:
        """Keep numerical optimization finite while MCMC retains strict support."""
        value = _neg_log_posterior(
            candidate_theta,
            phase_days,
            flux,
            flux_err,
            ephemeris,
            rho_prior_solar,
            rho_prior_log10_sigma,
            eccentric,
            ldtk_prior,
            sector_index=sector_index,
            exposure_seconds_by_sector=exposure_seconds_by_sector,
            n_sectors=n_sectors,
        )
        return value if math.isfinite(value) else 1e100

    baseline_stop = 3 + n_sectors
    for rp_delta, rho_delta, impact_delta, q1_delta, q2_delta, eccentric_delta in offsets:
        for jitter_delta in jitter_starts:
            candidate = start.copy()
            candidate[0] += rp_delta
            candidate[1] += rho_delta
            candidate[2] += impact_delta
            candidate[baseline_stop] += jitter_delta
            candidate[baseline_stop + 1] += q1_delta
            candidate[baseline_stop + 2] += q2_delta
            if eccentric:
                candidate[baseline_stop + 3] += eccentric_delta
                candidate[baseline_stop + 4] += eccentric_delta
            result = minimize(
                optimizer_objective,
                candidate,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 400, "ftol": 1e-9},
            )
            if np.isfinite(result.fun) and result.fun < best_objective:
                best_objective = float(result.fun)
                best_point = np.asarray(result.x, dtype=float)
    return best_point


def _quantile_summary(chain: np.ndarray) -> Dict[str, float]:
    """Compute median, 16th, and 84th percentiles for 1-D posterior samples.

    These are the standard posterior summary quantiles for symmetric
    (approximately Gaussian) marginal posteriors.  The ``plus`` and ``minus``
    fields provide the 1-sigma-equivalent interval half-widths.

    Parameters
    ----------
    chain : ndarray
        1-D array of posterior draws.

    Returns
    -------
    dict
        Keys ``p16``, ``median``, ``p84``, ``plus``, ``minus``.
    """
    quantiles = np.quantile(chain, [0.16, 0.50, 0.84])
    return {
        "p16": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p84": float(quantiles[2]),
        "plus": float(quantiles[2] - quantiles[1]),
        "minus": float(quantiles[1] - quantiles[0]),
    }


def _thin_joint_posterior(chain: np.ndarray, target_samples: int = 1000) -> np.ndarray:
    """Thin a joint posterior matrix with a uniform stride, preserving covariance.

    Every retained row is an actual joint draw from the MCMC chain; no
    cross-row pairing occurs. This replaces per-column chunk medians, which
    synthesized physically impossible parameter combinations by breaking the
    joint ``(rp/rs, a/Rs, b, q1, q2)`` probability manifold.

    Args:
        chain: 2-D array of shape ``(n_samples, n_params)`` of joint draws.
        target_samples: Maximum number of joint rows to retain.

    Returns:
        Thinned array of shape ``(min(n_samples, target_samples), n_params)``.
    """
    samples = np.asarray(chain, dtype=float)
    n_rows = samples.shape[0]
    if n_rows <= target_samples:
        return samples
    step = max(1, n_rows // target_samples)
    return samples[::step][:target_samples]


def _posterior_summaries(
    chain: np.ndarray,
    ephemeris: Dict[str, Any],
    eccentric: bool,
    n_sectors: int = 1,
    sector_labels: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Summarize sampled and derived transit parameters from an equal-weight chain.

    For every sample in the flattened post-burn-in chain, this function
    computes derived quantities that depend on more than one fitted
    parameter:

    - ``a_rs``: scaled semi-major axis from Kepler's law
      (:func:`stellar_density_a_rs`)
    - ``inclination_deg``: orbital inclination from impact parameter
      (:func:`inclination_deg_from_impact_parameter`)
    - ``rho_star_solar``: :math:`10^{\\log_{10}\\rho}` in solar units
    - ``u1, u2``: standard quadratic limb-darkening coefficients
      transformed from ``(q1, q2)`` via Kipping (2013).

    Astrophysical Rationale
    -----------------------
    Derived quantities are computed at the sample level rather than from
    summary statistics to capture covariances correctly.  For example,
    ``a_rs`` depends on both ``log_rho_star`` and the period, and
    ``inclination_deg`` couples ``a_rs`` (via the conjunction distance),
    ``b``, and eccentricity.
    """
    names = _parameter_names(eccentric, n_sectors, sector_labels)
    if chain.ndim != 2 or chain.shape[1] != len(names):
        raise ValueError("transit posterior chain has an invalid sector layout")
    posteriors: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(names):
        posteriors[name] = _quantile_summary(chain[:, index])

    rp_samples = chain[:, 0]
    rho_samples = 10.0 ** chain[:, 1]
    b_samples = chain[:, 2]
    q1_index = 4 + n_sectors
    q2_index = 5 + n_sectors
    q1_samples = chain[:, q1_index]
    q2_samples = chain[:, q2_index]
    a_rs_samples = np.array(
        [stellar_density_a_rs(rho, ephemeris["period_days"]) for rho in rho_samples]
    )
    if eccentric:
        se_cos_samples = chain[:, 6 + n_sectors]
        se_sin_samples = chain[:, 7 + n_sectors]
        eccentricity_samples = se_cos_samples**2 + se_sin_samples**2
        omega_samples = np.degrees(np.arctan2(se_sin_samples, se_cos_samples))
    else:
        eccentricity_samples = np.zeros_like(rp_samples)
        omega_samples = np.full_like(rp_samples, 90.0)
    conjunction_distance_samples = np.asarray(
        [
            conjunction_distance_a_rs(a_rs, eccentricity_value, omega_value)
            for a_rs, eccentricity_value, omega_value in zip(
                a_rs_samples, eccentricity_samples, omega_samples
            )
        ],
        dtype=float,
    )
    projection_samples = b_samples / conjunction_distance_samples
    # NUMERICAL_GUARD: ``arccos`` only accepts the physical projection range.
    # Retain the edge-on mapping for legacy posterior summaries, but expose how
    # often a sampled impact parameter exceeded the conjunction distance.
    inclination_clipped = np.isfinite(projection_samples) & (
        (projection_samples < 0.0) | (projection_samples > 1.0)
    )
    inc_samples = np.degrees(np.arccos(np.clip(projection_samples, 0.0, 1.0)))
    area_ppm = (rp_samples**2) * 1e6
    # COVARIANCE_GUARD: thin the joint parameter matrix with a uniform stride
    # so every evaluated tuple is an actual posterior draw. Independent
    # per-column chunk medians would pair samples from different iterations,
    # synthesizing unphysical (rp/rs, a/Rs, b, q1, q2) combinations.
    joint_depth_columns = np.column_stack(
        [
            rp_samples,
            a_rs_samples,
            b_samples,
            q1_samples,
            q2_samples,
            eccentricity_samples,
            omega_samples,
        ]
    )
    depth_values = []
    for median_rp, median_a, median_b, median_q1, median_q2, median_eccentricity, median_omega in _thin_joint_posterior(
        joint_depth_columns, target_samples=1000
    ):
        model = batman_transit_flux(
            np.array([0.0]),
            ephemeris["period_days"],
            float(median_rp),
            float(median_a),
            float(median_b),
            float(median_q1),
            float(median_q2),
            1.0,
            eccentricity=float(median_eccentricity),
            omega_deg=float(median_omega),
            # ASTROPHYSICAL_GUARD: evaluate the instantaneous mid-transit depth
            # at native TESS cadence. The default 480 s supersampling window
            # smears short transits and biases the derived depth low relative
            # to the geometric area ratio (rp/rs)^2.
            exposure_seconds=EXPTIME_SECONDS,
        )
        if model is None or model.shape != (1,) or not np.isfinite(model[0]):
            raise RuntimeError(
                "batman failed while evaluating posterior-derived mid-transit depths"
            )
        depth_values.append(1.0 - model[0])
    depth_ppm_samples = np.asarray(depth_values) * 1e6

    if not (
        np.all(np.isfinite(q1_samples))
        and np.all(np.isfinite(q2_samples))
        and np.all((q1_samples >= 0.0) & (q1_samples <= 1.0))
        and np.all((q2_samples >= 0.0) & (q2_samples <= 1.0))
    ):
        raise ValueError("transit posterior q1 and q2 samples must be finite values in [0, 1]")
    sqrt_q1 = np.sqrt(q1_samples)
    u1_samples = 2.0 * sqrt_q1 * q2_samples
    u2_samples = sqrt_q1 * (1.0 - 2.0 * q2_samples)

    inclination_summary = _quantile_summary(inc_samples)
    inclination_summary["conjunction_distance_clip_fraction"] = float(
        np.mean(inclination_clipped)
    )
    posteriors["inclination_deg"] = inclination_summary
    posteriors["a_rs"] = _quantile_summary(a_rs_samples)
    posteriors["conjunction_distance_a_rs"] = _quantile_summary(conjunction_distance_samples)
    posteriors["rho_star_solar"] = _quantile_summary(rho_samples)
    posteriors["area_ratio_ppm"] = _quantile_summary(area_ppm)
    posteriors["mid_transit_depth_ppm"] = _quantile_summary(depth_ppm_samples)
    posteriors["u1"] = _quantile_summary(u1_samples)
    posteriors["u2"] = _quantile_summary(u2_samples)
    if eccentric:
        posteriors["eccentricity"] = _quantile_summary(eccentricity_samples)
        posteriors["omega_deg"] = _quantile_summary(omega_samples)
    return posteriors


def _resample_weighted_posterior(
    samples: np.ndarray, weights: np.ndarray, seed: int
) -> Tuple[np.ndarray, float]:
    """Systematically resample normalized nested-sampling weights with a fixed seed.

    This uses the ``searchsorted`` method (systematic resampling) that
    preserves the equal-weight property better than multinomial draws and
    guarantees a deterministic outcome for a given seed (Handley et al.
    2015).

    Mathematical Formulation
    ------------------------
    After normalizing weights :math:`w_i` to sum to unity, the systematic
    resampling algorithm draws :math:`N` new indices by

    .. math::

        u_k = \\frac{k + \\xi}{N}, \\quad k = 0, \\dots, N-1,

    where :math:`\\xi \\sim U(0, 1)`, and assigns each new draw to the
    original sample whose cumulative weight first exceeds :math:`u_k`.

    Parameters
    ----------
    samples : ndarray
        Posterior samples, shape ``(n_live, n_params)``.
    weights : ndarray
        Nested-sampling weights of shape ``(n_live,)``.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    resampled : ndarray
        Equal-weight posterior draws.
    ess : float
        Effective sample size :math:`1 / \\sum w_i^2`.
    """
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if samples.ndim != 2 or weights.shape != (samples.shape[0],):
        raise ValueError("nested samples and weights have incompatible shapes")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("nested samples and weights must be finite with non-negative weights")
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise ValueError("nested weights must have positive total weight")
    weights = weights / total_weight
    positions = (np.random.default_rng(seed).random() + np.arange(weights.size)) / weights.size
    indices = np.searchsorted(np.cumsum(weights), positions, side="right")
    indices = np.minimum(indices, samples.shape[0] - 1)
    return samples[indices], float(1.0 / np.sum(weights**2))


def _make_dynesty_prior_transform(
    rho_prior_solar: float,
    rho_prior_log10_sigma: float,
    eccentric: bool,
    ldtk_prior: Optional[Dict[str, Any]],
    n_sectors: int = 1,
):
    """Create a normalized prior transform for dynesty's likelihood-only API.

    dynesty requires sampling from the unit hypercube and applying a
    prior transform :math:`T: [0, 1]^d \\to \\Theta`.  This function
    builds that transform for the transit parameter space.

    Each parameter is mapped from a unit-uniform variate via inverse-CDF
    sampling of a truncated Gaussian distribution, except for the
    Kipping ``(q1, q2)`` parameters which use a gridded inverse-CDF
    constructed from the product of the LDTk Gaussian likelihood on
    ``(u1, u2)`` and the uniform Dirichlet prior on ``(q1, q2)``
    (Kipping 2013).  This ensures that the evidence :math:`\\ln Z`
    reported by dynesty is computed under a proper prior.

    Mathematical Formulation
    ------------------------
    For a truncated Gaussian with mean :math:`\\mu`, standard deviation
    :math:`\\sigma`, and bounds :math:`[a, b]`, the prior transform is

    .. math::

        T(u) = \\mu + \\sigma\\,\\Phi^{-1}\\!\\bigl(
            \\Phi(\\alpha) + u\\,[\\Phi(\\beta) - \\Phi(\\alpha)]
        \\bigr),

    where :math:`\\alpha = (a - \\mu)/\\sigma`,
    :math:`\\beta = (b - \\mu)/\\sigma`, :math:`\\Phi` is the standard
    normal CDF, and :math:`\\Phi^{-1}` its inverse (percent-point
    function).

    Parameters
    ----------
    rho_prior_solar : float
        Prior mean density in solar units.
    rho_prior_log10_sigma : float
        Prior density width in :math:`\\log_{10}` space.
    eccentric : bool
        Whether eccentricity components are active.
    ldtk_prior : dict or None
        LDTk limb-darkening prior.
    n_sectors : int, optional
        Number of observation sectors.

    Returns
    -------
    callable
        A function ``ptform(u)`` that maps ``u`` in ``[0, 1]^d`` to the
        physical parameter vector in the order expected by
        :func:`_unpack_theta`.
    """
    from scipy.special import ndtr, ndtri

    if rho_prior_solar <= 0 or rho_prior_log10_sigma <= 0:
        raise ValueError("stellar density and density uncertainty must be positive")

    def truncated_normal(unit_value: float, mean: float, sigma: float, lower: float, upper: float) -> float:
        clipped = float(np.clip(unit_value, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
        lower_cdf = ndtr((lower - mean) / sigma)
        upper_cdf = ndtr((upper - mean) / sigma)
        return float(mean + sigma * ndtri(lower_cdf + clipped * (upper_cdf - lower_cdf)))

    def truncated_half_cauchy_log_jitter(unit_value: float) -> float:
        clipped = float(np.clip(unit_value, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
        lower = math.exp(JITTER_LOG_LOWER)
        upper = math.exp(JITTER_LOG_UPPER)
        lower_cdf = 2.0 / math.pi * math.atan(lower / JITTER_HALF_CAUCHY_SCALE)
        upper_cdf = 2.0 / math.pi * math.atan(upper / JITTER_HALF_CAUCHY_SCALE)
        jitter = JITTER_HALF_CAUCHY_SCALE * math.tan(
            0.5 * math.pi * (lower_cdf + clipped * (upper_cdf - lower_cdf))
        )
        return float(math.log(jitter))

    q_transform = None
    if ldtk_prior is not None:
        # The emcee path uses uniform Kipping parameters multiplied by the LDTk
        # density. Build its equivalent normalized two-dimensional prior once,
        # then use inverse-CDF sampling so it remains a prior for the evidence.
        from scipy.integrate import cumulative_trapezoid, trapezoid

        q_grid = np.linspace(0.01, 0.99, 513)
        q1_grid, q2_grid = np.meshgrid(q_grid, q_grid, indexing="ij")
        root_q1 = np.sqrt(q1_grid)
        u1_grid = 2.0 * root_q1 * q2_grid
        u2_grid = root_q1 * (1.0 - 2.0 * q2_grid)
        log_density = -0.5 * ((u1_grid - ldtk_prior["u1"]) / ldtk_prior["u1_err"]) ** 2
        log_density += -0.5 * ((u2_grid - ldtk_prior["u2"]) / ldtk_prior["u2_err"]) ** 2
        density = np.exp(log_density - float(np.max(log_density)))
        # COMPATIBILITY: np.trapz was removed in NumPy 2.0+; scipy.integrate
        # trapezoid is the drop-in replacement with identical numerics.
        marginal = trapezoid(density, q_grid, axis=1)
        q1_cdf = cumulative_trapezoid(marginal, q_grid, initial=0.0)
        q1_cdf /= q1_cdf[-1]

        def q_transform(q1_unit: float, q2_unit: float) -> Tuple[float, float]:
            q1_value = float(np.interp(q1_unit, q1_cdf, q_grid))
            root_q1_value = math.sqrt(q1_value)
            u1_values = 2.0 * root_q1_value * q_grid
            u2_values = root_q1_value * (1.0 - 2.0 * q_grid)
            conditional = np.exp(
                -0.5 * ((u1_values - ldtk_prior["u1"]) / ldtk_prior["u1_err"]) ** 2
                -0.5 * ((u2_values - ldtk_prior["u2"]) / ldtk_prior["u2_err"]) ** 2
                - float(np.max(log_density))
            )
            q2_cdf = cumulative_trapezoid(conditional, q_grid, initial=0.0)
            q2_cdf /= q2_cdf[-1]
            return q1_value, float(np.interp(q2_unit, q2_cdf, q_grid))

    def prior_transform(unit_cube: np.ndarray) -> np.ndarray:
        unit_cube = np.asarray(unit_cube, dtype=float)
        expected_dimensions = _parameter_count(eccentric, n_sectors)
        if unit_cube.shape != (expected_dimensions,) or not np.all(np.isfinite(unit_cube)):
            raise ValueError("dynesty prior transform received an invalid unit-cube point")
        if np.any(unit_cube < 0.0) or np.any(unit_cube > 1.0):
            raise ValueError("dynesty prior transform requires values in [0, 1]")
        theta = np.empty(expected_dimensions, dtype=float)
        theta[0] = 0.001 + unit_cube[0] * (0.3 - 0.001)
        theta[1] = truncated_normal(
            unit_cube[1], math.log10(rho_prior_solar), rho_prior_log10_sigma, -2.0, 1.5
        )
        theta[2] = unit_cube[2] * 1.2
        baseline_stop = 3 + n_sectors
        theta[3:baseline_stop] = 0.99 + unit_cube[3:baseline_stop] * 0.02
        theta[baseline_stop] = truncated_half_cauchy_log_jitter(unit_cube[baseline_stop])
        q1_index = baseline_stop + 1
        q2_index = baseline_stop + 2
        if q_transform is None:
            theta[q1_index] = 0.01 + unit_cube[q1_index] * 0.98
            theta[q2_index] = 0.01 + unit_cube[q2_index] * 0.98
        else:
            theta[q1_index], theta[q2_index] = q_transform(
                unit_cube[q1_index], unit_cube[q2_index]
            )
        if eccentric:
            radius = math.sqrt(unit_cube[baseline_stop + 3])
            angle = 2.0 * math.pi * unit_cube[baseline_stop + 4]
            theta[baseline_stop + 3] = radius * math.cos(angle)
            theta[baseline_stop + 4] = radius * math.sin(angle)
        return theta

    return prior_transform


def _synthetic_transit_table(
    ephemeris: Dict[str, Any], rng_seed: int = 5
) -> Dict[str, np.ndarray]:
    """Deterministic test-only transit light curve.

    The injected radius is derived from the ephemeris depth so the synthetic
    signal is self-consistent with the fitter's initialization.

    .. warning::

        This function produces **synthetic** photometry with hardcoded
        limb-darkening coefficients (q1=0.35, q2=0.3) and 80 ppm Gaussian
        noise.  It exists for development and testing only.
    """
    rng = np.random.default_rng(seed=rng_seed)
    cadence_days = 120.0 / SECONDS_PER_DAY
    duration_days = max(12.0, 4.0 * float(ephemeris["period_days"]))
    time = np.arange(0.0, duration_days, cadence_days)
    phase_days = (
        (time - ephemeris["epoch_btjd"] + 0.5 * ephemeris["period_days"])
        % ephemeris["period_days"]
    ) - 0.5 * ephemeris["period_days"]
    injected_rp = math.sqrt(max(float(ephemeris["depth_ppm"]) * 1e-6, 1e-8))
    rho_solar = 1.0
    a_rs = stellar_density_a_rs(rho_solar, ephemeris["period_days"])
    model = batman_transit_flux(
        phase_days,
        ephemeris["period_days"],
        injected_rp,
        a_rs,
        0.3,
        0.35,
        0.3,
        1.0,
        exposure_seconds=EXPTIME_SECONDS,
    )
    if model is None:
        raise RuntimeError("batman failed while generating a synthetic transit fixture")
    flux = np.asarray(model)
    flux = flux + rng.normal(0.0, 80e-6, size=time.shape)
    flux_err = np.full_like(flux, 80e-6)
    sector_values = np.ones(time.size, dtype=int)
    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "sector": sector_values,
    }


def _mcmc_convergence_diagnostics(
    raw_chain: np.ndarray, tau_values: Optional[np.ndarray], parameter_names: Sequence[str]
) -> Dict[str, Any]:
    """Assess ensemble-chain mixing without claiming independent-chain validation.

    Computes the split-:math:`\\hat{R}` statistic (Gelman-Rubin, Gelman et
    al. 2014) and the integrated autocorrelation time via ``emcee``'s
    built-in estimator.  The effective sample size is

    .. math::

        \\text{ESS} = \\frac{N_{\\rm steps} \\times N_{\\rm walkers}}{\\tau},

    where :math:`\\tau` is the integrated autocorrelation time.

    .. warning::

        Ensemble diagnostics are **not** a substitute for independently
        initialised chains or a calibrated correlated-noise likelihood.
        The ``scientific_posterior_eligible`` flag is always ``False``
        because convergence under i.i.d. Gaussian likelihood with a single
        ensemble does not demonstrate that the posterior is physically
        meaningful.  This is a :ref:`SCIENTIFIC_BOUNDARY` — a calibrated
        model-comparison framework is required for scientific claims.

    Parameters
    ----------
    raw_chain : ndarray
        Post-burn-in ensemble chain of shape ``(n_steps, n_walkers,
        n_params)``.
    tau_values : ndarray or None
        Per-parameter integrated autocorrelation times.
    parameter_names : sequence of str
        Parameter names for the diagnostic keys.

    Returns
    -------
    dict
        Keys: ``split_r_hat``, ``effective_samples``,
        ``chain_length_over_tau``, ``thresholds``, ``basic_mixing_passed``,
        ``status``, ``scientific_posterior_eligible``, ``reason``.
    """
    chain = np.asarray(raw_chain, dtype=float)
    if chain.ndim != 3 or chain.shape[0] < 4 or chain.shape[1] < 2 or not np.all(np.isfinite(chain)):
        return {
            "status": "not-demonstrated",
            "scientific_posterior_eligible": False,
            "reason": "insufficient finite post-burn-in ensemble chain for convergence diagnostics",
        }
    half = chain.shape[0] // 2
    split = np.concatenate((chain[:half], chain[-half:]), axis=1)
    within = np.mean(np.var(split, axis=0, ddof=1), axis=0)
    means = np.mean(split, axis=0)
    between = half * np.var(means, axis=0, ddof=1)
    variance_plus = ((half - 1.0) / half) * within + between / half
    rhat = np.sqrt(np.divide(variance_plus, within, out=np.full_like(within, np.nan), where=within > 0))
    tau = np.asarray(tau_values, dtype=float) if tau_values is not None else np.full(chain.shape[2], np.nan)
    ess = np.divide(
        chain.shape[0] * chain.shape[1], tau, out=np.full(chain.shape[2], np.nan), where=tau > 0
    )
    ratios = np.divide(chain.shape[0], tau, out=np.full(chain.shape[2], np.nan), where=tau > 0)
    diagnostics = {
        "split_r_hat": {parameter_names[index]: float(value) if np.isfinite(value) else None for index, value in enumerate(rhat)},
        "effective_samples": {parameter_names[index]: float(value) if np.isfinite(value) else None for index, value in enumerate(ess)},
        "chain_length_over_tau": {parameter_names[index]: float(value) if np.isfinite(value) else None for index, value in enumerate(ratios)},
        "thresholds": {"split_r_hat_max": 1.01, "effective_samples_min": 400.0, "chain_length_over_tau_min": 50.0},
    }
    basic_pass = bool(np.all(rhat <= 1.01) and np.all(ess >= 400.0) and np.all(ratios >= 50.0))
    diagnostics.update({
        "basic_mixing_passed": basic_pass,
        "status": "basic-ensemble-mixing-passed" if basic_pass else "not-demonstrated",
        "scientific_posterior_eligible": False,
        "reason": "ensemble-walker diagnostics are not a substitute for independently initialized chains or a calibrated correlated-noise likelihood",
    })
    return diagnostics


def run_mcmc_transit_fit(
    workspace: CandidateWorkspace,
    n_samples: int = CPU_EMCEE_PRODUCTION,
    eccentric: bool = False,
    n_walkers: Optional[int] = None,
    burn_in: Optional[int] = None,
    seed: int = 5,
    signal: Optional[str] = None,
    use_ldtk_prior: bool = False,
    sampler: str = "auto",
    device: str = "auto",
    detrending_method: Optional[str] = None,
    dlogz_tolerance: float = 0.5,
    n_jobs: int = 1,
    progress: bool = False,
    resume: Optional[str] = None,
    checkpoint_interval: int = 250,
    gpu_num_warmup: int = GPU_NUTS_WARMUP,
    gpu_num_samples: int = GPU_NUTS_SAMPLES,
    gpu_target_accept_prob: float = GPU_NUTS_TARGET_ACCEPT_PROB,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Fit candidate photometry with automatic CUDA NUTS or CPU fallback.

    ``sampler='auto'`` and ``device='auto'`` select the JAX/NumPyro GPU path
    when its 64-bit CUDA stack is available. Missing optional dependencies,
    absent CUDA hardware, ``--device cpu``, emcee-only checkpoint resume, and
    multiprocessing requests retain the historical batman/emcee path. The
    dynesty mode remains an explicitly selected CPU nested sampler.
    """
    if sampler not in ("auto", "emcee", "numpyro", "dynesty"):
        raise ValueError("sampler must be one of: auto, emcee, numpyro, dynesty")
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError("device must be one of: auto, cpu, gpu")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if (
        not isinstance(gpu_num_warmup, int)
        or isinstance(gpu_num_warmup, bool)
        or gpu_num_warmup <= 0
        or not isinstance(gpu_num_samples, int)
        or isinstance(gpu_num_samples, bool)
        or gpu_num_samples <= 0
        or not isinstance(gpu_target_accept_prob, (float, int))
        or not math.isfinite(float(gpu_target_accept_prob))
        or not 0.0 < float(gpu_target_accept_prob) < 1.0
    ):
        raise ValueError("GPU NUTS controls must be positive with target acceptance in (0, 1)")

    signal = validate_signal_suffix(signal)
    if sampler == "dynesty":
        return _run_dynesty_transit_fit(
            workspace,
            n_samples=n_samples,
            eccentric=eccentric,
            seed=seed,
            signal=signal,
            use_ldtk_prior=use_ldtk_prior,
            detrending_method=detrending_method,
            dlogz_tolerance=dlogz_tolerance,
        )
    if sampler == "emcee" or device == "cpu":
        return _fit_emcee_candidate_transit_fit(
            workspace,
            n_samples=n_samples,
            eccentric=eccentric,
            n_walkers=n_walkers,
            burn_in=burn_in,
            seed=seed,
            signal=signal,
            use_ldtk_prior=use_ldtk_prior,
            detrending_method=detrending_method,
            n_jobs=n_jobs,
            progress=progress,
            resume=resume,
            checkpoint_interval=checkpoint_interval,
            progress_callback=progress_callback,
            backend_fallback_reason=(
                "CPU requested explicitly" if device == "cpu" else "emcee sampler selected explicitly"
            ),
        )
    if resume is not None or n_jobs > 1:
        return _fit_emcee_candidate_transit_fit(
            workspace,
            n_samples=n_samples,
            eccentric=eccentric,
            n_walkers=n_walkers,
            burn_in=burn_in,
            seed=seed,
            signal=signal,
            use_ldtk_prior=use_ldtk_prior,
            detrending_method=detrending_method,
            n_jobs=n_jobs,
            progress=progress,
            resume=resume,
            checkpoint_interval=checkpoint_interval,
            progress_callback=progress_callback,
            backend_fallback_reason=(
                "GPU NUTS does not support emcee checkpoints or multiprocessing worker pools"
            ),
        )
    try:
        stack = _load_jax_gpu_stack()
    except _GpuBackendUnavailable as exc:
        return _fit_emcee_candidate_transit_fit(
            workspace,
            n_samples=n_samples,
            eccentric=eccentric,
            n_walkers=n_walkers,
            burn_in=burn_in,
            seed=seed,
            signal=signal,
            use_ldtk_prior=use_ldtk_prior,
            detrending_method=detrending_method,
            n_jobs=n_jobs,
            progress=progress,
            resume=resume,
            checkpoint_interval=checkpoint_interval,
            backend_fallback_reason=str(exc),
        )
    return _run_numpyro_candidate_transit_fit(
        workspace,
        eccentric=eccentric,
        seed=seed,
        signal=signal,
        use_ldtk_prior=use_ldtk_prior,
        detrending_method=detrending_method,
        num_warmup=gpu_num_warmup,
        num_samples=gpu_num_samples,
        target_accept_prob=float(gpu_target_accept_prob),
        progress=progress,
        stack=stack,
    )


def _fit_emcee_candidate_transit_fit(
    workspace: CandidateWorkspace,
    n_samples: int = CPU_EMCEE_PRODUCTION,
    eccentric: bool = False,
    n_walkers: Optional[int] = None,
    burn_in: Optional[int] = None,
    seed: int = 5,
    signal: Optional[str] = None,
    use_ldtk_prior: bool = False,
    detrending_method: Optional[str] = None,
    n_jobs: int = 1,
    progress: bool = False,
    resume: Optional[str] = None,
    checkpoint_interval: int = 250,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    backend_fallback_reason: Optional[str] = None,
) -> Path:
    """Run the candidate-local batman/emcee transit fit.

    This is the top-level entry point for transit fitting.  It loads the
    candidate light curve, stellar parameters, and ephemeris; constructs the
    prior and likelihood; performs MAP optimisation; initialises ensemble
    walkers; runs the affine-invariant MCMC chain (Goodman & Weare 2010)
    via ``emcee``; computes posterior summaries and convergence diagnostics;
    and serialises the results to the candidate's ``outputs/`` directory.

    The native-cadence likelihood uses a Gaussian independent-noise model
    with a fitted jitter term and per-sector baseline parameters.  The
    resulting posterior is **exploratory** and explicitly does not qualify
    as a validated posterior or a planet-claim probability.

    Parameters
    ----------
    workspace : CandidateWorkspace
        The candidate workspace.
    n_samples : int, optional
        Number of production MCMC steps per walker (default 2500).
    eccentric : bool, optional
        Whether to fit eccentric orbit components (default False).
    n_walkers : int, optional
        Number of ensemble walkers; auto-set if None.
    burn_in : int, optional
        Number of burn-in steps; auto-set if None.
    seed : int, optional
        RNG seed for reproducibility (default 5).
    signal : str, optional
        Signal suffix for multi-signal candidates.
    use_ldtk_prior : bool, optional
        Whether to apply the LDTk limb-darkening prior (default False).
    detrending_method : str, optional
        Detrending method for light curve retrieval.
    n_jobs : int, optional
        Number of parallel worker processes for ensemble likelihood evaluation
        (``pool`` via ``multiprocessing``; default 1, i.e. serial).
    progress : bool, optional
        Report per-step progress on stderr (default False).
    resume : str, optional
        Path to a previous checkpoint ``.npz`` file to resume from
        (only valid for emcee; default None).
    checkpoint_interval : int, optional
        Number of steps between intermediate chain saves (default 250).
    backend_fallback_reason : str, optional
        Reason automatic device selection retained the CPU backend.

    Returns
    -------
    Path
        Path to the written ``mcmc_transit_fit.json`` artifact.

    Raises
    ------
    RuntimeError
        If the required ``batman-package`` dependency is unavailable, or
        photometry, ephemeris, or stellar parameters are synthetic or missing.
    """
    signal = validate_signal_suffix(signal)
    suffix = f".{signal.lstrip('.')}" if signal else ""
    _require_batman()
    import emcee

    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ephemeris = load_transit_ephemeris(workspace, signal=signal)
    if ephemeris["source"] == "synthetic-demo" or any(
        value == "synthetic-demo" for value in ephemeris.get("field_sources", {}).values()
    ):
        raise RuntimeError("transit fitting requires a complete candidate-derived transit ephemeris")
    stellar = load_stellar_parameters(workspace)
    if stellar["source"] != "candidate-data":
        raise RuntimeError("transit fitting requires complete candidate-derived stellar parameters")
    density_prior = _stellar_density_prior(stellar)
    rho_prior_solar = density_prior["rho_solar"]
    rho_prior_log10_sigma = density_prior["log10_sigma"]
    ldtk_prior = _load_ldtk_prior(workspace) if use_ldtk_prior else None

    table = load_light_curve_table(
        workspace,
        max_points=None,
        require_raw_provenance=True,
        detrending_method=detrending_method,
    )
    if table is None:
        raise RuntimeError("transit fitting requires observed candidate photometry")
    source = "candidate-data"

    try:
        (
            phase_days,
            native_flux,
            native_error,
            sector_index,
            sector_labels,
            exposure_seconds_by_sector,
        ) = _native_transit_window_data(
            table, ephemeris
        )
    except ValueError as exc:
        raise RuntimeError("candidate photometry cannot support a transit fit") from exc

    depth_ppm = float(ephemeris["depth_ppm"])
    n_sectors = len(sector_labels)
    jitter_prior = _jitter_prior_assumption()
    scatter = float(np.std(native_flux - np.median(native_flux)))
    start = _initial_fit_parameters(
        depth_ppm, rho_prior_solar, scatter, eccentric, n_sectors=n_sectors
    )
    map_point = _map_optimize(
        phase_days,
        native_flux,
        native_error,
        ephemeris,
        rho_prior_solar,
        rho_prior_log10_sigma,
        eccentric,
        start,
        ldtk_prior,
        sector_index=sector_index,
        exposure_seconds_by_sector=exposure_seconds_by_sector,
        n_sectors=n_sectors,
    )

    ndim = int(map_point.size)
    if n_walkers is None:
        # Keep the requested robust default while respecting emcee's
        # red-blue move requirement for high-dimensional multi-sector fits.
        n_walkers = max(CPU_EMCEE_WALKERS, 2 * ndim)
    if burn_in is None:
        burn_in = CPU_EMCEE_BURN_IN
    rng = np.random.default_rng(seed=seed)

    if resume is None:
        _resume_done = False

        def valid_walker_start(candidate_theta: np.ndarray) -> bool:
            return math.isfinite(
                _neg_log_posterior(
                    candidate_theta,
                    phase_days,
                    native_flux,
                    native_error,
                    ephemeris,
                    rho_prior_solar,
                    rho_prior_log10_sigma,
                    eccentric,
                    ldtk_prior,
                    sector_index=sector_index,
                    exposure_seconds_by_sector=exposure_seconds_by_sector,
                    n_sectors=n_sectors,
                )
            )

        p0 = np.empty((n_walkers, ndim), dtype=float)
        for walker_index in range(n_walkers):
            accepted_start = None
            for center in (map_point, start):
                for _attempt in range(100):
                    candidate_theta = center + 1e-3 * rng.normal(size=ndim)
                    candidate_theta[2] = np.clip(candidate_theta[2], 0.0, 1.1)
                    baseline_stop = 3 + n_sectors
                    candidate_theta[3:baseline_stop] = np.clip(
                        candidate_theta[3:baseline_stop], 0.995, 1.005
                    )
                    candidate_theta[baseline_stop + 1] = np.clip(
                        candidate_theta[baseline_stop + 1], 0.01, 0.99
                    )
                    candidate_theta[baseline_stop + 2] = np.clip(
                        candidate_theta[baseline_stop + 2], 0.01, 0.99
                    )
                    if eccentric:
                        eccentric_radius = math.hypot(
                            candidate_theta[baseline_stop + 3], candidate_theta[baseline_stop + 4]
                        )
                        if eccentric_radius >= 0.99:
                            candidate_theta[baseline_stop + 3:baseline_stop + 5] *= 0.99 / eccentric_radius
                    if valid_walker_start(candidate_theta):
                        accepted_start = candidate_theta
                        break
                if accepted_start is not None:
                    break
            if accepted_start is None:
                raise RuntimeError("could not initialize a physically valid transit-fit walker")
            p0[walker_index] = accepted_start
        saved_rstate = np.random.RandomState(seed).get_state()
        remaining_steps = burn_in + n_samples
    else:
        ck = np.load(resume, allow_pickle=True)
        saved_chain = ck["chain"]        # shape (n_iter, n_walkers, ndim)
        saved_iter = int(ck["iteration"])
        saved_burn = int(ck["burn_in"])
        saved_rstate = ck["random_state"].item()
        p0 = saved_chain[-1]             # last walker positions
        remaining_steps = burn_in + n_samples - saved_iter
        if remaining_steps <= 0:
            # Chain already finished; skip sampling, go straight to posterior.
            raw_chain = saved_chain[saved_burn:] if saved_burn < saved_chain.shape[0] else saved_chain
            chain = raw_chain.reshape((-1, raw_chain.shape[-1]))
            _resume_done = True
        else:
            _resume_done = False

    # Reproducibility: walker positions and emcee's StretchMove RNG are both
    # explicitly seeded without mutating NumPy's process-global RNG.
    # ASTROPHYSICAL_HEURISTIC: StretchMove a=1.5 is the Goodman & Weare
    # (2010) recommended value for general-purpose MCMC; it yields an
    # acceptance fraction near the optimal ~25% for multi-dimensional
    # Gaussians.

    # ---- Parallel worker pool -------------------------------------------------
    if n_jobs > 1:
        worker_context: Dict[str, Any] = {
            "phase_days": phase_days,
            "native_flux": native_flux,
            "native_error": native_error,
            "ephemeris": ephemeris,
            "rho_prior_solar": rho_prior_solar,
            "rho_prior_log10_sigma": rho_prior_log10_sigma,
            "eccentric": eccentric,
            "ldtk_prior": ldtk_prior,
            "sector_index": sector_index,
            "exposure_seconds_by_sector": exposure_seconds_by_sector,
            "n_sectors": n_sectors,
        }
        # Prevent MKL/OpenBLAS oversubscription before spawning workers.
        for _env_var in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ.setdefault(_env_var, "1")
        ctx = multiprocessing.get_context("spawn")
        pool: Any = ctx.Pool(n_jobs, initializer=_init_worker, initargs=(worker_context,))
    else:
        pool = None

    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        _log_prob_worker if pool is not None else lambda x: -_neg_log_posterior(
            x,
            phase_days,
            native_flux,
            native_error,
            ephemeris,
            rho_prior_solar,
            rho_prior_log10_sigma,
            eccentric,
            ldtk_prior,
            sector_index=sector_index,
            exposure_seconds_by_sector=exposure_seconds_by_sector,
            n_sectors=n_sectors,
        ),
        pool=pool,
        moves=emcee.moves.StretchMove(a=1.5),
    )
    sampler.random_state = saved_rstate
    total_steps = remaining_steps
    checkpoint_path = outputs_dir / f"mcmc_transit_fit_chain{suffix}.checkpoint.npz"
    old_tau: Optional[float] = float("inf")
    early_stop = False

    # ---- Sampling loop (generator: progress + checkpoint + early stop) -------
    try:
        if not _resume_done:
            for _state in sampler.sample(p0, iterations=total_steps, progress=progress):
                iteration: int = getattr(sampler, "iteration", 0)
                # Telemetry hook: report (iteration, total) to an external HUD.
                if progress_callback is not None:
                    try:
                        progress_callback(int(iteration), int(total_steps))
                    except Exception:  # noqa: BLE001
                        pass
                # Periodic checkpoint save (best-effort).
                if checkpoint_interval > 0 and iteration % checkpoint_interval == 0 and iteration > 0:
                    try:
                        raw_snap = sampler.get_chain(flat=False)
                        np.savez(
                            str(checkpoint_path),
                            chain=raw_snap,
                            iteration=iteration,
                            burn_in=burn_in,
                            random_state=sampler.random_state,
                        )
                    except Exception:
                        pass

                # Autocorrelation-based early convergence (every 100 steps).
                if iteration % 100 == 0 and iteration >= burn_in + 100:
                    try:
                        tau_values = sampler.get_autocorr_time(tol=0)
                        tau_mean = float(np.mean(tau_values))
                        if np.isfinite(tau_mean) and tau_mean > 0:
                            prod_steps = int(iteration) - int(burn_in)
                            if old_tau is not None and prod_steps >= max(100 * tau_mean, 500):
                                if abs(old_tau - tau_mean) / tau_mean < 0.01:
                                    early_stop = True
                                    break
                            old_tau = tau_mean
                    except Exception:
                        pass
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if not _resume_done:
        raw_chain = sampler.get_chain(discard=burn_in, flat=False)
        chain = raw_chain.reshape((-1, raw_chain.shape[-1]))
    # Clean up checkpoint on successful completion.
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink(missing_ok=True)
        except OSError:
            pass

    names = _parameter_names(eccentric, n_sectors, sector_labels)
    posteriors = _posterior_summaries(
        chain, ephemeris, eccentric, n_sectors=n_sectors, sector_labels=sector_labels
    )

    try:
        import logging
        import warnings

        emcee_logger = logging.getLogger("emcee")
        previous_level = emcee_logger.level
        emcee_logger.setLevel(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                tau_values = sampler.get_autocorr_time(discard=burn_in // 2, quiet=True)
            finally:
                emcee_logger.setLevel(previous_level)
        tau_dict = {
            names[index]: float(tau) if np.isfinite(tau) else None
            for index, tau in enumerate(tau_values)
        }
        convergence = _mcmc_convergence_diagnostics(raw_chain, np.asarray(tau_values), names)
    except Exception as exc:
        tau_dict = {"_error": "{0}: {1}".format(type(exc).__name__, exc)}
        convergence = _mcmc_convergence_diagnostics(raw_chain, None, names)

    accelerated_posterior = _candidate_accelerated_posterior_summary(
        chain,
        ephemeris,
        eccentric,
        n_sectors,
        phase_days,
        native_flux,
        native_error,
        rho_prior_solar,
        rho_prior_log10_sigma,
        backend="emcee-cpu",
        sampler_metadata={
            "sampler": "emcee.EnsembleSampler",
            "walkers": int(n_walkers),
            "burn_in": int(burn_in),
            "production": int(n_samples),
            "flat_samples": int(chain.shape[0]),
            "random_seed": int(seed),
            "fallback_reason": backend_fallback_reason,
        },
    )

    payload = {
        "schema_version": "1.0",
        "work_package": "MCMC_TRANSIT_FIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "backend": "emcee-cpu",
        "sampler": "emcee",
        # SCIENTIFIC_BOUNDARY: the posterior is labeled exploratory because
        # the fit does not include a calibrated correlated-noise model,
        # independent-chain convergence validation, or dilution/contamination
        # correction.  A full validation-grade posterior requires the
        # scene-model framework (methods/phasecurve-secondary-control.md).
        "scientific_status": "exploratory-native-cadence-inference",
        "validation_eligible": False,
        "validation_reason": (
            "This likelihood has per-sector flux normalizations but no "
            "calibrated correlated-noise model or independent-chain analysis."
        ),
        "model": (
            "batman quadratic limb darkening, stellar-density locked, eccentric orbit"
            if eccentric
            else "batman quadratic limb darkening, stellar-density locked, circular orbit"
        ),
        "ephemeris": {
            "period_days": ephemeris["period_days"],
            "epoch_btjd": ephemeris["epoch_btjd"],
            "source": ephemeris["source"],
        },
        "density_prior_solar": float(rho_prior_solar),
        "density_prior": {
            **density_prior,
            "propagation": "first-order independent symmetric mass and radius uncertainties",
        },
        "assumptions": {"jitter_prior": jitter_prior},
        "limb_darkening_prior": (
            {"source": "ldtk", "path": ldtk_prior["path"]} if ldtk_prior is not None else None
        ),
        # CONTRACT: The numeric chain is only meaningful with this exact
        # parameter order. Consumers must not infer eccentric coordinates from
        # a positional convention alone.
        "parameter_names": names,
        "posterior": posteriors,
        "accelerated_posterior": accelerated_posterior,
        "mcmc": {
            "backend": "emcee-cpu",
            "walkers": int(n_walkers),
            "burn_in": int(burn_in),
            "production": int(n_samples),
            "actual_production": int(chain.shape[0]),
            "stopped_early": early_stop,
            "stopped_early_reason": "autocorrelation_converged" if early_stop else None,
            "n_jobs": int(n_jobs),
            "flat_samples": int(chain.shape[0]),
            "random_seed": int(seed),
            "random_generator": "numpy.random.RandomState (MT19937)",
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "autocorrelation_times": tau_dict,
            "convergence": convergence,
            "fallback_reason": backend_fallback_reason,
        },
        "likelihood": {
            "cadence": "native",
            "n_points": int(phase_days.size),
            "sector_labels": sector_labels,
            "exposure_seconds_by_sector": {
                str(label): float(exposure_seconds_by_sector[index])
                for index, label in enumerate(sector_labels)
            },
            "flux_err_sources": list(table.get("flux_err_sources", [])),
            "preprocessing": table.get("detrending", {"kind": "pipeline-normalization"}),
        },
        "signal": signal,
        "caveat": (
            "Exploratory native-cadence fit with independent Gaussian residuals; "
            "not an adopted posterior or validation claim."
        ),
    }
    output_path = outputs_dir / f"mcmc_transit_fit{suffix}.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.save(str(outputs_dir / f"mcmc_transit_fit_chain{suffix}.npy"), chain)
    return output_path


def _candidate_accelerated_posterior_summary(
    chain: np.ndarray,
    ephemeris: Dict[str, Any],
    eccentric: bool,
    n_sectors: int,
    phase_days: np.ndarray,
    native_flux: np.ndarray,
    native_error: np.ndarray,
    rho_prior_solar: float,
    rho_prior_log10_sigma: float,
    *,
    backend: str,
    sampler_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Expose the common accelerator schema alongside the legacy artifact."""
    samples = np.asarray(chain, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise RuntimeError("candidate transit sampler returned an empty posterior chain")
    jitter_index = 3 + n_sectors
    q1_index = jitter_index + 1
    q2_index = jitter_index + 2
    values: Dict[str, np.ndarray] = {
        "rp_rstar": samples[:, 0],
        # The legacy chain stores log10 stellar density, while the standalone
        # schema derives a/Rstar from this explicit cgs density array.
        "log_rho_star": samples[:, 1],
        "rho_star_g_cm3": np.power(10.0, samples[:, 1]) * SOLAR_MEAN_DENSITY_G_CM3,
        "impact_parameter": samples[:, 2],
        "log_jitter": samples[:, jitter_index],
        "q1": samples[:, q1_index],
        "q2": samples[:, q2_index],
        "period": np.full(samples.shape[0], float(ephemeris["period_days"])),
        "t0": np.full(samples.shape[0], float(ephemeris["epoch_btjd"])),
    }
    if eccentric:
        values["sqe_cosw"] = samples[:, q2_index + 1]
        values["sqe_sinw"] = samples[:, q2_index + 2]
    density_cgs = rho_prior_solar * SOLAR_MEAN_DENSITY_G_CM3
    density_sigma_cgs = density_cgs * rho_prior_log10_sigma * math.log(10.0)
    data = _AcceleratedTransitFitData(
        time_days=np.asarray(phase_days, dtype=float),
        flux=np.asarray(native_flux, dtype=float),
        flux_err=np.asarray(native_error, dtype=float),
        period_days=float(ephemeris["period_days"]),
        t0_days=float(ephemeris["epoch_btjd"]),
        rho_star_g_cm3=float(density_cgs),
        rho_star_sigma_g_cm3=float(density_sigma_cgs),
        period_sigma_days=None,
        t0_sigma_days=None,
        exposure_seconds=EXPTIME_SECONDS,
        eccentric=eccentric,
    )
    return _summarize_accelerated_samples(
        values,
        data,
        backend=backend,
        sampler_metadata=sampler_metadata,
    )


def _run_numpyro_candidate_transit_fit(
    workspace: CandidateWorkspace,
    *,
    eccentric: bool,
    seed: int,
    signal: Optional[str],
    use_ldtk_prior: bool,
    detrending_method: Optional[str],
    num_warmup: int,
    num_samples: int,
    target_accept_prob: float,
    progress: bool,
    stack: Dict[str, Any],
) -> Path:
    """Run the candidate-native multi-sector GPU NUTS implementation."""
    signal = validate_signal_suffix(signal)
    suffix = f".{signal.lstrip('.')}" if signal else ""
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ephemeris = load_transit_ephemeris(workspace, signal=signal)
    if ephemeris["source"] == "synthetic-demo" or any(
        value == "synthetic-demo" for value in ephemeris.get("field_sources", {}).values()
    ):
        raise RuntimeError("transit fitting requires a complete candidate-derived transit ephemeris")
    stellar = load_stellar_parameters(workspace)
    if stellar["source"] != "candidate-data":
        raise RuntimeError("transit fitting requires complete candidate-derived stellar parameters")
    density_prior = _stellar_density_prior(stellar)
    rho_prior_solar = density_prior["rho_solar"]
    rho_prior_log10_sigma = density_prior["log10_sigma"]
    ldtk_prior = _load_ldtk_prior(workspace) if use_ldtk_prior else None
    table = load_light_curve_table(
        workspace,
        max_points=None,
        require_raw_provenance=True,
        detrending_method=detrending_method,
    )
    if table is None:
        raise RuntimeError("transit fitting requires observed candidate photometry")
    try:
        (
            phase_days,
            native_flux,
            native_error,
            sector_index,
            sector_labels,
            exposure_seconds_by_sector,
        ) = _native_transit_window_data(table, ephemeris)
    except ValueError as exc:
        raise RuntimeError("candidate photometry cannot support a transit fit") from exc

    jax = stack["jax"]
    jnp = stack["jnp"]
    numpyro = stack["numpyro"]
    dist = stack["dist"]
    MCMC = stack["MCMC"]
    NUTS = stack["NUTS"]
    TransitOrbit = stack["TransitOrbit"]
    limb_dark_light_curve = stack["limb_dark_light_curve"]
    gpu_device = stack["device"]
    n_sectors = len(sector_labels)
    phase = jax.device_put(jnp.asarray(phase_days, dtype=jnp.float64), gpu_device)
    flux = jax.device_put(jnp.asarray(native_flux, dtype=jnp.float64), gpu_device)
    flux_err = jax.device_put(jnp.asarray(native_error, dtype=jnp.float64), gpu_device)
    sector_indices = jax.device_put(jnp.asarray(sector_index, dtype=jnp.int32), gpu_device)
    sector_exposures = jax.device_put(
        jnp.asarray(exposure_seconds_by_sector, dtype=jnp.float64), gpu_device
    )
    cadence_days = sector_exposures[sector_indices] / SECONDS_PER_DAY
    sub_sample_offsets = (
        jnp.arange(SUPERSAMPLE_FACTOR, dtype=jnp.float64) - 0.5 * (SUPERSAMPLE_FACTOR - 1)
    ) * (cadence_days[:, None] / SUPERSAMPLE_FACTOR)
    period_days = float(ephemeris["period_days"])
    machine_epsilon = jnp.finfo(jnp.float64).eps

    def transit_model() -> None:
        """Pure multi-sector potential energy model compiled by NumPyro."""
        rp_rs = numpyro.sample("rp_rs", dist.Uniform(0.001, 0.3))
        log_rho_star = numpyro.sample("log_rho_star", dist.Uniform(-2.0, 1.5))
        impact_parameter = numpyro.sample("impact_parameter", dist.Uniform(0.0, 1.2))
        baselines = numpyro.sample(
            "baselines", dist.Uniform(0.99, 1.01).expand([n_sectors]).to_event(1)
        )
        log_jitter = numpyro.sample("log_jitter", dist.Uniform(JITTER_LOG_LOWER, JITTER_LOG_UPPER))
        q1 = numpyro.sample("q1", dist.Uniform(0.01, 0.99))
        q2 = numpyro.sample("q2", dist.Uniform(0.01, 0.99))
        if eccentric:
            sqe_cosw = numpyro.sample("sqe_cosw", dist.Uniform(-1.0, 1.0))
            sqe_sinw = numpyro.sample("sqe_sinw", dist.Uniform(-1.0, 1.0))
        else:
            sqe_cosw = jnp.asarray(0.0, dtype=jnp.float64)
            sqe_sinw = jnp.asarray(0.0, dtype=jnp.float64)

        eccentricity = jnp.square(sqe_cosw) + jnp.square(sqe_sinw)
        safe_eccentricity = jnp.minimum(eccentricity, 1.0 - machine_epsilon)
        sqrt_eccentricity = jnp.sqrt(safe_eccentricity)
        conjunction_factor = jnp.maximum(1.0 + sqrt_eccentricity * sqe_sinw, machine_epsilon)
        rho_solar = jnp.power(10.0, log_rho_star)
        a_rstar = jnp.cbrt(
            GRAVITATIONAL_CONSTANT_CGS
            * jnp.square(period_days * SECONDS_PER_DAY)
            * rho_solar
            * SOLAR_MEAN_DENSITY_G_CM3
            / (3.0 * math.pi)
        )
        conjunction_distance = a_rstar * (1.0 - safe_eccentricity) / conjunction_factor
        cosine_inclination = impact_parameter / jnp.maximum(conjunction_distance, machine_epsilon)
        sine_inclination = jnp.sqrt(jnp.maximum(1.0 - jnp.square(cosine_inclination), machine_epsilon))
        sky_speed = (
            2.0
            * math.pi
            * a_rstar
            / period_days
            * conjunction_factor
            / jnp.sqrt(jnp.maximum(1.0 - jnp.square(safe_eccentricity), machine_epsilon))
            * sine_inclination
        )
        valid_geometry = (
            (eccentricity < 1.0)
            & (impact_parameter < 1.0 + rp_rs)
            & (cosine_inclination >= 0.0)
            & (cosine_inclination <= 1.0)
            & jnp.isfinite(a_rstar)
            & jnp.isfinite(sky_speed)
            & (sky_speed > 0.0)
        )
        numpyro.factor("physical_geometry", jnp.where(valid_geometry, 0.0, -jnp.inf))
        numpyro.factor(
            "stellar_density_prior",
            -0.5 * jnp.square((log_rho_star - math.log10(rho_prior_solar)) / rho_prior_log10_sigma),
        )
        jitter = jnp.exp(log_jitter)
        half_cauchy_ratio = jitter / JITTER_HALF_CAUCHY_SCALE
        numpyro.factor(
            "log_jitter_prior",
            jnp.log(2.0 / (math.pi * JITTER_HALF_CAUCHY_SCALE))
            - jnp.log1p(jnp.square(half_cauchy_ratio))
            + log_jitter,
        )
        u1 = 2.0 * jnp.sqrt(q1) * q2
        u2 = jnp.sqrt(q1) * (1.0 - 2.0 * q2)
        if ldtk_prior is not None:
            numpyro.factor(
                "ldtk_quadratic_prior",
                -0.5 * jnp.square((u1 - ldtk_prior["u1"]) / ldtk_prior["u1_err"])
                -0.5 * jnp.square((u2 - ldtk_prior["u2"]) / ldtk_prior["u2_err"]),
            )
        orbit = TransitOrbit(
            period=jnp.asarray(period_days, dtype=jnp.float64),
            speed=jnp.maximum(sky_speed, machine_epsilon),
            time_transit=jnp.asarray(0.0, dtype=jnp.float64),
            impact_param=impact_parameter,
            radius_ratio=rp_rs,
        )
        light_curve = limb_dark_light_curve(orbit, u1, u2)
        delta_flux = sum(
            light_curve(phase + sub_sample_offsets[:, index])
            for index in range(SUPERSAMPLE_FACTOR)
        ) / SUPERSAMPLE_FACTOR
        model_flux = baselines[sector_indices] * (1.0 + delta_flux)
        total_sigma = jnp.sqrt(jnp.square(flux_err) + jnp.square(jitter))
        numpyro.sample("observed_flux", dist.Normal(model_flux, total_sigma).to_event(1), obs=flux)

    sampler = MCMC(
        NUTS(transit_model, target_accept_prob=target_accept_prob),
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=progress,
        jit_model_args=True,
    )
    sampler.run(jax.device_put(jax.random.PRNGKey(seed), gpu_device), extra_fields=("diverging",))
    raw_samples = sampler.get_samples(group_by_chain=False)
    samples = {name: np.asarray(value, dtype=float) for name, value in raw_samples.items()}
    baselines = samples["baselines"]
    chain_columns = [
        samples["rp_rs"],
        samples["log_rho_star"],
        samples["impact_parameter"],
    ]
    chain_columns.extend(baselines[:, index] for index in range(n_sectors))
    chain_columns.extend([samples["log_jitter"], samples["q1"], samples["q2"]])
    if eccentric:
        chain_columns.extend([samples["sqe_cosw"], samples["sqe_sinw"]])
    chain = np.column_stack(chain_columns)
    if not np.all(np.isfinite(chain)):
        raise RuntimeError("NumPyro returned a non-finite candidate transit posterior")
    _require_batman()
    names = _parameter_names(eccentric, n_sectors, sector_labels)
    posteriors = _posterior_summaries(
        chain, ephemeris, eccentric, n_sectors=n_sectors, sector_labels=sector_labels
    )
    extra_fields = sampler.get_extra_fields(group_by_chain=False)
    divergences = int(np.sum(np.asarray(extra_fields.get("diverging", ()), dtype=bool)))
    sampler_metadata = {
        "sampler": "numpyro.NUTS",
        "num_warmup": int(num_warmup),
        "num_samples": int(num_samples),
        "target_accept_prob": float(target_accept_prob),
        "random_seed": int(seed),
        "gpu_device": str(gpu_device),
        "divergences": divergences,
        "jax_enable_x64": True,
    }
    accelerated_posterior = _candidate_accelerated_posterior_summary(
        chain,
        ephemeris,
        eccentric,
        n_sectors,
        phase_days,
        native_flux,
        native_error,
        rho_prior_solar,
        rho_prior_log10_sigma,
        backend="jax-gpu",
        sampler_metadata=sampler_metadata,
    )
    jitter_prior = _jitter_prior_assumption()
    payload = {
        "schema_version": "1.0",
        "work_package": "MCMC_TRANSIT_FIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "candidate-data",
        "backend": "jax-gpu",
        "sampler": "numpyro",
        "scientific_status": "exploratory-native-cadence-inference",
        "validation_eligible": False,
        "validation_reason": (
            "This likelihood has per-sector flux normalizations but no calibrated correlated-noise "
            "model or independent-chain analysis."
        ),
        "model": (
            "jaxoplanet TransitOrbit quadratic limb darkening, stellar-density locked, eccentric orbit"
            if eccentric
            else "jaxoplanet TransitOrbit quadratic limb darkening, stellar-density locked, circular orbit"
        ),
        "ephemeris": {
            "period_days": ephemeris["period_days"],
            "epoch_btjd": ephemeris["epoch_btjd"],
            "source": ephemeris["source"],
        },
        "density_prior_solar": float(rho_prior_solar),
        "density_prior": {
            **density_prior,
            "propagation": "first-order independent symmetric mass and radius uncertainties",
        },
        "assumptions": {"jitter_prior": jitter_prior},
        "limb_darkening_prior": (
            {"source": "ldtk", "path": ldtk_prior["path"]} if ldtk_prior is not None else None
        ),
        "parameter_names": names,
        "posterior": posteriors,
        "accelerated_posterior": accelerated_posterior,
        "mcmc": {
            "backend": "jax-gpu",
            "chains": 1,
            "warmup": int(num_warmup),
            "production": int(num_samples),
            "actual_production": int(chain.shape[0]),
            "flat_samples": int(chain.shape[0]),
            "random_seed": int(seed),
            "random_generator": "jax.random.PRNGKey",
            "target_accept_prob": float(target_accept_prob),
            "divergences": divergences,
            "convergence": {
                "status": "not-demonstrated",
                "scientific_posterior_eligible": False,
                "reason": "single-chain NUTS diagnostics are not independent-chain validation",
            },
        },
        "likelihood": {
            "cadence": "native",
            "n_points": int(phase_days.size),
            "sector_labels": sector_labels,
            "exposure_seconds_by_sector": {
                str(label): float(exposure_seconds_by_sector[index])
                for index, label in enumerate(sector_labels)
            },
            "flux_err_sources": list(table.get("flux_err_sources", [])),
            "preprocessing": table.get("detrending", {"kind": "pipeline-normalization"}),
        },
        "signal": signal,
        "caveat": (
            "Exploratory native-cadence GPU fit with independent Gaussian residuals; "
            "not an adopted posterior or validation claim."
        ),
    }
    output_path = outputs_dir / f"mcmc_transit_fit{suffix}.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.save(str(outputs_dir / f"mcmc_transit_fit_chain{suffix}.npy"), chain)
    return output_path


def compute_bayesian_model_comparison(
    ln_z_circular: float,
    ln_z_circular_err: float,
    ln_z_eccentric: float,
    ln_z_eccentric_err: float,
) -> Dict[str, Any]:
    """Compute Bayesian model comparison between eccentric and circular transit fits.

    Evaluates the Bayes Factor B_10 = exp(Delta ln Z) where Delta ln Z = ln Z_eccentric - ln Z_circular,
    and classifies the evidence on the Kass & Raftery (1995) scale:
    - |Delta ln Z| < 1.0: inconclusive (barely worth mentioning)
    - 1.0 <= |Delta ln Z| < 2.5: substantial_evidence (3:1 to 12:1 odds)
    - 2.5 <= |Delta ln Z| < 5.0: strong_evidence (12:1 to 150:1 odds)
    - |Delta ln Z| >= 5.0: decisive_evidence (> 150:1 odds)
    """
    for val, name in (
        (ln_z_circular, "ln_z_circular"),
        (ln_z_circular_err, "ln_z_circular_err"),
        (ln_z_eccentric, "ln_z_eccentric"),
        (ln_z_eccentric_err, "ln_z_eccentric_err"),
    ):
        if not math.isfinite(float(val)):
            raise ValueError(f"{name} must be a finite number")

    delta_ln_z = float(ln_z_eccentric - ln_z_circular)
    delta_ln_z_err = float(math.sqrt(float(ln_z_circular_err) ** 2 + float(ln_z_eccentric_err) ** 2))

    # Guard against overflow when evaluating Bayes factor exp(Delta ln Z)
    clipped_delta = float(np.clip(delta_ln_z, -700.0, 700.0))
    bayes_factor = float(math.exp(clipped_delta))

    abs_delta = abs(delta_ln_z)
    if abs_delta < 1.0:
        kass_raftery_scale = "inconclusive"
        evidence_phrase = "inconclusive"
    elif abs_delta < 2.5:
        kass_raftery_scale = "substantial_evidence"
        evidence_phrase = "substantially favored"
    elif abs_delta < 5.0:
        kass_raftery_scale = "strong_evidence"
        evidence_phrase = "strongly favored"
    else:
        kass_raftery_scale = "decisive_evidence"
        evidence_phrase = "decisively favored"

    if delta_ln_z > 0.0:
        preferred_model = "eccentric"
        interpretation = f"Eccentric orbital geometry is {evidence_phrase} over the circular model."
    elif delta_ln_z < 0.0:
        preferred_model = "circular"
        interpretation = f"Circular orbital geometry is {evidence_phrase} over the eccentric model."
    else:
        preferred_model = "inconclusive"
        interpretation = "Neither orbital model is favored (Delta ln Z = 0.0)."

    return {
        "ln_z_circular": float(ln_z_circular),
        "ln_z_circular_err": float(ln_z_circular_err),
        "ln_z_eccentric": float(ln_z_eccentric),
        "ln_z_eccentric_err": float(ln_z_eccentric_err),
        "delta_ln_z": delta_ln_z,
        "delta_ln_z_err": delta_ln_z_err,
        "bayes_factor": bayes_factor,
        "preferred_model": preferred_model,
        "kass_raftery_scale": kass_raftery_scale,
        "interpretation": interpretation,
    }


def _run_dynesty_transit_fit(
    workspace: CandidateWorkspace,
    n_samples: int,
    eccentric: bool,
    seed: int,
    signal: Optional[str],
    use_ldtk_prior: bool,
    detrending_method: Optional[str],
    dlogz_tolerance: float = 0.5,
) -> Path:
    """Run optional dynamic nested sampling with an explicit normalized prior transform.

    This function wraps dynesty (Speagle 2020) to draw posterior samples
    under the same likelihood and prior model used by the emcee path, but
    with the addition of a custom prior transform that maps the unit
    hypercube through truncated-Gaussian and LDTk-weighted inverse-CDF
    transforms (see :func:`_make_dynesty_prior_transform`).

    Nested sampling simultaneously estimates the Bayesian evidence

    .. math::

        Z = \\int L(\\theta) \\, \\pi(\\theta) \\, d\\theta

    which is reported as ``log_z`` with its estimated numerical uncertainty
    ``log_z_err``.  This evidence is **descriptive** — it is not a planet
    validation probability and depends on the choice of prior, likelihood,
    and stopping rule.

    .. note::

        The evidence reported by dynesty is the *model* evidence under the
        Gaussian independent-noise likelihood, not a model-comparison odds
        ratio.  Comparing ``log_z`` between circular and eccentric fits is
        not statistically rigorous without calibrating the prior odds and
        accounting for correlated noise.

    Parameters
    ----------
    workspace : CandidateWorkspace
        The candidate workspace.
    n_samples : int
        Number of live points proxy; used to scale ``nlive_init``.
    eccentric : bool
        Whether eccentricity components are active.
    seed : int
        RNG seed.
    signal : str or None
        Signal suffix.
    use_ldtk_prior : bool
        Whether to apply LDTk limb-darkening prior.
    detrending_method : str or None
        Detrending method for light curve retrieval.
    dlogz_tolerance : float, optional
        Initial convergence tolerance :math:`\\Delta\\ln Z` (default 0.5).

    Returns
    -------
    Path
        Path to the written ``mcmc_transit_fit.json`` artifact.

    Raises
    ------
    RuntimeError
        If dynesty is not installed, photometry/parameters are synthetic
        or missing, or the nested-sampling results are incomplete.
    """
    signal = validate_signal_suffix(signal)
    try:
        import dynesty
    except ImportError as exc:
        raise RuntimeError(
            "dynesty is required for --sampler dynesty; install the pinned optional dependency with "
            'pip install -e ".[inference]"'
        ) from exc
    if n_samples <= 0:
        raise ValueError("n_samples must be positive for dynesty")
    _require_batman()

    ephemeris = load_transit_ephemeris(workspace, signal=signal)
    if ephemeris["source"] == "synthetic-demo" or any(
        value == "synthetic-demo" for value in ephemeris.get("field_sources", {}).values()
    ):
        raise RuntimeError("transit fitting requires a complete candidate-derived transit ephemeris")
    stellar = load_stellar_parameters(workspace)
    if stellar["source"] != "candidate-data":
        raise RuntimeError("transit fitting requires complete candidate-derived stellar parameters")
    density_prior = _stellar_density_prior(stellar)
    rho_prior_solar = density_prior["rho_solar"]
    rho_prior_log10_sigma = density_prior["log10_sigma"]
    ldtk_prior = _load_ldtk_prior(workspace) if use_ldtk_prior else None
    table = load_light_curve_table(
        workspace,
        max_points=None,
        require_raw_provenance=True,
        detrending_method=detrending_method,
    )
    if table is None:
        raise RuntimeError("transit fitting requires observed candidate photometry")
    source = "candidate-data"
    try:
        (
            phase_days,
            native_flux,
            native_error,
            sector_index,
            sector_labels,
            exposure_seconds_by_sector,
        ) = _native_transit_window_data(
            table, ephemeris
        )
    except ValueError as exc:
        raise RuntimeError("candidate photometry cannot support a transit fit") from exc

    n_sectors = len(sector_labels)
    jitter_prior = _jitter_prior_assumption()
    prior_transform = _make_dynesty_prior_transform(
        rho_prior_solar,
        rho_prior_log10_sigma,
        eccentric,
        ldtk_prior,
        n_sectors=n_sectors,
    )
    ndim = _parameter_count(eccentric, n_sectors)
    # ASTROPHYSICAL_HEURISTIC: initial live point count is at least
    # 2*ndim+1 (minimal for reliable evidence integration; Skilling 2006)
    # and is capped at 500 to limit runtime.  For very large samples the
    # live point count is scaled as n_samples // 10.
    initial_live_points = max(2 * ndim + 1, min(500, max(50, n_samples // 10)))
    # SCIENTIFIC_BOUNDARY: ``dlogz_init`` applies only to dynesty's initial
    # baseline run. The later dynamic allocation uses dynesty's default
    # stopping function because this call supplies no custom stop kwargs.
    # Neither numerical condition is a physical validation criterion.
    nested_sampler = dynesty.DynamicNestedSampler(
        lambda theta: _log_likelihood(
            theta,
            phase_days,
            native_flux,
            native_error,
            ephemeris,
            eccentric,
            sector_index=sector_index,
            exposure_seconds_by_sector=exposure_seconds_by_sector,
            n_sectors=n_sectors,
        ),
        prior_transform,
        ndim,
        rstate=np.random.default_rng(seed),
    )
    run_nested_kwargs = {
        "nlive_init": initial_live_points,
        "dlogz_init": dlogz_tolerance,
        "print_progress": False,
    }
    nested_sampler.run_nested(**run_nested_kwargs)
    results = nested_sampler.results
    samples = np.asarray(results.samples, dtype=float)
    log_weights = np.asarray(results.logwt, dtype=float)
    log_evidence = np.asarray(results.logz, dtype=float)
    log_evidence_error = np.asarray(results.logzerr, dtype=float)
    if (
        samples.ndim != 2
        or samples.shape[1] != ndim
        or samples.shape[0] == 0
        or log_weights.shape != (samples.shape[0],)
        or log_evidence.size == 0
        or log_evidence_error.size == 0
        or not np.all(np.isfinite(samples))
        or not np.all(np.isfinite(log_weights))
        or not math.isfinite(float(log_evidence[-1]))
        or not math.isfinite(float(log_evidence_error[-1]))
    ):
        raise RuntimeError("dynesty returned incomplete or non-finite nested-sampling results")
    posterior_weights = np.exp(log_weights - float(log_evidence[-1]))
    chain, effective_samples = _resample_weighted_posterior(samples, posterior_weights, seed)
    names = _parameter_names(eccentric, n_sectors, sector_labels)
    posteriors = _posterior_summaries(
        chain, ephemeris, eccentric, n_sectors=n_sectors, sector_labels=sector_labels
    )
    sampling_efficiency = float(getattr(results, "eff", np.nan))
    if not math.isfinite(sampling_efficiency):
        sampling_efficiency = None

    bayesian_model_comparison = None
    try:
        if eccentric:
            log_z_ecc = float(log_evidence[-1])
            log_z_ecc_err = float(log_evidence_error[-1])
            prior_transform_circ = _make_dynesty_prior_transform(
                rho_prior_solar,
                rho_prior_log10_sigma,
                False,
                ldtk_prior,
                n_sectors=n_sectors,
            )
            ndim_circ = _parameter_count(False, n_sectors)
            live_circ = max(2 * ndim_circ + 1, min(300, max(50, n_samples // 10)))
            sampler_circ = dynesty.DynamicNestedSampler(
                lambda theta: _log_likelihood(
                    theta,
                    phase_days,
                    native_flux,
                    native_error,
                    ephemeris,
                    False,
                    sector_index=sector_index,
                    exposure_seconds_by_sector=exposure_seconds_by_sector,
                    n_sectors=n_sectors,
                ),
                prior_transform_circ,
                ndim_circ,
                rstate=np.random.default_rng(seed + 1),
            )
            sampler_circ.run_nested(nlive_init=live_circ, dlogz_init=dlogz_tolerance, print_progress=False)
            res_circ = sampler_circ.results
            log_z_circ = float(np.asarray(res_circ.logz, dtype=float)[-1])
            log_z_circ_err = float(np.asarray(res_circ.logzerr, dtype=float)[-1])
            bayesian_model_comparison = compute_bayesian_model_comparison(
                log_z_circ, log_z_circ_err, log_z_ecc, log_z_ecc_err
            )
        else:
            log_z_circ = float(log_evidence[-1])
            log_z_circ_err = float(log_evidence_error[-1])
            prior_transform_ecc = _make_dynesty_prior_transform(
                rho_prior_solar,
                rho_prior_log10_sigma,
                True,
                ldtk_prior,
                n_sectors=n_sectors,
            )
            ndim_ecc = _parameter_count(True, n_sectors)
            live_ecc = max(2 * ndim_ecc + 1, min(300, max(50, n_samples // 10)))
            sampler_ecc = dynesty.DynamicNestedSampler(
                lambda theta: _log_likelihood(
                    theta,
                    phase_days,
                    native_flux,
                    native_error,
                    ephemeris,
                    True,
                    sector_index=sector_index,
                    exposure_seconds_by_sector=exposure_seconds_by_sector,
                    n_sectors=n_sectors,
                ),
                prior_transform_ecc,
                ndim_ecc,
                rstate=np.random.default_rng(seed + 1),
            )
            sampler_ecc.run_nested(nlive_init=live_ecc, dlogz_init=dlogz_tolerance, print_progress=False)
            res_ecc = sampler_ecc.results
            log_z_ecc = float(np.asarray(res_ecc.logz, dtype=float)[-1])
            log_z_ecc_err = float(np.asarray(res_ecc.logzerr, dtype=float)[-1])
            bayesian_model_comparison = compute_bayesian_model_comparison(
                log_z_circ, log_z_circ_err, log_z_ecc, log_z_ecc_err
            )
    except Exception:
        pass

    payload = {
        "schema_version": "1.0",
        "work_package": "NESTED_TRANSIT_FIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "scientific_status": "exploratory-native-cadence-inference",
        "validation_eligible": False,
        "validation_reason": (
            "Nested sampling does not make this an adopted posterior without a "
            "calibrated correlated-noise model and independent reproducibility checks."
        ),
        "sampler": "dynesty",
        "dynesty_version": getattr(dynesty, "__version__", "unknown"),
        "model": (
            "batman quadratic limb darkening, stellar-density locked, eccentric orbit"
            if eccentric
            else "batman quadratic limb darkening, stellar-density locked, circular orbit"
        ),
        "ephemeris": {
            "period_days": ephemeris["period_days"],
            "epoch_btjd": ephemeris["epoch_btjd"],
            "source": ephemeris["source"],
        },
        "density_prior_solar": float(rho_prior_solar),
        "density_prior": {
            **density_prior,
            "propagation": "first-order independent symmetric mass and radius uncertainties",
        },
        "assumptions": {"jitter_prior": jitter_prior},
        "limb_darkening_prior": (
            {"source": "ldtk", "path": ldtk_prior["path"]} if ldtk_prior is not None else None
        ),
        # CONTRACT: Persist the resampled-chain order for positional consumers.
        "parameter_names": names,
        "posterior": posteriors,
        # SCIENTIFIC_BOUNDARY: nested-sampling log Z is the model evidence
        # under the Gaussian independent-noise likelihood and chosen prior;
        # it does not constitute a validation probability or calibrated
        # model-comparison odds ratio.
        "evidence": {
            "log_z": float(log_evidence[-1]),
            "log_z_err": float(log_evidence_error[-1]),
            "meaning": "Nested-sampling model evidence; not a validation probability.",
        },
        "bayesian_model_comparison": bayesian_model_comparison,
        "diagnostics": {
            "initial_live_points": int(initial_live_points),
            "sampler_niter": int(getattr(results, "niter", 0)),
            "sampler_stop_criterion": None,
            "sampler_stop_criterion_status": "not-reported-by-dynesty-results",
            "dlogz_init_tolerance": float(dlogz_tolerance),
            "dlogz_tolerance": float(dlogz_tolerance),
            "dynesty_run_configuration": {
                "initial_baseline": {
                    "nlive_init": int(run_nested_kwargs["nlive_init"]),
                    "criterion": "dlogz_init",
                    "dlogz_init": float(run_nested_kwargs["dlogz_init"]),
                },
                "final_dynamic_stopping": {
                    "criterion": "dynesty-default-stopping-function",
                    "custom_stop_function": None,
                    "custom_stop_kwargs": {},
                    "use_stop": {
                        "value": True,
                        "source": "dynesty API default; not overridden by this run",
                    },
                    "configured_hard_stop_kwargs": {},
                    "result_criterion": None,
                    "result_criterion_status": "not-reported-by-dynesty-results",
                },
            },
            "iterations": int(getattr(results, "niter", samples.shape[0])),
            "likelihood_calls": int(np.sum(np.asarray(getattr(results, "ncall", 0)))),
            "sampling_efficiency_percent": sampling_efficiency,
            "weighted_samples": int(samples.shape[0]),
            "effective_samples": effective_samples,
            "resampled_samples": int(chain.shape[0]),
            "resampling": "systematic equal-weight resampling",
            "resampling_seed": int(seed),
            "prior_transform": (
                "LDTk-weighted Kipping prior via numerical inverse CDF"
                if ldtk_prior is not None
                else "analytic bounded and truncated priors"
            ),
        },
        "likelihood": {
            "cadence": "native",
            "n_points": int(phase_days.size),
            "sector_labels": sector_labels,
            "exposure_seconds_by_sector": {
                str(label): float(exposure_seconds_by_sector[index])
                for index, label in enumerate(sector_labels)
            },
            "flux_err_sources": list(table.get("flux_err_sources", [])),
            "preprocessing": table.get("detrending", {"kind": "pipeline-normalization"}),
        },
        "signal": signal,
        "caveat": (
            "Exploratory native-cadence fit with independent Gaussian residuals; "
            "nested evidence is not a validation probability or an adopted posterior."
        ),
    }
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{signal.lstrip('.')}" if signal else ""
    output_path = outputs_dir / f"mcmc_transit_fit{suffix}.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.save(str(outputs_dir / f"mcmc_transit_fit_chain{suffix}.npy"), chain)
    return output_path
