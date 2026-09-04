# Detrending and transit inference adapters

## Scope

`exonym detrend` writes an opt-in candidate-local processed series and manifest. It does not replace raw photometry or the frozen survey detrending comparison. The fit command can use an LDTk prior, emcee, optional dynesty, or the auto-selected NumPyro/JAX backend, with telemetry and an emcee resume checkpoint where applicable. These operations estimate descriptive model parameters and posterior summaries. None validates a planet.

## Equations and units

For the running-median backend, the trend at cadence `i` is:

```
T_i = median({f_j : |t_j - t_i| is within the selected window})
f_detrended,i = f_i / T_i
```

`t` is BTJD in days, `f` and `T` are normalized relative flux, and the window is days. Wotan receives the same time and normalized-flux arrays with its declared biweight backend. Celerite uses a Matern 3/2 GP trend:

```
k(tau) = sigma^2 (1 + sqrt(3) |tau| / rho) exp(-sqrt(3) |tau| / rho)
```

`sigma` is normalized-flux amplitude and `rho` is days. The output divides flux and, when supplied, flux uncertainty by the absolute trend.

The transit model uses a quadratic limb-darkening law:

```
I(mu) / I(1) = 1 - u1 (1 - mu) - u2 (1 - mu)^2
```

`u1` and `u2` are dimensionless. LDTk obtains them from candidate-local effective temperature in K, log g in log10(cm s^-2), and metallicity in dex. The fitter uses Kipping coordinates:

```
u1 = 2 sqrt(q1) q2
u2 = sqrt(q1) (1 - 2 q2)
```

and derives the dimensionless scaled semi-major axis from stellar density and period:

```
(a / R_star)^3 = G P^2 rho_star / (3 pi)
```

`P` is days in the input and converts to seconds for this equation; density is g cm^-3. The likelihood uses independent Gaussian errors with fitted normalized-flux jitter. Dynesty reports descriptive log evidence `log Z` and its estimated numerical uncertainty. Corner plots visualize samples only.

## Assumptions and failure modes

- Median, biweight, and GP trends can attenuate transits, absorb stellar variability, or create depth changes when the selected timescale is poor.
- The celerite GP assumes the stated stationary kernel and finite positive flux uncertainties. Wotan and celerite are explicit optional dependencies.
- LDTk priors assume supplied stellar parameters, uncertainties, and passband represent the observed star and data. A missing dependency, missing stellar uncertainty, non-finite coefficient, or candidate mismatch writes no prior.
- The phase-folded, median-binned fit is descriptive. It does not model every cadence, time-correlated noise, dilution uncertainty, or stellar variability.
- Nested-sampling evidence depends on the likelihood, prior, stopping rule, and numerical settings. It is not a planet-validation probability. A missing dynesty package writes no fit output.
- `fit --sampler auto` records whether a compatible GPU NumPyro/JAX runtime was selected or why CPU emcee was used. An explicit GPU runtime failure remains a failed run; it must not silently change backend. `--resume` applies only to the intermediate emcee sampler checkpoint and is separate from a workspace checkpoint.
- Live telemetry is presentation-only. Memory, elapsed time, progress, and step labels are never convergence evidence and must not replace the recorded sampler diagnostics.

## Sources

- Hippke et al., "Wotan: Comprehensive time-series de-trending," *AJ* 158, 143 (2019), ADS `2019AJ....158..143H`, DOI `10.3847/1538-3881/ab3984`.
- Foreman-Mackey et al., "Fast and scalable Gaussian process modeling with applications to astronomical time series," *AJ* 154, 220 (2017), ADS `2017AJ....154..220F`, DOI `10.3847/1538-3881/aa9332`.
- **Provenance gap:** the LDTk primary source is not yet registered in `literature/README.md`. LDTk remains an optional prior adapter and no unverified citation metadata is asserted here.
- Kipping, "Efficient, uninformative sampling of limb darkening coefficients," *MNRAS* 435, 2152 (2013), ADS `2013MNRAS.435.2152K`, DOI `10.1093/mnras/stt1435`.
- Mandel and Agol, "Analytic light curves for planetary transit searches," *ApJL* 580, L171 (2002), ADS `2002ApJ...580L.171M`, DOI `10.1086/345520`.
- **Provenance gap:** the optional dynesty package source is not currently registered as a retained primary PDF; log evidence remains descriptive.
- `corner` is rendering software, not a physical relation. Its package source is not asserted as retained primary literature.
