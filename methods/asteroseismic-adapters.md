# Asteroseismic analysis and adapters

## Scope

The asteroseismology command estimates a solar-like oscillation envelope from candidate-local light curves. It can record a pySYD cross-check and a tess-atl runtime status. Native results and optional adapter outputs are descriptive stellar evidence. They do not validate a planet or determine a candidate disposition.

## Equations and units

The native search uses a Lomb-Scargle power spectral density, fits a bounded
Harvey-style granulation background, and divides by the smoothed background
estimate. The whitened power is:

```
W(nu) = P(nu) / B(nu)
```

Frequency `nu` is microhertz, and `P`, `B`, and `W` use the PSD normalization returned by Astropy. It estimates the envelope maximum `nu_max` in microhertz and the large separation `Delta_nu` in microhertz through the correlation of the local whitened spectrum with a frequency-shifted copy.

When both observables are usable, scaling relations report:

```
R/R_sun = (nu_max/nu_max,sun) (Delta_nu/Delta_nu,sun)^-2 (T_eff/T_eff,sun)^1/2
M/M_sun = (nu_max/nu_max,sun)^3 (Delta_nu/Delta_nu,sun)^-4 (T_eff/T_eff,sun)^3/2
```

`T_eff` is K, `nu_max` and `Delta_nu` are microhertz, and mass and radius are solar units. A Delta-nu correction is applied only when a candidate-local evidence record supplies one; otherwise the identity correction is recorded explicitly. pySYD receives candidate-local time and flux data and records raw adapter estimates with their units as supplied by pySYD. tess-atl only records whether its runtime/interface is available; it does not synthesize stellar values.

## Assumptions and failure modes

- Solar scaling relations are approximate and can carry systematic error, especially for evolved stars and when `Delta_nu` needs a model-dependent correction.
- Finite cadence, gaps, window functions, instrumental systematics, and granulation can create spurious envelope or spacing peaks.
- A result outside physical bounds, or inconsistent with a catalog radius prior where supplied, is flagged as implausible rather than accepted.
- Frequency bounds are clipped to the supported PSD range and the result records the effective bounds and Rayleigh resolution; a requested range is not silently treated as observed support.
- Missing pySYD or tess-atl, an adapter exception, malformed optional output, or no adapter output produces a candidate-local status manifest without a substituted stellar result.

## Sources

- **Provenance gap:** the historical scaling source and optional pySYD/tess-atl adapter sources are not currently registered as retained primary literature. The implementation uses the retained scaling sources below; optional adapter output remains descriptive status evidence.
- Huber et al., "Testing the asteroseismic mass and radius relations for solar-type stars," *ApJ* 743, 143 (2011), ADS `2011ApJ...743..143H`, DOI `10.1088/0004-637X/743/2/143`.
- Chaplin et al., "Asteroseismic Fundamental Properties of Solar-type Stars Observed by the NASA Kepler Mission," *ApJS* 210, 1 (2014), ADS `2014ApJS..210....1C`, DOI `10.1088/0067-0049/210/1/1`.
