# TTV and orbital-dynamics diagnostics

## Scope

`exonym ttv` fits candidate-local transit templates to observed events and
reports timing residuals, model comparisons, and first-order mean-motion
resonance super-period diagnostics. It is candidate evidence only; it does not
confirm a companion or validate a planet.

## Fit binding

The timing template reads the matching candidate transit-fit posterior median
and records the fit artifact SHA-256 digest. Missing, stale, ambiguous,
candidate-mismatched, or tampered fit artifacts block the analysis rather than
allowing an unbound template or synthetic ephemeris.

## Models and units

Observed-minus-calculated timing is reported in minutes. Linear and quadratic
ephemerides are compared using the retained timing uncertainties. With
`--fit-orbital-decay`, the quadratic derivative is reported in the declared
period/time units and the model comparison includes BIC. A first-order
resonance super-period uses:

```text
1 / P_super = abs(j / P_outer - (j - 1) / P_inner)
```

Periods and epochs retain their candidate-declared time system. The optional
orbital-decay derivative and BIC are formal diagnostics: a preferred quadratic
model is not by itself evidence of physical tidal decay, a third body, or a
planetary mass.

## Failure and interpretation limits

Low-SNR event fits can absorb shape, baseline, detrending, or activity changes.
Sparse timing coverage, aliases, correlated noise, and an incorrect fixed
ephemeris can dominate the result. The output must be interpreted with the
source fit, timing uncertainties, residual diagnostics, and provenance hashes;
it cannot unlock the analysis gate or create a validation claim.

## Retained sources

- Mandel & Agol (2002), *ApJ* 580, L171, ADS `2002ApJ...580L.171M`, DOI
  `10.1086/345520` (template flux context).
- Lithwick, Xie & Wu (2012), *ApJ* 761, 122, ADS `2012ApJ...761..122L`, DOI
  `10.1088/0004-637X/761/2/122` (first-order MMR super-period context).
- Danby & Burkardt (1983), *CeMec* 31, 95, ADS `1983CeMec..31...95D`, DOI
  `10.1007/BF01686811` (Kepler-equation iteration).

The retained source registry contains no primary model that turns a formal
quadratic ephemeris/BIC preference into tidal decay or a companion claim;
EXONYM therefore reports it only as fit-bound diagnostic evidence.
