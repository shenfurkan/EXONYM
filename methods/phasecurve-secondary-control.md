# Phase-curve secondary-eclipse control

## Scope

`exonym phasecurve` fits a simultaneous circular-harmonic regression and a
secondary-eclipse control. It is an exploratory photometric diagnostic. Its
amplitudes, component significances, and eclipse control do not establish a
detection, false-alarm probability, or validation result.

## Circular control

Without a compatible candidate-local eccentric transit fit, the secondary box
is centered at orbital phase 0.5 and uses the candidate ephemeris transit
duration. The output calls this a circular-orbit control. It does not apply the
circular assumption to an eccentric interpretation.

## Eccentric posterior control

When `outputs/mcmc_transit_fit.json` records a candidate-data eccentric model
with the same period and epoch, the command reads the paired numeric chain.
It records SHA-256 digests for both files in the phase-curve result. Invalid,
missing, or mismatched eccentric inputs stop the command instead of falling
back to an unlabelled circular box.

For retained posterior draws, the control calculates transit and occultation
conjunctions with an edge-on Keplerian approximation. It derives the elapsed
mean anomaly between them, which gives the secondary phase. It scales the
primary duration by the small-angle chord and transverse-velocity ratio. Draws
that do not produce a geometric occultation contribute zero to the control.
The regression uses the average box template over a deterministic subset of
at most 512 retained chain draws.

The result reports phase and duration quantiles, the sampled occultation
fraction, template sample count, source digests, and the selected template
method. Phase intervals retain an explicit wrap flag when an interval crosses
phase zero.

## Limits

The circular harmonic terms remain a circular basis. The secondary template
does not model a planetary brightness map, hotspot offsets, ephemeris
uncertainty, light-travel time, a full occultation light curve, or correlated
photometric noise. The transit posterior is exploratory and its paramete
uncertainty is only used to construct the diagnostic box control. Review the
candidate-local fit, chain diagnostics, and phase-curve output together before
making an astrophysical interpretation.

## Retained sources and covariance provenance gap

- Faigler & Mazeh (2011), *MNRAS* 415, 3921, ADS `2011MNRAS.415.3921F`, DOI
  `10.1111/j.1365-2966.2011.19011.x`.
- Morris (1985), *ApJ* 295, 143, ADS `1985ApJ...295..143M`, DOI
  `10.1086/163359`.
- Shporer (2017), *PASP* 129, 072001, ADS `2017PASP..129g2001S`, DOI
  `10.1088/1538-3873/aa7112`.

The finite-sample cluster-sandwich covariance and day-block policy are robust
regression diagnostics. Their exact primary source has not yet been registered
as a retained local PDF, so they must not be described as a calibrated red-noise
or false-alarm model.
