# Gaussian-process noise model reference

## Current implementation status

This document specifies the Gaussian-process (GP) treatment intended fo
future light-curve analyses.  The current production commands do **not** fit
a GP noise model.  In particular, the white-noise jitter term used by the
transit-fitting workflow is not a GP and must not be described as one in a
report, claim, or manuscript.

No analysis may claim GP detrending or GP-derived planetary parameters until
the fit has been performed on the candidate-local data and its run record has
been preserved with the candidate.

## Scope and notation

The covariance model applies to residual normalized flux after the selected
deterministic model (for example, a transit model and baseline) has been
evaluated.  Let \(\Delta t = |t_i-t_j|\), measured in days, and let
\(\sigma\) be a correlated-noise amplitude in the same units as normalized
flux.  If amplitudes are reported in ppm, that unit must be stated explicitly.

Length scales such as \(\rho\) and \(\tau\) are in days.  Angular frequencies
are in radians per day.  Whenever a sampler uses logarithmic coordinates for a
dimensional quantity, the log is taken relative to the declared reference unit
(for example, \(\log(\rho / 1\,{\rm day})\)); the reference unit belongs in the
run metadata.

## Candidate kernels

### Ornstein--Uhlenbeck / Matérn-1/2

The exponential covariance is

\[
k_{1/2}(\Delta t) = \sigma^2 \exp\left(-\frac{\Delta t}{\rho}\right).
\]

It is suitable for short-memory stochastic variability, but its sample paths
are not differentiable.  It should not be used merely because it is
computationally convenient: residual diagnostics and comparison models must
support the choice.

### Matérn-3/2

The Matérn-3/2 covariance is

\[
k_{3/2}(\Delta t) = \sigma^2
\left(1 + \sqrt{3}\frac{\Delta t}{\rho}\right)
\exp\left(-\sqrt{3}\frac{\Delta t}{\rho}\right).
\]

It provides once-mean-square-differentiable variations and is often a useful
starting comparison model for smoothly correlated photometric residuals.  It
is a model choice, not a project-wide default.

### Stochastically driven, damped harmonic oscillato

For celerite-compatible implementations, a stochastically driven harmonic
oscillator (SHO) is commonly specified through the power spectral density

\[
S(\omega) = \sqrt{\frac{2}{\pi}}
\frac{S_0\,\omega_0^4}
{(\omega^2-\omega_0^2)^2 + \omega_0^2\omega^2/Q^2}.
\]

Here \(\omega_0\) is the characteristic angular frequency, \(Q\) is the
quality factor, and \(S_0\) sets the PSD normalization.  The critical-damping
boundary is \(Q=1/2\).  \(Q=1/\sqrt{2}\) is a frequently used
granulation-like, non-oscillatory choice; it is not the critical-damping
value.  Values \(Q>1/2\) are underdamped, and large values can represent
quasi-periodic structure.

## Planned implementation and validation requirements

Before a GP result can be used scientifically, the implementation must:

1. fit the transit, baseline, and GP jointly to the actual candidate-local
   time, flux, and uncertainty arrays;
2. retain the selected kernel, priors, parameter units, masks, input hashes,
   package and code versions, random seed, posterior summaries, and
   convergence diagnostics in a candidate-local run record;
3. compare the selected model with a white-noise baseline and at least one
   scientifically motivated alternative where the data support that comparison;
4. demonstrate that the detrending does not absorb or distort injected transit
   signals through injection--recovery tests; and
5. record any exclusion of cadences, flares, momentum dumps, or other data
   treatments that can influence the inference.

A GP configuration is therefore evidence for a particular run, not a global
assumption that applies to every candidate.

## Reporting boundary

Until those requirements are met, documentation may say that GP modelling is
planned or available for future integration, but it must not report a kernel,
hyperparameter posterior, evidence comparison, or GP-improved uncertainty as
an obtained result.  Once implemented, manuscript methods should identify the
kernel, priors, units, data treatment, diagnostics, and robustness checks
sufficiently for an independent reader to reproduce the inference.

## Retained source and provenance gap

- Foreman-Mackey, D., Agol, E., Ambikasaran, S., & Angus, R. (2017),
  *Fast and Scalable Gaussian Process Modeling with Applications to
  Astronomical Time Series*, AJ, 154, 220, ADS `2017AJ....154..220F`, DOI
  `10.3847/1538-3881/aa9332`.

Other GP reference material is not retained primary literature in this
repository. This document therefore does not use it to claim a calibrated
EXONYM GP implementation.
