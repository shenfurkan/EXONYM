# Transit Fitting Protocol :: {{CANDIDATE_ID}}

Target: {{TOI}} / {{TIC}} | Frozen: {{TIMESTAMP}} | Phase: analysis

## Prior Specification

Document the model family, parameterization, priors, bounds, and transforms
before any result-bearing execution:

- Orbital period and epoch prior sources
- Radius ratio, impact parameter, and scaled semimajor-axis priors
- Limb-darkening treatment and its stellar parameter inputs
- Detrending window and baseline polynomial selection
- Noise model (white or correlated) and jitter treatment

## Noise model status

The current transit-fitting pipeline uses independent white errors with an
additive jitter term. It does not run a Gaussian-process noise model. Treat
this as a white-jitter baseline, not as evidence that correlated noise has
been modeled.

Before a future Gaussian-process result supports a scientific claim, record:

- Kernel family, implementation, and package version
- Input flux, uncertainties, quality masks, and detrending choices
- Hyperparameter priors, fitted values, and constraints
- Optimizer or sampler, initialization, random seed, and convergence diagnostics
- Residual and posterior-predictive checks against the white-jitter baseline
- Run provenance, configuration, and artifact hashes

## Sampling and Numerical Execution

- Optimizer/sampler and start configuration
- Convergence diagnostics (R-hat, integrated autocorrelation time)
- Random seed policy and worker layout

## Synthetic Calibration

- Calibration classes and truth distributions
- Coverage and bias metrics
- Failure thresholds and stop rules

## Required Artifacts

List every output artifact with its schema and no-clobber rule.

## Binding Gates

| Gate ID | Metric | Threshold | PASS effect | FAIL effect |
|---|---|---|---|---|
| | | | | |
