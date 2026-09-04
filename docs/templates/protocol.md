# Scientific Protocol

Protocol ID: PROTOCOL-YYYY-NNN

Version:

Frozen date:

Status: `DRAFT | FROZEN | EXECUTED | SUPERSEDED`

## Purpose and Claim Dependency

State the exact scientific question and claim(s) governed by this protocol.

## Prior Exposure Disclosure

List target results already observed. State whether this is blind,
prospective-after-exposure, or retrospective.

## Upstream Dependencies

| Dependency | Path/identifier | Version/hash | Required status |
|---|---|---|---|
| | | | |

## Inputs

| Input | Units/schema | Size | SHA-256 | Lineage |
|---|---|---:|---|---|
| | | | | |

## Model and Assumptions

Document equations, likelihood, priors, bounds, transforms, shared/nuisance
parameters, approximations, and known non-identifiability.

## Data Splits and Leakage Controls

Define training, validation, held-out, transit, control, mask, branch, and
overlap rules.

## Randomness and Parallelism

| Item | Rule |
|---|---|
| Master seed | |
| Per-record seed derivation | |
| Worker-order invariance | |
| Random generator/library | |

## Synthetic Calibration

Define classes, realizations, truth distributions, observation sampling,
success metrics, coverage, bias, selector, and failure thresholds.

## Numerical Execution

Record optimizer/sampler, starts, convergence, movement, objective, gradients,
Hessian/covariance, effective sample size, R-hat, MCSE, and resource limits.

## Scientific Failure Diagnosis

State the ordered diagnostic boundary checks and their stop behavior. At a
minimum include ownership/schema/hash lineage; declared time system and units;
finite/domain-valid inputs; random-seed and parallel-mode reproducibility; and
the model's calibration/applicability limit. Classify each failure as software,
environment, data/provenance, applicability, numerical, or interpretation. A
failed or unavailable engine must produce an unresolved execution record, never
a scientific rejection or a claim.

| Boundary | Evidence/artifact | Failure action | Downstream interpretation |
|---|---|---|---|
| Ownership and provenance | | | |
| Units and time system | | | |
| Input applicability | | | |
| Numerical execution | | | |
| Calibration and interpretation | | | |

## Scientific Failure Diagnosis

State the ordered diagnostic boundary checks and their stop behavior. At a
minimum include ownership/schema/hash lineage; declared time system and units;
finite/domain-valid inputs; random-seed and parallel-mode reproducibility; and
the model's calibration/applicability limit. Classify each failure as software,
environment, data/provenance, applicability, numerical, or interpretation. A
failed or unavailable engine must produce an unresolved execution record, never
a scientific rejection or a claim.

| Boundary | Evidence/artifact | Failure action | Downstream interpretation |
|---|---|---|---|
| Ownership and provenance | | | |
| Units and time system | | | |
| Input applicability | | | |
| Numerical execution | | | |
| Calibration and interpretation | | | |

## Required Artifacts

| Artifact | Schema | No-clobber | Verification |
|---|---|---|---|
| | | | |

## Binding Gates

| Gate ID | Metric | Threshold | PASS effect | FAIL effect |
|---|---|---|---|---|
| | | | | |

## Stop Rules

State terminal failure behavior and maximum allowed fallback attempts.

## Claim Effects

| Outcome | Claims opened | Claims remaining closed | Permitted wording |
|---|---|---|---|
| PASS | | | |
| FAIL | | | |

## Approvals

| Role | Name | Date | Protocol hash | Disposition |
|---|---|---|---|---|
| Scientific owner | | | | |
| Independent verifier | | | | |
