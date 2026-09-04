# Capability assessment

## Strengths

EXONYM already solves operational problems that many small candidate projects leave informal:

- Candidate-local isolation enforces checks intended to prevent target facts from leaking into shared code and documentation.
- Provenance sidecars, JSON schemas, gate records, and lifecycle events make decisions traceable.
- SPOC light-curve and target-pixel ingestion use canonical file names and sidecars.
- The framework includes fixed-ephemeris checks, archive queries, pixel-depth screening, dilution checks, phase-curve screening, transit fitting, and follow-up records.
- The novelty audit has a schema, evidence digests, candidate matching, and expiry handling.

These are a credible foundation for an independent research workflow.

## Gaps that block an original-candidate programme

| Capability | Current limitation | Required change |
| --- | --- | --- |
| Search preparation | The survey search compares frozen-sector controls and the standalone `exonym detrend` command offers running-median, Wotan, and GP-celerite backends. Population-wide calibration, alternate-aperture coverage, and broad per-sector quality surfaces remain limited. | Add survey-scale quality and sensitivity reports without weakening candidate-local provenance. |
| Detection reliability | The survey search runs inverted-flux and deterministic scrambled-flux null controls. A project-specific population false-alarm rate and empirical alert calibration are not yet provided. | Add population-level calibration and retain the existing null-control artifacts. |
| Completeness | Candidate-level sensitivity runs fixed injection-recovery trials across periods, durations, depths, and phases, but explicitly declares itself ineligible for completeness claims. Broad recovery surfaces across stellar/crowding regimes remain absent. | Add a preregistered survey completeness study before making population claims. |
| Multi-signal discovery | No iterative masking and re-search. | Add candidate masking, re-search, and alias bookkeeping. |
| Data validation | Gates, schemas, provenance hashes, engine manifests, and automated triage check record structure and readiness; they do not validate the scientific judgement behind a record. | Keep machine checks and independent scientific review separate. |
| Localization | Pixel depth mapping and nonnegative Gaussian source screening are implemented. Calibrated mission PRF library fitting, formal position uncertainty propagation, and source-scene constraints are not yet provided. | Implement difference-image/PRF fitting with uncertainty propagation and source-scene constraints. |
| FPP | Native vectorized TREX engine and TRICERATOPS wrapper provide probabilistic candidate false-positive screening (FPP/NFPP) with hash-bound inputs; full scene constraints and high-resolution imaging limits remain required for formal validation claims. | Integrate calibrated difference-image scene constraints, host-star posteriors, and high-resolution contrast limits into the vetting decision pipeline. |
| Follow-up | Candidate-local RV ingest/fit and specialized adapter records exist, while contrast curves, spectroscopy summaries, and resolved imaging constraints remain incomplete. | Add provenance-bound follow-up evidence models as separate claim-gated inputs. |
| Replay | `freeze` and `verify-release` validate detached hashes and replay offline source/workspace loading; they do not rerun scientific engines, network retrievals, or external services. | Preserve the distinction between integrity replay and scientific reproduction. |

## High-risk design issues

### Workflow completion is not scientific completion

Checklist and phase gates certify that required records exist. They must remain separate from a claim that the underlying evidence is sufficient. The public release model should require machine-checked evidence provenance plus independent scientific review.

### Synthetic demonstrations must fail closed in discovery mode

Synthetic outputs are useful for tests and demonstrations, but a blind-discovery command should stop when real, readable candidate photometry is absent. A discovery workflow should never continue from a demonstration output.

### Candidate novelty needs geometric matching

Identifier checks are insufficient. A novelty audit needs sky-position, period, epoch, duration, and harmonic comparisons against current TOI, community-candidate, EB, variable-star, and literature records. The audit must record query time and versions because catalogues change.

## Definition of done for a first original candidate

1. The blind search configuration and input set were frozen before human review.
2. The signal appears in multiple events and survives alternate detrendings and apertures.
3. Project-specific null tests and injection-recovery results support the alert's reliability and sensitivity.
4. DV and TPF diagnostics do not identify an instrumental, EB, or off-target explanation.
5. Position and ephemeris novelty checks are current and recorded.
6. The host scene, dilution, and stellar properties are adequate for the stated claim.
7. The release states "independently detected candidate" unless a separate validation or confirmation evidence package meets the stronger standard.

## Source and unit audit

Every capability above must retain the units and primary-source scope of its
production caller. The target-neutral
[`../../scientific-method-contract.md`](../../scientific-method-contract.md)
indexes every `src/exonym/` module, identifies formula-bearing APIs, and names
the retained ADS/DOI sources. It also records unresolved provenance gaps rather
than treating a convenient planning relation or external catalog transform as a
calibrated physical result.
