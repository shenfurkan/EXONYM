# Roadmap

## Current status

The repository now has frozen survey manifests, TOI-free novelty harvesting,
candidate-level null controls, fixed sensitivity trials, engine-run manifests,
bounded auto-vet/run-loop orchestration, ephemeris matching, and offline
release-integrity replay. These are operational foundations, not population
calibration or validation.

## Phase A: Make discovery statistics trustworthy

**Objective:** produce a calibrated list of transit-like alerts from real photometry.

- The versioned survey manifest, frozen sectors, review threshold, TOI-free registration, and candidate-local search outcomes are operational. Keep extending the denominator record to cover every source row, unavailable target, failed fetch, veto, and alert in a defined data release.
- Candidate-level transit-preserving branches, null controls, and fixed injection-recovery trials are operational. Survey-scale sector quality surfaces, alternate-aperture coverage, and representative observing-window controls remain limited.
- Calibrate detection reliability with a preregistered population of null and injected targets.
- Extend candidate-level injection-recovery into representative stellar, crowding, and observing-window recovery surfaces.

**Exit criterion:** every target has a retained search outcome, and every alert has a calibrated detection statistic, a recovery/completeness context, and a reproducible input manifest.

## Phase B: Make candidate triage defensible

**Objective:** reject ordinary artifacts and false positives before follow-up.

- Candidate-local SPOC light-curve/TPF ingestion, fixed-ephemeris odd-even/half-phase/alternating-event screening, schema checks, provenance checks, and automated triage are operational. SPOC DV product ingestion and a complete machine-checked DV gate remain open.
- Catalog ephemeris matching is operational for fresh NEA/TOI rows and reviewed candidate-recorded BJD_TDB evidence. Broader EB/variable-star provider coverage and human-reviewed harmonic matching remain open.
- Survey search already compares its declared preprocessing branches. General candidate-scale alternate-aperture and detrending orchestration remains open.
- Upgrade localization to calibrated difference-image and PRF fitting with uncertainties.

**Exit criterion:** an independently detected candidate has a complete, machine-checked internal vetting package.

## Phase C: Support validation-quality evidence

**Objective:** decide whether a candidate can be statistically validated.

- SED, MIST consistency, asteroseismic, and archive records are operational exploratory inputs with provenance. A validation-grade stellar-property posterior record remains open.
- Gaia archive context and descriptive RV/adapter records are operational; calibrated contrast curves, reconnaissance spectroscopy, and resolved-photometry constraints remain open.
- Replace the exploratory FPP wrapper with a real-photometry, scene-aware scenario model.
- Report global and nearby-source false-positive probabilities, assumptions, and sensitivity to priors.
- Require independent scientific review before a validation claim.

**Exit criterion:** a release can explain why each planet and false-positive scenario is constrained by candidate-specific evidence.

## Phase D: Confirmation and durable release

**Objective:** retain the scientific record and distinguish statistical validation from mass confirmation.

- Extend the current descriptive RV and orbital-decay/TTV diagnostics with independent dynamical evidence and activity/line-profile inputs.
- Add a confirmation claim type only when a planetary-mass inference and review policy exist.
- Preserve the current freeze/verify-release integrity replay. Add scientific rerun/reproduction only where external services and inputs can be pinned; workspace checkpoints remain recovery snapshots rather than releases.
- Re-run novelty and catalogue checks immediately before release.

**Exit criterion:** the published record can be replayed and its status is stated precisely as candidate, validated planet, or confirmed planet.

## Remaining vertical slice

Keep the current work bounded to a single TESS data-release cohort and do not broaden missions or catalogues yet. The operational parts of the original slice now exist; the remaining work is to harden and calibrate them:

1. Complete population-wide denominator and per-sector quality accounting.
2. Add representative population null/injection calibration and a completeness surface.
3. Add provenance-bound DV products, calibrated localization, and scene constraints.
4. Expand ephemeris/provider coverage with explicit human harmonic review.
5. Replace the exploratory FPP path only after real-photometry scene modeling and independent review are available.

This slice makes the project useful for original-candidate discovery. Validation-quality FPP and follow-up integration come after it.
