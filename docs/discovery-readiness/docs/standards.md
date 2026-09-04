# Standards comparison

## Scope of this comparison

No single public document provides a universal NASA certification standard for an independent exoplanet-discovery codebase. NASA mission pipelines, archive catalogues, and follow-up programs define data products and scientific practice for their purposes. Community papers define common validation methods and thresholds. This assessment compares EXONYM with those public expectations; it does not claim formal NASA approval or compliance.

## Public practice that applies

| Practice | Public basis | EXONYM status |
| --- | --- | --- |
| Preserve the specific light-curve, target-pixel, DV XML, summary and full DV PDFs, and DV time series used for a disposition, including their pipeline-run identifiers | [MAST TESS data products](https://archive.stsci.edu/missions-and-data/tess/data-products) | Raw SPOC products and provenance sidecars are supported. DV product ingestion and pipeline-run capture are incomplete. |
| Review transit consistency, odd/even events, secondary eclipses, out-of-eclipse variability, image localization, contamination, and ephemeris matches | [TESS SPOC pipeline and DV](https://ui.adsabs.harvard.edu/abs/2018AJ....156..171T/abstract) | Odd/even, half-phase, archive context, exploratory localization, and catalog ephemeris matching exist. Calibrated DV/source-scene evidence remains incomplete. |
| Distinguish a candidate, false positive, false alarm, validated planet, and confirmed planet | [NASA Exoplanet Archive philosophy](https://exoplanetarchive.ipac.caltech.edu/docs/ExoPlanArchPhilosophy.html) | The workflow records a disposition, but the scientific criteria are not yet programmatically enforced at this level. |
| Treat TOIs as evolving catalogue records rather than validation claims | [NASA Exoplanet Archive TOI fields](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html) | The discovery-first policy is now documented. Novelty checks need position and ephemeris matching, not only identifiers. |
| Model planet and false-positive scenarios using photometric, stellar, imaging, and field constraints | [Morton 2016](https://doi.org/10.3847/0004-637X/822/2/86), [TRICERATOPS](https://ui.adsabs.harvard.edu/abs/2021AJ....161...24G/abstract) | An optional FPP wrapper exists, but it does not yet use observed light curves, measured contrast curves, or calibrated localization. |
| Obtain follow-up evidence appropriate to blend and companion risks | [TESS Follow-up Observing Program](https://tess.mit.edu/followup/) | Follow-up status, candidate-local RV ingest/fit, and selected adapters exist. Contrast curves, reconnaissance spectroscopy, and resolved imaging ingestion remain incomplete. |

## EXONYM claim taxonomy

The following is EXONYM's proposed claim taxonomy. NASA archive disposition terms inform the candidate and false-positive categories; statistical validation and confirmation require the separate evidence standards described below.

| Claim | Minimum basis | EXONYM position now |
| --- | --- | --- |
| Transit-like signal | Search output in real photometry with retained provenance. | Supported. |
| Independently detected candidate | Blind detection, repeatability across events and data treatments, a dated novelty audit, and initial false-positive screening. | Achievable after discovery-calibration work. |
| Statistically validated planet | Candidate-specific light-curve model, stellar and field constraints, source-location evidence, relevant follow-up, scenario probabilities, and an independently reviewed validation analysis. | Not supported by the current FPP path. See [Morton 2016](https://doi.org/10.3847/0004-637X/822/2/86). |
| Confirmed planet | Direct planetary-mass evidence, usually RVs or a dynamical TTV solution. | External follow-up outcome. |

## NASA data products are evidence, not a certificate

SPOC Data Validation products are designed to support vetting. They do not make an independent project automatically compliant with a NASA standard, and they do not replace candidate-specific review. EXONYM should preserve the exact product URI, retrieval time, pipeline version when available, and the diagnostic outcomes used in each disposition.

## Thresholds

A numerical FPP threshold is conditional on its scenarios, priors, photometry, imaging limits, and stellar information. The project must not treat a generic low-FPP value as a universal validation certificate. Likewise, a detection threshold must be calibrated for the actual period-duration search space, detrending, noise properties, and human triage process. A threshold inherited from another mission or pipeline is not automatically transferable.

The exact implementation-facing unit and source contract is maintained in
[`../../scientific-method-contract.md`](../../scientific-method-contract.md).
It documents why the present TRICERATOPS/TREX path retains
`claim_eligible: false`: a finite conditional FPP/NFPP is not enough without
calibrated, provenance-bound scene-model constraints.
