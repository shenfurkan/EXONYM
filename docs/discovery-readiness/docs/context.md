# Research context

## Questions and outcomes

1. Can EXONYM support its own TOI-free discoveries? It can operate bounded TOI-free surveys, harvest novelty evidence, and triage real candidate data, but its population-level detection reliability and source-vetting methods are not calibrated enough for a public planet claim.
2. Does the project meet all NASA or exoplanet-handbook standards? No finite public standard set certifies this kind of independent codebase. The relevant public practice is documented here, and several required capabilities remain incomplete.
3. What should the project build next? Population-level detection/completeness calibration and calibrated scene localization first, then a scene-aware validation workflow and richer follow-up ingestion.

## Local context examined

- Shared workflow and scientific-use boundaries in the repository README.
- Candidate lifecycle and novelty-audit requirements in `docs/lifecycle.md`.
- Workspace templates and gates under `templates/` and packaged resource templates.
- Ingestion, search, screening, localization, archive, FPP, release, schema, and gate modules under `src/exonym/`.
- Tests and documentation warnings about synthetic demonstrations and exploratory outputs.

No candidate workspace names, identifiers, measurements, or payloads are included in this target-neutral report.

## External sources consulted

- MAST TESS data products: https://archive.stsci.edu/missions-and-data/tess/data-products
- NASA Exoplanet Archive catalogue philosophy: https://exoplanetarchive.ipac.caltech.edu/docs/ExoPlanArchPhilosophy.html
- NASA Exoplanet Archive TOI field definitions: https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html
- TESS SPOC pipeline and data validation: https://ui.adsabs.harvard.edu/abs/2018AJ....156..171T/abstract
- TESS Follow-up Observing Program: https://tess.mit.edu/followup/
- TRICERATOPS: https://ui.adsabs.harvard.edu/abs/2021AJ....161...24G/abstract
- Morton false-positive methodology: https://doi.org/10.3847/0004-637X/822/2/86
- Ephemeris matching methodology: https://doi.org/10.3847/0067-0049/224/1/12

## Commands used

- Repository content searches and reads through the workspace tools.
- Focused `pytest` commands only when source behavior changes; editorial-only documentation changes do not require pytest.
- `exonym verify --source` and `exonym verify --schemas-only` for shared isolation/schema checks after neutral-zone edits.
- `uvx zensical build` was attempted, but `uvx` is unavailable in this environment.
- `python -m pip install zensical` installed PyPI package version `0.0.2`, which has no command entry point or `__main__` module; `python -m zensical build` and `zensical build` could not run. The site source is present, but its static build remains unverified until a working Zensical distribution is available.

## Limits and follow-up

- This is a code and documentation assessment, not a benchmark of measured detection yield.
- Official NASA pages describe data products and catalogue practice; they do not certify independent discovery pipelines.
- A future report should include population-scale injection-recovery, false-alarm calibration, runtime, and scientific-reproduction results; current sensitivity and release replay records do not provide those claims.
