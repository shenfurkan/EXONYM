---
title: 'EXONYM: candidate-isolated, reproducible transit-signal research'
tags:
  - Python
  - astronomy
  - exoplanets
  - reproducibility
authors:
  - name: 'AUTHOR METADATA PENDING'
    orcid: 'https://orcid.org/0000-0000-0000-0000'
    affiliation: 1
affiliations:
  - name: 'AFFILIATION METADATA PENDING'
    index: 1
date: 25 August 2026
bibliography: paper.bib
---

# Summary

EXONYM is a Python command-line package for organizing transit-signal research one candidate at a time. It separates reusable methods from candidate-owned observations, identifiers, decisions, and outputs. Shared source code remains target-neutral, while every candidate workspace retains its inputs, provenance, diagnostics, and release material.

The package covers ingestion, light-curve preparation, transit searches, fixed-ephemeris screening, catalog context, pixel localization, stellar diagnostics, transit fitting, timing analysis, and reproducibility exports. Each scientific command records its assumptions and limitations. The current workflow does not make statistical validation claims.

# Statement of need

Transit-signal studies combine archive products, catalog context, numerical models, and human review. A collection of scripts can reproduce an individual result, but it often leaves provenance, candidate identity, and decision records in different locations. This makes review and later replay difficult, especially when several targets share one code checkout.

EXONYM provides a candidate-isolated workspace model for that problem. Candidate-specific data stays below `candidate/<candidate-id>/`; shared modules, tests, schemas, and templates stay neutral. The isolation audit checks file ownership, catalog aliases in shared text, reparse points, schema structure, and target-like literals in shared source. The result is a clear boundary between a generic method and the evidence used for one target.

# State of the field

Python astronomy software already provides the numerical and domain libraries needed for much of this work. Astropy supplies time, coordinate, and FITS support [@astropy:2018; @astropy:2022]. Lightkurve supports analysis of space-based light curves [@lightkurve:2018]. NumPy and SciPy provide array operations and numerical routines [@numpy:2020; @scipy:2020], while emcee supports ensemble Markov-chain Monte Carlo sampling [@emcee:2013]. EXONYM connects these tools through candidate-owned inputs, structured artifacts, and lifecycle checks rather than replacing their numerical implementations.

The package also records limits that remain outside its present calibration. Search scores are ranking diagnostics, localization is a source-competition screen, and optional external adapters are single-hypothesis interpretations. A blocked analysis gate prevents these outputs from being promoted into a statistical validation claim.

# Quality control

EXONYM uses deterministic synthetic tests for shared methods and JSON Schema 2020-12 contracts for candidate records, provenance sidecars, selected scientific outputs, and optional-adapter reports. Root schemas are mirrored in packaged resources so installed wheels retain the same contracts. Candidate releases include a content manifest and a detached digest; release verification replays the frozen source and workspace without rerunning network services or scientific engines.

The repository keeps optional external engines separate from the verified core. An adapter records the package version, declared candidate input, raw result, and normalized report. Unsupported interfaces and unavailable packages produce a candidate-local status manifest rather than a placeholder scientific result.

# Acknowledgements

AUTHOR ACKNOWLEDGEMENT METADATA PENDING. This paper cites the open-source projects used by EXONYM and should be updated with project-specific acknowledgements before submission.

# References
