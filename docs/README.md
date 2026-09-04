# Documentation

Target-neutral documentation and record templates.

| Path | Purpose |
| --- | --- |
| `governance/README.md` | Roles, change classes, claims, and review authority |
| `lifecycle.md` | Candidate lifecycle, states, and archive semantics |
| `EXONYM_SCIENTIFIC_ARCHITECTURE_AND_HOW_TO.md` | End-to-end scientific pipeline, uncertainty contracts, calibration roadmap, and paper workflow |
| `scientific-method-contract.md` | Source-module inventory; verified ADS/DOI formula register; exact unit, applicability, fail-closed, and claim-ineligibility contract |
| `EXONYM_SCIENTIFIC_ARCHITECTURE_AND_HOW_TO.md#scientific-debugging-and-incident-response` | Evidence-first diagnosis of scientific, numerical, provenance, and optional-engine failures |
| `discovery-readiness/` | Current survey/discovery capability assessment, remaining calibration gaps, and claim ladder |
| `templates/` | Generic record scaffolds inside `docs/` (charter, protocol, dataset, gate, decision, postmortem, handover, run record) |
| `../methods/` | Root `methods/`: command-level scientific method records, literature notes, and interpretation limits |
| `../protocols/` | Root `protocols/`: frozen protocol definitions and their record scaffolds |

The root `templates/` directory is a source-checkout resource (mirrored under
`src/exonym/_resources/`) and is not a documentation location; `docs/templates/`
is the documentation record-scaffold directory described above.

Target-specific records never belong here. They live in
`candidate/<candidate-id>/` (for example under that candidate's `docs/`,
`protocols/`, `gates/`, and `claims/` directories).

Operational workspace checkpoints, engine runs, telemetry, and automation
manifests are candidate-local records; this directory documents their contract
but never stores their payloads.
