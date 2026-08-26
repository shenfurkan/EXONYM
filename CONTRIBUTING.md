# Contributing to EXONYM

EXONYM accepts fixes, tests, documentation, and method improvements that preserve candidate isolation and the project’s scientific boundaries.

## Before you change code

Keep target-specific identifiers, observations, ephemerides, sectors, figures, and research outputs under `candidate/<candidate-id>/`. Shared source, schemas, templates, tests, and documentation must remain target-neutral. Tests use synthetic inputs only.

Read `AGENTS.md` before changing source, schemas, templates, tests, workflow gates, or candidate records. Keep reusable behavior out of the CLI dispatcher. Update root and packaged schema or template mirrors in the same patch.

## Development checks

Add deterministic regression coverage for changed behavior. Use focused tests for the affected modules rather than the full suite during normal development. Editorial-only changes do not require pytest, but target-neutral documentation and agent-rule changes must be checked for identifier leaks. Run source isolation and schema checks after shared changes. Review optional dependencies and licenses before adding an external engine.

Do not use an exploratory diagnostic as a validation claim. New scientific artifacts must state their inputs, units, assumptions, uncertainty treatment, applicability limits, and candidate-owned provenance.

## Pull requests

Describe the behavioral change, affected artifact contracts, documentation/skill contracts, and test coverage. Keep unrelated formatting or refactoring out of the same change. Because `AGENTS.md`, `docs/`, and `.agents/` are ignored by broad repository rules, inspect their contents directly and include explicit file-level evidence in the handoff. Do not include candidate payloads, secrets, or private paths.
