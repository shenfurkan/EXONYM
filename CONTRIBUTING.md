# Contributing

## Scope

Contributions may improve the shared pipeline (`src/`), repository policy
(`docs/`, `schemas/`, `policy/`), shared tests (`tests/`), or a specific
candidate workspace (`candidate/<candidate-id>/`). Target research never
leaves `candidate/`.

## Before Work

1. Read `docs/governance/README.md` and `docs/lifecycle.md`.
2. Classify the change: C0 editorial, C1 implementation, C2 input/data,
   C3 protocol, C4 claim, C5 release, C6 security.
3. Inspect `git status`; do not overwrite unrelated work.
4. Check `exonym verify` before and after.

## Rules

1. No target identifiers, aliases, fixed sectors, ephemerides, or target
   payload formats outside `candidate/`.
2. Shared code takes target identity, paths, sectors, and thresholds as
   explicit inputs; no real candidate defaults.
3. Shared tests use synthetic fixtures only.
4. Candidate-specific code imports `exonym`; it never reads a sibling
   candidate.
5. Frozen protocols, results, gates, and release receipts are append-only.
   Corrections use new versioned files with explicit supersession links.
6. Never weaken a test or threshold to make a result pass; a gate change is a
   protocol change (C3).

## Verification

```powershell
python -m compileall -q src tests
python -m pytest -q
exonym verify
```

> **Testing Policy**: To prevent unnecessary operational overhead ("red tape"), automated test suites (`pytest`) do not need to be run after minor tasks where no Python code (`src/`, `tests/`) has been created or modified. Once a candidate manuscript or campaign milestone takes shape as a draft, test suites should be executed more frequently to ensure data integrity and baseline stability.

Report exact failures; do not hide known baseline failures.

## Policy Enforcement

The repository guard layer runs on every push and pull request:

1. `python -m pytest -q` — shared-layer unit tests.
2. `exonym verify` — repository isolation audit.
3. `exonym verify --schemas-only` — JSON schema validation of candidate
   records, provenance sidecars, and claim assertions.

Install the local pre-commit hook to run the audit before every commit:

```powershell
pip install pre-commit
pre-commit install
```

The hook runs `exonym verify` on the full repository (`always_run`) and
rejects target identifiers, research payloads, symlinks, and schema
violations outside `candidate/`. Policy, schema, and template files are
owner-reviewed (`CODEOWNERS`); exceptions require an entry in
`policy/isolation-exceptions.json` with an expiry date.

## Commit and Release

Commit only intended files. Do not commit secrets. Do not tag, publish,
deposit, or claim a DOI without explicit release authorization and a passing
release gate.
