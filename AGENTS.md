# AGENTS.md — EXONYM

## Core Invariant (read this first)

**No target-specific data, identifiers, or constants may exist outside `candidate/`.**
Every byte outside `candidate/` must be demonstrably target-neutral.
Run `exonym verify` after any edit that could violate this — it enforces five layers:
1. Forbidden top-level dirs (`archive/`, `data/`)
2. Research payload extensions (`.fits`, `.csv`, `.png`, `.ipynb`, …) outside `candidate/`
3. TOI / TIC / catalog ID strings in neutral-zone text
4. Registered candidate aliases (from `candidate.json` records) in neutral-zone text
5. AST scan of `src/`: no numeric literals bound to sector/ephemeris variable names

## Essential Commands

```powershell
# Install (Python 3.9 required — pyproject.toml enforces ==3.9.*)
pip install -e ".[test]"                         # core + test deps
pip install -e ".[discovery]"                    # optional TLS discovery engine
pip install -e ".[screening]"                    # optional TRICERATOPS screening
pip install -e ".[asteroseismology]"             # optional analysis extras

# Run all tests (takes ~3 minutes due to BLS search)
python -m pytest -q

# Run a single test file
python -m pytest tests/test_gates.py -v

# Isolation + schema audit (run after any structural edit)
exonym verify
exonym verify --schemas-only                     # schema validation only

# Compile check (quick)
python -m compileall -q src tests

# CLI entry point
python -m exonym <command> [--root <repo_root>]   # works before the script is on PATH
```

## CLI Commands

| Command | Purpose |
|---|---|
| `exonym init <id> --tic <n> --mission tess` | Provision an independent-discovery workspace + clone templates |
| `exonym ingest <id> --sectors 14 15 --exptime 120` | Download SPOC FITS + write provenance sidecars |
| `exonym ingest <id> --products tp --sectors 47` | Download SPOC target pixel files (canonical `sNNNN_tp.fits` + sidecars) |
| `exonym advance <id>` | Validate gate and promote workflow phase |
| `exonym set-state <id> --state paused --reason "..."` | Set lifecycle state (safe, event-logged; never hand-edit candidate.json) |
| `exonym track <id>` | ANSI dashboard of checklist completion |
| `exonym freeze <id> --version v1.0.0` | Build reproducibility bundle |
| `exonym verify` | Full isolation + schema audit |
| `exonym search <id> [--engine bls|tls]` | BLS or optional TLS transit search |
| `exonym vet <id>` | TRICERATOPS Monte Carlo FPP simulation (writes `outputs/triceratops_report.json` & `claims/fpp_claim.json`) |
| `exonym plot <id>` | Diagnostic figures to `figures/` |

## Architecture

```
src/exonym/         ← target-neutral library; zero candidate constants allowed
candidate/<id>/     ← all research payload; isolated per target
schemas/            ← JSON Schema 2020-12 for candidate.json, provenance, claims
templates/          ← cloned into every workspace created by exonym init
tests/              ← synthetic fixtures only; no real target data
policy/isolation-exceptions.json  ← approved isolation rule exceptions (requires expiry date)
```

**CI** (`.github/workflows/policy.yml`) runs on every push/PR:
1. `python -m pytest -q`
2. `python -m exonym --root . verify`
3. `python -m exonym --root . verify --schemas-only`

**Verification audit** (`exonym verify`) enforces isolation rules and schema integrity before push/PR.

## Workflow Phases & Gate Rules

Sequential 7-phase state machine. `exonym advance` blocks unless **all** `[MANDATORY]` checkboxes in the phase document are checked AND programmatic gate conditions pass:

| Phase | Gate document | Extra programmatic gate |
|---|---|---|
| `intake` | `docs/01_intake_manifest.md` | — |
| `feasibility` | `docs/02_feasibility_report.md` | — |
| `acquisition` | *(none)* | Every `.fits` in `data/raw/` has a `.provenance.json` sidecar |
| `vetting` | `docs/03_spoc_dv_vetting.md` | — |
| `followup` | `docs/04_tfop_sg_followup.md` | — |
| `analysis` | *(none)* | A `claims/*.json` with `parameter=fpp` and `value < 0.01` exists |
| `review` | `decisions/review_gate.md` | Advancing locks lifecycle to `published` |

Checklist syntax: `- [x] [MANDATORY] description` / `- [ ] [MANDATORY] description`
Gate records written to `candidate/<id>/gates/gate-NNN-<phase>.json`.
Lifecycle events appended to `candidate/<id>/lifecycle/events.jsonl`.

## Schema System

- `schemas/candidate.schema.json` — `candidate.json` (Schema v2, `additionalProperties: false`)
- `schemas/provenance.schema.json` — `*.provenance.json` sidecars (SHA-256 64-char hex + URI)
- `schemas/claim.schema.json` — `claims/*.json` (parameters: `period_days`, `radius_earth`, `mass_earth`, `fpp`)
- `legacy-project/` subtrees are explicitly excluded from schema validation (`schemas.py:LEGACY_SUBTREE`)

`candidate.json` required fields: `schema_version` (must be `2`), `candidate_id`, `identifiers` (with `toi`, `tic`, `aliases`), `lifecycle`, `workflow`, `scientific_disposition`, `publication`, `created_at`.
Always include `mission` in `identifiers` — it is optional in the schema but required for completeness.

## Vetting Maths (do not change thresholds without a C3 protocol change)

| Test | Formula | Pass threshold |
|---|---|---|
| Centroid Z | `sqrt((Δα·cos δ)² + (Δδ)²) / σ` | Z < 3.0 σ |
| Odd-even Z | `\|d_odd − d_even\| / sqrt(σ_odd² + σ_even²)` | Z < 3.0 σ |
| FPP | TRICERATOPS output | FPP < 0.01 |

## Isolation Exceptions

To suppress a genuine violation, add an entry to `policy/isolation-exceptions.json`:
```json
{ "path": "...", "line": 42, "rule": "target-id-in-neutral-zone", "reason": "...", "expires": "2027-01-01" }
```
Key tuple `(path, line, rule)` must match exactly. Entries require an expiry date.

## Testing Policy

- **Skip `pytest` for non-code tasks** (editorial edits, metadata tagging, markdown updates).
- **Run `pytest` + `exonym verify`** when modifying `src/`, `tests/`, `schemas/`, or `templates/`.
- Run both when milestone drafts or gate sign-offs are being recorded.
- `tests/conftest.py` auto-sets `EXONYM_REPO_ROOT` so `test_self_check_of_actual_repository` runs without environment configuration.
- The BLS search tests take ~2–3 minutes total — this is expected.

## Common Pitfalls

- **Do not hardcode sector numbers or ephemeris values in `src/`** — the AST scanner will flag them. Pass all target-specific parameters as function arguments.
- **`exonym init` ID format**: lowercase, `^[a-z0-9][a-z0-9._-]*$`, no Windows reserved names (`CON`, `NUL`, `COM1`…).
- **`freeze` requires `requirements-lock.txt` at the repo root** — it fails with `FileNotFoundError` if absent.
- **Provenance sidecar naming**: `<stem>.provenance.json` (not `<full_filename>.provenance.json`) — `gatekeeper.py` uses `p.with_name(p.stem + ".provenance.json")`.
- **TPF downloads must use `exonym ingest --products tp`** — hand-rolled download scripts have repeatedly produced mis-named sidecars (`s0001_tp.fits.provenance.json` fails the acquisition gate).
- **Never hand-edit `candidate.json` from a shell** — PowerShell/JSON round-trips corrupt the file (BOM, `\u003e` escapes). Use `exonym set-state <id> --state <s> --reason "..."` (validated + event-logged). `set-state` requires a reason when leaving `published`/`archived`.
- **`exonym archive` Gaia results are validated against target presence (≤2″)** — check the `backend` and `validated` fields in `archival_vetting_report.json`; a `validated: false` result may be a stale mirror and must be re-run or treated as degraded. Note the default `--radius-arcsec` is 10″ — bright neighbors beyond it (2026-08-05: a G≈14 star at 25″) are invisible to the archival report but matter for dilution; widen the radius for crowding studies.
- **Asteroseismology scaling can return absurd parameters from noise** (e.g., M≈26 M☉). Results are flagged `scaling_rejected_unphysical` in `asteroseismic_results.json` via `seismic_sanity_check` — trust only `plausible: true` outputs.
- **BLS wide-range searches can return a 2×/3× harmonic alias of the true period** — `find_transits` now refines the peak and tests fractional-phase offsets to recover the fundamental period (see `tests/test_search.py::test_find_transits_resolves_double_period_alias`). Still sanity-check recovered periods against the catalog.
- **`exonym verify` runs against the working directory** — always run from repo root.
- **Adding a new candidate alias in `candidate.json`** triggers alias-leak checks against all neutral-zone text — run `exonym verify` after.
- **`legacy-project/` directories** are deliberately excluded from schema validation and isolation checks; do not move their content out of that subtree.
- **`python -m exonym`** also works as the CLI entry (equivalent to `exonym`), useful before the package is installed.
