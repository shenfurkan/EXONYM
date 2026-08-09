# EXONYM

> **Target-neutral Python framework & CLI for TESS exoplanet candidate disposition & characterization.**
> Integrates automated BLS transit detection, photocenter offset statistics, odd-even depth comparisons, and Bayesian TRICERATOPS FPP calculations into candidate-isolated, 7-phase gated workflows with complete reproducibility tracking.

## Etymology & philosophy

The name **Exonym** comes from Ancient Greek **ἔξω** *éxō* meaning outside and **ὄνομα** *ónoma* meaning name. In linguistics, an [exonym](https://en.wikipedia.org/wiki/Exonym_and_endonym) is a name assigned to an entity or place by external observers, unlike an [endonym](https://en.wikipedia.org/wiki/Exonym_and_endonym) *éndon* meaning inside which is the native internal name.

In exoplanet astronomy, catalog designations like TIC, TOI, or Gaia IDs are external labels given by Earth-based surveys to distant star systems. We built EXONYM around this distinction:

- **Target-neutral core**: The shared library `src/exonym/` acts purely as an outside observer. It contains zero hardcoded candidate IDs, sector numbers, or ephemerides.
- **Isolated candidate data**: All target-specific identifiers, raw FITS data, light curves, decisions, and claims live strictly inside `candidate/<candidate-id>/`.
- **Reproducible research**: Decoupling software logic from candidate state prevents hardcoding errors and keeps every target workspace self-contained.

## Workflow & gate methodology

Candidate disposition proceeds through a sequential 7-phase state machine governed by programmatic gate checks via `exonym advance`:

```text
intake ──> feasibility ──> acquisition ──> vetting ──> followup ──> analysis ──> review
```

- **Phase Checklist Sign-offs**: Progression through `intake`, `feasibility`, `vetting`, `followup`, and `review` requires candidate-local markdown checklist completion `- [x] [MANDATORY] ...`.
- **Programmatic Gate Conditions**:
  - `feasibility` & `review`: Require an active, hash-backed `decisions/novelty_audit.json` record.
  - `acquisition`: Enforces matching `<stem>.provenance.json` sidecars with SHA-256 hashes and URIs for all raw FITS products under `data/raw/`.
  - `analysis`: Requires a candidate-local claim artifact `claims/fpp_claim.json` confirming a false-positive probability below the protocol threshold $\text{FPP} < 0.01$.
  - `review`: Passing review permanently locks candidate lifecycle state to `published`.
- **Target Isolation Invariant**: Evaluated on every change via `exonym verify`. The shared codebase `src/exonym/` is scanned via AST parsing to prohibit numeric literals bound to sector or ephemeris variable names.

## Vetting tests & scientific capabilities

EXONYM implements automated screening tests and exploratory astrophysics modules:

| Test / Analysis | Method & Formula | Pass Threshold / Output |
| --- | --- | --- |
| **BLS Transit Search** | Box Least Squares period grid refinement & fractional-phase harmonic resolution (`exonym search`) | Recovered period, duration, epoch, & depth |
| **Odd-Even Depth Test** | $Z = \|d_{\text{odd}} - d_{\text{even}}\| / \sqrt{\sigma_{\text{odd}}^2 + \sigma_{\text{even}}^2}$ | $Z < 3.0\sigma$ |
| **Photocenter / Centroid Z** | $Z = \sqrt{(\Delta\alpha \cos\delta)^2 + (\Delta\delta)^2} / \sigma$ | $Z < 3.0\sigma$ |
| **False-Positive Probability** | TRICERATOPS Bayesian Monte Carlo scenario probability calculation (`exonym vet`) | $\text{FPP} < 0.01$ |
| **Sub-pixel PRF Localization** | Pixel Response Function source location on Target Pixel Files (`exonym localization`) | Pixel-level transit source offset |
| **Archival & Dilution** | Gaia DR3 neighbor search with 2″ target validation and wide-radius dilution assessment (`exonym archive`, `dilution`) | Neighboring star contamination factor |
| **Exploratory Physics** | MCMC transit fitting `fit`, TTV analysis `ttv`, SED fitting `sed`, Asteroseismology scaling sanity check `asteroseismology` | Fitted parameters with physical sanity flags |

## Quickstart

### Installation

Requires Python `3.9.*`.

```powershell
# Core package & test suite
pip install -e ".[test]"

# With optional screening and asteroseismology
pip install -e ".[test,screening,asteroseismology]"
```

### Usage workflow

```powershell
# Initialize a candidate workspace
exonym init candidate-id --toi <TOI-NUMBER> --tic <TIC-NUMBER> --mission tess

# Check workspace boundary & track phase
exonym verify
exonym track candidate-id

# Ingest TESS light curves & target pixel files
exonym ingest candidate-id --products lc,tp

# Fetch catalog priors & run signal search
exonym fetch-priors candidate-id
exonym search candidate-id
exonym screen candidate-id --signal .01

# Validate current phase gate & advance
exonym advance candidate-id
```

## Repository & workspace layout

```text
src/exonym/                 Target-neutral Python library and CLI
candidate/<candidate-id>/   Isolated research workspace per target
  candidate.json            Identity, lifecycle, workflow, & publication state
  config/                   Signal priors and local configuration
  data/                     Raw FITS files and .provenance.json sidecars
  docs/                     Phase checklists 01_intake through 04_followup
  decisions/                Novelty audit and review gate sign-offs
  outputs/, figures/        BLS search results, TRICERATOPS reports, plots
  claims/                   Structured scientific assertions and FPP claims
  gates/, lifecycle/        Gate validation records & audit log
  releases/                 Frozen reproducibility bundles
schemas/                    JSON Schema Draft 2020-12 definitions
templates/                  Workspace templates cloned on init
tests/                      Synthetic, target-neutral unit tests
policy/                     Target-isolation policy registry
LICENSE                     GNU General Public License v3.0 text
```

## CLI command reference

| Command | Category | Description |
| --- | --- | --- |
| `init` | Workspace | Provision a new isolated candidate workspace. |
| `list` | Workspace | List registered candidate workspaces with filters. |
| `status` | Workspace | Display candidate metadata and workspace paths. |
| `track` | Workspace | Render phase-progress dashboard and checklists. |
| `advance` | Workspace | Validate current phase gate and promote candidate phase. |
| `set-state` | Workspace | Audit-log candidate lifecycle state transitions. |
| `freeze` | Workspace | Build a candidate-local reproducibility bundle. |
| `verify` | Audit | Audit target-isolation and JSON schemas `--schemas-only`. |
| `ingest` | Screening | Download SPOC light curves / TPFs and write provenance sidecars. |
| `fetch-priors` | Screening | Retrieve TIC catalog transit priors into `config/signals/`. |
| `search` | Screening | Perform targeted or blind BLS transit search. |
| `screen` | Screening | Measure fixed-ephemeris primary, odd-even, half-phase, and doubled-period alternating-event diagnostics. |
| `vet` | Screening | Run TRICERATOPS Monte Carlo FPP false-positive simulation. |
| `archive` | Screening | Query Gaia DR3 and ExoFOP archival evidence. |
| `plot` | Screening | Render diagnostic figures to `figures/`. |
| `fit`, `ttv`, `sed` ... | Analysis | Exploratory transit fitting, TTV, SED, phase curve, activity, localization. |

## Verification & testing

Run the test suite and isolation audit before submitting changes:

```powershell
python -m pytest -q
exonym verify
```

## License

EXONYM is open-source software licensed under the **GNU General Public License v3.0**. See [LICENSE](file:///d:/Exonym/LICENSE) for details.
