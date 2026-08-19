<div align="center">
  <img src="images/logo.png" alt="EXONYM Logo" width="75">
  <h1>EXONYM</h1>
</div>

EXONYM is a Python 3.9 command-line tool for working through TESS transit signals one candidate at a time. It creates an isolated workspace, downloads SPOC light curves and target-pixel files, and keeps the evidence, decisions, and outputs together. Shared code never contains a target identifier, sector choice, ephemeris, or research payload.

The project is for screening and exploratory characterization. It cannot confirm or statistically validate a planet on its own. A planet claim still needs checks for systematics, contamination, false-positive scenarios, stellar properties, and, when needed, follow-up photometry, imaging, spectroscopy, or radial velocities.

## Scope and scientific status

EXONYM keeps two things separate:

- The shared code implements generic procedures such as transit searches, fixed-ephemeris screening, catalog context, and posterior sampling.
- Each candidate workspace contains the identity, inputs, provenance, diagnostic products, decisions, claims, and release records for one target.

That separation keeps each result inspectable without turning the shared package into a store of target facts. You can see which inputs, method, and decision led to a conclusion.

You can label a workspace with several missions, but data ingestion and most commands are TESS-focused. Treat the repository thresholds as screening rules, not universal boundaries between planets and false positives.

### Independent discovery policy

Independent discovery starts with a TIC target that has no assigned TOI or cTOI. Record the ExoFOP and literature checks in the intake workspace. A known TOI is still useful for method checks, comparisons, or follow-up, but it is not an independent EXONYM discovery unless the workspace documents a separate contribution.

### Read this before interpreting an output

- Several exploratory commands can produce records marked `source: "synthetic-demo"` when candidate data are unavailable. Those records exercise software paths; they are not scientific evidence for a target.
- `exonym screen` intentionally requires real photometry and a real ephemeris. It does not substitute synthetic data.
- Workflow gates verify the existence, structure, and checklist state of evidence artifacts. They do not independently reproduce the scientific judgement written in a checklist or claim.
- The diagnostic centroid panel produced by `exonym plot` is currently a fixed-seed visualization, not a TPF-derived centroid measurement. Use a measured difference-image or pixel-level analysis for centroid evidence.
- A low false-positive probability, a clean odd-even statistic, or a small centroid offset is necessary evidence in some cases, but none alone proves a planetary companion.

## Design principles

| Principle | What EXONYM enforces |
| --- | --- |
| Target isolation | All target-specific material lives below `candidate/<candidate-id>/`. Shared source, tests, templates, schemas, and documentation stay target-neutral. |
| Evidence traceability | Inputs, derived artifacts, decisions, phase-gate records, and lifecycle events are stored in the candidate workspace. |
| Schema-bound records | Candidate metadata, provenance sidecars, and scientific claims are checked against JSON Schema 2020-12 definitions. |
| Sequential review | A seven-phase workflow prevents a candidate from skipping intake, feasibility, acquisition, vetting, follow-up, analysis, or final review. |
| Reproducibility metadata | Release bundles capture environment definitions, dependency locks, a manifest, and source-control metadata. They are not automatic full data archives. |

### Why the name EXONYM?

According to the [United Nations](https://unstats.un.org/unsd/publication/seriesm/seriesm_88e.pdf), an **exonym** is an externally assigned name used by an outer community to refer to an entity outside its jurisdiction, adapted without altering the original local *endonym*.

In this framework, **EXONYM** serves as an architectural metaphor: the shared codebase acts as a target-neutral outer observer that interacts with candidates solely through external identifiers and standardized placeholders, while the true candidate identity, raw inputs, and research payloads remain strictly isolated within `candidate/<candidate-id>/`.

## Installation

EXONYM requires Python `3.9.*`. The version constraint is exact because the package and its pinned dependencies are tested against that interpreter series.

### First-time setup

Run these commands from the repository root in PowerShell:

```powershell
py -3.9 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m exonym --root . verify
```

Install optional analysis engines separately when they are needed:

```powershell
# Native-cadence Transit Least Squares discovery
python -m pip install -e ".[discovery]"

# TRICERATOPS screening
python -m pip install -e ".[screening]"

# Asteroseismology tools
python -m pip install -e ".[asteroseismology]"

# Confirm the installed command
exonym --version
```

The command-line entry point and module invocation are equivalent:

```powershell
exonym --help
python -m exonym --help
```

Place the optional repository-root argument before the subcommand when operating outside the repository root:

```powershell
exonym --root <repository-root> verify
```

Networked operations need access to their upstream services. In particular, SPOC ingestion needs a catalog identifier and network access, Gaia and ExoFOP queries depend on remote services, and TRICERATOPS and TLS require their optional dependency groups.

## First discovery run

The placeholders below are deliberately generic; do not place real catalog identifiers or target aliases in shared documentation, source code, or tests.

```powershell
# Create a workspace for an independent-discovery target.
exonym init <candidate-id> --tic <tic> --mission tess

# Inspect the workspace created by init and confirm repository isolation.
exonym status <candidate-id>
exonym verify

# Download SPOC light curves and target pixel files for selected sectors.
exonym ingest <candidate-id> --products both --sectors <sector>

# Search the acquired real photometry with the default BLS engine.
exonym search <candidate-id> --engine bls

# If the discovery extra is installed, also run native-cadence TLS.
exonym search <candidate-id> --engine tls

# Inspect outstanding evidence requirements before advancing a phase.
exonym track <candidate-id>
exonym advance <candidate-id>
```

`exonym advance` only checks whether the required records and checklist items are present. It does not decide whether the science is sound. Check a box only after the candidate evidence supports it, and write caveats in the relevant candidate document.

### Blind-discovery surveys

`exonym survey` records a bounded TESS cohort below
`candidate/_surveys/<survey-id>/`. Each registered target keeps an explicit
survey outcome, including novelty-audit blocks and searches without alerts.
Survey search uses only the manifest's frozen sectors and a preregistered SNR
threshold. For each target it runs the following controls before routing any
alert to human review: a BLS search across a one-, two-, and four-hour
duration grid comparing both normalized and per-sector running-median flux;
an inverted-flux null search; three deterministic scrambled-flux searches with
fixed seeds; and candidate-scale transit injection at three phase offsets to
check period and epoch recovery at the survey SNR threshold. An alert requires
the reference BLS, the normalized search, and the duration-grid search to all
clear the threshold, both diagnostic periods to agree with the reference within
one percent, every null-control SNR to stay below the threshold, and at least
two of three injections to recover. The command does not set a scientific
disposition, make a planet claim, or replace the required candidate-local
screening sequence. These controls do not provide a population false-alarm
calibration, a broad completeness map, source localization, or a statistical
validation. Localization, fitting, follow-up, archive checks, and a final FPP
run remain required before an independent-detection claim.

Pass `--toi <toi>` only when a known TOI is deliberately being analyzed for validation, comparison, or follow-up rather than independent discovery.

## Candidate lifecycle and gate logic

Candidate disposition follows one ordered state machine:

```text
intake -> feasibility -> acquisition -> vetting -> followup -> analysis -> review
```

The active phase advances only when its gate passes. `stopped` workspaces cannot advance. For Markdown-gated phases, the parser looks for the required document and checked mandatory items. It checks the checklist structure, not the scientific judgement behind the text.

| Phase | Candidate-local evidence | Gate behavior |
| --- | --- | --- |
| `intake` | `docs/01_intake_manifest.md` records catalog identity, astrometry, stellar context, collision checks, catalog review, and literature screening. | Every mandatory checklist item must be checked. |
| `feasibility` | `docs/02_feasibility_report.md` records contamination, expected signal-to-noise ratio, observing coverage, stellar parameters, and novelty assessment. | The checklist must pass and `decisions/novelty_audit.json` must be current, schema-valid, candidate-matched, and marked eligible. |
| `acquisition` | Raw FITS products and provenance sidecars under `data/raw/`. | At least one `.fits` or `.fz` file must exist, and every such file needs a matching `<stem>.provenance.json` sidecar. |
| `vetting` | `docs/03_spoc_dv_vetting.md` records odd-even, difference-image centroid, ephemeris-match, and secondary-eclipse assessments. | Every mandatory checklist item must be checked. The gate does not recompute these diagnostics. |
| `followup` | `docs/04_tfop_sg_followup.md` records photometry, reconnaissance spectroscopy, high-resolution imaging, and precision-RV status. | Every mandatory checklist item must be checked. |
| `analysis` | Structured scientific claims in `claims/`. | Any parseable claim with `parameter: "fpp"` and numeric `value < 0.01` passes this gate. The claim need not have a particular filename, so method and provenance still require reviewer scrutiny. |
| `review` | `decisions/review_gate.md`, a current novelty audit, and the complete candidate record. | The checklist and novelty audit must pass. Successful review sets the lifecycle state to `published` and writes a final gate record. Leaving `published` later requires an explicit reason through `set-state`. |

The novelty-audit record must have a valid schema, match the workspace candidate, declare `status: "eligible"`, and contain a nonexpired, timezone-aware evidence trail. This prevents a stale or mismatched literature check from satisfying feasibility or review.

Use lifecycle changes rather than direct JSON edits:

```powershell
exonym set-state <candidate-id> --state paused --reason "Awaiting follow-up observations"
```

Valid lifecycle states are `active`, `paused`, `stopped`, `published`, and `archived`. Lifecycle events are appended to `lifecycle/events.jsonl` within the candidate workspace.

## Isolation and data stewardship

The central invariant is strict: no target-specific data, identifiers, aliases, or constants may exist outside `candidate/`. `exonym verify` audits the working tree through five layers:

| Audit layer | Check |
| --- | --- |
| Repository layout | Rejects forbidden top-level `data/` and `archive/` directories. |
| Research payloads | Rejects scientific payload extensions outside `candidate/`, including FITS, CSV, image, notebook, and array files. |
| Catalog identifiers | Detects TOI, TIC, and related catalog-ID strings in target-neutral text. |
| Candidate aliases | Detects aliases registered in candidate metadata when they appear in the neutral zone. |
| Shared-source constants | Parses `src/` with the Python AST and rejects numeric literals assigned to sector or ephemeris-like variable names. |

Run the audit whenever a change could affect the boundary:

```powershell
exonym verify
exonym verify --schemas-only
```

The full audit also validates supported candidate records. A policy exception, when genuinely needed, belongs in `policy/isolation-exceptions.json` and must identify the exact path, line, rule, reason, and an expiry date.

## Workspace anatomy

`exonym init` creates the core candidate-local workspace shown below. `data/processed/` is an optional input location that loaders inspect before raw products. The candidate identifier appears only in this subtree.

```text
candidate/<candidate-id>/
  candidate.json              Candidate metadata, lifecycle, and workflow state
  config/signals/             Catalog priors and declared transit-signal inputs
  data/raw/                   Downloaded FITS products and provenance sidecars
  data/processed/             Optional processed light-curve inputs
  docs/                       Intake, feasibility, vetting, and follow-up records
  decisions/                  Novelty audit and review decisions
  outputs/                    Machine-readable screening and analysis artifacts
  figures/                    Candidate-local diagnostic figures
  claims/                     Structured scientific assertions
  gates/                      Immutable phase-gate validation records
  lifecycle/                  Append-only state-transition events
  releases/                   Reproducibility-bundle directories

src/exonym/                   Target-neutral library and CLI implementation
schemas/                      JSON Schema 2020-12 definitions
templates/                    Files cloned into candidate workspaces created by init
policy/                       Isolation policy and approved exceptions
tests/                        Target-neutral automated test suite
```

### Structured records

| Record | Purpose | Important constraint |
| --- | --- | --- |
| `candidate.json` | Candidate identity, mission, lifecycle, workflow, disposition, publication state, and creation time. | Schema version is `2`; top-level objects reject undeclared properties. Use the CLI to change lifecycle state. |
| `<stem>.provenance.json` | URI, acquisition time, fetcher, and SHA-256 digest for a downloaded product. | The acquisition gate checks sidecar presence. Schema validation checks the record format, but `verify` does not recompute the digest against FITS bytes. |
| `claims/*.json` | A parameter, value, uncertainties, unit, and method for a scientific assertion. | Supported claim parameters include period, radius, mass, and false-positive probability. A claim is an assertion with provenance, not a publication-grade result by itself. |
| `decisions/novelty_audit.json` | Evidence that the signal is eligible for the workflow's novelty criterion. | Requires timestamped evidence, a decision basis, and a valid expiry time. |

The sidecar naming rule matters: a raw product named `s0001_lc.fits` must use `s0001_lc.provenance.json`, not `s0001_lc.fits.provenance.json`.

## Command reference

All commands accept the global form `exonym [--root <repository-root>] <command>`. Run `exonym <command> --help` for argparse help and the full option list.

### Workspace, lifecycle, and audit commands

| Command | Key options | Result |
| --- | --- | --- |
| `init <candidate-id>` | `--toi`, `--tic`, `--mission {tess,kepler,k2,plato,cheops}`, repeatable `--tag` | Provisions a workspace, clones templates, validates metadata, and prints candidate JSON. The identifier alone is accepted, although catalog identity is needed by some downstream operations. |
| `list` | `--phase`, `--tag`, `--mission` | Prints metadata records matching the filters. |
| `status <candidate-id>` | None | Prints candidate metadata and candidate-relative workspace paths. |
| `tag <candidate-id> <tag> [<tag> ...]` | Positional tags | Adds tags and prints the resulting tag list. |
| `track <candidate-id>` | None | Renders an ANSI/ASCII progress dashboard from candidate-local checklists. |
| `advance <candidate-id>` | None | Validates the current gate, writes a gate record and lifecycle event, then promotes the workflow when allowed. |
| `set-state <candidate-id>` | Required `--state`; optional `--reason` | Performs an audit-logged lifecycle transition. A reason is required when leaving `stopped`, `published`, or `archived`. |
| `freeze <candidate-id>` | `--version` | Creates a candidate-local reproducibility bundle and prints its path. |
| `survey init <survey-id>` | `--mission tess --sectors <int> [<int> ...] --review-snr <float>` | Creates a bounded survey and freezes its internal BLS triage threshold below `candidate/_surveys/`. |
| `survey add-target <survey-id> <candidate-id>` | None | Adds one TOI-free TESS workspace to the cohort denominator. |
| `survey search <survey-id> <candidate-id>` | None | Runs BLS only on the survey sectors using its frozen threshold after a current eligible novelty audit; it records a triage outcome, not a planet claim. |
| `survey exclude <survey-id> <candidate-id>` | `--reason <text>` | Retains a documented pre-search exclusion without changing the candidate lifecycle. |
| `survey report <survey-id>` | None | Prints every registered target and its recorded outcome. |
| `verify` | `--schemas-only` | Runs isolation and schema checks. It exits nonzero when violations are found. |

### Acquisition, search, and screening commands

| Command | Key options | Primary artifact or result |
| --- | --- | --- |
| `ingest <candidate-id>` | `--sectors <int> [<int> ...]`, `--exptime <int>`, `--products {lc,tp,both}` | Downloads SPOC light curves and/or target pixel files into `data/raw/` and writes provenance sidecars. `lc` is the default product choice. |
| `fetch-priors <candidate-id>` | None | Retrieves available catalog transit priors into `config/signals/transit_config.NN.json`. It can legitimately return an empty list. |
| `search <candidate-id>` | `--engine {bls,tls}`, `--period-min`, `--period-max`, `--signal` | Writes engine-specific search results and a content-addressed input manifest. The default blind period interval is 0.5 to 15.0 days. TLS requires the `discovery` extra. |
| `screen <candidate-id>` | `--signal` | Writes `outputs/fixed_ephemeris_screen.json` or a signal-scoped equivalent after fixed-ephemeris primary, odd-even, half-phase, and alternating-event checks. |
| `vet <candidate-id>` | `--n-draws`, `--signal` | Runs the optional TRICERATOPS wrapper, writes `outputs/triceratops_report.json`, and on success writes an FPP claim. The default draw count is 2000. |
| `archive <candidate-id>` | `--radius-arcsec` | Writes `outputs/archival_vetting_report.json` from Gaia DR3 and available ExoFOP context. The default search radius is 10 arcsec. |
| `plot <candidate-id>` | `--signal` | Writes phase-folded light-curve and centroid-offset figures under `figures/`. See the scientific-use warning for the centroid-panel limitation. |

### Exploratory characterization commands

| Command | Key options | Primary artifact or result |
| --- | --- | --- |
| `asteroseismology <candidate-id>` | `--numax-min`, `--numax-max` | Writes `outputs/asteroseismic_results.json`. |
| `localization <candidate-id>` | `--search-radius` | Writes `outputs/prf_localization_results.json` from pixel-depth and Gaussian-template screening. |
| `sed <candidate-id>` | None | Writes `outputs/sed_fit_results.json` and an MCMC chain array. |
| `fit <candidate-id>` | `--n-samples`, `--eccentric`, `--signal` | Writes `outputs/mcmc_transit_fit.json` and an MCMC chain array. The default chain length is 5000 samples. |
| `phasecurve <candidate-id>` | None | Writes `outputs/phase_curve_results.json`. |
| `ttv <candidate-id>` | `--signal` | Writes `outputs/ttv_analysis_results.json` and may create a timing diagram. |
| `activity <candidate-id>` | None | Writes `outputs/stellar_activity_results.json`. |
| `dilution <candidate-id>` | None | Writes `outputs/dilution_sensitivity_results.json` using supplied or previously archived neighbor information. |

## Scientific methods and interpretation

The following descriptions document the implemented procedures, not an abstract ideal pipeline. Each method should be interpreted together with its input provenance, diagnostic flags, and limitations.

### Transit search

`exonym search --engine bls` uses a custom box-shaped periodic search. For an in-transit and out-of-transit partition, it estimates depth as:

```text
d = median(f_out) - median(f_in)
```

It reports a heuristic signal-to-noise ratio using:

```text
N_eff = N_in * N_out / (N_in + N_out)
SNR = d * sqrt(N_eff) / sigma_out
```

The blind search uses a fixed three-hour duration, refines candidate periods, and checks twofold and threefold harmonic aliases. `best_duration_hours` therefore describes the supplied search duration rather than a recovered physical transit duration. The score uses median-box statistics rather than a complete transit likelihood, weighted photometric uncertainties, a detrending model, or a correlated-noise model. Inspect recovered periods against the catalog and phase-folded data before treating a peak as an astrophysical event.

`exonym search --engine tls` uses the optional Transit Least Squares engine on native-cadence photometry and per-cadence flux uncertainties. It reports TLS Signal Detection Efficiency alongside period, epoch, depth, and duration. TLS improves transit-shape matching but does not calibrate its own false-alarm rate; injection-recovery and null searches remain required before ranking alerts as a survey result.

### Fixed-ephemeris photometric screening

`exonym screen` measures a primary window, odd and even event depths, a half-phase window for secondary-eclipse evidence, and doubled-period alternating-event behavior. It reports the odd-even consistency statistic:

```text
Z_odd-even = abs(d_odd - d_even) / sqrt(sigma_odd^2 + sigma_even^2)
```

The nominal helper criterion is `Z_odd-even < 3`. Its uncertainty calculation uses median-depth approximations and does not model red noise, detrending choices, crowding, dilution uncertainty, or ephemeris uncertainty. A value below the threshold is evidence against a resolved odd-even difference under those assumptions; it is not proof that the signal is planetary.

The centroid helper computes angular displacement significance as:

```text
Z_centroid = sqrt((delta_RA * cos(dec))^2 + delta_Dec^2) / sigma
```

`Z_centroid < 3` is likewise a screening convention. It is not an automatic gate condition and cannot replace a validated difference-image centroid analysis. The shared vetting utilities also include an ellipsoidal-amplitude estimate based on mass ratio, stellar radius, orbital separation, and inclination. Treat it as a plausibility diagnostic, especially for short-period stellar companions, rather than a complete binary model.

### False-positive probability and archival context

`exonym vet` uses the optional [TRICERATOPS](https://github.com/stevengiacalone/triceratops) package for false-positive scenarios. It reads the period, depth, and duration from a signal configuration or BLS result, combines them with workspace metadata, and writes a report and FPP claim when the run succeeds.

The current wrapper gives TRICERATOPS a simplified box-shaped light curve instead of the observed candidate photometry. It also uses a fixed fractional uncertainty rather than a full posterior. Treat its output as a screening input, then check whether the stellar properties, aperture contamination, contrast limits, light-curve treatment, and population assumptions make sense for the candidate.

> [!IMPORTANT]
> TRICERATOPS can keep a machine busy for a long time and may use significant CPU or memory. Dedicated or remote execution is suitable for a long exploratory run when data policy permits. Keep a candidate-local record of the command, package version, input hashes, and output, then rerun any result that supports a claim in the project's frozen environment.

`exonym archive` queries Gaia DR3 through available TAP, VizieR, and mirror backends. It validates target association within two arcsec, including proper-motion propagation where applicable. A Renormalised Unit Weight Error value above 1.4 flags possible unresolved multiplicity, but does not establish binarity. The command's default ten-arcsec archive radius is suitable for local catalog context, not a complete crowding analysis when brighter contaminants sit farther from the target aperture.

### Pixel localization and dilution

`exonym localization` constructs a pixel depth map:

```text
D = (F_out - F_in) / F_out
```

It fits nonnegative amplitudes of isotropic Gaussian source templates using nonnegative least squares. This procedure is a depth-centroid and source-competition screen. It is not a calibrated fit to a mission PRF library, and a source should not be called target-dominated unless competing sources are modeled.

`exonym dilution` reports a contamination ratio:

```text
C_contam = sum(F_neighbor / F_target)
```

When only Gaia magnitudes are available, it uses the approximate flux ratio `10^(-0.4 * (G_neighbor - G_target))`. It compares pipeline and square-aperture assumptions. Gaia G-band contrast is not TESS-band contrast, and a catalog neighbor is not a model of its eclipse depth, so the result bounds plausible dilution rather than resolving every blend scenario. The dilution command consumes supplied neighbor data or a validated archival report; it does not query Gaia itself.

### Transit fitting and stellar parameters

`exonym fit` uses Batman transit models with MCMC sampling. Quadratic limb-darkening parameters are sampled through Kipping-style transformed variables:

```text
u1 = 2 * sqrt(q1) * q2
u2 = sqrt(q1) * (1 - 2 * q2)
```

When a stellar-density constraint is available, the model uses:

```text
a_over_Rstar = (G * P^2 * rho_star / (3 * pi))^(1/3)
```

The current fitter uses phase-folded, median-binned data and independent white errors with a jitter term. It does not fit a Gaussian-process noise model or provide a native-cadence adopted posterior. Inspect posterior correlations, prior sensitivity, dilution treatment, and cadence integration before converting a fitted radius ratio into a physical companion radius.

`exonym sed` fits a reddened blackbody representation at catalog pivot wavelengths with `emcee`. It uses parallax information to infer radius and luminosity. It is useful for a transparent first-pass stellar context, but it is not a passband-integrated atmosphere analysis and should not replace a dedicated stellar-characterization study.

`exonym asteroseismology` estimates a power spectral density, whitens a median background, searches for a smoothed oscillation envelope, and estimates the large frequency separation from lag correlation. It applies standard solar-like scaling relations, including:

```text
R_star / R_sun = (numax / numax_sun) * (Dnu / Dnu_sun)^(-2) * (Teff / Teff_sun)^(1/2)
M_star / M_sun = (numax / numax_sun)^3 * (Dnu / Dnu_sun)^(-4) * (Teff / Teff_sun)^(3/2)
```

The output includes physical sanity checks. Scaling relations are empirical approximations with regime-dependent systematics; a plausible result is not a validated seismic solution.

### Variability, phase curves, and timing

`exonym phasecurve` regresses sector offsets and slopes with reflection, Doppler-beaming, ellipsoidal, harmonic-control, and phase-0.5 eclipse terms. It uses block-clustered covariance over half-day blocks. A component at or above three standard deviations remains a follow-up prompt, not a physical detection without systematics tests and an independent model comparison.

`exonym ttv` fits fixed transit templates to expected events and reports observed-minus-calculated timing in minutes:

```text
O_minus_C_minutes = 1440 * (t_observed - t_calculated)
```

For a proposed first-order resonance, it also reports the conventional super-period relation:

```text
1 / P_TTV = abs(j / P_outer - (j - 1) / P_inner)
```

Low signal-to-noise timing estimates can be dominated by shape and baseline noise. The module has no standalone timing-variation detection threshold.

`exonym activity` applies a per-sector generalized Lomb-Scargle search over one to twenty days, combines periods using power-weighted estimates, and fits a sinusoidal amplitude. Rotation or activity interpretations require window-function, harmonic, and persistence checks beyond the reported periodogram peak.

## Methodological limits

Use the framework to organize and test evidence. Do not treat it as an automatic planet-validation certificate.

- Transit-like signals can arise from eclipsing binaries, background blends, instrumental artifacts, stellar variability, or detrending behavior.
- BLS period aliases, sparse sampling, transit-duration assumptions, and time-correlated noise can bias detection statistics and ephemerides.
- Odd-even and centroid screening require appropriate photometry, aperture context, and uncertainty models. They do not replace image-domain vetting.
- Catalog queries can be incomplete, stale, or unavailable. A clean query result is not proof of isolation from contaminants.
- FPP estimates depend on the input light curve, stellar properties, contrast constraints, catalog completeness, and scenario priors. The current wrapper has the additional simplified-light-curve limitation described above.
- SED, seismic, phase-curve, activity, TTV, localization, and dilution outputs are exploratory unless their candidate-local inputs, uncertainty model, and validation checks support a stronger claim.
- A workflow phase and a publication lifecycle state record process completion. They do not elevate a screening product into a confirmed exoplanet.

## Reproducibility, testing, and release records

`exonym freeze <candidate-id> --version <version>` creates a release directory below `candidate/<candidate-id>/releases/`. The bundle captures dependency and container definitions, a manifest, source-control metadata, and a metadata hash. It requires `requirements-lock.txt` at the repository root.

Freezing does not copy the full candidate workspace, raw FITS data, derived outputs, or claims into the release directory. Before calling a release reproducible, check that the source inputs, candidate records, and external-data retrieval conditions are still available and documented.

Run the following before a code, schema, template, or scientific-record milestone:

```powershell
python -m compileall -q src tests
python -m pytest -q
exonym verify
exonym verify --schemas-only
```

For an editorial-only README change, the repository policy does not require the full Python test suite. The isolation audit remains appropriate because target-neutral documentation can accidentally contain target-specific identifiers or aliases.

Continuous integration runs the test suite, full isolation audit, and schema-only audit on pushes and pull requests.

## License

EXONYM is licensed under the [GNU General Public License v3.0](LICENSE).
