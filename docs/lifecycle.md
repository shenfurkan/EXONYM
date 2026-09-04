# Candidate lifecycle

A candidate workspace is a permanent directory. Lifecycle changes are recorded
events, never directory moves. A lifecycle state describes the workspace's
status, while a workflow phase describes its position in the analysis sequence.
`exonym advance` evaluates both.

Scientific disposition, publication, human-review status, and retention class
are separate metadata axes. Retention is an operational label only: it neve
authorizes deleting or moving candidate data.

## States

| State | Meaning | Workflow behavior |
| --- | --- | --- |
| `active` | Scientific or publication work is active. | Phase gates determine whether advancement is permitted. |
| `paused` | Work is temporarily on hold. Preserve handover and next-action information. | This administrative state does not itself block `exonym advance`. Use `stopped` when advancement must be disabled. |
| `stopped` | Work was intentionally discontinued. Record the decision and supported negative results. | `exonym advance` rejects the workspace before evaluating phase gates. Leaving this state requires a non-empty reason. |
| `published` | At least one immutable, citable release exists. Publication does not imply validation. | At review, an already published workspace cannot advance again. Changing this state requires a non-empty reason. |
| `archived` | The scientific payload is frozen for long-term retention. | At review, an archived workspace cannot advance. Changing this state requires a non-empty reason. |

Every transition appends an event with the previous state, new state, timestamp,
candidate identifier, and reason.

## Stopping and resuming work

Use the lifecycle command instead of editing `candidate.json` directly:

```powershell
exonym set-state <candidate-id> --state stopped --reason "<decision and evidence>"
```

A stopped workspace remains available for inspection, preservation, and new
documentation, but it cannot move through the workflow. To resume work, move it
to an appropriate lifecycle state with a non-empty reason, then satisfy the
current phase gate again:

```powershell
exonym set-state <candidate-id> --state active --reason "<why work is resuming>"
```

Changing the lifecycle state never bypasses an incomplete, missing, or failed
workflow gate.

Classification changes use an evidence-backed review record rather than direct
metadata edits:

```powershell
exonym review <candidate-id> --reviewer <name> --reason "<decision>" --evidence <candidate-relative-path> --review-status reviewed
```

The command writes a versioned record under `decisions/reviews/`, captures the
SHA-256 of every evidence file, and updates only the current classification
summary in `candidate.json`. A `cold` retention label is not proof of an
external archive and does not permit local deletion.

### Operational recovery and automation

`exonym checkpoint save` creates a compressed snapshot of mutable candidate
state for operational recovery. It excludes raw FITS, append-only provenance,
and the checkpoint directory; restore verifies the archive before changes,
replaces mutable entries atomically, and appends an audit event. A checkpoint
does not alter lifecycle state, gate records, claims, or scientific eligibility.
Each checkpoint manifest follows `checkpoint-manifest.schema.json`; both save
and restore validate that manifest before accepting the checkpoint.

`survey auto-vet` and `survey run-loop` are bounded execution helpers. Thei
engine-run manifests and failed-step records are evidence of execution status,
not a lifecycle transition, disposition, validation result, or claim.

## Workflow Phases

`intake`, `feasibility`, `acquisition`, `vetting`, `followup`, `analysis`,
`review`. The phase describes where work is, not the state of the candidate.
Phases are gate-protected and fail closed. `exonym advance` promotes a phase
only after every `[MANDATORY]` checkbox in the phase document is checked and
the phase's programmatic checks pass. A missing phase document, a document with
no mandatory items, an unchecked mandatory item, or unavailable required
evidence blocks advancement. Legacy records that predate this ordering used
`writing`, `submission`, or `post-publication`; those map to `review`.

### Novelty audit requirement

Feasibility and review both require the candidate-local
`decisions/novelty_audit.json` record. The audit must be schema-valid, belong to
the workspace, include a decision basis, and contain at least one evidence item
with a source URI, retrieval timestamp, finding, and SHA-256 evidence digest.
Its status must be `eligible`, and its `freshness.expires_at` timestamp must be
later than both retrieval and the current time.

The statuses `ineligible`, `inconclusive`, and `unavailable` are valid audit
records, but each blocks advancement. A missing, malformed, stale, future-dated,
or candidate-mismatched audit also blocks advancement. Record unavailable
sources as unavailable evidence; do not treat an inaccessible source as an
absence of prior work or follow-up.

### Discovery target policy

Independent discovery starts with a TIC target that has no assigned TOI or cTOI. Intake must record current ExoFOP and literature checks before the workspace can advance. A known TOI may remain in the repository for method validation, comparison, or follow-up, but it does not count as an independent discovery without a separately documented contribution.

## Freeze and Archive Semantics

An archived candidate stays at `candidate/<candidate-id>/`. A freeze should:

1. Assign a freeze ID and set lifecycle to `archived`.
2. Append the final transition event.
3. Inventory claim-bearing sources, protocols, inputs, results, manuscripts,
   environment records, and release objects with paths, sizes, and hashes.
4. Verify the manifest from a clean location.
5. Protect the corresponding commit/tag and external release packages.

While archived: manifest-listed files cannot be edited, deleted, renamed, o
regenerated in place. Corrections use new versioned files with explicit
supersession links. Reopening preserves the old snapshot before new work.

This release process is distinct from workspace checkpoints. `freeze` captures
the release contract and dependency closure; `verify-release` validates hashes
and offline source/workspace loading, but does not rerun engines, network
queries, or remote services. A checkpoint is only a mutable-state recovery
mechanism and is never a citable scientific release.

## Verification Layers

| Layer | Purpose |
| ---: | --- |
| Q0 | Syntax and schema |
| Q1 | Unit behavior |
| Q2 | Artifact integrity (existence, size, hash, lineage) |
| Q3 | Numerical reproduction from frozen inputs |
| Q4 | Independent calculation |
| Q5 | Scientific gate (thresholds, stop rules, applicability) |
| Q6 | Claim audit |
| Q7 | Release verification (manifest, extraction, offline run) |
| Q8 | External verification |

A release claim must state the highest completed layer.

The layer does not alter a method's scientific applicability. Before a review o
release, use [`scientific-method-contract.md`](scientific-method-contract.md)
to check the exact unit/time-scale contract, primary-source provenance, and
fail-closed boundary of every formula-bearing source module. In particular,
passing Q0--Q4 cannot override the present `claim_eligible: false` invariant:
the analysis gate remains blocked until calibrated, provenance-bound scene-model
constraints are integrated and independently reviewed.

### Verification Cadence Rule

To prevent unnecessary administrative red tape, layer Q1 (`pytest`) verification is skipped for non-code tasks where no source code has changed. As a candidate manuscript or milestone takes shape as a draft, Q1–Q6 verification checks are executed more frequently prior to formal release freezing (Q7).
