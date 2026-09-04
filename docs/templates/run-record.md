# Run Record

Run ID: RUN-YYYYMMDD-NNN

Protocol ID/version:

Status: `PLANNED | RUNNING | COMPLETE | FAILED | INVALID | QUARANTINED`

## Execution Identity

| Field | Value |
|---|---|
| Start/end UTC | |
| Operator | |
| Working directory | |
| Commit/code hash | |
| Input manifest hash | |
| Environment/container digest | |
| Host/OS/CPU/GPU | |
| Command and arguments | |
| Exit code | |
| Engine/automation manifest | |
| Workspace checkpoint relation | None / checkpoint ID |

## Randomness

Record master seed, per-task seed derivation, generator implementation, worker
count, and task ordering.

## Resources

| Resource | Planned | Observed |
|---|---:|---:|
| Wall time | | |
| Peak memory | | |
| Storage written | | |

## Outputs

| Artifact | Size bytes | SHA-256 | Schema/status |
|---|---:|---|---|
| | | | |

## Diagnostics

Record convergence, movement from starts, objective values, warnings, retries,
boundary behavior, and residual checks.

## Deviations

List every protocol deviation. If none, state `None`. Undocumented deviations
invalidate result-bearing execution.

## Disposition

State whether the run may proceed to gate review and which artifacts are
authoritative, diagnostic, invalid, or quarantined. Engine availability,
telemetry, and auto-vet completion do not by themselves authorize a gate or
claim.
