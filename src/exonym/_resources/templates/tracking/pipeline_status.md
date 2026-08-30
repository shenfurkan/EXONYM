# Pipeline Status :: {{CANDIDATE_ID}}

Target: {{TOI}} / {{TIC}} | Updated: {{TIMESTAMP}}

## Current Checkpoint

- Lifecycle: {{STATUS}}
- Workflow phase: {{PHASE}}
- Next gate: see the phase document in `docs/`

## Local Telemetry

This file is updated at each checkpoint. Run `exonym track {{CANDIDATE_ID}}`
for the machine-parsed gate progress dashboard.

Record workspace checkpoint IDs separately from fit sampler checkpoints. Live
telemetry reports progress and resources only; it is not convergence evidence.

## Gate status

- Gate behavior: fail closed
- Feasibility and review: require a current, eligible, evidence-backed `decisions/novelty_audit.json`
- Stopped lifecycle: workflow advancement is disabled until a reasoned state change is recorded
