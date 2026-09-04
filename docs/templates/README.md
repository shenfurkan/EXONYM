# Research Record Templates

These templates operationalize `../governance/README.md` and
`../lifecycle.md`. Copy a template into the project record area,
assign a stable ID, and preserve approved versions through explicit
supersession.

| Template | Lifecycle use |
|---|---|
| `project-charter.md` | L0 intake, scope, claims, roles, risks, and stop rules |
| `claim-charter.md` | L0/L6 permitted and prohibited scientific wording |
| `dataset-record.md` | L1 source, terms, integrity, lineage, and restoration |
| `protocol.md` | L2 pre-result model, calibration, gates, and outputs |
| `run-record.md` | L3/L4 execution identity, resources, outputs, and deviations |
| `gate-review.md` | L5 verification and claim-opening disposition |
| `decision-record.md` | Consequential scientific, operational, release, or security decision |
| `postmortem.md` | Failed, invalid, surprising, or security-relevant work |
| `self-assessment.md` | Monthly and milestone lab maturity review |
| `handover.md` | Ownership transition, pause, or milestone continuity record |

Workspace checkpoints, engine-run manifests, and bounded auto-vet manifests are
candidate-local machine records rather than generic templates. Reference their
IDs and hashes from a handover or run record, but do not treat them as gate or
claim evidence by themselves.

Required practices:

1. Do not fill records retrospectively without disclosing prior exposure.
2. Bind material records to exact inputs, code, environment, and artifact
   hashes.
3. Use `SELF_REVIEW_ONLY` when review is not independent.
4. Do not overwrite approved/frozen records.
5. Keep failed and superseded records accessible.
