# Review Gate :: {{CANDIDATE_ID}}

Target: {{TOI}} / {{TIC}} | Created: {{TIMESTAMP}} | Phase: review

## Peer Review and Disposition

- [ ] [MANDATORY] Peer review of scientific claims against gate evidence
- [ ] [MANDATORY] Manuscript mathematics and claim audit
- [ ] [MANDATORY] Reproducibility bundle verified in a clean environment
- [ ] [MANDATORY] Final scientific disposition recorded
- [ ] [MANDATORY] Confirm the novelty audit remains eligible and unexpired
- [ ] Record reviewer names and dispositions

## Gate Notes

Review fails closed. `exonym advance` revalidates the candidate-local
`decisions/novelty_audit.json` record, so a missing, nonconforming, non-eligible,
or expired audit blocks publication even when every checklist item is checked.
Refresh the audit before review if its freshness window has elapsed.

After all requirements pass, advancing review sets the lifecycle to `published`.
A workspace in lifecycle state `stopped` cannot advance. Resume work with
`exonym set-state <candidate-id> --state active --reason "<why work is resuming>"`,
then satisfy the review gate again.
