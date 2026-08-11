# Feasibility Report :: {{CANDIDATE_ID}}

Target: {{TOI}} / {{TIC}} | Created: {{TIMESTAMP}} | Phase: feasibility

## Contamination and Signal

- [ ] [MANDATORY] Compute contamination ratio from Gaia DR3 G-band fluxes
- [ ] [MANDATORY] Estimate expected transit SNR for the candidate
- [ ] [MANDATORY] Inventory available TESS sectors and cadences
- [ ] [MANDATORY] Record stellar parameters (log g, T_eff, R_*)
- [ ] [MANDATORY] Record a current, eligible novelty audit in `decisions/novelty_audit.json`
- [ ] Confirm the target remained TOI-free through feasibility, or record the approved known-TOI exception
- [ ] Write the one-page go/no-go feasibility decision
- [ ] Record required follow-up resources and risks

## Originality and evidence

- Record the audit's decision basis and the evidence supporting the proposed contribution
- Record material prior work and follow-up saturation found by the audit
- Record an inaccessible source as `inconclusive` or `unavailable`, rather than as positive evidence

## Gate Notes

SNR above the preregistered threshold and contamination within acceptable
bounds for the target depth are required before this phase can advance.
This gate fails closed. Alongside the mandatory checklist, `exonym advance`
requires `decisions/novelty_audit.json` to be schema-valid, match this
workspace, contain evidence with source URI, retrieval timestamp, finding, and
SHA-256 digest, and have status `eligible` with a future
`freshness.expires_at`. Missing, stale, malformed, mismatched, `ineligible`,
`inconclusive`, or `unavailable` records block advancement.
