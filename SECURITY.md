# Security Policy

## Reporting

Report vulnerabilities through GitHub Private Vulnerability Reporting:

https://github.com/shenfurkan/EXONYM/security/advisories/new

Do not open a public issue containing credentials, tokens, private paths,
unreleased data, or an exploitable vulnerability. Include the affected version
or commit, Python and operating-system details, reproduction steps, impact, and
any relevant logs with secrets and target payloads redacted.

## Supported Scope

Security review applies to shared analysis code, network queries,
dependencies, release packages, candidate data handling, and external binaries
or data under `candidate/`.

## Mandatory Controls

1. Do not disable TLS certificate verification globally.
2. Do not commit secrets, API keys, cookies, private keys, or credentials.
3. Use least-privilege, short-lived credentials for archive and release tasks.
4. Record external downloads used in scientific inference in the owning
   candidate workspace with source URI, retrieval date, terms, and a SHA-256
   digest. FITS ingestion writes provenance sidecars; archive responses and
   other external records need equivalent candidate-local provenance before
   they support a release claim.
5. Isolate optional or legacy dependencies from the verified shared core.
6. Review dependencies and licenses before release.
7. Before release, scan release archives for secrets, unsafe paths, and
   unexpected binaries. `exonym freeze` creates a manifest but does not perform
   this scan automatically.
8. If archive-extraction code is added, it must reject absolute member paths,
   traversal, symlinks, and reserved names.
9. Workspace checkpoint restore must verify the archive hash before modifying
   mutable state, reject traversal/link/device members, exclude raw FITS and
   append-only provenance, and use atomic replacement. A checkpoint is not a
   release or a way to bypass lifecycle gates.
10. Install `.[security]` before local security checks. CI runs Bandit, isolated
   Semgrep, and a full-history Gitleaks scan on every push and pull request.

## Research Isolation

Target-specific research is confined to `candidate/`. The isolation checker
enforces its listed rules: target identifiers and aliases in shared zones,
hardcoded sectors and ephemerides in shared code, research payload formats
outside `candidate/`, and symlink/reparse-point payloads anywhere.

## Incident Handling

1. Contain the affected code, credential, environment, or release object.
2. Preserve logs and hashes.
3. Rotate exposed credentials.
4. Determine whether scientific artifacts or manifests are affected.
5. Create a postmortem and corrective-action record.
6. Re-run affected verification and rebuild affected release objects.
