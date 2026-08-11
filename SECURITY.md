# Security Policy

## Reporting

Do not open a public issue containing credentials, tokens, private paths,
unreleased data, or an exploitable vulnerability. Report privately to the
repository owner using the contact listed in the current candidate or release
metadata.

## Supported Scope

Security review applies to shared analysis code, network queries,
dependencies, release packages, candidate data handling, and external binaries
or data under `candidate/`.

## Mandatory Controls

1. Do not disable TLS certificate verification globally.
2. Do not commit secrets, API keys, cookies, private keys, or credentials.
3. Use least-privilege, short-lived credentials for archive and release tasks.
4. Record and hash external downloads used in scientific inference; retain
   source, retrieval date, and terms in the owning candidate workspace.
5. Isolate optional or legacy dependencies from the verified shared core.
6. Review dependencies and licenses before release.
7. Scan release archives for secrets, unsafe paths, and unexpected binaries.
8. Archive-extraction helpers must reject absolute member paths, traversal,
   symlinks, and reserved names.

## Research Isolation

Target-specific research is confined to `candidate/`. The isolation checker
rejects target identifiers and aliases in shared zones, hardcoded sectors and
ephemerides in shared code, research payload formats outside `candidate/`,
and symlink/reparse-point payloads anywhere.

## Incident Handling

1. Contain the affected code, credential, environment, or release object.
2. Preserve logs and hashes.
3. Rotate exposed credentials.
4. Determine whether scientific artifacts or manifests are affected.
5. Create a postmortem and corrective-action record.
6. Re-run affected verification and rebuild affected release objects.
