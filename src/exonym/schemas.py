"""Machine-readable JSON schema validation for candidate workspaces.

Validates every ``candidate/<id>/candidate.json`` against
``schemas/candidate.schema.json``, every ``*.provenance.json`` sidecar against
``schemas/provenance.schema.json``, and every ``claims/*.json`` assertion
against ``schemas/claim.schema.json`` (JSON Schema draft 2020-12).

Frozen legacy evidence under ``candidate/<id>/legacy-project/`` is excluded:
it predates the schema system and is preserved as-is.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Callable, Dict

from .isolation import IsolationReport
from .resources import ResourceUnavailableError, read_schema_text

SCHEMA_DIRECTORY = "schemas"
CANDIDATE_SCHEMA = "candidate.schema.json"
PROVENANCE_SCHEMA = "provenance.schema.json"
CLAIM_SCHEMA = "claim.schema.json"
NOVELTY_AUDIT_SCHEMA = "novelty-audit.schema.json"
LEGACY_SUBTREE = "legacy-project"


def _load_schemas(root: Path, report: IsolationReport) -> Dict[str, object]:
    loaded: Dict[str, object] = {}
    for name in (CANDIDATE_SCHEMA, PROVENANCE_SCHEMA, CLAIM_SCHEMA, NOVELTY_AUDIT_SCHEMA):
        path = root / SCHEMA_DIRECTORY / name
        try:
            content = read_schema_text(root, name)
        except FileNotFoundError:
            report.add(path, "schema-file-missing", "schema file not found")
            continue
        except ResourceUnavailableError as exc:
            report.add(path, "schema-resource-unavailable", str(exc))
            continue
        except (OSError, UnicodeError) as exc:
            report.add(path, "schema-file-unreadable", str(exc))
            continue
        try:
            loaded[name] = json.loads(content)
        except json.JSONDecodeError as exc:
            report.add(path, "schema-file-invalid", "invalid JSON: {0}".format(exc))
    return loaded


def _validate(
    report: IsolationReport,
    path: Path,
    instance: object,
    schema: object,
    validate_func: Callable[[object, object], None],
) -> None:
    try:
        validate_func(instance, schema)
    except Exception as exc:  # ValidationError or SchemaError
        detail = str(exc).splitlines()
        report.add(path, "schema-violation", detail[0][:300] if detail else str(exc))


def validate_schemas(root: Path, report: IsolationReport) -> None:
    """Append schema violations for every candidate record in the tree."""
    root = Path(root).resolve()
    try:
        import jsonschema
    except ImportError as exc:
        report.add(root, "schema-validation-unavailable", "jsonschema not installed: {0}".format(exc))
        return

    schemas = _load_schemas(root, report)
    if CANDIDATE_SCHEMA not in schemas or PROVENANCE_SCHEMA not in schemas:
        return
    format_checker = jsonschema.FormatChecker()
    validate_func = partial(jsonschema.validate, format_checker=format_checker)

    candidate_root = root / "candidate"
    if not candidate_root.is_dir():
        return

    for workspace_dir in sorted(candidate_root.iterdir()):
        if not workspace_dir.is_dir() or workspace_dir.name.startswith("_"):
            continue

        metadata_path = workspace_dir / "candidate.json"
        if metadata_path.is_file():
            try:
                instance = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                report.add(metadata_path, "schema-violation", "invalid JSON: {0}".format(exc))
            else:
                _validate(report, metadata_path, instance, schemas[CANDIDATE_SCHEMA], validate_func)

        for path in workspace_dir.rglob("*.provenance.json"):
            if LEGACY_SUBTREE in path.parts:
                continue
            try:
                instance = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                report.add(path, "schema-violation", "invalid JSON: {0}".format(exc))
                continue
            _validate(report, path, instance, schemas[PROVENANCE_SCHEMA], validate_func)

        claim_schema = schemas.get(CLAIM_SCHEMA)
        if claim_schema is not None:
            claims_dir = workspace_dir / "claims"
            if claims_dir.is_dir():
                for path in sorted(claims_dir.glob("*.json")):
                    try:
                        instance = json.loads(path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        report.add(path, "schema-violation", "invalid JSON: {0}".format(exc))
                        continue
                    _validate(report, path, instance, claim_schema, validate_func)

        novelty_audit_schema = schemas.get(NOVELTY_AUDIT_SCHEMA)
        novelty_audit_path = workspace_dir / "decisions" / "novelty_audit.json"
        if novelty_audit_schema is not None and novelty_audit_path.is_file():
            try:
                instance = json.loads(novelty_audit_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                report.add(novelty_audit_path, "schema-violation", "invalid JSON: {0}".format(exc))
            else:
                _validate(report, novelty_audit_path, instance, novelty_audit_schema, validate_func)
