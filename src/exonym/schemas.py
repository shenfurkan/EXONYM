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
import math
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
SURVEY_SCHEMA = "survey.schema.json"
SURVEY_TARGET_SCHEMA = "survey-target.schema.json"
SURVEY_ROBUSTNESS_SCHEMA = "survey-robustness.schema.json"
LEGACY_SUBTREE = "legacy-project"


def _parse_finite_float(value: str) -> float:
    """Parse one JSON float while rejecting values outside the real number line."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_nonfinite_constant(value: str) -> object:
    """Reject non-standard JSON constants such as NaN and Infinity."""
    raise ValueError("non-finite JSON number: {0}".format(value))


def _parse_json(content: str) -> object:
    """Load strict JSON without accepting non-finite numeric constants."""
    return json.loads(
        content,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_parse_finite_float,
    )


def _read_json(path: Path) -> object:
    """Read one UTF-8 JSON file with strict finite-number parsing."""
    return _parse_json(path.read_text(encoding="utf-8"))


def _load_schemas(root: Path, report: IsolationReport) -> Dict[str, object]:
    loaded: Dict[str, object] = {}
    for name in (
        CANDIDATE_SCHEMA,
        PROVENANCE_SCHEMA,
        CLAIM_SCHEMA,
        NOVELTY_AUDIT_SCHEMA,
        SURVEY_SCHEMA,
        SURVEY_TARGET_SCHEMA,
        SURVEY_ROBUSTNESS_SCHEMA,
    ):
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
            loaded[name] = _parse_json(content)
        except ValueError as exc:
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

    survey_robustness_schema = schemas.get(SURVEY_ROBUSTNESS_SCHEMA)
    surveys_root = candidate_root / "_surveys"
    for workspace_dir in sorted(candidate_root.iterdir()):
        if not workspace_dir.is_dir() or workspace_dir.name == "_surveys":
            continue

        metadata_path = workspace_dir / "candidate.json"
        if metadata_path.is_file():
            try:
                instance = _read_json(metadata_path)
            except (OSError, UnicodeError, ValueError) as exc:
                report.add(metadata_path, "schema-violation", "invalid JSON: {0}".format(exc))
            else:
                _validate(report, metadata_path, instance, schemas[CANDIDATE_SCHEMA], validate_func)

        for path in workspace_dir.rglob("*.provenance.json"):
            if LEGACY_SUBTREE in path.parts:
                continue
            try:
                instance = _read_json(path)
            except (OSError, UnicodeError, ValueError) as exc:
                report.add(path, "schema-violation", "invalid JSON: {0}".format(exc))
                continue
            _validate(report, path, instance, schemas[PROVENANCE_SCHEMA], validate_func)

        claim_schema = schemas.get(CLAIM_SCHEMA)
        if claim_schema is not None:
            claims_dir = workspace_dir / "claims"
            if claims_dir.is_dir():
                for path in sorted(claims_dir.glob("*.json")):
                    try:
                        instance = _read_json(path)
                    except (OSError, UnicodeError, ValueError) as exc:
                        report.add(path, "schema-violation", "invalid JSON: {0}".format(exc))
                        continue
                    _validate(report, path, instance, claim_schema, validate_func)

        novelty_audit_schema = schemas.get(NOVELTY_AUDIT_SCHEMA)
        novelty_audit_path = workspace_dir / "decisions" / "novelty_audit.json"
        if novelty_audit_schema is not None and novelty_audit_path.is_file():
            try:
                instance = _read_json(novelty_audit_path)
            except (OSError, UnicodeError, ValueError) as exc:
                report.add(novelty_audit_path, "schema-violation", "invalid JSON: {0}".format(exc))
            else:
                _validate(report, novelty_audit_path, instance, novelty_audit_schema, validate_func)

        if survey_robustness_schema is not None:
            outputs_dir = workspace_dir / "outputs"
            if outputs_dir.is_dir():
                for path in sorted(outputs_dir.glob("survey_robustness*.json")):
                    if not metadata_path.is_file():
                        report.add(
                            path,
                            "survey-robustness-outside-candidate",
                            "robustness artifacts require an owning candidate.json workspace",
                        )
                        continue
                    try:
                        instance = _read_json(path)
                    except (OSError, UnicodeError, ValueError) as exc:
                        report.add(path, "schema-violation", "unreadable JSON: {0}".format(exc))
                        continue
                    _validate(report, path, instance, survey_robustness_schema, validate_func)
                    if not isinstance(instance, dict):
                        continue
                    if instance.get("candidate_id") != workspace_dir.name:
                        report.add(
                            path,
                            "schema-violation",
                            "robustness artifact candidate_id does not match its workspace",
                        )
                    survey_id = instance.get("survey_id")
                    if isinstance(survey_id, str):
                        expected_name = "survey_robustness.survey-{0}.json".format(survey_id)
                        if path.name != expected_name:
                            report.add(
                                path,
                                "schema-violation",
                                "robustness artifact filename does not match its survey_id",
                            )
                        if Path(survey_id).name != survey_id or survey_id in (".", ".."):
                            continue
                        survey_path = surveys_root / survey_id / "survey.json"
                        if not survey_path.is_file():
                            report.add(
                                path,
                                "schema-violation",
                                "robustness artifact has no matching survey record",
                            )
                            continue
                        try:
                            survey_record = _read_json(survey_path)
                        except (OSError, UnicodeError, ValueError) as exc:
                            report.add(
                                path,
                                "schema-violation",
                                "matching survey record is unreadable: {0}".format(exc),
                            )
                            continue
                        if not isinstance(survey_record, dict) or survey_record.get("survey_id") != survey_id:
                            report.add(
                                path,
                                "schema-violation",
                                "robustness artifact survey_id does not match its survey record",
                            )
                            continue
                        target_path = (
                            surveys_root
                            / survey_id
                            / "targets"
                            / workspace_dir.name
                            / "target.json"
                        )
                        if not target_path.is_file():
                            report.add(
                                path,
                                "schema-violation",
                                "robustness artifact has no matching survey target record",
                            )
                            continue
                        try:
                            target_record = _read_json(target_path)
                        except (OSError, UnicodeError, ValueError) as exc:
                            report.add(
                                path,
                                "schema-violation",
                                "matching survey target record is unreadable: {0}".format(exc),
                            )
                            continue
                        if (
                            not isinstance(target_record, dict)
                            or target_record.get("survey_id") != survey_id
                            or target_record.get("candidate_id") != workspace_dir.name
                        ):
                            report.add(
                                path,
                                "schema-violation",
                                "robustness artifact does not match its survey target record",
                            )

    survey_schema = schemas.get(SURVEY_SCHEMA)
    survey_target_schema = schemas.get(SURVEY_TARGET_SCHEMA)
    if survey_schema is not None and survey_target_schema is not None and surveys_root.is_dir():
        for survey_dir in sorted(path for path in surveys_root.iterdir() if path.is_dir()):
            survey_path = survey_dir / "survey.json"
            if survey_path.is_file():
                try:
                    instance = _read_json(survey_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(survey_path, "schema-violation", "invalid JSON: {0}".format(exc))
                else:
                    _validate(report, survey_path, instance, survey_schema, validate_func)
            targets_dir = survey_dir / "targets"
            if not targets_dir.is_dir():
                continue
            for target_path in sorted(targets_dir.glob("*/target.json")):
                try:
                    instance = _read_json(target_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(target_path, "schema-violation", "invalid JSON: {0}".format(exc))
                else:
                    _validate(report, target_path, instance, survey_target_schema, validate_func)

    for path in sorted(root.rglob("survey_robustness*.json")):
        if not path.is_file():
            continue
        if LEGACY_SUBTREE in path.parts:
            continue
        try:
            relative = path.relative_to(candidate_root)
        except ValueError:
            relative = Path()
        is_candidate_local = (
            len(relative.parts) == 3
            and relative.parts[0] != "_surveys"
            and relative.parts[1] == "outputs"
        )
        if not is_candidate_local:
            report.add(
                path,
                "survey-robustness-outside-candidate",
                "robustness artifacts must be direct files in candidate/<id>/outputs/",
            )
