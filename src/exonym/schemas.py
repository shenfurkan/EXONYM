"""Machine-readable JSON schema validation for candidate workspaces.

Validates every ``candidate/<id>/candidate.json`` against
``schemas/candidate.schema.json``, every ``*.provenance.json`` sidecar against
``schemas/provenance.schema.json``, and every ``claims/*.json`` assertion
against ``schemas/claim.schema.json`` (JSON Schema draft 2020-12).

Frozen legacy evidence under ``candidate/<id>/legacy-project/`` is excluded:
it predates the schema system and is preserved as-is.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional

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
ENGINE_RUN_SCHEMA = "engine-run.schema.json"
AUTOMATED_TRIAGE_SCHEMA = "automated-triage.schema.json"
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


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for one regular candidate-local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_artifact_path(workspace_dir: Path, relative_path: object) -> Optional[Path]:
    """Resolve a manifest path only when it remains inside its owning workspace."""
    if not isinstance(relative_path, str):
        return None
    workspace_root = workspace_dir.resolve()
    artifact_path = (workspace_root / relative_path).resolve()
    try:
        artifact_path.relative_to(workspace_root)
    except ValueError:
        return None
    return artifact_path


def _validate_artifacts(
    report: IsolationReport,
    record_path: Path,
    workspace_dir: Path,
    artifacts: object,
    label: str,
) -> None:
    """Ensure every manifest artifact exists in the workspace with its recorded hash."""
    if not isinstance(artifacts, list):
        return
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_path = _candidate_artifact_path(workspace_dir, artifact.get("path"))
        if artifact_path is None or not artifact_path.is_file():
            report.add(record_path, "artifact-reference-invalid", "{0} artifact does not exist in the candidate workspace".format(label))
            continue
        if artifact.get("sha256") != _file_sha256(artifact_path):
            report.add(record_path, "artifact-hash-mismatch", "{0} artifact SHA-256 does not match its manifest".format(label))


def _validate_triage_records(
    report: IsolationReport,
    triage_path: Path,
    workspace_dir: Path,
    instance: object,
    engine_run_schema: object,
    validate_func: Callable[[object, object], None],
) -> None:
    """Verify that non-blocked triage records reference exact successful run outputs."""
    if not isinstance(instance, dict) or not isinstance(instance.get("records"), list):
        return
    for record in instance["records"]:
        if not isinstance(record, dict) or record.get("status") == "blocked":
            continue
        manifest_path = _candidate_artifact_path(workspace_dir, record.get("run_manifest_path"))
        artifact_path = _candidate_artifact_path(workspace_dir, record.get("artifact_path"))
        if manifest_path is None or not manifest_path.is_file() or artifact_path is None or not artifact_path.is_file():
            report.add(triage_path, "triage-provenance-invalid", "triage record references a missing candidate-local artifact")
            continue
        if record.get("run_manifest_sha256") != _file_sha256(manifest_path):
            report.add(triage_path, "triage-provenance-invalid", "triage run manifest SHA-256 does not match")
            continue
        if record.get("artifact_sha256") != _file_sha256(artifact_path):
            report.add(triage_path, "triage-provenance-invalid", "triage evidence artifact SHA-256 does not match")
            continue
        try:
            manifest = _read_json(manifest_path)
        except (OSError, UnicodeError, ValueError) as exc:
            report.add(triage_path, "triage-provenance-invalid", "triage run manifest is unreadable: {0}".format(exc))
            continue
        _validate(report, manifest_path, manifest, engine_run_schema, validate_func)
        if not isinstance(manifest, dict):
            continue
        artifact_relative = artifact_path.relative_to(workspace_dir.resolve()).as_posix()
        outputs = manifest.get("outputs")
        if (
            manifest.get("candidate_id") != workspace_dir.name
            or manifest.get("engine") != record.get("engine")
            or manifest.get("status") != "succeeded"
            or not isinstance(outputs, list)
            or not any(
                isinstance(output, dict)
                and output.get("path") == artifact_relative
                and output.get("sha256") == record.get("artifact_sha256")
                for output in outputs
            )
        ):
            report.add(triage_path, "triage-provenance-invalid", "triage evidence is not a successful output of the referenced engine run")


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
        ENGINE_RUN_SCHEMA,
        AUTOMATED_TRIAGE_SCHEMA,
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
    engine_run_schema = schemas.get(ENGINE_RUN_SCHEMA)
    automated_triage_schema = schemas.get(AUTOMATED_TRIAGE_SCHEMA)
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

        if automated_triage_schema is not None:
            triage_path = workspace_dir / "decisions" / "automated_triage.json"
            if triage_path.is_file():
                try:
                    instance = _read_json(triage_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(triage_path, "schema-violation", "invalid JSON: {0}".format(exc))
                else:
                    _validate(report, triage_path, instance, automated_triage_schema, validate_func)
                    if isinstance(instance, dict) and instance.get("candidate_id") != workspace_dir.name:
                        report.add(
                            triage_path,
                            "schema-violation",
                            "automated triage candidate_id does not match its workspace",
                        )
                    if engine_run_schema is not None:
                        _validate_triage_records(
                            report,
                            triage_path,
                            workspace_dir,
                            instance,
                            engine_run_schema,
                            validate_func,
                        )

        if engine_run_schema is not None:
            for run_path in sorted(workspace_dir.glob("runs/*/*/engine-run.json")):
                try:
                    instance = _read_json(run_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(run_path, "schema-violation", "invalid JSON: {0}".format(exc))
                    continue
                _validate(report, run_path, instance, engine_run_schema, validate_func)
                if not isinstance(instance, dict):
                    continue
                relative = run_path.relative_to(workspace_dir)
                if instance.get("candidate_id") != workspace_dir.name:
                    report.add(
                        run_path,
                        "schema-violation",
                        "engine run candidate_id does not match its workspace",
                    )
                if instance.get("engine") != relative.parts[1]:
                    report.add(
                        run_path,
                        "schema-violation",
                        "engine run engine does not match its directory",
                    )
                if instance.get("run_id") != relative.parts[2]:
                    report.add(
                        run_path,
                        "schema-violation",
                        "engine run run_id does not match its directory",
                    )
                _validate_artifacts(report, run_path, workspace_dir, instance.get("inputs"), "input")
                _validate_artifacts(report, run_path, workspace_dir, instance.get("outputs"), "output")

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
            len(relative.parts) >= 3
            and relative.parts[-2] == "outputs"
            and not any(p.startswith("_") for p in relative.parts[:-2])
        )
        if not is_candidate_local:
            report.add(
                path,
                "survey-robustness-outside-candidate",
                "robustness artifacts must be direct files in candidate/<id>/outputs/",
            )

    for path in sorted(root.rglob("engine-run.json")):
        if not path.is_file() or LEGACY_SUBTREE in path.parts:
            continue
        try:
            relative = path.relative_to(candidate_root)
        except ValueError:
            relative = Path()
        is_candidate_local = (
            len(relative.parts) >= 5
            and relative.parts[-4] == "runs"
            and relative.parts[-1] == "engine-run.json"
            and not any(p.startswith("_") for p in relative.parts[:-4])
        )
        if not is_candidate_local:
            report.add(
                path,
                "engine-run-outside-candidate",
                "engine manifests must be direct files in candidate/<id>/runs/<engine>/<run-id>/",
            )

    for path in sorted(root.rglob("automated_triage.json")):
        if not path.is_file() or LEGACY_SUBTREE in path.parts:
            continue
        try:
            relative = path.relative_to(candidate_root)
        except ValueError:
            relative = Path()
        is_candidate_local = (
            len(relative.parts) >= 3
            and relative.parts[-2] == "decisions"
            and relative.parts[-1] == "automated_triage.json"
            and not any(p.startswith("_") for p in relative.parts[:-2])
        )
        if not is_candidate_local:
            report.add(
                path,
                "automated-triage-outside-candidate",
                "automated triage records must be direct files in candidate/<id>/decisions/",
            )
