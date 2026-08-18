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
RV_OBSERVATIONS_SCHEMA = "radial-velocity-observations.schema.json"
RV_FIT_SCHEMA = "rv-keplerian-fit.schema.json"
PLANETSYNTH_CHARACTERIZATION_SCHEMA = "planetsynth-characterization.schema.json"
ANOMALOUS_TRANSIT_HYPOTHESIS_SCHEMA = "anomalous-transit-hypothesis.schema.json"
PLANETSYNTH_INTERPRETATION_SCHEMA = "planetsynth-interpretation.schema.json"
PYPPLUSS_HYPOTHESIS_TEST_SCHEMA = "pyppluss-hypothesis-test.schema.json"
STATISTICAL_VETTING_EVIDENCE_SCHEMA = "statistical-vetting-evidence.schema.json"
DECISIVE_REJECTION_SCHEMA = "decisive-rejection.schema.json"
CATALOG_QUERY_MANIFEST_SCHEMA = "catalog-query-manifest.schema.json"
CATALOG_RAW_RESPONSE_METADATA_SCHEMA = "catalog-raw-response-metadata.schema.json"
CATALOG_SNAPSHOT_SCHEMA = "catalog-snapshot.schema.json"
CATALOG_STELLAR_PARAMETERS_SCHEMA = "catalog-stellar-parameters.schema.json"
CATALOG_STELLAR_PHOTOMETRY_SCHEMA = "catalog-stellar-photometry.schema.json"
CATALOG_ARCHIVE_DISCOVERY_SCHEMA = "catalog-archive-discovery.schema.json"
CATALOG_CONTRAST_CURVES_SCHEMA = "catalog-contrast-curves.schema.json"
CATALOG_CONTEXT_SCHEMA = "catalog-context.schema.json"
CATALOG_CROSS_MATCH_SCHEMA = "catalog-cross-match.schema.json"
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
        artifact_path = _candidate_artifact_path(workspace_dir, record.get("artifact_path"))
        if artifact_path is None or not artifact_path.is_file():
            report.add(triage_path, "triage-provenance-invalid", "triage record references a missing candidate-local artifact")
            continue
        if record.get("artifact_sha256") != _file_sha256(artifact_path):
            report.add(triage_path, "triage-provenance-invalid", "triage evidence artifact SHA-256 does not match")
            continue
        if "run_manifest_path" not in record and "run_manifest_sha256" not in record:
            continue
        manifest_path = _candidate_artifact_path(workspace_dir, record.get("run_manifest_path"))
        if manifest_path is None or not manifest_path.is_file():
            report.add(triage_path, "triage-provenance-invalid", "triage record references a missing engine manifest")
            continue
        if record.get("run_manifest_sha256") != _file_sha256(manifest_path):
            report.add(triage_path, "triage-provenance-invalid", "triage run manifest SHA-256 does not match")
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
        RV_OBSERVATIONS_SCHEMA,
        RV_FIT_SCHEMA,
        PLANETSYNTH_CHARACTERIZATION_SCHEMA,
        ANOMALOUS_TRANSIT_HYPOTHESIS_SCHEMA,
        PLANETSYNTH_INTERPRETATION_SCHEMA,
        PYPPLUSS_HYPOTHESIS_TEST_SCHEMA,
        STATISTICAL_VETTING_EVIDENCE_SCHEMA,
        DECISIVE_REJECTION_SCHEMA,
        CATALOG_QUERY_MANIFEST_SCHEMA,
        CATALOG_RAW_RESPONSE_METADATA_SCHEMA,
        CATALOG_SNAPSHOT_SCHEMA,
        CATALOG_STELLAR_PARAMETERS_SCHEMA,
        CATALOG_STELLAR_PHOTOMETRY_SCHEMA,
        CATALOG_ARCHIVE_DISCOVERY_SCHEMA,
        CATALOG_CONTRAST_CURVES_SCHEMA,
        CATALOG_CONTEXT_SCHEMA,
        CATALOG_CROSS_MATCH_SCHEMA,
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
    rv_observations_schema = schemas.get(RV_OBSERVATIONS_SCHEMA)
    rv_fit_schema = schemas.get(RV_FIT_SCHEMA)
    planetsynth_characterization_schema = schemas.get(PLANETSYNTH_CHARACTERIZATION_SCHEMA)
    anomalous_transit_hypothesis_schema = schemas.get(ANOMALOUS_TRANSIT_HYPOTHESIS_SCHEMA)
    planetsynth_interpretation_schema = schemas.get(PLANETSYNTH_INTERPRETATION_SCHEMA)
    pyppluss_hypothesis_test_schema = schemas.get(PYPPLUSS_HYPOTHESIS_TEST_SCHEMA)
    statistical_vetting_schema = schemas.get(STATISTICAL_VETTING_EVIDENCE_SCHEMA)
    decisive_rejection_schema = schemas.get(DECISIVE_REJECTION_SCHEMA)
    catalog_query_schema = schemas.get(CATALOG_QUERY_MANIFEST_SCHEMA)
    catalog_raw_metadata_schema = schemas.get(CATALOG_RAW_RESPONSE_METADATA_SCHEMA)
    catalog_snapshot_schema = schemas.get(CATALOG_SNAPSHOT_SCHEMA)
    catalog_record_schemas = {
        "stellar-parameters.json": schemas.get(CATALOG_STELLAR_PARAMETERS_SCHEMA),
        "stellar-photometry.json": schemas.get(CATALOG_STELLAR_PHOTOMETRY_SCHEMA),
        "archive-discovery.json": schemas.get(CATALOG_ARCHIVE_DISCOVERY_SCHEMA),
        "contrast-curves.json": schemas.get(CATALOG_CONTRAST_CURVES_SCHEMA),
    }
    catalog_context_schema = schemas.get(CATALOG_CONTEXT_SCHEMA)
    catalog_cross_match_schema = schemas.get(CATALOG_CROSS_MATCH_SCHEMA)
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

        if rv_observations_schema is not None:
            rv_observations_path = workspace_dir / "data" / "external" / "radial_velocity_observations.json"
            if rv_observations_path.is_file():
                try:
                    instance = _read_json(rv_observations_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(rv_observations_path, "schema-violation", "invalid JSON: {0}".format(exc))
                else:
                    _validate(report, rv_observations_path, instance, rv_observations_schema, validate_func)
                    if isinstance(instance, dict) and instance.get("candidate_id") != workspace_dir.name:
                        report.add(rv_observations_path, "schema-violation", "RV observation candidate_id does not match its workspace")

        if rv_fit_schema is not None:
            rv_fit_path = workspace_dir / "outputs" / "rv_keplerian_fit.json"
            if rv_fit_path.is_file():
                try:
                    instance = _read_json(rv_fit_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(rv_fit_path, "schema-violation", "invalid JSON: {0}".format(exc))
                else:
                    _validate(report, rv_fit_path, instance, rv_fit_schema, validate_func)
                    if isinstance(instance, dict):
                        if instance.get("candidate_id") != workspace_dir.name:
                            report.add(rv_fit_path, "schema-violation", "RV fit candidate_id does not match its workspace")
                        _validate_artifacts(report, rv_fit_path, workspace_dir, instance.get("input_artifacts"), "RV fit input")

        if statistical_vetting_schema is not None:
            for path in sorted((workspace_dir / "outputs").glob("statistical_vetting_evidence*.json")):
                try:
                    instance = _read_json(path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(path, "schema-violation", "invalid JSON: {0}".format(exc))
                    continue
                _validate(report, path, instance, statistical_vetting_schema, validate_func)
                if not isinstance(instance, dict):
                    continue
                if instance.get("candidate_id") != workspace_dir.name:
                    report.add(path, "schema-violation", "statistical vetting evidence candidate_id does not match its workspace")
                diagnostics = instance.get("diagnostics")
                if isinstance(diagnostics, list):
                    for diagnostic in diagnostics:
                        if isinstance(diagnostic, dict) and isinstance(diagnostic.get("artifact"), dict):
                            _validate_artifacts(
                                report,
                                path,
                                workspace_dir,
                                [diagnostic["artifact"]],
                                "statistical vetting",
                            )

        if decisive_rejection_schema is not None:
            rejection_path = workspace_dir / "decisions" / "decisive_rejection.json"
            if rejection_path.is_file():
                try:
                    instance = _read_json(rejection_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(rejection_path, "schema-violation", "invalid JSON: {0}".format(exc))
                else:
                    _validate(report, rejection_path, instance, decisive_rejection_schema, validate_func)
                    if isinstance(instance, dict):
                        if instance.get("candidate_id") != workspace_dir.name:
                            report.add(rejection_path, "schema-violation", "decisive rejection candidate_id does not match its workspace")
                        if isinstance(instance.get("evidence"), dict):
                            _validate_artifacts(
                                report,
                                rejection_path,
                                workspace_dir,
                                [instance["evidence"]],
                                "decisive rejection",
                            )

        if catalog_query_schema is not None:
            catalog_runs = workspace_dir / "runs" / "catalog"
            for manifest_path in sorted(catalog_runs.glob("*/*/query-manifest.json")):
                try:
                    instance = _read_json(manifest_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(manifest_path, "schema-violation", "invalid JSON: {0}".format(exc))
                    continue
                _validate(report, manifest_path, instance, catalog_query_schema, validate_func)
                relative = manifest_path.relative_to(workspace_dir)
                if not isinstance(instance, dict):
                    continue
                if (
                    instance.get("candidate_id") != workspace_dir.name
                    or instance.get("provider") != relative.parts[2]
                    or instance.get("retrieval_id") != relative.parts[3]
                ):
                    report.add(manifest_path, "schema-violation", "catalog manifest does not match candidate or directory ownership")
                _validate_artifacts(report, manifest_path, workspace_dir, instance.get("artifacts"), "catalog")
                provider, retrieval_id = relative.parts[2], relative.parts[3]
                expected_artifacts = {
                    "raw-response": "data/external/catalog/{0}/{1}/response.bin".format(provider, retrieval_id),
                    "raw-metadata": "data/external/catalog/{0}/{1}/response-metadata.json".format(provider, retrieval_id),
                    "normalized-snapshot": "runs/catalog/{0}/{1}/snapshot.json".format(provider, retrieval_id),
                    "parser-log": "runs/catalog/{0}/{1}/parser-log.json".format(provider, retrieval_id),
                    "cross-match": "runs/catalog/{0}/{1}/cross-match.json".format(provider, retrieval_id),
                }
                artifacts = instance.get("artifacts")
                artifact_paths = {}
                if isinstance(artifacts, list):
                    artifact_paths = {
                        item.get("role"): item.get("path")
                        for item in artifacts if isinstance(item, dict)
                    }
                if artifact_paths != expected_artifacts:
                    report.add(manifest_path, "catalog-provenance-invalid", "catalog manifest artifacts must use the exact retrieval paths and roles")
                raw_metadata = (
                    workspace_dir / "data" / "external" / "catalog" / provider / retrieval_id / "response-metadata.json"
                )
                snapshot = manifest_path.with_name("snapshot.json")
                if not raw_metadata.is_file() or not snapshot.is_file():
                    report.add(manifest_path, "catalog-provenance-invalid", "catalog manifest requires matching raw metadata and snapshot artifacts")
                    continue
                try:
                    raw_instance = _read_json(raw_metadata)
                    snapshot_instance = _read_json(snapshot)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(manifest_path, "catalog-provenance-invalid", "catalog retrieval companion record is unreadable: {0}".format(exc))
                    continue
                raw_response = workspace_dir / expected_artifacts["raw-response"]
                snapshot_response = snapshot_instance.get("raw_response") if isinstance(snapshot_instance, dict) else None
                matching_fields = ("candidate_id", "provider", "retrieval_id", "source_uri", "release", "query_template_id", "status")
                if (
                    not isinstance(raw_instance, dict)
                    or not isinstance(snapshot_instance, dict)
                    or any(raw_instance.get(field) != instance.get(field) for field in matching_fields)
                    or any(snapshot_instance.get(field) != instance.get(field) for field in matching_fields)
                    or not isinstance(snapshot_response, dict)
                    or not raw_response.is_file()
                    or snapshot_response.get("path") != expected_artifacts["raw-response"]
                    or snapshot_response.get("sha256") != _file_sha256(raw_response)
                    or raw_instance.get("response_sha256") != _file_sha256(raw_response)
                ):
                    report.add(manifest_path, "catalog-provenance-invalid", "catalog manifest, raw metadata, and snapshot must describe the same exact retrieval")

        if catalog_raw_metadata_schema is not None:
            for metadata_path in sorted((workspace_dir / "data" / "external" / "catalog").glob("*/*/response-metadata.json")):
                try:
                    instance = _read_json(metadata_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(metadata_path, "schema-violation", "invalid JSON: {0}".format(exc))
                    continue
                _validate(report, metadata_path, instance, catalog_raw_metadata_schema, validate_func)
                relative = metadata_path.relative_to(workspace_dir)
                if isinstance(instance, dict) and (
                    instance.get("candidate_id") != workspace_dir.name
                    or instance.get("provider") != relative.parts[3]
                    or instance.get("retrieval_id") != relative.parts[4]
                ):
                    report.add(metadata_path, "schema-violation", "catalog raw metadata does not match candidate or directory ownership")
                response_path = metadata_path.with_name("response.bin")
                if isinstance(instance, dict) and (
                    not response_path.is_file() or instance.get("response_sha256") != _file_sha256(response_path)
                ):
                    report.add(metadata_path, "catalog-provenance-invalid", "catalog raw metadata does not hash its exact raw response")

        if catalog_snapshot_schema is not None:
            for snapshot_path in sorted((workspace_dir / "runs" / "catalog").glob("*/*/snapshot.json")):
                try:
                    instance = _read_json(snapshot_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(snapshot_path, "schema-violation", "invalid JSON: {0}".format(exc))
                    continue
                _validate(report, snapshot_path, instance, catalog_snapshot_schema, validate_func)
                relative = snapshot_path.relative_to(workspace_dir)
                if isinstance(instance, dict) and (
                    instance.get("candidate_id") != workspace_dir.name
                    or instance.get("provider") != relative.parts[2]
                    or instance.get("retrieval_id") != relative.parts[3]
                ):
                    report.add(snapshot_path, "schema-violation", "catalog snapshot does not match candidate or directory ownership")
                if isinstance(instance, dict) and isinstance(instance.get("raw_response"), dict):
                    _validate_artifacts(report, snapshot_path, workspace_dir, [instance["raw_response"]], "catalog raw response")

        if catalog_cross_match_schema is not None:
            for cross_match_path in sorted((workspace_dir / "runs" / "catalog").glob("*/*/cross-match.json")):
                try:
                    instance = _read_json(cross_match_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(cross_match_path, "schema-violation", "invalid JSON: {0}".format(exc))
                    continue
                _validate(report, cross_match_path, instance, catalog_cross_match_schema, validate_func)
                relative = cross_match_path.relative_to(workspace_dir)
                if not isinstance(instance, dict):
                    continue
                if (
                    instance.get("candidate_id") != workspace_dir.name
                    or instance.get("provider") != relative.parts[2]
                    or instance.get("retrieval_id") != relative.parts[3]
                ):
                    report.add(cross_match_path, "schema-violation", "catalog cross-match does not match candidate or directory ownership")
                snapshot = _candidate_artifact_path(workspace_dir, instance.get("snapshot_path"))
                if (
                    snapshot is None
                    or snapshot != cross_match_path.with_name("snapshot.json")
                    or not snapshot.is_file()
                    or instance.get("snapshot_sha256") != _file_sha256(snapshot)
                ):
                    report.add(cross_match_path, "catalog-provenance-invalid", "catalog cross-match does not reference its exact snapshot")

        for filename, record_schema in catalog_record_schemas.items():
            if record_schema is None:
                continue
            for record_path in sorted((workspace_dir / "runs" / "catalog").glob("*/*/{0}".format(filename))):
                try:
                    instance = _read_json(record_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(record_path, "schema-violation", "invalid JSON: {0}".format(exc))
                    continue
                _validate(report, record_path, instance, record_schema, validate_func)
                relative = record_path.relative_to(workspace_dir)
                if not isinstance(instance, dict):
                    continue
                if (
                    instance.get("candidate_id") != workspace_dir.name
                    or instance.get("provider") != relative.parts[2]
                    or instance.get("retrieval_id") != relative.parts[3]
                ):
                    report.add(record_path, "schema-violation", "catalog normalized record does not match candidate or directory ownership")
                snapshot = _candidate_artifact_path(workspace_dir, instance.get("snapshot_path"))
                if (
                    snapshot is None
                    or snapshot != record_path.with_name("snapshot.json")
                    or not snapshot.is_file()
                    or instance.get("snapshot_sha256") != _file_sha256(snapshot)
                ):
                    report.add(record_path, "catalog-provenance-invalid", "catalog normalized record does not reference its exact snapshot")

        context_path = workspace_dir / "outputs" / "catalog_context.json"
        if catalog_context_schema is not None and context_path.is_file():
            try:
                instance = _read_json(context_path)
            except (OSError, UnicodeError, ValueError) as exc:
                report.add(context_path, "schema-violation", "invalid JSON: {0}".format(exc))
            else:
                _validate(report, context_path, instance, catalog_context_schema, validate_func)
                if isinstance(instance, dict) and instance.get("candidate_id") != workspace_dir.name:
                    report.add(context_path, "schema-violation", "catalog context candidate_id does not match its workspace")
                if isinstance(instance, dict) and isinstance(instance.get("retrievals"), list):
                    for retrieval in instance["retrievals"]:
                        if not isinstance(retrieval, dict):
                            continue
                        manifest = _candidate_artifact_path(workspace_dir, retrieval.get("manifest_path"))
                        if (
                            manifest is None
                            or not manifest.is_file()
                            or retrieval.get("manifest_sha256") != _file_sha256(manifest)
                        ):
                            report.add(context_path, "catalog-provenance-invalid", "catalog context does not reference an exact retrieval manifest")
                            continue
                        try:
                            manifest_instance = _read_json(manifest)
                        except (OSError, UnicodeError, ValueError) as exc:
                            report.add(context_path, "catalog-provenance-invalid", "catalog context retrieval manifest is unreadable: {0}".format(exc))
                            continue
                        expected_manifest = "runs/catalog/{0}/{1}/query-manifest.json".format(
                            retrieval.get("provider"), retrieval.get("retrieval_id")
                        )
                        if (
                            not isinstance(manifest_instance, dict)
                            or retrieval.get("manifest_path") != expected_manifest
                            or any(
                                retrieval.get(field) != manifest_instance.get(field)
                                for field in ("provider", "retrieval_id", "status", "retrieved_at", "expires_at", "citation")
                            )
                        ):
                            report.add(context_path, "catalog-provenance-invalid", "catalog context retrieval fields do not match its exact manifest")

        specialized_records = [
            (
                workspace_dir / "data" / "external" / "planetsynth_characterization.json",
                planetsynth_characterization_schema,
                "PlanetSynth characterization",
                None,
            ),
            (
                workspace_dir / "data" / "external" / "anomalous_transit_hypothesis.json",
                anomalous_transit_hypothesis_schema,
                "anomalous-transit hypothesis",
                None,
            ),
        ]
        for filename_pattern, record_schema, label, engine in (
            ("planetsynth_interpretation.*.json", planetsynth_interpretation_schema, "PlanetSynth interpretation", "planetsynth"),
            ("pyppluss_hypothesis_test.*.json", pyppluss_hypothesis_test_schema, "pyPplusS hypothesis test", "pyppluss"),
        ):
            for record_path in sorted((workspace_dir / "outputs").glob(filename_pattern)):
                specialized_records.append((record_path, record_schema, label, "raw_result_artifact", engine))
        for record in specialized_records:
            record_path, record_schema, label, raw_artifact_field = record[:4]
            engine = record[4] if len(record) > 4 else None
            if record_schema is None or not record_path.is_file():
                continue
            try:
                instance = _read_json(record_path)
            except (OSError, UnicodeError, ValueError) as exc:
                report.add(record_path, "schema-violation", "invalid JSON: {0}".format(exc))
                continue
            _validate(report, record_path, instance, record_schema, validate_func)
            if not isinstance(instance, dict):
                continue
            if instance.get("candidate_id") != workspace_dir.name:
                report.add(record_path, "schema-violation", "{0} candidate_id does not match its workspace".format(label))
            if raw_artifact_field is not None:
                _validate_artifacts(report, record_path, workspace_dir, [instance.get("input_artifact")], "{0} input".format(label))
                _validate_artifacts(report, record_path, workspace_dir, [instance.get(raw_artifact_field)], "{0} raw result".format(label))
                run_id = instance.get("run_id")
                expected_name = "{0}.{1}.json".format(record_path.name.split(".", 1)[0], run_id)
                if not isinstance(run_id, str) or record_path.name != expected_name:
                    report.add(record_path, "schema-violation", "{0} filename does not match its run_id".format(label))
                    continue
                manifest_path = workspace_dir / "runs" / str(engine) / run_id / "engine-run.json"
                if not manifest_path.is_file():
                    report.add(record_path, "schema-violation", "{0} has no matching engine manifest".format(label))
                    continue
                try:
                    manifest = _read_json(manifest_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(record_path, "schema-violation", "matching engine manifest is unreadable: {0}".format(exc))
                    continue
                if not isinstance(manifest, dict) or manifest.get("status") != "succeeded":
                    report.add(record_path, "schema-violation", "{0} requires a successful matching engine manifest".format(label))
                    continue
                output_path = record_path.relative_to(workspace_dir).as_posix()
                output_hash = _file_sha256(record_path)
                if not any(
                    isinstance(artifact, dict)
                    and artifact.get("path") == output_path
                    and artifact.get("sha256") == output_hash
                    for artifact in manifest.get("outputs", [])
                ):
                    report.add(record_path, "schema-violation", "{0} is not hash-recorded by its engine manifest".format(label))

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

    for filename, expected_parts, rule, message in (
        (
            "query-manifest.json",
            ("runs", "catalog"),
            "catalog-query-manifest-outside-candidate",
            "catalog query manifests must be direct files in candidate/<id>/runs/catalog/<provider>/<retrieval-id>/",
        ),
        (
            "snapshot.json",
            ("runs", "catalog"),
            "catalog-snapshot-outside-candidate",
            "catalog snapshots must be direct files in candidate/<id>/runs/catalog/<provider>/<retrieval-id>/",
        ),
        (
            "cross-match.json",
            ("runs", "catalog"),
            "catalog-cross-match-outside-candidate",
            "catalog cross-match records must be direct files in candidate/<id>/runs/catalog/<provider>/<retrieval-id>/",
        ),
        (
            "response-metadata.json",
            ("data", "external", "catalog"),
            "catalog-raw-metadata-outside-candidate",
            "catalog raw metadata must be direct files in candidate/<id>/data/external/catalog/<provider>/<retrieval-id>/",
        ),
    ):
        for path in sorted(root.rglob(filename)):
            if not path.is_file() or LEGACY_SUBTREE in path.parts:
                continue
            try:
                relative = path.relative_to(candidate_root)
            except ValueError:
                relative = Path()
            is_candidate_local = (
                len(relative.parts) == len(expected_parts) + 4
                and tuple(relative.parts[1:1 + len(expected_parts)]) == expected_parts
                and not relative.parts[0].startswith("_")
            )
            if not is_candidate_local:
                report.add(path, rule, message)

    for filename in (
        "stellar-parameters.json",
        "stellar-photometry.json",
        "archive-discovery.json",
        "contrast-curves.json",
    ):
        for path in sorted(root.rglob(filename)):
            if not path.is_file() or LEGACY_SUBTREE in path.parts:
                continue
            try:
                relative = path.relative_to(candidate_root)
            except ValueError:
                relative = Path()
            is_candidate_local = (
                len(relative.parts) == 6
                and tuple(relative.parts[1:3]) == ("runs", "catalog")
                and not relative.parts[0].startswith("_")
            )
            if not is_candidate_local:
                report.add(path, "catalog-normalized-record-outside-candidate", "catalog normalized records must be direct files in candidate/<id>/runs/catalog/<provider>/<retrieval-id>/")

    for filename, expected_parts, rule, message in (
        (
            "radial_velocity_observations.json",
            ("data", "external", "radial_velocity_observations.json"),
            "rv-observations-outside-candidate",
            "RV observations must be direct files in candidate/<id>/data/external/",
        ),
        (
            "rv_keplerian_fit.json",
            ("outputs", "rv_keplerian_fit.json"),
            "rv-fit-outside-candidate",
            "RV fit reports must be direct files in candidate/<id>/outputs/",
        ),
        (
            "planetsynth_characterization.json",
            ("data", "external", "planetsynth_characterization.json"),
            "planetsynth-characterization-outside-candidate",
            "PlanetSynth characterization inputs must be direct files in candidate/<id>/data/external/",
        ),
        (
            "anomalous_transit_hypothesis.json",
            ("data", "external", "anomalous_transit_hypothesis.json"),
            "anomalous-transit-hypothesis-outside-candidate",
            "anomalous-transit hypothesis inputs must be direct files in candidate/<id>/data/external/",
        ),
        (
            "decisive_rejection.json",
            ("decisions", "decisive_rejection.json"),
            "decisive-rejection-outside-candidate",
            "decisive rejection records must be direct files in candidate/<id>/decisions/",
        ),
    ):
        for path in sorted(root.rglob(filename)):
            if not path.is_file() or LEGACY_SUBTREE in path.parts:
                continue
            try:
                relative = path.relative_to(candidate_root)
            except ValueError:
                relative = Path()
            is_candidate_local = (
                len(relative.parts) == len(expected_parts) + 1
                and tuple(relative.parts[-len(expected_parts):]) == expected_parts
                and not relative.parts[0].startswith("_")
            )
            if not is_candidate_local:
                report.add(path, rule, message)

    for filename_pattern, rule, message in (
        (
            "planetsynth_interpretation.*.json",
            "planetsynth-interpretation-outside-candidate",
            "PlanetSynth interpretation reports must be direct files in candidate/<id>/outputs/",
        ),
        (
            "pyppluss_hypothesis_test.*.json",
            "pyppluss-hypothesis-test-outside-candidate",
            "pyPplusS hypothesis test reports must be direct files in candidate/<id>/outputs/",
        ),
    ):
        for path in sorted(root.rglob(filename_pattern)):
            if not path.is_file() or LEGACY_SUBTREE in path.parts:
                continue
            try:
                relative = path.relative_to(candidate_root)
            except ValueError:
                relative = Path()
            is_candidate_local = (
                len(relative.parts) == 3
                and relative.parts[-2] == "outputs"
                and not relative.parts[0].startswith("_")
            )
            if not is_candidate_local:
                report.add(path, rule, message)

    for path in sorted(root.rglob("statistical_vetting_evidence*.json")):
        if not path.is_file() or LEGACY_SUBTREE in path.parts:
            continue
        try:
            relative = path.relative_to(candidate_root)
        except ValueError:
            relative = Path()
        is_candidate_local = (
            (
                len(relative.parts) == 3
                and relative.parts[1] == "outputs"
                and not relative.parts[0].startswith("_")
            )
            or (
                len(relative.parts) == 4
                and relative.parts[0] == "active"
                and relative.parts[2] == "outputs"
            )
        )
        if not is_candidate_local:
            report.add(
                path,
                "statistical-vetting-evidence-outside-candidate",
                "statistical vetting evidence must be direct files in candidate/<id>/outputs/",
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
