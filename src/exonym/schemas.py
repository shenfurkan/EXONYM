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
import re
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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
SURVEY_SENSITIVITY_SCHEMA = "survey-sensitivity.schema.json"
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
KNOWN_SIGNAL_EPHEMERIS_MATCH_SCHEMA = "known-signal-ephemeris-match.schema.json"
KNOWN_SIGNAL_EPHEMERIS_EVIDENCE_SCHEMA = "known-signal-ephemeris-evidence.schema.json"
STELLAR_ACTIVITY_SCHEMA = "stellar-activity.schema.json"
PHASE_CURVE_SCHEMA = "phase-curve.schema.json"
DETRENDING_MANIFEST_SCHEMA = "detrending-manifest.schema.json"
LDTK_QUADRATIC_LIMB_DARKENING_PRIOR_SCHEMA = "ldtk-quadratic-limb-darkening-prior.schema.json"
EXOFOP_PRIOR_RETRIEVAL_SCHEMA = "exofop-prior-retrieval.schema.json"
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


def _reject_duplicate_object_keys(pairs: Sequence[Tuple[str, object]]) -> Dict[str, Any]:
    """Reject ambiguous JSON objects rather than keeping their last key value."""
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {0}".format(key))
        result[key] = value
    return result


def _parse_json(content: str) -> object:
    """Load strict JSON without accepting non-finite numeric constants."""
    return json.loads(
        content,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_parse_finite_float,
        object_pairs_hook=_reject_duplicate_object_keys,
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


def _provenance_product_path(sidecar_path: Path) -> Optional[Path]:
    """Resolve the adjacent FITS/FZ file named by a provenance sidecar."""
    suffix = ".provenance.json"
    if not sidecar_path.name.endswith(suffix):
        return None
    stem = sidecar_path.name[: -len(suffix)]
    for filename in (stem, stem + ".fits", stem + ".fz"):
        candidate = sidecar_path.with_name(filename)
        if candidate.is_file():
            return candidate
    return None


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


def _validate_novelty_audit_evidence(
    report: IsolationReport,
    audit_path: Path,
    instance: object,
    workspace_dir: Path,
) -> None:
    """Check v2 novelty response paths and digests without rewriting v1 history."""
    if not isinstance(instance, dict) or instance.get("schema_version") != 2:
        return
    evidence = instance.get("evidence")
    if not isinstance(evidence, list):
        return
    expected_urls: Optional[Dict[str, str]] = None
    if instance.get("status") == "eligible":
        try:
            metadata = _read_json(workspace_dir / "candidate.json")
            identifiers = metadata.get("identifiers") if isinstance(metadata, dict) else None
            tic = identifiers.get("tic") if isinstance(identifiers, dict) else None
            from .survey_harvest import novelty_provider_urls

            expected_urls = dict(novelty_provider_urls(str(tic)))
        except (OSError, UnicodeError, ValueError):
            report.add(
                audit_path,
                "novelty-evidence-provenance-invalid",
                "eligible novelty evidence requires a candidate TIC and canonical registry queries",
            )
    response_paths = set()
    retrieval_ids = set()
    for entry in evidence:
        if not isinstance(entry, dict):
            continue
        response_path = entry.get("response_path")
        resolved_path = _candidate_artifact_path(workspace_dir, response_path)
        relative_path = Path(response_path) if isinstance(response_path, str) else None
        if (
            relative_path is None
            or relative_path.is_absolute()
            or len(relative_path.parts) != 5
            or relative_path.parts[:3] != ("data", "external", "novelty")
            or resolved_path is None
            or not resolved_path.is_file()
        ):
            report.add(
                audit_path,
                "novelty-evidence-provenance-invalid",
                "novelty evidence response path is missing or outside the candidate workspace",
            )
            continue
        hash_matches = entry.get("evidence_sha256") == _file_sha256(resolved_path)
        if not hash_matches:
            report.add(
                audit_path,
                "novelty-evidence-provenance-invalid",
                "novelty evidence response SHA-256 does not match",
            )
        elif expected_urls is not None:
            provider = entry.get("provider")
            source_uri = entry.get("source_uri")
            if provider not in expected_urls or source_uri != expected_urls[provider]:
                report.add(
                    audit_path,
                    "novelty-evidence-provenance-invalid",
                    "eligible novelty evidence does not use its candidate-matched provider query",
                )
            else:
                try:
                    from .survey_harvest import novelty_response_has_registration

                    if novelty_response_has_registration(
                        provider, source_uri, resolved_path.read_bytes(), str(tic)
                    ):
                        report.add(
                            audit_path,
                            "novelty-evidence-provenance-invalid",
                            "eligible novelty evidence contains a registered source record",
                        )
                except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                    report.add(
                        audit_path,
                        "novelty-evidence-provenance-invalid",
                        "eligible novelty evidence response is not semantically valid: {0}".format(exc),
                    )
        response_paths.add(resolved_path)
        retrieval_ids.add(relative_path.parts[3])
    if len(response_paths) != len(evidence) or len(retrieval_ids) > 1:
        report.add(
            audit_path,
            "novelty-evidence-provenance-invalid",
            "novelty evidence responses must be distinct files from one retrieval",
        )


def _positive_finite_number(value: object) -> bool:
    """Return whether one JSON value is a positive finite real number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _validate_triceratops_scientific_evidence(
    report: IsolationReport, workspace_dir: Path
) -> None:
    """Check that active TRICERATOPS reports cannot imply unsupported validation.

    The current implementation records observed phase-folded photometry but
    deliberately does not integrate a calibrated scene model into the FPP.
    These checks therefore require the observed-input provenance and prohibit
    an active report from declaring itself claim-eligible.
    """
    outputs_dir = workspace_dir / "outputs"
    if not outputs_dir.is_dir():
        return
    for report_path in sorted(outputs_dir.glob("triceratops_report*.json")):
        try:
            instance = _read_json(report_path)
        except (OSError, UnicodeError, ValueError) as exc:
            report.add(report_path, "scientific-triceratops-invalid", "invalid JSON: {0}".format(exc))
            continue
        if not isinstance(instance, dict):
            report.add(report_path, "scientific-triceratops-invalid", "TRICERATOPS report must be a JSON object")
            continue
        if instance.get("candidate_id") != workspace_dir.name:
            report.add(report_path, "scientific-triceratops-ownership-invalid", "TRICERATOPS candidate_id does not match its workspace")
        if instance.get("claim_eligible") is not False:
            report.add(
                report_path,
                "scientific-fpp-claim-disabled",
                "TRICERATOPS may not declare an FPP claim eligible before calibrated scene constraints are integrated",
            )

        provenance = instance.get("input_provenance")
        if not isinstance(provenance, dict):
            report.add(report_path, "scientific-observed-photometry-missing", "TRICERATOPS report has no observed-photometry provenance")
            continue
        if provenance.get("representation") != "phase-folded observed candidate photometry":
            report.add(report_path, "scientific-observed-photometry-invalid", "TRICERATOPS input is not identified as observed candidate photometry")
        if provenance.get("flux_error_source") != "reported per-cadence uncertainties":
            report.add(report_path, "scientific-observed-photometry-invalid", "TRICERATOPS does not record reported per-cadence flux uncertainties")
        if not isinstance(provenance.get("raw_cadence_count"), int) or provenance["raw_cadence_count"] < 50:
            report.add(report_path, "scientific-observed-photometry-invalid", "TRICERATOPS requires at least fifty observed cadences")
        if not isinstance(provenance.get("phase_bin_count"), int) or provenance["phase_bin_count"] < 10:
            report.add(report_path, "scientific-observed-photometry-invalid", "TRICERATOPS requires at least ten populated phase bins")
        for field in ("flux_error_scalar", "exposure_days", "observed_depth_ppm"):
            if not _positive_finite_number(provenance.get(field)):
                report.add(report_path, "scientific-observed-photometry-invalid", "TRICERATOPS provenance has no positive finite {0}".format(field))
        input_files = provenance.get("input_files")
        if not isinstance(input_files, list) or not input_files:
            report.add(report_path, "scientific-observed-photometry-invalid", "TRICERATOPS provenance has no hash-bound input photometry")
        else:
            _validate_artifacts(report, report_path, workspace_dir, input_files, "TRICERATOPS input")
        field_sources = provenance.get("ephemeris_field_sources")
        required_fields = ("period_days", "epoch_btjd", "duration_days")
        if not isinstance(field_sources, dict) or any(
            field_sources.get(field) in (None, "synthetic-demo") for field in required_fields
        ):
            report.add(report_path, "scientific-ephemeris-provenance-invalid", "TRICERATOPS requires candidate-derived period, epoch, and duration provenance")
        if not isinstance(instance.get("random_seed"), int) or isinstance(instance.get("random_seed"), bool):
            report.add(report_path, "scientific-triceratops-reproducibility-invalid", "TRICERATOPS report must record an integer random seed")


def _validate_localization_scientific_evidence(report: IsolationReport, workspace_dir: Path) -> None:
    """Prevent uncalibrated source-localization output from becoming a claim."""
    report_path = workspace_dir / "outputs" / "prf_localization_results.json"
    if not report_path.is_file():
        return
    try:
        instance = _read_json(report_path)
    except (OSError, UnicodeError, ValueError) as exc:
        report.add(report_path, "scientific-localization-invalid", "invalid JSON: {0}".format(exc))
        return
    if not isinstance(instance, dict):
        report.add(report_path, "scientific-localization-invalid", "localization report must be a JSON object")
        return
    if instance.get("candidate_id") != workspace_dir.name:
        report.add(report_path, "scientific-localization-ownership-invalid", "localization candidate_id does not match its workspace")
    calibration_status = instance.get("calibration_status")
    summary = instance.get("summary")
    conclusion = summary.get("conclusion") if isinstance(summary, dict) else None
    if calibration_status == "uncalibrated":
        if not isinstance(conclusion, str) or not conclusion.startswith("inconclusive_"):
            report.add(report_path, "scientific-localization-overclaim", "uncalibrated localization must retain an inconclusive conclusion")
    elif calibration_status == "calibrated":
        report.add(
            report_path,
            "scientific-localization-calibration-unsupported",
            "the current Gaussian PRF implementation has no mission-calibrated scene-model contract",
        )
    else:
        report.add(report_path, "scientific-localization-calibration-invalid", "localization calibration_status must be uncalibrated")


def _release_member_path(release_dir: Path, relative_path: object) -> Optional[Path]:
    """Resolve one release-manifest member only when it stays inside the bundle."""
    if not isinstance(relative_path, str) or not relative_path:
        return None
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    release_root = release_dir.resolve()
    member = (release_root / candidate).resolve()
    try:
        member.relative_to(release_root)
    except ValueError:
        return None
    return member


def _validate_release_bundle(report: IsolationReport, release_dir: Path, candidate_id: str) -> None:
    """Verify one self-contained freeze inventory and its candidate ownership."""
    manifest_path = release_dir / "manifest.json"
    if not manifest_path.is_file():
        report.add(release_dir, "release-manifest-missing", "release directory has no manifest.json")
        return
    try:
        manifest = _read_json(manifest_path)
    except (OSError, UnicodeError, ValueError) as exc:
        report.add(manifest_path, "release-manifest-invalid", "invalid JSON: {0}".format(exc))
        return
    if not isinstance(manifest, dict):
        report.add(manifest_path, "release-manifest-invalid", "release manifest must be a JSON object")
        return
    manifest_schema = manifest.get("schema")
    if manifest_schema not in ("exonym-freeze-2", "exonym-freeze-3"):
        report.add(manifest_path, "release-manifest-invalid", "release manifest schema must be exonym-freeze-2 or exonym-freeze-3")
    if manifest.get("version") != release_dir.name:
        report.add(manifest_path, "release-manifest-invalid", "release manifest version does not match its directory")
    if manifest.get("candidate_id") != candidate_id:
        report.add(manifest_path, "release-manifest-invalid", "release manifest candidate_id does not match its workspace")
    expected_replay_status = (
        "integrity-checked-source-import-and-workspace-load"
        if manifest_schema == "exonym-freeze-3"
        else "self-contained-source-and-candidate-evidence-snapshot"
    )
    if manifest.get("replay_status") != expected_replay_status:
        report.add(manifest_path, "release-manifest-invalid", "release does not declare its required replay boundary")

    source_snapshot = manifest.get("source_snapshot")
    workspace_snapshot = manifest.get("workspace_snapshot")
    if not isinstance(source_snapshot, dict) or source_snapshot.get("package_definition") != "source/pyproject.toml":
        report.add(manifest_path, "release-manifest-invalid", "release source snapshot is incomplete")
    if (
        not isinstance(workspace_snapshot, dict)
        or workspace_snapshot.get("candidate_path") != "workspace/candidate/{0}".format(candidate_id)
    ):
        report.add(manifest_path, "release-manifest-invalid", "release workspace snapshot is not candidate-owned")

    expected_candidate = release_dir / "workspace" / "candidate" / candidate_id / "candidate.json"
    if not expected_candidate.is_file():
        report.add(manifest_path, "release-snapshot-missing", "release snapshot has no candidate.json")
    elif manifest.get("candidate_json_sha256") != _file_sha256(expected_candidate):
        report.add(manifest_path, "release-snapshot-hash-mismatch", "candidate snapshot hash does not match the release manifest")
    lock_path = release_dir / "requirements.lock.txt"
    if not lock_path.is_file():
        report.add(manifest_path, "release-snapshot-missing", "release snapshot has no requirements lock")
    elif manifest.get("requirements_lock_sha256") != _file_sha256(lock_path):
        report.add(manifest_path, "release-snapshot-hash-mismatch", "requirements lock hash does not match the release manifest")

    digest_path = release_dir / "manifest.sha256"
    if manifest_schema == "exonym-freeze-3":
        if not digest_path.is_file() or digest_path.is_symlink():
            report.add(manifest_path, "release-manifest-digest-missing", "release has no detached manifest.sha256 digest")
        else:
            try:
                digest_fields = digest_path.read_text(encoding="ascii").split()
            except (OSError, UnicodeError) as exc:
                report.add(digest_path, "release-manifest-digest-invalid", "detached manifest digest is unreadable: {0}".format(exc))
            else:
                if (
                    len(digest_fields) != 2
                    or digest_fields[1] != "manifest.json"
                    or re.fullmatch(r"[0-9a-f]{64}", digest_fields[0]) is None
                ):
                    report.add(digest_path, "release-manifest-digest-invalid", "detached manifest digest must contain '<sha256>  manifest.json'")
                elif digest_fields[0] != _file_sha256(manifest_path):
                    report.add(digest_path, "release-manifest-digest-mismatch", "detached manifest digest does not match manifest.json")
        lock_metadata = manifest.get("requirements_lock")
        if (
            not lock_path.is_file()
            or not isinstance(lock_metadata, dict)
            or lock_metadata.get("format") != "fully-pinned-requirements"
            or not isinstance(lock_metadata.get("package_count"), int)
            or lock_metadata.get("package_count") <= 0
            or lock_metadata.get("sha256") != _file_sha256(lock_path)
        ):
            report.add(manifest_path, "release-manifest-invalid", "release does not record a valid fully pinned lock inventory")
        source_project = release_dir / "source" / "pyproject.toml"
        if (
            not source_project.is_file()
            or manifest.get("source_pyproject_toml_sha256") != _file_sha256(source_project)
        ):
            report.add(manifest_path, "release-snapshot-hash-mismatch", "frozen source package definition hash does not match the release manifest")

    files = manifest.get("files")
    if not isinstance(files, list):
        report.add(manifest_path, "release-manifest-invalid", "release manifest files must be an array")
        return
    expected_paths: Set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            report.add(manifest_path, "release-manifest-invalid", "release file record must be an object")
            continue
        relative_path = entry.get("path")
        member = _release_member_path(release_dir, relative_path)
        if member is None or not member.is_file():
            report.add(manifest_path, "release-file-missing", "release manifest references a missing or unsafe file")
            continue
        if not isinstance(relative_path, str):
            continue
        if relative_path in expected_paths:
            report.add(manifest_path, "release-manifest-invalid", "release manifest contains duplicate file paths")
        expected_paths.add(relative_path)
        if entry.get("sha256") != _file_sha256(member):
            report.add(member, "release-file-hash-mismatch", "release file SHA-256 does not match manifest")
        if entry.get("size_bytes") != member.stat().st_size:
            report.add(member, "release-file-size-mismatch", "release file size does not match manifest")

    actual_paths = {
        path.relative_to(release_dir).as_posix()
        for path in release_dir.rglob("*")
        if path.is_file() and path.name not in ("manifest.json", "manifest.sha256")
    }
    if expected_paths != actual_paths:
        report.add(manifest_path, "release-inventory-mismatch", "release manifest inventory does not match bundle contents")


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
        SURVEY_SENSITIVITY_SCHEMA,
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
        KNOWN_SIGNAL_EPHEMERIS_MATCH_SCHEMA,
        KNOWN_SIGNAL_EPHEMERIS_EVIDENCE_SCHEMA,
        STELLAR_ACTIVITY_SCHEMA,
        PHASE_CURVE_SCHEMA,
        DETRENDING_MANIFEST_SCHEMA,
        LDTK_QUADRATIC_LIMB_DARKENING_PRIOR_SCHEMA,
        EXOFOP_PRIOR_RETRIEVAL_SCHEMA,
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


def _validate_survey_sensitivity_artifact(
    report: IsolationReport,
    path: Path,
    instance: Dict[str, object],
    workspace_dir: Path,
    surveys_root: Path,
) -> None:
    """Validate ownership, inputs, and recovery accounting for one grid artifact."""
    survey_id = instance.get("survey_id")
    if instance.get("candidate_id") != workspace_dir.name:
        report.add(path, "schema-violation", "sensitivity artifact candidate_id does not match its workspace")
        return
    if not isinstance(survey_id, str):
        return
    expected_name = "survey_sensitivity.survey-{0}.json".format(survey_id)
    if path.name != expected_name:
        report.add(path, "schema-violation", "sensitivity artifact filename does not match its survey_id")
        return
    if Path(survey_id).name != survey_id or survey_id in (".", ".."):
        report.add(path, "schema-violation", "sensitivity artifact has an unsafe survey_id")
        return
    survey_path = surveys_root / survey_id / "survey.json"
    target_path = surveys_root / survey_id / "targets" / workspace_dir.name / "target.json"
    try:
        survey_record = _read_json(survey_path)
        target_record = _read_json(target_path)
    except (OSError, UnicodeError, ValueError) as exc:
        report.add(path, "schema-violation", "sensitivity artifact survey ownership is unreadable: {0}".format(exc))
        return
    if (
        not isinstance(survey_record, dict)
        or survey_record.get("survey_id") != survey_id
        or not isinstance(target_record, dict)
        or target_record.get("survey_id") != survey_id
        or target_record.get("candidate_id") != workspace_dir.name
    ):
        report.add(path, "schema-violation", "sensitivity artifact does not match survey and target records")
        return
    if instance.get("sectors") != survey_record.get("sectors"):
        report.add(path, "schema-violation", "sensitivity artifact sectors do not match the frozen survey")

    configuration = instance.get("configuration")
    if not isinstance(configuration, dict):
        return
    try:
        if not math.isclose(
            float(configuration["minimum_snr"]), float(survey_record["review_snr"])
        ):
            report.add(path, "schema-violation", "sensitivity minimum_snr does not match survey review_snr")
    except (KeyError, TypeError, ValueError):
        report.add(path, "schema-violation", "sensitivity configuration has no usable review SNR")
        return

    manifest = instance.get("input_manifest")
    input_files = instance.get("input_files")
    if not isinstance(manifest, list) or not isinstance(input_files, list):
        return
    manifest_paths: List[str] = []
    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        relative_path = entry.get("path")
        artifact_path = _candidate_artifact_path(workspace_dir, relative_path)
        if not isinstance(relative_path, str) or artifact_path is None or not artifact_path.is_file():
            report.add(path, "schema-violation", "sensitivity input manifest references a missing candidate file")
            continue
        manifest_paths.append(relative_path)
        if entry.get("sha256") != _file_sha256(artifact_path):
            report.add(path, "schema-violation", "sensitivity input manifest hash does not match candidate file")
    if len(manifest_paths) != len(set(manifest_paths)) or input_files != manifest_paths:
        report.add(path, "schema-violation", "sensitivity input_files do not match unique hash-recorded inputs")

    try:
        periods = [float(value) for value in configuration["period_days"]]
        durations = [float(value) for value in configuration["duration_hours"]]
        depths = [float(value) for value in configuration["depth_ppm"]]
        phases = [float(value) for value in configuration["phase_offsets"]]
        tolerance = float(configuration["period_agreement_fraction"])
        minimum_snr = float(configuration["minimum_snr"])
        epoch_fraction = float(configuration["epoch_tolerance_duration_fraction"])
    except (KeyError, TypeError, ValueError):
        return
    expected_cells = {(period, duration, depth) for period in periods for duration in durations for depth in depths}
    grouped = {key: [] for key in expected_cells}
    recovery = instance.get("injection_recovery")
    if not isinstance(recovery, list):
        return
    for entry in recovery:
        if not isinstance(entry, dict):
            continue
        try:
            injection = entry["injection"]
            key = (
                float(injection["period_days"]),
                float(injection["duration_hours"]),
                float(injection["depth_ppm"]),
            )
            phase = float(injection["phase_offset"])
            branches = entry["branches"]
        except (KeyError, TypeError, ValueError):
            continue
        if key not in expected_cells or phase not in phases:
            report.add(path, "schema-violation", "sensitivity recovery contains an undeclared grid point")
            continue
        if not isinstance(branches, dict) or set(branches) != {"normalized", "running-median"}:
            report.add(path, "schema-violation", "sensitivity recovery must include both preprocessing branches")
            continue
        branch_outcomes: List[Dict[str, object]] = []
        valid_entry = True
        for branch in branches.values():
            try:
                best = branch["best"]
                period_match = abs(float(best["best_period"]) / key[0] - 1.0) <= tolerance
                delta_days = (float(best["best_epoch"]) - float(injection["epoch_btjd"]) + 0.5 * key[0]) % key[0] - 0.5 * key[0]
                epoch_match = abs(delta_days * 24.0) <= key[1] * epoch_fraction
                snr_pass = float(best["snr"]) >= minimum_snr and int(best["n_distinct_transit_events"]) >= 2
                expected_branch = {
                    "period_match": period_match,
                    "epoch_match": epoch_match,
                    "snr_pass": snr_pass,
                    "recovered": bool(period_match and epoch_match and snr_pass),
                    "best": best,
                }
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                valid_entry = False
                break
            if branch != expected_branch:
                valid_entry = False
                break
            branch_outcomes.append(expected_branch)
        if not valid_entry:
            report.add(path, "schema-violation", "sensitivity recovery branch outcome does not match its BLS result")
            continue
        normalized = branches["normalized"]
        if (
            entry.get("best") != normalized["best"]
            or entry.get("period_match") != all(item["period_match"] for item in branch_outcomes)
            or entry.get("epoch_match") != all(item["epoch_match"] for item in branch_outcomes)
            or entry.get("snr_pass") != all(item["snr_pass"] for item in branch_outcomes)
            or entry.get("recovered") != all(item["recovered"] for item in branch_outcomes)
            or entry.get("epoch_tolerance_hours") != key[1] * epoch_fraction
        ):
            report.add(path, "schema-violation", "sensitivity aggregate recovery does not match both branches")
            continue
        grouped[key].append(bool(entry["recovered"]))

    summary = instance.get("summary")
    if not isinstance(summary, dict):
        return
    expected_trial_count = len(expected_cells) * len(phases)
    if len(recovery) != expected_trial_count or any(len(values) != len(phases) for values in grouped.values()):
        report.add(path, "schema-violation", "sensitivity recovery does not cover the frozen grid")
        return
    expected_recovered_count = sum(sum(values) for values in grouped.values())
    expected_fraction = float(expected_recovered_count) / expected_trial_count
    cells = summary.get("cells")
    if not isinstance(cells, list):
        return
    reported_cells = {
        (float(cell["period_days"]), float(cell["duration_hours"]), float(cell["depth_ppm"])): cell
        for cell in cells
        if isinstance(cell, dict)
    }
    if (
        set(reported_cells) != expected_cells
        or len(cells) != len(expected_cells)
        or summary.get("grid_cell_count") != len(expected_cells)
        or summary.get("trial_count") != expected_trial_count
        or summary.get("recovered_count") != expected_recovered_count
        or not math.isclose(float(summary.get("recovery_fraction", -1)), expected_fraction)
    ):
        report.add(path, "schema-violation", "sensitivity summary does not match recovery trials")
        return
    for key, outcomes in grouped.items():
        cell = reported_cells[key]
        count = sum(outcomes)
        if (
            cell.get("trial_count") != len(outcomes)
            or cell.get("recovered_count") != count
            or not math.isclose(float(cell.get("recovery_fraction", -1)), float(count) / len(outcomes))
        ):
            report.add(path, "schema-violation", "sensitivity grid cell does not match recovery trials")
            break


def _validate_exofop_prior_retrieval(
    report: IsolationReport, path: Path, instance: Dict[str, object], workspace_dir: Path
) -> None:
    """Check ExoFOP raw-response provenance and append-only retrieval ownership."""
    retrieval_id = instance.get("retrieval_id")
    if instance.get("candidate_id") != workspace_dir.name:
        report.add(path, "schema-violation", "ExoFOP prior retrieval candidate_id does not match its workspace")
        return
    if not isinstance(retrieval_id, str):
        return
    expected_relative = Path("runs") / "catalog" / "exofop-priors" / retrieval_id / "exofop-prior-manifest.json"
    if path.relative_to(workspace_dir) != expected_relative:
        report.add(path, "schema-violation", "ExoFOP prior manifest does not match its retrieval directory")
        return
    expected_artifacts = {
        "raw_response": Path("data") / "external" / "catalog" / "exofop-priors" / retrieval_id / "response.csv",
        "raw_metadata": Path("data") / "external" / "catalog" / "exofop-priors" / retrieval_id / "prior-response-metadata.json",
    }
    for field, expected_path in expected_artifacts.items():
        artifact = instance.get(field)
        if not isinstance(artifact, dict):
            continue
        relative_path = artifact.get("path")
        resolved = _candidate_artifact_path(workspace_dir, relative_path)
        if (
            not isinstance(relative_path, str)
            or Path(relative_path) != expected_path
            or resolved is None
            or not resolved.is_file()
        ):
            report.add(path, "schema-violation", "ExoFOP prior {0} path is missing or mismatched".format(field))
            continue
        if artifact.get("sha256") != _file_sha256(resolved):
            report.add(path, "schema-violation", "ExoFOP prior {0} hash does not match".format(field))

    metadata = instance.get("raw_metadata")
    if isinstance(metadata, dict):
        metadata_path = _candidate_artifact_path(workspace_dir, metadata.get("path"))
        if metadata_path is not None and metadata_path.is_file():
            try:
                metadata_record = _read_json(metadata_path)
            except (OSError, UnicodeError, ValueError) as exc:
                report.add(path, "schema-violation", "ExoFOP prior metadata is unreadable: {0}".format(exc))
            else:
                raw_response = instance.get("raw_response")
                if (
                    not isinstance(metadata_record, dict)
                    or metadata_record.get("candidate_id") != workspace_dir.name
                    or metadata_record.get("provider") != "exofop-priors"
                    or metadata_record.get("retrieval_id") != retrieval_id
                    or not isinstance(raw_response, dict)
                    or metadata_record.get("response_sha256") != raw_response.get("sha256")
                ):
                    report.add(path, "schema-violation", "ExoFOP prior metadata does not bind the raw response")

    signals = instance.get("signals")
    if isinstance(signals, list):
        suffixes = [entry.get("signal_suffix") for entry in signals if isinstance(entry, dict)]
        source_rows = [entry.get("source_row_number") for entry in signals if isinstance(entry, dict)]
        if len(suffixes) != len(set(suffixes)) or len(source_rows) != len(set(source_rows)):
            report.add(path, "schema-violation", "ExoFOP prior retrieval contains duplicate signal or source-row identities")


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
    survey_sensitivity_schema = schemas.get(SURVEY_SENSITIVITY_SCHEMA)
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
    known_signal_ephemeris_match_schema = schemas.get(KNOWN_SIGNAL_EPHEMERIS_MATCH_SCHEMA)
    known_signal_ephemeris_evidence_schema = schemas.get(KNOWN_SIGNAL_EPHEMERIS_EVIDENCE_SCHEMA)
    stellar_activity_schema = schemas.get(STELLAR_ACTIVITY_SCHEMA)
    phase_curve_schema = schemas.get(PHASE_CURVE_SCHEMA)
    detrending_manifest_schema = schemas.get(DETRENDING_MANIFEST_SCHEMA)
    ldtk_prior_schema = schemas.get(LDTK_QUADRATIC_LIMB_DARKENING_PRIOR_SCHEMA)
    exofop_prior_schema = schemas.get(EXOFOP_PRIOR_RETRIEVAL_SCHEMA)
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

        releases_dir = workspace_dir / "releases"
        if releases_dir.is_dir():
            for staging_dir in sorted(releases_dir.glob(".*.staging-*")):
                report.add(staging_dir, "release-staging-leftover", "abandoned release staging directory must be removed")
            for release_dir in sorted(releases_dir.iterdir()):
                if release_dir.is_dir() and not release_dir.name.startswith("."):
                    _validate_release_bundle(report, release_dir, workspace_dir.name)

        for path in workspace_dir.rglob("*.provenance.json"):
            if LEGACY_SUBTREE in path.parts or "releases" in path.relative_to(workspace_dir).parts:
                continue
            try:
                instance = _read_json(path)
            except (OSError, UnicodeError, ValueError) as exc:
                report.add(path, "schema-violation", "invalid JSON: {0}".format(exc))
                continue
            _validate(report, path, instance, schemas[PROVENANCE_SCHEMA], validate_func)
            relative = path.relative_to(workspace_dir)
            is_product_sidecar = (
                len(relative.parts) >= 3
                and relative.parts[0:2] == ("data", "raw")
            )
            if is_product_sidecar:
                product_path = _provenance_product_path(path)
                if product_path is None:
                    report.add(
                        path,
                        "provenance-product-missing",
                        "raw-product provenance sidecar has no adjacent FITS/FZ product",
                    )
                elif not isinstance(instance, dict) or instance.get("sha256") != _file_sha256(product_path):
                    report.add(
                        path,
                        "provenance-hash-mismatch",
                        "raw-product provenance SHA-256 does not match adjacent product bytes",
                    )

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
                    if isinstance(instance, dict) and instance.get("parameter") == "fpp":
                        report.add(
                            path,
                            "scientific-fpp-claim-disabled",
                            "FPP claims are disabled until calibrated scene constraints are integrated into TRICERATOPS",
                        )

        novelty_audit_schema = schemas.get(NOVELTY_AUDIT_SCHEMA)
        novelty_audit_path = workspace_dir / "decisions" / "novelty_audit.json"
        if novelty_audit_schema is not None and novelty_audit_path.is_file():
            try:
                instance = _read_json(novelty_audit_path)
            except (OSError, UnicodeError, ValueError) as exc:
                report.add(novelty_audit_path, "schema-violation", "invalid JSON: {0}".format(exc))
            else:
                _validate(report, novelty_audit_path, instance, novelty_audit_schema, validate_func)
                _validate_novelty_audit_evidence(report, novelty_audit_path, instance, workspace_dir)

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

        if detrending_manifest_schema is not None:
            for manifest_path in sorted(
                (workspace_dir / "outputs").glob("detrending_manifest.*.json")
            ):
                try:
                    instance = _read_json(manifest_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(manifest_path, "schema-violation", "invalid JSON: {0}".format(exc))
                    continue
                _validate(report, manifest_path, instance, detrending_manifest_schema, validate_func)
                if not isinstance(instance, dict):
                    continue
                if instance.get("candidate_id") != workspace_dir.name:
                    report.add(
                        manifest_path,
                        "schema-violation",
                        "detrending manifest candidate_id does not match its workspace",
                    )
                method = instance.get("method")
                if manifest_path.name != "detrending_manifest.{0}.json".format(method):
                    report.add(
                        manifest_path,
                        "schema-violation",
                        "detrending manifest method does not match its filename",
                    )
                _validate_artifacts(
                    report,
                    manifest_path,
                    workspace_dir,
                    [instance.get("artifact")],
                    "detrending",
                )

        if ldtk_prior_schema is not None:
            prior_path = workspace_dir / "outputs" / "ldtk_quadratic_limb_darkening_prior.json"
            if prior_path.is_file():
                try:
                    instance = _read_json(prior_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(prior_path, "schema-violation", "invalid JSON: {0}".format(exc))
                else:
                    _validate(report, prior_path, instance, ldtk_prior_schema, validate_func)
                    if isinstance(instance, dict):
                        if instance.get("candidate_id") != workspace_dir.name:
                            report.add(
                                prior_path,
                                "schema-violation",
                                "LDTk prior candidate_id does not match its workspace",
                            )
                        provenance = instance.get("input_provenance")
                        if isinstance(provenance, dict):
                            _validate_artifacts(
                                report,
                                prior_path,
                                workspace_dir,
                                [
                                    {
                                        "path": provenance.get("stellar_parameters_path"),
                                        "sha256": provenance.get("stellar_parameters_sha256"),
                                    }
                                ],
                                "LDTk stellar-parameter input",
                            )

        if exofop_prior_schema is not None:
            prior_runs = workspace_dir / "runs" / "catalog" / "exofop-priors"
            for manifest_path in sorted(prior_runs.glob("*/exofop-prior-manifest.json")):
                try:
                    instance = _read_json(manifest_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(manifest_path, "schema-violation", "invalid JSON: {0}".format(exc))
                    continue
                _validate(report, manifest_path, instance, exofop_prior_schema, validate_func)
                if isinstance(instance, dict):
                    _validate_exofop_prior_retrieval(report, manifest_path, instance, workspace_dir)

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

        if known_signal_ephemeris_evidence_schema is not None:
            evidence_path = workspace_dir / "decisions" / "known_signal_ephemerides.json"
            if evidence_path.is_file():
                try:
                    evidence_instance = _read_json(evidence_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(evidence_path, "schema-violation", "invalid JSON: {0}".format(exc))
                else:
                    _validate(report, evidence_path, evidence_instance, known_signal_ephemeris_evidence_schema, validate_func)
                    if isinstance(evidence_instance, dict):
                        if evidence_instance.get("candidate_id") != workspace_dir.name:
                            report.add(evidence_path, "schema-violation", "known-signal evidence candidate_id does not match its workspace")
                        records = evidence_instance.get("records")
                        if isinstance(records, list):
                            for record in records:
                                if isinstance(record, dict):
                                    _validate_artifacts(
                                        report,
                                        evidence_path,
                                        workspace_dir,
                                        [record.get("raw_artifact")],
                                        "known-signal evidence raw input",
                                    )

        if known_signal_ephemeris_match_schema is not None:
            for match_path in sorted((workspace_dir / "outputs").glob("known_signal_ephemeris_match*.json")):
                try:
                    instance = _read_json(match_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(match_path, "schema-violation", "invalid JSON: {0}".format(exc))
                    continue
                _validate(report, match_path, instance, known_signal_ephemeris_match_schema, validate_func)
                if not isinstance(instance, dict):
                    continue
                signal = instance.get("signal")
                expected_name = "known_signal_ephemeris_match{0}.json".format(signal or "")
                if instance.get("candidate_id") != workspace_dir.name or match_path.name != expected_name:
                    report.add(match_path, "schema-violation", "known-signal ephemeris match does not match candidate or signal filename ownership")
                candidate_ephemeris = instance.get("candidate_ephemeris")
                if isinstance(candidate_ephemeris, dict):
                    input_artifact = candidate_ephemeris.get("input_artifact")
                    if isinstance(input_artifact, dict):
                        _validate_artifacts(report, match_path, workspace_dir, [input_artifact], "known-signal ephemeris input")
                source_snapshots = instance.get("source_snapshots")
                if isinstance(source_snapshots, list):
                    for source_snapshot in source_snapshots:
                        if not isinstance(source_snapshot, dict):
                            continue
                        artifacts = [
                            source_snapshot.get("query_manifest"),
                            source_snapshot.get("snapshot"),
                        ]
                        if all(isinstance(item, dict) for item in artifacts):
                            _validate_artifacts(report, match_path, workspace_dir, artifacts, "known-signal catalog inputs")

        if stellar_activity_schema is not None:
            activity_path = workspace_dir / "outputs" / "stellar_activity_results.json"
            if activity_path.is_file():
                try:
                    activity_instance = _read_json(activity_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(activity_path, "schema-violation", "invalid JSON: {0}".format(exc))
                else:
                    _validate(report, activity_path, activity_instance, stellar_activity_schema, validate_func)

        if phase_curve_schema is not None:
            phase_curve_path = workspace_dir / "outputs" / "phase_curve_results.json"
            if phase_curve_path.is_file():
                try:
                    phase_curve_instance = _read_json(phase_curve_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    report.add(phase_curve_path, "schema-violation", "invalid JSON: {0}".format(exc))
                else:
                    _validate(report, phase_curve_path, phase_curve_instance, phase_curve_schema, validate_func)

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

        if survey_sensitivity_schema is not None:
            outputs_dir = workspace_dir / "outputs"
            if outputs_dir.is_dir():
                for path in sorted(outputs_dir.glob("survey_sensitivity*.json")):
                    if not metadata_path.is_file():
                        report.add(
                            path,
                            "survey-sensitivity-outside-candidate",
                            "sensitivity artifacts require an owning candidate.json workspace",
                        )
                        continue
                    try:
                        instance = _read_json(path)
                    except (OSError, UnicodeError, ValueError) as exc:
                        report.add(path, "schema-violation", "unreadable JSON: {0}".format(exc))
                        continue
                    _validate(report, path, instance, survey_sensitivity_schema, validate_func)
                    if isinstance(instance, dict):
                        _validate_survey_sensitivity_artifact(
                            report, path, instance, workspace_dir, surveys_root
                        )

        _validate_triceratops_scientific_evidence(report, workspace_dir)
        _validate_localization_scientific_evidence(report, workspace_dir)

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

    for path in sorted(root.rglob("survey_sensitivity*.json")):
        if not path.is_file() or LEGACY_SUBTREE in path.parts:
            continue
        try:
            relative = path.relative_to(candidate_root)
        except ValueError:
            relative = Path()
        is_candidate_local = (
            len(relative.parts) >= 3
            and relative.parts[-2] == "outputs"
            and not any(part.startswith("_") for part in relative.parts[:-2])
        )
        if not is_candidate_local:
            report.add(
                path,
                "survey-sensitivity-outside-candidate",
                "sensitivity artifacts must be direct files in candidate/<id>/outputs/",
            )

    for path in sorted(root.rglob("exofop-prior-manifest.json")):
        if not path.is_file() or LEGACY_SUBTREE in path.parts:
            continue
        try:
            relative = path.relative_to(candidate_root)
        except ValueError:
            relative = Path()
        is_candidate_local = (
            len(relative.parts) == 6
            and relative.parts[1:4] == ("runs", "catalog", "exofop-priors")
            and not relative.parts[0].startswith("_")
        )
        if not is_candidate_local:
            report.add(
                path,
                "exofop-prior-manifest-outside-candidate",
                "ExoFOP prior manifests must be direct files in candidate/<id>/runs/catalog/exofop-priors/<retrieval-id>/",
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
                len(relative.parts) >= len(expected_parts) + 3
                and tuple(relative.parts[-len(expected_parts) - 3:-3]) == expected_parts
                and not any(p.startswith("_") for p in relative.parts[:-len(expected_parts) - 3])
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
                len(relative.parts) >= 5
                and tuple(relative.parts[-5:-3]) == ("runs", "catalog")
                and not any(p.startswith("_") for p in relative.parts[:-5])
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
                len(relative.parts) >= len(expected_parts) + 1
                and tuple(relative.parts[-len(expected_parts):]) == expected_parts
                and not any(p.startswith("_") for p in relative.parts[:-len(expected_parts)])
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
                len(relative.parts) >= 3
                and relative.parts[-2] == "outputs"
                and not any(p.startswith("_") for p in relative.parts[:-2])
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
            len(relative.parts) >= 3
            and relative.parts[-2] == "outputs"
            and not any(p.startswith("_") for p in relative.parts[:-2])
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
