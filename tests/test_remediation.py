"""Tests for safe derived-artifact remediation."""

import hashlib
import json

import numpy as np
import pytest

from exonym.detrending import detrend_candidate
from exonym.remediation import numerical_npz_sha256, remediate_candidate_drift, semantic_json_sha256
from exonym.workspace import create_candidate


def _detrended_artifact(tmp_path):
    workspace = create_candidate(tmp_path, "remediation-synthetic")
    raw_path = workspace.path / "data" / "raw" / "source.fits"
    raw_path.write_bytes(b"synthetic raw product")
    raw_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    raw_path.with_name("source.provenance.json").write_text(
        json.dumps(
            {
                "source_uri": "https://example.invalid/source",
                "download_timestamp_utc": "2026-01-01T00:00:00Z",
                "sha256": raw_digest,
                "fetched_by": "synthetic-test",
            }
        ),
        encoding="utf-8",
    )
    time = np.linspace(0.0, 8.0, 101)
    result = detrend_candidate(
        workspace,
        time,
        1.0 + 0.001 * np.sin(time),
        window_days=0.5,
        sector=np.ones(time.size, dtype=int),
        input_products=[{"path": "data/raw/source.fits", "sha256": raw_digest}],
    )
    return workspace, result


def test_remediation_refreshes_semantically_identical_detrending_archive(tmp_path):
    workspace, result = _detrended_artifact(tmp_path)
    with np.load(result.artifact_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    np.savez(result.artifact_path, **payload)

    actions = remediate_candidate_drift(tmp_path)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert actions[workspace.candidate_id] == [
        "refreshed outputs/detrending_manifest.running-median.json artifact digest"
    ]
    assert manifest["artifact"]["sha256"] == hashlib.sha256(result.artifact_path.read_bytes()).hexdigest()


def test_remediation_skips_unreadable_detrending_artifacts(tmp_path):
    _, result = _detrended_artifact(tmp_path)
    result.artifact_path.write_bytes(b"truncated")

    assert remediate_candidate_drift(tmp_path) == {}


def test_remediation_keeps_bls_derived_config_bound_to_refreshed_manifest(tmp_path):
    workspace = create_candidate(tmp_path, "remediation-bls")
    outputs = workspace.path / "outputs"
    result_path = outputs / "bls_search_results.json"
    result_path.write_text(
        json.dumps({"generated_utc": "2026-01-01T00:00:00Z", "result": "stable"}),
        encoding="utf-8",
    )
    manifest_path = outputs / "bls_search_manifest.json"
    manifest = {
        "schema": "exonym-bls-search-manifest-1",
        "result_path": "outputs/bls_search_results.json",
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "result_semantic_sha256": semantic_json_sha256(result_path),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = workspace.path / "config" / "transit_config.json"
    config_path.write_text(
        json.dumps(
            {
                "source": "candidate-data-bls",
                "bls_provenance": {
                    "result": {
                        "path": manifest["result_path"],
                        "sha256": manifest["result_sha256"],
                    },
                    "manifest": {
                        "path": "outputs/bls_search_manifest.json",
                        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps({"generated_utc": "2026-01-02T00:00:00Z", "result": "stable"}),
        encoding="utf-8",
    )

    actions = remediate_candidate_drift(tmp_path)

    refreshed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refreshed_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert actions[workspace.candidate_id] == [
        "refreshed outputs/bls_search_manifest.json result digest",
        "refreshed config/transit_config.json BLS provenance",
    ]
    assert refreshed_config["bls_provenance"]["result"]["sha256"] == refreshed_manifest["result_sha256"]
    assert refreshed_config["bls_provenance"]["manifest"]["sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


def test_remediation_never_rewrites_existing_triage_evidence(tmp_path, monkeypatch):
    # Arrange
    workspace = create_candidate(tmp_path, "remediation-triage-policy")
    triage_path = workspace.path / "decisions" / "automated_triage.json"
    triage_path.write_text(
        json.dumps(
            {
                "policy_id": "custom-pre-vetting-policy",
                "policy_version": "2.4.1",
            }
        ),
        encoding="utf-8",
    )
    original = triage_path.read_bytes()
    run_manifest_path = (
        workspace.path / "runs" / "statistical-vetting" / "run-001" / "engine-run.json"
    )
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path.write_text(
        json.dumps(
            {
                "outputs": [
                    {
                        "path": "decisions/automated_triage.json",
                        "sha256": hashlib.sha256(original).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    original_manifest = run_manifest_path.read_bytes()
    monkeypatch.setattr(
        "exonym.engines.run_automated_triage",
        lambda *_args, **_kwargs: pytest.fail("--fix must not rerun triage evidence"),
    )

    # Act
    actions = remediate_candidate_drift(tmp_path)

    # Assert
    assert actions == {}
    assert triage_path.read_bytes() == original
    assert run_manifest_path.read_bytes() == original_manifest
    assert json.loads(run_manifest_path.read_text(encoding="utf-8"))["outputs"][0]["sha256"] == hashlib.sha256(
        triage_path.read_bytes()
    ).hexdigest()


def test_remediation_skips_manifest_paths_that_escape_the_workspace(tmp_path):
    workspace = create_candidate(tmp_path, "remediation-contained-paths")
    external_npz = tmp_path / "outside.npz"
    np.savez(external_npz, time=np.array([0.0, 1.0]), flux=np.array([1.0, 1.0]))
    external_result = tmp_path / "outside.json"
    external_result.write_text(
        json.dumps({"generated_utc": "2026-01-01T00:00:00Z", "result": "stable"}),
        encoding="utf-8",
    )
    detrending_manifest_path = workspace.path / "outputs" / "detrending_manifest.running-median.json"
    detrending_manifest_path.write_text(
        json.dumps(
            {
                "artifact": {
                    "path": "../../outside.npz",
                    "data_sha256": numerical_npz_sha256(external_npz),
                    "sha256": "0" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    search_manifest_path = workspace.path / "outputs" / "bls_search_manifest.json"
    search_manifest_path.write_text(
        json.dumps(
            {
                "result_path": "../../outside.json",
                "result_semantic_sha256": semantic_json_sha256(external_result),
                "result_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    original_detrending_manifest = detrending_manifest_path.read_bytes()
    original_search_manifest = search_manifest_path.read_bytes()

    actions = remediate_candidate_drift(tmp_path)

    assert actions == {}
    assert detrending_manifest_path.read_bytes() == original_detrending_manifest
    assert search_manifest_path.read_bytes() == original_search_manifest
