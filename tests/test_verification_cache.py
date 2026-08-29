"""Tests for candidate-local verification cache behavior."""

import hashlib
import json
import os

from exonym.verification_cache import CandidateVerificationCache
from exonym.workspace import create_candidate


def test_candidate_cache_reuses_unchanged_hash_and_metadata(tmp_path):
    workspace = create_candidate(tmp_path, "cache-synthetic")
    artifact = workspace.path / "outputs" / "artifact.json"
    artifact.write_text('{"result": "synthetic"}\n', encoding="utf-8")
    metadata_path = workspace.path / "candidate.json"

    first = CandidateVerificationCache(tmp_path)
    expected_hash = first.sha256(artifact)
    assert first.read_candidate_json(metadata_path, json.loads)["candidate_id"] == "cache-synthetic"
    first.save()

    second = CandidateVerificationCache(tmp_path)
    assert second.sha256(artifact) == expected_hash
    assert second.read_candidate_json(metadata_path, json.loads)["candidate_id"] == "cache-synthetic"
    assert second.statistics() == {
        "hash_cache_hits": 1,
        "hash_cache_misses": 0,
        "candidate_json_cache_hits": 1,
        "candidate_json_cache_misses": 0,
    }


def test_changed_candidate_json_drops_stale_digest_before_hashing(tmp_path):
    workspace = create_candidate(tmp_path, "cache-fingerprint-synthetic")
    metadata_path = workspace.path / "candidate.json"

    first = CandidateVerificationCache(tmp_path)
    stale_digest = first.sha256(metadata_path)
    first.save()

    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    expected_digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    assert expected_digest != stale_digest

    second = CandidateVerificationCache(tmp_path)
    assert second.read_candidate_json(metadata_path, json.loads)["candidate_id"] == (
        "cache-fingerprint-synthetic"
    )
    assert second.sha256(metadata_path) == expected_digest
    assert second.statistics() == {
        "hash_cache_hits": 1,
        "hash_cache_misses": 0,
        "candidate_json_cache_hits": 0,
        "candidate_json_cache_misses": 1,
    }

    second.save()
    persisted = json.loads(
        (workspace.path / "outputs" / ".exonym-verify-cache.json").read_text(encoding="utf-8")
    )
    assert persisted["files"]["candidate.json"]["sha256"] == expected_digest
    third = CandidateVerificationCache(tmp_path)
    assert third.sha256(metadata_path) == expected_digest
    assert third.statistics()["hash_cache_hits"] == 1


def test_cache_rehashes_and_reparses_same_size_mtime_rewrites(tmp_path):
    workspace = create_candidate(tmp_path, "cache-synthetic")
    artifact = workspace.path / "outputs" / "artifact.json"
    artifact.write_text("original", encoding="utf-8")
    metadata_path = workspace.path / "candidate.json"

    first = CandidateVerificationCache(tmp_path)
    first.sha256(artifact)
    first.read_candidate_json(metadata_path, json.loads)
    first.save()

    artifact_stat = artifact.stat()
    metadata_stat = metadata_path.stat()
    artifact.write_text("rewrote!", encoding="utf-8")
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8").replace("cache-synthetic", "cache-rewritten"),
        encoding="utf-8",
    )
    os.utime(artifact, ns=(artifact_stat.st_atime_ns, artifact_stat.st_mtime_ns))
    os.utime(metadata_path, ns=(metadata_stat.st_atime_ns, metadata_stat.st_mtime_ns))
    assert artifact.stat().st_size == artifact_stat.st_size
    assert artifact.stat().st_mtime_ns == artifact_stat.st_mtime_ns
    assert metadata_path.stat().st_size == metadata_stat.st_size
    assert metadata_path.stat().st_mtime_ns == metadata_stat.st_mtime_ns

    second = CandidateVerificationCache(tmp_path)
    assert second.sha256(artifact) == hashlib.sha256(b"rewrote!").hexdigest()
    assert second.read_candidate_json(metadata_path, json.loads)["candidate_id"] == "cache-rewritten"
    assert second.statistics() == {
        "hash_cache_hits": 0,
        "hash_cache_misses": 1,
        "candidate_json_cache_hits": 0,
        "candidate_json_cache_misses": 1,
    }


def test_copied_cache_is_not_trusted_in_another_workspace(tmp_path):
    source_root = tmp_path / "source-checkout"
    target_root = tmp_path / "target-checkout"
    source = create_candidate(source_root, "cache-source")
    target = create_candidate(target_root, "cache-target")
    source_artifact = source.path / "outputs" / "artifact.json"
    target_artifact = target.path / "outputs" / "artifact.json"
    source_artifact.write_text('{"result": "source"}\n', encoding="utf-8")
    target_artifact.write_text('{"result": "target"}\n', encoding="utf-8")

    source_cache = CandidateVerificationCache(source_root)
    source_cache.sha256(source_artifact)
    source_cache.read_candidate_json(source.path / "candidate.json", json.loads)
    source_cache.save()

    copied = json.loads(
        (source.path / "outputs" / ".exonym-verify-cache.json").read_text(encoding="utf-8")
    )
    target_artifact_fingerprint = target_artifact.stat()
    target_metadata = target.path / "candidate.json"
    target_metadata_fingerprint = target_metadata.stat()
    copied["files"]["outputs/artifact.json"].update(
        {
            "mtime_ns": target_artifact_fingerprint.st_mtime_ns,
            "size": target_artifact_fingerprint.st_size,
            "sha256": "0" * 64,
        }
    )
    copied["files"]["candidate.json"].update(
        {
            "mtime_ns": target_metadata_fingerprint.st_mtime_ns,
            "size": target_metadata_fingerprint.st_size,
            "sha256": "0" * 64,
            "json": {"candidate_id": "untrusted-cache"},
        }
    )
    target_cache_path = target.path / "outputs" / ".exonym-verify-cache.json"
    target_cache_path.write_text(json.dumps(copied), encoding="utf-8")

    target_cache = CandidateVerificationCache(target_root)
    assert target_cache.sha256(target_artifact) == hashlib.sha256(
        target_artifact.read_bytes()
    ).hexdigest()
    assert target_cache.read_candidate_json(target_metadata, json.loads)["candidate_id"] == "cache-target"
    assert target_cache.statistics() == {
        "hash_cache_hits": 0,
        "hash_cache_misses": 1,
        "candidate_json_cache_hits": 0,
        "candidate_json_cache_misses": 1,
    }

    target_cache.save()
    refreshed = json.loads(target_cache_path.read_text(encoding="utf-8"))
    assert refreshed["workspace_scope"] != copied["workspace_scope"]
    assert refreshed["files"]["outputs/artifact.json"]["sha256"] != "0" * 64
