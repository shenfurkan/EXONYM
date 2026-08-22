"""Tests for candidate-local verification cache behavior."""

import json

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
