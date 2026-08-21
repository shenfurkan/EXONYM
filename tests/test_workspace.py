import json

import pytest

from exonym.workspace import (
    create_candidate,
    discover_candidates,
    load_candidate,
    validate_candidate_id,
    validate_metadata,
    workspace_layout,
)


def test_create_candidate_builds_standard_workspace(tmp_path):
    candidate = create_candidate(
        tmp_path, "candidate-alpha", toi="1234.01", tic="123456789"
    )

    assert candidate.candidate_id == "candidate-alpha"
    assert candidate.metadata["schema_version"] == 2
    assert candidate.metadata["lifecycle"]["state"] == "active"
    assert candidate.metadata["workflow"]["phase"] == "intake"
    assert candidate.metadata["scientific_disposition"] == "unknown"
    assert candidate.path.joinpath("candidate.json").is_file()
    for path in workspace_layout(candidate).values():
        if path.name != "candidate.json":
            assert path.exists()

    metadata = json.loads(candidate.path.joinpath("candidate.json").read_text(encoding="utf-8"))
    assert metadata["identifiers"]["tic"] == "123456789"


def test_discover_and_load_candidates(tmp_path):
    create_candidate(tmp_path, "candidate-beta")
    create_candidate(tmp_path, "candidate-alpha")

    assert [candidate.candidate_id for candidate in discover_candidates(tmp_path)] == [
        "candidate-alpha",
        "candidate-beta",
    ]
    loaded = load_candidate(tmp_path, "candidate-alpha")
    assert loaded.metadata["identifiers"]["toi"] is None


def test_discovery_and_loading_reject_nested_candidate_workspaces(tmp_path):
    nested = tmp_path / "candidate" / "legacy-group" / "candidate-alpha"
    nested.mkdir(parents=True)
    (nested / "candidate.json").write_text(
        '{"schema_version": 2, "candidate_id": "candidate-alpha"}\n', encoding="utf-8"
    )

    assert discover_candidates(tmp_path) == []
    with pytest.raises(FileNotFoundError):
        load_candidate(tmp_path, "candidate-alpha")


def test_candidate_creation_never_overwrites_an_existing_workspace(tmp_path):
    create_candidate(tmp_path, "candidate-alpha")

    with pytest.raises(FileExistsError):
        create_candidate(tmp_path, "candidate-alpha")


def test_candidate_id_collision_is_case_insensitive(tmp_path):
    create_candidate(tmp_path, "candidate-alpha")

    with pytest.raises(FileExistsError):
        create_candidate(tmp_path, "CANDIDATE-ALPHA")


def test_validate_metadata_rejects_bad_lifecycle(tmp_path):
    candidate = create_candidate(tmp_path, "candidate-alpha")
    broken = dict(candidate.metadata)
    broken["lifecycle"] = {"state": "mystery", "state_since": "x"}

    with pytest.raises(ValueError, match="lifecycle"):
        validate_metadata(broken, "candidate-alpha")


@pytest.mark.parametrize(
    "candidate_id",
    ["../escape", "candidate space", "", "/absolute", "CON", "alpha.", "nul"],
)
def test_candidate_id_rejects_unsafe_names(candidate_id):
    with pytest.raises(ValueError):
        validate_candidate_id(candidate_id)


def test_create_candidate_validates_toi_and_tic_format(tmp_path):
    with pytest.raises(ValueError, match="toi"):
        create_candidate(tmp_path, "candidate-alpha", toi="not-a-number")
    with pytest.raises(ValueError, match="tic"):
        create_candidate(tmp_path, "candidate-alpha", tic="0abc")


@pytest.mark.parametrize(
    "payload, message",
    (
        ('{"candidate_id":"candidate-alpha","candidate_id":"other"}', "duplicate JSON key"),
        ('{"schema_version":1e999}', "non-finite JSON number"),
    ),
)
def test_load_candidate_rejects_ambiguous_or_nonfinite_metadata(tmp_path, payload, message):
    candidate = create_candidate(tmp_path, "candidate-alpha")
    candidate.path.joinpath("candidate.json").write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_candidate(tmp_path, "candidate-alpha")
