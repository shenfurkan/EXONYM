"""Unit coverage for the workspace checkpoint engine."""

import json
from pathlib import Path

import pytest

from exonym import checkpoints
from exonym.workspace import create_candidate, load_candidate


@pytest.fixture()
def workspace(tmp_path):
    return create_candidate(tmp_path, "checkpoint-fixture")


def test_checkpoint_save_manifest_and_list_roundtrip(workspace):
    (workspace.path / "outputs").mkdir(parents=True, exist_ok=True)
    artifact = workspace.path / "outputs" / "probe.txt"
    artifact.write_text("synthetic analysis state", encoding="utf-8")

    manifest_path = checkpoints.save_checkpoint(workspace, "Pre-Fit")
    archive_path = manifest_path.with_suffix("").with_suffix(".tar.gz")

    assert archive_path.is_file() and archive_path.name.endswith(".tar.gz")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidate_id"] == workspace.candidate_id
    assert manifest["label"] == "pre-fit"
    assert len(manifest["archive"]["sha256"]) == 64
    paths = {entry["path"] for entry in manifest["files"]}
    assert "candidate.json" in paths and "outputs/probe.txt" in paths
    # Exclusions are structural, not incidental.
    assert not any(path.startswith("data/raw") for path in paths)

    records = checkpoints.list_checkpoints(workspace)
    assert [r["checkpoint_id"] for r in records] == [manifest["checkpoint_id"]]
    assert records[0]["lifecycle_state"] == workspace.metadata["lifecycle"]["state"]


def test_checkpoint_restore_is_hash_verified_and_atomic(workspace):
    outputs = workspace.path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    probe = outputs / "probe.txt"
    probe.write_text("original", encoding="utf-8")
    manifest_path = checkpoints.save_checkpoint(workspace, "baseline")
    checkpoint_id = json.loads(manifest_path.read_text(encoding="utf-8"))["checkpoint_id"]

    probe.write_text("mutated by a failed experiment", encoding="utf-8")

    summary = checkpoints.restore_checkpoint(workspace, checkpoint_id, assume_yes=True)

    assert probe.read_text(encoding="utf-8") == "original"
    assert "outputs" in summary["restored_entries"]
    audit_log = workspace.path / "audit_log.jsonl"
    lines = audit_log.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[-1])
    assert record["action"] == "checkpoint_restore"
    assert record["result"] == "success"

    # Tampering with the archive must abort before touching the workspace.
    archive_path = workspace.path / "checkpoints" / (checkpoint_id + ".tar.gz")
    original_bytes = archive_path.read_bytes()
    archive_path.write_bytes(original_bytes[:-1] + bytes([original_bytes[-1] ^ 0xFF]))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        checkpoints.restore_checkpoint(workspace, checkpoint_id, assume_yes=True)
    assert probe.read_text(encoding="utf-8") == "original"

    checkpoints.delete_checkpoint(workspace, checkpoint_id)
    assert not archive_path.exists()
    assert not manifest_path.exists()


def test_checkpoint_rejects_unsafe_identifiers_and_unknown_ids(workspace):
    with pytest.raises(ValueError):
        checkpoints.restore_checkpoint(workspace, "../escape", assume_yes=True)
    with pytest.raises(FileNotFoundError):
        checkpoints.restore_checkpoint(
            workspace, "20260101T000000Z_missing", assume_yes=True
        )
    with pytest.raises(ValueError):
        checkpoints.save_checkpoint(workspace, "Bad Label!")


def test_cli_checkpoint_save_list_delete(tmp_path, capsys):
    from exonym.__main__ import main

    root = ["--root", str(tmp_path)]
    assert main(root + ["init", "ckpt-cli-target"]) == 0
    assert main(root + ["checkpoint", "save", "ckpt-cli-target", "--name", "snap"]) == 0

    listed = main(root + ["checkpoint", "list", "ckpt-cli-target"])
    assert listed == 0
    out = capsys.readouterr().out
    assert "snap" in out

    checkpoint_dir = tmp_path / "candidate" / "ckpt-cli-target" / "checkpoints"
    checkpoint_id = next(
        path.name[: -len(".manifest.json")]
        for path in checkpoint_dir.glob("*.manifest.json")
    )
    assert main(root + ["checkpoint", "delete", "ckpt-cli-target", "--id", checkpoint_id]) == 0
    assert list(checkpoint_dir.glob("*")) == []
