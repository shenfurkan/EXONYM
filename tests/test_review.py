import hashlib
import json

import pytest

from exonym.__main__ import main
from exonym.classification import batch_classify, verify_classification_records
from exonym.gatekeeper import set_lifecycle_state
from exonym.isolation import IsolationReport
from exonym.review import apply_classification_review
from exonym.schemas import validate_schemas
from exonym.storage import build_storage_report
from exonym.workspace import create_candidate, load_candidate


def test_review_updates_summary_and_records_hash_bound_evidence(tmp_path):
    candidate = create_candidate(tmp_path, "candidate-alpha")
    evidence = candidate.path / "outputs" / "screen.json"
    evidence.write_text('{"status":"review"}\n', encoding="utf-8")

    review_path = apply_classification_review(
        candidate,
        reviewer="operator",
        reason="Synthetic evidence is sufficient for a routing disposition.",
        evidence_paths=["outputs/screen.json"],
        scientific_disposition="inconclusive",
        review_status="adjudicated",
        retention_class="hold",
    )

    reloaded = load_candidate(tmp_path, "candidate-alpha")
    assert reloaded.metadata["scientific_disposition"] == "inconclusive"
    assert reloaded.metadata["review_status"] == "adjudicated"
    assert reloaded.metadata["retention_class"] == "hold"
    record = json.loads(review_path.read_text(encoding="utf-8"))
    assert record["previous"]["review_status"] == "unreviewed"
    assert record["current"]["retention_class"] == "hold"
    assert record["evidence"][0] == {
        "path": "outputs/screen.json",
        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }

    report = IsolationReport()
    validate_schemas(tmp_path, report)
    assert report.ok, [str(violation) for violation in report.violations]


def test_review_rejects_missing_or_external_evidence(tmp_path):
    candidate = create_candidate(tmp_path, "candidate-alpha")

    with pytest.raises(FileNotFoundError):
        apply_classification_review(
            candidate,
            reviewer="operator",
            reason="Missing evidence must fail closed.",
            evidence_paths=["outputs/missing.json"],
            review_status="reviewed",
        )
    with pytest.raises(ValueError, match="relative candidate-local"):
        apply_classification_review(
            candidate,
            reviewer="operator",
            reason="External evidence must fail closed.",
            evidence_paths=["../outside.json"],
            review_status="reviewed",
        )


def test_schema_validation_detects_review_evidence_tampering(tmp_path):
    candidate = create_candidate(tmp_path, "candidate-alpha")
    evidence = candidate.path / "outputs" / "screen.json"
    evidence.write_text("before\n", encoding="utf-8")
    apply_classification_review(
        candidate,
        reviewer="operator",
        reason="Synthetic review.",
        evidence_paths=["outputs/screen.json"],
        review_status="reviewed",
    )
    evidence.write_text("after\n", encoding="utf-8")

    report = IsolationReport()
    validate_schemas(tmp_path, report)
    assert any(
        violation.rule == "review-evidence-hash-mismatch"
        for violation in report.violations
    )


def test_cli_review_and_classification_filters(tmp_path, capsys):
    root = ["--root", str(tmp_path)]
    assert main(root + ["init", "candidate-alpha"]) == 0
    evidence = tmp_path / "candidate" / "candidate-alpha" / "outputs" / "screen.json"
    evidence.write_text("synthetic\n", encoding="utf-8")

    assert main(
        root
        + [
            "review",
            "candidate-alpha",
            "--reviewer",
            "operator",
            "--reason",
            "Synthetic review.",
            "--evidence",
            "outputs/screen.json",
            "--disposition",
            "false_positive",
            "--review-status",
            "adjudicated",
            "--retention-class",
            "cold",
        ]
    ) == 0
    assert "classification-review-" in capsys.readouterr().out

    assert main(root + ["list", "--disposition", "false_positive"]) == 0
    assert "candidate-alpha" in capsys.readouterr().out
    assert main(root + ["list", "--retention-class", "hold"]) == 0
    assert "candidate-alpha" not in capsys.readouterr().out


def test_storage_report_is_stat_only_and_buckets_bytes(tmp_path):
    candidate = create_candidate(tmp_path, "candidate-alpha")
    raw = candidate.path / "data" / "raw" / "sample.fits"
    output = candidate.path / "outputs" / "result.json"
    raw.write_bytes(b"raw")
    output.write_bytes(b"result")

    report = build_storage_report(tmp_path, "candidate-alpha")
    assert report["scope"] == "candidate"
    assert report["total"]["bytes"] >= len(b"raw") + len(b"result")
    assert report["buckets"]["data/raw"] == {"files": 1, "bytes": 3}
    assert report["buckets"]["outputs"] == {"files": 1, "bytes": 6}
    assert report["stat_errors"] == 0


def test_batch_classification_is_conservative_and_audited(tmp_path):
    active = create_candidate(tmp_path, "candidate-alpha")
    paused = create_candidate(tmp_path, "candidate-beta")
    set_lifecycle_state(paused, "paused", reason="Synthetic administrative hold.")
    (paused.path / "decisions" / "automated_triage.json").write_text(
        "synthetic triage evidence\n", encoding="utf-8"
    )
    for candidate in (active, paused):
        metadata_path = candidate.path / "candidate.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.pop("review_status")
        metadata.pop("retention_class")
        metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    dry_run = batch_classify(tmp_path)
    assert {
        key: dry_run["summary"][key]
        for key in ("total", "proposed", "unchanged", "missing_basis")
    } == {
        "total": 2,
        "proposed": 2,
        "unchanged": 0,
        "missing_basis": 0,
    }
    by_id = {item["candidate_id"]: item for item in dry_run["candidates"]}
    assert by_id["candidate-alpha"]["proposed"]["retention_class"] == "hot"
    assert by_id["candidate-beta"]["proposed"]["retention_class"] == "warm"
    assert by_id["candidate-beta"]["proposed"]["review_status"] == "triaged"
    assert active.metadata["scientific_disposition"] == "unknown"

    applied = batch_classify(tmp_path, apply=True)
    assert applied["summary"]["proposed"] == 2
    assert len(applied["applied_reviews"]) == 2
    assert load_candidate(tmp_path, "candidate-alpha").metadata["retention_class"] == "hot"
    assert load_candidate(tmp_path, "candidate-beta").metadata["retention_class"] == "warm"
    assert load_candidate(tmp_path, "candidate-beta").metadata["review_status"] == "triaged"
    assert load_candidate(tmp_path, "candidate-alpha").metadata["scientific_disposition"] == "unknown"
    integrity = verify_classification_records(tmp_path)
    assert integrity["status"] == "pass"
    assert integrity["review_files"] == 2
    assert integrity["valid_reviews"] == 2


def test_cli_classify_supports_dry_run_then_apply(tmp_path, capsys):
    root = ["--root", str(tmp_path)]
    assert main(root + ["init", "candidate-alpha"]) == 0
    capsys.readouterr()
    metadata_path = tmp_path / "candidate" / "candidate-alpha" / "candidate.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("review_status")
    metadata.pop("retention_class")
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    assert main(root + ["classify"]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["apply"] is False
    assert dry_run["summary"]["proposed"] == 1

    assert main(root + ["classify", "--apply"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["apply"] is True
    assert applied["summary"]["proposed"] == 1
    assert load_candidate(tmp_path, "candidate-alpha").metadata["retention_class"] == "hot"


def test_cli_classify_verify_checks_review_hashes(tmp_path, capsys):
    root = ["--root", str(tmp_path)]
    assert main(root + ["init", "candidate-alpha"]) == 0
    capsys.readouterr()
    metadata_path = tmp_path / "candidate" / "candidate-alpha" / "candidate.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("review_status")
    metadata.pop("retention_class")
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    assert main(root + ["classify", "--apply"]) == 0
    capsys.readouterr()

    assert main(root + ["classify", "--verify"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pass"
    assert result["review_files"] == 1
    assert result["valid_reviews"] == 1
