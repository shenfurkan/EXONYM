import json
import shutil

import pytest

from exonym.isolation import IsolationReport
from exonym.schemas import validate_schemas
from exonym.workspace import create_candidate


def _make_repo(tmp_path, with_templates=True):
    create_candidate(tmp_path, "candidate-alpha", toi="1234.01", tic="123456789")
    (tmp_path / "schemas").mkdir(parents=True, exist_ok=True)
    for name in (
        "candidate.schema.json",
        "provenance.schema.json",
        "claim.schema.json",
        "novelty-audit.schema.json",
    ):
        shutil.copy2(
            "schemas/{0}".format(name), tmp_path / "schemas" / name
        )
    return tmp_path


def _audit(tmp_path):
    report = IsolationReport()
    validate_schemas(tmp_path, report)
    return report


def test_clean_repository_passes_schema_validation(tmp_path):
    report = _audit(_make_repo(tmp_path))
    assert report.ok


def test_invalid_candidate_record_is_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    path = repo / "candidate" / "candidate-alpha" / "candidate.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["lifecycle"]["state"] = "mystery"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    report = _audit(repo)
    assert not report.ok
    assert any(v.rule == "schema-violation" for v in report.violations)


def test_invalid_provenance_sidecar_is_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    raw = repo / "candidate" / "candidate-alpha" / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "lc.fits").write_bytes(b"fits")
    (raw / "lc.provenance.json").write_text(
        json.dumps({"sha256": "not-a-hash"}), encoding="utf-8"
    )

    report = _audit(repo)
    assert not report.ok
    assert any(v.rule == "schema-violation" for v in report.violations)


def test_valid_provenance_sidecar_passes(tmp_path):
    repo = _make_repo(tmp_path)
    raw = repo / "candidate" / "candidate-alpha" / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "lc.fits").write_bytes(b"fits")
    (raw / "lc.provenance.json").write_text(
        json.dumps(
            {
                "source_uri": "https://archive.stsci.edu/example",
                "download_timestamp_utc": "2026-08-04T00:00:00Z",
                "sha256": "a" * 64,
                "fetched_by": "test",
            }
        ),
        encoding="utf-8",
    )

    assert _audit(repo).ok


def test_invalid_claim_is_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    claims = repo / "candidate" / "candidate-alpha" / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    (claims / "bad.json").write_text(
        json.dumps({"parameter": "period_days"}), encoding="utf-8"
    )

    report = _audit(repo)
    assert not report.ok
    assert any(v.rule == "schema-violation" for v in report.violations)


def _valid_novelty_audit():
    return {
        "schema_version": 1,
        "candidate_id": "candidate-alpha",
        "retrieved_at": "2000-01-01T00:00:00Z",
        "freshness": {"expires_at": "2099-01-01T00:00:00Z"},
        "status": "eligible",
        "decision_basis": "The recorded evidence supports the documented eligibility decision.",
        "evidence": [
            {
                "source_uri": "https://example.invalid/novelty-evidence",
                "retrieved_at": "2000-01-01T00:00:00Z",
                "finding": "A source was reviewed using the recorded method.",
                "evidence_sha256": "a" * 64,
            }
        ],
    }


def test_valid_novelty_audit_passes_schema_validation(tmp_path):
    repo = _make_repo(tmp_path)
    path = repo / "candidate" / "candidate-alpha" / "decisions" / "novelty_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_valid_novelty_audit()), encoding="utf-8")

    assert _audit(repo).ok


def test_invalid_novelty_audit_is_flagged(tmp_path):
    repo = _make_repo(tmp_path)
    audit = _valid_novelty_audit()
    audit["evidence"] = []
    path = repo / "candidate" / "candidate-alpha" / "decisions" / "novelty_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit), encoding="utf-8")

    report = _audit(repo)
    assert not report.ok
    assert any(v.rule == "schema-violation" for v in report.violations)


def test_legacy_subtree_sidecars_are_skipped(tmp_path):
    repo = _make_repo(tmp_path)
    legacy = repo / "candidate" / "candidate-alpha" / "legacy-project" / "data"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "old.provenance.json").write_text(
        json.dumps({"legacy": "format"}), encoding="utf-8"
    )

    assert _audit(repo).ok


def test_missing_schema_file_is_reported(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "schemas" / "provenance.schema.json").unlink()

    report = _audit(repo)
    assert not report.ok
    assert any(v.rule == "schema-file-missing" for v in report.violations)
