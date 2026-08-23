import json
from pathlib import Path

import pytest

import exonym.isolation as isolation
from exonym.isolation import check_neutral_repository, check_repository, format_report
from exonym.workspace import new_candidate_metadata


def _write(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _create_candidate(root, candidate_id, toi=None, tic=None):
    """Create only the metadata needed by the isolation unit tests."""
    metadata = new_candidate_metadata(
        candidate_id,
        toi=toi,
        tic=tic,
        mission="tess",
    )
    path = root / "candidate" / candidate_id
    path.mkdir(parents=True)
    _write(path, "candidate.json", json.dumps(metadata, indent=2))
    return path


def _make_repo(tmp_path, with_candidate=True):
    if with_candidate:
        _create_candidate(tmp_path, "candidate-alpha", toi="1234.01", tic="123456789")
        _write(
            tmp_path,
            "candidate/candidate-alpha/docs/note.md",
            "# alpha intake note\n",
        )
    return tmp_path


def test_clean_repository_passes(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo, "src/example.py", "def parse_sector(label):\n    return label\n")
    _write(repo, "README.md", "# Factory\nNeutral documentation.\n")

    report = check_repository(repo)
    assert report.ok, format_report(report)


def test_neutral_audit_does_not_traverse_candidate_workspaces(tmp_path, monkeypatch):
    # A default shared-code audit must remain fast even when candidate data is large.
    repo = _make_repo(tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("candidate workspace was traversed")

    monkeypatch.setattr(isolation, "_alias_tokens", fail_if_called)
    monkeypatch.setattr(isolation, "_scan_candidate_reparse_points", fail_if_called)

    assert check_neutral_repository(repo).ok


def test_catalog_identifier_outside_candidate_fails(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo, "docs/note.md", "Plan: study TOI-99999.01 next.\n")

    report = check_repository(repo)
    assert not report.ok
    assert any(v.rule == "target-id-in-neutral-zone" for v in report.violations)


def test_root_level_files_are_operator_workspace(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo, "notes.md", "Plan: study TOI-99999.01 next.\n")
    _write(repo, "figure.pdf", "%PDF-1.7 simulated binary payload\n")
    _write(
        repo,
        "docs/note.md",
        "# Neutral\nSubdirectory documents remain fully audited.\n",
    )

    report = check_repository(repo)
    assert report.ok, format_report(report)


def test_root_level_symlink_still_rejected(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo, "docs/target.txt", "neutral")
    link = repo / "link.txt"
    try:
        link.symlink_to(repo / "docs" / "target.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    report = check_repository(repo)
    assert not report.ok
    assert any(v.rule == "symlink-or-reparse-point" for v in report.violations)


def test_catalog_identifier_inside_candidate_passes(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo, "candidate/candidate-alpha/docs/note.md", "TOI-1234.01 target notes.\n")

    report = check_repository(repo)
    assert report.ok, format_report(report)


def test_registered_alias_leak_outside_candidate_fails(tmp_path):
    repo = _make_repo(tmp_path)
    _create_candidate(tmp_path, "candidate-beta")
    metadata_path = tmp_path / "candidate" / "candidate-beta" / "candidate.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["identifiers"]["aliases"] = ["HD 99999"]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write(repo, "docs/note.md", "Reference: HD 99999 was checked.\n")

    report = check_repository(repo)
    assert not report.ok
    assert any(v.rule == "registered-alias-leak" for v in report.violations)


def test_hardcoded_sectors_in_shared_source_produce_warnings(tmp_path):
    repo = _make_repo(tmp_path)
    _write(
        repo,
        "src/analysis.py",
        "SECTORS = [37, 63, 64]\nPERIOD_DAYS = 9.2224171\n",
    )

    report = check_repository(repo)
    # AST scan downgraded to warnings: report.ok must be True.
    assert report.ok
    assert any(v.rule == "hardcoded-target-literal" for v in report.violations)
    assert all(v.severity == "warning" for v in report.violations if v.rule == "hardcoded-target-literal")


def test_generic_source_passes(tmp_path):
    repo = _make_repo(tmp_path)
    _write(
        repo,
        "src/analysis.py",
        "def phase_hours(time, period_days, epoch_btjd):\n"
        "    return time % period_days\n"
        "def parse_sector(label):\n"
        "    import re\n"
        "    return re.search(r'Sector\\s+(\\d+)', str(label))\n",
    )

    report = check_repository(repo)
    assert report.ok, format_report(report)


def test_ephemeris_keyword_literal_warning_but_trivial_passes(tmp_path):
    repo = _make_repo(tmp_path)
    _write(
        repo,
        "src/analysis.py",
        "fit(period_days=9.2224171, epoch_btjd=2281.123)\n",
    )
    report = check_repository(repo)
    # AST scan now produces warnings, not errors.
    assert report.ok
    assert any(v.rule == "hardcoded-ephemeris-keyword" for v in report.violations)
    assert all(v.severity == "warning" for v in report.violations if v.rule == "hardcoded-ephemeris-keyword")

    _write(repo, "src/analysis.py", "fit(period_days=1.0, epoch_btjd=0.0)\n")
    report = check_repository(repo)
    assert report.ok


def test_research_payload_outside_candidate_fails(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo, "data/raw/lightcurve.fits", "binary")
    _write(repo, "results/outputs.csv", "a,b\n1,2\n")
    _write(repo, "docs/notes.ipynb", "{}")

    report = check_repository(repo)
    assert not report.ok
    assert any(v.rule == "research-payload-outside-candidate" for v in report.violations)


def test_research_payload_inside_candidate_passes(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo, "candidate/candidate-alpha/data/raw/lc.fits", "binary")
    _write(repo, "candidate/candidate-alpha/outputs/out.csv", "a,b\n")

    report = check_repository(repo)
    assert report.ok, format_report(report)


def test_only_the_exact_paper_template_is_allowed_outside_candidate(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo, "templates/paper/paper_template.tex", "% target-neutral template\n")
    _write(repo, "templates/paper/supplement.tex", "% forbidden payload\n")

    report = check_repository(repo)

    assert any(
        violation.path.endswith("templates/paper/supplement.tex")
        and violation.rule == "research-payload-outside-candidate"
        for violation in report.violations
    )


def test_top_level_archive_is_forbidden(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo, "archive/old-project/README.md", "frozen evidence\n")

    report = check_repository(repo)
    assert not report.ok
    assert any(v.rule == "top-level-archive-forbidden" for v in report.violations)


def test_top_level_data_is_forbidden(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo, "data/raw/lc.fits", "binary")

    report = check_repository(repo)
    assert not report.ok
    assert any(v.rule == "top-level-data-forbidden" for v in report.violations)


def test_exception_registry_suppresses_matching_violation(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo, "docs/note.md", "Plan: study TOI-99999.01 next.\n")

    violations_before = check_repository(repo).violations
    assert violations_before

    target = violations_before[0]
    exception = {
        "path": Path(target.path).relative_to(repo.resolve()).as_posix(),
        "line": target.line,
        "rule": target.rule,
        "reason": "test fixture",
        "expires": "2099-01-01",
    }
    _write(
        repo,
        "policy/isolation-exceptions.json",
        json.dumps({"entries": [exception]}, indent=2),
    )

    assert check_repository(repo).ok


def test_unlisted_neutral_directory_is_scanned(tmp_path):
    # Arrange: `scratch/` is not a project source directory allowlist entry.
    repo = _make_repo(tmp_path)
    _write(repo, "scratch/intake.md", "Plan: study TOI-99999.01 next.\n")
    _write(repo, "scratch/measurement.csv", "time,flux\n")

    # Act
    report = check_repository(repo)

    # Assert
    rules = {violation.rule for violation in report.violations}
    assert "target-id-in-neutral-zone" in rules
    assert "research-payload-outside-candidate" in rules


def test_expired_exception_does_not_suppress_a_violation(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    _write(repo, "docs/note.md", "Plan: study TOI-99999.01 next.\n")
    target = check_repository(repo).violations[0]
    exception = {
        "path": Path(target.path).relative_to(repo.resolve()).as_posix(),
        "line": target.line,
        "rule": target.rule,
        "reason": "expired test fixture",
        "expires": "2000-01-01",
    }
    _write(repo, "policy/isolation-exceptions.json", json.dumps({"entries": [exception]}))

    # Act
    report = check_repository(repo)

    # Assert
    rules = {violation.rule for violation in report.violations}
    assert "expired-isolation-exception" in rules
    assert target.rule in rules


@pytest.mark.parametrize(
    "entry_updates",
    [
        {"reason": ""},
        {"expires": "2099/01/01"},
        {"line": True},
    ],
)
def test_malformed_exception_does_not_suppress_a_violation(tmp_path, entry_updates):
    # Arrange
    repo = _make_repo(tmp_path)
    _write(repo, "docs/note.md", "Plan: study TOI-99999.01 next.\n")
    target = check_repository(repo).violations[0]
    exception = {
        "path": Path(target.path).relative_to(repo.resolve()).as_posix(),
        "line": target.line,
        "rule": target.rule,
        "reason": "test fixture",
        "expires": "2099-01-01",
    }
    exception.update(entry_updates)
    _write(repo, "policy/isolation-exceptions.json", json.dumps({"entries": [exception]}))

    # Act
    report = check_repository(repo)

    # Assert
    rules = {violation.rule for violation in report.violations}
    assert "invalid-isolation-exception" in rules
    assert target.rule in rules


def test_symlink_or_reparse_point_rejected(tmp_path):
    repo = _make_repo(tmp_path)
    _write(repo, "src/target.txt", "neutral")
    link = repo / "src/link.txt"
    try:
        link.symlink_to(repo / "src" / "target.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    report = check_repository(repo)
    assert not report.ok
    assert any(v.rule == "symlink-or-reparse-point" for v in report.violations)


def test_directory_symlink_or_reparse_point_rejected(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    _write(repo, "scratch/plain/note.md", "neutral\n")
    link = repo / "scratch" / "linked"
    try:
        link.symlink_to(repo / "scratch" / "plain", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable on this platform")

    # Act
    report = check_repository(repo)

    # Assert
    assert any(v.rule == "symlink-or-reparse-point" for v in report.violations)


def test_detected_directory_reparse_point_is_rejected(tmp_path, monkeypatch):
    # Arrange
    repo = _make_repo(tmp_path)
    reparse_directory = repo / "scratch" / "linked-directory"
    reparse_directory.mkdir(parents=True)
    monkeypatch.setattr(
        isolation,
        "is_reparse_point",
        lambda path: Path(path) == reparse_directory,
    )

    # Act
    report = check_repository(repo)

    # Assert
    assert any(v.rule == "symlink-or-reparse-point" for v in report.violations)


def test_candidate_directory_symlink_or_reparse_point_rejected(tmp_path):
    # Arrange
    repo = _make_repo(tmp_path)
    source = repo / "candidate" / "candidate-alpha" / "docs"
    link = repo / "candidate" / "candidate-alpha" / "linked-docs"
    try:
        link.symlink_to(source, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable on this platform")

    # Act
    report = check_repository(repo)

    # Assert
    assert any(v.rule == "symlink-or-reparse-point" for v in report.violations)


def test_survey_collection_reparse_point_is_rejected(tmp_path, monkeypatch):
    # Arrange
    repo = _make_repo(tmp_path)
    survey_root = repo / "candidate" / "_surveys"
    survey_root.mkdir()
    monkeypatch.setattr(
        isolation,
        "is_reparse_point",
        lambda path: Path(path) == survey_root,
    )

    # Act
    report = check_repository(repo)

    # Assert
    assert any(
        violation.rule == "symlink-or-reparse-point" and Path(violation.path) == survey_root
        for violation in report.violations
    )


def test_self_check_of_actual_repository():
    import os

    root = os.environ.get("EXONYM_REPO_ROOT")
    if not root:
        pytest.skip("EXONYM_REPO_ROOT not set")
    report = check_repository(root)
    assert report.ok, format_report(report)
