"""Regression coverage for the candidate-free source debugger."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exonym.__main__ import _build_parser
from exonym import debugger
from exonym import isolation


def test_parser_registers_candidate_free_debug_command() -> None:
    """The public CLI exposes changed/full modes without a candidate argument."""
    parser = _build_parser()

    changed = parser.parse_args(["debug", "--changed", "--format", "json"])
    full = parser.parse_args(["debug", "--full"])

    assert changed.debug_mode == "changed"
    assert changed.debug_format == "json"
    assert full.debug_mode == "full"
    assert not hasattr(full, "candidate_id")


def test_full_source_selection_never_descends_into_candidate(tmp_path: Path) -> None:
    """A full audit selects only protected target-neutral inputs."""
    (tmp_path / "src" / "exonym").mkdir(parents=True)
    (tmp_path / "src" / "exonym" / "module.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "candidate" / "real-target").mkdir(parents=True)
    (tmp_path / "candidate" / "real-target" / "private.py").write_text(
        "raise RuntimeError('must not be scanned')\n", encoding="utf-8"
    )

    selected = debugger._select_source_paths(tmp_path, mode="full", since=None)

    assert [path.as_posix() for path in selected] == ["src/exonym/module.py"]


def test_changed_source_selection_includes_untracked_code_but_excludes_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    """New source files are checked locally without ever selecting candidate files."""
    outputs = {
        ("git", "diff", "--name-only", "--diff-filter=ACMR", "origin/main...HEAD"): "tests/test_old.py\n",
        ("git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"): "src/exonym/edited.py\n",
        ("git", "ls-files", "--others", "--exclude-standard"): "src/exonym/new.py\ncandidate/private.py\n",
    }

    def fake_run(command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=outputs[tuple(command)])

    monkeypatch.setattr(debugger.subprocess, "run", fake_run)

    selected = debugger._select_source_paths(tmp_path, mode="changed", since="origin/main")

    assert [path.as_posix() for path in selected] == [
        "src/exonym/edited.py",
        "src/exonym/new.py",
        "tests/test_old.py",
    ]


def test_successful_debugger_run_preserves_candidate_and_cleans_tmp(
    tmp_path: Path, monkeypatch
) -> None:
    """A successful source run leaves the real candidate tree byte-for-byte unchanged."""
    candidate_file = tmp_path / "candidate" / "real-target" / "candidate.json"
    candidate_file.parent.mkdir(parents=True)
    candidate_file.write_text('{"private": true}\n', encoding="utf-8")
    before = candidate_file.read_bytes()
    monkeypatch.setattr(
        isolation,
        "_alias_tokens",
        lambda *_args, **_kwargs: pytest.fail("debugger must not inspect candidate metadata"),
    )
    monkeypatch.setattr(debugger, "_select_source_paths", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(debugger, "_tool_commands", lambda *_args, **_kwargs: [])

    report = debugger.run_debug(tmp_path, mode="changed")

    assert report.exit_code == 0
    assert candidate_file.read_bytes() == before
    assert not (report.run_dir / "tmp").exists()
    assert (report.run_dir / "report" / "debug-report.json").is_file()
    payload = json.loads((report.run_dir / "report" / "debug-report.json").read_text(encoding="utf-8"))
    assert payload["candidate_access"] == "forbidden"
    assert payload["tmp_preserved"] is False


def test_blocking_static_finding_preserves_sandbox(tmp_path: Path, monkeypatch) -> None:
    """A failed source audit retains only its log-local temporary sandbox."""
    source = tmp_path / "src" / "exonym" / "unsafe.py"
    source.parent.mkdir(parents=True)
    source.write_text("import numpy as np\nnp.load('x.npy', allow_pickle=True)\n", encoding="utf-8")
    monkeypatch.setattr(
        debugger,
        "_select_source_paths",
        lambda *_args, **_kwargs: [Path("src/exonym/unsafe.py")],
    )
    monkeypatch.setattr(debugger, "_tool_commands", lambda *_args, **_kwargs: [])

    report = debugger.run_debug(tmp_path, mode="changed")

    assert report.exit_code == 1
    assert (report.run_dir / "tmp").is_dir()
    static = next(result for result in report.results if result.name == "static-contract")
    assert static.status == "failed"
    assert static.findings[0].rule_id == "EXD003"


def test_sarif_contains_static_source_locations(tmp_path: Path, monkeypatch) -> None:
    """SARIF output carries source locations for editor and CI integration."""
    source = tmp_path / "src" / "exonym" / "unsafe.py"
    source.parent.mkdir(parents=True)
    source.write_text("import os\nos.system('whoami')\n", encoding="utf-8")
    monkeypatch.setattr(
        debugger,
        "_select_source_paths",
        lambda *_args, **_kwargs: [Path("src/exonym/unsafe.py")],
    )
    monkeypatch.setattr(debugger, "_tool_commands", lambda *_args, **_kwargs: [])

    report = debugger.run_debug(tmp_path, mode="changed")
    sarif = json.loads(debugger.format_debug_report(report, "sarif"))

    result = sarif["runs"][0]["results"][0]
    assert result["ruleId"] == "EXD001"
    assert result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "src/exonym/unsafe.py"


def test_debug_environment_redirects_all_temporary_roots(tmp_path: Path) -> None:
    """Subprocesses receive only the log-local temporary paths."""
    environment = debugger._debug_environment(tmp_path)

    assert environment["TMP"] == str(tmp_path)
    assert environment["TEMP"] == str(tmp_path)
    assert environment["TMPDIR"] == str(tmp_path)
    assert environment["MPLCONFIGDIR"] == str(tmp_path)
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["EXONYM_REPO_ROOT"] == ""


def test_in_memory_compile_scan_does_not_write_bytecode(tmp_path: Path) -> None:
    """Compilation validates syntax without writing outside the debugger log tree."""
    source = tmp_path / "src" / "exonym" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    tools_dir = tmp_path / "log" / "debug" / "run" / "tools"
    tools_dir.mkdir(parents=True)

    result = debugger._run_in_memory_compile_scan(tmp_path, tools_dir)

    assert result.status == "passed"
    assert not list((tmp_path / "src").rglob("__pycache__"))


def test_ruff_baseline_warns_for_old_findings_and_blocks_new_ones(
    tmp_path: Path, monkeypatch
) -> None:
    """Ruff debt remains visible, while a newly introduced finding fails the run."""
    source = tmp_path / "src" / "exonym" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("unused = 1\n", encoding="utf-8")
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "debug-baseline.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ruff": [{"path": "src/exonym/module.py", "line": 1, "code": "F401"}],
            }
        ),
        encoding="utf-8",
    )
    tools_dir = tmp_path / "log" / "debug" / "run" / "tools"
    tools_dir.mkdir(parents=True)
    diagnostics = json.dumps(
        [
            {
                "filename": str(source),
                "location": {"row": 1},
                "code": "F401",
                "message": "unused import",
            },
            {
                "filename": str(source),
                "location": {"row": 2},
                "code": "F841",
                "message": "unused local variable",
            },
        ]
    )
    monkeypatch.setattr(debugger.shutil, "which", lambda _command: "ruff")
    monkeypatch.setattr(
        debugger.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=diagnostics),
    )

    result = debugger._run_ruff(
        tmp_path,
        "full",
        [],
        debugger._debug_environment(tmp_path / "tmp"),
        tools_dir,
    )

    assert result.status == "failed"
    assert [finding.severity for finding in result.findings] == ["warning", "blocker"]
    assert "1 new finding(s), 1 baselined" == result.detail
