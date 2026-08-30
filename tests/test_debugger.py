"""Regression coverage for the candidate-free source debugger."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
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


def test_invalid_since_writes_a_failed_report(tmp_path: Path, monkeypatch) -> None:
    """An unusable baseline cannot silently produce an incomplete changed audit."""
    monkeypatch.setattr(
        debugger.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=128, stdout=""),
    )

    report = debugger.run_debug(tmp_path, mode="changed", since="missing-revision")

    assert report.exit_code == 1
    failure = report.run_dir / "tools" / "debugger-internal-error.txt"
    assert "invalid --since revision: missing-revision" in failure.read_text(encoding="utf-8")
    assert json.loads(
        (report.run_dir / "report" / "debug-report.json").read_text(encoding="utf-8")
    )["exit_code"] == 1


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


def test_debugger_infrastructure_failure_keeps_traceback_and_sandbox(
    tmp_path: Path, monkeypatch
) -> None:
    """Unexpected debugger faults preserve reproducible failure evidence."""
    monkeypatch.setattr(
        debugger,
        "_run_static_contract_scan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic fault")),
    )

    report = debugger.run_debug(tmp_path, mode="changed")

    failure = report.run_dir / "tools" / "debugger-internal-error.txt"
    assert report.exit_code == 1
    assert (report.run_dir / "tmp").is_dir()
    assert "RuntimeError: synthetic fault" in failure.read_text(encoding="utf-8")


def test_child_process_interrupt_writes_reports_before_reraising(
    tmp_path: Path, monkeypatch
) -> None:
    """An interrupted diagnostic preserves complete report evidence and the sandbox."""
    monkeypatch.setattr(debugger, "_select_source_paths", lambda *_args, **_kwargs: [])

    def passed(name):
        return debugger.ToolResult(name, "passed", [], 0, "tools/{0}.txt".format(name))

    monkeypatch.setattr(debugger, "_run_static_contract_scan", lambda *_args: passed("static-contract"))
    monkeypatch.setattr(debugger, "_run_debug_source_audit", lambda *_args: passed("source-verify"))
    monkeypatch.setattr(debugger, "_run_in_memory_compile_scan", lambda *_args: passed("compileall"))
    monkeypatch.setattr(debugger, "_run_ruff", lambda *_args: passed("ruff"))
    monkeypatch.setattr(
        debugger,
        "_tool_commands",
        lambda *_args: [("synthetic", ["synthetic"])],
    )
    monkeypatch.setattr(debugger.shutil, "which", lambda _command: "synthetic")
    monkeypatch.setattr(
        debugger.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        debugger.run_debug(tmp_path, mode="changed")

    run_dir = next((tmp_path / "log" / "debug").iterdir())
    payload = json.loads((run_dir / "report" / "debug-report.json").read_text(encoding="utf-8"))
    assert payload["exit_code"] == 1
    assert payload["tmp_preserved"] is True
    assert (run_dir / "tools" / "debugger-interrupted.txt").is_file()
    assert not list((run_dir / "report").glob("*.tmp"))


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
    assert environment["PIP_CACHE_DIR"] == str(tmp_path / "pip-cache")


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


def test_full_debug_requires_semgrep_when_unavailable(tmp_path: Path) -> None:
    """A full audit is incomplete when the required Semgrep scan cannot run."""
    report = debugger.DebugReport(
        root=tmp_path,
        run_dir=tmp_path / "run",
        mode="full",
        changed_paths=[],
        results=[debugger.ToolResult("semgrep", "skipped", [], None, "tools/semgrep.txt")],
        started_at="start",
        completed_at="end",
        tmp_preserved=False,
    )

    assert [result.name for result in report.unavailable_required_tools] == ["semgrep"]
    assert report.exit_code == 2


def test_full_wheel_smoke_builds_installs_and_imports_isolated_wheel(
    tmp_path: Path, monkeypatch
) -> None:
    """The full debugger checks the built wheel, not the editable source tree."""
    tools_dir = tmp_path / "log" / "debug" / "run" / "tools"
    tmp_dir = tmp_path / "log" / "debug" / "run" / "tmp"
    tools_dir.mkdir(parents=True)
    tmp_dir.mkdir()
    (tmp_path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("synthetic\n", encoding="utf-8")
    (tmp_path / "src" / "exonym").mkdir(parents=True)
    (tmp_path / "src" / "exonym" / "__init__.py").write_text("\n", encoding="utf-8")
    commands = []

    def fake_run_tool(name, command, _root, _environment, _tools_dir):
        commands.append((name, command, _root))
        if name == "wheel-build":
            wheel_dir = Path(command[command.index("--wheel-dir") + 1])
            wheel_dir.mkdir()
            (wheel_dir / "exonym-2.0.0-py3-none-any.whl").write_bytes(b"synthetic")
        return debugger.ToolResult(name, "passed", command, 0, "tools/{0}.txt".format(name))

    monkeypatch.setattr(debugger, "_run_tool", fake_run_tool)

    results = debugger._run_wheel_install_smoke(
        tmp_path,
        debugger._debug_environment(tmp_dir),
        tools_dir,
        tmp_dir,
    )

    assert [result.name for result in results] == [
        "wheel-build",
        "wheel-install",
        "wheel-import-smoke",
    ]
    build_command = commands[0][1]
    assert "--no-deps" in build_command
    assert "--no-build-isolation" in build_command
    assert commands[0][2] == tmp_dir / "wheel-source"
    assert commands[1][2] == tmp_dir
    assert commands[2][2] == tmp_dir
    assert commands[1][1][3:7] == ["install", "--no-deps", "--no-cache-dir", "--target"]
    assert commands[1][1][-2] == str(tmp_dir / "wheel-install")
    assert commands[2][1][1] == "-I"
    assert repr(str(tmp_dir / "wheel-install")) in commands[2][1][-1]


def test_changed_test_file_is_its_own_focused_regression(tmp_path: Path) -> None:
    """A changed test must run even when no source module maps to it."""
    test_path = tmp_path / "tests" / "test_synthetic.py"
    test_path.parent.mkdir()
    test_path.write_text("def test_synthetic():\n    assert True\n", encoding="utf-8")

    selected = debugger._select_tests(tmp_path, "changed", [Path("tests/test_synthetic.py")])

    assert selected == ["tests/test_synthetic.py"]


def test_tool_start_oserror_is_a_blocking_debugger_failure(tmp_path: Path, monkeypatch) -> None:
    """A discovered executable that cannot start must fail the debugger exit code."""
    tools_dir = tmp_path / "log" / "debug" / "run" / "tools"
    tools_dir.mkdir(parents=True)
    monkeypatch.setattr(debugger.shutil, "which", lambda _command: "synthetic")
    monkeypatch.setattr(
        debugger.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic startup failure")),
    )

    result = debugger._run_tool(
        "synthetic",
        ["synthetic"],
        tmp_path,
        debugger._debug_environment(tmp_path / "tmp"),
        tools_dir,
    )
    report = debugger.DebugReport(
        root=tmp_path,
        run_dir=tools_dir.parent,
        mode="changed",
        changed_paths=[],
        results=[result],
        started_at="start",
        completed_at="end",
        tmp_preserved=True,
    )

    assert result.status == "error"
    assert report.exit_code == 1


def test_process_isolated_debugger_runs_do_not_share_state(tmp_path: Path) -> None:
    """Two fresh interpreter runs create independent reports without external tools."""
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from exonym import debugger

root = Path(sys.argv[2])
def passed(name):
    return debugger.ToolResult(name, "passed", [], 0, "tools/{}.txt".format(name))
debugger._select_source_paths = lambda *_args, **_kwargs: []
debugger._run_static_contract_scan = lambda *_args: passed("static-contract")
debugger._run_debug_source_audit = lambda *_args: passed("source-verify")
debugger._run_in_memory_compile_scan = lambda *_args: passed("compileall")
debugger._run_ruff = lambda *_args: passed("ruff")
debugger._tool_commands = lambda *_args: []
report = debugger.run_debug(root, mode="changed")
print(json.dumps({"run_dir": str(report.run_dir), "exit_code": report.exit_code}))
"""

    runs = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script, str(source_root), str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        runs.append(json.loads(completed.stdout))

    assert [run["exit_code"] for run in runs] == [0, 0]
    assert runs[0]["run_dir"] != runs[1]["run_dir"]
    for run in runs:
        run_dir = Path(run["run_dir"])
        assert (run_dir / "report" / "debug-report.json").is_file()
        assert not (run_dir / "tmp").exists()
