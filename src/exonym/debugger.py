"""Candidate-free development diagnostics for EXONYM source changes.

``exonym debug`` audits only target-neutral source, schemas, templates, and
synthetic tests.  It deliberately never traverses or passes the repository's
``candidate/`` directory to a subprocess.  All mutable debugger state lives
under the ignored ``log/debug/<run-id>/`` directory.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import signal
import shutil
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .isolation import format_report, run_debug_source_audit


_SOURCE_ROOTS = ("src", "schemas", "templates", "tests", "policy")
_SOURCE_FILES = ("pyproject.toml", ".pre-commit-config.yaml")
_SCIENTIFIC_MODULES = {
    "asteroseismology.py",
    "constants.py",
    "detrending.py",
    "phasecurve.py",
    "radial_velocity.py",
    "sed.py",
    "transit_fit.py",
    "ttv.py",
}
_NESTED_TEST_MAP = {
    "src/exonym/vetting/tricera_parse.py": (
        "tests/test_signal_suffixes.py",
        "tests/test_vetting.py",
    ),
}
_SELF_CHECK_EXPRESSION = "not test_self_check_of_actual_repository"
_RUFF_BASELINE_PATH = Path("policy") / "debug-baseline.json"


@dataclass
class Finding:
    """One source-level diagnostic finding."""

    rule_id: str
    severity: str
    path: str
    line: int
    message: str
    remediation: str

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-safe report representation."""
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "path": self.path,
            "line": self.line,
            "message": self.message,
            "remediation": self.remediation,
        }


@dataclass
class ToolResult:
    """One command or deterministic source scan in a debugger run."""

    name: str
    status: str
    command: List[str]
    returncode: Optional[int]
    output_path: str
    detail: str = ""
    findings: List[Finding] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-safe report representation."""
        return {
            "name": self.name,
            "status": self.status,
            "command": self.command,
            "returncode": self.returncode,
            "output_path": self.output_path,
            "detail": self.detail,
            "findings": [finding.as_dict() for finding in self.findings],
        }


@dataclass
class DebugReport:
    """Completed candidate-free source debugger report."""

    root: Path
    run_dir: Path
    mode: str
    changed_paths: List[str]
    results: List[ToolResult]
    started_at: str
    completed_at: str
    tmp_preserved: bool

    @property
    def blockers(self) -> List[ToolResult]:
        """Return failed checks and source scans."""
        return [result for result in self.results if result.status in {"failed", "error"}]

    @property
    def unavailable_required_tools(self) -> List[ToolResult]:
        """Return unavailable tools that make a full audit incomplete."""
        if self.mode != "full":
            return []
        return [
            result
            for result in self.results
            if result.status == "skipped"
            and result.name in {"ruff", "bandit", "pytest-cov", "semgrep"}
        ]

    @property
    def exit_code(self) -> int:
        """Return the documented debugger exit code."""
        if self.blockers:
            return 1
        if self.unavailable_required_tools:
            return 2
        return 0

    def as_dict(self) -> Dict[str, object]:
        """Return the complete versioned JSON report."""
        return {
            "schema_version": 1,
            "kind": "exonym-source-debug-report",
            "mode": self.mode,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "repository_root": str(self.root),
            "run_directory": str(self.run_dir),
            "candidate_access": "forbidden",
            "changed_paths": self.changed_paths,
            "tmp_preserved": self.tmp_preserved,
            "exit_code": self.exit_code,
            "results": [result.as_dict() for result in self.results],
        }


def add_debug_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the candidate-free source debugger."""
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--changed",
        action="store_const",
        const="changed",
        dest="debug_mode",
        help="Audit changed target-neutral files and their focused synthetic tests (default).",
    )
    mode.add_argument(
        "--full",
        action="store_const",
        const="full",
        dest="debug_mode",
        help="Run the complete target-neutral audit and synthetic test suite.",
    )
    parser.set_defaults(debug_mode="changed")
    parser.add_argument(
        "--since",
        default=None,
        help="Git revision used as the changed-file baseline.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "sarif"),
        default="text",
        dest="debug_format",
        help="Console report format; all report files are written regardless.",
    )


def run_debug(
    root: Path,
    *,
    mode: str = "changed",
    since: Optional[str] = None,
) -> DebugReport:
    """Run candidate-free source diagnostics and write a report below ``log/``.

    The function intentionally does not inspect ``root / 'candidate'``.  Its
    subprocess environment redirects temporary files to the run-local sandbox,
    and its pytest invocation excludes the one test that audits the live
    repository candidate tree.
    """
    if mode not in ("changed", "full"):
        raise ValueError("debug mode must be 'changed' or 'full'")

    root = Path(root).resolve()
    run_dir = _create_run_dir(root)
    report_dir = run_dir / "report"
    tools_dir = run_dir / "tools"
    tmp_dir = run_dir / "tmp"
    for directory in (report_dir, tools_dir, tmp_dir):
        directory.mkdir(parents=True, exist_ok=False)

    started_at = _utc_now()
    changed_paths: List[Path] = []
    environment = _debug_environment(tmp_dir)
    results: List[ToolResult] = []
    preserve_tmp = True
    interrupted: Optional[KeyboardInterrupt] = None

    try:
        changed_paths = _select_source_paths(root, mode=mode, since=since)
        results.append(_run_static_contract_scan(root, changed_paths, tools_dir))
        source_audit = _run_debug_source_audit(root, tools_dir)
        results.append(source_audit)
        if source_audit.status == "passed":
            results.append(_run_in_memory_compile_scan(root, tools_dir))
            results.append(_run_ruff(root, mode, changed_paths, environment, tools_dir))
            for name, command in _tool_commands(root, mode, changed_paths, tmp_dir):
                results.append(_run_tool(name, command, root, environment, tools_dir))
            if mode == "full":
                results.extend(_run_wheel_install_smoke(root, environment, tools_dir, tmp_dir))
        preserve_tmp = any(result.status in ("failed", "error") for result in results)
    except KeyboardInterrupt as exc:
        _record_debugger_failure(
            results,
            run_dir,
            tools_dir,
            "debugger-interrupted.txt",
            "Debugger interrupted; report and temporary sandbox retained.",
        )
        preserve_tmp = True
        interrupted = exc
    except Exception as exc:  # exonym: fail-closed - preserve sandbox on debugger faults.
        _record_debugger_failure(
            results,
            run_dir,
            tools_dir,
            "debugger-internal-error.txt",
            "Unexpected debugger infrastructure failure: {0}".format(type(exc).__name__),
        )
        preserve_tmp = True

    completed_at = _utc_now()
    report = DebugReport(
        root=root,
        run_dir=run_dir,
        mode=mode,
        changed_paths=[path.as_posix() for path in changed_paths],
        results=results,
        started_at=started_at,
        completed_at=completed_at,
        tmp_preserved=preserve_tmp,
    )
    _write_reports(report)
    if not preserve_tmp:
        shutil.rmtree(tmp_dir)
    if interrupted is not None:
        raise interrupted
    return report


def format_debug_report(report: DebugReport, output_format: str) -> str:
    """Render a completed debugger report for the selected console format."""
    if output_format == "json":
        return json.dumps(report.as_dict(), indent=2, sort_keys=True)
    if output_format == "sarif":
        return json.dumps(_sarif(report), indent=2, sort_keys=True)
    status = "PASS" if report.exit_code == 0 else "FAIL" if report.exit_code == 1 else "INCOMPLETE"
    lines = ["DEBUG: {0} ({1})".format(status, report.mode)]
    lines.append("run: {0}".format(report.run_dir))
    lines.append("candidate access: forbidden")
    for result in report.results:
        detail = " ({0})".format(result.detail) if result.detail else ""
        lines.append("  [{0}] {1}{2}".format(result.status.upper(), result.name, detail))
        for finding in result.findings:
            lines.append(
                "    {0} {1}:{2} {3}".format(
                    finding.severity.upper(), finding.path, finding.line, finding.message
                )
            )
    return "\n".join(lines)


def _create_run_dir(root: Path) -> Path:
    """Create a collision-resistant run directory below the ignored log tree."""
    run_id = "{0}-{1}".format(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), uuid.uuid4().hex[:10]
    )
    return root / "log" / "debug" / run_id


def _utc_now() -> str:
    """Return a UTC timestamp with an explicit timezone offset."""
    return datetime.now(timezone.utc).isoformat()


def _record_debugger_failure(
    results: List[ToolResult],
    run_dir: Path,
    tools_dir: Path,
    filename: str,
    detail: str,
) -> None:
    """Retain a traceback for a debugger failure before emitting its report."""
    failure_path = tools_dir / filename
    failure_path.write_text(traceback.format_exc(), encoding="utf-8")
    results.append(
        ToolResult(
            name="debugger",
            status="failed",
            command=[],
            returncode=None,
            output_path=_relative(run_dir, failure_path),
            detail=detail,
        )
    )


def _relative(root: Path, path: Path) -> str:
    """Return a portable path relative to one debugger run."""
    return path.relative_to(root).as_posix()


def _is_source_path(relative: Path) -> bool:
    """Return whether a path is target-neutral debugger input."""
    if not relative.parts or relative.parts[0] == "candidate":
        return False
    return relative.name in _SOURCE_FILES or relative.parts[0] in _SOURCE_ROOTS


def _normalize_since(since: Optional[str]) -> Optional[str]:
    """Normalize and discard empty, whitespace, or Git null-SHA revisions."""
    if since is None:
        return None
    cleaned = str(since).strip()
    if not cleaned or (len(cleaned) in (40, 64) and set(cleaned) == {"0"}):
        return None
    return cleaned


def _select_source_paths(root: Path, *, mode: str, since: Optional[str]) -> List[Path]:
    """Select target-neutral paths without traversing candidate workspaces."""
    since = _normalize_since(since)
    if mode == "full":
        paths: List[Path] = []
        for name in _SOURCE_ROOTS:
            path = root / name
            if path.exists():
                paths.extend(item.relative_to(root) for item in path.rglob("*") if item.is_file())
        paths.extend(Path(name) for name in _SOURCE_FILES if (root / name).is_file())
        return sorted(set(paths))

    commands = [
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            "{0}...HEAD".format(since) if since else "HEAD",
        ],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ]
    # A pre-commit run may have uncommitted changes on top of a branch diff.
    # Include them even when ``--since`` selects a PR merge-base.
    if since:
        commands.insert(1, ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"])

    paths = set()
    for index, command in enumerate(commands):
        try:
            completed = subprocess.run(
                command,
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except OSError as exc:
            if since and index == 0:
                raise ValueError("could not resolve --since revision: {0}".format(since)) from exc
            continue
        if since and index == 0 and completed.returncode != 0:
            raise ValueError("invalid --since revision: {0}".format(since))
        if completed.returncode == 0:
            paths.update(
                Path(line.strip()) for line in completed.stdout.splitlines() if line.strip()
            )
    return sorted(path for path in paths if _is_source_path(path))


def _debug_environment(tmp_dir: Path) -> Dict[str, str]:
    """Build an isolated environment for diagnostic subprocesses."""
    environment = dict(os.environ)
    for key in ("TMP", "TEMP", "TMPDIR", "MPLCONFIGDIR"):
        environment[key] = str(tmp_dir)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["EXONYM_REPO_ROOT"] = ""
    environment["PIP_CACHE_DIR"] = str(tmp_dir / "pip-cache")
    return environment


def _tool_commands(
    root: Path,
    mode: str,
    changed_paths: Sequence[Path],
    tmp_dir: Path,
) -> Iterable[Tuple[str, List[str]]]:
    """Yield source-only commands for one debugger mode."""
    yield "bandit", ["bandit", "-r", "src", "-lll"]
    yield "semgrep", ["semgrep", "scan", "--config", "p/python", "--severity", "ERROR", "--error", "--metrics=off", "src"]

    tests = _select_tests(root, mode, changed_paths)
    if tests:
        command = ["pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(tmp_dir / "pytest")]
        command.extend(tests)
        command.extend(["-k", _SELF_CHECK_EXPRESSION])
        if mode == "full":
            if _pytest_cov_available():
                command.extend(["--cov=src", "--cov-report=term-missing:skip-covered"])
                yield "pytest-cov", command
            else:
                yield "pytest-cov", []
        else:
            yield "pytest", command
    yield "import-smoke", [sys.executable, "-c", "import exonym; print(exonym.__version__)"]


def _run_wheel_install_smoke(
    root: Path,
    environment: Dict[str, str],
    tools_dir: Path,
    tmp_dir: Path,
) -> List[ToolResult]:
    """Build, install, and import a wheel without using the source checkout."""
    wheel_dir = tmp_dir / "wheel"
    install_dir = tmp_dir / "wheel-install"
    source_dir = tmp_dir / "wheel-source"
    source_dir.mkdir()
    for filename in ("pyproject.toml", "LICENSE"):
        shutil.copy2(root / filename, source_dir / filename)
    shutil.copytree(root / "src", source_dir / "src")
    build = _run_tool(
        "wheel-build",
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-cache-dir",
            "--wheel-dir",
            str(wheel_dir),
            ".",
        ],
        source_dir,
        environment,
        tools_dir,
    )
    results = [build]
    if build.status != "passed":
        return results

    wheels = sorted(wheel_dir.glob("exonym-*.whl"))
    if len(wheels) != 1:
        output_path = tools_dir / "wheel-install.txt"
        output_path.write_text("expected exactly one Exonym wheel\n", encoding="utf-8")
        results.append(
            ToolResult(
                name="wheel-install",
                status="failed",
                command=[],
                returncode=None,
                output_path=_relative(tools_dir.parent, output_path),
                detail="wheel build did not produce exactly one Exonym wheel",
            )
        )
        return results

    install = _run_tool(
        "wheel-install",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-cache-dir",
            "--target",
            str(install_dir),
            str(wheels[0]),
        ],
        tmp_dir,
        environment,
        tools_dir,
    )
    results.append(install)
    if install.status != "passed":
        return results

    smoke_code = (
        "import sys\n"
        "sys.path.insert(0, {0!r})\n"
        "import exonym\n"
        "from exonym.resources import _bundled_directory\n"
        "assert exonym.__file__.startswith({0!r})\n"
        "assert _bundled_directory('schemas').is_dir()\n"
        "print(exonym.__version__)\n"
    ).format(str(install_dir))
    results.append(
        _run_tool(
            "wheel-import-smoke",
            [sys.executable, "-I", "-c", smoke_code],
            tmp_dir,
            environment,
            tools_dir,
        )
    )
    return results


def _run_debug_source_audit(root: Path, tools_dir: Path) -> ToolResult:
    """Run the source verifier variant that is forbidden from reading candidate/."""
    output_path = tools_dir / "source-verify.txt"
    report = run_debug_source_audit(root)
    output_path.write_text(format_report(report) + "\n", encoding="utf-8")
    return ToolResult(
        name="source-verify",
        status="passed" if report.ok else "failed",
        command=["internal", "candidate-free-source-audit"],
        returncode=0 if report.ok else 1,
        output_path=_relative(tools_dir.parent, output_path),
        detail="candidate access forbidden",
    )


def _run_ruff(
    root: Path,
    mode: str,
    changed_paths: Sequence[Path],
    environment: Dict[str, str],
    tools_dir: Path,
) -> ToolResult:
    """Run Ruff and fail only findings absent from the reviewed baseline."""
    output_path = tools_dir / "ruff.json"
    python_paths = [path.as_posix() for path in changed_paths if path.suffix == ".py"]
    paths = python_paths if mode == "changed" else ["src", "tests"]
    if not paths:
        output_path.write_text("[]\n", encoding="utf-8")
        return ToolResult(
            name="ruff",
            status="skipped",
            command=[],
            returncode=None,
            output_path=_relative(tools_dir.parent, output_path),
            detail="no changed Python files",
        )
    if shutil.which("ruff") is None:
        output_path.write_text("command not found: ruff\n", encoding="utf-8")
        return ToolResult(
            name="ruff",
            status="skipped",
            command=["ruff", "check", *paths],
            returncode=None,
            output_path=_relative(tools_dir.parent, output_path),
            detail="command is not installed",
        )
    command = ["ruff", "check", "--output-format", "json", *paths]
    completed = subprocess.run(
        command,
        cwd=str(root),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output_path.write_text(completed.stdout or "", encoding="utf-8")
    try:
        diagnostics = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        return ToolResult(
            name="ruff",
            status="failed",
            command=command,
            returncode=completed.returncode,
            output_path=_relative(tools_dir.parent, output_path),
            detail="Ruff returned non-JSON diagnostic output",
        )
    baseline = _ruff_baseline(root)
    findings: List[Finding] = []
    new_count = 0
    baseline_count = 0
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        filename = diagnostic.get("filename")
        location = diagnostic.get("location")
        code = diagnostic.get("code")
        if not isinstance(filename, str) or not isinstance(location, dict) or not isinstance(code, str):
            continue
        line = location.get("row")
        if not isinstance(line, int):
            continue
        try:
            path = Path(filename).resolve().relative_to(root).as_posix()
        except ValueError:
            path = Path(filename).as_posix()
        key = (path, line, code)
        is_baselined = key in baseline
        baseline_count += int(is_baselined)
        new_count += int(not is_baselined)
        findings.append(
            Finding(
                "RUFF-{0}".format(code),
                "warning" if is_baselined else "blocker",
                path,
                line,
                str(diagnostic.get("message", code)),
                "Remove the finding or add one reviewed, line-specific baseline entry.",
            )
        )
    failed = completed.returncode not in (0, 1) or new_count > 0
    return ToolResult(
        name="ruff",
        status="failed" if failed else "passed",
        command=command,
        returncode=completed.returncode,
        output_path=_relative(tools_dir.parent, output_path),
        detail="{0} new finding(s), {1} baselined".format(new_count, baseline_count),
        findings=findings,
    )


def _ruff_baseline(root: Path) -> set:
    """Load reviewed Ruff findings that predate the debugger rollout."""
    path = root / _RUFF_BASELINE_PATH
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, dict) or not isinstance(payload.get("ruff"), list):
        return set()
    keys = set()
    for entry in payload["ruff"]:
        if not isinstance(entry, dict):
            continue
        path_value = entry.get("path")
        line = entry.get("line")
        code = entry.get("code")
        if isinstance(path_value, str) and isinstance(line, int) and isinstance(code, str):
            keys.add((path_value, line, code))
    return keys


def _run_in_memory_compile_scan(root: Path, tools_dir: Path) -> ToolResult:
    """Compile source and tests in memory without emitting ``__pycache__`` files."""
    failures: List[Finding] = []
    for directory_name in ("src", "tests"):
        directory = root / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(root)
            try:
                compile(path.read_text(encoding="utf-8-sig"), relative.as_posix(), "exec")
            except (OSError, SyntaxError, UnicodeError) as exc:
                failures.append(
                    Finding(
                        "EXD010",
                        "blocker",
                        relative.as_posix(),
                        int(getattr(exc, "lineno", 1) or 1),
                        "In-memory compile failed: {0}".format(exc),
                        "Fix the syntax or encoding before rerunning exonym debug.",
                    )
                )
    output_path = tools_dir / "compileall.json"
    output_path.write_text(
        json.dumps([finding.as_dict() for finding in failures], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ToolResult(
        name="compileall",
        status="failed" if failures else "passed",
        command=["internal", "in-memory-compile"],
        returncode=1 if failures else 0,
        output_path=_relative(tools_dir.parent, output_path),
        detail="{0} failure(s); no bytecode written".format(len(failures)),
        findings=failures,
    )


def _select_tests(root: Path, mode: str, changed_paths: Sequence[Path]) -> List[str]:
    """Map changed neutral files to synthetic regression tests."""
    if mode == "full":
        return ["tests"]

    tests = set()
    for path in changed_paths:
        if path.parts and path.parts[0] == "tests" and path.suffix == ".py":
            tests.add(path.as_posix())
        if path.parts[:2] == ("src", "exonym") and path.suffix == ".py":
            if path.name == "__main__.py":
                tests.add("tests/test_cli.py")
            else:
                candidate = "tests/test_{0}".format(path.name)
                if (root / candidate).is_file():
                    tests.add(candidate)
                else:
                    nested_tests = _NESTED_TEST_MAP.get(path.as_posix())
                    if nested_tests is not None:
                        tests.update(nested_tests)
                    else:
                        package_test = "tests/test_{0}.py".format(path.parent.name)
                        tests.add(package_test if (root / package_test).is_file() else "tests")
            if path.name in _SCIENTIFIC_MODULES:
                tests.update(
                    {
                        "tests/test_astrophysical_benchmarks.py",
                        "tests/test_astrophysical_input_guards.py",
                    }
                )
        if path.parts and path.parts[0] in {"schemas", "templates"}:
            tests.update({"tests/test_resources.py", "tests/test_schemas.py"})
    return sorted(path for path in tests if (root / path).is_file())


def _pytest_cov_available() -> bool:
    """Return whether the pytest-cov plugin is importable without importing it."""
    try:
        import importlib.util

        return importlib.util.find_spec("pytest_cov") is not None
    except (ImportError, AttributeError):
        return False


def _run_tool(
    name: str,
    command: List[str],
    root: Path,
    environment: Dict[str, str],
    tools_dir: Path,
) -> ToolResult:
    """Run one command and capture stdout and stderr under the run directory."""
    output_path = tools_dir / "{0}.txt".format(name)
    if not command:
        output_path.write_text("pytest-cov is not installed\n", encoding="utf-8")
        return ToolResult(
            name=name,
            status="skipped",
            command=[],
            returncode=None,
            output_path=_relative(tools_dir.parent, output_path),
            detail="pytest-cov is not installed",
        )
    if shutil.which(command[0]) is None and command[0] != sys.executable:
        output_path.write_text("command not found: {0}\n".format(command[0]), encoding="utf-8")
        return ToolResult(
            name=name,
            status="skipped",
            command=command,
            returncode=None,
            output_path=_relative(tools_dir.parent, output_path),
            detail="command is not installed",
        )
    popen_kwargs = {
        "cwd": str(root),
        "env": environment,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **popen_kwargs)
        stdout, _ = process.communicate()
    except KeyboardInterrupt:
        if "process" in locals():
            _terminate_process_tree(process)
        raise
    except OSError as exc:
        output_path.write_text("{0}: {1}\n".format(type(exc).__name__, exc), encoding="utf-8")
        return ToolResult(
            name=name,
            status="error",
            command=command,
            returncode=None,
            output_path=_relative(tools_dir.parent, output_path),
            detail="could not start command",
        )
    output_path.write_text(stdout or "", encoding="utf-8")
    return ToolResult(
        name=name,
        status="passed" if process.returncode == 0 else "failed",
        command=command,
        returncode=process.returncode,
        output_path=_relative(tools_dir.parent, output_path),
    )


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Terminate a debugger tool and descendants after an interruption."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        process.kill()
        process.communicate()


def _run_static_contract_scan(root: Path, paths: Sequence[Path], tools_dir: Path) -> ToolResult:
    """Run deterministic security and exception-contract AST checks."""
    findings: List[Finding] = []
    python_paths = [path for path in paths if path.suffix == ".py" and path.parts[:2] == ("src", "exonym")]
    for relative in python_paths:
        source_path = root / relative
        try:
            source = source_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(relative))
        except (OSError, SyntaxError, UnicodeError) as exc:
            findings.append(
                Finding(
                    "EXD000", "blocker", relative.as_posix(), 1,
                    "Source could not be parsed: {0}".format(exc),
                    "Fix the Python syntax or encoding before rerunning exonym debug.",
                )
            )
            continue
        findings.extend(_scan_tree(tree, relative, source))

    output_path = tools_dir / "static-contract.json"
    output_path.write_text(
        json.dumps([finding.as_dict() for finding in findings], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    has_blocker = any(finding.severity == "blocker" for finding in findings)
    return ToolResult(
        name="static-contract",
        status="failed" if has_blocker else "passed",
        command=["internal", "ast-contract-scan"],
        returncode=1 if has_blocker else 0,
        output_path=_relative(tools_dir.parent, output_path),
        detail="{0} finding(s)".format(len(findings)),
        findings=findings,
    )


def _scan_tree(tree: ast.AST, relative: Path, source: str) -> List[Finding]:
    """Return high-signal unsafe-operation and exception findings for one module."""
    findings: List[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = _dotted_name(node.func)
            if dotted in {"pickle.load", "pickle.loads", "os.system"}:
                findings.append(_finding("EXD001", "blocker", relative, node, "unsafe deserialization or shell execution", "Use a safe parser or subprocess argument list."))
            if dotted == "yaml.load":
                findings.append(_finding("EXD002", "blocker", relative, node, "unsafe yaml.load invocation", "Use yaml.safe_load or an explicit SafeLoader."))
            if dotted in {"numpy.load", "np.load"} and any(
                keyword.arg == "allow_pickle" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                for keyword in node.keywords
            ):
                findings.append(_finding("EXD003", "blocker", relative, node, "np.load allows pickle deserialization", "Reject pickle-backed arrays or use allow_pickle=False."))
            if dotted.startswith("subprocess.") and any(
                keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                for keyword in node.keywords
            ):
                findings.append(_finding("EXD004", "blocker", relative, node, "subprocess call enables shell=True", "Pass an argument list with shell disabled."))
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                findings.append(_finding("EXD101", "warning", relative, node, "bare except hides control-flow and system exceptions", "Catch the narrow expected exception type."))
            elif (
                isinstance(node.type, ast.Name)
                and node.type.id == "Exception"
                and not _is_documented_fail_closed_handler(node, source)
            ):
                findings.append(_finding("EXD102", "warning", relative, node, "broad Exception handler requires review", "Catch explicit failure types or document the fail-closed boundary."))
    return findings


def _is_documented_fail_closed_handler(handler: ast.ExceptHandler, source: str) -> bool:
    """Return whether an intentional broad handler has an explicit rationale."""
    lines = source.splitlines()
    line_index = handler.lineno - 1
    return (
        0 <= line_index < len(lines)
        and "# exonym: fail-closed" in lines[line_index].lower()
    )


def _dotted_name(node: ast.AST) -> str:
    """Return a dotted call name when it can be statically resolved."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return "{0}.{1}".format(prefix, node.attr) if prefix else node.attr
    return ""


def _finding(rule_id: str, severity: str, path: Path, node: ast.AST, message: str, remediation: str) -> Finding:
    """Create one source-location finding."""
    return Finding(rule_id, severity, path.as_posix(), int(getattr(node, "lineno", 1)), message, remediation)


def _write_reports(report: DebugReport) -> None:
    """Write JSON, Markdown, and SARIF reports atomically."""
    report_dir = report.run_dir / "report"
    _atomic_json(report_dir / "debug-report.json", report.as_dict())
    _atomic_json(report_dir / "debug-report.sarif", _sarif(report))
    _atomic_text(report_dir / "debug-report.md", _markdown(report))


def _atomic_json(path: Path, value: object) -> None:
    """Write one deterministic JSON document without partial reports."""
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    """Atomically replace one UTF-8 report file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _markdown(report: DebugReport) -> str:
    """Render a concise human review report."""
    lines = ["# Exonym source debugger", "", "- Mode: `{0}`".format(report.mode)]
    lines.append("- Candidate access: `forbidden`")
    lines.append("- Exit code: `{0}`".format(report.exit_code))
    lines.append("- Temporary sandbox preserved: `{0}`".format(str(report.tmp_preserved).lower()))
    lines.extend(["", "| Check | Status | Detail |", "| --- | --- | --- |"])
    for result in report.results:
        lines.append("| {0} | {1} | {2} |".format(result.name, result.status, result.detail or "-"))
    findings = [finding for result in report.results for finding in result.findings]
    if findings:
        lines.extend(["", "## Findings", ""])
        for finding in findings:
            lines.append(
                "- `{0}` {1}:{2} — {3}. Remediation: {4}".format(
                    finding.rule_id, finding.path, finding.line, finding.message, finding.remediation
                )
            )
    return "\n".join(lines) + "\n"


def _sarif(report: DebugReport) -> Dict[str, object]:
    """Return SARIF findings suitable for CI code-scanning upload."""
    results = []
    for tool_result in report.results:
        for finding in tool_result.findings:
            results.append(
                {
                    "ruleId": finding.rule_id,
                    "level": "error" if finding.severity == "blocker" else "warning",
                    "message": {"text": finding.message},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {"uri": finding.path},
                                "region": {"startLine": finding.line},
                            }
                        }
                    ],
                }
            )
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{"tool": {"driver": {"name": "exonym-debugger"}}, "results": results}],
    }
