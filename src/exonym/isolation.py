"""Repository isolation enforcement.

Invariant: every target-specific path and byte must live under
``candidate/<candidate-id>/``. Everything outside ``candidate/`` must be
demonstrably target-neutral.

Checks implemented here:

1. Path ownership: no top-level ``archive/`` or ``data/``; no research payload
   formats (FITS/CSV/NPY/NPZ/PDF/TeX/ZIP/TAR/GZ/IPYNB/...) outside
   ``candidate/``.
2. Registered alias scan: TOI/TIC and alias tokens derived from every
   ``candidate.json`` must not appear in neutral-zone text.
3. Python AST scan (``src/`` only): no numeric literals bound to
   sector/ephemeris names or ephemeris call keywords.
4. Symlink/junction/reparse-point rejection across the whole tree.

The module ships with a test suite and a CLI:
``exonym verify`` or ``python -m exonym.isolation --root .``
"""

from __future__ import annotations

import argparse
import ast
import ctypes
from datetime import date
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .workspace import discover_candidates

CANDIDATE_DIRECTORY = "candidate"

NEUTRAL_EXTENSIONS = {
    ".py", ".md", ".toml", ".txt", ".json", ".yaml", ".yml", ".gitignore", ".cff", "",
}

# `.agents` and `.opencode` are host-injected tool state, not Exonym repository
# content; their nested third-party repositories and dependencies are excluded
# explicitly. Every other non-candidate directory is audited. Do not turn this
# into an allowlist of project source folders.
EXCLUDED_TOP_LEVEL_DIRECTORIES = {".agents", ".opencode"}
EXCLUDED_DIRECTORY_NAMES = {".git", "__pycache__", ".pytest_cache"}

RESEARCH_PAYLOAD_EXTENSIONS = {
    ".fits", ".fit", ".fz", ".csv", ".tsv", ".parquet",
    ".npy", ".npz", ".h5", ".hdf5", ".pkl", ".joblib",
    ".pdf", ".tex", ".zip", ".tar", ".gz", ".ipynb",
    ".png", ".jpg", ".jpeg", ".log",
}

TOI_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])TOI[\s._-]*\d{1,7}(?:\.\d{1,2})?(?![A-Za-z0-9])", re.IGNORECASE
)
TIC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])TIC[\s._:-]*\d{5,12}(?![A-Za-z0-9])", re.IGNORECASE
)
COMPACT_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:TOI|TIC)\d{4,}(?![A-Za-z0-9])", re.IGNORECASE)

SECTOR_NAME = re.compile(r"^(?:tess_)?sectors?(?:_ids?|_numbers?)?$", re.IGNORECASE)
EPHEMERIS_NAME = re.compile(
    r"^(?:period|epoch|t0|duration|ephemeris|transit_time)(?:_days?|_hours?|_btjd|_bjd|_jd|_tdb)?$",
    re.IGNORECASE,
)
EPHEMERIS_KEYWORDS = {
    "period_days", "epoch_btjd", "epoch_bjd", "epoch_jd", "epoch_tdb", "t0",
    "duration_hours", "duration_days", "transit_time", "ephemeris",
}
TRIVIAL_VALUES = {0.0, 1.0, -1.0, 0.5, -0.5}

EXCEPTIONS_PATH = Path("policy") / "isolation-exceptions.json"
EXCEPTION_ENTRY_FIELDS = {"path", "line", "rule", "reason", "expires"}
EXCEPTION_REGISTRY_RULES = {
    "invalid-isolation-exception-registry",
    "invalid-isolation-exception",
    "expired-isolation-exception",
}


@dataclass(frozen=True)
class Violation:
    path: str
    rule: str
    detail: str
    line: Optional[int] = None

    def __str__(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else self.path
        return f"[{self.rule}] {location}: {self.detail}"


@dataclass
class IsolationReport:
    violations: List[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, path: Path, rule: str, detail: str, line: Optional[int] = None) -> None:
        self.violations.append(
            Violation(path.as_posix(), rule, detail, line=line)
        )


def is_reparse_point(path: Path) -> bool:
    """Detect symlinks and junctions on Windows and POSIX."""
    if path.is_symlink():
        return True
    if os.name != "nt":
        return False
    attributes = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    return bool(attributes != 0xFFFFFFFF and (attributes & 0x400))


def _text_payload(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _alias_tokens(candidate_root: Path) -> Dict[str, str]:
    """Return {escaped-token: owning candidate_id} for every registered alias."""
    tokens: Dict[str, str] = {}
    for workspace in discover_candidates(candidate_root.parent):
        for alias in workspace.metadata.get("identifiers", {}).get("aliases", []):
            compact = re.sub(r"[^0-9A-Za-z]", "", str(alias))
            if compact and len(compact) >= 4:
                tokens[re.escape(str(alias))] = workspace.candidate_id
                tokens[re.escape(compact)] = workspace.candidate_id
    return tokens


def _scan_text_for_ids(
    report: IsolationReport,
    path: Path,
    content: str,
    alias_tokens: Dict[str, str],
    scan_catalog_patterns: bool = True,
) -> None:
    for line_number, line in enumerate(content.splitlines(), start=1):
        if scan_catalog_patterns and (
            TOI_PATTERN.search(line) or TIC_PATTERN.search(line) or COMPACT_ID_PATTERN.search(line)
        ):
            report.add(
                path,
                "target-id-in-neutral-zone",
                f"catalog identifier found: {line.strip()[:120]}",
                line_number,
            )
        for token, owner in alias_tokens.items():
            if re.search(token, line, flags=re.IGNORECASE):
                report.add(
                    path,
                    "registered-alias-leak",
                    f"alias owned by {owner}: {line.strip()[:120]}",
                    line_number,
                )
                break


def _scan_ast(report: IsolationReport, path: Path) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if not isinstance(value, ast.Constant) or isinstance(value.value, bool):
                continue
            try:
                number = float(value.value)
            except (TypeError, ValueError):
                continue
            if number in TRIVIAL_VALUES:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and (
                    SECTOR_NAME.fullmatch(target.id) or EPHEMERIS_NAME.fullmatch(target.id)
                ):
                    report.add(
                        path,
                        "hardcoded-target-literal",
                        f"{target.id} = {number!r}",
                        node.lineno,
                    )
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in EPHEMERIS_KEYWORDS and isinstance(
                    keyword.value, ast.Constant
                ):
                    try:
                        number = float(keyword.value.value)
                    except (TypeError, ValueError):
                        continue
                    if number not in TRIVIAL_VALUES and float(number).is_integer() is False:
                        report.add(
                            path,
                            "hardcoded-ephemeris-keyword",
                            f"{keyword.arg}={number!r}",
                            node.lineno,
                        )


def _add_exception_violation(
    report: IsolationReport,
    path: Path,
    rule: str,
    detail: str,
) -> None:
    """Record a registry error that cannot itself be excepted."""
    report.add(path, rule, detail)


def _validate_exception_entry(
    entry: Any,
    index: int,
    path: Path,
    report: IsolationReport,
) -> Optional[Tuple[str, Optional[int], str]]:
    """Return a safe exception key, or report why the entry is unusable."""
    prefix = f"entry {index}"
    if not isinstance(entry, dict):
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception",
            f"{prefix} must be an object",
        )
        return None

    entry_fields = set(entry)
    if entry_fields != EXCEPTION_ENTRY_FIELDS:
        missing = sorted(EXCEPTION_ENTRY_FIELDS - entry_fields)
        unexpected = sorted(entry_fields - EXCEPTION_ENTRY_FIELDS)
        details: List[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected fields: {', '.join(unexpected)}")
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception",
            f"{prefix} has invalid shape ({'; '.join(details)})",
        )
        return None

    exception_path = entry["path"]
    if not isinstance(exception_path, str) or not exception_path.strip():
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception",
            f"{prefix} path must be a non-empty relative POSIX path",
        )
        return None
    if "\\" in exception_path:
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception",
            f"{prefix} path must use POSIX separators",
        )
        return None
    posix_path = PurePosixPath(exception_path)
    if posix_path.is_absolute() or ".." in posix_path.parts or exception_path == ".":
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception",
            f"{prefix} path must remain inside the repository",
        )
        return None

    line = entry["line"]
    if line is not None and (isinstance(line, bool) or not isinstance(line, int) or line < 1):
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception",
            f"{prefix} line must be a positive integer or null",
        )
        return None

    rule = entry["rule"]
    if not isinstance(rule, str) or not rule.strip():
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception",
            f"{prefix} rule must be a non-empty string",
        )
        return None

    reason = entry["reason"]
    if not isinstance(reason, str) or not reason.strip():
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception",
            f"{prefix} reason must be a non-empty string",
        )
        return None

    expires = entry["expires"]
    if not isinstance(expires, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", expires):
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception",
            f"{prefix} expires must be an ISO date (YYYY-MM-DD)",
        )
        return None
    try:
        expiry_date = date.fromisoformat(expires)
    except ValueError:
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception",
            f"{prefix} expires must be a valid ISO date",
        )
        return None
    if expiry_date < date.today():
        _add_exception_violation(
            report,
            path,
            "expired-isolation-exception",
            f"{prefix} expired on {expires}",
        )
        return None

    return (posix_path.as_posix(), line, rule)


def _load_exceptions(
    root: Path,
    report: IsolationReport,
) -> Set[Tuple[str, Optional[int], str]]:
    """Load only well-formed, unexpired, repository-relative exceptions."""
    path = root / EXCEPTIONS_PATH
    if not path.exists() and not path.is_symlink():
        return set()
    if is_reparse_point(path):
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception-registry",
            "registry must not be a symlink or reparse point",
        )
        return set()
    if not path.is_file():
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception-registry",
            "registry must be a JSON file",
        )
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception-registry",
            f"could not parse registry: {exc}",
        )
        return set()
    if not isinstance(payload, dict) or set(payload) != {"entries"}:
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception-registry",
            "registry must be an object with exactly one 'entries' field",
        )
        return set()
    entries = payload["entries"]
    if not isinstance(entries, list):
        _add_exception_violation(
            report,
            path,
            "invalid-isolation-exception-registry",
            "registry entries must be a list",
        )
        return set()

    exception_paths: Set[Tuple[str, Optional[int], str]] = set()
    for index, entry in enumerate(entries, start=1):
        key = _validate_exception_entry(entry, index, path, report)
        if key is not None:
            exception_paths.add(key)
    return exception_paths


def _is_excluded_neutral_directory(relative: Path) -> bool:
    """Return whether a path belongs to host/VCS state outside the repository."""
    parts = relative.parts
    return bool(
        parts
        and (
            parts[0] in EXCLUDED_TOP_LEVEL_DIRECTORIES
            or any(part in EXCLUDED_DIRECTORY_NAMES for part in parts)
        )
    )


def _iter_neutral_entries(root: Path) -> Iterable[Path]:
    """Yield every auditable entry outside candidate/ without following links."""
    for current, directory_names, file_names in os.walk(
        str(root), topdown=True, followlinks=False
    ):
        directory = Path(current)
        next_directories: List[str] = []
        for name in sorted(directory_names):
            path = directory / name
            relative = path.relative_to(root)
            if relative.parts[0] == CANDIDATE_DIRECTORY or _is_excluded_neutral_directory(relative):
                continue
            yield path
            if not is_reparse_point(path):
                next_directories.append(name)
        directory_names[:] = next_directories

        for name in sorted(file_names):
            path = directory / name
            relative = path.relative_to(root)
            if not _is_excluded_neutral_directory(relative):
                yield path


def _scan_candidate_reparse_points(report: IsolationReport, candidate_root: Path) -> None:
    """Reject links in candidate workspaces without inspecting their payload."""
    if not candidate_root.exists() and not candidate_root.is_symlink():
        return
    if is_reparse_point(candidate_root):
        report.add(
            candidate_root,
            "symlink-or-reparse-point",
            "candidate workspaces must not be linked",
        )
        return

    for current, directory_names, file_names in os.walk(
        str(candidate_root), topdown=True, followlinks=False
    ):
        directory = Path(current)
        next_directories: List[str] = []
        for name in sorted(directory_names):
            if name in EXCLUDED_DIRECTORY_NAMES:
                continue
            path = directory / name
            if is_reparse_point(path):
                report.add(path, "symlink-or-reparse-point", "not permitted in candidate workspaces")
                continue
            next_directories.append(name)
        directory_names[:] = next_directories
        for name in sorted(file_names):
            path = directory / name
            if is_reparse_point(path):
                report.add(path, "symlink-or-reparse-point", "not permitted in candidate workspaces")


def check_repository(root: Path) -> IsolationReport:
    """Run the full isolation check over a repository tree."""
    requested_root = Path(root)
    report = IsolationReport()
    if is_reparse_point(requested_root):
        report.add(
            requested_root,
            "symlink-or-reparse-point",
            "repository root must not be linked",
        )
    root = requested_root.resolve()
    exception_paths = _load_exceptions(root, report)
    alias_tokens = _alias_tokens(root / CANDIDATE_DIRECTORY)

    archive_root = root / "archive"
    if archive_root.exists() or archive_root.is_symlink():
        report.add(
            archive_root,
            "top-level-archive-forbidden",
            "archived targets must remain under candidate/<candidate-id>/",
        )
    data_root = root / "data"
    if data_root.exists() or data_root.is_symlink():
        report.add(
            data_root,
            "top-level-data-forbidden",
            "target data must live under candidate/<candidate-id>/data/",
        )

    for path in _iter_neutral_entries(root):
        if is_reparse_point(path):
            report.add(path, "symlink-or-reparse-point", "not permitted outside candidate/")
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            report.add(
                path,
                "unsupported-neutral-entry",
                "neutral-zone entries must be regular files or directories",
            )
            continue
        relative = path.relative_to(root)
        if path.suffix.lower() not in NEUTRAL_EXTENSIONS:
            report.add(
                path,
                "research-payload-outside-candidate",
                f"format {path.suffix or '(none)'} is only allowed under candidate/",
            )
            continue
        content = _text_payload(path)
        if content is None:
            report.add(
                path,
                "unreadable-neutral-content",
                "neutral-zone files must be UTF-8 text",
            )
            continue
        # Shared tests necessarily exercise ID-detection fixtures, so generic
        # catalog patterns are skipped there; real registered aliases are
        # still scanned everywhere.
        _scan_text_for_ids(
            report,
            path,
            content,
            alias_tokens,
            scan_catalog_patterns=not (relative.parts and relative.parts[0] == "tests"),
        )
        if path.suffix.lower() == ".py" and relative.parts and relative.parts[0] == "src":
            _scan_ast(report, path)

    candidate_root = root / CANDIDATE_DIRECTORY
    _scan_candidate_reparse_points(report, candidate_root)

    if exception_paths:
        kept: List[Violation] = []
        for violation in report.violations:
            try:
                relative_path = Path(violation.path).relative_to(root).as_posix()
            except ValueError:
                relative_path = None
            key = (relative_path, violation.line, violation.rule)
            if violation.rule not in EXCEPTION_REGISTRY_RULES and key in exception_paths:
                continue
            kept.append(violation)
        report.violations = kept
    return report


def run_audit(root: Path) -> IsolationReport:
    """Run the full repository audit: isolation checks plus JSON schema
    validation of candidate records, provenance sidecars, and claims."""
    report = check_repository(root)
    try:
        from .schemas import validate_schemas

        validate_schemas(root, report)
    except Exception as exc:  # pragma: no cover - defensive
        report.add(Path(root), "schema-validation-error", str(exc))
    return report


def format_report(report: IsolationReport) -> str:
    if report.ok:
        return "ISOLATION: PASS (no violations)"
    lines = [f"ISOLATION: FAIL ({len(report.violations)} violation(s))"]
    lines.extend(str(violation) for violation in report.violations)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enforce candidate/ research isolation and schema integrity."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument(
        "--schemas-only",
        action="store_true",
        help="Validate JSON schemas only (skip the isolation scan).",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.schemas_only:
        report = IsolationReport()
        try:
            from .schemas import validate_schemas

            validate_schemas(root, report)
        except Exception as exc:  # pragma: no cover - defensive
            report.add(Path(root), "schema-validation-error", str(exc))
    else:
        report = run_audit(root)
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
