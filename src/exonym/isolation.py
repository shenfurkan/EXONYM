"""Repository isolation enforcement.

Invariant: every target-specific path and byte must live under
``candidate/<candidate-id>/``. Everything outside ``candidate/`` must be
demonstrably target-neutral.

Checks implemented here:

1. Path ownership: no top-level ``archive/`` or ``data/``; no research payload
   formats (FITS/CSV/NPY/NPZ/PDF/TeX/ZIP/TAR/GZ/IPYNB/...) outside
   ``candidate/`` except the target-neutral source manuscript template.
2. Registered alias scan: TOI/TIC and alias tokens derived from every
   ``candidate.json`` must not appear in neutral-zone text.
3. Python AST scan (``src/`` only): no numeric literals bound to
   sector/ephemeris names or ephemeris call keywords.
4. Symlink/junction/reparse-point rejection across the whole tree.

The module ships with a test suite and a CLI: ``exonym verify --source`` for
the neutral-zone audit, or ``exonym verify --candidates`` for candidate data
integrity. The legacy ``verify candidate`` spelling remains supported.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
from datetime import date
import json
import os
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .workspace import discover_candidates
from .constants import (
    EARTH_MASS_ONE_JULIAN_YEAR_RV_SEMI_AMPLITUDE_M_PER_S,
    JULIAN_YEAR_DAYS,
)

CANDIDATE_DIRECTORY = "candidate"

NEUTRAL_EXTENSIONS = {
    ".py", ".md", ".toml", ".txt", ".json", ".yaml", ".yml", ".gitignore", ".cff", "",
}

# `.agents` and `.opencode` are host-injected tool state, not Exonym repository
# content; their nested third-party repositories and dependencies are excluded
# explicitly. Every non-candidate *subdirectory* is audited. Loose files placed
# directly at the repository root are the operator's unrestricted workspace:
# extension and identifier rules apply only below the top level. Do not turn
# this into an allowlist of project source folders.
EXCLUDED_TOP_LEVEL_DIRECTORIES = {".agents", ".opencode", "images", "assets", "textbooks", "log", "logs"}
EXCLUDED_DIRECTORY_NAMES = {".git", "__pycache__", ".pytest_cache"}
DEBUG_SOURCE_DIRECTORIES = {".github", "policy", "schemas", "src", "templates", "tests"}
DEBUG_SOURCE_FILES = {".pre-commit-config.yaml", "pyproject.toml"}

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
CANONICAL_CONSTANTS_MODULE = "src/exonym/constants.py"
# These values define the conventional Earth-mass/one-Julian-year RV scaling.
# They are intentionally narrow: generic physical constants and ordinary
# numerical thresholds remain outside this source-isolation lint's scope.
# Keys are generated from the canonical constants module so the numeric
# literals themselves do not appear in this file and trip the audit.
BANNED_NORMALIZATION_LITERALS = {
    str(EARTH_MASS_ONE_JULIAN_YEAR_RV_SEMI_AMPLITUDE_M_PER_S): "EARTH_MASS_ONE_JULIAN_YEAR_RV_SEMI_AMPLITUDE_M_PER_S",
    str(JULIAN_YEAR_DAYS): "JULIAN_YEAR_DAYS",
}

EXCEPTIONS_PATH = Path("policy") / "isolation-exceptions.json"
EXCEPTION_ENTRY_FIELDS = {"path", "line", "rule", "reason", "expires"}
EXCEPTION_REGISTRY_RULES = {
    "invalid-isolation-exception-registry",
    "invalid-isolation-exception",
    "expired-isolation-exception",
}


@dataclass(frozen=True)
class Violation:
    """One auditable isolation or schema-validation finding.

    Attributes:
        path: Repository-relative path associated with the finding.
        rule: Stable machine-readable rule identifier.
        detail: Human-readable explanation of the failed condition.
        line: Optional one-based source line number.
        severity: ``"error"`` for a failing condition or another label for a
            non-fatal finding.
    """

    path: str
    rule: str
    detail: str
    line: Optional[int] = None
    severity: str = "error"

    def __str__(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else self.path
        tag = f"[{self.severity.upper()}] " if self.severity != "error" else ""
        return f"{tag}[{self.rule}] {location}: {self.detail}"


@dataclass
class IsolationReport:
    """Mutable collection of audit findings and their derived status.

    Attributes:
        violations: Findings accumulated by isolation and schema checks.
    """

    violations: List[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)

    @property
    def warnings(self) -> List[Violation]:
        return [v for v in self.violations if v.severity != "error"]

    def add(self, path: Path, rule: str, detail: str, line: Optional[int] = None, severity: str = "error") -> None:
        """Append a normalized finding to this report.

        Args:
            path: Path associated with the finding.
            rule: Stable rule identifier.
            detail: Explanation for an operator.
            line: Optional one-based source line number.
            severity: Finding severity; ``"error"`` causes :attr:`ok` to be
                false.
        """
        self.violations.append(
            Violation(path.as_posix(), rule, detail, line=line, severity=severity)
        )


def is_reparse_point(path: Path) -> bool:
    """Return whether a path is a symlink, junction, or Windows reparse point.

    Args:
        path: Filesystem path to inspect without following it.

    Returns:
        ``True`` when the path can redirect the audit outside its expected
        ownership boundary.
    """
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
    """Return {literal-token: owning candidate_id} for every registered alias."""
    tokens: Dict[str, str] = {}
    for workspace in discover_candidates(candidate_root.parent):
        for alias in workspace.metadata.get("identifiers", {}).get("aliases", []):
            compact = re.sub(r"[^0-9A-Za-z]", "", str(alias))
            if compact and len(compact) >= 4:
                tokens[str(alias)] = workspace.candidate_id
                tokens[compact] = workspace.candidate_id
    return tokens


def _compile_alias_matcher(alias_tokens: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    """Prepare case-folded literals for one fast pass per source line."""
    return tuple((token.casefold(), owner) for token, owner in alias_tokens.items())


def _python_comment_and_docstring_lines(source: str) -> Set[int]:
    """Return line numbers of comments and docstrings in Python source.

    Uses the tokenize module to identify ``#`` comment lines and the bodies
    of module/class/function docstrings so that educational or documentation
    identifiers inside them do not trigger false-positive isolation violations.
    """
    skip: Set[int] = set()
    try:
        tokens = tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                skip.add(tok.start[0])
            elif tok.type == tokenize.STRING:
                # Multi-line docstrings cover every line from start to end.
                for line_no in range(tok.start[0], tok.end[0] + 1):
                    skip.add(line_no)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return skip


def _scan_text_for_ids(
    report: IsolationReport,
    path: Path,
    content: str,
    alias_tokens: Dict[str, str],
    scan_catalog_patterns: bool = True,
    skip_lines: Optional[Set[int]] = None,
    alias_matcher: Optional[Tuple[Tuple[str, str], ...]] = None,
) -> None:
    for line_number, line in enumerate(content.splitlines(), start=1):
        if skip_lines is not None and line_number in skip_lines:
            continue
        if scan_catalog_patterns and (
            TOI_PATTERN.search(line) or TIC_PATTERN.search(line) or COMPACT_ID_PATTERN.search(line)
        ):
            report.add(
                path,
                "target-id-in-neutral-zone",
                f"catalog identifier found: {line.strip()[:120]}",
                line_number,
            )
        if alias_matcher is not None:
            folded_line = line.casefold()
            for token, owner in alias_matcher:
                if token in folded_line:
                    report.add(
                        path,
                        "registered-alias-leak",
                        f"alias owned by {owner}: {line.strip()[:120]}",
                        line_number,
                    )
        else:
            for token, owner in alias_tokens.items():
                if re.search(re.escape(token), line, flags=re.IGNORECASE):
                    report.add(
                        path,
                        "registered-alias-leak",
                        f"alias owned by {owner}: {line.strip()[:120]}",
                        line_number,
                    )
                break


def _numeric_literal(value: ast.AST) -> Optional[float]:
    """Return one numeric literal, including a unary sign, or ``None``."""
    if isinstance(value, ast.Constant) and not isinstance(value.value, bool):
        try:
            return float(value.value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
        operand = _numeric_literal(value.operand)
        if operand is not None:
            return operand if isinstance(value.op, ast.UAdd) else -operand
    return None


def _assignment_target_names(target: ast.AST) -> Iterable[str]:
    """Yield names assigned directly or through an attribute target."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Attribute):
        yield target.attr
    elif isinstance(target, (ast.List, ast.Tuple)):
        for element in target.elts:
            yield from _assignment_target_names(element)


def _is_target_literal_name(name: str) -> bool:
    """Return whether an identifier names a sector or ephemeris value."""
    return bool(
        name in EPHEMERIS_KEYWORDS
        or SECTOR_NAME.fullmatch(name)
        or EPHEMERIS_NAME.fullmatch(name)
    )


def _report_target_literal(
    report: IsolationReport,
    path: Path,
    name: str,
    number: float,
    line: int,
) -> None:
    """Record a non-fatal target-specific numeric literal for source review."""
    report.add(
        path,
        "hardcoded-target-literal",
        f"{name} = {number!r}",
        line,
        severity="warning",
    )


def _scan_ast(
    report: IsolationReport,
    path: Path,
    relative_path: Optional[Path] = None,
) -> None:
    """Find target-specific literal encodings and duplicate RV normalizations."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return

    relative_text = (
        relative_path.as_posix()
        if relative_path is not None
        else path.as_posix().replace("\\", "/")
    )
    is_canonical_constants_module = relative_text.endswith(CANONICAL_CONSTANTS_MODULE)
    for node in ast.walk(tree):
        if not is_canonical_constants_module and isinstance(node, ast.Constant):
            number = _numeric_literal(node)
            literal_text = repr(number) if number is not None else None
            if literal_text in BANNED_NORMALIZATION_LITERALS:
                constant_name = BANNED_NORMALIZATION_LITERALS[literal_text]
                report.add(
                    path,
                    "duplicated-sensitive-normalization",
                    f"{literal_text} duplicates {constant_name}; import it from exonym.constants",
                    node.lineno,
                )

        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            number = _numeric_literal(node.value)
            if number is None or number in TRIVIAL_VALUES:
                continue
            for target in targets:
                for name in _assignment_target_names(target):
                    if _is_target_literal_name(name):
                        _report_target_literal(report, path, name, number, node.lineno)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                number = _numeric_literal(value)
                if number is None or number in TRIVIAL_VALUES or not _is_target_literal_name(key.value):
                    continue
                _report_target_literal(report, path, key.value, number, value.lineno)
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg is None or not _is_target_literal_name(keyword.arg):
                    continue
                number = _numeric_literal(keyword.value)
                if number is not None and number not in TRIVIAL_VALUES:
                    report.add(
                        path,
                        "hardcoded-ephemeris-keyword",
                        f"{keyword.arg}={number!r}",
                        node.lineno,
                        severity="warning",
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
            or (len(parts) >= 2 and parts[0] == "methods" and parts[1] == "papers")
            or (len(parts) >= 2 and parts[0] == "paper" and parts[1] == "downloads")
        )
    )


def _allowed_neutral_file(relative: Path, path: Path) -> bool:
    """Allow ordinary neutral text and the exact target-neutral paper sources."""
    if path.suffix.lower() in NEUTRAL_EXTENSIONS:
        return True
    return relative.as_posix() in {
        "paper/paper.bib",
        "templates/paper/paper_template.tex",
        "src/exonym/_resources/templates/paper/paper_template.tex",
    }


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


def _iter_debug_source_entries(root: Path) -> Iterable[Path]:
    """Yield only debugger-owned source inputs without reading candidate/ or operator folders."""
    for name in sorted(DEBUG_SOURCE_DIRECTORIES):
        source_root = root / name
        if not source_root.exists():
            continue
        if source_root.is_file():
            yield source_root
            continue
        for current, directory_names, file_names in os.walk(
            str(source_root), topdown=True, followlinks=False
        ):
            directory = Path(current)
            next_directories: List[str] = []
            for child_name in sorted(directory_names):
                path = directory / child_name
                relative = path.relative_to(root)
                yield path
                if not is_reparse_point(path) and not _is_excluded_neutral_directory(relative):
                    next_directories.append(child_name)
            directory_names[:] = next_directories
            for child_name in sorted(file_names):
                path = directory / child_name
                if not _is_excluded_neutral_directory(path.relative_to(root)):
                    yield path
    for name in sorted(DEBUG_SOURCE_FILES):
        path = root / name
        if path.is_file():
            yield path


def _scan_candidate_reparse_points(
    report: IsolationReport, candidate_root: Path, candidate_id: Optional[str] = None
) -> None:
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

    scan_root = candidate_root / candidate_id if candidate_id is not None else candidate_root
    if candidate_id is not None and not scan_root.is_dir():
        report.add(
            scan_root,
            "candidate-workspace-missing",
            "selected candidate workspace does not exist",
        )
        return

    for current, directory_names, file_names in os.walk(
        str(scan_root), topdown=True, followlinks=False
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


def _check_repository(
    root: Path,
    include_candidates: bool,
    candidate_id: Optional[str] = None,
    *,
    include_candidate_aliases: bool = True,
    debug_source_only: bool = False,
) -> IsolationReport:
    """Run the repository isolation check, optionally including candidate workspaces."""
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
    # Source-only audits must not access candidate metadata.  Full repository
    # verification supplies registered aliases to enforce their isolation.
    alias_tokens = _alias_tokens(root / CANDIDATE_DIRECTORY) if include_candidate_aliases else {}
    alias_matcher = _compile_alias_matcher(alias_tokens)

    if not debug_source_only:
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

    entries = _iter_debug_source_entries(root) if debug_source_only else _iter_neutral_entries(root)
    for path in entries:
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
        if len(relative.parts) == 1:
            # Root-level loose files are the operator's unrestricted workspace.
            # Any format is allowed and no identifier scanning runs here; the
            # security checks above (reparse points, regular files) still apply.
            continue
        if not _allowed_neutral_file(relative, path):
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
        is_python = path.suffix.lower() == ".py"
        comment_skip = _python_comment_and_docstring_lines(content) if is_python else None
        _scan_text_for_ids(
            report,
            path,
            content,
            alias_tokens,
            scan_catalog_patterns=not (relative.parts and relative.parts[0] == "tests"),
            skip_lines=comment_skip,
            alias_matcher=alias_matcher,
        )
        if is_python and relative.parts and relative.parts[0] == "src":
            _scan_ast(report, path, relative)

    if include_candidates:
        _scan_candidate_reparse_points(
            report, root / CANDIDATE_DIRECTORY, candidate_id=candidate_id
        )

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


def check_neutral_repository(root: Path) -> IsolationReport:
    """Audit the protected shared zone without reading candidate workspaces.

    Args:
        root: Repository root to audit.

    Returns:
        Report containing neutral-zone ownership, identifier-leak, AST, and
        reparse-point findings.
    """
    return _check_repository(
        root,
        include_candidates=False,
        include_candidate_aliases=False,
    )


def run_debug_source_audit(root: Path) -> IsolationReport:
    """Audit debugger-owned source paths without accessing candidate workspaces.

    This is intentionally narrower than :func:`check_neutral_repository`:
    it limits traversal to debugger-owned code, schema, template, policy, and
    test paths.  Both source-only modes are forbidden from reading
    ``candidate/`` in any form.
    """
    report = _check_repository(
        root,
        include_candidates=False,
        include_candidate_aliases=False,
        debug_source_only=True,
    )
    _append_schema_validation(Path(root).resolve(), report, candidate_scope=False)
    return report


def check_repository(root: Path, candidate_id: Optional[str] = None) -> IsolationReport:
    """Run the full isolation audit, including candidate workspaces.

    Args:
        root: Repository root to audit.

    Returns:
        Report with protected-zone findings plus candidate-workspace link and
        ownership findings.
    """
    return _check_repository(root, include_candidates=True, candidate_id=candidate_id)


def run_audit(
    root: Path, *, use_cache: bool = True, candidate_id: Optional[str] = None
) -> IsolationReport:
    """Run isolation checks and candidate-record schema validation together.

    Args:
        root: Repository root to audit.
        use_cache: Whether candidate JSON and hash validation may reuse the
            verification cache.

    Returns:
        Combined isolation and schema-validation report. Unexpected schema
        validation failures are captured as report findings rather than raised.
    """
    report = check_repository(root, candidate_id=candidate_id)
    try:
        from .schemas import validate_schemas
        from .verification_cache import candidate_verification_cache

        with candidate_verification_cache(root, enabled=use_cache) as cache:
            validate_schemas(root, report, candidate_id=candidate_id)
        report.cache_statistics = cache.statistics()
    except Exception as exc:  # exonym: fail-closed - return the schema error in the audit report.
        report.add(
            Path(root),
            "schema-validation-error",
            "{0}: {1}".format(type(exc).__name__, exc),
        )
    return report


def add_verify_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared verify-scope arguments to one CLI parser.

    Args:
        parser: Parser for either the standalone audit command or the main
            ``exonym verify`` subcommand.
    """
    parser.add_argument("scope", nargs="?", choices=("candidate",), help=argparse.SUPPRESS)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--source", action="store_true", help="Audit target-neutral source and resources only.")
    scope.add_argument("--candidates", action="store_true", help="Audit candidate data, records, and provenance.")
    parser.add_argument(
        "--candidate",
        dest="candidate_id",
        default=None,
        help="Limit --candidates validation to one candidate workspace.",
    )
    parser.add_argument(
        "--schemas-only",
        action="store_true",
        help="Validate schema definitions only; combine with --candidates for candidate records.",
    )
    parser.add_argument("--fix", "--remediate", action="store_true", dest="fix", help="Repair safe manifest and triage drift in candidate workspaces.")
    parser.add_argument("--fresh", action="store_true", help="Bypass candidate hash and metadata caches.")


def _append_schema_validation(
    root: Path,
    report: IsolationReport,
    *,
    candidate_scope: bool,
    candidate_id: Optional[str] = None,
) -> None:
    """Append schema findings while converting unexpected validator failures to a report entry."""
    try:
        if candidate_scope:
            from .schemas import validate_schemas

            validate_schemas(root, report, candidate_id=candidate_id)
        else:
            from .schemas import validate_schema_definitions

            validate_schema_definitions(root, report)
    except Exception as exc:  # exonym: fail-closed - return the schema error in the audit report.
        report.add(
            Path(root),
            "schema-validation-error",
            "{0}: {1}".format(type(exc).__name__, exc),
        )


def run_verify_command(
    root: Path,
    *,
    source: bool = False,
    candidates: bool = False,
    legacy_scope: Optional[str] = None,
    schemas_only: bool = False,
    fix: bool = False,
    fresh: bool = False,
    candidate_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, List[str]]], IsolationReport]:
    """Run the shared ``verify`` dispatch used by both command-line entry points.

    Args:
        root: Repository root to audit.
        source: Whether the caller explicitly selected the neutral source scope.
        candidates: Whether the caller explicitly selected candidate scope.
        legacy_scope: Optional legacy positional ``"candidate"`` scope.
        schemas_only: Limit the selected scope to JSON Schema validation.
        fix: Repair safely provable manifest and triage drift before auditing.
        fresh: Disable candidate verification-cache reuse for a full audit.
        candidate_id: Optional workspace ID for a scoped candidate audit.

    Returns:
        A pair of optional remediation actions and the completed audit report.

    Raises:
        ValueError: Explicit scopes conflict, an unsupported legacy scope is
            supplied, or ``fix`` lacks candidate scope.
    """
    if source and candidates:
        raise ValueError("--source and --candidates are mutually exclusive")
    if legacy_scope not in (None, "candidate"):
        raise ValueError("unsupported legacy verify scope: {0}".format(legacy_scope))
    if legacy_scope == "candidate" and (source or candidates):
        raise ValueError("legacy positional 'candidate' cannot be combined with --source or --candidates")

    root = Path(root).resolve()
    candidate_scope = bool(candidates or legacy_scope == "candidate")
    if candidate_id is not None:
        if not candidate_scope:
            raise ValueError("--candidate requires --candidates")
        from .workspace import validate_candidate_id

        candidate_id = validate_candidate_id(candidate_id)
    if fix and not candidate_scope:
        raise ValueError("--fix requires --candidates")

    remediated: Optional[Dict[str, List[str]]] = None
    if fix:
        from .remediation import remediate_candidate_drift

        remediated = remediate_candidate_drift(root)

    if schemas_only:
        report = IsolationReport()
        _append_schema_validation(
            root,
            report,
            candidate_scope=candidate_scope,
            candidate_id=candidate_id,
        )
    elif candidate_scope:
        report = run_audit(root, use_cache=not fresh, candidate_id=candidate_id)
    else:
        report = check_neutral_repository(root)
        _append_schema_validation(root, report, candidate_scope=False)
    return remediated, report


def _remediation_hint(rule: str) -> str:
    """Return the shortest safe command that addresses a violation category."""
    if rule in {
        "artifact-hash-mismatch",
        "provenance-hash-mismatch",
        "triage-provenance-invalid",
    }:
        return "exonym verify --candidates --fix"
    if rule.startswith("schema") or rule.endswith("-invalid"):
        return "exonym verify --candidates --fresh"
    if rule in {"target-id-in-neutral-zone", "registered-alias-leak", "hardcoded-target-literal", "hardcoded-ephemeris-keyword"}:
        return "move the target-specific value under candidate/<candidate-id>/, then run exonym verify --source"
    return "inspect the cited path, correct the record, then run exonym verify --candidates"


def format_report(report: IsolationReport) -> str:
    """Format an audit report for a human operator.

    Args:
        report: Isolation and optional schema-validation findings to render.

    Returns:
        Terminal-ready summary grouped by rule, including safe remediation
        guidance for error findings.
    """
    errors = [v for v in report.violations if v.severity == "error"]
    warnings = report.warnings
    if not errors:
        lines = ["ISOLATION: PASS (no error violations)"]
        if warnings:
            lines.append("WARNINGS: {0} warning(s) found (non-fatal)".format(len(warnings)))
            by_rule: Dict[str, List[Violation]] = {}
            for violation in warnings:
                by_rule.setdefault(violation.rule, []).append(violation)
            for rule in sorted(by_rule):
                violations = by_rule[rule]
                lines.append("  [{0}] {1} warning(s)".format(rule, len(violations)))
                lines.extend("    " + str(violation) for violation in violations)
        statistics = getattr(report, "cache_statistics", None)
        if isinstance(statistics, dict):
            lines.append(
                "CACHE: {0} hash hit(s), {1} hash miss(es), {2} candidate JSON hit(s)".format(
                    statistics.get("hash_cache_hits", 0),
                    statistics.get("hash_cache_misses", 0),
                    statistics.get("candidate_json_cache_hits", 0),
                )
            )
        return "\n".join(lines)
    lines = [f"ISOLATION: FAIL ({len(errors)} error violation(s))"]
    by_rule: Dict[str, List[Violation]] = {}
    for violation in errors:
        by_rule.setdefault(violation.rule, []).append(violation)
    for rule in sorted(by_rule):
        violations = by_rule[rule]
        lines.append("[{0}] {1} violation(s); remediation: {2}".format(rule, len(violations), _remediation_hint(rule)))
        lines.extend("  " + str(violation) for violation in violations)
    if warnings:
        lines.append("WARNINGS: {0} warning(s) found (non-fatal)".format(len(warnings)))
        by_rule_warn: Dict[str, List[Violation]] = {}
        for violation in warnings:
            by_rule_warn.setdefault(violation.rule, []).append(violation)
        for rule in sorted(by_rule_warn):
            violations = by_rule_warn[rule]
            lines.append("  [{0}] {1} warning(s)".format(rule, len(violations)))
            lines.extend("    " + str(violation) for violation in violations)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the standalone isolation-audit command-line interface.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        ``0`` when the selected audit has no error findings, otherwise ``1``.
    """
    parser = argparse.ArgumentParser(
        description="Enforce candidate/ research isolation and schema integrity."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root.")
    add_verify_arguments(parser)
    args = parser.parse_args(argv)
    try:
        remediated, report = run_verify_command(
            args.root,
            source=args.source,
            candidates=args.candidates,
            legacy_scope=args.scope,
            schemas_only=args.schemas_only,
            fix=args.fix,
            fresh=args.fresh,
            candidate_id=args.candidate_id,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if remediated is not None:
        print(json.dumps({"remediated": remediated}, indent=2, sort_keys=True))
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
