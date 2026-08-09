"""Access to target-neutral templates and schemas shipped with EXONYM.

Projects checked out from source may keep editable ``templates/`` and
``schemas/`` directories at their repository root.  Installed wheels do not
have those source-tree directories, so this module provides a packaged
fallback.  A present local directory always takes precedence; an absent local
directory falls back to the immutable package copy.

The functions deliberately fail loudly if neither resource source is usable.
Creating a candidate without its protocol templates would otherwise leave a
workspace that cannot pass its workflow gates.
"""

from __future__ import annotations

from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Iterator, Tuple


RESOURCE_PACKAGE = "exonym._resources"
TEMPLATE_DIRECTORY = "templates"
SCHEMA_DIRECTORY = "schemas"


class ResourceUnavailableError(RuntimeError):
    """Raised when a required bundled EXONYM resource cannot be loaded."""


def _bundled_directory(name: str) -> Any:
    """Return a bundled resource directory or raise a descriptive error."""
    try:
        directory = importlib_resources.files(RESOURCE_PACKAGE).joinpath(name)
    except (AttributeError, ModuleNotFoundError) as exc:
        raise ResourceUnavailableError(
            "installed package does not expose bundled {0}".format(name)
        ) from exc
    if not directory.is_dir():
        raise ResourceUnavailableError(
            "installed package is missing bundled {0}".format(name)
        )
    return directory


def _walk_bundled_files(directory: Any, relative: Path = Path()) -> Iterator[Tuple[Path, str]]:
    """Yield UTF-8 files from an ``importlib.resources`` directory tree."""
    for child in sorted(directory.iterdir(), key=lambda item: item.name):
        child_relative = relative / child.name
        if child.is_dir():
            yield from _walk_bundled_files(child, child_relative)
        elif child.is_file():
            yield child_relative, child.read_text(encoding="utf-8")


def iter_template_texts(repository_root: Path) -> Iterator[Tuple[Path, str]]:
    """Yield relative template paths and UTF-8 contents for a workspace.

    A project-local ``templates/`` tree is an intentional editable override.
    If it is absent, use templates embedded in the installed distribution.  A
    local but empty template directory is an invalid project configuration and
    is rejected rather than silently producing an incomplete workspace.
    """
    template_root = Path(repository_root).resolve() / TEMPLATE_DIRECTORY
    if template_root.exists():
        if not template_root.is_dir():
            raise FileNotFoundError(
                "template path is not a directory: {0}".format(template_root)
            )
        templates = [path for path in sorted(template_root.rglob("*")) if path.is_file()]
        if not templates:
            raise FileNotFoundError(
                "template directory contains no files: {0}".format(template_root)
            )
        for template in templates:
            yield template.relative_to(template_root), template.read_text(encoding="utf-8")
        return

    bundled_templates = list(_walk_bundled_files(_bundled_directory(TEMPLATE_DIRECTORY)))
    if not bundled_templates:
        raise ResourceUnavailableError("bundled template directory contains no files")
    yield from bundled_templates


def read_schema_text(repository_root: Path, filename: str) -> str:
    """Read a schema from the repository or, for installed wheels, the package.

    If a repository defines a ``schemas/`` directory, it is authoritative:
    missing files are reported rather than being masked by the bundled copy.
    This keeps source-tree integrity checks strict while allowing a new
    workspace initialized by an installed wheel to validate itself.
    """
    if Path(filename).name != filename:
        raise ValueError("schema filename must not contain path components")

    schema_root = Path(repository_root).resolve() / SCHEMA_DIRECTORY
    if schema_root.exists():
        if not schema_root.is_dir():
            raise FileNotFoundError(
                "schema path is not a directory: {0}".format(schema_root)
            )
        path = schema_root / filename
        if not path.is_file():
            raise FileNotFoundError("schema file not found: {0}".format(path))
        return path.read_text(encoding="utf-8")

    resource = _bundled_directory(SCHEMA_DIRECTORY).joinpath(filename)
    if not resource.is_file():
        raise ResourceUnavailableError(
            "installed package is missing bundled schema {0}".format(filename)
        )
    return resource.read_text(encoding="utf-8")
