from pathlib import Path

import pytest

import exonym.__main__ as cli
from exonym.__main__ import main
from exonym.workspace import create_candidate


def _normalized_text(path: Path) -> str:
    """Return UTF-8 text with platform-neutral line endings."""
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_cli_initializes_and_verifies_without_source_resource_directories(tmp_path):
    # Arrange
    root = ["--root", str(tmp_path)]

    # Act
    initialized = main(root + ["init", "package-resource-test"])
    verified = main(root + ["verify"])

    # Assert
    workspace = tmp_path / "candidate" / "package-resource-test"
    assert initialized == 0
    assert verified == 0
    assert workspace.joinpath("docs", "01_intake_manifest.md").is_file()
    assert workspace.joinpath("paper", "paper_template.tex").is_file()
    assert "package-resource-test" in workspace.joinpath("docs", "01_intake_manifest.md").read_text(
        encoding="utf-8"
    )


def test_empty_local_template_directory_prevents_partial_workspace(tmp_path):
    # Arrange
    (tmp_path / "templates").mkdir()

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="contains no files"):
        create_candidate(tmp_path, "empty-template-test")
    assert not (tmp_path / "candidate" / "empty-template-test").exists()


def test_default_root_uses_cwd_for_an_installed_package(monkeypatch, tmp_path):
    # Arrange
    installed_module = tmp_path / "site-packages" / "exonym" / "__main__.py"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(cli, "__file__", str(installed_module))
    monkeypatch.chdir(workspace)

    # Act
    default_root = cli._default_repository_root()

    # Assert
    assert default_root == workspace.resolve()
