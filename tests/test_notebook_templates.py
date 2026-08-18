import json
from pathlib import Path

from exonym.workspace import create_candidate


def test_notebook_template_is_valid_target_neutral_notebook_and_is_mirrored(tmp_path):
    # Arrange
    repository_root = Path(__file__).resolve().parents[1]
    source = repository_root / "templates" / "notebooks" / "evidence_review.ipynb"
    bundled = (
        repository_root
        / "src"
        / "exonym"
        / "_resources"
        / "templates"
        / "notebooks"
        / "evidence_review.ipynb"
    )

    # Act
    notebook = json.loads(source.read_text(encoding="utf-8"))
    candidate = create_candidate(tmp_path, "notebook-template-test")

    # Assert
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    assert source.read_text(encoding="utf-8") == bundled.read_text(encoding="utf-8")
    assert candidate.path.joinpath("notebooks", "evidence_review.ipynb").is_file()
