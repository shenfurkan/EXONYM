import json

import pytest

from exonym.ds9 import export_ds9_regions


def _workspace(tmp_path):
    return type("Workspace", (), {"path": tmp_path, "candidate_id": "ds9-test"})()


def _write_archival_report(workspace, sources):
    outputs = workspace.path / "outputs"
    outputs.mkdir()
    outputs.joinpath("archival_vetting_report.json").write_text(
        json.dumps(
            {
                "candidate_id": workspace.candidate_id,
                "target_coordinates": {"ra_deg": 99.0, "dec_deg": 88.0},
                "gaia_astrometry": {
                    "validated": True,
                    "query_status": "ok",
                    "target_source_id": "synthetic-target",
                    "sources": sources,
                },
            }
        ),
        encoding="utf-8",
    )


def test_export_ds9_regions_uses_archival_coordinates_and_localization_labels(tmp_path):
    # Arrange
    workspace = _workspace(tmp_path)
    _write_archival_report(
        workspace,
        [
            {"source_id": "synthetic-target", "ra_deg": 10.0, "dec_deg": -20.0},
            {"source_id": "synthetic-neighbor", "ra_deg": 10.01, "dec_deg": -20.01},
            {"source_id": "missing-coordinate", "ra_deg": None, "dec_deg": -20.02},
        ],
    )
    workspace.path.joinpath("outputs", "prf_localization_results.json").write_text(
        json.dumps(
            {
                "sector_results": [
                    {"skipped": False, "fit_dominant_source_id": "synthetic-neighbor"}
                ]
            }
        ),
        encoding="utf-8",
    )

    # Act
    result = export_ds9_regions(workspace)

    # Assert
    text = result.read_text(encoding="utf-8")
    assert result == workspace.path / "figures" / "ds9_sources.reg"
    assert "point(10.00000000,-20.00000000)" in text
    assert "point(10.01000000,-20.01000000)" in text
    assert "archive-target" in text
    assert "prf-fit-dominant" in text
    assert "99.00000000" not in text
    assert "missing-coordinate" not in text


def test_export_ds9_regions_requires_validated_archival_source_coordinates(tmp_path):
    # Arrange
    workspace = _workspace(tmp_path)
    _write_archival_report(
        workspace,
        [{"source_id": "synthetic-target", "ra_deg": "not-a-coordinate", "dec_deg": None}],
    )

    # Act / Assert
    with pytest.raises(ValueError, match="finite sky coordinates"):
        export_ds9_regions(workspace)
    assert not workspace.path.joinpath("figures", "ds9_sources.reg").exists()
