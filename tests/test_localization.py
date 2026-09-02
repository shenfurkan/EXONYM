import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from exonym.workspace import create_candidate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_calibrated_assets(workspace, fully_declared: bool = False) -> Path:
    fits = pytest.importorskip("astropy.io.fits")
    external = workspace.path / "data" / "external"
    external.mkdir(parents=True, exist_ok=True)
    prf_path = external / "tess_prf.fits"
    fits.PrimaryHDU(
        data=np.array(
            [[0.0, 0.1, 0.0], [0.1, 1.0, 0.1], [0.0, 0.1, 0.0]], dtype=float
        )
    ).writeto(prf_path, overwrite=True)
    digest = _sha256(prf_path)
    manifest_path = external / "tess_prf.manifest.json"
    if fully_declared:
        recovery_source = external / "prf_recovery_source.json"
        recovery_source.write_text(json.dumps({"fixture": "structural-contract"}), encoding="utf-8")
        manifest_payload = {
            "schema_version": 1,
            "candidate_id": workspace.candidate_id,
            "mission": "TESS",
            "sector": 1,
            "camera": 1,
            "ccd": 1,
            "field_position": {"column": 100.0, "row": 200.0},
            "prf_sha256": digest,
            "provenance": {
                "source": "TESS calibration archive",
                "official_uri": "https://archive.stsci.edu/missions/tess/doc/tess-prf-models.html",
                "retrieved_utc": "2025-01-01T00:00:00Z",
            },
        }
        recovery_payload = {
            "schema_version": 1,
            "candidate_id": workspace.candidate_id,
            "mission": "TESS",
            "prf_sha256": digest,
            "recovery_passed": True,
            "recovery_results": [{"injected_flux": 5.0, "recovered_flux": 5.0}],
            "source_artifacts": [
                {"path": "data/external/prf_recovery_source.json", "sha256": _sha256(recovery_source), "role": "injected-scene"}
            ],
        }
    else:
        manifest_payload = {"prf_sha256": digest, "provenance": {"source": "synthetic"}}
        recovery_payload = {"prf_sha256": digest, "recovery_passed": True}
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    (external / "tess_prf.recovery_calibration.json").write_text(json.dumps(recovery_payload), encoding="utf-8")
    return prf_path


def test_calibrated_prf_assets_require_fully_declared_mission_contract(tmp_path: Path):
    from exonym.localization import calibrated_prf_assets

    workspace = create_candidate(tmp_path, "synthetic-localization")
    prf_path = _write_calibrated_assets(workspace)

    evidence, error = calibrated_prf_assets(workspace)

    assert evidence is None
    assert "required property" in error
    prf_path = _write_calibrated_assets(workspace, fully_declared=True)
    evidence, error = calibrated_prf_assets(workspace)

    assert error is None
    assert evidence["prf_template"] == {
        "path": "data/external/tess_prf.fits",
        "sha256": _sha256(prf_path),
    }
    manifest_path = workspace.path / "data" / "external" / "tess_prf.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["source"] = "synthetic"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert calibrated_prf_assets(workspace)[0] is None
    _write_calibrated_assets(workspace, fully_declared=True)
    prf_path.write_bytes(prf_path.read_bytes() + b"changed")
    assert calibrated_prf_assets(workspace)[0] is None


def test_calibrated_prf_nnls_recovers_empirical_template_amplitudes(tmp_path: Path):
    from exonym.localization import (
        calibrated_prf_kernel,
        fit_calibrated_difference_image_prf,
        load_calibrated_prf_template,
    )

    workspace = create_candidate(tmp_path, "synthetic-localization-nnls")
    prf_path = _write_calibrated_assets(workspace)
    template = load_calibrated_prf_template(prf_path)
    yy, xx = np.indices((13, 13), dtype=float)
    injected = np.array([8.0, 3.0])
    difference_image = (
        injected[0] * calibrated_prf_kernel(xx, yy, 5.25, 6.0, template)
        + injected[1] * calibrated_prf_kernel(xx, yy, 8.0, 6.0, template)
    )

    amplitudes, residual, n_pixels = fit_calibrated_difference_image_prf(
        difference_image, np.ones_like(difference_image, dtype=bool), [5.25, 8.0], [6.0, 6.0], template
    )

    assert n_pixels == difference_image.size
    assert amplitudes == pytest.approx(injected, rel=0.02)
    assert residual == pytest.approx(0.0, abs=1e-8)


def test_runner_keeps_fabricated_recovery_and_missing_wcs_uncalibrated(tmp_path: Path, monkeypatch):
    from exonym.localization import (
        calibrated_prf_kernel,
        load_calibrated_prf_template,
        run_prf_localization,
    )

    workspace = create_candidate(tmp_path, "synthetic-localization-runner")
    prf_path = _write_calibrated_assets(workspace, fully_declared=True)
    template = load_calibrated_prf_template(prf_path)
    yy, xx = np.indices((7, 7), dtype=float)
    difference = 5.0 * calibrated_prf_kernel(xx, yy, 3.0, 3.0, template)
    ephemeris = {
        "period_days": 1.5,
        "epoch_btjd": 1.0,
        "duration_days": 0.1,
        "source": "candidate-data",
        "field_sources": {
            "period_days": "candidate-data",
            "epoch_btjd": "candidate-data",
            "duration_days": "candidate-data",
        },
    }
    monkeypatch.setattr("exonym.localization.load_transit_ephemeris", lambda _: ephemeris)
    monkeypatch.setattr("exonym.localization._load_archival_gaia_neighbors", lambda _: ([], {}))
    monkeypatch.setattr(
        "exonym.localization.load_tpf_cubes",
        lambda *_args, **_kwargs: [{"path": tmp_path / "missing-tpf.fits", "sector": 1, "header": {"SECTOR": 1, "CAMERA": 1, "CCD": 1}}],
    )
    monkeypatch.setattr(
        "exonym.localization.extract_tpf_difference_image",
        lambda *_args: (difference, np.ones_like(difference, dtype=bool), 3.0, 3.0, 12, 12),
    )

    report = json.loads(run_prf_localization(workspace).read_text(encoding="utf-8"))

    assert report["source"] == "candidate-data"
    assert report["calibrated"] is False
    assert report["calibration_status"] == "uncalibrated"
    assert report["summary"]["conclusion"] == "inconclusive_wcs_unavailable"
    assert report["sector_results"][0]["source_assignment_interpretable"] is False
    assert report["sector_results"][0]["target_difference_flux_amplitude"] == pytest.approx(5.0)


def test_localization_keeps_projected_ra_and_safely_deprojects_coordinate_ra():
    from exonym.localization import localize_difference_image

    difference = np.array([[0.2, 0.8, 0.2], [0.8, 2.0, 0.8], [0.2, 0.8, 0.2]])
    result = localize_difference_image(
        difference, np.ones_like(difference, dtype=bool), 0.0, 1.0, cos_dec=0.0
    )

    assert result["ra_cosdec_offset_arcsec"] == result["ra_offset_arcsec"]
    assert np.isfinite(result["ra_coordinate_offset_arcsec"])


def test_localization_uses_wcs_tangent_offsets_without_double_cosine():
    from exonym.localization import localize_difference_image

    class RotatedWcs:
        def pixel_to_world_values(self, x, y):
            # A synthetic 90-degree detector roll: columns map to Dec, rows to RA.
            return 30.0 + y / 3600.0, 70.0 + x / 3600.0

    difference = np.array([[0.1, 0.4, 0.1], [0.4, 2.0, 0.4], [0.1, 0.4, 0.1]])
    result = localize_difference_image(
        difference, np.ones_like(difference, dtype=bool), 0.0, 1.0, wcs=RotatedWcs()
    )

    assert result["coordinate_method"] == "fits-wcs-spherical-tangent-offset"
    assert result["dec_offset_arcsec"] > 0.0


def test_source_selection_keeps_all_projected_catalog_neighbors():
    from exonym.localization import _select_sources

    class IdentityWcs:
        @staticmethod
        def world_to_pixel_values(ra, dec):
            return ra, dec

    neighbors = [
        {
            "source_id": "synthetic-{}".format(index),
            "ra": float(index),
            "dec": 1.0,
            "g_mag": 15.0 + index,
            "flux_ratio": None,
            "separation_arcsec": 1.0,
            "is_target": False,
        }
        for index in range(8)
    ]

    sources, _, _ = _select_sources(
        np.ones((4, 12)), np.ones((4, 12), dtype=bool), 0.0, 1.0,
        neighbors, 10.0, IdentityWcs(), 1.0,
    )

    assert len(sources) == 9
    assert all(source["flux_ratio"] is None for source in sources[1:])
