"""Synthetic unit tests for engine execution, run manifests, and automated triage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import lightkurve as lk

from exonym.__main__ import main
from exonym.engines import report_candidate_engines, run_automated_triage, run_engine
from exonym.isolation import IsolationReport
from exonym.schemas import validate_schemas
from exonym.workspace import create_candidate, load_candidate


def _write_synthetic_lc_fits(path: Path, sector: int = 1, period: float = 2.5, depth_ppm: float = 1200.0) -> None:
    """Write a synthetic light curve FITS file with a clean transit box."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    time = np.arange(100.0, 125.0, 2.0 / 1440.0)
    flux = 1.0 + rng.normal(0.0, 100e-6, time.size)
    phase = ((time - 100.5) % period) / period
    in_tr = (phase < 0.02) | (phase > 0.98)
    flux[in_tr] -= depth_ppm * 1e-6
    err = np.full_like(time, 100e-6)

    meta = {
        "MISSION": "TESS",
        "TELESCOP": "TESS",
        "SECTOR": sector,
        "TIMEDEL": 120.0 / 86400.0,
        "TIMEUNIT": "BJD",
        "BJDREFI": 2457000,
        "BJDREFF": 0.0,
    }
    lk.LightCurve(time=time, flux=flux, flux_err=err, meta=meta).to_fits(
        path=path, overwrite=True
    )


def _setup_synthetic_workspace(tmp_path: Path, candidate_id: str = "synth-triage-target") -> Path:
    """Create a fully populated synthetic candidate workspace with light curve and ephemeris."""
    workspace = create_candidate(
        repository_root=tmp_path,
        candidate_id=candidate_id,
        tic="987654321",
        mission="tess",
    )
    cand_path = workspace.path

    # 1. Ephemeris config
    ephem_payload = {
        "period_days": 2.5,
        "epoch_btjd": 100.5,
        "duration_days": 0.1,
        "depth_ppm": 1200.0,
    }
    (cand_path / "config" / "transit_config.json").write_text(
        json.dumps(ephem_payload, indent=2), encoding="utf-8"
    )

    # 2. Synthetic light curve FITS
    _write_synthetic_lc_fits(cand_path / "data" / "processed" / "s0001_lc.fits", sector=1)

    return cand_path


def test_engine_run_generates_valid_schema_manifest(tmp_path: Path):
    _setup_synthetic_workspace(tmp_path, "synth-engine-manifest")
    cand = load_candidate(tmp_path, "synth-engine-manifest")

    manifest_path = run_engine(cand, "screen")
    assert manifest_path.is_file()

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["schema_version"] == 1
    assert manifest_data["candidate_id"] == "synth-engine-manifest"
    assert manifest_data["engine"] == "screen"
    assert manifest_data["status"] == "succeeded"
    assert len(manifest_data["inputs"]) >= 1
    assert len(manifest_data["outputs"]) >= 1

    # Verify that schemas.py validates the workspace with 0 violations
    report = IsolationReport()
    validate_schemas(tmp_path, report)
    assert report.ok, f"Schema violations: {[v.message for v in report.violations]}"


def test_engine_report_lists_completed_runs(tmp_path: Path):
    _setup_synthetic_workspace(tmp_path, "synth-report-candidate")
    cand = load_candidate(tmp_path, "synth-report-candidate")

    run_engine(cand, "screen")
    runs = report_candidate_engines(cand)
    assert len(runs) == 1
    assert runs[0]["engine"] == "screen"
    assert runs[0]["status"] == "succeeded"


def test_automated_triage_pass_verdict(tmp_path: Path):
    _setup_synthetic_workspace(tmp_path, "synth-triage-pass")
    cand = load_candidate(tmp_path, "synth-triage-pass")

    triage_path = run_automated_triage(cand)
    assert triage_path.is_file()

    triage_data = json.loads(triage_path.read_text(encoding="utf-8"))
    assert triage_data["schema_version"] == 1
    assert triage_data["candidate_id"] == "synth-triage-pass"
    assert triage_data["status"] == "pass"
    assert len(triage_data["records"]) >= 1

    report = IsolationReport()
    validate_schemas(tmp_path, report)
    assert report.ok, f"Schema violations: {[v.message for v in report.violations]}"


def test_automated_triage_review_required_on_odd_even_anomaly(tmp_path: Path):
    cand_path = _setup_synthetic_workspace(tmp_path, "synth-triage-anomaly")
    cand = load_candidate(tmp_path, "synth-triage-anomaly")

    # Manually plant an odd-even failure in screening outputs
    outputs_dir = cand_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    screen_artifact = {
        "screen": {
            "odd_even": {
                "status": "measured",
                "z": 4.5,
                "consistent_at_threshold": False,
            }
        }
    }
    (outputs_dir / "fixed_ephemeris_screening.json").write_text(
        json.dumps(screen_artifact, indent=2), encoding="utf-8"
    )

    triage_path = run_automated_triage(cand)
    triage_data = json.loads(triage_path.read_text(encoding="utf-8"))
    assert triage_data["status"] == "review-required"
    assert any(r["status"] == "review-required" for r in triage_data["records"])

    report = IsolationReport()
    validate_schemas(tmp_path, report)
    assert report.ok, f"Schema violations: {[v.message for v in report.violations]}"


def test_cli_engine_run_and_triage(tmp_path: Path, capsys: pytest.CaptureFixture):
    _setup_synthetic_workspace(tmp_path, "synth-cli-candidate")

    # Test 'exonym engine run'
    rc_run = main(["--root", str(tmp_path), "engine", "run", "screen", "synth-cli-candidate"])
    assert rc_run == 0
    captured_run = capsys.readouterr().out
    assert "engine-run.json" in captured_run

    # Test 'exonym engine report'
    rc_rep = main(["--root", str(tmp_path), "engine", "report", "synth-cli-candidate"])
    assert rc_rep == 0
    captured_rep = capsys.readouterr().out
    assert "screen" in captured_rep

    # Test 'exonym triage'
    rc_tri = main(["--root", str(tmp_path), "triage", "synth-cli-candidate"])
    assert rc_tri == 0
    captured_tri = capsys.readouterr().out
    assert "automated_triage.json" in captured_tri
