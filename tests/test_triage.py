"""Synthetic unit tests for engine execution, run manifests, and automated triage."""

from __future__ import annotations

import json
import hashlib
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


def _write_passing_pre_vetting_artifacts(candidate_path: Path, candidate_id: str) -> None:
    """Write minimal candidate-data diagnostic reports for routing tests."""
    outputs = candidate_path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "fixed_ephemeris_screen.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "source": "candidate-data",
                "ephemeris": {"period_days": 2.5},
                "screen": {
                    "primary": {"status": "measured"},
                    "odd_even": {
                        "z": 0.2,
                        "consistent_at_threshold": True,
                        "consistency_threshold_sigma": 3.0,
                    },
                    "half_phase_control": {"depth_significance_sigma": 0.1},
                    "double_period_hypothesis": {
                        "alternating_event": {"depth_significance_sigma": 0.1}
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs / "archival_vetting_report.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "gaia_astrometry": {"validated": True, "query_status": "ok", "ruwe": 1.0},
                "scientific_assessment": {
                    "1_is_hidden_binary": {"answer": False},
                    "2_has_nearby_contaminants": {"answer": False},
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs / "prf_localization_results.json").write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "summary": {
                    "conclusion": "target_dominant_among_modeled_sources",
                    "median_target_to_other_ratio": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs / "stellar_activity_results.json").write_text(
        json.dumps(
            {"source": "candidate-data", "rotation_period_days": 7.0, "rotation_period_std_days": 0.2}
        ),
        encoding="utf-8",
    )
    (outputs / "dilution_sensitivity_results.json").write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "depth_stability": {"interpretation": "stable", "max_variation_relative_to_median": 0.1},
                "contamination": {"availability": "available", "contamination_factor": 0.01},
            }
        ),
        encoding="utf-8",
    )


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
    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-triage-pass")
    cand = load_candidate(tmp_path, "synth-triage-pass")

    _write_passing_pre_vetting_artifacts(candidate_path, cand.candidate_id)
    triage_path = run_automated_triage(cand)
    assert triage_path.is_file()

    triage_data = json.loads(triage_path.read_text(encoding="utf-8"))
    assert triage_data["schema_version"] == 1
    assert triage_data["candidate_id"] == "synth-triage-pass"
    assert triage_data["status"] == "pass"
    assert len(triage_data["records"]) == 5
    evidence = json.loads(
        (candidate_path / "outputs" / "statistical_vetting_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["status"] == "pass"
    assert {item["name"] for item in evidence["diagnostics"]} == {
        "screening", "archive", "localization", "activity", "dilution"
    }

    report = IsolationReport()
    validate_schemas(tmp_path, report)
    assert report.ok, f"Schema violations: {[v.message for v in report.violations]}"


def test_automated_triage_review_required_on_odd_even_anomaly(tmp_path: Path):
    cand_path = _setup_synthetic_workspace(tmp_path, "synth-triage-anomaly")
    cand = load_candidate(tmp_path, "synth-triage-anomaly")
    manifest_path = run_engine(cand, "screen")
    _write_passing_pre_vetting_artifacts(cand_path, cand.candidate_id)

    # Manually plant an odd-even failure in screening outputs
    outputs_dir = cand_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    screen_artifact = {
        "candidate_id": cand.candidate_id,
        "source": "candidate-data",
        "screen": {
            "primary": {"status": "measured"},
            "odd_even": {
                "status": "measured",
                "z": 4.5,
                "consistent_at_threshold": False,
                "consistency_threshold_sigma": 3.0,
            }
        }
    }
    screen_path = cand_path / "outputs" / "fixed_ephemeris_screen.json"
    screen_path.write_text(
        json.dumps(screen_artifact, indent=2), encoding="utf-8"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][0]["sha256"] = hashlib.sha256(screen_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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


def test_engine_run_failure_is_recorded_and_returns_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture):
    create_candidate(tmp_path, "synth-engine-failure", tic="123456789", mission="tess")

    rc = main(["--root", str(tmp_path), "engine", "run", "screen", "synth-engine-failure"])

    assert rc == 1
    output = capsys.readouterr().out.strip()
    manifest_path = tmp_path / output
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"]["code"]


def test_engine_run_with_no_output_is_blocked_without_scientific_artifacts(tmp_path: Path, monkeypatch):
    _setup_synthetic_workspace(tmp_path, "synth-engine-no-output")
    cand = load_candidate(tmp_path, "synth-engine-no-output")
    monkeypatch.setattr("exonym.screening.run_fixed_ephemeris_screen", lambda *args, **kwargs: None)

    manifest_path = run_engine(cand, "screen")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["outputs"] == []
    assert manifest["failure"]["code"] == "no-output-artifacts"
    assert not (cand.path / "outputs" / "fixed_ephemeris_screen.json").exists()
    assert not list((cand.path / "claims").glob("*.json"))


def test_automated_triage_blocks_without_manifest_backed_evidence(tmp_path: Path):
    _setup_synthetic_workspace(tmp_path, "synth-triage-blocked")
    cand = load_candidate(tmp_path, "synth-triage-blocked")

    triage_path = run_automated_triage(cand)

    triage_data = json.loads(triage_path.read_text(encoding="utf-8"))
    assert triage_data["status"] == "blocked"
    assert triage_data["records"][0]["status"] == "blocked"


def test_activity_harmonic_requires_human_review(tmp_path: Path):
    from exonym.statistical_vetting import build_statistical_vetting_evidence

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-activity-harmonic")
    cand = load_candidate(tmp_path, "synth-activity-harmonic")
    _write_passing_pre_vetting_artifacts(candidate_path, cand.candidate_id)
    (candidate_path / "outputs" / "stellar_activity_results.json").write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "rotation_period_days": 5.0,
                "rotation_period_std_days": 0.2,
            }
        ),
        encoding="utf-8",
    )

    evidence = json.loads(build_statistical_vetting_evidence(cand).read_text(encoding="utf-8"))

    activity = next(record for record in evidence["diagnostics"] if record["name"] == "activity")
    assert activity["status"] == "review-required"
    assert evidence["status"] == "review-required"


def test_vetting_readiness_refuses_review_required_evidence_without_claims(tmp_path: Path):
    from exonym.statistical_vetting import require_vetting_readiness

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-vet-review")
    cand = load_candidate(tmp_path, "synth-vet-review")
    run_engine(cand, "screen")
    _write_passing_pre_vetting_artifacts(candidate_path, cand.candidate_id)
    localization = candidate_path / "outputs" / "prf_localization_results.json"
    localization.write_text(
        json.dumps({"source": "candidate-data", "summary": {"conclusion": "inconclusive_no_competing_sources_modeled"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="current routing is review-required"):
        require_vetting_readiness(cand)

    triage = json.loads((candidate_path / "decisions" / "automated_triage.json").read_text(encoding="utf-8"))
    assert triage["status"] == "review-required"
    assert not list((candidate_path / "claims").glob("*.json"))


def test_decisive_rejection_prohibits_triceratops(tmp_path: Path):
    from exonym.statistical_vetting import record_decisive_rejection, require_vetting_readiness

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-vet-rejection")
    cand = load_candidate(tmp_path, "synth-vet-rejection")
    evidence = candidate_path / "outputs" / "decisive.json"
    evidence.write_text('{"result": "synthetic rejection"}\n', encoding="utf-8")
    record_decisive_rejection(cand, "Synthetic decisive alias.", "outputs/decisive.json")

    with pytest.raises(RuntimeError, match="decisive rejection"):
        require_vetting_readiness(cand)


def test_decisive_rejection_rejects_evidence_outside_the_candidate_workspace(tmp_path: Path):
    from exonym.statistical_vetting import record_decisive_rejection

    candidate_path = _setup_synthetic_workspace(tmp_path, "synth-rejection-boundary")
    cand = load_candidate(tmp_path, "synth-rejection-boundary")
    outside_evidence = tmp_path / "outside.json"
    outside_evidence.write_text('{"synthetic": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="inside the candidate workspace"):
        record_decisive_rejection(cand, "Synthetic boundary check.", "../outside.json")

    assert not (candidate_path / "decisions" / "decisive_rejection.json").exists()
