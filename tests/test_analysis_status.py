"""Tests for candidate-local analysis coverage records."""

import json

from exonym.analysis_status import build_analysis_status
from exonym.workspace import create_candidate


def test_analysis_status_marks_complete_when_every_stage_has_an_artifact(tmp_path):
    workspace = create_candidate(tmp_path, "analysis-status-complete")
    paths = (
        "data/raw/synthetic_lc.fits",
        "data/processed/detrended-running-median.npz",
        "outputs/bls_search_results.json",
        "outputs/fixed_ephemeris_screen.json",
        "outputs/archival_vetting_report.json",
        "outputs/prf_localization_results.json",
        "outputs/stellar_activity_results.json",
        "outputs/sed_fit_results.json",
        "outputs/dilution_sensitivity_results.json",
        "outputs/mcmc_transit_fit.json",
        "outputs/ttv_analysis_results.json",
        "outputs/phase_curve_results.json",
        "decisions/automated_triage.json",
        "decisions/triceratops_vetting_decision.json",
    )
    for relative in paths:
        path = workspace.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic artifact")

    status_path = build_analysis_status(workspace)

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_path == workspace.path / "decisions" / "analysis_completion.json"
    assert payload["candidate_id"] == workspace.candidate_id
    assert payload["overall_status"] == "complete"
    assert payload["claim_eligible"] is False
    assert {stage["status"] for stage in payload["stages"]} == {"succeeded"}
    assert all(stage["evidence"][0]["sha256"] for stage in payload["stages"])


def test_analysis_status_marks_missing_stages_unavailable_without_scientific_inference(tmp_path):
    workspace = create_candidate(tmp_path, "analysis-status-partial")

    status_path = build_analysis_status(workspace)

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "partial"
    assert payload["claim_eligible"] is False
    assert "not scientific null results" in payload["scientific_interpretation"]
    assert all(stage["status"] == "unavailable" for stage in payload["stages"])
    assert all(stage["evidence"] == [] for stage in payload["stages"])
