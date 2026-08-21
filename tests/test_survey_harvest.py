"""Tests for streamed survey harvesting and bounded autonomous execution."""

import json

import pytest

from exonym.autonomous import auto_vet_candidate
from exonym.gatekeeper import _gate_novelty_audit
from exonym.survey import create_survey, load_survey
from exonym.survey_harvest import TceFilters, harvest_tces
from exonym.workspace import create_candidate, load_candidate


def _filters():
    return TceFilters(20.0, 1.0, 15.0, 200.0, 1500.0, 1.2, 3.5, 1.3, 12.5)


def test_harvest_streams_rows_and_provisions_only_eligible_candidate(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "harvest-test", "tess", [17])
    source = tmp_path / "tces.csv"
    source.write_text(
        "tic_id,snr,period_days,depth_ppm,planet_radius_earth,stellar_radius_solar,tmag\n"
        "123456789,22,3.2,700,2.1,1.0,10.5\n"
        "987654321,8,3.2,700,2.1,1.0,10.5\n",
        encoding="utf-8",
    )

    def transport(url, timeout):
        assert url.startswith("https://")
        if "target.php" in url:
            return b"{}"
        return b"tic_id\n"

    # Act
    outcomes = harvest_tces(survey, str(source), _filters(), 5, transport=transport)

    # Assert
    assert outcomes == [{"candidate_id": "tce-123456789", "status": "registered"}]
    workspace = tmp_path / "candidate" / "tce-123456789"
    audit = json.loads((workspace / "decisions" / "novelty_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "eligible"
    assert len(audit["evidence"]) == 3
    assert (workspace / "data" / "external" / "novelty").is_dir()
    assert _gate_novelty_audit(load_candidate(tmp_path, "tce-123456789"))[0]
    assert load_survey(tmp_path, "harvest-test").path.joinpath("targets", "tce-123456789", "target.json").is_file()


def test_harvest_does_not_provision_when_a_registry_is_unavailable(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "harvest-test", "tess", [17])
    source = tmp_path / "tces.csv"
    source.write_text(
        "tic_id,snr,period_days,depth_ppm,planet_radius_earth,stellar_radius_solar,tmag\n"
        "123456789,22,3.2,700,2.1,1.0,10.5\n",
        encoding="utf-8",
    )

    # Act
    outcomes = harvest_tces(
        survey,
        str(source),
        _filters(),
        5,
        transport=lambda *_args: (_ for _ in ()).throw(OSError("offline")),
    )

    # Assert
    assert outcomes[0]["status"] == "unavailable"
    assert not (tmp_path / "candidate" / "tce-123456789").exists()


def test_auto_vet_records_blocked_steps_without_state_or_claim_changes(tmp_path, monkeypatch):
    # Arrange
    candidate = create_candidate(tmp_path, "automation-test", tic="123456789", mission="tess")

    def write_output(name):
        def operation(workspace, *args, **kwargs):
            path = workspace.path / "outputs" / name
            path.write_text("{}\n", encoding="utf-8")
            return path

        return operation

    monkeypatch.setattr("exonym.screening.run_fixed_ephemeris_screen", write_output("screen.json"))
    monkeypatch.setattr("exonym.archive.run_archival_vetting", write_output("archive.json"))
    monkeypatch.setattr("exonym.localization.run_prf_localization", write_output("localization.json"))
    monkeypatch.setattr("exonym.activity.run_stellar_activity", write_output("activity.json"))
    monkeypatch.setattr("exonym.asteroseismology.run_asteroseismology", write_output("astero.json"))
    monkeypatch.setattr("exonym.sed.run_sed_fit", write_output("sed.json"))
    monkeypatch.setattr("exonym.dilution.run_dilution_sensitivity", write_output("dilution.json"))
    monkeypatch.setattr("exonym.transit_fit.run_mcmc_transit_fit", write_output("fit.json"))
    monkeypatch.setattr("exonym.ttv.run_ttv_analysis", write_output("ttv.json"))
    monkeypatch.setattr("exonym.phasecurve.run_phase_curve_search", write_output("phasecurve.json"))
    monkeypatch.setattr("exonym.plotting.generate_candidate_plots", lambda workspace: [write_output("plot.json")(workspace)])
    monkeypatch.setattr("exonym.engines.run_automated_triage", write_output("triage.json"))
    monkeypatch.setattr("exonym.vetting.tricera_parse.run_triceratops_simulation", write_output("triceratops.json"))

    # Act
    manifest_path = auto_vet_candidate(candidate, download=False)

    # Assert
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["automation"]["claim_eligible"] is False
    assert manifest["automation"]["disposition_changed"] is False
    assert manifest["automation"]["workflow_advanced"] is False
    assert next(step for step in manifest["automation"]["steps"] if step["name"] == "search")["status"] == "blocked"
    assert candidate.metadata["workflow"]["phase"] == "intake"
    assert candidate.metadata["scientific_disposition"] == "unknown"


@pytest.mark.parametrize("value", (0, -1, True))
def test_harvest_rejects_invalid_limit(tmp_path, value):
    survey = create_survey(tmp_path, "harvest-test", "tess", [17])

    with pytest.raises(ValueError, match="max_candidates"):
        harvest_tces(survey, "missing.csv", _filters(), value)
