"""Tests for streamed survey harvesting and bounded autonomous execution."""

import json
from contextlib import contextmanager
import hashlib
from pathlib import Path
from urllib.error import URLError

import pytest

from exonym import __version__
from exonym.autonomous import (
    _available_common_sectors,
    _select_download_sectors,
    auto_vet_candidate,
    record_autonomous_incident,
)
from exonym.gatekeeper import _gate_novelty_audit
from exonym.survey import create_survey, load_survey
from exonym.survey_harvest import (
    TceFilters,
    evaluate_live_novelty,
    harvest_tces,
    novelty_provider_urls,
    novelty_response_has_registration,
    stream_tce_rows,
)
from exonym.workspace import create_candidate, load_candidate


def _filters():
    return TceFilters(20.0, 1.0, 15.0, 200.0, 1500.0, 1.2, 3.5, 1.3, 12.5)


def _eligible_exofop_response(tic="123456789"):
    return json.dumps(
        {
            "basic_info": {"tic_id": tic},
            "tois": [],
            "ctois": [],
            "planet_parameters": [],
        }
    ).encode("utf-8")


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
            return _eligible_exofop_response()
        if "from+toi" in url:
            return b"toi,tid\n"
        return b"pl_name,tic_id\n"

    # Act
    outcomes = harvest_tces(survey, str(source), _filters(), 5, transport=transport)

    # Assert
    assert outcomes == [{"candidate_id": "tce-123456789", "status": "registered"}]
    workspace = tmp_path / "candidate" / "tce-123456789"
    audit = json.loads((workspace / "decisions" / "novelty_audit.json").read_text(encoding="utf-8"))
    assert audit["schema_version"] == 2
    assert audit["status"] == "eligible"
    assert len(audit["evidence"]) == 3
    assert {entry["provider"] for entry in audit["evidence"]} == {
        "nasa-toi",
        "nasa-confirmed",
        "exofop",
    }
    for entry in audit["evidence"]:
        response_path = workspace / entry["response_path"]
        assert response_path.is_file()
        assert hashlib.sha256(response_path.read_bytes()).hexdigest() == entry["evidence_sha256"]
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


def test_harvest_does_not_count_unavailable_novelty_results_toward_candidate_cap(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "harvest-test", "tess", [17])
    source = tmp_path / "tces.csv"
    source.write_text(
        "tic_id,snr,period_days,depth_ppm,planet_radius_earth,stellar_radius_solar,tmag\n"
        "123456789,22,3.2,700,2.1,1.0,10.5\n"
        "987654321,22,3.2,700,2.1,1.0,10.5\n",
        encoding="utf-8",
    )

    def transport(url, _timeout):
        if "123456789" in url:
            raise OSError("synthetic registry outage")
        if "target.php" in url:
            return _eligible_exofop_response("987654321")
        return b"toi,tid\n" if "from+toi" in url else b"pl_name,tic_id\n"

    # Act
    outcomes = harvest_tces(survey, str(source), _filters(), 1, transport=transport)

    # Assert
    assert [outcome["status"] for outcome in outcomes] == ["unavailable", "registered"]
    assert outcomes[1]["candidate_id"] == "tce-987654321"


def test_harvest_marks_incomplete_existing_workspace_as_rollback_leftover(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "harvest-test", "tess", [17])
    create_candidate(
        tmp_path,
        "tce-123456789",
        tic="123456789",
        mission="tess",
        tags=["survey-harvest"],
    )
    source = tmp_path / "tces.csv"
    source.write_text(
        "tic_id,snr,period_days,depth_ppm,planet_radius_earth,stellar_radius_solar,tmag\n"
        "123456789,22,3.2,700,2.1,1.0,10.5\n",
        encoding="utf-8",
    )

    def transport(url, _timeout):
        if "target.php" in url:
            return _eligible_exofop_response()
        return b"toi,tid\n" if "from+toi" in url else b"pl_name,tic_id\n"

    # Act
    outcomes = harvest_tces(survey, str(source), _filters(), 1, transport=transport)

    # Assert
    assert outcomes == [
        {
            "candidate_id": "tce-123456789",
            "status": "rollback-leftover",
            "reason": (
                "Existing candidate workspace has no valid matching survey target "
                "record and requires operator inspection."
            ),
        }
    ]


def test_harvest_reports_completed_existing_registration_as_already_provisioned(tmp_path):
    # Arrange
    survey = create_survey(tmp_path, "harvest-test", "tess", [17])
    source = tmp_path / "tces.csv"
    source.write_text(
        "tic_id,snr,period_days,depth_ppm,planet_radius_earth,stellar_radius_solar,tmag\n"
        "123456789,22,3.2,700,2.1,1.0,10.5\n",
        encoding="utf-8",
    )

    def transport(url, _timeout):
        if "target.php" in url:
            return _eligible_exofop_response()
        return b"toi,tid\n" if "from+toi" in url else b"pl_name,tic_id\n"

    # Act
    first_outcomes = harvest_tces(survey, str(source), _filters(), 1, transport=transport)
    second_outcomes = harvest_tces(survey, str(source), _filters(), 1, transport=transport)

    # Assert
    assert first_outcomes == [{"candidate_id": "tce-123456789", "status": "registered"}]
    assert second_outcomes == [
        {"candidate_id": "tce-123456789", "status": "already-provisioned"}
    ]


@pytest.mark.parametrize(
    ("nasa_response", "exofop_response"),
    [
        (b"<html>temporary gateway error</html>", _eligible_exofop_response()),
        (b"toi,tid,pl_name,tic_id\n", b"{}"),
    ],
)
def test_live_novelty_rejects_malformed_success_bodies(nasa_response, exofop_response):
    def transport(url, _timeout):
        return exofop_response if "target.php" in url else nasa_response

    result = evaluate_live_novelty("123456789", transport=transport)

    assert result.status == "unavailable"


@pytest.mark.parametrize(
    "payload",
    (
        b"toi,tid,tid\n",
        b"toi,tid,extra\n",
        b"toi,tid\n1,2,unexpected\n",
    ),
)
def test_nasa_novelty_response_rejects_ambiguous_or_unrequested_columns(payload):
    source_uri = dict(novelty_provider_urls("123456789"))["nasa-toi"]

    with pytest.raises(ValueError, match="NASA Archive response"):
        novelty_response_has_registration("nasa-toi", source_uri, payload, "123456789")


def test_exofop_nonempty_registration_and_ambiguous_tic_keys_fail_closed():
    source_uri = dict(novelty_provider_urls("123456789"))["exofop"]
    registered = json.dumps(
        {
            "basic_info": {"tic_id": "123456789"},
            "tois": [{}],
            "ctois": [],
            "planet_parameters": [],
        }
    ).encode("utf-8")
    ambiguous = json.dumps(
        {
            "basic_info": {"tic_id": "123456789", "tic-id": "123456789"},
            "tois": [],
            "ctois": [],
            "planet_parameters": [],
        }
    ).encode("utf-8")

    assert novelty_response_has_registration("exofop", source_uri, registered, "123456789")
    with pytest.raises(ValueError, match="ambiguous basic_info"):
        novelty_response_has_registration("exofop", source_uri, ambiguous, "123456789")


def test_harvest_rolls_back_a_new_candidate_when_audit_write_fails(tmp_path, monkeypatch):
    survey = create_survey(tmp_path, "harvest-test", "tess", [17])
    source = tmp_path / "tces.csv"
    source.write_text(
        "tic_id,snr,period_days,depth_ppm,planet_radius_earth,stellar_radius_solar,tmag\n"
        "123456789,22,3.2,700,2.1,1.0,10.5\n",
        encoding="utf-8",
    )

    def transport(url, _timeout):
        if "target.php" in url:
            return _eligible_exofop_response()
        return b"toi,tid\n" if "from+toi" in url else b"pl_name,tic_id\n"

    monkeypatch.setattr(
        "exonym.survey_harvest.write_novelty_audit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic write failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic write failure"):
        harvest_tces(survey, str(source), _filters(), 5, transport=transport)

    assert not (tmp_path / "candidate" / "tce-123456789").exists()
    assert not (survey.path / "targets" / "tce-123456789").exists()


def test_harvest_surfaces_rollback_leftovers_when_cleanup_cannot_remove_workspace(
    tmp_path, monkeypatch
):
    # Arrange
    from exonym import survey_harvest

    survey = create_survey(tmp_path, "harvest-test", "tess", [17])
    source = tmp_path / "tces.csv"
    source.write_text(
        "tic_id,snr,period_days,depth_ppm,planet_radius_earth,stellar_radius_solar,tmag\n"
        "123456789,22,3.2,700,2.1,1.0,10.5\n",
        encoding="utf-8",
    )

    def transport(url, _timeout):
        if "target.php" in url:
            return _eligible_exofop_response()
        return b"toi,tid\n" if "from+toi" in url else b"pl_name,tic_id\n"

    workspace_path = tmp_path / "candidate" / "tce-123456789"
    monkeypatch.setattr(
        "exonym.survey_harvest.write_novelty_audit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic write failure")),
    )
    original_rmtree = survey_harvest.shutil.rmtree

    def leave_new_workspace(path):
        if Path(path) == workspace_path:
            return
        original_rmtree(path)

    monkeypatch.setattr("exonym.survey_harvest.shutil.rmtree", leave_new_workspace)

    # Act / Assert
    with pytest.raises(RuntimeError, match="rollback left incomplete paths"):
        harvest_tces(survey, str(source), _filters(), 1, transport=transport)

    assert workspace_path.is_dir()


def test_live_catalog_transport_retries_transient_https_failure(monkeypatch):
    from exonym import survey_harvest

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def read():
            return b"tic_id\n"

    calls = []

    def fake_urlopen(request, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise URLError("temporary outage")
        return Response()

    monkeypatch.setattr(survey_harvest.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(survey_harvest.time, "sleep", lambda _: None)

    assert survey_harvest._https_bytes("https://example.invalid/catalog.csv", 12.0) == b"tic_id\n"
    assert calls == [12.0, 12.0]


def test_auto_vet_records_blocked_steps_without_state_or_claim_changes(tmp_path, monkeypatch):
    # Arrange
    candidate = create_candidate(tmp_path, "automation-test", tic="123456789", mission="tess")

    fit_samples = []

    def write_output(name):
        def operation(workspace, *args, **kwargs):
            if name == "fit.json":
                fit_samples.append(kwargs.get("n_samples"))
            path = workspace.path / "outputs" / name
            path.write_text("{}\n", encoding="utf-8")
            return path

        return operation

    def unexpected_screen_failure(*_args, **_kwargs):
        raise TypeError("synthetic unexpected screen failure")

    monkeypatch.setattr("exonym.screening.run_fixed_ephemeris_screen", unexpected_screen_failure)
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
    manifest_path = auto_vet_candidate(candidate, sectors=[5, 2, 5], download=False)

    # Assert
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["automation"]["claim_eligible"] is False
    assert manifest["automation"]["disposition_changed"] is False
    assert manifest["automation"]["workflow_advanced"] is False
    assert manifest["automation"]["sectors_used"] == [2, 5]
    assert next(step for step in manifest["automation"]["steps"] if step["name"] == "search")["status"] == "blocked"
    assert candidate.metadata["workflow"]["phase"] == "intake"
    assert candidate.metadata["scientific_disposition"] == "unknown"
    assert fit_samples == [2500]
    assert next(step for step in manifest["automation"]["steps"] if step["name"] == "screen")["status"] == "blocked"
    incidents = list((tmp_path / "log").glob("issue-*.md"))
    assert len(incidents) == 1
    incident = incidents[0].read_text(encoding="utf-8")
    assert "Full Python Traceback" in incident
    assert "synthetic unexpected screen failure" in incident
    assert "exonym survey auto-vet {0}".format(candidate.candidate_id) in incident
    assert manifest["runtime"] == {
        "kind": "direct",
        "version": __version__,
        "version_known": True,
        "executable": "exonym.autonomous",
    }


def test_auto_vet_retains_atomic_incremental_manifest_on_interrupt(tmp_path, monkeypatch):
    candidate = create_candidate(tmp_path, "automation-interrupt", tic="123456789", mission="tess")

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr("exonym.screening.run_fixed_ephemeris_screen", interrupt)

    with pytest.raises(KeyboardInterrupt):
        auto_vet_candidate(candidate, download=False)

    manifests = list((candidate.path / "runs" / "auto-vet").glob("*/engine-run.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["failure"]["code"] == "auto-vet-incomplete"
    assert [step["name"] for step in manifest["automation"]["steps"]] == ["ingest", "search"]
    assert not list(manifests[0].parent.glob("*.tmp"))


def test_auto_vet_all_reports_invalid_workspace_as_incomplete(tmp_path, monkeypatch, capsys):
    from exonym.__main__ import main

    candidate = create_candidate(tmp_path, "automation-valid", tic="123456789", mission="tess")
    (tmp_path / "candidate" / "automation-broken").mkdir()
    manifest = candidate.path / "runs" / "auto-vet" / "synthetic" / "engine-run.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("exonym.autonomous.auto_vet_candidate", lambda *_args, **_kwargs: manifest)

    assert main(["--root", str(tmp_path), "survey", "auto-vet", "--all", "--no-download"]) == 1

    report = json.loads(capsys.readouterr().out)
    assert {entry["status"] for entry in report["outcomes"]} == {"completed", "incomplete"}
    assert report["claim_eligible"] is False


def test_auto_vet_cli_records_incident_when_candidate_run_cannot_start(tmp_path, monkeypatch, capsys):
    from exonym.__main__ import main

    candidate = create_candidate(tmp_path, "automation-start-failure", tic="123456789", mission="tess")

    def fail_auto_vet(*_args, **_kwargs):
        raise RuntimeError("synthetic startup failure")

    monkeypatch.setattr("exonym.autonomous.auto_vet_candidate", fail_auto_vet)

    assert main(["--root", str(tmp_path), "survey", "auto-vet", candidate.candidate_id]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["outcomes"][0]["status"] == "failed"
    incidents = list((tmp_path / "log").glob("issue-*.md"))
    assert len(incidents) == 1
    assert "synthetic startup failure" in incidents[0].read_text(encoding="utf-8")


def test_run_loop_resumes_already_provisioned_candidate_without_auto_vet_run(
    tmp_path, monkeypatch, capsys
):
    from exonym.__main__ import main

    survey = create_survey(tmp_path, "loop-test", "tess", [17])
    candidate = create_candidate(tmp_path, "loop-target", tic="123456789", mission="tess")
    manifest = candidate.path / "runs" / "auto-vet" / "resumed" / "engine-run.json"
    calls = []

    monkeypatch.setattr(
        "exonym.survey_harvest.harvest_tces",
        lambda *_args, **_kwargs: [{"candidate_id": candidate.candidate_id, "status": "already-provisioned"}],
    )

    def run_auto_vet(workspace, **_kwargs):
        calls.append(workspace.candidate_id)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr("exonym.autonomous.auto_vet_candidate", run_auto_vet)

    assert main(
        [
            "--root",
            str(tmp_path),
            "survey",
            "run-loop",
            survey.survey_id,
            "--source",
            "https://example.invalid/tces.csv",
        ]
    ) == 0

    report = json.loads(capsys.readouterr().out)
    journal = json.loads((tmp_path / report["journal"]).read_text(encoding="utf-8"))
    assert calls == [candidate.candidate_id]
    assert journal["status"] == "completed"
    assert journal["cycles"][0]["auto_vet"] == [
        {
            "candidate_id": candidate.candidate_id,
            "status": "completed",
            "manifest": manifest.relative_to(tmp_path).as_posix(),
        }
    ]
    assert journal["claim_eligible"] is False


def test_autonomous_cli_failure_records_atomic_incident(tmp_path, monkeypatch):
    from exonym.__main__ import main

    candidate = create_candidate(tmp_path, "incident-target", tic="123456789", mission="tess")
    monkeypatch.setattr(
        "exonym.survey.load_survey_candidate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--root", str(tmp_path), "survey", "auto-vet", candidate.candidate_id, "--no-download"])

    assert exc_info.value.code == 2
    incidents = list((tmp_path / "log").glob("issue-*.md"))
    assert len(incidents) == 1
    incident = incidents[0].read_text(encoding="utf-8")
    assert "Exonym Version" in incident
    assert "Full Python Traceback" in incident
    assert "synthetic failure" in incident
    assert not list((tmp_path / "log").glob("*.tmp"))


def test_incident_recorder_writes_required_sections_atomically(tmp_path):
    try:
        raise RuntimeError("synthetic incident")
    except RuntimeError as exc:
        path = record_autonomous_incident(tmp_path, "exonym survey run-loop loop-test", exc)

    content = path.read_text(encoding="utf-8")
    assert "UTC Timestamp" in content
    assert "Expected Behavior" in content
    assert "Observed Behavior" in content
    assert "Root Cause And Affected Modules" in content
    assert "Actionable Remediation" in content
    assert not list(path.parent.glob("*.tmp"))


def test_auto_vet_intersects_requested_sectors_with_common_archive_products():
    common = _available_common_sectors(
        ["s0002", "s0005", "unusable"],
        [2, 5, 8],
    )

    assert common == [2, 5]
    assert _select_download_sectors(common, [5, 9]) == [5]
    assert _select_download_sectors(common, None) == [2]
    with pytest.raises(RuntimeError, match="No requested sectors"):
        _select_download_sectors(common, [9])


def test_remote_tce_stream_is_fully_staged_before_first_row(monkeypatch):
    class Response:
        def __init__(self):
            self._body = b"tic,period\n123456789,3.0\n987654321,4.0\n"
            self.read_calls = 0
            self.finished = False

        def read(self, size=-1):
            self.read_calls += 1
            if not self._body:
                self.finished = True
                return b""
            if size is None or size < 0:
                chunk, self._body = self._body, b""
                return chunk
            chunk, self._body = self._body[:size], self._body[size:]
            return chunk

    response = Response()

    @contextmanager
    def fake_https_response(*_args, **_kwargs):
        yield response

    monkeypatch.setattr("exonym.survey_harvest._https_response", fake_https_response)

    rows = stream_tce_rows("https://example.invalid/release.csv")
    assert next(rows)["tic"] == "123456789"
    assert response.finished
    assert response.read_calls >= 2
    rows.close()


@pytest.mark.parametrize("value", (0, -1, True))
def test_harvest_rejects_invalid_limit(tmp_path, value):
    survey = create_survey(tmp_path, "harvest-test", "tess", [17])

    with pytest.raises(ValueError, match="max_candidates"):
        harvest_tces(survey, "missing.csv", _filters(), value)
