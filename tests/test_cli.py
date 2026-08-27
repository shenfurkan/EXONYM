import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from exonym.__main__ import main, _build_parser


def _repo(tmp_path):
    for name in (
        "docs/01_intake_manifest.md",
        "docs/02_feasibility_report.md",
        "docs/03_spoc_dv_vetting.md",
        "docs/04_tfop_sg_followup.md",
    ):
        path = tmp_path / "templates" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("- [ ] [MANDATORY] task\n", encoding="utf-8")
    (tmp_path / "templates/decisions/review_gate.md").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "templates/decisions/review_gate.md").write_text(
        "- [ ] [MANDATORY] task\n", encoding="utf-8"
    )
    (tmp_path / "templates/protocols").mkdir(parents=True, exist_ok=True)
    (tmp_path / "templates/tracking").mkdir(parents=True, exist_ok=True)
    (tmp_path / "schemas").mkdir(parents=True, exist_ok=True)
    for name in (
        "candidate.schema.json",
        "provenance.schema.json",
        "claim.schema.json",
        "novelty-audit.schema.json",
        "survey.schema.json",
        "survey-target.schema.json",
        "survey-robustness.schema.json",
        "survey-sensitivity.schema.json",
        "engine-run.schema.json",
        "automated-triage.schema.json",
        "radial-velocity-observations.schema.json",
        "rv-keplerian-fit.schema.json",
        "planetsynth-characterization.schema.json",
        "anomalous-transit-hypothesis.schema.json",
        "planetsynth-interpretation.schema.json",
        "pyppluss-hypothesis-test.schema.json",
        "asymmetric-transit-hypothesis.schema.json",
        "terminator-asymmetry-test.schema.json",
        "mist-main-sequence-input.schema.json",
        "sed-fit-results.schema.json",
        "ttv-analysis.schema.json",
        "statistical-vetting-evidence.schema.json",
        "decisive-rejection.schema.json",
        "triceratops-vetting-decision.schema.json",
        "catalog-query-manifest.schema.json",
        "catalog-raw-response-metadata.schema.json",
        "catalog-snapshot.schema.json",
        "catalog-stellar-parameters.schema.json",
        "catalog-stellar-photometry.schema.json",
        "catalog-archive-discovery.schema.json",
        "catalog-contrast-curves.schema.json",
        "catalog-context.schema.json",
        "catalog-cross-match.schema.json",
        "known-signal-ephemeris-match.schema.json",
        "known-signal-ephemeris-evidence.schema.json",
        "stellar-activity.schema.json",
        "phase-curve.schema.json",
        "detrending-manifest.schema.json",
        "ldtk-quadratic-limb-darkening-prior.schema.json",
        "exofop-prior-retrieval.schema.json",
    ):
        shutil.copy2("schemas/{0}".format(name), tmp_path / "schemas" / name)
    (tmp_path / "requirements-lock.txt").write_text(
        "numpy==1.26.4\nscipy==1.13.1\n", encoding="utf-8"
    )
    return tmp_path


def test_cli_full_lifecycle(tmp_path, capsys):
    repo = _repo(tmp_path)
    shutil.copy2("pyproject.toml", repo / "pyproject.toml")
    shutil.copytree("src", repo / "src")
    root = ["--root", str(repo)]

    assert main(root + ["init", "candidate-alpha", "--toi", "1234.01", "--tic", "123456789"]) == 0
    assert main(root + ["list"]) == 0
    assert main(root + ["list", "--mission", "tess"]) == 0
    assert main(root + ["status", "candidate-alpha"]) == 0
    assert main(root + ["track", "candidate-alpha"]) == 0

    with pytest.raises(SystemExit) as exc_info:
        main(root + ["advance", "candidate-alpha"])
    assert exc_info.value.code == 2

    assert main(root + ["tag", "candidate-alpha", "sg1-cleared"]) == 0
    assert main(root + ["freeze", "candidate-alpha", "--version", "v1.0.0"]) == 0
    assert main(root + ["verify-release", "candidate-alpha", "--version", "v1.0.0"]) == 0
    assert main(root + ["verify"]) == 0
    output = capsys.readouterr().out
    assert '"checked_file_count"' in output
    assert "ISOLATION: PASS" in output


def test_cli_verify_candidate_isolated_from_default_shared_audit(tmp_path):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    assert main(root + ["init", "candidate-alpha"]) == 0
    (repo / "candidate" / "candidate-alpha" / "candidate.json").write_text("{\n", encoding="utf-8")

    assert main(root + ["verify"]) == 0
    assert main(root + ["verify", "candidate"]) == 1


@pytest.mark.parametrize("scope_flag", ("--source", "--candidates"))
def test_cli_verify_rejects_legacy_scope_combined_with_explicit_scope(tmp_path, capsys, scope_flag):
    repo = _repo(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        main(["--root", str(repo), "verify", scope_flag, "candidate"])

    assert exc_info.value.code == 2
    assert "legacy positional 'candidate' cannot be combined" in capsys.readouterr().err


def test_cli_survey_sensitivity_dispatches_without_changing_candidate_state(
    tmp_path, capsys, monkeypatch
):
    # Arrange
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    assert main(root + ["init", "survey-target", "--tic", "123456789", "--mission", "tess"]) == 0
    assert main(root + ["survey", "init", "test-survey", "--mission", "tess", "--sectors", "17"]) == 0
    assert main(root + ["survey", "add-target", "test-survey", "survey-target"]) == 0
    calls = []

    def fake_sensitivity(survey, candidate):
        calls.append((survey.survey_id, candidate.candidate_id))
        output = candidate.path / "outputs" / "survey_sensitivity.survey-test-survey.json"
        output.write_text("{}\n", encoding="utf-8")
        return output

    monkeypatch.setattr("exonym.survey.run_survey_sensitivity", fake_sensitivity)

    # Act / Assert
    assert main(root + ["survey", "sensitivity", "test-survey", "survey-target"]) == 0
    assert calls == [("test-survey", "survey-target")]
    assert "survey_sensitivity.survey-test-survey.json" in capsys.readouterr().out


def test_cli_survey_harvest_dispatches_with_explicit_source(tmp_path, capsys, monkeypatch):
    # Arrange
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    assert main(root + ["survey", "init", "test-survey", "--mission", "tess", "--sectors", "17"]) == 0
    calls = []

    def fake_harvest(survey, source, filters, max_candidates, novelty_timeout, freshness_hours):
        calls.append((survey.survey_id, source, filters.minimum_snr, max_candidates, novelty_timeout, freshness_hours))
        return []

    monkeypatch.setattr("exonym.survey_harvest.harvest_tces", fake_harvest)

    # Act / Assert
    assert main(root + ["survey", "harvest", "test-survey", "--source", "https://example.invalid/tces.csv"]) == 0
    assert calls == [("test-survey", "https://example.invalid/tces.csv", 20.0, 25, 20.0, 24.0)]
    assert capsys.readouterr().out.endswith("[]\n")

def test_cli_init_sets_mission_and_tags(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(
        root
        + ["init", "candidate-beta", "--toi", "1234.01", "--tic", "123456789",
           "--mission", "tess", "--tag", "priority-1"]
    )
    output = capsys.readouterr().out
    assert '"mission": "tess"' in output
    assert '"priority-1"' in output


def test_cli_list_filters_by_phase(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(root + ["init", "candidate-alpha", "--toi", "1234.01", "--tic", "123456789"])
    main(root + ["init", "candidate-beta"])
    main(root + ["list", "--phase", "intake"])
    output = capsys.readouterr().out
    assert "candidate-alpha" in output
    assert "candidate-beta" in output

    main(root + ["list", "--phase", "analysis"])
    output = capsys.readouterr().out
    assert "candidate-alpha" not in output


def test_cli_survey_registers_a_toi_free_target(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(root + ["survey", "init", "test-survey", "--mission", "tess", "--sectors", "17"])
    main(root + ["init", "survey-target", "--tic", "123456789", "--mission", "tess"])

    assert main(root + ["survey", "add-target", "test-survey", "survey-target"]) == 0
    assert main(root + ["survey", "report", "test-survey"]) == 0

    output = capsys.readouterr().out
    assert "pending-eligibility" in output


def test_cli_ingest_requires_tic(tmp_path):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(root + ["init", "candidate-alpha"])
    with pytest.raises(SystemExit) as exc_info:
        main(root + ["ingest", "candidate-alpha", "--sectors", "37"])
    assert exc_info.value.code == 2


def test_cli_vet_command(tmp_path, capsys, monkeypatch):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(root + ["init", "candidate-alpha", "--toi", "1234.01", "--tic", "123456789"])

    calls = []

    def fake_run_triceratops(candidate, n_draws=2000, signal=None):
        calls.append({"candidate": candidate.candidate_id, "n_draws": n_draws, "signal": signal})
        output = candidate.path / "outputs" / "triceratops_report.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"source": "test-stub"}\n', encoding="utf-8")
        return output

    monkeypatch.setattr(
        "exonym.vetting.tricera_parse.run_triceratops_simulation", fake_run_triceratops
    )
    monkeypatch.setattr(
        "exonym.statistical_vetting.require_vetting_readiness", lambda candidate, signal=None: candidate.path
    )

    assert main(root + ["vet", "candidate-alpha", "--n-draws", "100"]) == 0
    assert calls == [{"candidate": "candidate-alpha", "n_draws": 100, "signal": None}]
    output = capsys.readouterr().out
    assert "triceratops_report.json" in output


def test_cli_vet_blocks_before_triceratops_without_required_evidence(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(root + ["init", "candidate-alpha", "--tic", "123456789"])

    with pytest.raises(SystemExit) as exc_info:
        main(root + ["vet", "candidate-alpha"])

    assert exc_info.value.code == 2
    decision_path = repo / "candidate" / "candidate-alpha" / "decisions" / "triceratops_vetting_decision.json"
    assert decision_path.is_file()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["execution_status"] == "blocked"
    assert decision["triage_status"] == "not-run"
    assert not (repo / "candidate" / "candidate-alpha" / "decisions" / "automated_triage.json").exists()


def test_cli_vet_does_not_accept_force_bypass(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(root + ["init", "candidate-alpha", "--tic", "123456789"])
    called = []
    monkeypatch.setattr(
        "exonym.vetting.tricera_parse.run_triceratops_simulation",
        lambda *args, **kwargs: called.append(True),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(root + ["vet", "candidate-alpha", "--force"])

    assert exc_info.value.code == 2
    assert called == []


def test_cli_search_forwards_tls_engine(tmp_path, capsys, monkeypatch):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(root + ["init", "candidate-alpha"])
    calls = []

    def fake_search(candidate, period_min, period_max, signal, engine, detrending_method):
        calls.append(
            {
                "candidate": candidate.candidate_id,
                "period_min": period_min,
                "period_max": period_max,
                "signal": signal,
                "engine": engine,
                "detrending_method": detrending_method,
            }
        )
        output = candidate.path / "outputs" / "tls_search_results.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}\n", encoding="utf-8")
        return output

    monkeypatch.setattr("exonym.search.run_bls_on_candidate", fake_search)

    assert main(root + ["search", "candidate-alpha", "--engine", "tls"]) == 0
    assert calls == [
        {
            "candidate": "candidate-alpha",
            "period_min": None,
            "period_max": None,
            "signal": None,
            "engine": "tls",
            "detrending_method": None,
        }
    ]
    assert "tls_search_results.json" in capsys.readouterr().out


def test_cli_detrend_derives_and_forwards_candidate_ephemeris_mask(tmp_path, capsys, monkeypatch):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    assert main(root + ["init", "candidate-alpha"]) == 0
    candidate_path = repo / "candidate" / "candidate-alpha"
    time = np.array([0.75, 1.0, 1.25, 2.0, 3.0])
    table = {
        "time": time,
        "flux": np.ones(time.size),
        "flux_err": np.full(time.size, 0.0001),
        "sector": np.ones(time.size, dtype=int),
        "input_files": [candidate_path / "data" / "raw" / "source.fits"],
        "input_sha256s": ["a" * 64],
        "time_system": "BTJD_TDB",
    }
    ephemeris = {
        "period_days": 2.0,
        "epoch_btjd": 1.0,
        "duration_days": 0.4,
        "time_system": "BTJD_TDB",
        "source": "candidate-config",
        "field_sources": {
            "period_days": "candidate-config",
            "epoch_btjd": "candidate-config",
            "duration_days": "candidate-config",
        },
    }
    calls = []

    def fake_detrend(candidate, time_btjd, flux, **kwargs):
        calls.append(
            {
                "candidate": candidate.candidate_id,
                "time": time_btjd,
                "mask": kwargs["transit_mask"],
                "ephemeris": kwargs["transit_mask_ephemeris"],
            }
        )
        return SimpleNamespace(
            artifact_path=candidate.path / "data" / "processed" / "detrended-running-median.npz",
            manifest_path=candidate.path / "outputs" / "detrending_manifest.running-median.json",
        )

    monkeypatch.setattr(
        "exonym.inputs.load_light_curve_table", lambda *_args, **_kwargs: table
    )
    monkeypatch.setattr(
        "exonym.inputs.load_transit_ephemeris", lambda *_args, **_kwargs: ephemeris
    )
    monkeypatch.setattr("exonym.detrending.detrend_candidate", fake_detrend)

    assert main(root + ["detrend", "candidate-alpha", "--window-days", "0.5"]) == 0
    assert len(calls) == 1
    assert calls[0]["candidate"] == "candidate-alpha"
    assert np.array_equal(calls[0]["time"], time)
    assert np.array_equal(calls[0]["mask"], np.array([False, True, False, False, True]))
    assert calls[0]["ephemeris"] == ephemeris
    assert "detrended-running-median.npz" in capsys.readouterr().out


def test_cli_detrend_rejects_a_synthetic_ephemeris_before_writing_output(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    assert main(root + ["init", "candidate-alpha"]) == 0
    candidate_path = repo / "candidate" / "candidate-alpha"
    time = np.array([0.75, 1.0, 1.25, 2.0, 3.0])
    table = {
        "time": time,
        "flux": np.ones(time.size),
        "flux_err": np.full(time.size, 0.0001),
        "sector": np.ones(time.size, dtype=int),
        "input_files": [candidate_path / "data" / "raw" / "source.fits"],
        "input_sha256s": ["a" * 64],
        "time_system": "BTJD_TDB",
    }
    incomplete_ephemeris = {
        "period_days": 2.0,
        "epoch_btjd": 1.0,
        "duration_days": 0.4,
        "time_system": "BTJD_TDB",
        "source": "partial-candidate-config",
        "field_sources": {
            "period_days": "candidate-config",
            "epoch_btjd": "synthetic-demo",
            "duration_days": "candidate-config",
        },
    }
    called = []

    monkeypatch.setattr(
        "exonym.inputs.load_light_curve_table", lambda *_args, **_kwargs: table
    )
    monkeypatch.setattr(
        "exonym.inputs.load_transit_ephemeris", lambda *_args, **_kwargs: incomplete_ephemeris
    )
    monkeypatch.setattr(
        "exonym.detrending.detrend_candidate", lambda *_args, **_kwargs: called.append(True)
    )

    with pytest.raises(SystemExit) as exc_info:
        main(root + ["detrend", "candidate-alpha", "--window-days", "0.5"])

    assert exc_info.value.code == 2
    assert called == []


def test_cli_fetch_priors_command(tmp_path, capsys, monkeypatch):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(root + ["init", "candidate-alpha", "--toi", "1234.01", "--tic", "123456789"])
    calls = []

    def fake_fetch_priors(candidate):
        calls.append(candidate.candidate_id)
        output = candidate.path / "config" / "signals" / "transit_config.01.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}\n", encoding="utf-8")
        return [output]

    monkeypatch.setattr("exonym.priors.fetch_exofop_priors", fake_fetch_priors)

    assert main(root + ["fetch-priors", "candidate-alpha"]) == 0
    assert calls == ["candidate-alpha"]
    assert "config/signals/transit_config.01.json" in capsys.readouterr().out


def _init_alpha(repo):
    return ["--root", str(repo)] + ["init", "candidate-alpha", "--toi", "1234.01", "--tic", "123456789"]


def test_cli_asteroseismology_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    with pytest.raises(SystemExit) as exc_info:
        main(root + ["asteroseismology", "candidate-alpha"])
    assert exc_info.value.code == 2
    assert not (repo / "candidate" / "candidate-alpha" / "outputs" / "asteroseismic_results.json").exists()


def test_cli_asteroseismology_accepts_numax_bounds(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    with pytest.raises(SystemExit) as exc_info:
        main(root + ["asteroseismology", "candidate-alpha", "--numax-min", "50", "--numax-max", "900"])
    assert exc_info.value.code == 2
    assert not (repo / "candidate" / "candidate-alpha" / "outputs" / "asteroseismic_results.json").exists()


def test_cli_localization_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    assert main(root + ["localization", "candidate-alpha", "--search-radius", "30"]) == 0
    assert "prf_localization_results.json" in capsys.readouterr().out


def _write_candidate_sed_photometry(repo):
    path = repo / "candidate" / "candidate-alpha" / "data" / "external" / "stellar_photometry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "2MASS": {
                    "J": {"mag": 10.1, "error": 0.03},
                    "H": {"mag": 9.8, "error": 0.03},
                    "Ks": {"mag": 9.7, "error": 0.03},
                },
                "AllWISE": {"W1": {"mag": 9.6, "error": 0.03}},
            }
        ),
        encoding="utf-8",
    )
    (path.parent / "stellar_params.json").write_text(
        json.dumps(
            {"teff_k": 5700.0, "logg_cgs": 4.4, "feh": 0.0, "mass_solar": 1.0,
             "radius_solar": 1.0, "parallax_mas": 10.0, "parallax_mas_err": 0.05}
        ),
        encoding="utf-8",
    )


def test_cli_sed_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    _write_candidate_sed_photometry(repo)
    assert main(root + ["sed", "candidate-alpha"]) == 0
    assert "sed_fit_results.json" in capsys.readouterr().out


def test_cli_fit_requires_candidate_derived_inputs(tmp_path):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    _write_candidate_sed_photometry(repo)
    with pytest.raises(SystemExit) as exc_info:
        main(root + ["fit", "candidate-alpha", "--n-samples", "200"])
    assert exc_info.value.code == 2
    assert not (repo / "candidate" / "candidate-alpha" / "outputs" / "mcmc_transit_fit.json").exists()


def test_cli_fit_passes_dynesty_sampler(tmp_path, capsys, mocker):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    output = repo / "candidate" / "candidate-alpha" / "outputs" / "mcmc_transit_fit.json"
    mock_fit = mocker.patch("exonym.transit_fit.run_mcmc_transit_fit", return_value=output)

    assert main(root + ["fit", "candidate-alpha", "--sampler", "dynesty"]) == 0
    assert mock_fit.call_args.kwargs["sampler"] == "dynesty"
    assert mock_fit.call_args.kwargs["device"] == "auto"
    assert "mcmc_transit_fit.json" in capsys.readouterr().out


def test_cli_fit_n_jobs_flag(tmp_path):
    """--n-jobs accepts integer and defaults to 1."""
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    parser = _build_parser()
    ns = parser.parse_args(["fit", "tic-123", "--n-jobs", "2"])
    assert ns.n_jobs == 2
    # Default
    ns_default = parser.parse_args(["fit", "tic-123"])
    assert ns_default.n_jobs == 1


def test_cli_fit_device_flag_defaults_to_auto_and_accepts_cpu(tmp_path):
    parser = _build_parser()

    assert parser.parse_args(["fit", "candidate-x"]).device == "auto"
    assert parser.parse_args(["fit", "candidate-x", "--device", "cpu"]).device == "cpu"


def test_cli_fit_progress_flag(tmp_path):
    """--progress is a boolean store_true."""
    parser = _build_parser()
    ns = parser.parse_args(["fit", "tic-123", "--progress"])
    assert ns.progress is True


def test_cli_parser_registers_checkpoint_group():
    parser = _build_parser()

    save = parser.parse_args(["checkpoint", "save", "candidate-x", "--name", "pre-fit"])
    assert (save.checkpoint_action, save.candidate_id, save.name) == (
        "save", "candidate-x", "pre-fit",
    )
    listing = parser.parse_args(["checkpoint", "list", "candidate-x"])
    assert listing.checkpoint_action == "list"
    restore = parser.parse_args(
        ["checkpoint", "restore", "candidate-x", "--id", "20260101T000000Z_snap", "--yes"]
    )
    assert restore.checkpoint_action == "restore" and restore.yes is True
    delete = parser.parse_args(
        ["checkpoint", "delete", "candidate-x", "--id", "20260101T000000Z_snap"]
    )
    assert delete.checkpoint_action == "delete"


def test_cli_parser_registers_wizard_command():
    parser = _build_parser()

    with_target = parser.parse_args(["wizard", "candidate-x"])
    assert with_target.candidate_id == "candidate-x"
    without_target = parser.parse_args(["wizard"])
    assert without_target.candidate_id is None


def test_cli_fit_resume_flag(tmp_path):
    """--resume accepts a string path."""
    parser = _build_parser()
    ns = parser.parse_args(["fit", "tic-123", "--resume", "checkpoint.npz"])
    assert ns.resume == "checkpoint.npz"
    ns_default = parser.parse_args(["fit", "tic-123"])
    assert ns_default.resume is None


def test_cli_eccentric_fit_requires_candidate_derived_inputs(tmp_path):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    with pytest.raises(SystemExit) as exc_info:
        main(root + ["fit", "candidate-alpha", "--n-samples", "200", "--eccentric"])
    assert exc_info.value.code == 2
    assert not (repo / "candidate" / "candidate-alpha" / "outputs" / "mcmc_transit_fit.json").exists()


def test_cli_phasecurve_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    with pytest.raises(SystemExit) as exc_info:
        main(root + ["phasecurve", "candidate-alpha"])
    assert exc_info.value.code == 2
    assert not (repo / "candidate" / "candidate-alpha" / "outputs" / "phase_curve_results.json").exists()


def test_cli_parser_accepts_ttv_decay_and_specialized_commands():
    parser = _build_parser()

    args = parser.parse_args(["ttv", "candidate-x", "--signal", ".01", "--fit-orbital-decay"])
    assert args.candidate_id == "candidate-x"
    assert args.signal == ".01"
    assert args.fit_orbital_decay is True

    assert parser.parse_args(["fit", "candidate-x"]).n_samples == 2500
    assert (
        parser.parse_args(["survey", "auto-vet", "candidate-x"]).fit_samples
        == 2500
    )
    assert (
        parser.parse_args(["survey", "run-loop", "loop-1", "--source", "https://example.invalid"]).fit_samples
        == 2500
    )

    for command in ("planetsynth", "pyppluss", "catwoman", "squishyplanet"):
        parsed = parser.parse_args([command, "candidate-x"])
        assert parsed.candidate_id == "candidate-x"


def test_cli_ttv_requires_observed_candidate_photometry(tmp_path):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    with pytest.raises(SystemExit) as exc_info:
        main(root + ["ttv", "candidate-alpha"])
    assert exc_info.value.code == 2
    assert not (repo / "candidate" / "candidate-alpha" / "outputs" / "ttv_analysis_results.json").exists()


def test_cli_activity_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    with pytest.raises(SystemExit) as exc_info:
        main(root + ["activity", "candidate-alpha"])
    assert exc_info.value.code == 2
    assert not (repo / "candidate" / "candidate-alpha" / "outputs" / "stellar_activity_results.json").exists()


def test_cli_dilution_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    with pytest.raises(SystemExit) as exc_info:
        main(root + ["dilution", "candidate-alpha"])
    assert exc_info.value.code == 2
    assert not (repo / "candidate" / "candidate-alpha" / "outputs" / "dilution_sensitivity_results.json").exists()


def test_cli_science_outputs_exist_on_disk(tmp_path):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    _write_candidate_sed_photometry(repo)
    commands = [
        ["localization", "candidate-alpha"],
        ["sed", "candidate-alpha"],
    ]
    for command in commands:
        assert main(root + command) == 0
    outputs_dir = repo / "candidate" / "candidate-alpha" / "outputs"
    for filename in (
        "prf_localization_results.json",
        "sed_fit_results.json",
    ):
        assert (outputs_dir / filename).is_file()
