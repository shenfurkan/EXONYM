import shutil

import pytest

from exonym.__main__ import main


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
        "engine-run.schema.json",
        "automated-triage.schema.json",
        "radial-velocity-observations.schema.json",
        "rv-keplerian-fit.schema.json",
        "planetsynth-characterization.schema.json",
        "anomalous-transit-hypothesis.schema.json",
        "planetsynth-interpretation.schema.json",
        "pyppluss-hypothesis-test.schema.json",
        "statistical-vetting-evidence.schema.json",
        "decisive-rejection.schema.json",
        "catalog-query-manifest.schema.json",
        "catalog-raw-response-metadata.schema.json",
        "catalog-snapshot.schema.json",
        "catalog-stellar-parameters.schema.json",
        "catalog-stellar-photometry.schema.json",
        "catalog-archive-discovery.schema.json",
        "catalog-contrast-curves.schema.json",
        "catalog-context.schema.json",
        "catalog-cross-match.schema.json",
    ):
        shutil.copy2("schemas/{0}".format(name), tmp_path / "schemas" / name)
    (tmp_path / "requirements-lock.txt").write_text(
        "numpy==1.26.4\nscipy==1.13.1\n", encoding="utf-8"
    )
    return tmp_path


def test_cli_full_lifecycle(tmp_path, capsys):
    repo = _repo(tmp_path)
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
    assert main(root + ["verify"]) == 0

    output = capsys.readouterr().out
    assert "ISOLATION: PASS" in output


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
    called = []
    monkeypatch.setattr(
        "exonym.vetting.tricera_parse.run_triceratops_simulation",
        lambda *args, **kwargs: called.append(True),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(root + ["vet", "candidate-alpha"])

    assert exc_info.value.code == 2
    assert called == []
    assert (repo / "candidate" / "candidate-alpha" / "outputs" / "statistical_vetting_evidence.json").is_file()
    assert (repo / "candidate" / "candidate-alpha" / "decisions" / "automated_triage.json").is_file()


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

    def fake_search(candidate, period_min, period_max, signal, engine):
        calls.append(
            {
                "candidate": candidate.candidate_id,
                "period_min": period_min,
                "period_max": period_max,
                "signal": signal,
                "engine": engine,
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
            "period_min": 0.5,
            "period_max": 15.0,
            "signal": None,
            "engine": "tls",
        }
    ]
    assert "tls_search_results.json" in capsys.readouterr().out


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
    assert main(root + ["asteroseismology", "candidate-alpha"]) == 0
    assert "asteroseismic_results.json" in capsys.readouterr().out


def test_cli_asteroseismology_accepts_numax_bounds(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    assert main(root + ["asteroseismology", "candidate-alpha", "--numax-min", "50", "--numax-max", "900"]) == 0
    assert "asteroseismic_results.json" in capsys.readouterr().out


def test_cli_localization_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    assert main(root + ["localization", "candidate-alpha", "--search-radius", "30"]) == 0
    assert "prf_localization_results.json" in capsys.readouterr().out


def test_cli_sed_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    assert main(root + ["sed", "candidate-alpha"]) == 0
    assert "sed_fit_results.json" in capsys.readouterr().out


def test_cli_fit_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    assert main(root + ["fit", "candidate-alpha", "--n-samples", "200"]) == 0
    assert "mcmc_transit_fit.json" in capsys.readouterr().out


def test_cli_fit_passes_dynesty_sampler(tmp_path, capsys, mocker):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    output = repo / "candidate" / "candidate-alpha" / "outputs" / "mcmc_transit_fit.json"
    mock_fit = mocker.patch("exonym.transit_fit.run_mcmc_transit_fit", return_value=output)

    assert main(root + ["fit", "candidate-alpha", "--sampler", "dynesty"]) == 0
    assert mock_fit.call_args.kwargs["sampler"] == "dynesty"
    assert "mcmc_transit_fit.json" in capsys.readouterr().out


def test_cli_fit_eccentric_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    assert main(root + ["fit", "candidate-alpha", "--n-samples", "200", "--eccentric"]) == 0
    assert "mcmc_transit_fit.json" in capsys.readouterr().out


def test_cli_phasecurve_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    assert main(root + ["phasecurve", "candidate-alpha"]) == 0
    assert "phase_curve_results.json" in capsys.readouterr().out


def test_cli_ttv_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    assert main(root + ["ttv", "candidate-alpha"]) == 0
    assert "ttv_analysis_results.json" in capsys.readouterr().out


def test_cli_activity_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    assert main(root + ["activity", "candidate-alpha"]) == 0
    assert "stellar_activity_results.json" in capsys.readouterr().out


def test_cli_dilution_command(tmp_path, capsys):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    assert main(root + ["dilution", "candidate-alpha"]) == 0
    assert "dilution_sensitivity_results.json" in capsys.readouterr().out


def test_cli_science_outputs_exist_on_disk(tmp_path):
    repo = _repo(tmp_path)
    root = ["--root", str(repo)]
    main(_init_alpha(repo))
    commands = [
        ["asteroseismology", "candidate-alpha"],
        ["localization", "candidate-alpha"],
        ["sed", "candidate-alpha"],
        ["phasecurve", "candidate-alpha"],
        ["ttv", "candidate-alpha"],
        ["activity", "candidate-alpha"],
        ["dilution", "candidate-alpha"],
    ]
    for command in commands:
        assert main(root + command) == 0
    outputs_dir = repo / "candidate" / "candidate-alpha" / "outputs"
    for filename in (
        "asteroseismic_results.json",
        "prf_localization_results.json",
        "sed_fit_results.json",
        "phase_curve_results.json",
        "ttv_analysis_results.json",
        "stellar_activity_results.json",
        "dilution_sensitivity_results.json",
    ):
        assert (outputs_dir / filename).is_file()
