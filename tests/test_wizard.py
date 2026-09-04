"""Unit coverage for the interactive wizard: builders, validation, and flow."""

import pytest

import exonym.wizard as wizard
from exonym.wizard import (
    build_detrend_argv,
    build_fit_argv,
    build_ingest_argv,
    build_init_argv,
    build_search_argv,
    build_vet_argv,
    parse_sectors,
    validate_period_bounds,
    validate_window_days,
)


def test_argv_builders_match_registered_cli_flags():
    assert build_init_argv("tgt-01", "tess", "12345", ["a", "b"]) == [
        "init", "tgt-01", "--mission", "tess", "--tic", "12345",
        "--tag", "a", "--tag", "b",
    ]
    assert build_ingest_argv("tgt-01", [3, 4], "lc") == [
        "ingest", "tgt-01", "--sectors", "3", "4", "--products", "lc",
    ]
    assert build_detrend_argv("tgt-01", "wotan", 0.75) == [
        "detrend", "tgt-01", "--method", "wotan", "--window-days", "0.75",
    ]
    assert build_search_argv("tgt-01", "tls", 0.5, 15.0, "running-median") == [
        "search", "tgt-01", "--engine", "tls",
        "--period-min", "0.5", "--period-max", "15.0",
        "--detrending-method", "running-median",
    ]
    assert build_fit_argv("tgt-01", 2500, True) == [
        "fit", "tgt-01", "--n-samples", "2500", "--eccentric",
    ]
    assert build_vet_argv("tgt-01", 2000) == ["vet", "tgt-01", "--n-draws", "2000"]


def test_ingest_builder_never_selects_a_static_scientific_cadence():
    argv = build_ingest_argv("tgt-01", [3, 4], "both")

    assert "--exptime" not in argv
    assert argv == ["ingest", "tgt-01", "--sectors", "3", "4", "--products", "both"]


def test_validation_helpers_reject_bad_input():
    with pytest.raises(ValueError):
        parse_sectors("")
    with pytest.raises(ValueError):
        parse_sectors("1 two")
    with pytest.raises(ValueError):
        parse_sectors("3 3")
    with pytest.raises(ValueError):
        parse_sectors("0")
    assert parse_sectors("1, 2 3") == [1, 2, 3]

    with pytest.raises(ValueError):
        validate_period_bounds(15.0, 0.5)
    with pytest.raises(ValueError):
        validate_period_bounds(-1.0, 5.0)
    assert validate_period_bounds(0.5, 15.0) == (0.5, 15.0)

    with pytest.raises(ValueError):
        validate_window_days(0.0)
    assert validate_window_days(2) == 2.0


def _script(monkeypatch, texts=(), floats=(), ints=(), confirms=()):
    """Replace wizard prompt adapters with deterministic scripted queues."""
    text_queue = list(texts)
    float_queue = list(floats)
    int_queue = list(ints)
    confirm_queue = list(confirms)
    monkeypatch.setattr(
        wizard, "_ask_text", lambda console, msg, default="": text_queue.pop(0)
    )
    monkeypatch.setattr(
        wizard, "_ask_float", lambda console, msg, default: (
            float_queue.pop(0) if float_queue else default
        )
    )
    monkeypatch.setattr(
        wizard, "_ask_int", lambda console, msg, default: (
            int_queue.pop(0) if int_queue else default
        )
    )
    monkeypatch.setattr(wizard, "_ask_choice", lambda console, msg, choices, default: default)
    monkeypatch.setattr(
        wizard, "_ask_confirm", lambda console, msg, default: bool(confirm_queue.pop(0))
    )


def test_wizard_requires_interactive_terminal(tmp_path, capsys):
    # pytest stdin/stdout are not TTYs.
    assert wizard.run_wizard(tmp_path, None) == 2
    err = capsys.readouterr().err
    assert "interactive terminal" in err


def test_wizard_happy_path_executes_confirmed_steps_only(tmp_path, monkeypatch):
    calls = []

    def fake_main(argv):
        calls.append(list(argv))
        return 0

    monkeypatch.setattr("exonym.__main__.main", fake_main)
    # Text answers: sectors only (candidate supplied -> no init prompts).
    texts = ["1 2"]
    # Floats in flow order: detrend window, Pmin, Pmax.
    floats = [0.75, 0.5, 15.0]
    # Ints in flow order: fit samples, vet draws.
    ints = [2500, 2000]
    # Confirms in flow order: ingest-plan Y, detrend N, search N, archive N,
    # sed N, eccentric N, fit-plan Y, localization N, dilution N, vet-plan N.
    confirms = [True, False, False, False, False, False, True, False, False, False]
    _script(monkeypatch, texts=texts, floats=floats, ints=ints, confirms=confirms)

    code = wizard.run_wizard(tmp_path, "wizard-flow-target", interactive=True)

    assert code == 0
    root_prefix = ["--root", str(tmp_path)]
    executed = [argv[len(root_prefix):][0] for argv in calls if argv[:2] == root_prefix]
    assert executed == ["ingest", "fit"]
    ingest_argv = calls[0][len(root_prefix):]
    assert ingest_argv == [
        "ingest", "wizard-flow-target",
        "--sectors", "1", "2", "--products", "lc",
    ]
    fit_argv = calls[1][len(root_prefix):]
    assert fit_argv == ["fit", "wizard-flow-target", "--n-samples", "2500"]


def test_wizard_counts_failed_step_as_failure(tmp_path, monkeypatch):
    def failing_main(argv):
        return 3

    monkeypatch.setattr("exonym.__main__.main", failing_main)
    _script(
        monkeypatch,
        texts=["7"],
        floats=[0.75, 0.5, 15.0],
        ints=[2500, 100],
        confirms=[True] * 10,
    )

    assert wizard.run_wizard(tmp_path, "wizard-fail-target", interactive=True) == 1
