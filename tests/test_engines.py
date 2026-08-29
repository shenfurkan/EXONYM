"""Unit tests for target-neutral engine registry and CLI commands."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from exonym.engines import (
    EngineStatus,
    check_engine,
    get_engine,
    iter_engines,
    run_engine,
)
from exonym.__main__ import main
from exonym.workspace import create_candidate


def test_iter_engines_returns_all_catalog_entries():
    engines = iter_engines()
    assert len(engines) >= 10
    names = {e.name for e in engines}
    assert "bls" in names
    assert "batman" in names
    assert "emcee" in names
    assert "triceratops" in names
    assert "pysyd" in names


def test_get_engine_existing_core():
    status = get_engine("bls")
    assert status is not None
    assert status.name == "bls"
    assert status.capability == "search"
    assert status.optional_group == "core"
    assert status.installed is True


def test_get_engine_case_insensitive():
    status = get_engine("  BaTmAn  ")
    assert status is not None
    assert status.name == "batman"
    assert status.capability == "fitting"


def test_get_engine_unknown():
    assert get_engine("nonexistent_fake_engine") is None


def test_check_engine_known():
    ready, msg = check_engine("bls")
    assert ready is True
    assert "bls" in msg
    assert "installed and ready" in msg


def test_check_engine_unknown():
    ready, msg = check_engine("unknown_engine_xyz")
    assert ready is False
    assert "Unknown engine" in msg


def test_check_engine_unconfigured_dependency_has_direct_install_hint(monkeypatch):
    missing = EngineStatus(
        name="dynesty",
        capability="sampler",
        optional_group="optional",
        module_name="dynesty",
        description="test",
        installed=False,
        version=None,
    )
    monkeypatch.setattr("exonym.engines.get_engine", lambda name: missing)

    ready, message = check_engine("dynesty")

    assert ready is False
    assert "pip install dynesty" in message


def test_check_engine_rejects_incompatible_installed_dependency(monkeypatch):
    def fake_version(name):
        return {"triceratops": "1.0.20", "pytransit": "2.6.11"}[name]

    monkeypatch.setattr("exonym.engines.importlib.util.find_spec", lambda _name: object())
    monkeypatch.setattr(
        "exonym.engines.distribution",
        lambda name: SimpleNamespace(requires=["pytransit == 2.2"]) if name == "triceratops" else None,
    )
    monkeypatch.setattr("exonym.engines.version", fake_version)

    status = get_engine("triceratops")
    ready, message = check_engine("triceratops")

    assert status is not None
    assert status.dependency_issues == ("requires pytransit==2.2, but pytransit 2.6.11 is installed",)
    assert ready is False
    assert "dependency/interface contract is incompatible" in message
    assert "pytransit 2.6.11" in message


def test_run_engine_blocks_an_incompatible_runtime_before_execution(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "engine-incompatible-runtime")
    incompatible = EngineStatus(
        name="screen",
        capability="screening",
        optional_group="core",
        module_name="numpy",
        description="test",
        installed=True,
        version="1.0",
        dependency_issues=("requires synthetic-dependency==1.0, but 2.0 is installed",),
    )
    monkeypatch.setattr("exonym.engines.get_engine", lambda _name: incompatible)

    with pytest.raises(RuntimeError, match="dependency/interface contract is incompatible"):
        run_engine(workspace, "screen")


def test_cli_engine_list(capsys):
    ret = main(["engine", "list"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Engine" in captured.out
    assert "bls" in captured.out
    assert "batman" in captured.out


def test_cli_engine_list_json(capsys):
    ret = main(["engine", "list", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, list)
    names = [item["name"] for item in data]
    assert "bls" in names
    assert "emcee" in names


def test_cli_engine_check_success(capsys):
    ret = main(["engine", "check", "bls"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "is installed and ready" in captured.out


def test_cli_engine_check_failure(capsys):
    ret = main(["engine", "check", "nonexistent_engine"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Unknown engine" in captured.out


def test_run_engine_forwards_ttv_keyword_arguments(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "engine-ttv-kwargs")
    output = workspace.path / "outputs" / "ttv_analysis_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{}\n", encoding="utf-8")
    captured = {}

    def fake_ttv(candidate, signal=None, **kwargs):
        captured["signal"] = signal
        captured.update(kwargs)
        return output

    monkeypatch.setattr("exonym.ttv.run_ttv_analysis", fake_ttv)

    manifest = run_engine(
        workspace,
        "ttv",
        signal=".01",
        ephemeris_model="quadratic",
        fit_orbital_decay=True,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert captured == {
        "signal": ".01",
        "ephemeris_model": "quadratic",
        "fit_orbital_decay": True,
    }
    assert payload["engine"] == "ttv"
    assert payload["status"] == "succeeded"
