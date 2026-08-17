"""Unit tests for target-neutral engine registry and CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from exonym.engines import (
    EngineStatus,
    check_engine,
    get_engine,
    iter_engines,
)
from exonym.__main__ import main


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
