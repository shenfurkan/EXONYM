"""Tests for per-signal result suffix conventions and collision prevention.

Verifies that multi-signal runs (e.g. signal .01, .02) write distinct, suffixed
artifacts across transit fitting, TTV analysis, TRICERATOPS vetting, and
diagnostic figure generation without colliding or overwriting un-suffixed
default artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from exonym.plotting import generate_candidate_plots
from exonym.ttv import run_ttv_analysis
from exonym.transit_fit import run_mcmc_transit_fit
from exonym.vetting.tricera_parse import run_triceratops_simulation
from exonym.workspace import create_candidate


def test_plotting_signal_suffix(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-test-signal-plot-suffix")
    
    # 1. Un-suffixed run
    plots_default = generate_candidate_plots(workspace)
    assert len(plots_default) == 2
    assert plots_default[0].name == "phase_folded_lc.png"
    assert plots_default[1].name == "centroid_offset.png"
    for p in plots_default:
        assert p.is_file()

    # 2. Suffixed run (.01)
    plots_signal = generate_candidate_plots(workspace, signal=".01")
    assert len(plots_signal) == 2
    assert plots_signal[0].name == "phase_folded_lc.01.png"
    assert plots_signal[1].name == "centroid_offset.01.png"
    for p in plots_signal:
        assert p.is_file()

    # Verify both sets exist simultaneously without overwriting
    assert (workspace.path / "figures" / "phase_folded_lc.png").is_file()
    assert (workspace.path / "figures" / "phase_folded_lc.01.png").is_file()


def test_ttv_signal_suffix(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-test-signal-ttv-suffix")

    # Mock inputs so we don't need real lightkurve downloads in synthetic test
    synthetic_table = {
        "time": np.linspace(0, 30, 500),
        "flux": np.ones(500),
        "flux_err": np.full(500, 0.0001),
        "sector": np.full(500, 1),
    }

    with patch("exonym.ttv.load_light_curve_table", return_value=synthetic_table):
        # 1. Default un-suffixed run
        out_default = run_ttv_analysis(workspace)
        assert out_default.name == "ttv_analysis_results.json"
        assert out_default.is_file()

        # 2. Suffixed run (.02)
        out_signal = run_ttv_analysis(workspace, signal=".02")
        assert out_signal.name == "ttv_analysis_results.02.json"
        assert out_signal.is_file()

        # Check payload records signal field
        payload = json.loads(out_signal.read_text(encoding="utf-8"))
        assert payload["signal"] == ".02"

        # Verify both coexist
        assert (workspace.path / "outputs" / "ttv_analysis_results.json").is_file()
        assert (workspace.path / "outputs" / "ttv_analysis_results.02.json").is_file()


def test_triceratops_signal_suffix(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-test-signal-vet-suffix")

    # 1. Default un-suffixed run with allow_fallback=True
    report_default = run_triceratops_simulation(workspace, n_draws=100, allow_fallback=True)
    assert report_default.name == "triceratops_report.json"
    assert report_default.is_file()
    assert (workspace.path / "claims" / "fpp_claim.json").is_file()

    # 2. Suffixed run (.01)
    with pytest.warns(UserWarning, match="could not read signal transit config"):
        report_signal = run_triceratops_simulation(workspace, n_draws=100, signal=".01", allow_fallback=True)
    assert report_signal.name == "triceratops_report.01.json"
    assert report_signal.is_file()
    assert (workspace.path / "claims" / "fpp_claim.01.json").is_file()

    # Verify both coexist
    assert (workspace.path / "outputs" / "triceratops_report.json").is_file()
    assert (workspace.path / "outputs" / "triceratops_report.01.json").is_file()
    assert (workspace.path / "claims" / "fpp_claim.json").is_file()
    assert (workspace.path / "claims" / "fpp_claim.01.json").is_file()


def test_transit_fit_signal_suffix(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-test-signal-fit-suffix")

    synthetic_table = {
        "time": np.linspace(0, 10, 300),
        "flux": np.ones(300),
        "flux_err": np.full(300, 0.0002),
        "sector": np.full(300, 1),
    }

    with patch("exonym.transit_fit.load_light_curve_table", return_value=synthetic_table):
        # 1. Default un-suffixed run (low sample count for quick test)
        out_default = run_mcmc_transit_fit(workspace, n_samples=10)
        assert out_default.name == "mcmc_transit_fit.json"
        assert out_default.is_file()
        assert (workspace.path / "outputs" / "mcmc_transit_fit_chain.npy").is_file()

        # 2. Suffixed run (.01)
        out_signal = run_mcmc_transit_fit(workspace, n_samples=10, signal=".01")
        assert out_signal.name == "mcmc_transit_fit.01.json"
        assert out_signal.is_file()
        assert (workspace.path / "outputs" / "mcmc_transit_fit_chain.01.npy").is_file()

        # Check payload
        payload = json.loads(out_signal.read_text(encoding="utf-8"))
        assert payload["signal"] == ".01"

        # Verify both coexist
        assert (workspace.path / "outputs" / "mcmc_transit_fit.json").is_file()
        assert (workspace.path / "outputs" / "mcmc_transit_fit.01.json").is_file()
        assert (workspace.path / "outputs" / "mcmc_transit_fit_chain.npy").is_file()
        assert (workspace.path / "outputs" / "mcmc_transit_fit_chain.01.npy").is_file()
