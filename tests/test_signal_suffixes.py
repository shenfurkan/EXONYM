"""Tests for per-signal result suffix conventions and collision prevention.

Verifies that multi-signal runs (e.g. signal .01, .02) write distinct, suffixed
artifacts across transit fitting, TTV analysis, TRICERATOPS vetting, and
diagnostic figure generation without colliding or overwriting un-suffixed
default artifacts.
"""

from __future__ import annotations

import json
from unittest.mock import patch
import numpy as np
import pytest

from exonym.plotting import generate_candidate_plots
from exonym.ttv import run_ttv_analysis
from exonym.transit_fit import run_mcmc_transit_fit
from exonym.vetting.tricera_parse import run_triceratops_simulation
from exonym.workspace import create_candidate


def test_plotting_signal_suffix(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-test-signal-plot-suffix")
    (workspace.path / "config" / "transit_config.json").write_text(
        '{"transit": {"period_days": 2.5, "epoch_btjd": 0.5, "duration_hours": 2.0}}\n',
        encoding="utf-8",
    )
    signal_config = workspace.path / "config" / "signals" / "transit_config.01.json"
    signal_config.parent.mkdir(parents=True, exist_ok=True)
    signal_config.write_text(
        '{"transit": {"period_days": 3.0, "epoch_btjd": 0.5, "duration_hours": 2.0}}\n',
        encoding="utf-8",
    )
    light_curve = (np.linspace(0.0, 10.0, 200), np.ones(200))

    with patch("exonym.plotting.load_candidate_light_curve", return_value=light_curve):
        # 1. Un-suffixed run
        plots_default = generate_candidate_plots(workspace)
        assert [path.name for path in plots_default] == ["phase_folded_lc.png"]
        for path in plots_default:
            assert path.is_file()

        # 2. Suffixed run (.01)
        plots_signal = generate_candidate_plots(workspace, signal=".01")
        assert [path.name for path in plots_signal] == ["phase_folded_lc.01.png"]
        for path in plots_signal:
            assert path.is_file()

    # Verify both sets exist simultaneously without overwriting
    assert (workspace.path / "figures" / "phase_folded_lc.png").is_file()
    assert (workspace.path / "figures" / "phase_folded_lc.01.png").is_file()
    assert not (workspace.path / "figures" / "centroid_offset.png").exists()
    assert not (workspace.path / "figures" / "centroid_offset.01.png").exists()


def test_ttv_signal_suffix(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-test-signal-ttv-suffix")

    def write_transit_fit_artifact(signal):
        suffix = f".{signal.lstrip('.')}" if signal else ""
        path = workspace.path / "outputs" / f"mcmc_transit_fit{suffix}.json"
        path.write_text(
            json.dumps(
                {
                    "work_package": "MCMC_TRANSIT_FIT",
                    "source": "candidate-data",
                    "signal": signal,
                    "parameter_names": [
                        "rp_rs",
                        "log_rho_star",
                        "impact_parameter",
                        "baseline",
                        "log_jitter",
                        "q1",
                        "q2",
                    ],
                    "ephemeris": {
                        "period_days": 3.0,
                        "epoch_btjd": 1.0,
                        "source": "candidate-config",
                    },
                    "posterior": {
                        "impact_parameter": {"median": 0.3},
                        "q1": {"median": 0.3},
                        "q2": {"median": 0.3},
                    },
                }
            ),
            encoding="utf-8",
        )

    write_transit_fit_artifact(None)
    write_transit_fit_artifact(".02")

    # Mock inputs so we don't need real lightkurve downloads in synthetic test
    synthetic_table = {
        "time": np.linspace(0, 30, 500),
        "flux": np.ones(500),
        "flux_err": np.full(500, 0.0001),
        "sector": np.full(500, 1),
    }

    ephemeris = {
        "period_days": 3.0,
        "epoch_btjd": 1.0,
        "duration_days": 2.0 / 24.0,
        "depth_ppm": 1000.0,
        "source": "candidate-config",
        "field_sources": {
            "period_days": "candidate-config",
            "epoch_btjd": "candidate-config",
            "duration_days": "candidate-config",
            "depth_ppm": "candidate-config",
        },
    }
    stellar = {
        "mass_solar": 1.0,
        "mass_solar_err": 0.1,
        "radius_solar": 1.0,
        "radius_solar_err": 0.05,
        "source": "candidate-data",
    }
    with patch("exonym.ttv.load_light_curve_table", return_value=synthetic_table), patch(
        "exonym.ttv.load_transit_ephemeris", return_value=ephemeris
    ), patch("exonym.ttv.load_stellar_parameters", return_value=stellar):
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


def test_ttv_refuses_to_fabricate_photometry(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-test-ttv-no-photometry")

    with patch("exonym.ttv.load_light_curve_table", return_value=None):
        with pytest.raises(RuntimeError, match="observed candidate photometry"):
            run_ttv_analysis(workspace)


def test_triceratops_signal_suffix(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-test-signal-vet-suffix")

    # 1. Default un-suffixed run with allow_fallback=True
    report_default = run_triceratops_simulation(workspace, n_draws=100, allow_fallback=True)
    assert report_default.name == "triceratops_report.json"
    assert report_default.is_file()
    assert not (workspace.path / "claims" / "fpp_claim.json").exists()

    # 2. Suffixed run (.01)
    with pytest.warns(UserWarning, match="could not read signal transit config"):
        report_signal = run_triceratops_simulation(workspace, n_draws=100, signal=".01", allow_fallback=True)
    assert report_signal.name == "triceratops_report.01.json"
    assert report_signal.is_file()
    assert not (workspace.path / "claims" / "fpp_claim.01.json").exists()

    # Verify both coexist
    assert (workspace.path / "outputs" / "triceratops_report.json").is_file()
    assert (workspace.path / "outputs" / "triceratops_report.01.json").is_file()


def test_transit_fit_signal_suffix(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-test-signal-fit-suffix")

    ephemeris = {
        "period_days": 3.0,
        "epoch_btjd": 1.0,
        "duration_days": 2.0 / 24.0,
        "depth_ppm": 1000.0,
        "source": "candidate-config",
        "field_sources": {
            "period_days": "candidate-config",
            "epoch_btjd": "candidate-config",
            "duration_days": "candidate-config",
            "depth_ppm": "candidate-config",
        },
    }
    stellar = {
        "mass_solar": 1.0,
        "mass_solar_err": 0.1,
        "radius_solar": 1.0,
        "radius_solar_err": 0.05,
        "source": "candidate-data",
    }
    from tests.fixtures.synthetic_observations import _synthetic_transit_table

    synthetic_table = _synthetic_transit_table(ephemeris)
    with patch("exonym.transit_fit.load_light_curve_table", return_value=synthetic_table), patch(
        "exonym.transit_fit.load_transit_ephemeris", return_value=ephemeris
    ), patch("exonym.transit_fit.load_stellar_parameters", return_value=stellar):
        # 1. Default un-suffixed run (low sample count for quick test)
        out_default = run_mcmc_transit_fit(workspace, n_samples=10, n_walkers=16, burn_in=20)
        assert out_default.name == "mcmc_transit_fit.json"
        assert out_default.is_file()
        assert (workspace.path / "outputs" / "mcmc_transit_fit_chain.npy").is_file()

        # 2. Suffixed run (.01)
        out_signal = run_mcmc_transit_fit(
            workspace, n_samples=10, n_walkers=16, burn_in=20, signal=".01"
        )
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
