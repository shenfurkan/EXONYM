"""Tests for headless diagnostic figure generation."""

import hashlib
from unittest.mock import patch

import numpy as np

from exonym.plotting import generate_candidate_plots, plot_centroid_offsets, plot_phase_folded_lc
from exonym.workspace import create_candidate


def test_plot_phase_folded_lc(tmp_path):
    time = np.linspace(0, 10, 200)
    flux = np.ones_like(time)
    out = tmp_path / "test_lc.png"
    result = plot_phase_folded_lc(time, flux, period_days=2.5, epoch_btjd=0.5, output_path=out)
    assert result.is_file()
    assert result.stat().st_size > 0


def test_plot_centroid_offsets(tmp_path):
    out = tmp_path / "test_centroid.png"
    result = plot_centroid_offsets([0.1, -0.1], [0.2, -0.05], sigma_arcsec=0.1, output_path=out)
    assert result.is_file()
    assert result.stat().st_size > 0


def test_generate_candidate_plots(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-test-plot")
    plots = generate_candidate_plots(workspace)
    assert len(plots) == 2
    for path in plots:
        assert path.is_file()


def test_generate_candidate_plots_deterministic(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-test-deterministic")
    first = generate_candidate_plots(workspace)
    second = generate_candidate_plots(workspace)
    for first_path, second_path in zip(first, second):
        digest_one = hashlib.sha256(first_path.read_bytes()).hexdigest()
        digest_two = hashlib.sha256(second_path.read_bytes()).hexdigest()
        assert digest_one == digest_two


def test_generate_candidate_plots_uses_requested_signal_config(tmp_path):
    # Arrange
    workspace = create_candidate(tmp_path, "candidate-test-signal-plot")
    signal_config = workspace.path / "config" / "signals" / "transit_config.03.json"
    signal_config.parent.mkdir(parents=True, exist_ok=True)
    signal_config.write_text(
        '{"transit": {"period_days": 2.25, "epoch_btjd": 1.5, "duration_hours": 2.0}}',
        encoding="utf-8",
    )

    # Act
    with patch("exonym.plotting.plot_phase_folded_lc") as phase_plot:
        generate_candidate_plots(workspace, signal=".03")

    # Assert
    assert phase_plot.call_args.args[2] == 2.25
    assert phase_plot.call_args.args[3] == 1.5
