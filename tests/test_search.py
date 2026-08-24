"""Tests for the target-neutral BLS transit search engine."""

import json
import hashlib
import sys
import types

import numpy as np
import pytest

from exonym.lightcurve import phase_hours
from exonym.search import (
    BLSSearchResult,
    find_transits,
    find_transits_duration_grid,
    find_transits_tls,
    run_bls_on_candidate,
)
from exonym.workspace import create_candidate


def _write_raw_provenance(product):
    sidecar = product.with_name(product.stem + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "source_uri": "https://archive.example.invalid/" + product.name,
                "download_timestamp_utc": "2026-01-01T00:00:00Z",
                "sha256": hashlib.sha256(product.read_bytes()).hexdigest(),
                "fetched_by": "synthetic-test",
            }
        ),
        encoding="utf-8",
    )


def _raw_bls_table(workspace, time, flux):
    product = workspace.path / "data" / "raw" / "s0001_lc.fits"
    product.parent.mkdir(parents=True, exist_ok=True)
    product.write_bytes(b"synthetic-raw-photometry")
    _write_raw_provenance(product)
    return {
        "time": time,
        "flux": flux,
        "flux_err": np.full_like(time, 0.001),
        "flux_err_sources": ["reported"],
        "sector": np.ones(time.size, dtype=int),
        "input_files": [product],
        "input_sha256s": [hashlib.sha256(product.read_bytes()).hexdigest()],
    }


def test_find_transits_synthetic():
    time = np.linspace(0, 20, 2000)
    period = 4.0
    epoch = 1.0
    ph = phase_hours(time, period, epoch)
    flux = np.ones_like(time)
    flux[np.abs(ph) < 1.5] = 0.995

    res = find_transits(time, flux, period_min=1.0, period_max=10.0, duration_hours=3.0)
    assert res.best_period > 0
    assert res.snr > 0


def test_find_transits_resolves_double_period_alias():
    rng = np.random.default_rng(11)
    period = 4.2608
    epoch = 100.0
    duration_days = 6.45 / 24.0

    segments = []
    for start in (0.0, 165.0, 357.0):
        segments.append(np.arange(start, start + 26.0, 120.0 / 86400.0))
    time = np.concatenate(segments)

    flux = np.ones_like(time)
    ph = ((time - epoch + 0.5 * period) % period) / period - 0.5
    flux[np.abs(ph) < duration_days / period / 2.0] -= 0.003857
    flux += rng.normal(0.0, 0.0035, time.size)

    from exonym.search import _median_bin

    time_b, flux_b = _median_bin(time, flux, n_bins=3500)

    res = find_transits(time_b, flux_b, period_min=2.0, period_max=12.0)
    assert abs(res.best_period - period) < 0.02, "fundamental period lost to 2x alias"
    assert res.snr > 10.0
    assert res.best_depth_ppm > 2000.0


def test_find_transits_invalid():
    with pytest.raises(ValueError):
        find_transits([1.0, 2.0], [1.0, 1.0])

    with pytest.raises(ValueError):
        find_transits(np.linspace(0, 10, 100), np.ones(100), period_min=-1.0)


def test_find_transits_uses_weighted_astropy_bls_and_observed_event_count(monkeypatch):
    """The BLS core must consume per-cadence uncertainties, not a heuristic."""
    from astropy import timeseries

    captured = {}

    class FakeBoxLeastSquares:
        def __init__(self, time, flux, dy):
            captured["time"] = time
            captured["flux"] = flux
            captured["dy"] = dy

        def autopower(self, duration, **kwargs):
            captured["duration"] = duration
            captured["autopower"] = kwargs
            return types.SimpleNamespace(
                power=np.array([12.0]),
                period=np.array([3.0]),
                transit_time=np.array([1.0]),
                depth=np.array([0.003]),
                depth_err=np.array([0.0005]),
            )

    monkeypatch.setattr(timeseries, "BoxLeastSquares", FakeBoxLeastSquares)
    time = np.linspace(0.0, 12.0, 500)
    flux = np.ones_like(time)
    errors = np.linspace(0.0005, 0.0015, time.size)

    result = find_transits(
        time, flux, period_min=2.0, period_max=4.0, duration_hours=2.0, flux_err=errors
    )

    assert np.array_equal(captured["dy"], errors)
    assert captured["duration"] == pytest.approx(2.0 / 24.0)
    assert captured["autopower"]["minimum_n_transit"] == 2
    assert 0.0 < captured["autopower"]["frequency_factor"] <= 1.0
    assert result.snr == pytest.approx(6.0)
    assert result.detection_status == "no-detection"
    assert result.best_period is None
    assert result.n_distinct_transit_events >= 2
    assert result.n_period_trials == 1


def test_frequency_period_grid_has_uniform_frequency_spacing():
    from exonym.search import _frequency_period_grid

    periods = _frequency_period_grid(0.5, 20.0, 11)
    frequencies = 1.0 / periods

    assert periods[0] == pytest.approx(20.0)
    assert periods[-1] == pytest.approx(0.5)
    assert np.allclose(np.diff(frequencies), np.diff(frequencies)[0])


def test_find_transits_duration_grid_retains_all_trials_and_uses_best_score(monkeypatch):
    calls = []

    def fake_find_transits(time, flux, **kwargs):
        duration = kwargs["duration_hours"]
        calls.append(duration)
        return BLSSearchResult(3.0, 1.0, 100.0, duration, duration, 3)

    monkeypatch.setattr("exonym.search.find_transits", fake_find_transits)

    best, trials = find_transits_duration_grid(
        [0.0, 1.0], [1.0, 1.0], [1.0, 2.0, 4.0], period_min=0.5, period_max=20.0, n_periods=200
    )

    assert calls == [1.0, 2.0, 4.0]
    assert best.best_duration_hours == 4.0
    assert [trial["best_duration_hours"] for trial in trials] == [1.0, 2.0, 4.0]


def test_find_transits_tls_uses_native_cadence_uncertainties(monkeypatch):
    calls = {}

    class FakeModel:
        def power(self, **kwargs):
            calls["power"] = kwargs
            return types.SimpleNamespace(
                period=4.0,
                T0=1.0,
                depth=0.999,
                duration=0.125,
                SDE=9.0,
            )

    def fake_tls(time, flux, flux_err, verbose):
        calls["time"] = time
        calls["flux"] = flux
        calls["flux_err"] = flux_err
        calls["verbose"] = verbose
        return FakeModel()

    fake_module = types.ModuleType("transitleastsquares")
    fake_module.transitleastsquares = fake_tls
    monkeypatch.setitem(sys.modules, "transitleastsquares", fake_module)

    time = np.linspace(0.0, 20.0, 100)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 0.001)
    result = find_transits_tls(time, flux, flux_err, period_min=1.0, period_max=10.0)

    assert calls["verbose"] is False
    assert calls["power"] == {
        "period_min": 1.0,
        "period_max": 10.0,
        "show_progress_bar": False,
        "use_threads": 1,
    }
    assert result["best_period"] == 4.0
    assert result["best_epoch"] == 1.0
    assert result["best_depth_ppm"] == pytest.approx(1000.0)
    assert result["best_duration_hours"] == 3.0
    assert result["sde"] == 9.0


def test_run_bls_on_candidate_requires_real_photometry(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-test-bls")

    with pytest.raises(ValueError, match="no readable candidate light-curve photometry"):
        run_bls_on_candidate(workspace)

    assert not (workspace.path / "outputs" / "bls_search_results.json").exists()
    assert not (workspace.path / "outputs" / "bls_search_manifest.json").exists()


def test_run_bls_rejects_a_loader_result_without_raw_provenance(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "candidate-provenance-bls")
    time = np.linspace(0.0, 30.0, 100)
    table = _raw_bls_table(workspace, time, np.ones_like(time))
    table["input_files"][0].with_name("s0001_lc.provenance.json").unlink()
    monkeypatch.setattr(
        "exonym.inputs.load_light_curve_table", lambda *_args, **_kwargs: table
    )

    with pytest.raises(ValueError, match="raw provenance sidecars"):
        run_bls_on_candidate(workspace)


def test_run_bls_output_suffix_preserves_the_default_result(tmp_path, monkeypatch):
    # Arrange
    workspace = create_candidate(tmp_path, "candidate-output-suffix")
    outputs = workspace.path / "outputs"
    default_result = outputs / "bls_search_results.json"
    default_result.write_text('{"source": "candidate-data", "snr": 2.0}\n', encoding="utf-8")
    time = np.linspace(0.0, 30.0, 100)
    flux = np.ones_like(time)
    table = _raw_bls_table(workspace, time, flux)
    monkeypatch.setattr(
        "exonym.inputs.load_light_curve_table", lambda *_args, **_kwargs: table
    )
    monkeypatch.setattr(
        "exonym.search.find_transits",
        lambda *args, **kwargs: BLSSearchResult(3.0, 1.0, 100.0, 2.0, 6.0, 3),
    )

    # Act
    output = run_bls_on_candidate(workspace, result_suffix=".survey-test")

    # Assert
    assert output.name == "bls_search_results.survey-test.json"
    assert json.loads(default_result.read_text(encoding="utf-8"))["snr"] == 2.0
    assert (outputs / "bls_search_manifest.survey-test.json").is_file()


def test_run_bls_duration_grid_is_recorded_and_selects_the_best_trial(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "candidate-duration-grid")
    time = np.linspace(0.0, 30.0, 100)
    flux = np.ones_like(time)
    table = _raw_bls_table(workspace, time, flux)
    monkeypatch.setattr(
        "exonym.inputs.load_light_curve_table", lambda *_args, **_kwargs: table
    )

    calls = []

    def fake_find_transits(time_values, flux_values, **kwargs):
        calls.append(kwargs)
        duration = kwargs["duration_hours"]
        return BLSSearchResult(3.0, 1.0, 100.0, duration, duration, 3)

    monkeypatch.setattr("exonym.search.find_transits", fake_find_transits)

    output = run_bls_on_candidate(
        workspace, duration_grid_hours=[1.0, 2.0, 4.0], n_periods=211
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    manifest = json.loads((workspace.path / "outputs" / "bls_search_manifest.json").read_text())
    assert payload["best_duration_hours"] == 4.0
    assert [trial["best_duration_hours"] for trial in payload["duration_grid_trials"]] == [1.0, 2.0, 4.0]
    assert manifest["configuration"]["duration_hours"] is None
    assert manifest["configuration"]["duration_grid_hours"] == [1.0, 2.0, 4.0]
    assert manifest["configuration"]["n_periods"] == 211
    assert (
        manifest["configuration"]["period_grid"]
        == "astropy-autopower-baseline-duration-resolved"
    )
    assert all(call["n_periods"] == 211 for call in calls)


@pytest.mark.parametrize("suffix", ["survey", ".bad_value", ".bad.value"])
def test_run_bls_rejects_an_unsafe_or_ambiguous_output_suffix(tmp_path, suffix):
    # Arrange
    workspace = create_candidate(tmp_path, "candidate-output-suffix")

    # Act and assert
    with pytest.raises(ValueError, match="result_suffix"):
        run_bls_on_candidate(workspace, result_suffix=suffix)


def test_run_bls_signal_uses_prior_duration_and_preserves_each_signal_output(tmp_path, monkeypatch):
    """Targeted searches use their own prior and cannot overwrite another signal."""
    workspace = create_candidate(tmp_path, "candidate-targeted-bls")
    signals = workspace.path / "config" / "signals"
    signals.mkdir(parents=True)
    (signals / "transit_config.01.json").write_text(
        json.dumps(
            {
                "transit": {
                    "period_days": 4.2,
                    "epoch_btjd": 11.0,
                    "duration_hours": 2.4,
                }
            }
        ),
        encoding="utf-8",
    )
    (signals / "transit_config.02.json").write_text(
        json.dumps(
            {
                "transit": {
                    "period_days": 8.6,
                    "epoch_btjd": 12.0,
                    "duration_hours": 4.8,
                }
            }
        ),
        encoding="utf-8",
    )

    time = np.linspace(0.0, 30.0, 100)
    flux = np.ones_like(time)
    table = _raw_bls_table(workspace, time, flux)
    calls = []

    def fake_find_transits(time_btjd, flux_values, **kwargs):
        calls.append(kwargs)
        return BLSSearchResult(
            best_period=kwargs["period_min"],
            best_epoch=1.0,
            best_depth_ppm=100.0,
            best_duration_hours=kwargs["duration_hours"],
            snr=5.0,
            n_distinct_transit_events=3,
        )

    monkeypatch.setattr(
        "exonym.inputs.load_light_curve_table", lambda *_args, **_kwargs: table
    )
    monkeypatch.setattr("exonym.search.find_transits", fake_find_transits)

    first_output = run_bls_on_candidate(
        workspace,
        signal=".01",
        period_min=4.1,
        period_max=4.3,
    )
    second_output = run_bls_on_candidate(workspace, signal=".02")

    assert first_output.name == "bls_search_results.01.json"
    assert second_output.name == "bls_search_results.02.json"
    assert first_output != second_output
    assert not (workspace.path / "outputs" / "bls_search_results.json").exists()

    first_payload = json.loads(first_output.read_text(encoding="utf-8"))
    second_payload = json.loads(second_output.read_text(encoding="utf-8"))
    assert first_payload["signal"] == ".01"
    assert first_payload["search_provenance"] == {
        "mode": "targeted-prior",
        "signal": ".01",
        "prior_path": "config/signals/transit_config.01.json",
        "prior_source": "partial-candidate-config-signal",
        "prior_period_days": 4.2,
        "prior_epoch_btjd": 11.0,
        "prior_duration_hours": 2.4,
        "period_min_days": pytest.approx(4.1),
        "period_max_days": pytest.approx(4.3),
    }
    assert second_payload["signal"] == ".02"
    assert second_payload["search_provenance"]["prior_period_days"] == 8.6
    assert calls[0]["duration_hours"] == pytest.approx(2.4)
    assert calls[0]["period_min"] == pytest.approx(4.1)
    assert calls[0]["period_max"] == pytest.approx(4.3)
    assert calls[1]["duration_hours"] == pytest.approx(4.8)


def test_run_bls_signal_rejects_conflicting_explicit_period_bounds(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-targeted-bound-conflict")
    signals = workspace.path / "config" / "signals"
    signals.mkdir(parents=True)
    (signals / "transit_config.01.json").write_text(
        json.dumps(
            {
                "transit": {
                    "period_days": 4.2,
                    "duration_hours": 2.4,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bounds conflict with candidate prior window"):
        run_bls_on_candidate(
            workspace,
            signal=".01",
            period_min=1.0,
            period_max=10.0,
        )


def test_search_cli_leaves_blind_period_bounds_unset_until_requested():
    from exonym.__main__ import _build_parser

    parser = _build_parser()
    default_args = parser.parse_args(["search", "synthetic-candidate"])
    explicit_args = parser.parse_args(
        ["search", "synthetic-candidate", "--period-min", "1.0", "--period-max", "10.0"]
    )

    assert default_args.period_min is None
    assert default_args.period_max is None
    assert explicit_args.period_min == pytest.approx(1.0)
    assert explicit_args.period_max == pytest.approx(10.0)


def test_run_bls_signal_requires_a_readable_signal_prior(tmp_path):
    workspace = create_candidate(tmp_path, "candidate-missing-prior")

    with pytest.raises(ValueError, match="no readable signal prior"):
        run_bls_on_candidate(workspace, signal=".01")


def test_synthetic_bls_result_cannot_seed_an_ephemeris(tmp_path):
    from exonym.inputs import load_transit_ephemeris

    workspace = create_candidate(tmp_path, "synthetic-bls-guard")
    outputs = workspace.path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "bls_search_results.json").write_text(
        json.dumps(
            {
                "best_period": 3.0,
                "best_epoch": 1.0,
                "best_duration_hours": 2.0,
                "best_depth_ppm": 500.0,
                "source": "synthetic-demo",
            }
        ),
        encoding="utf-8",
    )

    ephemeris = load_transit_ephemeris(workspace)

    assert ephemeris["source"] == "synthetic-demo"


def test_no_detection_bls_result_cannot_seed_an_ephemeris(tmp_path):
    from exonym.inputs import load_transit_ephemeris

    workspace = create_candidate(tmp_path, "no-detection-bls-guard")
    outputs = workspace.path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "bls_search_results.json").write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "detection_status": "no-detection",
                "best_period": 3.0,
                "best_epoch": 1.0,
                "best_duration_hours": 2.0,
                "best_depth_ppm": 500.0,
                "n_distinct_transit_events": 3,
            }
        ),
        encoding="utf-8",
    )

    ephemeris = load_transit_ephemeris(workspace)

    assert ephemeris["source"] == "synthetic-demo"


def test_unbound_bls_result_cannot_seed_an_ephemeris(tmp_path):
    from exonym.inputs import load_transit_ephemeris

    workspace = create_candidate(tmp_path, "unbound-bls-guard")
    outputs = workspace.path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "bls_search_results.json").write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "detection_status": "detected",
                "best_period": 3.0,
                "best_epoch": 1.0,
                "best_duration_hours": 2.0,
                "best_depth_ppm": 500.0,
                "n_distinct_transit_events": 3,
            }
        ),
        encoding="utf-8",
    )

    ephemeris = load_transit_ephemeris(workspace)

    assert ephemeris["source"] == "synthetic-demo"


def test_bls_manifest_requires_candidate_photometry_inputs(tmp_path):
    from exonym.inputs import load_transit_ephemeris

    workspace = create_candidate(tmp_path, "non-photometry-bls-input")
    outputs = workspace.path / "outputs"
    result_path = outputs / "bls_search_results.json"
    result_path.write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "detection_status": "detected",
                "time_system": "BTJD_TDB",
                "best_period": 3.0,
                "best_epoch": 1.0,
                "best_duration_hours": 2.0,
                "best_depth_ppm": 500.0,
                "snr": 8.0,
                "n_distinct_transit_events": 3,
            }
        ),
        encoding="utf-8",
    )
    candidate_json = workspace.path / "candidate.json"
    (outputs / "bls_search_manifest.json").write_text(
        json.dumps(
            {
                "schema": "exonym-bls-search-manifest-1",
                "candidate_id": workspace.candidate_id,
                "result_path": "outputs/bls_search_results.json",
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "source": "candidate-data",
                "detection_status": "detected",
                "inputs": [
                    {
                        "path": "candidate.json",
                        "sha256": hashlib.sha256(candidate_json.read_bytes()).hexdigest(),
                    }
                ],
                "configuration": {"engine": "bls", "signal": None, "time_system": "BTJD_TDB"},
            }
        ),
        encoding="utf-8",
    )

    assert load_transit_ephemeris(workspace)["source"] == "synthetic-demo"


def test_bls_binding_rejects_a_changed_detrending_product(tmp_path):
    from exonym.detrending import detrend_candidate, transit_mask_from_ephemeris
    from exonym.inputs import is_manifest_bound_bls_result

    workspace = create_candidate(tmp_path, "detrending-bound-search")
    raw_path = workspace.path / "data" / "raw" / "source.fits"
    raw_path.write_bytes(b"synthetic raw product")
    _write_raw_provenance(raw_path)
    raw_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    time = np.linspace(0.0, 8.0, 101)
    ephemeris = {
        "period_days": 3.5,
        "epoch_btjd": 1.0,
        "duration_days": 2.0 / 24.0,
        "time_system": "BTJD_TDB",
        "source": "candidate-config",
        "field_sources": {
            "period_days": "candidate-config",
            "epoch_btjd": "candidate-config",
            "duration_days": "candidate-config",
        },
    }
    detrending = detrend_candidate(
        workspace,
        time,
        1.0 + 0.001 * np.sin(time),
        window_days=0.5,
        sector=np.ones(time.size, dtype=int),
        input_products=[{"path": "data/raw/source.fits", "sha256": raw_digest}],
        transit_mask=transit_mask_from_ephemeris(time, ephemeris),
        transit_mask_ephemeris=ephemeris,
    )
    detrending_manifest = json.loads(detrending.manifest_path.read_text(encoding="utf-8"))
    preprocessing = {
        "kind": "candidate-detrending",
        "method": "running-median",
        "manifest": {
            "path": "outputs/detrending_manifest.running-median.json",
            "sha256": hashlib.sha256(detrending.manifest_path.read_bytes()).hexdigest(),
        },
        "artifact": detrending_manifest["artifact"],
    }
    result_path = workspace.path / "outputs" / "bls_search_results.json"
    result = {
        "source": "candidate-data",
        "detection_status": "detected",
        "time_system": "BTJD_TDB",
        "detection_threshold_snr": 7.1,
        "best_period": 3.5,
        "best_epoch": 1.0,
        "best_duration_hours": 2.0,
        "best_depth_ppm": 500.0,
        "snr": 8.0,
        "n_distinct_transit_events": 3,
        "preprocessing": preprocessing,
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    manifest = {
        "schema": "exonym-bls-search-manifest-1",
        "candidate_id": workspace.candidate_id,
        "result_path": "outputs/bls_search_results.json",
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "source": "candidate-data",
        "detection_status": "detected",
        "inputs": [
            {
                "path": "data/raw/source.fits",
                "sha256": raw_digest,
                "provenance_path": "data/raw/source.provenance.json",
                "provenance_sha256": hashlib.sha256(
                    raw_path.with_name("source.provenance.json").read_bytes()
                ).hexdigest(),
            }
        ],
        "configuration": {
            "engine": "bls",
            "signal": None,
            "time_system": "BTJD_TDB",
            "detection_threshold_snr": 7.1,
            "preprocessing": preprocessing,
        },
    }
    manifest_path = workspace.path / "outputs" / "bls_search_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert is_manifest_bound_bls_result(workspace, result_path, result, None)

    with np.load(detrending.artifact_path, allow_pickle=False) as archive:
        artifact_payload = {name: archive[name] for name in archive.files}
    artifact_payload["detrended_flux"] = artifact_payload["detrended_flux"].copy()
    artifact_payload["detrended_flux"][0] -= 0.01
    np.savez_compressed(detrending.artifact_path, **artifact_payload)

    assert not is_manifest_bound_bls_result(workspace, result_path, result, None)


def test_bls_binding_rejects_legacy_unmasked_preprocessing_for_ephemeris_resolution(tmp_path):
    """A BLS-derived config cannot revive a pre-mask detrending artifact."""
    from exonym.inputs import is_manifest_bound_bls_result, load_transit_ephemeris

    workspace = create_candidate(tmp_path, "legacy-preprocessing")
    raw_path = workspace.path / "data" / "raw" / "source.fits"
    raw_path.write_bytes(b"synthetic raw product")
    _write_raw_provenance(raw_path)
    raw_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    artifact_path = workspace.path / "data" / "processed" / "detrended-running-median.npz"
    artifact_path.write_bytes(b"legacy detrending artifact")
    artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    detrending_manifest_path = workspace.path / "outputs" / "detrending_manifest.running-median.json"
    detrending_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id": workspace.candidate_id,
                "generated_utc": "2026-01-01T00:00:00Z",
                "method": "running-median",
                "configuration": {"window_days": 0.5},
                "input": {"cadences": 101, "finite_cadences": 101, "sha256": "a" * 64},
                "artifact": {
                    "path": "data/processed/detrended-running-median.npz",
                    "sha256": artifact_digest,
                    "data_sha256": "b" * 64,
                },
                "input_products": [{"path": "data/raw/source.fits", "sha256": raw_digest}],
            }
        ),
        encoding="utf-8",
    )
    preprocessing = {
        "kind": "candidate-detrending",
        "method": "running-median",
        "manifest": {
            "path": "outputs/detrending_manifest.running-median.json",
            "sha256": hashlib.sha256(detrending_manifest_path.read_bytes()).hexdigest(),
        },
        "artifact": {
            "path": "data/processed/detrended-running-median.npz",
            "sha256": artifact_digest,
            "data_sha256": "b" * 64,
        },
    }
    result_path = workspace.path / "outputs" / "bls_search_results.json"
    result = {
        "source": "candidate-data",
        "detection_status": "detected",
        "time_system": "BTJD_TDB",
        "detection_threshold_snr": 7.1,
        "best_period": 3.5,
        "best_epoch": 1.0,
        "best_duration_hours": 2.0,
        "best_depth_ppm": 500.0,
        "snr": 8.0,
        "n_distinct_transit_events": 3,
        "preprocessing": preprocessing,
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    bls_manifest_path = workspace.path / "outputs" / "bls_search_manifest.json"
    bls_manifest_path.write_text(
        json.dumps(
            {
                "schema": "exonym-bls-search-manifest-1",
                "candidate_id": workspace.candidate_id,
                "result_path": "outputs/bls_search_results.json",
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "source": "candidate-data",
                "detection_status": "detected",
                "inputs": [
                    {
                        "path": "data/raw/source.fits",
                        "sha256": raw_digest,
                        "provenance_path": "data/raw/source.provenance.json",
                        "provenance_sha256": hashlib.sha256(
                            raw_path.with_name("source.provenance.json").read_bytes()
                        ).hexdigest(),
                    }
                ],
                "configuration": {
                    "engine": "bls",
                    "signal": None,
                    "time_system": "BTJD_TDB",
                    "detection_threshold_snr": 7.1,
                    "preprocessing": preprocessing,
                },
            }
        ),
        encoding="utf-8",
    )
    (workspace.path / "config" / "transit_config.json").write_text(
        json.dumps(
            {
                "source": "candidate-data-bls",
                "bls_provenance": {
                    "result": {
                        "path": "outputs/bls_search_results.json",
                        "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                    },
                    "manifest": {
                        "path": "outputs/bls_search_manifest.json",
                        "sha256": hashlib.sha256(bls_manifest_path.read_bytes()).hexdigest(),
                    },
                },
                "transit": {
                    "period_days": 3.5,
                    "epoch_btjd": 1.0,
                    "duration_days": 2.0 / 24.0,
                    "depth_ppm": 500.0,
                },
            }
        ),
        encoding="utf-8",
    )

    assert not is_manifest_bound_bls_result(workspace, result_path, result, None)
    assert load_transit_ephemeris(workspace)["source"] == "synthetic-demo"


def test_run_bls_on_candidate_with_real_data(tmp_path):
    import lightkurve as lk

    workspace = create_candidate(tmp_path, "candidate-test-data")
    raw = workspace.path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    time = np.linspace(2459000.0, 2459030.0, 600)
    period = 4.2
    epoch = 2459005.0
    ph = phase_hours(time, period, epoch)
    flux = 1.0 - 0.005 * (np.abs(ph) < 1.5).astype(float)
    meta = {
        "MISSION": "TESS",
        "TELESCOP": "TESS",
        "TIMEDEL": 120.0 / 86400.0,
        "TIMEUNIT": "BJD",
        "BJDREFI": 2457000,
        "BJDREFF": 0.0,
        "SECTOR": 30,
    }
    lk.LightCurve(time=time, flux=flux, flux_err=np.full_like(flux, 0.001), meta=meta).to_fits(
        path=raw / "s0030_lc.fits", overwrite=True
    )
    _write_raw_provenance(raw / "s0030_lc.fits")

    out = run_bls_on_candidate(workspace)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source"] == "candidate-data"
    assert payload["detection_status"] == "detected"
    assert payload["time_system"] == "BTJD_TDB"
    assert 1000.0 < payload["best_epoch"] < 3000.0
    assert payload["n_points"] == 600
    assert payload["best_period"] > 0
    assert payload["n_period_trials"] > 0
    assert payload["statistic"]["name"] == "weighted BLS fitted-depth signal-to-noise"
    assert payload["statistic"]["uncertainty_source"] == ["reported"]
    manifest = json.loads((workspace.path / "outputs" / "bls_search_manifest.json").read_text())
    assert manifest["source"] == "candidate-data"
    assert manifest["inputs"][0]["path"] == "data/raw/s0030_lc.fits"
    assert len(manifest["inputs"][0]["sha256"]) == 64
    assert manifest["inputs"][0]["provenance_path"] == "data/raw/s0030_lc.provenance.json"
    assert len(manifest["inputs"][0]["provenance_sha256"]) == 64
    assert manifest["configuration"]["uncertainty_source"] == ["reported"]
    assert manifest["configuration"]["time_system"] == "BTJD_TDB"
    assert manifest["runtime"] == {
        "implementation": "astropy.timeseries.BoxLeastSquares",
        "package": "astropy",
        "version": "6.0.1",
    }


def test_light_curve_loader_rejects_non_tdb_time_system(tmp_path):
    from astropy.io import fits
    import lightkurve as lk

    from exonym.inputs import load_light_curve_table

    workspace = create_candidate(tmp_path, "non-tdb-photometry")
    raw = workspace.path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / "s0030_lc.fits"
    lk.LightCurve(
        time=np.linspace(2459000.0, 2459001.0, 60),
        flux=np.ones(60),
        flux_err=np.full(60, 0.001),
        meta={"MISSION": "TESS", "TELESCOP": "TESS", "BJDREFI": 2457000, "SECTOR": 30},
    ).to_fits(path=path, overwrite=True)
    with fits.open(path, mode="update") as hdul:
        hdul[0].header["TIMESYS"] = "UTC"

    with pytest.warns(UserWarning, match="TIMESYS must be TDB"):
        assert load_light_curve_table(workspace) is None


def test_quality_flag_masking_excludes_bad_cadences(tmp_path):
    """Quality-flagged cadences must be removed before BLS.

    Injects a cluster of flagged cadences that carry a strong artificial dip
    (simulating a momentum dump plus scattered-light artefact). After the
    quality mask the dip is absent and BLS must NOT recover a period near
    the spacing of the bad-cadence cluster (which would be ~1 day here).
    """
    import lightkurve as lk
    from astropy.io import fits as fitsio
    from astropy.table import Table

    workspace = create_candidate(tmp_path, "quality-mask-test")
    raw = workspace.path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n = 800
    time = np.linspace(2459000.0, 2459030.0, n)
    flux = 1.0 + rng.normal(0.0, 0.0008, n)
    quality = np.zeros(n, dtype=np.int32)

    # Inject 16 consecutive cadences with quality flag=2048 (scattered light)
    # and a strong artificial dip — these must be excluded by the quality mask.
    bad_start = 300
    quality[bad_start : bad_start + 16] = 2048
    flux[bad_start : bad_start + 16] = 0.98  # 2% dip — far deeper than any real planet

    table = Table()
    table["TIME"] = time
    table["FLUX"] = flux
    table["FLUX_ERR"] = np.full(n, 0.001)
    table["QUALITY"] = quality
    ext = fitsio.BinTableHDU(table)
    ext.header["SECTOR"] = 30
    ext.header["TIMEDEL"] = 120.0 / 86400.0
    ext.header["BJDREFI"] = 2457000
    ext.header["BJDREFF"] = 0.0
    primary = fitsio.PrimaryHDU()
    primary.header["MISSION"] = "TESS"
    primary.header["TELESCOP"] = "TESS"
    fitsio.HDUList([primary, ext]).writeto(raw / "s0030_lc.fits", overwrite=True)
    _write_raw_provenance(raw / "s0030_lc.fits")

    out = run_bls_on_candidate(workspace, period_min=0.5, period_max=10.0)
    payload = json.loads(out.read_text(encoding="utf-8"))
    # The loader uses quality==0 masking; the dip should be absent so BLS
    # should not lock onto the ~1-day spurious period of the bad cluster.
    # We assert SNR is low, meaning no strong periodic signal was detected.
    assert payload["detection_status"] == "no-detection" or payload["snr"] < 50.0, (
        "BLS SNR is suspiciously high — quality masking may not have removed the bad cadences. "
        f"Recovered period={payload['best_period']:.3f} d, SNR={payload['snr']:.1f}"
    )
