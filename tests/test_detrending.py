"""Deterministic, synthetic tests for opt-in candidate-local detrending."""

import hashlib
import importlib
import json
import sys

import numpy as np
import pytest

from exonym.detrending import (
    OptionalBackendUnavailable,
    detrend_candidate,
    transit_mask_from_ephemeris,
    transit_mask_provenance_from_ephemeris,
    validate_transit_mask_provenance,
)
from exonym.workspace import create_candidate


def _synthetic_flux():
    time = np.linspace(0.0, 8.0, 401)
    trend = 1.0 + 0.015 * np.sin(2.0 * np.pi * time / 8.0)
    transit = np.where(np.abs(time - 4.0) < 0.06, 0.997, 1.0)
    return time, trend * transit


def _raw_input_product(workspace):
    raw_path = workspace.path / "data" / "raw" / "source.fits"
    raw_path.write_bytes(b"synthetic source bytes")
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    raw_path.with_name("source.provenance.json").write_text(
        json.dumps(
            {
                "source_uri": "https://example.invalid/source",
                "download_timestamp_utc": "2026-01-01T00:00:00Z",
                "sha256": digest,
                "fetched_by": "synthetic-test",
            }
        ),
        encoding="utf-8",
    )
    return {
        "path": "data/raw/source.fits",
        "sha256": digest,
    }


def test_running_median_writes_processed_artifact_and_manifest_without_touching_raw(tmp_path):
    # Arrange
    workspace = create_candidate(tmp_path, "detrending-synthetic")
    raw_product = _raw_input_product(workspace)
    raw_path = workspace.path / raw_product["path"]
    raw_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    time, flux = _synthetic_flux()

    # Act
    result = detrend_candidate(
        workspace,
        time,
        flux,
        method="running-median",
        window_days=0.5,
        sector=np.ones(time.size, dtype=int),
        input_products=[raw_product],
    )

    # Assert
    assert result.artifact_path == workspace.path / "data" / "processed" / "detrended-running-median.npz"
    assert result.manifest_path == workspace.path / "outputs" / "detrending_manifest.running-median.json"
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_digest
    with np.load(result.artifact_path) as artifact:
        assert set(artifact.files) == {"time_btjd", "flux", "trend", "detrended_flux", "sector"}
        assert np.median(artifact["detrended_flux"]) == pytest.approx(1.0, abs=0.002)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidate_id"] == "detrending-synthetic"
    assert manifest["method"] == "running-median"
    assert manifest["artifact"]["path"] == "data/processed/detrended-running-median.npz"
    assert manifest["artifact"]["sha256"] == hashlib.sha256(result.artifact_path.read_bytes()).hexdigest()
    assert len(manifest["artifact"]["data_sha256"]) == 64
    assert manifest["input_products"] == [raw_product]


def test_running_median_is_deterministic(tmp_path):
    # Arrange
    workspace = create_candidate(tmp_path, "detrending-repeatable")
    time, flux = _synthetic_flux()
    raw_product = _raw_input_product(workspace)
    sectors = np.ones(time.size, dtype=int)

    # Act
    first = detrend_candidate(
        workspace,
        time,
        flux,
        method="running-median",
        window_days=0.5,
        sector=sectors,
        input_products=[raw_product],
    )
    with np.load(first.artifact_path) as artifact:
        first_flux = artifact["detrended_flux"].copy()
    second = detrend_candidate(
        workspace,
        time,
        flux,
        method="running-median",
        window_days=0.5,
        sector=sectors,
        input_products=[raw_product],
    )
    with np.load(second.artifact_path) as artifact:
        second_flux = artifact["detrended_flux"].copy()

    # Assert
    assert np.array_equal(first_flux, second_flux)


@pytest.mark.parametrize("method", ["wotan", "celerite"])
def test_unavailable_optional_backend_writes_no_science_output(tmp_path, monkeypatch, method):
    # Arrange
    workspace = create_candidate(tmp_path, "detrending-optional-" + method)
    time, flux = _synthetic_flux()
    raw_product = _raw_input_product(workspace)
    real_import = importlib.import_module

    def unavailable(name):
        if name == method:
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr("exonym.detrending.importlib.import_module", unavailable)

    # Act / Assert
    with pytest.raises(OptionalBackendUnavailable, match="requested"):
        detrend_candidate(
            workspace,
            time,
            flux,
            method=method,
            window_days=0.5,
            sector=np.ones(time.size, dtype=int),
            input_products=[raw_product],
        )
    assert not (workspace.path / "data" / "processed" / ("detrended-" + method + ".npz")).exists()
    assert not (workspace.path / "outputs" / ("detrending_manifest." + method + ".json")).exists()


def test_celerite_rejects_missing_observational_uncertainties(tmp_path, monkeypatch):
    workspace = create_candidate(tmp_path, "celerite-requires-errors")
    time, flux = _synthetic_flux()
    raw_product = _raw_input_product(workspace)

    monkeypatch.setitem(
        sys.modules,
        "celerite",
        type("Celerite", (), {"terms": object(), "GP": object()})(),
    )

    with pytest.raises(ValueError, match="reported per-cadence flux_err"):
        detrend_candidate(
            workspace,
            time,
            flux,
            method="celerite",
            window_days=0.5,
            sector=np.ones(time.size, dtype=int),
            input_products=[raw_product],
        )
    assert not (workspace.path / "data" / "processed" / "detrended-celerite.npz").exists()

def _synthetic_transit_light_curve():
    """Return time, flux, and transit_mask with an injected box transit (depth 0.01)."""
    time = np.linspace(0.0, 8.0, 401)
    trend = 1.0 + 0.015 * np.sin(2.0 * np.pi * time / 8.0)
    depth = 0.01
    ephemeris = {
        "period_days": 20.0,
        "epoch_btjd": 4.0,
        "duration_days": 0.12,
        "time_system": "BTJD_TDB",
        "source": "candidate-config",
        "field_sources": {
            "period_days": "candidate-config",
            "epoch_btjd": "candidate-config",
            "duration_days": "candidate-config",
        },
    }
    transit_mask = transit_mask_from_ephemeris(time, ephemeris)
    flux = trend * np.where(transit_mask, 1.0 - depth, 1.0)
    return time, flux, transit_mask, depth, ephemeris


def _assert_depth_preserved(detrended_flux, transit_mask, injected_depth):
    in_transit = detrended_flux[transit_mask]
    measured_depth = 1.0 - float(np.median(in_transit[np.isfinite(in_transit)]))
    fractional_loss = abs(measured_depth - injected_depth) / injected_depth
    assert fractional_loss <= 0.05, (
        "masked detrending changed depth from {0:.6f} to {1:.6f} "
        "({2:.1%} loss)".format(injected_depth, measured_depth, fractional_loss)
    )


def test_detrending_depth_preservation_running_median(tmp_path):
    """Running-median backend must preserve injected transit depth with a correct mask."""
    workspace = create_candidate(tmp_path, "depth-running-median")
    raw_product = _raw_input_product(workspace)
    time, flux, transit_mask, depth, ephemeris = _synthetic_transit_light_curve()

    result = detrend_candidate(
        workspace,
        time,
        flux,
        method="running-median",
        window_days=0.5,
        sector=np.ones(time.size, dtype=int),
        input_products=[raw_product],
        transit_mask=transit_mask,
        transit_mask_ephemeris=ephemeris,
    )

    data = np.load(result.artifact_path)
    _assert_depth_preserved(data["detrended_flux"], transit_mask, depth)


def test_detrending_depth_preservation_wotan(tmp_path, monkeypatch):
    """Wotan receives the true in-transit mask and preserves depth to five percent."""
    import sys
    from types import SimpleNamespace

    observed_masks = []

    # Model Wotan's public mask polarity: True identifies in-transit points
    # which the backend excludes internally before estimating its trend.
    def _mock_flatten(time, values, **kwargs):
        from scipy.ndimage import median_filter

        mask = np.asarray(kwargs["mask"], dtype=bool)
        observed_masks.append(mask.copy())
        width = max(3, int(round(0.5 / float(np.median(np.diff(time))))))
        if width % 2 == 0:
            width += 1
        working = values.copy()
        indices = np.arange(values.size)
        working[mask] = np.interp(indices[mask], indices[~mask], values[~mask])
        return None, median_filter(working, size=width, mode="nearest")

    fake_wotan = SimpleNamespace(flatten=_mock_flatten)
    monkeypatch.setitem(sys.modules, "wotan", fake_wotan)

    workspace = create_candidate(tmp_path, "depth-wotan")
    raw_product = _raw_input_product(workspace)
    time, flux, transit_mask, depth, ephemeris = _synthetic_transit_light_curve()

    result = detrend_candidate(
        workspace,
        time,
        flux,
        method="wotan",
        window_days=0.5,
        sector=np.ones(time.size, dtype=int),
        input_products=[raw_product],
        transit_mask=transit_mask,
        transit_mask_ephemeris=ephemeris,
    )

    data = np.load(result.artifact_path)
    assert len(observed_masks) == 1
    assert np.array_equal(observed_masks[0], transit_mask)
    _assert_depth_preserved(data["detrended_flux"], transit_mask, depth)


def test_detrending_depth_preservation_celerite(tmp_path, monkeypatch):
    """Celerite conditions only on out-of-transit samples then predicts all cadences."""
    import sys
    from types import SimpleNamespace

    observed = {}

    class _MockTerm:
        def __init__(self, log_sigma, log_rho):
            self.initial_parameters = np.array([log_sigma, log_rho], dtype=float)

    class _MockGP:
        def __init__(self, kernel, mean):
            self.parameters = kernel.initial_parameters.copy()

        def compute(self, time, yerr):
            observed["condition_time"] = np.asarray(time).copy()
            observed["condition_error"] = np.asarray(yerr).copy()

        def get_parameter_vector(self):
            return self.parameters.copy()

        def get_parameter_names(self):
            return ("kernel:log_sigma", "kernel:log_rho")

        def set_parameter_vector(self, parameters):
            self.parameters = np.asarray(parameters, dtype=float).copy()

        def log_likelihood(self, residuals):
            return -0.5 * float(np.sum((residuals / 1.0e-4) ** 2))

        def grad_log_likelihood(self, residuals):
            return np.zeros_like(self.parameters, dtype=float)

        def predict(self, y, t, return_cov=False):
            observed["condition_residual"] = np.asarray(y).copy()
            observed["prediction_time"] = np.asarray(t).copy()
            return np.zeros_like(t, dtype=float)

    fake_celerite = SimpleNamespace(terms=SimpleNamespace(Matern32Term=_MockTerm), GP=_MockGP)
    monkeypatch.setitem(sys.modules, "celerite", fake_celerite)

    workspace = create_candidate(tmp_path, "depth-celerite")
    raw_product = _raw_input_product(workspace)
    time, flux, transit_mask, depth, ephemeris = _synthetic_transit_light_curve()
    flux_err = np.linspace(0.0001, 0.0002, time.size)

    result = detrend_candidate(
        workspace,
        time,
        flux,
        method="celerite",
        window_days=0.5,
        flux_err=flux_err,
        sector=np.ones(time.size, dtype=int),
        input_products=[raw_product],
        transit_mask=transit_mask,
        transit_mask_ephemeris=ephemeris,
    )

    data = np.load(result.artifact_path)
    unmasked = ~transit_mask
    baseline = float(np.median(flux[unmasked]))
    assert np.array_equal(observed["condition_time"], time[unmasked])
    assert np.array_equal(observed["condition_error"], flux_err[unmasked])
    assert np.allclose(observed["condition_residual"], (flux - baseline)[unmasked])
    assert np.array_equal(observed["prediction_time"], time)
    _assert_depth_preserved(data["detrended_flux"], transit_mask, depth)


def test_detrending_rejects_an_all_transit_mask_before_running_a_backend(tmp_path):
    workspace = create_candidate(tmp_path, "all-transit-mask")
    raw_product = _raw_input_product(workspace)
    time, flux, _transit_mask, _depth, _ephemeris = _synthetic_transit_light_curve()

    with pytest.raises(ValueError, match="out-of-transit cadence"):
        detrend_candidate(
            workspace,
            time,
            flux,
            method="celerite",
            window_days=0.5,
            sector=np.ones(time.size, dtype=int),
            input_products=[raw_product],
            transit_mask=np.ones(time.size, dtype=bool),
        )


def test_transit_mask_requires_complete_candidate_derived_btjd_ephemeris():
    time = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
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

    mask = transit_mask_from_ephemeris(time, ephemeris)

    assert np.array_equal(mask, np.array([False, True, False, False, True]))
    ephemeris["field_sources"]["epoch_btjd"] = "synthetic-demo"
    with pytest.raises(ValueError, match="complete candidate-derived"):
        transit_mask_from_ephemeris(time, ephemeris)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("period_days", 2.1),
        ("epoch_btjd", 1.1),
        ("duration_days", 0.5),
        ("source", "candidate-config-revised"),
    ),
)
def test_transit_mask_provenance_rejects_changed_ephemeris_values(field, replacement):
    time = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
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
    provenance = transit_mask_provenance_from_ephemeris(time, ephemeris)
    changed = dict(ephemeris)
    changed["field_sources"] = dict(ephemeris["field_sources"])
    changed[field] = replacement

    with pytest.raises(ValueError, match="stale or mismatched"):
        validate_transit_mask_provenance(time, provenance, changed)


def test_rejects_invalid_inputs_before_creating_outputs(tmp_path):
    # Arrange
    workspace = create_candidate(tmp_path, "detrending-invalid")

    # Act / Assert
    with pytest.raises(ValueError, match="matching one-dimensional"):
        detrend_candidate(workspace, [0.0, 1.0, 2.0], [1.0, 1.0], window_days=0.5)
    assert not list((workspace.path / "data" / "processed").glob("detrended-*.npz"))
    assert not list((workspace.path / "outputs").glob("detrending_manifest.*.json"))


def test_requires_sector_labels_and_raw_input_provenance(tmp_path):
    workspace = create_candidate(tmp_path, "detrending-required-provenance")
    time, flux = _synthetic_flux()

    with pytest.raises(ValueError, match="TESS sector"):
        detrend_candidate(workspace, time, flux, window_days=0.5)

    with pytest.raises(ValueError, match="raw input products"):
        detrend_candidate(
            workspace,
            time,
            flux,
            window_days=0.5,
            sector=np.ones(time.size, dtype=int),
        )
