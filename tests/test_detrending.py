"""Deterministic, synthetic tests for opt-in candidate-local detrending."""

import hashlib
import importlib
import json

import numpy as np
import pytest

from exonym.detrending import OptionalBackendUnavailable, detrend_candidate
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

def _synthetic_transit_light_curve():
    """Return time, flux, and transit_mask with an injected box transit (depth 0.01)."""
    import numpy as np

    time = np.linspace(0.0, 8.0, 401)
    trend = 1.0 + 0.015 * np.sin(2.0 * np.pi * time / 8.0)
    depth = 0.01
    transit_mask = np.abs(time - 4.0) < 0.06
    flux = trend * np.where(transit_mask, 1.0 - depth, 1.0)
    return time, flux, transit_mask


def test_detrending_depth_preservation_running_median(tmp_path):
    """Running-median backend must preserve injected transit depth with a correct mask."""
    import numpy as np

    from exonym.detrending import detrend_candidate

    workspace = create_candidate(tmp_path, "depth-running-median")
    raw_product = _raw_input_product(workspace)
    time, flux, transit_mask = _synthetic_transit_light_curve()

    result = detrend_candidate(
        workspace,
        time,
        flux,
        method="running-median",
        window_days=0.5,
        sector=np.ones(time.size, dtype=int),
        input_products=[raw_product],
        transit_mask=transit_mask,
    )

    data = np.load(result.artifact_path)
    detrended = data["detrended_flux"]
    in_transit = detrended[transit_mask]
    measured_depth = 1.0 - float(np.median(in_transit[np.isfinite(in_transit)]))
    assert 0.008 <= measured_depth <= 0.012, (
        "running-median depth {0:.6f} outside [0.008, 0.012]".format(measured_depth)
    )


def test_detrending_depth_preservation_wotan(tmp_path, monkeypatch):
    """Wotan backend must preserve injected transit depth with a correct mask."""
    import sys
    from types import SimpleNamespace

    import numpy as np

    from exonym.detrending import detrend_candidate

    # Mock wotan to return a simple running-median trend so the test is self-contained.
    def _mock_flatten(time, values, **kwargs):
        from scipy.ndimage import median_filter
        width = max(3, int(round(0.5 / float(np.median(np.diff(time))))))
        if width % 2 == 0:
            width += 1
        return None, median_filter(values, size=width, mode="nearest")

    fake_wotan = SimpleNamespace(flatten=_mock_flatten)
    monkeypatch.setitem(sys.modules, "wotan", fake_wotan)

    workspace = create_candidate(tmp_path, "depth-wotan")
    raw_product = _raw_input_product(workspace)
    time, flux, transit_mask = _synthetic_transit_light_curve()

    result = detrend_candidate(
        workspace,
        time,
        flux,
        method="wotan",
        window_days=0.5,
        sector=np.ones(time.size, dtype=int),
        input_products=[raw_product],
        transit_mask=transit_mask,
    )

    data = np.load(result.artifact_path)
    detrended = data["detrended_flux"]
    in_transit = detrended[transit_mask]
    measured_depth = 1.0 - float(np.median(in_transit[np.isfinite(in_transit)]))
    assert 0.008 <= measured_depth <= 0.012, (
        "wotan depth {0:.6f} outside [0.008, 0.012]".format(measured_depth)
    )


def test_detrending_depth_preservation_celerite(tmp_path, monkeypatch):
    """Celerite GP backend must preserve injected transit depth with a correct mask."""
    import sys
    from types import SimpleNamespace

    import numpy as np

    from exonym.detrending import detrend_candidate

    class _MockTerm:
        def __init__(self, log_sigma, log_rho):
            pass

    class _MockGP:
        def __init__(self, kernel, mean):
            pass

        def compute(self, time, yerr):
            pass

        def predict(self, y, t, return_cov=False):
            from scipy.ndimage import median_filter
            width = max(3, int(round(0.5 / float(np.median(np.diff(t))))))
            if width % 2 == 0:
                width += 1
            return median_filter(y, size=width, mode="nearest")

    fake_celerite = SimpleNamespace(terms=SimpleNamespace(Matern32Term=_MockTerm), GP=_MockGP)
    monkeypatch.setitem(sys.modules, "celerite", fake_celerite)

    workspace = create_candidate(tmp_path, "depth-celerite")
    raw_product = _raw_input_product(workspace)
    time, flux, transit_mask = _synthetic_transit_light_curve()

    result = detrend_candidate(
        workspace,
        time,
        flux,
        method="celerite",
        window_days=0.5,
        sector=np.ones(time.size, dtype=int),
        input_products=[raw_product],
        transit_mask=transit_mask,
    )

    data = np.load(result.artifact_path)
    detrended = data["detrended_flux"]
    in_transit = detrended[transit_mask]
    measured_depth = 1.0 - float(np.median(in_transit[np.isfinite(in_transit)]))
    assert 0.008 <= measured_depth <= 0.012, (
        "celerite depth {0:.6f} outside [0.008, 0.012]".format(measured_depth)
    )

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
