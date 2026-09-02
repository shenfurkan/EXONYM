import sys
import types
import json
import hashlib

import numpy as np
import pytest

from exonym.inputs import (
    _time_values_to_btjd_tdb,
    load_light_curve_table,
    load_stellar_parameters,
    load_transit_ephemeris,
)
from exonym.detrending import detrend_candidate, transit_mask_from_ephemeris
from exonym.workspace import create_candidate


class _Quantity:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)


class _UnscopedLightCurve:
    def __init__(self):
        count = 64
        self.time = _Quantity(np.linspace(0.0, 2.0, count))
        self.flux = _Quantity(np.ones(count))
        self.flux_err = _Quantity(np.full(count, 1e-4))
        self.quality = _Quantity(np.zeros(count, dtype=int))
        self.meta = {}

    def __getitem__(self, key):
        # Minimal subscriptability for quality-mask indexing
        return self

    def remove_nans(self):
        return self

    def normalize(self):
        return self


def test_light_curve_loader_rejects_products_without_a_verified_sector(tmp_path, monkeypatch):
    # Arrange
    workspace = create_candidate(tmp_path, "unscoped-lightcurve")
    product = workspace.path / "data" / "raw" / "custom_lightcurve.fits"
    product.write_bytes(b"synthetic")
    monkeypatch.setitem(
        sys.modules,
        "lightkurve",
        types.SimpleNamespace(read=lambda _path: _UnscopedLightCurve()),
    )

    # Act
    with pytest.warns(UserWarning, match="sector cannot be verified"):
        table = load_light_curve_table(workspace)

    # Assert
    assert table is None

def test_inputs_missing_quality_column_rejected(tmp_path, monkeypatch):
    """Products without a QUALITY column must be skipped with a warning."""
    workspace = create_candidate(tmp_path, "no-quality-lc")
    product = workspace.path / "data" / "raw" / "no_quality_lc.fits"
    product.write_bytes(b"synthetic")

    class _NoQualityLC:
        def __init__(self):
            count = 64
            self.time = _Quantity(np.linspace(0.0, 2.0, count))
            self.flux = _Quantity(np.ones(count))
            self.flux_err = _Quantity(np.full(count, 1e-4))
            self.meta = {}

        def remove_nans(self):
            return self

        def normalize(self):
            return self

    monkeypatch.setitem(
        sys.modules,
        "lightkurve",
        types.SimpleNamespace(read=lambda _path: _NoQualityLC()),
    )

    with pytest.warns(UserWarning, match="no QUALITY column"):
        table = load_light_curve_table(workspace)

    assert table is None

@pytest.mark.parametrize(
    "header",
    (
        {"TELESCOP": "TESS", "BJDREFI": 2457000},
        {"TIMESYS": "TDB", "BJDREFI": 2457000},
    ),
)
def test_time_normalization_requires_declared_scale_and_day_units(header):
    with pytest.raises(ValueError):
        _time_values_to_btjd_tdb(np.array([100.0, 101.0]), header)


def test_explicit_btjd_field_rejects_a_conflicting_time_system(tmp_path):
    workspace = create_candidate(tmp_path, "conflicting-epoch-system")
    (workspace.path / "config" / "transit_config.json").write_text(
        json.dumps(
            {
                "period_days": 3.0,
                "epoch_btjd": 100.0,
                "epoch_time_system": "UTC",
                "duration_days": 0.1,
                "depth_ppm": 500.0,
            }
        ),
        encoding="utf-8",
    )

    ephemeris = load_transit_ephemeris(workspace)

    assert ephemeris["source"] == "partial-candidate-config"
    assert ephemeris["field_sources"]["epoch_btjd"] is None
    assert ephemeris["time_system"] is None


def test_detrended_loader_requires_hash_bound_raw_provenance(tmp_path):
    workspace = create_candidate(tmp_path, "detrended-loader")
    raw_path = workspace.path / "data" / "raw" / "source.fits"
    raw_path.write_bytes(b"synthetic raw product")
    raw_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    raw_path.with_name("source.provenance.json").write_text(
        json.dumps(
            {
                "source_uri": "https://example.invalid/source",
                "download_timestamp_utc": "2026-01-01T00:00:00Z",
                "sha256": raw_digest,
                "fetched_by": "test",
            }
        ),
        encoding="utf-8",
    )
    (workspace.path / "config" / "transit_config.json").write_text(
        json.dumps(
            {
                "period_days": 3.0,
                "epoch_btjd": 1.0,
                "duration_days": 0.12,
                "depth_ppm": 1000.0,
                "time_system": "BTJD_TDB",
            }
        ),
        encoding="utf-8",
    )
    ephemeris = load_transit_ephemeris(workspace)
    time = np.linspace(0.0, 8.0, 101)
    flux = 1.0 + 0.001 * np.sin(time)
    transit_mask = transit_mask_from_ephemeris(time, ephemeris)
    detrend_candidate(
        workspace,
        time,
        flux,
        window_days=0.5,
        sector=np.full(time.size, 1, dtype=int),
        input_products=[{"path": "data/raw/source.fits", "sha256": raw_digest}],
        transit_mask=transit_mask,
        transit_mask_ephemeris=ephemeris,
    )

    table = load_light_curve_table(
        workspace,
        max_points=None,
        require_raw_provenance=True,
        detrending_method="running-median",
    )

    assert table is not None
    assert table["input_files"] == [raw_path]
    assert np.all(table["sector"] == 1)
    assert table["detrending"]["artifact"]["path"] == "data/processed/detrended-running-median.npz"

    raw_path.write_bytes(b"tampered raw product")
    with pytest.raises(ValueError, match="raw provenance"):
        load_light_curve_table(
            workspace,
            require_raw_provenance=True,
            detrending_method="running-median",
        )


def test_detrended_loader_rejects_unbound_and_legacy_manifests(tmp_path):
    workspace = create_candidate(tmp_path, "detrended-loader-unbound")
    raw_path = workspace.path / "data" / "raw" / "source.fits"
    raw_path.write_bytes(b"synthetic raw product")
    raw_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    raw_path.with_name("source.provenance.json").write_text(
        json.dumps(
            {
                "source_uri": "https://example.invalid/source",
                "download_timestamp_utc": "2026-01-01T00:00:00Z",
                "sha256": raw_digest,
                "fetched_by": "test",
            }
        ),
        encoding="utf-8",
    )
    time = np.linspace(0.0, 8.0, 101)
    result = detrend_candidate(
        workspace,
        time,
        1.0 + 0.001 * np.sin(time),
        window_days=0.5,
        sector=np.full(time.size, 1, dtype=int),
        input_products=[{"path": "data/raw/source.fits", "sha256": raw_digest}],
    )

    with pytest.raises(ValueError, match="regenerate with `exonym detrend`"):
        load_light_curve_table(workspace, detrending_method="running-median")
    assert load_light_curve_table(
        workspace,
        detrending_method="running-median",
        require_transit_mask=False,
    ) is not None

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest["configuration"] = {"window_days": 0.5}
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy manifest; regenerate with `exonym detrend`"):
        load_light_curve_table(workspace, detrending_method="running-median")


def test_detrended_loader_rejects_a_changed_transit_mask_ephemeris(tmp_path):
    workspace = create_candidate(tmp_path, "detrended-mask-provenance")
    raw_path = workspace.path / "data" / "raw" / "source.fits"
    raw_path.write_bytes(b"synthetic raw product")
    raw_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    raw_path.with_name("source.provenance.json").write_text(
        json.dumps(
            {
                "source_uri": "https://example.invalid/source",
                "download_timestamp_utc": "2026-01-01T00:00:00Z",
                "sha256": raw_digest,
                "fetched_by": "test",
            }
        ),
        encoding="utf-8",
    )
    ephemeris_path = workspace.path / "config" / "transit_config.json"

    def write_ephemeris(epoch_btjd):
        ephemeris_path.write_text(
            json.dumps(
                {
                    "period_days": 3.0,
                    "epoch_btjd": epoch_btjd,
                    "duration_days": 0.12,
                    "depth_ppm": 1000.0,
                    "time_system": "BTJD_TDB",
                }
            ),
            encoding="utf-8",
        )

    write_ephemeris(1.0)
    ephemeris = load_transit_ephemeris(workspace)
    time = np.linspace(0.0, 8.0, 101)
    flux = 1.0 + 0.001 * np.sin(time)
    transit_mask = transit_mask_from_ephemeris(time, ephemeris)
    result = detrend_candidate(
        workspace,
        time,
        flux,
        window_days=0.5,
        sector=np.full(time.size, 1, dtype=int),
        input_products=[{"path": "data/raw/source.fits", "sha256": raw_digest}],
        transit_mask=transit_mask,
        transit_mask_ephemeris=ephemeris,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["configuration"]["transit_mask_provenance"]
    assert provenance["ephemeris"]["epoch_btjd"] == 1.0
    assert len(provenance["ephemeris_sha256"]) == 64
    assert len(provenance["mask_sha256"]) == 64
    unchanged = load_light_curve_table(
        workspace,
        max_points=None,
        require_raw_provenance=True,
        detrending_method="running-median",
    )
    assert unchanged is not None

    ephemeris_path.write_text(
        json.dumps(
            {
                "source": "candidate-data-bls",
                "transit": {
                    "period_days": 3.0,
                    "epoch_btjd": 1.0,
                    "duration_days": 0.12,
                    "depth_ppm": 1000.0,
                },
                "bls_provenance": {},
            }
        ),
        encoding="utf-8",
    )
    # An invalidated BLS binding resolves to unavailable while a fresh BLS
    # rerun is pending. The existing detrended artifact remains usable because
    # its own immutable mask provenance is complete and hash-bound.
    assert load_light_curve_table(
        workspace,
        max_points=None,
        require_raw_provenance=True,
        detrending_method="running-median",
    ) is not None

    write_ephemeris(1.1)
    with pytest.raises(ValueError, match="transit mask provenance is stale or mismatched"):
        load_light_curve_table(
            workspace,
            max_points=None,
            require_raw_provenance=True,
            detrending_method="running-median",
        )


def test_load_transit_ephemeris_and_stellar_parameters_fail_closed_when_absent(tmp_path):
    """Missing ephemeris/stellar parameter files must yield source='unavailable' and None fields."""
    workspace = create_candidate(tmp_path, "empty-candidate")

    ephemeris = load_transit_ephemeris(workspace)
    assert ephemeris["source"] == "unavailable"
    assert ephemeris["period_days"] is None
    assert ephemeris["epoch_btjd"] is None
    assert ephemeris["duration_days"] is None
    assert ephemeris["depth_ppm"] is None
    assert ephemeris["time_system"] is None

    stellar = load_stellar_parameters(workspace)
    assert stellar["source"] == "unavailable"
    assert stellar["teff_k"] is None
    assert stellar["logg_cgs"] is None
    assert stellar["feh"] is None
    assert stellar["mass_solar"] is None
    assert stellar["radius_solar"] is None
    assert stellar["parallax_mas"] is None
