import sys
import types
import json

import numpy as np
import pytest

from exonym.inputs import _time_values_to_btjd_tdb, load_light_curve_table, load_transit_ephemeris
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
        self.meta = {}

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
    assert ephemeris["field_sources"]["epoch_btjd"] == "synthetic-demo"
    assert ephemeris["time_system"] == "synthetic-demo"
