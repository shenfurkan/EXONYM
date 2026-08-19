import sys
import types

import numpy as np
import pytest

from exonym.inputs import load_light_curve_table
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
