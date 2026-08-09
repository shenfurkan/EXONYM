import numpy as np
import pytest

from exonym.lightcurve import (
    bin_phase_folded_flux,
    parse_tess_sector,
    phase_hours,
    robust_transit_depth,
)


def test_parse_tess_sector_accepts_mast_labels():
    assert parse_tess_sector("TESS Sector 42") == 42
    assert parse_tess_sector("sector 7") == 7
    assert parse_tess_sector("Kepler Quarter 1") is None


def test_phase_hours_centers_transit_at_zero():
    result = phase_hours([0.0, 0.5, 1.0], period_days=1.0, epoch_btjd=0.0)
    np.testing.assert_allclose(result, [0.0, -12.0, 0.0])


def test_robust_transit_depth_reports_ppm_and_sample_counts():
    time = np.array([-0.09, -0.07, -0.01, 0.0, 0.01, 0.07, 0.09])
    flux = np.array([1.0, 1.0, 0.99, 0.99, 0.99, 1.0, 1.0])

    depth, uncertainty, in_count, out_count = robust_transit_depth(
        time, flux, period_days=1.0, epoch_btjd=0.0, duration_hours=1.0
    )

    assert depth == pytest.approx(10_000.0)
    assert uncertainty < 1e-9
    assert in_count == 3
    assert out_count == 4


def test_robust_transit_depth_requires_coverage():
    with pytest.raises(ValueError, match="coverage"):
        robust_transit_depth(
            [0.0, 0.01], [1.0, 0.99], period_days=1.0, epoch_btjd=0.0, duration_hours=1.0
        )


def test_bin_phase_folded_flux_produces_expected_bin_shapes():
    time = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    flux = np.array([1.0, 0.99, 0.99, 0.99, 1.0])
    centers, median, error = bin_phase_folded_flux(
        time, flux, period_days=1.0, epoch_btjd=0.0, limit_hours=1.0, bin_minutes=30.0
    )

    assert centers.shape == median.shape == error.shape == (4,)
    assert np.isfinite(median).sum() >= 1
