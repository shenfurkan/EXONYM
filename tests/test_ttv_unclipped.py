"""Target-neutral unit tests verifying unclipped transit timing uncertainties (TTV-001)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from exonym.transit_fit import stellar_density_a_rs
from exonym.ttv import (
    _template_flux,
    fit_transit_epoch,
    transit_template_parameters,
    transit_timing_analysis,
)
from tests.fixtures.synthetic_observations import _synthetic_timing_table


def test_fit_transit_epoch_preserves_high_precision_unclipped_sigma():
    """Verify that high-SNR physical transits retain sigma_t0 without artificial clamping."""
    t0_true = 100.0
    period = 2.0
    ephemeris = {
        "period_days": period,
        "epoch_btjd": t0_true,
        "duration_days": 0.08,
        "depth_ppm": 20000.0,
    }
    a_rs = stellar_density_a_rs(1.0, period)
    template = transit_template_parameters(
        ephemeris, a_rs=a_rs, impact_parameter=0.2, q1=0.3, q2=0.3
    )

    time = np.linspace(t0_true - 0.2, t0_true + 0.2, 300)
    # Physically evaluated Mandel-Agol limb-darkened transit model
    flux = _template_flux(template, time, t0_true)
    errors = np.full_like(time, 1e-4)

    fit = fit_transit_epoch(
        time=time,
        flux=flux,
        errors=errors,
        template=template,
        t0_expected=t0_true,
    )

    assert fit["rejection_reason"] is None
    assert fit["sigma_t0_clipped"] is False
    assert fit["sigma_t0"] == fit["sigma_t0_raw"]
    assert fit["sigma_t0"] > 0.0
    assert math.isfinite(fit["sigma_t0"])


def test_transit_timing_analysis_uncertainty_clipped_list_empty():
    """Verify end-to-end timing analysis reports no artificial clipping on clean physical data."""
    table = _synthetic_timing_table(ttv_amplitude_minutes=0.0)
    ephemeris = {
        "period_days": table.pop("_period_days"),
        "epoch_btjd": table.pop("_epoch_btjd"),
        "duration_days": table.pop("_duration_days"),
        "depth_ppm": table.pop("_depth_ppm"),
    }
    a_rs = stellar_density_a_rs(1.0, ephemeris["period_days"])
    template = transit_template_parameters(
        ephemeris, a_rs=a_rs, impact_parameter=0.3, q1=0.3, q2=0.3
    )

    analysis = transit_timing_analysis(
        table["time"],
        table["flux"],
        table["flux_err"],
        ephemeris,
        template,
    )

    assert analysis["n_transits_fit"] >= 5
    assert len(analysis["uncertainty_clipped_epochs"]) == 0
    for record in analysis["per_epoch"]:
        if record["rejection_reason"] is None:
            assert record["sigma_t0_clipped"] is False
