"""Multi-mirror Gaia DR3 TAP failover and exact Pogson error propagation tests.

The four-tier failover hierarchy (ESA Gea -> CDS VizieR -> AIP Leibniz ->
ASIAA), the 8-second timeout default, and the analytic magnitude-error factor
``sigma_m = (2.5 / ln 10) * (sigma_F / F)`` (Evans et al. 2018, A&A, 616, A4,
doi:10.1051/0004-6361/201832756) are exercised end-to-end through the
production ``query_gaia_astrometry`` caller with mocked transports.  No
network access is performed and no mock payload lives inside ``src/``.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
from astropy.table import Table

from exonym.archive import (
    ArchivalVettingService,
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    POGSON_MAGNITUDE_FACTOR,
)


def _source_row(sep_arcsec, source_id="benchmark-source", g_mag=12.0):
    """Full 17-column standard-TAP row: id, ra, dec, g, bp, rp, ruwe,
    pmra, pmdec, epoch, sep, g_flux, g_ferr, bp_flux, bp_ferr, rp_flux, rp_ferr."""
    return [
        source_id, 10.0, 20.0, g_mag, 12.4, 12.1, 1.05, 0.0, 0.0, 2016.0,
        sep_arcsec, 1000.0, 20.0, 800.0, 16.0, 600.0, 12.0,
    ]


def _parsed_source(sep_arcsec):
    """A parsed source dict as returned by the production TAP/VizieR parsers."""
    return {
        "source_id": "benchmark-source",
        "ra_deg": 10.0,
        "dec_deg": 20.0,
        "separation_arcsec": sep_arcsec,
        "ruwe": 1.05,
        "phot_g_mean_mag": 12.0,
        "phot_g_mean_mag_error": POGSON_MAGNITUDE_FACTOR * (20.0 / 1000.0),
        "phot_bp_mean_mag": 12.4,
        "phot_bp_mean_mag_error": POGSON_MAGNITUDE_FACTOR * (16.0 / 800.0),
        "phot_rp_mean_mag": 12.1,
        "phot_rp_mean_mag_error": POGSON_MAGNITUDE_FACTOR * (12.0 / 600.0),
        "pmra_mas_per_year": 0.0,
        "pmdec_mas_per_year": 0.0,
        "reference_epoch_jyear": 2016.0,
    }


def test_default_gaia_timeout_is_eight_seconds():
    assert DEFAULT_HTTP_TIMEOUT_SECONDS == pytest.approx(8.0)
    assert ArchivalVettingService().timeout == pytest.approx(8.0)


def test_pogson_factor_is_the_exact_analytic_identity():
    """The factor is 2.5 / ln(10) at full double precision, never truncated."""
    assert POGSON_MAGNITUDE_FACTOR == pytest.approx(2.5 / math.log(10.0), rel=1e-15)
    assert POGSON_MAGNITUDE_FACTOR == pytest.approx(1.0857362047581294, rel=1e-15)
    assert POGSON_MAGNITUDE_FACTOR != 1.0857
    assert POGSON_MAGNITUDE_FACTOR != 1.086


def test_pogson_magnitude_error_propagates_exactly_and_fails_closed():
    service = ArchivalVettingService()
    expected = POGSON_MAGNITUDE_FACTOR * (20.0 / 1000.0)
    assert service._pogson_magnitude_error(1000.0, 20.0) == pytest.approx(expected, rel=1e-15)
    for flux, flux_error in (
        (None, 1.0),
        (1.0, None),
        (-1.0, 1.0),
        (1.0, -1.0),
        (0.0, 1.0),
        (1.0, 0.0),
    ):
        assert service._pogson_magnitude_error(flux, flux_error) is None


def test_aip_leibniz_is_reached_after_esa_and_vizier_fail():
    service = ArchivalVettingService(max_retries=1)
    tap_calls = []
    vizier_calls = []

    def fake_tap(ra, dec, radius_arcsec, base_url, table_name):
        tap_calls.append((base_url, table_name))
        if base_url == service.ESA_GAIA_TAP_URL:
            raise RuntimeError("esa down")
        if base_url == service.AIP_GAIA_TAP_URL:
            return [_parsed_source(0.5)]
        raise AssertionError("unexpected backend reached")

    def fake_vizier(ra, dec, radius_arcsec):
        vizier_calls.append((ra, dec, radius_arcsec))
        return []

    with patch.object(service, "_gaia_sources_tap", side_effect=fake_tap), patch.object(
        service, "_gaia_sources_vizier", side_effect=fake_vizier
    ):
        result = service.query_gaia_astrometry(10.0, 20.0, radius_arcsec=10.0)

    assert result["query_status"] == "ok"
    assert result["backend"] == "gaia-aip"
    assert result["validated"] is True
    assert tap_calls == [
        (service.ESA_GAIA_TAP_URL, service.ESA_GAIA_TABLE),
        (service.AIP_GAIA_TAP_URL, service.MIRROR_GAIA_TABLE),
    ]
    assert len(vizier_calls) == 1
    assert service.MIRROR_GAIA_TAP_URL not in [url for url, _ in tap_calls]
    assert any(error.startswith("esa-tap") for error in result["query_errors"])

def test_asiaa_mirror_remains_the_final_fallback():
    service = ArchivalVettingService(max_retries=1)
    tap_calls = []

    def fake_tap(ra, dec, radius_arcsec, base_url, table_name):
        tap_calls.append((base_url, table_name))
        if base_url == service.MIRROR_GAIA_TAP_URL:
            return [_parsed_source(0.3)]
        return []

    with patch.object(service, "_gaia_sources_tap", side_effect=fake_tap), patch.object(
        service, "_gaia_sources_vizier", return_value=[]
    ):
        result = service.query_gaia_astrometry(10.0, 20.0, radius_arcsec=10.0)

    assert result["query_status"] == "ok"
    assert result["backend"] == "gaia-mirror"
    assert result["validated"] is True
    assert [url for url, _ in tap_calls] == [
        service.ESA_GAIA_TAP_URL,
        service.AIP_GAIA_TAP_URL,
        service.MIRROR_GAIA_TAP_URL,
    ]


def test_all_mirrors_down_fails_closed_without_invented_sources():
    service = ArchivalVettingService(max_retries=1)

    def fake_tap(ra, dec, radius_arcsec, base_url, table_name):
        raise RuntimeError("backend unreachable")

    with patch.object(service, "_gaia_sources_tap", side_effect=fake_tap), patch.object(
        service, "_gaia_sources_vizier", return_value=[]
    ):
        result = service.query_gaia_astrometry(10.0, 20.0, radius_arcsec=10.0)

    assert result["query_status"] == "unavailable"
    assert result["validated"] is False
    assert result["sources"] == []
    assert result["target_ra_deg"] == pytest.approx(10.0)
    assert result["target_dec_deg"] == pytest.approx(20.0)
    assert sum(1 for error in result["query_errors"] if error.startswith("gaia-")) == 2
    assert sum(1 for error in result["query_errors"] if error.startswith("esa-tap")) == 1


def test_standard_tap_flux_columns_propagate_exact_mag_errors():
    service = ArchivalVettingService(max_retries=1)
    rows = [
        _source_row(0.1),
        # Legacy 6-element row without flux columns still parses.
        ["legacy-row", 10.001, 20.001, 13.0, 1.3, 2.0],
    ]

    with patch.object(service, "_http_get_json", return_value={"data": rows}):
        sources = service._gaia_sources_tap(
            10.0, 20.0, 10.0, "https://example.invalid/tap/sync", "gaiadr3.gaia_source"
        )

    assert len(sources) == 2
    modern, legacy = sources
    assert modern["phot_g_mean_mag_error"] == pytest.approx(
        POGSON_MAGNITUDE_FACTOR * (20.0 / 1000.0), rel=1e-12
    )
    assert modern["phot_bp_mean_mag_error"] == pytest.approx(
        POGSON_MAGNITUDE_FACTOR * (16.0 / 800.0), rel=1e-12
    )
    assert modern["phot_rp_mean_mag_error"] == pytest.approx(
        POGSON_MAGNITUDE_FACTOR * (12.0 / 600.0), rel=1e-12
    )
    assert legacy["phot_g_mean_mag"] == pytest.approx(13.0)
    assert legacy["phot_g_mean_mag_error"] is None
    assert legacy["phot_bp_mean_mag_error"] is None
    assert legacy["phot_rp_mean_mag_error"] is None


def test_vizier_flux_columns_propagate_exact_mag_errors(monkeypatch):
    service = ArchivalVettingService(max_retries=1)

    class FakeVizier:
        def __init__(self, row_limit=-1, timeout=8.0):
            self.row_limit = row_limit
            self.timeout = timeout

        def query_region(self, coordinate, radius, catalog):
            rows = {
                "RA_ICRS": [10.0],
                "DE_ICRS": [20.0],
                "Gmag": [12.0],
                "BPmag": [12.4],
                "RPmag": [12.1],
                "RUWE": [1.05],
                "pmRA": [0.0],
                "pmDE": [0.0],
                "Epoch": [2016.0],
                "Source": [1234567890],
                "FG": [1000.0],
                "e_FG": [20.0],
                "FBP": [800.0],
                "e_FBP": [16.0],
                "FRP": [600.0],
                "e_FRP": [12.0],
            }
            return [Table(rows)]

    monkeypatch.setattr("astroquery.vizier.Vizier", FakeVizier)

    sources = service._gaia_sources_vizier(10.0, 20.0, 10.0)

    assert len(sources) == 1
    source = sources[0]
    assert source["phot_g_mean_mag"] == pytest.approx(12.0)
    assert source["phot_g_mean_mag_error"] == pytest.approx(
        POGSON_MAGNITUDE_FACTOR * (20.0 / 1000.0), rel=1e-12
    )
    assert source["phot_bp_mean_mag_error"] == pytest.approx(
        POGSON_MAGNITUDE_FACTOR * (16.0 / 800.0), rel=1e-12
    )
    assert source["phot_rp_mean_mag_error"] == pytest.approx(
        POGSON_MAGNITUDE_FACTOR * (12.0 / 600.0), rel=1e-12
    )

