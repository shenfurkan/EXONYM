import json
from pathlib import Path

import numpy as np
import pytest

from exonym.catalog import (
    IdentifierError,
    make_provenance,
    mission_for_identifier,
    parse_identifier,
    write_provenance_sidecar,
)
from exonym.ingest import ingest_products
from exonym.workspace import create_candidate, discover_candidates


@pytest.mark.parametrize(
    "raw, kind, mission, value",
    [
        ("TOI 1234.01", "toi", "tess", "1234.01"),
        ("toi-8888.01", "toi", "tess", "8888.01"),
        ("TIC 123456789", "tic", "tess", "123456789"),
        ("TIC:987654321", "tic", "tess", "987654321"),
        ("K00007.01", "koi", "kepler", "00007.01"),
        ("K7.01", "koi", "kepler", "7.01"),
        ("EPIC 201367065", "epic", "k2", "201367065"),
        ("PIC-12345", "pic", "plato", "12345"),
        ("CHEOPS-ABCD123", "cheops", "cheops", "ABCD123"),
    ],
)
def test_parse_identifier(raw, kind, mission, value):
    parsed = parse_identifier(raw)
    assert parsed["kind"] == kind
    assert parsed["mission"] == mission
    assert parsed["value"] == value
    assert mission_for_identifier(raw) == mission


@pytest.mark.parametrize("raw", ["", "garbage", "TOI", "1234.01", "K1234"])
def test_parse_identifier_rejects_unknown(raw):
    with pytest.raises(IdentifierError):
        parse_identifier(raw)


def test_make_provenance_records_sha256(tmp_path):
    product = tmp_path / "lc.fits"
    product.write_bytes(b"fits-data")
    record = make_provenance(
        product,
        "https://archive.stsci.edu/example",
        fetched_by="test",
        download_timestamp_utc="2026-08-04T00:00:00Z",
    )
    assert record["sha256"] == "a" * 64 or len(record["sha256"]) == 64
    assert record["download_timestamp_utc"] == "2026-08-04T00:00:00Z"
    assert record["fetched_by"] == "test"


def test_write_provenance_sidecar_matches_gatekeeper_convention(tmp_path):
    product = tmp_path / "s0001_lc.fits.gz"
    product.write_bytes(b"fits")
    sidecar = write_provenance_sidecar(product, "https://example.invalid/p", fetched_by="test")
    assert sidecar.name == "s0001_lc.fits.provenance.json"
    assert sidecar.is_file()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["source_uri"] == "https://example.invalid/p"


def test_ingest_products_copies_and_sidecars(tmp_path):
    create_candidate(tmp_path, "candidate-alpha")
    candidate = [c for c in discover_candidates(tmp_path)][0]

    staging = tmp_path / "staging"
    staging.mkdir()
    product = staging / "s0037_lc.fits"
    product.write_bytes(b"fits")

    written = ingest_products(
        candidate, [(product, "https://archive.stsci.edu/example")], fetched_by="test"
    )
    raw = candidate.path / "data" / "raw"
    assert (raw / "s0037_lc.fits").is_file()
    assert (raw / "s0037_lc.provenance.json").is_file()
    assert written == [(raw / "s0037_lc.fits")]

    with pytest.raises(FileExistsError):
        ingest_products(candidate, [(product, "https://archive.stsci.edu/example")])


def test_ingest_products_rolls_back_when_sidecar_staging_fails(tmp_path, monkeypatch):
    # Arrange
    create_candidate(tmp_path, "candidate-alpha")
    candidate = discover_candidates(tmp_path)[0]
    product = tmp_path / "s0037_lc.fits"
    product.write_bytes(b"fits")

    def fail_sidecar(*args, **kwargs):
        raise OSError("simulated sidecar failure")

    monkeypatch.setattr("exonym.ingest.write_provenance_sidecar", fail_sidecar)

    # Act / Assert
    with pytest.raises(OSError, match="sidecar failure"):
        ingest_products(candidate, [(product, "https://archive.stsci.edu/example")])
    raw = candidate.path / "data" / "raw"
    assert not (raw / product.name).exists()
    assert not (raw / "s0037_lc.provenance.json").exists()
    assert not list(raw.glob(".ingest-staging-*"))


def test_ingest_products_requires_tic_for_network_fetch(tmp_path):
    create_candidate(tmp_path, "candidate-beta")
    candidate = [c for c in discover_candidates(tmp_path) if c.candidate_id == "candidate-beta"][0]

    from exonym.ingest import fetch_tess_products

    with pytest.raises(ValueError, match="TIC"):
        fetch_tess_products(candidate, sectors=[37])


class _FakeDownload:
    def __init__(self, light_curve):
        self._light_curve = light_curve

    def download(self):
        return self._light_curve


class _FakeSearch:
    def __init__(self, table, light_curves):
        self.table = table
        self._light_curves = light_curves

    def __len__(self):
        return len(self._light_curves)

    def __getitem__(self, index):
        return _FakeDownload(self._light_curves[index])


class _FakeTessCut:
    def __init__(self, light_curve):
        self._light_curve = light_curve

    def to_lightcurve(self):
        return self._light_curve


def test_fetch_tess_products_writes_fits_and_ingests(tmp_path, monkeypatch):
    import lightkurve as lk
    from astropy.table import Table

    from exonym.ingest import fetch_tess_products, ingest_products

    create_candidate(tmp_path, "candidate-ingest", tic="123456789")
    candidate = [
        c for c in discover_candidates(tmp_path) if c.candidate_id == "candidate-ingest"
    ][0]

    table = Table(
        rows=[
            (
                "s0037",
                "tess2021000000000-s0037-0000000123456789-0218-s_lc.fits",
                "mast:TESS/product/tess2021000000000-s0037-0000000123456789-0218-s_lc.fits",
            )
        ],
        names=("sequence_number", "productFilename", "dataURI"),
    )
    fake_search = _FakeSearch(table, [object()])
    raw_bytes = b"unaltered-mast-spoc-lightcurve-bytes"

    monkeypatch.setattr(lk, "search_lightcurve", lambda *args, **kwargs: fake_search)
    monkeypatch.setattr(
        lk,
        "search_tesscut",
        lambda *args, **kwargs: pytest.fail("TESSCut must not be queried for a SPOC request"),
    )
    from astroquery.mast import Observations

    calls = []

    def download_file(data_uri, *, local_path, **kwargs):
        calls.append((data_uri, local_path))
        Path(local_path).write_bytes(raw_bytes)
        return "COMPLETE", None, "https://mast.stsci.edu/api/v0.1/Download/file"

    monkeypatch.setattr(Observations, "download_file", download_file)

    products = fetch_tess_products(candidate, exptime=120)
    assert len(products) == 1
    staged, source_uri = products[0]
    fetch_staging = products.staging_path
    assert fetch_staging is not None and fetch_staging.is_dir()
    assert staged.is_file()
    assert staged.read_bytes() == raw_bytes
    assert source_uri.startswith("https://mast.stsci.edu")
    assert "uri=mast:TESS/product/tess2021000000000-s0037-0000000123456789-0218-s_lc.fits" in source_uri
    assert calls[0][0] == "mast:TESS/product/tess2021000000000-s0037-0000000123456789-0218-s_lc.fits"

    written = ingest_products(candidate, products)
    raw = candidate.path / "data" / "raw"
    assert written[0].is_file()
    assert written[0].parent == raw
    sidecar = written[0].with_name(written[0].stem + ".provenance.json")
    assert sidecar.is_file()
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["source_uri"] == source_uri
    assert len(record["sha256"]) == 64
    assert not fetch_staging.exists()


@pytest.mark.parametrize(
    "search_name, fetch_name",
    [
        ("search_lightcurve", "fetch_tess_products"),
        ("search_targetpixelfile", "fetch_tess_tpfs"),
    ],
)
def test_fetch_with_explicit_sectors_skips_rows_without_sector_metadata(
    tmp_path, monkeypatch, search_name, fetch_name
):
    import lightkurve as lk
    from astropy.table import Table

    import exonym.ingest as ingest

    create_candidate(tmp_path, "candidate-sector-filter", tic="123456789")
    candidate = discover_candidates(tmp_path)[0]
    table = Table(
        rows=[("synthetic-product.fits",)],
        names=("productFilename",),
    )
    fake_search = _FakeSearch(table, [object()])
    monkeypatch.setattr(lk, search_name, lambda *args, **kwargs: fake_search)
    monkeypatch.setattr(
        ingest,
        "_download_spoc_product",
        lambda *args, **kwargs: pytest.fail("unknown-sector row must not download"),
    )

    products = getattr(ingest, fetch_name)(candidate, sectors=[1])

    assert products == []
    assert products.staging_path is None


def test_fetch_cleans_partial_staging_after_download_failure(tmp_path, monkeypatch):
    import lightkurve as lk
    from astropy.table import Table

    import exonym.ingest as ingest

    create_candidate(tmp_path, "candidate-staging-cleanup", tic="123456789")
    candidate = discover_candidates(tmp_path)[0]
    table = Table(
        rows=[("s0001", "synthetic-product.fits")],
        names=("sequence_number", "productFilename"),
    )
    fake_search = _FakeSearch(table, [object()])
    monkeypatch.setattr(lk, "search_lightcurve", lambda *args, **kwargs: fake_search)
    staging_paths = []

    def fail_download(row, staging):
        staging_paths.append(staging)
        (staging / "partial-product.fits").write_bytes(b"partial")
        raise RuntimeError("simulated archive failure")

    monkeypatch.setattr(ingest, "_download_spoc_product", fail_download)

    with pytest.raises(RuntimeError, match="simulated archive failure"):
        ingest.fetch_tess_products(candidate, sectors=[1])

    assert staging_paths
    assert not staging_paths[0].exists()


def test_fetch_tesscut_is_rejected_before_network_access(tmp_path, monkeypatch):
    import lightkurve as lk
    from astropy.table import Table

    from exonym.ingest import fetch_tess_products

    create_candidate(tmp_path, "candidate-tesscut", tic="123456789")
    candidate = [
        c for c in discover_candidates(tmp_path) if c.candidate_id == "candidate-tesscut"
    ][0]
    monkeypatch.setattr(
        lk, "search_lightcurve", lambda *args, **kwargs: pytest.fail("network must not run")
    )

    with pytest.raises(ValueError, match="unsupported TESS data provider"):
        fetch_tess_products(candidate, sectors=[37], provider="tesscut")


def test_tesscut_rejects_tpf_requests_without_a_network_call(tmp_path):
    from exonym.ingest import fetch_tess_tpfs

    create_candidate(tmp_path, "candidate-tesscut-tpf", tic="123456789")
    candidate = [
        c for c in discover_candidates(tmp_path) if c.candidate_id == "candidate-tesscut-tpf"
    ][0]

    with pytest.raises(ValueError, match="unsupported TESS data provider"):
        fetch_tess_tpfs(candidate, provider="tesscut")


def test_fetch_tess_products_rejects_an_unknown_provider_before_network_access(tmp_path):
    from exonym.ingest import fetch_tess_products

    create_candidate(tmp_path, "candidate-provider", tic="123456789")
    candidate = [
        c for c in discover_candidates(tmp_path) if c.candidate_id == "candidate-provider"
    ][0]

    with pytest.raises(ValueError, match="unsupported TESS data provider"):
        fetch_tess_products(candidate, provider="other")


class _FakeTPF:
    def __init__(self):
        self._data = None

    def download(self):
        return self

    def to_fits(self, path, overwrite=False):
        from astropy.io import fits

        hdul = fits.HDUList([fits.PrimaryHDU(data=np.zeros(8))])
        hdul.writeto(path, overwrite=overwrite)
        hdul.close()


class _FakeTPFSearch:
    def __init__(self, table):
        self.table = table

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        return _FakeTPF()


def test_fetch_tess_tpfs_stages_and_ingests_with_sidecars(tmp_path, monkeypatch):
    import lightkurve as lk
    from astropy.table import Table

    from exonym.ingest import fetch_tess_tpfs, ingest_products

    create_candidate(tmp_path, "candidate-tpf", tic="123456789")
    candidate = [
        c for c in discover_candidates(tmp_path) if c.candidate_id == "candidate-tpf"
    ][0]

    table = Table(
        rows=[
            ("s0047", "tess2021000000000-s0047-0000000123456789-0218-s_tp.fits"),
            ("s0053", "tess2021000000000-s0053-0000000123456789-0226-s_tp.fits"),
        ],
        names=("sequence_number", "productFilename"),
    )
    fake_search = _FakeTPFSearch(table)

    monkeypatch.setattr(lk, "search_targetpixelfile", lambda *args, **kwargs: fake_search)
    from astroquery.mast import Observations

    def download_file(data_uri, *, local_path, **kwargs):
        Path(local_path).write_bytes(b"unaltered-mast-spoc-tpf-bytes")
        return "COMPLETE", None, "https://mast.stsci.edu/api/v0.1/Download/file"

    monkeypatch.setattr(Observations, "download_file", download_file)

    products = fetch_tess_tpfs(candidate, exptime=120)
    assert len(products) == 2
    for staged, source_uri in products:
        assert staged.is_file()
        assert staged.stat().st_size > 0
        assert staged.name.endswith("_tp.fits")
        assert source_uri.startswith("https://mast.stsci.edu")

    written = ingest_products(candidate, products)
    raw = candidate.path / "data" / "raw"
    for path in written:
        assert path.is_file()
        assert path.parent == raw
        assert path.name.endswith("_tp.fits")
        sidecar = path.with_name(path.stem + ".provenance.json")
        assert sidecar.is_file(), "sidecar must follow <stem>.provenance.json convention"
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        assert len(record["sha256"]) == 64


def test_perryman_spectroscopic_and_atmospheric_calculations():
    from exonym.catalog import (
        calculate_astrometric_wobble_microarcsec,
        calculate_atmospheric_scale_height_km,
        calculate_radial_velocity_semi_amplitude,
        calculate_transmission_signal_ppm,
    )
    from exonym.lightcurve import (
        calculate_contact_durations,
        kipping_to_quadratic_limb_darkening,
        quadratic_to_kipping_limb_darkening,
    )
    from exonym.search import (
        calculate_ttv_super_period,
        compute_linear_ephemeris_residuals,
    )
    from exonym.tagging import evaluate_habitable_zone_tag
    from exonym.vetting import (
        centroid_offset_pvalue,
        ellipsoidal_gate,
        ellipsoidal_variation_amplitude_ppm,
    )

    k_one_year = calculate_radial_velocity_semi_amplitude(
        m_planet_earth=1.0, m_star_solar=1.0, period_days=365.25, inclination_deg=90.0
    )
    assert k_one_year == pytest.approx(0.0895, rel=1e-2), (
        "K at P=1 yr should equal the canonical constant 0.0895 m/s"
    )

    k_one_day = calculate_radial_velocity_semi_amplitude(
        m_planet_earth=1.0, m_star_solar=1.0, period_days=1.0, inclination_deg=90.0
    )
    assert k_one_day == pytest.approx(0.640, rel=1e-2), (
        "K at P=1 d should be 0.0895 * 365.25^(1/3) ≈ 0.640 m/s"
    )

    wobble = calculate_astrometric_wobble_microarcsec(
        m_planet_earth=317.83, m_star_solar=1.0, semi_major_axis_au=5.2, distance_pc=10.0
    )
    assert wobble > 0

    h_km = calculate_atmospheric_scale_height_km(
        t_eq_kelvin=300.0, m_planet_earth=1.0, r_planet_earth=1.0
    )
    assert h_km > 0

    delta_ppm = calculate_transmission_signal_ppm(
        r_star_solar=1.0, r_planet_earth=1.0, scale_height_km=h_km
    )
    assert delta_ppm > 0

    durations = calculate_contact_durations(
        period_days=3.0,
        r_star_solar=1.0,
        m_star_solar=1.0,
        r_planet_earth=1.0,
        impact_parameter_b=0.2,
    )
    assert durations["T14_hr"] > 0
    assert durations["grazing"] == 0.0

    u1, u2 = kipping_to_quadratic_limb_darkening(0.25, 0.5)
    q1, q2 = quadratic_to_kipping_limb_darkening(u1, u2)
    assert q1 == pytest.approx(0.25, abs=1e-5)
    assert q2 == pytest.approx(0.5, abs=1e-5)

    p_ttv = calculate_ttv_super_period(
        period_inner_days=10.0, period_outer_days=20.1, j_resonance=2
    )
    assert p_ttv > 0

    result = compute_linear_ephemeris_residuals(
        transit_times_btjd=[10.0, 20.001, 30.0], period_days=10.0, epoch_btjd=10.0
    )
    residuals = result["residuals_minutes"]
    assert len(residuals) == 3
    assert result["n_nonfinite_midtimes"] == 0

    p_val = centroid_offset_pvalue(2.0)
    assert p_val == pytest.approx(np.exp(-2.0), rel=1e-4)

    amp_ppm = ellipsoidal_variation_amplitude_ppm(
        m_companion_solar=0.001, m_host_solar=1.0, r_host_solar=1.0, semi_major_axis_au=0.05
    )
    passed, _ = ellipsoidal_gate(amp_ppm)
    assert passed

    assert evaluate_habitable_zone_tag(0.5) == "HabitableZoneCandidate"
