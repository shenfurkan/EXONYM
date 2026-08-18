import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from exonym.vetting.centroid import centroid_gate, centroid_offset_z
from exonym.vetting.oddeven import odd_even_gate, odd_even_z
from exonym.vetting.tricera_parse import fpp_gate, load_fpp_report


def test_centroid_offset_z_uses_cos_dec():
    z = centroid_offset_z(ra_offset_arcsec=0.0, dec_offset_arcsec=3.0, dec_deg=0.0, sigma_arcsec=1.0)
    assert z == pytest.approx(3.0)
    z_on_target = centroid_offset_z(0.5, 0.5, 0.0, 1.0)
    assert z_on_target < 3.0


def test_centroid_gate_threshold():
    passed, z = centroid_gate(0.1, 0.1, 0.0, 1.0)
    assert passed and z < 3.0
    failed, z = centroid_gate(3.0, 0.0, 0.0, 1.0)
    assert not failed and z >= 3.0


def test_centroid_requires_positive_sigma():
    with pytest.raises(ValueError):
        centroid_offset_z(0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "measurement",
    [
        (float("nan"), 0.0, 0.0, 1.0),
        (0.0, float("inf"), 0.0, 1.0),
        (0.0, 0.0, float("nan"), 1.0),
        (0.0, 0.0, 0.0, float("inf")),
    ],
)
def test_centroid_rejects_nonfinite_measurements(measurement):
    # Arrange
    ra_offset_arcsec, dec_offset_arcsec, dec_deg, sigma_arcsec = measurement

    # Act and assert
    with pytest.raises(ValueError, match="inputs must be finite"):
        centroid_offset_z(ra_offset_arcsec, dec_offset_arcsec, dec_deg, sigma_arcsec)


def test_centroid_rejects_unphysical_declination_and_threshold():
    # Arrange
    arguments = (0.1, 0.1, 91.0, 1.0)

    # Act and assert
    with pytest.raises(ValueError, match="dec_deg must be between"):
        centroid_offset_z(*arguments)
    with pytest.raises(ValueError, match="threshold must be finite and positive"):
        centroid_gate(0.1, 0.1, 0.0, 1.0, threshold=float("nan"))


def test_odd_even_z():
    z = odd_even_z(100.0, 10.0, 90.0, 10.0)
    assert z == pytest.approx(0.7071, abs=1e-3)
    assert odd_even_gate(100.0, 10.0, 90.0, 10.0)[0] is True
    assert odd_even_gate(100.0, 5.0, 70.0, 5.0)[0] is False


def test_fpp_gate_dict_and_value():
    report = {"fpp": 0.005, "nfpp": 0.0}
    passed, fpp = fpp_gate(report)
    assert passed and fpp == pytest.approx(0.005)
    assert fpp_gate(0.02)[0] is False


def test_fpp_report_probes_common_keys(tmp_path):
    path = tmp_path / "triceratops.json"
    path.write_text(json.dumps({"FPP_specific": 0.008}), encoding="utf-8")
    report = load_fpp_report(path)
    assert fpp_gate(report)[0] is True


def test_fpp_missing_value_raises(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"note": "no fpp"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no FPP"):
        fpp_gate(load_fpp_report(path))


def test_fpp_report_rejects_non_object_json(tmp_path):
    path = tmp_path / "triceratops.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_fpp_report(path)


def test_observed_sector_parser_prefers_products_then_uses_holdings_fallback(tmp_path):
    from exonym.vetting.tricera_parse import _observed_sectors

    workspace = type("Workspace", (), {"path": tmp_path})()
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "s0007_lc.fits").write_bytes(b"synthetic")
    (raw / "s0021_tp.fz").write_bytes(b"synthetic")
    holdings = tmp_path / "data" / "external" / "tess_holdings.json"
    holdings.parent.mkdir(parents=True)
    holdings.write_text(
        json.dumps({"pipelines": {"synthetic": [{"sector": 99}]}}), encoding="utf-8"
    )

    assert _observed_sectors(workspace) == [7, 21]

    for path in raw.iterdir():
        path.unlink()
    holdings.write_text(
        json.dumps({"pipelines": {"synthetic": [{"sector": 4}, {"sector": "5"}]}}),
        encoding="utf-8",
    )
    assert _observed_sectors(workspace) == [4]


def _vet_workspace_stub(tmp_path, candidate_id="vet-stub", tic=None):
    import types

    outputs = tmp_path / "outputs"
    claims = tmp_path / "claims"
    outputs.mkdir(parents=True, exist_ok=True)
    stub = types.SimpleNamespace(
        path=tmp_path,
        candidate_id=candidate_id,
        metadata={"identifiers": {"tic": tic}} if tic else {"identifiers": {}},
    )
    return stub, outputs


@pytest.mark.parametrize("period_key", ("period_days", "period", "p"))
def test_run_triceratops_prefers_signal_config_over_bls(tmp_path, period_key):
    from exonym.vetting.tricera_parse import run_triceratops_simulation

    stub, _ = _vet_workspace_stub(tmp_path)
    signal_dir = tmp_path / "config" / "signals"
    signal_dir.mkdir(parents=True)
    (signal_dir / "transit_config.01.json").write_text(
        json.dumps(
            {
                "signal": ".01",
                "transit": {
                    "depth_ppm": 341.4,
                    "duration_days": 0.0925,
                    "duration_hours": 2.22,
                    period_key: 4.5701356,
                    "t0_btjd": 2117.193359,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "outputs" / "bls_search_results.json").write_text(
        json.dumps({"best_period": 14.97546, "best_depth_ppm": 4234.08, "best_duration_hours": 3.0}),
        encoding="utf-8",
    )

    # No TIC in stub → Monte Carlo cannot run; allow_fallback=True to test
    # ephemeris routing (signal config takes priority over BLS).
    report_path = run_triceratops_simulation(stub, signal=".01", allow_fallback=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["signal"] == ".01"
    assert report["ephemeris"]["period_days"] == pytest.approx(4.5701356)
    assert report["ephemeris"]["depth_ppm"] == pytest.approx(341.4)
    assert report["ephemeris"]["duration_hours"] == pytest.approx(2.22)
    assert report["ephemeris"]["source"] == "candidate-config-signal"


def test_run_triceratops_defaults_to_bls_without_signal(tmp_path):
    from exonym.vetting.tricera_parse import run_triceratops_simulation

    stub, _ = _vet_workspace_stub(tmp_path)
    (tmp_path / "outputs" / "bls_search_results.json").write_text(
        json.dumps({"best_period": 7.5, "best_depth_ppm": 900.0, "best_duration_hours": 2.0}),
        encoding="utf-8",
    )

    # No TIC in stub → Monte Carlo cannot run; allow_fallback=True to test
    # ephemeris routing (BLS results used when no signal is given).
    report_path = run_triceratops_simulation(stub, signal=None, allow_fallback=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["signal"] is None
    assert report["ephemeris"]["period_days"] == pytest.approx(7.5)
    assert report["ephemeris"]["source"] == "bls-search"


def test_run_triceratops_falls_back_when_signal_config_missing(tmp_path):
    from exonym.vetting.tricera_parse import run_triceratops_simulation

    stub, _ = _vet_workspace_stub(tmp_path)
    # No TIC → Monte Carlo cannot run; allow_fallback=True required.
    with pytest.warns(UserWarning, match="could not read signal transit config"):
        report_path = run_triceratops_simulation(stub, signal=".99", allow_fallback=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ephemeris"]["source"] == "defaults"
    assert report["ephemeris"]["period_days"] == pytest.approx(2.5)


def test_run_triceratops_no_tic_raises_without_allow_fallback(tmp_path):
    """When TIC is absent the Monte Carlo cannot run.

    Without allow_fallback=True, run_triceratops_simulation must raise a
    RuntimeError rather than writing a claim with a hardcoded placeholder FPP.
    This prevents the analysis gate from being silently satisfied.
    """
    from exonym.vetting.tricera_parse import run_triceratops_simulation

    stub, _ = _vet_workspace_stub(tmp_path)  # no TIC
    with pytest.raises(RuntimeError, match="TRICERATOPS Monte Carlo did not run"):
        run_triceratops_simulation(stub, allow_fallback=False)


def test_run_triceratops_does_not_monkeypatch_tls_client(tmp_path, monkeypatch):
    # Arrange
    import sys
    import types

    from exonym.vetting.tricera_parse import run_triceratops_simulation

    stub, _ = _vet_workspace_stub(tmp_path, tic="123456789")
    package = types.ModuleType("triceratops")
    package.__path__ = []
    module = types.ModuleType("triceratops.triceratops")

    def query_trilegal(*args, **kwargs):
        return None

    class FakeTarget:
        def __init__(self, **kwargs):
            self.FPP = 0.02
            self.NFPP = 0.0

        def calc_depths(self, depth):
            return None

        def calc_probs(self, **kwargs):
            return None

    module.query_TRILEGAL = query_trilegal
    module.target = FakeTarget
    monkeypatch.setitem(sys.modules, "triceratops", package)
    monkeypatch.setitem(sys.modules, "triceratops.triceratops", module)

    # Act
    report_path = run_triceratops_simulation(stub, allow_fallback=False)

    # Assert
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["source"] == "triceratops-monte-carlo"
    assert module.query_TRILEGAL is query_trilegal


def test_run_triceratops_allow_fallback_writes_null_fpp_without_claim(tmp_path):
    """A non-Monte-Carlo fallback never creates an FPP claim."""
    from exonym.vetting.tricera_parse import run_triceratops_simulation

    stub, outputs = _vet_workspace_stub(tmp_path)  # no TIC
    report_path = run_triceratops_simulation(stub, allow_fallback=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["FPP"] is None, "FPP must be null, not a hardcoded passing value"
    assert report["source"] in ("not-run",), f"unexpected source: {report['source']}"
    assert report["triceratops_error"] is None  # no error — just no TIC

    claim_path = tmp_path / "claims" / "fpp_claim.json"
    assert not claim_path.exists()


def test_run_triceratops_config_parse_error_emits_warning(tmp_path):
    """A malformed transit config file must emit a UserWarning and fall back
    to default ephemeris values rather than silently using stale data.
    """
    import warnings as _w
    from exonym.vetting.tricera_parse import run_triceratops_simulation

    stub, _ = _vet_workspace_stub(tmp_path)
    signal_dir = tmp_path / "config" / "signals"
    signal_dir.mkdir(parents=True)
    (signal_dir / "transit_config.01.json").write_text(
        "NOT VALID JSON {{{", encoding="utf-8"
    )
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        run_triceratops_simulation(stub, signal=".01", allow_fallback=True)
    assert any("transit_config.01.json" in str(w.message) for w in caught), (
        "expected a warning mentioning the config filename"
    )


def test_load_transit_ephemeris_signal_takes_precedence(tmp_path):
    from exonym.inputs import load_transit_ephemeris
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "ephemeris-signal-test")
    signal_dir = workspace.path / "config" / "signals"
    signal_dir.mkdir(parents=True)
    (signal_dir / "transit_config.02.json").write_text(
        json.dumps(
            {
                "transit": {
                    "period": 14.7157672,
                    "t0_btjd": 2124.779194,
                    "duration_days": 0.125,
                    "depth_ppm": 560.7,
                }
            }
        ),
        encoding="utf-8",
    )

    per_signal = load_transit_ephemeris(workspace, signal=".02")
    assert per_signal["period_days"] == pytest.approx(14.7157672)
    assert per_signal["depth_ppm"] == pytest.approx(560.7)
    assert per_signal["source"] == "candidate-config-signal"

    fallback = load_transit_ephemeris(workspace, signal=".99")
    assert fallback["source"] == "synthetic-demo"

    default = load_transit_ephemeris(workspace)
    assert default["source"] == "synthetic-demo"


# ---------------------------------------------------------------------------
# Scientific analysis modules: asteroseismology
# ---------------------------------------------------------------------------


def test_asteroseismology_recovers_injected_comb():
    from exonym.asteroseismology import (
        _synthetic_oscillation_table,
        estimate_oscillation_envelope,
    )

    table = _synthetic_oscillation_table()
    result = estimate_oscillation_envelope(table["time"], table["flux"], 100.0, 1600.0)
    assert result["numax_candidate_uhz"] == pytest.approx(250.0, abs=15.0)
    assert result["dnu_candidate_uhz"] == pytest.approx(40.0, abs=5.0)
    assert result["dnu_correlation"] > 0.5


def test_seismic_scaling_relations():
    from exonym.asteroseismology import seismic_mass_radius

    solar = seismic_mass_radius(3090.0, 135.1, 5772.0)
    assert solar["mass_solar"] == pytest.approx(1.0, abs=0.01)
    assert solar["radius_solar"] == pytest.approx(1.0, abs=0.01)

    subgiant = seismic_mass_radius(250.0, 40.0, 5772.0)
    expected_radius = (250.0 / 3090.0) / (40.0 / 135.1) ** 2
    assert subgiant["radius_solar"] == pytest.approx(expected_radius, rel=0.01)
    assert subgiant["mass_solar"] == pytest.approx(expected_radius**3 * (40.0 / 135.1) ** 2, rel=0.02)


def test_seismic_mass_radius_falls_back_to_priors():
    from exonym.asteroseismology import seismic_mass_radius

    result = seismic_mass_radius(0.0, None, 5772.0, mass_prior_solar=1.2, radius_prior_solar=1.4)
    assert result["mass_solar"] == pytest.approx(1.2)
    assert result["radius_solar"] == pytest.approx(1.4)
    assert "priors" in result["method"]


def test_seismic_sanity_check_rejects_unphysical_scaling():
    from exonym.asteroseismology import seismic_sanity_check

    implausible = {"mass_solar": 25.99, "radius_solar": 6.68}
    verdict = seismic_sanity_check(
        implausible, radius_prior_solar=2.15, prior_is_catalog=True
    )
    assert verdict["plausible"] is False
    assert any("mass" in reason for reason in verdict["reasons"])
    assert any("prior" in reason for reason in verdict["reasons"])

    plausible = {"mass_solar": 1.9, "radius_solar": 2.1}
    assert seismic_sanity_check(plausible, radius_prior_solar=2.15, prior_is_catalog=True)[
        "plausible"
    ]

    synthetic_source = seismic_sanity_check(implausible)
    assert synthetic_source["plausible"] is False
    assert synthetic_source["reasons"] == ["mass outside plausible range"]


def test_asteroseismic_optional_adapters_write_hashed_candidate_local_manifests(tmp_path, monkeypatch):
    from pathlib import Path

    import exonym.asteroseismology as asteroseismology
    from exonym.isolation import IsolationReport
    from exonym.schemas import validate_schemas
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "astero-adapter-test")
    time = np.linspace(0.0, 1.0, 100)
    flux = np.zeros_like(time)

    def write_estimates(arguments):
        assert arguments[0] == "-f"
        assert Path(arguments[1]).is_file()
        (Path.cwd() / "estimates.csv").write_text("numax,dnu\n250,40\n", encoding="utf-8")

    monkeypatch.setattr(
        asteroseismology.importlib,
        "import_module",
        lambda name: SimpleNamespace(main=write_estimates, __version__="synthetic"),
    )
    succeeded = asteroseismology._run_pysyd_adapter(workspace, time, flux, 100.0, 500.0)

    assert succeeded["status"] == "succeeded"
    assert succeeded["crosscheck"]["estimates"] == [{"numax": 250.0, "dnu": 40.0}]
    manifest = json.loads(succeeded["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["engine"] == "pysyd"
    assert manifest["status"] == "succeeded"
    assert len(manifest["inputs"]) == 1
    assert len(manifest["outputs"]) == 1
    for artifact in manifest["inputs"] + manifest["outputs"]:
        artifact_path = workspace.path / artifact["path"]
        assert artifact_path.is_file()
        assert asteroseismology._sha256(artifact_path) == artifact["sha256"]

    def missing_pysyd(_name):
        raise ModuleNotFoundError("synthetic missing pySYD")

    monkeypatch.setattr(asteroseismology.importlib, "import_module", missing_pysyd)
    unavailable = asteroseismology._run_pysyd_adapter(workspace, time, flux, 100.0, 500.0)
    unavailable_manifest = json.loads(unavailable["manifest_path"].read_text(encoding="utf-8"))
    assert unavailable["status"] == "unavailable"
    assert unavailable_manifest["failure"]["code"] == "module-unavailable"
    assert unavailable_manifest["outputs"] == []

    def fail_pysyd(_arguments):
        raise RuntimeError("synthetic adapter failure")

    monkeypatch.setattr(
        asteroseismology.importlib,
        "import_module",
        lambda name: SimpleNamespace(main=fail_pysyd),
    )
    failed = asteroseismology._run_pysyd_adapter(workspace, time, flux, 100.0, 500.0)
    failed_manifest = json.loads(failed["manifest_path"].read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed_manifest["failure"]["code"] == "adapter-execution-failed"

    monkeypatch.setattr(asteroseismology.importlib.util, "find_spec", lambda name: None)
    tess_atl = asteroseismology._record_tess_atl_adapter(workspace)
    tess_manifest = json.loads(tess_atl["manifest_path"].read_text(encoding="utf-8"))
    assert tess_atl["status"] == "unavailable"
    assert tess_manifest["engine"] == "tess-atl"
    assert tess_manifest["failure"]["code"] == "module-unavailable"
    assert tess_manifest["outputs"] == []

    report = IsolationReport()
    validate_schemas(tmp_path, report)
    assert report.violations == []


# ---------------------------------------------------------------------------
# Scientific analysis modules: PRF localization
# ---------------------------------------------------------------------------


def test_prf_localization_recovers_deficit_offset():
    from exonym.localization import build_depth_map, localize_depth_deficit

    shape = (11, 11)
    target_x, target_y = 5.0, 5.0
    deficit_x, deficit_y = target_x + 0.3, target_y - 0.2
    sigma = 0.85
    yy, xx = np.indices(shape, dtype=float)
    out_image = 2000.0 + 20.0 * np.exp(
        -((xx - target_x) ** 2 + (yy - target_y) ** 2) / (2.0 * sigma**2)
    )
    deficit = 14.0 * np.exp(
        -((xx - deficit_x) ** 2 + (yy - deficit_y) ** 2) / (2.0 * sigma**2)
    )
    depth_map, valid = build_depth_map(out_image - deficit, out_image)
    aperture = np.zeros(shape, dtype=bool)
    aperture[2:-2, 2:-2] = True
    offset = localize_depth_deficit(depth_map, aperture, target_x, target_y)
    assert offset["ra_offset_arcsec"] == pytest.approx(6.3, abs=2.5)
    assert offset["dec_offset_arcsec"] == pytest.approx(-4.2, abs=2.5)
    assert offset["offset_arcsec"] > 2.0
    assert offset["n_depth_pixels"] >= 3


def test_prf_localization_marks_a_two_pixel_depth_core_unresolved():
    from exonym.localization import localize_depth_deficit

    depth_map = np.full((3, 3), 0.001, dtype=float)
    depth_map[1, 1] = 0.010
    depth_map[1, 2] = 0.009

    result = localize_depth_deficit(
        depth_map, np.ones_like(depth_map, dtype=bool), target_x=1.0, target_y=1.0
    )

    assert result["n_depth_pixels"] == 2
    assert np.isnan(result["offset_arcsec"])


def test_prf_localization_json_sanitizer_replaces_nonfinite_diagnostics():
    from exonym.localization import _json_safe

    assert _json_safe({"offset_arcsec": float("nan")}) == {"offset_arcsec": None}


def test_prf_localization_requires_a_competing_source_for_target_dominance(tmp_path):
    from exonym.localization import run_prf_localization

    # Arrange
    workspace = type("Workspace", (), {"path": tmp_path})()

    # Act
    with patch("exonym.localization.load_transit_ephemeris", return_value={}), patch(
        "exonym.localization.load_tpf_cubes", return_value=[]
    ):
        output = run_prf_localization(workspace)
    report = json.loads(output.read_text(encoding="utf-8"))

    # Assert
    assert report["summary"]["conclusion"] == "inconclusive_no_competing_sources_modeled"
    assert report["summary"]["sectors_with_competing_sources_modeled"] == 0
    assert report["sector_results"][0]["target_to_max_other_ratio"] is None


def test_prf_nnls_assigns_depth_to_target():
    from exonym.localization import fit_depth_map_prf, gaussian_prf_kernel

    shape = (11, 11)
    yy, xx = np.indices(shape, dtype=float)
    kernel = gaussian_prf_kernel(xx, yy, 5.0, 5.0)
    depth_map = 0.01 * kernel
    pixel_mask = np.ones(shape, dtype=bool)
    amplitudes, residual, n_pixels = fit_depth_map_prf(
        depth_map, pixel_mask, [5.0, 8.0], [5.0, 5.0]
    )
    assert n_pixels > 5
    assert amplitudes[0] > 10.0 * amplitudes[1]
    assert residual is not None


# ---------------------------------------------------------------------------
# Scientific analysis modules: SED
# ---------------------------------------------------------------------------


def test_sed_recovers_synthetic_photometry():
    from exonym.sed import _fit_blackbody, _synthetic_photometry

    stellar = {
        "teff_k": 5772.0,
        "logg_cgs": 4.438,
        "feh": 0.0,
        "mass_solar": 1.0,
        "radius_solar": 1.0,
        "parallax_mas": 10.0,
    }
    observations = _synthetic_photometry(stellar)
    result = _fit_blackbody(observations, stellar, n_walkers=24, burn_in=150, production=250)
    posterior = result["posterior"]
    assert posterior["teff_k"]["median"] == pytest.approx(5772.0, abs=250.0)
    assert posterior["radius_solar"]["median"] == pytest.approx(1.0, abs=0.35)
    assert posterior["logg_cgs"]["median"] == pytest.approx(4.438, abs=0.3)


def test_sed_percentile_summary():
    from exonym.sed import percentile_summary

    samples = np.linspace(0.0, 10.0, 1001)
    summary = percentile_summary(samples)
    assert summary["median"] == pytest.approx(5.0)
    assert summary["p16"] < summary["median"] < summary["p84"]
    assert summary["plus"] == pytest.approx(summary["p84"] - summary["median"])


# ---------------------------------------------------------------------------
# Scientific analysis modules: transit fit
# ---------------------------------------------------------------------------


def test_transit_fit_recovers_synthetic_parameters(tmp_path):
    from exonym.transit_fit import run_mcmc_transit_fit
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "fit-test")
    output = run_mcmc_transit_fit(
        workspace, n_samples=160, n_walkers=16, burn_in=40
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    posterior = payload["posterior"]
    injected_rp = 1200.0**0.5 / 1000.0
    assert posterior["rp_rs"]["median"] == pytest.approx(injected_rp, abs=0.004)
    assert posterior["impact_parameter"]["median"] == pytest.approx(0.3, abs=0.1)
    assert posterior["rho_star_solar"]["median"] == pytest.approx(1.0, rel=0.3)
    assert posterior["q1"]["median"] == pytest.approx(0.35, abs=0.05)
    assert posterior["q2"]["median"] == pytest.approx(0.3, abs=0.05)
    assert payload["source"] == "synthetic-demo"


def test_transit_fit_eccentric_mode_runs(tmp_path):
    from exonym.transit_fit import run_mcmc_transit_fit
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "fit-ecc-test")
    output = run_mcmc_transit_fit(
        workspace, n_samples=120, eccentric=True, n_walkers=20, burn_in=30
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "eccentricity" in payload["posterior"]
    assert "omega_deg" in payload["posterior"]
    assert payload["posterior"]["eccentricity"]["median"] < 0.3


def test_dynesty_fit_writes_evidence_and_reproducible_compatibility_chain(tmp_path):
    from exonym.transit_fit import run_mcmc_transit_fit
    from exonym.workspace import create_candidate

    class FakeDynamicNestedSampler:
        def __init__(self, log_likelihood, prior_transform, ndim, rstate):
            probe = prior_transform(np.full(ndim, 0.5))
            assert np.isfinite(log_likelihood(probe))
            samples = np.tile(probe, (4, 1))
            samples[:, 0] += np.array([-0.001, 0.0, 0.001, 0.002])
            self.results = SimpleNamespace(
                samples=samples,
                logwt=np.array([-5.0, -3.0, -1.0, 0.0]),
                logz=np.array([-5.0, -2.5, -0.5, 0.0]),
                logzerr=np.array([0.5, 0.3, 0.2, 0.1]),
                niter=4,
                ncall=np.array([3, 3, 3, 3]),
                eff=75.0,
            )

        def run_nested(self, **kwargs):
            assert kwargs["print_progress"] is False
            assert kwargs["nlive_init"] > 0

    workspace = create_candidate(tmp_path, "fit-dynesty-test")
    fake_dynesty = SimpleNamespace(DynamicNestedSampler=FakeDynamicNestedSampler, __version__="test")

    with patch.dict(sys.modules, {"dynesty": fake_dynesty}):
        output = run_mcmc_transit_fit(workspace, n_samples=40, sampler="dynesty", seed=5)
        first_chain = np.load(workspace.path / "outputs" / "mcmc_transit_fit_chain.npy")
        run_mcmc_transit_fit(workspace, n_samples=40, sampler="dynesty", seed=5)

    payload = json.loads(output.read_text(encoding="utf-8"))
    chain = np.load(workspace.path / "outputs" / "mcmc_transit_fit_chain.npy")
    assert output.name == "mcmc_transit_fit.json"
    assert payload["sampler"] == "dynesty"
    assert payload["evidence"]["log_z"] == pytest.approx(0.0)
    assert payload["diagnostics"]["resampling"] == "systematic equal-weight resampling"
    assert payload["diagnostics"]["resampling_seed"] == 5
    assert chain.shape == (4, 7)
    assert np.array_equal(chain, first_chain)


def test_dynesty_dependency_failure_writes_no_fit_output(tmp_path):
    from exonym.transit_fit import run_mcmc_transit_fit
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "fit-dynesty-missing")
    with patch.dict(sys.modules, {"dynesty": None}):
        with pytest.raises(RuntimeError, match=r"\[inference\]"):
            run_mcmc_transit_fit(workspace, sampler="dynesty")

    assert not (workspace.path / "outputs" / "mcmc_transit_fit.json").exists()
    assert not (workspace.path / "outputs" / "mcmc_transit_fit_chain.npy").exists()


def test_stellar_density_a_rs_monotonic():
    from exonym.transit_fit import stellar_density_a_rs

    assert stellar_density_a_rs(1.0, 3.5) > stellar_density_a_rs(1.0, 1.0)
    assert stellar_density_a_rs(4.0, 3.5) > stellar_density_a_rs(1.0, 3.5)
    with pytest.raises(ValueError):
        stellar_density_a_rs(0.0, 3.5)


# ---------------------------------------------------------------------------
# Scientific analysis modules: phase curve
# ---------------------------------------------------------------------------


def test_phase_curve_recovers_injected_reflection():
    from exonym.phasecurve import _synthetic_phase_curve_table, fit_phase_curve_components

    table = _synthetic_phase_curve_table()
    ephemeris = {
        "period_days": table.pop("_period_days"),
        "epoch_btjd": table.pop("_epoch_btjd"),
        "duration_days": table.pop("_duration_days"),
    }
    result = fit_phase_curve_components(
        table["time"], table["flux"], table["flux_err"], table["sector"], ephemeris
    )
    reflection = result["components"]["reflection_semiamplitude"]
    assert reflection["value_ppm"] == pytest.approx(150.0, abs=50.0)
    assert result["maximum_absolute_significance_sigma"] >= 2.0


def test_phase_curve_cluster_covariance_shapes():
    from exonym.phasecurve import cluster_sandwich_covariance

    rng = np.random.default_rng(seed=3)
    design = rng.normal(size=(200, 4))
    residual = rng.normal(size=200)
    sigma = np.full(200, 0.001)
    cluster = np.repeat(np.arange(20), 10)
    covariance, n_clusters = cluster_sandwich_covariance(design, residual, sigma, cluster)
    assert covariance.shape == (4, 4)
    assert n_clusters == 20


def test_cluster_sandwich_covariance_rank_deficient_design():
    from exonym.phasecurve import cluster_sandwich_covariance

    rng = np.random.default_rng(seed=11)
    col = rng.normal(size=200)
    design = np.column_stack([col, col, rng.normal(size=200)])
    residual = rng.normal(size=200)
    sigma = np.full(200, 0.001)
    cluster = np.repeat(np.arange(20), 10)
    covariance, n_clusters = cluster_sandwich_covariance(design, residual, sigma, cluster)
    assert covariance.shape == (3, 3)
    assert np.all(np.isfinite(covariance))
    assert n_clusters == 20


def test_cluster_sandwich_covariance_single_cluster_guard():
    from exonym.phasecurve import cluster_sandwich_covariance

    rng = np.random.default_rng(seed=5)
    design = rng.normal(size=(100, 3))
    residual = rng.normal(size=100)
    sigma = np.full(100, 0.001)
    cluster = np.zeros(100, dtype=int)
    covariance, n_clusters = cluster_sandwich_covariance(design, residual, sigma, cluster)
    assert covariance.shape == (3, 3)
    assert np.all(np.isfinite(covariance))
    assert n_clusters == 1


def test_phase_curve_multi_sector_duplicate_sector_no_singular():
    from exonym.lightcurve import phase_hours
    from exonym.phasecurve import fit_phase_curve_components

    rng = np.random.default_rng(seed=7)
    time = np.concatenate(
        [
            np.linspace(2459000.0, 2459030.0, 500),
            np.linspace(2459000.0, 2459030.0, 500),
            np.linspace(2459100.0, 2459130.0, 500),
        ]
    )
    period_days = 4.57
    epoch_btjd = 2459010.0
    phase_days = phase_hours(time, period_days, epoch_btjd) / 24.0
    flux = 1.0 + 50e-6 * (-np.cos(2.0 * np.pi * phase_days / period_days))
    flux += rng.normal(0.0, 0.001, time.size)
    flux_err = np.full(time.size, 0.001)
    sector_values = np.array([30] * 500 + [30] * 500 + [70] * 500)
    ephemeris = {"period_days": period_days, "epoch_btjd": epoch_btjd, "duration_days": 0.09}
    result = fit_phase_curve_components(time, flux, flux_err, sector_values, ephemeris)
    assert "components" in result
    assert result["n_sectors"] == 2
    assert result["components"]["secondary_eclipse_depth"]["block_robust_error_ppm"] > 0


def _write_test_lightcurve(path, sector, n_points=400, seed=1):
    from astropy.io import fits as fitsio
    from astropy.table import Table

    rng = np.random.default_rng(seed)
    table = Table()
    table["TIME"] = np.linspace(2459000.0, 2459030.0, n_points)
    table["FLUX"] = 1.0 + rng.normal(0.0, 0.001, n_points)
    table["FLUX_ERR"] = np.full(n_points, 0.001)
    table["QUALITY"] = np.zeros(n_points, dtype=np.int32)
    extension = fitsio.BinTableHDU(table)
    extension.header["SECTOR"] = sector
    extension.header["TIMEDEL"] = 120.0 / 86400.0
    extension.header["BJDREFI"] = 2457000
    extension.header["BJDREFF"] = 0.0
    primary = fitsio.PrimaryHDU()
    primary.header["MISSION"] = "TESS"
    primary.header["TELESCOP"] = "TESS"
    fitsio.HDUList([primary, extension]).writeto(path, overwrite=True)


def test_light_curve_table_prefers_processed_over_raw(tmp_path):
    from exonym.inputs import load_light_curve_table
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "dedup-test")
    proc = workspace.path / "data" / "processed"
    raw = workspace.path / "data" / "raw"
    proc.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    _write_test_lightcurve(proc / "s0001_lc.fits", sector=1, n_points=400, seed=1)
    _write_test_lightcurve(raw / "s0001_lc.fits", sector=1, n_points=700, seed=2)

    table = load_light_curve_table(workspace)
    assert table is not None
    assert len(table["time"]) == 400
    assert np.all(table["sector"] == 1)


def test_light_curve_table_dedupes_duplicate_sectors(tmp_path):
    from exonym.inputs import load_light_curve_table
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "dedup-sector-test")
    proc = workspace.path / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    _write_test_lightcurve(proc / "s0030_lc.fits", sector=30, n_points=400, seed=1)
    _write_test_lightcurve(proc / "s0030_qlp_lc.fits", sector=30, n_points=300, seed=2)
    _write_test_lightcurve(proc / "s0070_qlp_lc.fits", sector=70, n_points=300, seed=3)

    table = load_light_curve_table(workspace)
    assert table is not None
    assert set(np.unique(table["sector"])) == {30, 70}
    assert len(table["time"]) == 700


def test_load_gaia_neighbors_parses_csv(tmp_path):
    from exonym.localization import _load_gaia_neighbors
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "gaia-csv-test")
    ext = workspace.path / "data" / "external"
    ext.mkdir(parents=True, exist_ok=True)
    (ext / "gaia_neighbors.csv").write_text(
        "source_id,ra,dec,phot_g_mean_mag,separation_arcsec,flux_ratio_vs_target,is_target_match\n"
        "123,25.9281,-2.6281,10.28,0.7,1.0,true\n"
        "456,25.9242,-2.6270,20.28,11.5,0.0001,false\n",
        encoding="utf-8",
    )
    rows = _load_gaia_neighbors(workspace)
    assert len(rows) == 2
    assert rows[0]["is_target"] is True
    assert rows[0]["source_id"] == "123"
    assert rows[1]["separation_arcsec"] == pytest.approx(11.5)
    assert rows[1]["flux_ratio"] == pytest.approx(0.0001)


# ---------------------------------------------------------------------------
# Scientific analysis modules: TTV
# ---------------------------------------------------------------------------


def test_ttv_linear_ephemeris_has_small_residuals():
    from exonym.search import calculate_ttv_super_period
    from exonym.transit_fit import stellar_density_a_rs
    from exonym.ttv import _synthetic_timing_table, transit_timing_analysis

    table = _synthetic_timing_table(ttv_amplitude_minutes=0.0)
    ephemeris = {
        "period_days": table.pop("_period_days"),
        "epoch_btjd": table.pop("_epoch_btjd"),
        "duration_days": table.pop("_duration_days"),
        "depth_ppm": table.pop("_depth_ppm"),
    }
    a_rs = stellar_density_a_rs(1.0, ephemeris["period_days"])
    analysis = transit_timing_analysis(
        table["time"], table["flux"], table["flux_err"], ephemeris, a_rs
    )
    assert analysis["n_transits_fit"] >= 5
    assert analysis["oc_rms_minutes"] < 2.0

    assert calculate_ttv_super_period(3.5, 5.0, j_resonance=2) == pytest.approx(8.75, rel=0.01)


def test_ttv_injected_signal_is_detected():
    from exonym.transit_fit import stellar_density_a_rs
    from exonym.ttv import _synthetic_timing_table, transit_timing_analysis

    table = _synthetic_timing_table(ttv_amplitude_minutes=20.0)
    ephemeris = {
        "period_days": table.pop("_period_days"),
        "epoch_btjd": table.pop("_epoch_btjd"),
        "duration_days": table.pop("_duration_days"),
        "depth_ppm": table.pop("_depth_ppm"),
    }
    a_rs = stellar_density_a_rs(1.0, ephemeris["period_days"])
    analysis = transit_timing_analysis(
        table["time"], table["flux"], table["flux_err"], ephemeris, a_rs
    )
    assert analysis["oc_rms_minutes"] > 5.0


# ---------------------------------------------------------------------------
# Scientific analysis modules: stellar activity
# ---------------------------------------------------------------------------


def test_activity_recovers_rotation_period():
    from exonym.activity import (
        _synthetic_rotation_table,
        gls_periodogram,
        sinusoid_amplitude_ppm,
        weighted_period_summary,
    )

    table = _synthetic_rotation_table()
    periods, powers, fap = gls_periodogram(table["time"], table["flux"])
    best_period = float(periods[int(np.argmax(powers))])
    assert best_period == pytest.approx(5.0, abs=0.2)
    assert fap < 0.01

    amplitude = sinusoid_amplitude_ppm(table["time"], table["flux"], best_period)
    assert amplitude == pytest.approx(400.0, abs=100.0)

    summary = weighted_period_summary([5.0, 5.1], [1.0, 1.0])
    assert summary["weighted_mean_period_days"] == pytest.approx(5.05)


# ---------------------------------------------------------------------------
# Scientific analysis modules: dilution
# ---------------------------------------------------------------------------


def test_dilution_contamination_factor_sums_neighbors():
    from exonym.dilution import gaia_contamination_factor

    rows = [
        {"separation_arcsec": 10.0, "flux_ratio": 0.02, "is_target": False},
        {"separation_arcsec": 100.0, "flux_ratio": 0.5, "is_target": False},
        {"separation_arcsec": 5.0, "flux_ratio": None, "is_target": False, "g_mag": 14.0},
    ]
    result = gaia_contamination_factor(rows, search_radius_arcsec=60.0, target_g_mag=10.0)
    expected = 0.02 + 10.0 ** (-0.4 * (14.0 - 10.0))
    assert result["contamination_factor"] == pytest.approx(expected, abs=1e-6)
    assert result["n_neighbors_included"] == 2


def test_dilution_ignores_nonfinite_neighbor_measurements():
    from exonym.dilution import gaia_contamination_factor

    # Arrange
    rows = [
        {"separation_arcsec": 10.0, "flux_ratio": float("nan"), "is_target": False},
        {"separation_arcsec": float("inf"), "flux_ratio": 0.5, "is_target": False},
    ]

    # Act
    result = gaia_contamination_factor(rows)

    # Assert
    assert result["contamination_factor"] == 0.0
    assert result["n_neighbors_included"] == 0


def test_dilution_reads_validated_archival_neighbors(tmp_path):
    from exonym.dilution import _load_archival_gaia_neighbor_rows

    # Arrange
    workspace = type("Workspace", (), {"path": tmp_path})()
    output = tmp_path / "outputs"
    output.mkdir()
    output.joinpath("archival_vetting_report.json").write_text(
        json.dumps(
            {
                "gaia_astrometry": {
                    "validated": True,
                    "search_radius_arcsec": 30.0,
                    "target_source_id": "synthetic-target",
                    "sources": [
                        {
                            "source_id": "synthetic-target",
                            "separation_arcsec": 0.1,
                            "phot_g_mean_mag": 10.0,
                        },
                        {
                            "source_id": "synthetic-neighbor",
                            "separation_arcsec": 8.0,
                            "phot_g_mean_mag": 14.0,
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    # Act
    rows, target_g_mag, metadata = _load_archival_gaia_neighbor_rows(workspace)

    # Assert
    assert target_g_mag == pytest.approx(10.0)
    assert rows == [
        {
            "g_mag": 14.0,
            "separation_arcsec": 8.0,
            "flux_ratio": None,
            "is_target": False,
        }
    ]
    assert metadata["availability"] == "available"
    assert metadata["target_selection"] == "reported-target-source-id"


def test_dilution_rejects_unvalidated_archival_neighbors(tmp_path):
    from exonym.dilution import _load_archival_gaia_neighbor_rows

    # Arrange
    workspace = type("Workspace", (), {"path": tmp_path})()
    output = tmp_path / "outputs"
    output.mkdir()
    output.joinpath("archival_vetting_report.json").write_text(
        json.dumps({"gaia_astrometry": {"validated": False, "sources": []}}),
        encoding="utf-8",
    )

    # Act
    rows, target_g_mag, metadata = _load_archival_gaia_neighbor_rows(workspace)

    # Assert
    assert rows == []
    assert target_g_mag is None
    assert metadata["availability"] == "unavailable"


def test_dilution_aperture_depth_decreases_with_size():
    from exonym.dilution import (
        _extract_cube_light_curves,
        _synthetic_tpf_cube,
        aperture_depth_ppm,
    )

    cube = _synthetic_tpf_cube()
    ephemeris = {
        "period_days": cube.pop("_period_days"),
        "epoch_btjd": cube.pop("_epoch_btjd"),
        "duration_days": cube.pop("_duration_days"),
    }
    extracted = _extract_cube_light_curves(cube, ephemeris)
    small = aperture_depth_ppm(extracted["time"], extracted["light_curves"]["box_3x3"], ephemeris)
    large = aperture_depth_ppm(extracted["time"], extracted["light_curves"]["box_7x7"], ephemeris)
    assert small["depth_ppm"] > large["depth_ppm"]
    assert small["n_in_transit"] > 100
