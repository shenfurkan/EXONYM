import json
import hashlib
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from exonym.vetting.centroid import centroid_gate, centroid_offset_z
from exonym.vetting.oddeven import odd_even_gate, odd_even_z
from exonym.vetting.tricera_parse import fpp_gate, load_fpp_report


@pytest.fixture(autouse=True)
def test_centroid_offset_z_uses_cos_dec():
    """Offsets are on-sky projected arcseconds; cos(dec) must NOT rescale again."""
    z = centroid_offset_z(ra_offset_arcsec=0.0, dec_offset_arcsec=3.0, dec_deg=0.0, sigma_arcsec=1.0)
    assert z == pytest.approx(3.0)
    z_on_target = centroid_offset_z(0.5, 0.5, 0.0, 1.0)
    assert z_on_target < 3.0
    # Regression guard (Finding: double cos(dec)): a high-declination target
    # must produce the same significance as an equatorial one for identical
    # projected offsets.
    z_high_dec = centroid_offset_z(ra_offset_arcsec=0.0, dec_offset_arcsec=3.0, dec_deg=69.5, sigma_arcsec=1.0)
    assert z_high_dec == pytest.approx(3.0)


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
    assert fpp_gate({"FPP": 0.005, "NFPP": 0.02})[0] is False
    assert fpp_gate({"FPP": 0.005})[0] is False
    assert fpp_gate(0.02)[0] is False


def test_fpp_report_probes_common_keys(tmp_path):
    path = tmp_path / "triceratops.json"
    path.write_text(json.dumps({"FPP_specific": 0.008, "NFPP": 0.0}), encoding="utf-8")
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


@pytest.mark.parametrize(
    "contents",
    (
        '{"FPP": 0.1, "FPP": 0.2, "NFPP": 0.0}',
        '{"FPP": NaN, "NFPP": 0.0}',
        '{"FPP": 1e999, "NFPP": 0.0}',
    ),
)
def test_fpp_report_rejects_ambiguous_or_nonfinite_json(tmp_path, contents):
    path = tmp_path / "triceratops.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="strict finite JSON"):
        load_fpp_report(path)


@pytest.mark.parametrize("report", ({"FPP": True, "NFPP": 0.0}, {"FPP": 1.1, "NFPP": 0.0}))
def test_fpp_gate_rejects_non_numeric_or_out_of_range_probabilities(report):
    with pytest.raises(ValueError, match="FPP must be"):
        fpp_gate(report)


def test_observed_sector_parser_prefers_products_then_uses_holdings_fallback(tmp_path):
    from exonym.vetting.tricera_parse import _observed_sectors

    workspace = type("Workspace", (), {"path": tmp_path, "candidate_id": "dilution-test"})()
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
    outputs.mkdir(parents=True, exist_ok=True)
    stub = types.SimpleNamespace(
        path=tmp_path,
        candidate_id=candidate_id,
        repository_root=tmp_path,
        metadata={"identifiers": {"tic": tic}} if tic else {"identifiers": {}},
    )
    return stub, outputs


def _write_raw_provenance(input_path):
    sidecar = input_path.with_name(input_path.stem + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "source_uri": "https://archive.example.invalid/" + input_path.name,
                "download_timestamp_utc": "2026-01-01T00:00:00Z",
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "fetched_by": "synthetic-test",
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def _observed_input_stub(tmp_path):
    input_path = tmp_path / "data" / "raw" / "s0001_lc.fits"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"observed-photometry")
    return {
        "time_days": np.array([-0.10, -0.02, 0.0, 0.02, 0.10]),
        "flux": np.array([1.0, 0.999, 0.998, 0.999, 1.0]),
        "flux_err": 0.0002,
        "period_days": 2.0,
        "duration_hours": 2.4,
        "depth_ppm": 2000.0,
        "exposure_days": 120.0 / 86400.0,
        "sectors": [1],
        "provenance": {
            "representation": "phase-folded observed candidate photometry",
            "input_files": [
                {
                    "path": "data/raw/s0001_lc.fits",
                    "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                }
            ],
            "ephemeris_source": "candidate-config",
        },
    }


def _write_trex_scene_manifest(workspace, *, include_neighbor=True):
    external = workspace.path / "data" / "external"
    external.mkdir(parents=True, exist_ok=True)
    contrast_path = external / "synthetic_contrast_curve.csv"
    contrast_path.write_text("separation_arcsec,delta_mag\n0.1,2.0\n1.0,5.0\n", encoding="utf-8")
    background_path = external / "synthetic_trilegal.csv"
    background_path.write_text("synthetic background population\n", encoding="utf-8")
    target = {
        "source_id": "synthetic-target",
        "separation_arcsec": 0.0,
        "ra_deg": 10.0,
        "dec_deg": -20.0,
        "phot_g_mean_mag": 10.0,
    }
    sources = [target]
    neighbors = []
    if include_neighbor:
        sources.append(
            {
                "source_id": "synthetic-neighbor",
                "separation_arcsec": 1.0,
                "ra_deg": 10.01,
                "dec_deg": -20.01,
                "phot_g_mean_mag": 13.0,
            }
        )
        neighbors.append(
            {
                "source_id": "synthetic-neighbor",
                "mass_solar": 0.7,
                "radius_solar": 0.7,
                "delta_mag": 3.0,
                "separation_arcsec": 1.0,
            }
        )
    archival_path = workspace.path / "outputs" / "archival_vetting_report.json"
    archival_path.write_text(
        json.dumps(
            {
                "candidate_id": workspace.candidate_id,
                "target_coordinates": {"ra_deg": 10.0, "dec_deg": -20.0},
                "gaia_astrometry": {
                    "validated": True,
                    "query_status": "ok",
                    "target_source_id": "synthetic-target",
                    "nearby_sources_count": len(sources),
                    "sources": sources,
                },
            }
        ),
        encoding="utf-8",
    )
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "candidate_id": workspace.candidate_id,
        "source": "candidate-data",
        "target": {
            "ra_deg": 10.0,
            "dec_deg": -20.0,
            "mass_solar": 1.0,
            "radius_solar": 1.0,
            "teff_k": 5700.0,
            "parallax_mas": 2.0,
            "tess_mag": 10.0,
        },
        "archival_gaia": {
            "path": "outputs/archival_vetting_report.json",
            "sha256": digest(archival_path),
            "target_source_id": "synthetic-target",
            "neighbor_source_ids": [neighbor["source_id"] for neighbor in neighbors],
        },
        "contrast_curve": {
            "path": "data/external/synthetic_contrast_curve.csv",
            "sha256": digest(contrast_path),
            "separations_arcsec": [0.1, 1.0],
            "delta_magnitudes": [2.0, 5.0],
        },
        "background": {
            "path": "data/external/synthetic_trilegal.csv",
            "sha256": digest(background_path),
            "model": "trilegal",
            "star_count": 2,
        },
        "resolved_neighbors": neighbors,
    }
    manifest_path = external / "trex_scene.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_prepare_observed_transit_input_uses_measured_candidate_photometry(tmp_path, monkeypatch):
    from exonym.vetting.tricera_parse import _prepare_observed_transit_input

    workspace, _ = _vet_workspace_stub(tmp_path, tic="123456789")
    input_path = tmp_path / "data" / "raw" / "s0001_lc.fits"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"observed-photometry")
    time = np.arange(995.0, 1005.0, 0.01)
    phase = np.remainder(time - 1000.0 + 1.0, 2.0) - 1.0
    flux = np.ones_like(time)
    flux[np.abs(phase) <= 0.0625] -= 0.002
    table = {
        "time": time,
        "flux": flux,
        "flux_err": np.full_like(time, 0.0002),
        "flux_err_sources": ["reported"],
        "sector": np.ones(time.size, dtype=int),
        "input_files": [input_path],
    }
    ephemeris = {
        "period_days": 2.0,
        "epoch_btjd": 1000.0,
        "duration_days": 0.125,
        "time_system": "BTJD_TDB",
        "source": "candidate-config",
        "field_sources": {
            "period_days": "candidate-config",
            "epoch_btjd": "candidate-config",
            "duration_days": "candidate-config",
        },
    }
    monkeypatch.setattr("exonym.inputs.load_light_curve_table", lambda *_args, **_kwargs: table)
    monkeypatch.setattr("exonym.inputs.load_transit_ephemeris", lambda *_args, **_kwargs: ephemeris)

    prepared = _prepare_observed_transit_input(workspace, signal=None)

    assert 10 <= prepared["time_days"].size <= 100
    assert np.max(np.abs(prepared["time_days"])) <= 1.0
    assert prepared["depth_ppm"] == pytest.approx(2000.0, rel=0.05)
    assert prepared["exposure_days"] == pytest.approx(0.01)
    assert prepared["flux_err"] > 0.0
    provenance = prepared["provenance"]
    assert provenance["raw_cadence_count"] == time.size
    assert provenance["input_files"] == [
        {
            "path": "data/raw/s0001_lc.fits",
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        }
    ]
    assert provenance["ephemeris_field_sources"] == {
        "period_days": "candidate-config",
        "epoch_btjd": "candidate-config",
        "duration_days": "candidate-config",
    }


def test_prepare_observed_transit_input_resolves_low_duty_cycle_transit(tmp_path, monkeypatch):
    from exonym.vetting.tricera_parse import _prepare_observed_transit_input

    workspace, _ = _vet_workspace_stub(tmp_path, tic="123456789")
    input_path = tmp_path / "data" / "raw" / "s0001_lc.fits"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"observed-photometry")
    period_days = 12.75
    duration_days = 3.0 / 24.0
    epoch_btjd = 1000.0
    time = np.arange(950.0, 1050.0, 0.005)
    phase = np.remainder(time - epoch_btjd + 0.5 * period_days, period_days) - 0.5 * period_days
    flux = np.ones_like(time)
    flux[np.abs(phase) <= 0.5 * duration_days] -= 0.0015
    table = {
        "time": time,
        "flux": flux,
        "flux_err": np.full_like(time, 0.0002),
        "flux_err_sources": ["reported"],
        "sector": np.ones(time.size, dtype=int),
        "input_files": [input_path],
    }
    ephemeris = {
        "period_days": period_days,
        "epoch_btjd": epoch_btjd,
        "duration_days": duration_days,
        "time_system": "BTJD_TDB",
        "source": "candidate-config",
        "field_sources": {
            "period_days": "candidate-config",
            "epoch_btjd": "candidate-config",
            "duration_days": "candidate-config",
        },
    }
    monkeypatch.setattr("exonym.inputs.load_light_curve_table", lambda *_args, **_kwargs: table)
    monkeypatch.setattr("exonym.inputs.load_transit_ephemeris", lambda *_args, **_kwargs: ephemeris)

    prepared = _prepare_observed_transit_input(workspace, signal=None)

    in_transit = np.abs(prepared["time_days"]) <= 0.5 * duration_days
    assert np.count_nonzero(in_transit) == 5
    assert prepared["depth_ppm"] == pytest.approx(1500.0, rel=0.05)
    phase_binning = prepared["provenance"]["phase_binning"]
    assert phase_binning["method"] == "transit-centered-nonuniform"
    assert phase_binning["transit_bin_count"] == 5
    assert phase_binning["transit_bin_width_days"] == pytest.approx(duration_days / 5.0)


def test_prepare_observed_transit_input_rejects_estimated_errors(tmp_path, monkeypatch):
    from exonym.vetting.tricera_parse import _prepare_observed_transit_input

    workspace, _ = _vet_workspace_stub(tmp_path, tic="123456789")
    table = {
        "flux_err_sources": ["mad-estimate"],
    }
    ephemeris = {
        "period_days": 2.0,
        "epoch_btjd": 1000.0,
        "duration_days": 0.125,
        "time_system": "BTJD_TDB",
        "field_sources": {
            "period_days": "candidate-config",
            "epoch_btjd": "candidate-config",
            "duration_days": "candidate-config",
        },
    }
    monkeypatch.setattr("exonym.inputs.load_light_curve_table", lambda *_args, **_kwargs: table)
    monkeypatch.setattr("exonym.inputs.load_transit_ephemeris", lambda *_args, **_kwargs: ephemeris)

    with pytest.raises(ValueError, match="reported per-cadence flux uncertainties"):
        _prepare_observed_transit_input(workspace, signal=None)


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

    # No TIC in stub â†’ Monte Carlo cannot run; allow_fallback=True to test
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
    result_path = tmp_path / "outputs" / "bls_search_results.json"
    result_path.write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "detection_status": "detected",
                "time_system": "BTJD_TDB",
                "best_period": 7.5,
                "best_epoch": 1.0,
                "best_depth_ppm": 900.0,
                "best_duration_hours": 2.0,
                "snr": 10.0,
                "n_distinct_transit_events": 3,
                "detection_threshold_snr": 7.1,
            }
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "data" / "raw" / "observed.fits"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"candidate-photometry")
    provenance_path = _write_raw_provenance(input_path)
    (tmp_path / "outputs" / "bls_search_manifest.json").write_text(
        json.dumps(
            {
                "schema": "exonym-bls-search-manifest-1",
                "candidate_id": stub.candidate_id,
                "result_path": "outputs/bls_search_results.json",
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "source": "candidate-data",
                "detection_status": "detected",
                "inputs": [
                    {
                        "path": "data/raw/observed.fits",
                        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                        "provenance_path": "data/raw/observed.provenance.json",
                        "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
                    }
                ],
                "configuration": {
                    "engine": "bls",
                    "signal": None,
                    "time_system": "BTJD_TDB",
                    "detection_threshold_snr": 7.1,
                },
            }
        ),
        encoding="utf-8",
    )

    # No TIC in stub â†’ Monte Carlo cannot run; allow_fallback=True to test
    # ephemeris routing (BLS results used when no signal is given).
    report_path = run_triceratops_simulation(stub, signal=None, allow_fallback=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["signal"] is None
    assert report["ephemeris"]["period_days"] == pytest.approx(7.5)
    assert report["ephemeris"]["source"] == "bls-search"


def test_run_triceratops_falls_back_when_signal_config_missing(tmp_path):
    from exonym.vetting.tricera_parse import run_triceratops_simulation

    stub, _ = _vet_workspace_stub(tmp_path)
    # No TIC â†’ Monte Carlo cannot run; allow_fallback=True required.
    with pytest.warns(UserWarning, match="could not read signal transit config"):
        report_path = run_triceratops_simulation(stub, signal=".99", allow_fallback=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ephemeris"] == {
        "period_days": None,
        "epoch_btjd": None,
        "depth_ppm": None,
        "duration_hours": None,
        "source": "unavailable",
    }
    assert report["FPP"] is None
    assert report["claim_eligible"] is False


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
def test_run_triceratops_records_incompatible_runtime_without_execution(tmp_path, monkeypatch):
    from exonym.vetting.tricera_parse import run_triceratops_simulation

    workspace, _ = _vet_workspace_stub(tmp_path, tic="123456789")
    monkeypatch.setattr(
        "exonym.statistical_vetting.require_vetting_readiness",
        lambda *_args, **_kwargs: tmp_path / "outputs" / "statistical_vetting_evidence.json",
    )
    monkeypatch.setattr(
        "exonym.vetting.tricera_parse._prepare_observed_transit_input",
        lambda *_args, **_kwargs: _observed_input_stub(tmp_path),
    )
    monkeypatch.setattr(
        "exonym.engines.check_engine",
        lambda _name: (False, "requires pytransit==2.2, but pytransit 2.6.11 is installed"),
    )

    report_path = run_triceratops_simulation(workspace, allow_fallback=True)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    decision = json.loads(
        (tmp_path / "decisions" / "triceratops_vetting_decision.json").read_text(encoding="utf-8")
    )
    assert report["source"] == "triceratops-failed-UNVALIDATED"
    assert report["FPP"] is None
    assert decision["execution_status"] == "unavailable"
    assert decision["result_status"] == "unresolved"
    assert decision["error"]["code"] == "triceratops-runtime-incompatible"


def test_run_triceratops_passes_measured_exposure_to_trex(tmp_path, monkeypatch):
    from exonym.vetting.tricera_parse import run_triceratops_simulation

    workspace, _ = _vet_workspace_stub(tmp_path, tic="123456789")
    observed = _observed_input_stub(tmp_path)
    _write_trex_scene_manifest(workspace)
    observed["exposure_days"] = 600.0 / 86400.0
    captured = {}

    class Result:
        fpp = 0.1
        nfpp = 0.01

        @staticmethod
        def top_scenarios(_count):
            return []

    def fake_run_trex_vetting(*args, **kwargs):
        captured.update(kwargs)
        captured["scene"] = args[0]
        return Result()

    monkeypatch.setattr(
        "exonym.statistical_vetting.require_vetting_readiness",
        lambda *_args, **_kwargs: tmp_path / "outputs" / "statistical_vetting_evidence.json",
    )
    monkeypatch.setattr(
        "exonym.vetting.tricera_parse._prepare_observed_transit_input",
        lambda *_args, **_kwargs: observed,
    )
    monkeypatch.setattr("exonym.engines.check_engine", lambda _name: (True, "available"))
    monkeypatch.setattr("exonym.vetting.trex.run_trex_vetting", fake_run_trex_vetting)

    run_triceratops_simulation(workspace)

    assert captured["exptime_days"] == pytest.approx(observed["exposure_days"])
    assert captured["scene"].Teff_K == pytest.approx(5700.0)


def test_run_triceratops_marks_missing_scene_evidence_unavailable(tmp_path, monkeypatch):
    from exonym.vetting.tricera_parse import run_triceratops_simulation

    workspace, _ = _vet_workspace_stub(tmp_path, tic="123456789")
    attempted = False

    def fake_run_trex_vetting(*_args, **_kwargs):
        nonlocal attempted
        attempted = True

    monkeypatch.setattr(
        "exonym.statistical_vetting.require_vetting_readiness",
        lambda *_args, **_kwargs: tmp_path / "outputs" / "statistical_vetting_evidence.json",
    )
    monkeypatch.setattr(
        "exonym.vetting.tricera_parse._prepare_observed_transit_input",
        lambda *_args, **_kwargs: _observed_input_stub(tmp_path),
    )
    monkeypatch.setattr("exonym.engines.check_engine", lambda _name: (True, "available"))
    monkeypatch.setattr("exonym.vetting.trex.run_trex_vetting", fake_run_trex_vetting)

    report_path = run_triceratops_simulation(workspace, allow_fallback=True)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    decision = json.loads(
        (tmp_path / "decisions" / "triceratops_vetting_decision.json").read_text(encoding="utf-8")
    )
    assert attempted is False
    assert report["FPP"] is None
    assert report["claim_eligible"] is False
    assert decision["execution_status"] == "unavailable"
    assert decision["result_status"] == "unresolved"
    assert decision["error"]["code"] == "trex-scene-unavailable"


def test_trex_scene_requires_all_archival_gaia_neighbors(tmp_path):
    from exonym.vetting.tricera_parse import TrexSceneUnavailableError, _load_trex_scene

    workspace, _ = _vet_workspace_stub(tmp_path, tic="123456789")
    manifest_path = _write_trex_scene_manifest(workspace)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["archival_gaia"]["neighbor_source_ids"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TrexSceneUnavailableError, match="every archival Gaia neighbor"):
        _load_trex_scene(workspace, 123456789, [1])


def test_trex_scene_requires_all_finite_target_parameters(tmp_path):
    from exonym.vetting.tricera_parse import TrexSceneUnavailableError, _load_trex_scene

    workspace, _ = _vet_workspace_stub(tmp_path, tic="123456789")
    manifest_path = _write_trex_scene_manifest(workspace)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["target"]["teff_k"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TrexSceneUnavailableError, match="target.teff_k"):
        _load_trex_scene(workspace, 123456789, [1])


def test_run_triceratops_requires_readiness_inside_public_function(tmp_path, monkeypatch):
    from exonym.vetting.tricera_parse import run_triceratops_simulation

    stub, _ = _vet_workspace_stub(tmp_path, tic="123456789")

    def reject_readiness(*_args, **_kwargs):
        raise RuntimeError("pre-vetting evidence is incomplete")

    monkeypatch.setattr("exonym.statistical_vetting.require_vetting_readiness", reject_readiness)

    with pytest.raises(RuntimeError, match="pre-vetting evidence is incomplete"):
        run_triceratops_simulation(stub, allow_fallback=True)

    assert not (tmp_path / "outputs" / "triceratops_report.json").exists()
def test_run_triceratops_allow_fallback_writes_null_fpp_without_claim(tmp_path):
    """A non-Monte-Carlo fallback never creates an FPP claim."""
    from exonym.vetting.tricera_parse import run_triceratops_simulation

    stub, outputs = _vet_workspace_stub(tmp_path)  # no TIC
    report_path = run_triceratops_simulation(stub, allow_fallback=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["FPP"] is None, "FPP must be null, not a hardcoded passing value"
    assert report["source"] in ("not-run",), f"unexpected source: {report['source']}"
    assert report["triceratops_error"] is None  # no error â€” just no TIC
    assert report["audit_status"] == "invalid"
    assert report["audit_invalid_reason"]

    claim_path = tmp_path / "claims" / "fpp_claim.json"
    assert not claim_path.exists()


@pytest.mark.parametrize("random_seed", (True, -1, 2**32))
def test_run_triceratops_rejects_invalid_random_seed(tmp_path, random_seed):
    from exonym.vetting.tricera_parse import run_triceratops_simulation

    stub, _ = _vet_workspace_stub(tmp_path)

    with pytest.raises(ValueError, match="random_seed"):
        run_triceratops_simulation(stub, allow_fallback=True, random_seed=random_seed)


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


def test_triceratops_signal_decisions_are_isolated_and_atomic(tmp_path, monkeypatch):
    from exonym.statistical_vetting import (
        _write_json_atomic,
        triceratops_vetting_decision_path,
        write_triceratops_vetting_decision,
    )

    workspace, _ = _vet_workspace_stub(tmp_path)
    first = write_triceratops_vetting_decision(
        workspace,
        signal=".01",
        execution_status="unavailable",
        triage_status="not-run",
        error={"code": "test", "message": "Synthetic unavailable engine."},
    )
    second = write_triceratops_vetting_decision(
        workspace,
        signal=".02",
        execution_status="failed",
        triage_status="review-required",
        error={"code": "test", "message": "Synthetic failed engine."},
    )

    assert first == triceratops_vetting_decision_path(workspace, ".01")
    assert second == triceratops_vetting_decision_path(workspace, ".02")
    assert json.loads(first.read_text(encoding="utf-8"))["signal"] == ".01"
    assert json.loads(second.read_text(encoding="utf-8"))["signal"] == ".02"

    original = tmp_path / "decisions" / "atomic.json"
    original.write_text('{"old": true}\n', encoding="utf-8")
    monkeypatch.setattr(
        "exonym.statistical_vetting.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("synthetic replace failure")),
    )
    with pytest.raises(OSError, match="synthetic replace failure"):
        _write_json_atomic(original, {"new": True})
    assert json.loads(original.read_text(encoding="utf-8")) == {"old": True}
    assert not list(original.parent.glob("atomic.json.*.tmp"))
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
    assert per_signal["field_sources"] == {
        "period_days": "candidate-config-signal",
        "epoch_btjd": "candidate-config-signal",
        "duration_days": "candidate-config-signal",
        "depth_ppm": "candidate-config-signal",
    }

    fallback = load_transit_ephemeris(workspace, signal=".99")
    assert fallback["source"] == "unavailable"
    assert fallback["period_days"] is None
    assert fallback["epoch_btjd"] is None
    assert fallback["duration_days"] is None
    assert fallback["depth_ppm"] is None

    default = load_transit_ephemeris(workspace)
    assert default["source"] == "unavailable"
    assert default["period_days"] is None
    assert default["epoch_btjd"] is None
    assert default["duration_days"] is None
    assert default["depth_ppm"] is None


@pytest.mark.parametrize("signal", [".1", ".001", "01", "../escape", ".0/"])
def test_signal_suffix_requires_exactly_two_digits(tmp_path, signal):
    from exonym.inputs import load_transit_ephemeris
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "signal-validation")

    with pytest.raises(ValueError, match="\\.NN"):
        load_transit_ephemeris(workspace, signal=signal)


def test_load_transit_ephemeris_keeps_partial_config_provenance_explicit(tmp_path):
    from exonym.inputs import load_transit_ephemeris
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "partial-ephemeris-test")
    (workspace.path / "config" / "transit_config.json").write_text(
        json.dumps({"transit": {"period_days": 6.5}}), encoding="utf-8"
    )

    ephemeris = load_transit_ephemeris(workspace)

    assert ephemeris["source"] == "partial-candidate-config"
    assert ephemeris["field_sources"] == {
        "period_days": "candidate-config",
        "epoch_btjd": None,
        "duration_days": None,
        "depth_ppm": None,
    }


def test_load_transit_ephemeris_uses_matching_signal_bls_result(tmp_path):
    from exonym.inputs import load_transit_ephemeris
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "signal-bls-ephemeris-test")
    result_path = workspace.path / "outputs" / "bls_search_results.02.json"
    result_path.write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "detection_status": "detected",
                "time_system": "BTJD_TDB",
                "best_period": 4.25,
                "best_epoch": 1742.5,
                "best_duration_hours": 3.0,
                "best_depth_ppm": 700.0,
                "snr": 10.0,
                "n_distinct_transit_events": 3,
                "detection_threshold_snr": 7.1,
            }
        ),
        encoding="utf-8",
    )
    input_path = workspace.path / "data" / "raw" / "observed.fits"
    input_path.write_bytes(b"candidate-photometry")
    provenance_path = _write_raw_provenance(input_path)
    (workspace.path / "outputs" / "bls_search_manifest.02.json").write_text(
        json.dumps(
            {
                "schema": "exonym-bls-search-manifest-1",
                "candidate_id": workspace.candidate_id,
                "result_path": "outputs/bls_search_results.02.json",
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "source": "candidate-data",
                "detection_status": "detected",
                "inputs": [
                    {
                        "path": "data/raw/observed.fits",
                        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                        "provenance_path": "data/raw/observed.provenance.json",
                        "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
                    }
                ],
                "configuration": {
                    "engine": "bls",
                    "signal": ".02",
                    "time_system": "BTJD_TDB",
                    "detection_threshold_snr": 7.1,
                },
            }
        ),
        encoding="utf-8",
    )

    ephemeris = load_transit_ephemeris(workspace, signal=".02")

    assert ephemeris["period_days"] == pytest.approx(4.25)
    assert ephemeris["epoch_btjd"] == pytest.approx(1742.5)
    assert ephemeris["duration_days"] == pytest.approx(0.125)
    assert ephemeris["field_sources"] == {
        "period_days": "bls-search",
        "epoch_btjd": "bls-search",
        "duration_days": "bls-search",
        "depth_ppm": "bls-search",
    }

    signal_config = workspace.path / "config" / "signals" / "transit_config.02.json"
    signal_config.parent.mkdir(parents=True, exist_ok=True)
    signal_config.write_text(
        json.dumps(
            {
                "source": "candidate-data-bls",
                "bls_provenance": {
                    "result": {
                        "path": "outputs/bls_search_results.02.json",
                        "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                    },
                    "manifest": {
                        "path": "outputs/bls_search_manifest.02.json",
                        "sha256": hashlib.sha256(
                            (workspace.path / "outputs" / "bls_search_manifest.02.json").read_bytes()
                        ).hexdigest(),
                    },
                },
                "transit": {
                    "period_days": 4.25,
                    "epoch_btjd": 1742.5,
                    "epoch_time_system": "BTJD_TDB",
                    "duration_days": 3.0 / 24.0,
                    "depth_ppm": 900.0,
                },
            }
        ),
        encoding="utf-8",
    )

    rejected_config = load_transit_ephemeris(workspace, signal=".02")

    assert rejected_config["source"] == "bls-search"
    assert rejected_config["depth_ppm"] == pytest.approx(700.0)


# ---------------------------------------------------------------------------
# Scientific analysis modules: asteroseismology
# ---------------------------------------------------------------------------


def test_asteroseismology_recovers_injected_comb():
    from exonym.asteroseismology import estimate_oscillation_envelope
    from tests.fixtures.synthetic_observations import _synthetic_oscillation_table

    table = _synthetic_oscillation_table()
    result = estimate_oscillation_envelope(table["time"], table["flux"], 100.0, 1600.0)
    assert result["numax_candidate_uhz"] == pytest.approx(250.0, abs=15.0)
    # With DNU_MAX_UHZ widened to 200 uHz the correlation grid may land on a
    # harmonic alias (40, 80, or 120 uHz) instead of the injected 40 uHz.
    dnu = result["dnu_candidate_uhz"]
    assert any(abs(dnu - expected) < 6.0 for expected in (40.0, 80.0, 120.0)), (
        "dnu_candidate %.1f not near injected 40 uHz or its harmonics" % dnu
    )
    assert result["dnu_correlation"] > 0.5


def test_solar_analog_dnu_recovery():
    """A solar-like comb (Î”Î½ â‰ˆ 135.1 ÂµHz) must be recoverable after raising DNU_MAX_UHZ."""
    import math

    import numpy as np

    from exonym.asteroseismology import (
        CPD_PER_UHZ,
        estimate_oscillation_envelope,
    )

    rng = np.random.default_rng(seed=42)
    numax_demo_uhz = 1000.0
    dnu_demo_uhz = 135.1
    envelope_sigma_uhz = 2.5 * dnu_demo_uhz
    cadence_days = 120.0 / 86400.0
    time = np.arange(0.0, 27.0, cadence_days)
    flux = np.ones_like(time)
    for harmonic in range(-4, 5):
        amplitude = 120e-6 * math.exp(
            -((harmonic * dnu_demo_uhz) ** 2) / (2.0 * envelope_sigma_uhz**2)
        )
        frequency_cpd = (numax_demo_uhz + harmonic * dnu_demo_uhz) * CPD_PER_UHZ
        flux = flux + amplitude * np.sin(2.0 * np.pi * frequency_cpd * time)
    flux = flux + rng.normal(0.0, 30e-6, size=time.shape)

    result = estimate_oscillation_envelope(time, flux, 100.0, 2000.0)
    assert result["dnu_candidate_uhz"] == pytest.approx(135.1, abs=10.0), (
        "solar analog Î”Î½ recovery must succeed within 10 ÂµHz"
    )


def test_spacing_correlation_captures_a_120_uhz_comb():
    from exonym.asteroseismology import DNU_MAX_UHZ, spacing_correlation

    frequency_uhz = np.linspace(700.0, 1300.0, 6001)
    whitened = np.ones_like(frequency_uhz)
    for peak_uhz in (760.0, 880.0, 1000.0, 1120.0, 1240.0):
        whitened += 5.0 * np.exp(-0.5 * ((frequency_uhz - peak_uhz) / 1.5) ** 2)

    dnu_uhz, correlation, lag_grid = spacing_correlation(
        frequency_uhz,
        whitened,
        numax_uhz=1000.0,
    )

    assert DNU_MAX_UHZ == pytest.approx(200.0)
    assert lag_grid[-1] == pytest.approx(DNU_MAX_UHZ)
    assert dnu_uhz == pytest.approx(120.0, abs=1.0)
    assert correlation > 0.5


def test_spacing_correlation_reports_no_dnu_for_a_flat_spectrum():
    from exonym.asteroseismology import DNU_MAX_UHZ, spacing_correlation

    frequency_uhz = np.linspace(700.0, 1300.0, 6001)
    dnu_uhz, correlation, lag_grid = spacing_correlation(
        frequency_uhz,
        np.ones_like(frequency_uhz),
        numax_uhz=1000.0,
    )

    assert dnu_uhz is None
    assert correlation is None
    assert lag_grid[-1] == pytest.approx(DNU_MAX_UHZ)


def test_numax_clipping_reports_requested_and_effective_bounds(monkeypatch):
    import exonym.asteroseismology as asteroseismology

    time = np.linspace(0.0, 10.0, 100)
    support = asteroseismology.frequency_support(time)

    def fake_power_spectrum(_time, _flux, frequency_min_uhz, frequency_max_uhz):
        assert frequency_min_uhz == pytest.approx(50.0)
        assert frequency_max_uhz == pytest.approx(support["nyquist_uhz"])
        frequency = np.linspace(frequency_min_uhz, frequency_max_uhz, 32)
        return frequency, np.ones_like(frequency), np.ones_like(frequency), frequency

    monkeypatch.setattr(asteroseismology, "compute_power_spectrum", fake_power_spectrum)
    monkeypatch.setattr(
        asteroseismology,
        "spacing_correlation",
        lambda *_args: (120.0, 0.9, np.array([120.0])),
    )

    with pytest.warns(UserWarning) as warnings:
        result = asteroseismology.estimate_oscillation_envelope(
            time,
            np.ones(100),
            50.0,
            9000.0,
        )

    assert len(warnings) == 1
    assert result["numax_min_requested_uhz"] == pytest.approx(50.0)
    assert result["numax_max_requested_uhz"] == pytest.approx(9000.0)
    assert result["numax_min_used"] == pytest.approx(50.0)
    assert result["numax_max_used"] == pytest.approx(support["nyquist_uhz"])
    assert result["numax_min_clipped"] is False
    assert result["numax_max_clipped"] is True
    assert result["frequency_support"] == pytest.approx(support)


def test_asteroseismic_artifact_retains_numax_bound_provenance(tmp_path, monkeypatch):
    import exonym.asteroseismology as asteroseismology
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "asteroseismic-bound-test")
    time = np.linspace(0.0, 10.0, 200)
    table = {
        "time": time,
        "flux": np.ones_like(time),
        "flux_err": np.full_like(time, 1e-4),
        "sector": np.ones(time.size, dtype=int),
    }
    ephemeris = {
        "period_days": 2.0,
        "epoch_btjd": 1.0,
        "duration_days": 0.1,
        "source": "candidate-data",
        "field_sources": {
            "period_days": "candidate-data",
            "epoch_btjd": "candidate-data",
            "duration_days": "candidate-data",
        },
    }
    stellar = {
        "teff_k": 5700.0,
        "teff_k_err": 75.0,
        "mass_solar": 1.0,
        "radius_solar": 1.0,
        "source": "candidate-data",
    }
    envelope = {
        "numax_candidate_uhz": 1000.0,
        "dnu_candidate_uhz": 120.0,
        "dnu_correlation": 0.9,
        "envelope_peak_ratio": 2.0,
        "rayleigh_uhz": 1.0,
        "baseline_days": 10.0,
        "numax_min_requested_uhz": 50.0,
        "numax_max_requested_uhz": 9000.0,
        "numax_min_used": 100.0,
        "numax_max_used": 2000.0,
        "numax_min_clipped": True,
        "numax_max_clipped": True,
        "frequency_support": {
            "baseline_days": 10.0,
            "median_cadence_seconds": 120.0,
            "rayleigh_uhz": 1.0,
            "nyquist_uhz": 2000.0,
            "duty_cycle": 0.5,
        },
    }
    unavailable_pysyd = {
        "status": "unavailable",
        "manifest_path": workspace.path / "runs" / "pysyd" / "manifest.json",
        "crosscheck": None,
    }
    unavailable_tess_atl = {
        "status": "unavailable",
        "manifest_path": workspace.path / "runs" / "tess-atl" / "manifest.json",
    }

    monkeypatch.setattr(asteroseismology, "load_light_curve_table", lambda *_args, **_kwargs: table)
    monkeypatch.setattr(asteroseismology, "load_transit_ephemeris", lambda *_args, **_kwargs: ephemeris)
    monkeypatch.setattr(asteroseismology, "load_stellar_parameters", lambda *_args, **_kwargs: stellar)
    monkeypatch.setattr(
        asteroseismology,
        "_highpass_segments",
        lambda source_time, source_flux, *_args, **_kwargs: (source_time, source_flux),
    )
    monkeypatch.setattr(asteroseismology, "estimate_oscillation_envelope", lambda *_args: envelope)
    monkeypatch.setattr(
        asteroseismology,
        "seismic_mass_radius",
        lambda *_args, **_kwargs: {"mass_solar": 1.0, "radius_solar": 1.0, "method": "test"},
    )
    monkeypatch.setattr(asteroseismology, "seismic_sanity_check", lambda *_args, **_kwargs: {"plausible": True})
    monkeypatch.setattr(
        asteroseismology,
        "seismic_uncertainty_summary",
        lambda *_args, **_kwargs: {"status": "unavailable"},
    )
    monkeypatch.setattr(asteroseismology, "_run_pysyd_adapter", lambda *_args, **_kwargs: unavailable_pysyd)
    monkeypatch.setattr(asteroseismology, "_record_tess_atl_adapter", lambda *_args, **_kwargs: unavailable_tess_atl)

    output = asteroseismology.run_asteroseismology(workspace, 50.0, 9000.0)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["search_range_uhz"] == [100.0, 2000.0]
    assert payload["requested_search_range_uhz"] == [50.0, 9000.0]
    assert payload["numax_search_bounds"] == {
        "supported_range_uhz": [1.0, 2000.0],
        "lower_clipped": True,
        "upper_clipped": True,
    }


def test_frequency_support_rejects_an_unsupported_nyquist_range():
    from exonym.asteroseismology import compute_power_spectrum, frequency_support

    time = np.arange(100, dtype=float) * (30.0 / 1440.0)
    support = frequency_support(time)

    with pytest.raises(ValueError, match="cadence-supported"):
        compute_power_spectrum(time, np.ones_like(time), 1.0, support["nyquist_uhz"] * 1.1)


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


def test_seismic_sanity_check_requires_physical_values_or_catalog_consistency():
    from exonym.asteroseismology import seismic_sanity_check

    implausible = {"mass_solar": 25.99, "radius_solar": 6.68}
    verdict = seismic_sanity_check(
        implausible, radius_prior_solar=2.15, prior_is_catalog=True
    )
    assert verdict["plausible"] is False
    assert any("prior" in reason for reason in verdict["reasons"])

    plausible = {"mass_solar": 1.9, "radius_solar": 2.1}
    assert seismic_sanity_check(plausible, radius_prior_solar=2.15, prior_is_catalog=True)[
        "plausible"
    ]

    synthetic_source = seismic_sanity_check(implausible)
    assert synthetic_source["plausible"] is True

    invalid = seismic_sanity_check({"mass_solar": float("nan"), "radius_solar": -1.0})
    assert invalid["plausible"] is False
    assert invalid["reasons"] == ["mass is not positive and finite", "radius is not positive and finite"]


def test_asteroseismic_optional_adapters_write_hashed_candidate_local_manifests(tmp_path, monkeypatch):
    from pathlib import Path

    import exonym.asteroseismology as asteroseismology
    from exonym.isolation import IsolationReport
    from exonym.schemas import validate_schemas
    from exonym.workspace import create_candidate

    # Keep adapter import failures local to this module under test rather than
    # mutating the shared importlib module used by schema/resource loading.
    monkeypatch.setattr(
        asteroseismology,
        "importlib",
        SimpleNamespace(
            import_module=asteroseismology.importlib.import_module,
            util=SimpleNamespace(find_spec=asteroseismology.importlib.util.find_spec),
        ),
    )

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
    from exonym.localization import build_difference_image, localize_difference_image

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
    difference_image, valid = build_difference_image(out_image - deficit, out_image)
    assert np.allclose(difference_image[valid], deficit[valid])
    aperture = np.zeros(shape, dtype=bool)
    aperture[2:-2, 2:-2] = True
    offset = localize_difference_image(difference_image, aperture, target_x, target_y)
    assert offset["ra_offset_arcsec"] == pytest.approx(6.3, abs=2.5)
    assert offset["dec_offset_arcsec"] == pytest.approx(-4.2, abs=2.5)
    assert offset["offset_arcsec"] > 2.0
    assert offset["n_difference_pixels"] >= 3


def test_prf_localization_marks_a_two_pixel_depth_core_unresolved():
    from exonym.localization import localize_difference_image

    depth_map = np.full((3, 3), 0.001, dtype=float)
    depth_map[1, 1] = 0.010
    depth_map[1, 2] = 0.009

    result = localize_difference_image(
        depth_map, np.ones_like(depth_map, dtype=bool), target_x=1.0, target_y=1.0
    )

    assert result["n_difference_pixels"] == 2
    assert np.isnan(result["offset_arcsec"])


def test_prf_localization_json_sanitizer_replaces_nonfinite_diagnostics():
    from exonym.localization import _json_safe

    assert _json_safe({"offset_arcsec": float("nan")}) == {"offset_arcsec": None}


def test_prf_localization_requires_a_competing_source_for_target_dominance(tmp_path):
    from exonym.localization import run_prf_localization

    # Arrange
    workspace = type("Workspace", (), {"path": tmp_path, "candidate_id": "test-target"})()

    # Act
    with patch("exonym.localization.load_transit_ephemeris", return_value={}), patch(
        "exonym.localization.load_tpf_cubes", return_value=[]
    ):
        output = run_prf_localization(workspace)
    report = json.loads(output.read_text(encoding="utf-8"))

    # Assert
    assert report["source"] == "not-run-mission-calibrated-prf-required"
    assert report["summary"]["conclusion"] == "inconclusive_mission_calibrated_prf_required"
    assert report["summary"]["sectors_with_competing_sources_modeled"] == 0
    assert report["sector_results"] == []


def test_prf_localization_ignores_missing_competitor_ratio_in_summary(tmp_path):
    from exonym.localization import run_prf_localization

    workspace = type("Workspace", (), {"path": tmp_path, "candidate_id": "test-target"})()
    ephemeris = {
        "period_days": 2.0,
        "epoch_btjd": 0.5,
        "duration_days": 0.1,
        "source": "candidate-data",
        "field_sources": {
            "period_days": "candidate-data",
            "epoch_btjd": "candidate-data",
            "duration_days": "candidate-data",
        },
    }
    fake_row = {
        "sector": 1,
        "skipped": False,
        "n_modeled_neighbors": 1,
        "target_to_max_other_difference_ratio": None,
        "difference_centroid_offset_arcsec": 0.2,
    }

    with patch("exonym.localization.load_transit_ephemeris", return_value=ephemeris), patch(
        "exonym.localization._load_archival_gaia_neighbors", return_value=([], "test-catalog")
    ), patch(
        "exonym.localization.load_tpf_cubes",
        return_value=[{"path": tmp_path / "missing.fits", "sector": 1, "header": {}}],
    ), patch(
        "exonym.localization.extract_tpf_difference_image",
        return_value=(np.ones((3, 3)), np.ones((3, 3), dtype=bool), 1.0, 1.0, 4, 4),
    ), patch("exonym.localization._fit_one_difference_image", return_value=fake_row):
        output = run_prf_localization(workspace)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["sectors_with_competing_sources_modeled"] == 0
    assert report["summary"]["median_target_to_other_difference_ratio"] is None


def test_prf_localization_retains_skipped_tpf_diagnostics(tmp_path):
    from exonym.localization import run_prf_localization

    workspace = type("Workspace", (), {"path": tmp_path, "candidate_id": "test-target"})()

    def fake_tpf_loader(*args, **kwargs):
        kwargs["skipped_products"].append(
            {"path": "data/raw/s0031_tp.fits", "reason": "missing-quality-column"}
        )
        return []

    with patch("exonym.localization.load_transit_ephemeris", return_value={}), patch(
        "exonym.localization.load_tpf_cubes", side_effect=fake_tpf_loader
    ):
        output = run_prf_localization(workspace)
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["skipped_tpf_products"] == []


def test_tpf_loader_skips_unverified_sector_and_uses_canonical_sector_name(tmp_path):
    from astropy.io import fits

    from exonym.inputs import load_tpf_cubes
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "tpf-sector-test")
    raw = workspace.path / "data" / "raw"
    cadence_count = 60
    pixels = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TIME", format="D", array=np.arange(cadence_count, dtype=float)),
            fits.Column(name="QUALITY", format="J", array=np.zeros(cadence_count, dtype=np.int32)),
            fits.Column(
                name="FLUX",
                format="4E",
                dim="(2,2)",
                array=np.ones((cadence_count, 2, 2), dtype=np.float32),
            ),
        ]
    )
    aperture = fits.ImageHDU(data=np.ones((2, 2), dtype=np.int16))
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "TESS"
    primary.header["TIMESYS"] = "TDB"
    primary.header["TIMEUNIT"] = "d"
    primary.header["BJDREFI"] = 2457000
    for filename in ("unverified_tp.fits", "s0030_tp.fits"):
        fits.HDUList([primary, pixels, aperture]).writeto(raw / filename)

    with pytest.warns(UserWarning, match="sector cannot be verified"):
        cubes = load_tpf_cubes(workspace)

    assert len(cubes) == 1
    assert cubes[0]["path"].name == "s0030_tp.fits"
    assert cubes[0]["sector"] == 30


def test_tpf_loader_skips_missing_quality_with_a_retained_diagnostic(tmp_path):
    from astropy.io import fits

    from exonym.inputs import load_tpf_cubes
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "tpf-quality-test")
    raw = workspace.path / "data" / "raw"
    cadence_count = 60
    pixels = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TIME", format="D", array=np.arange(cadence_count, dtype=float)),
            fits.Column(
                name="FLUX",
                format="4E",
                dim="(2,2)",
                array=np.ones((cadence_count, 2, 2), dtype=np.float32),
            ),
        ]
    )
    aperture = fits.ImageHDU(data=np.ones((2, 2), dtype=np.int16))
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "TESS"
    primary.header["TIMESYS"] = "TDB"
    primary.header["TIMEUNIT"] = "d"
    primary.header["BJDREFI"] = 2457000
    product = raw / "s0031_tp.fits"
    fits.HDUList([primary, pixels, aperture]).writeto(product)
    skipped_products = []

    with pytest.warns(UserWarning, match="no QUALITY column"):
        cubes = load_tpf_cubes(workspace, skipped_products=skipped_products)

    assert cubes == []
    assert skipped_products == [
        {
            "path": "data/raw/s0031_tp.fits",
            "reason": "missing-quality-column",
        }
    ]


@pytest.mark.parametrize(
    "quality_values",
    (
        np.full(60, 0.5, dtype=float),
        np.full(60, np.nan, dtype=float),
    ),
    ids=("fractional", "non-finite"),
)
def test_tpf_loader_skips_nonintegral_or_nonfinite_quality_values(tmp_path, quality_values):
    from astropy.io import fits

    from exonym.inputs import load_tpf_cubes
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "tpf-quality-values")
    raw = workspace.path / "data" / "raw"
    cadence_count = quality_values.size
    pixels = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TIME", format="D", array=np.arange(cadence_count, dtype=float)),
            fits.Column(
                name="FLUX",
                format="4E",
                dim="(2,2)",
                array=np.ones((cadence_count, 2, 2), dtype=np.float32),
            ),
            fits.Column(name="QUALITY", format="D", array=quality_values),
        ]
    )
    aperture = fits.ImageHDU(data=np.ones((2, 2), dtype=np.int16))
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "TESS"
    primary.header["TIMESYS"] = "TDB"
    primary.header["TIMEUNIT"] = "d"
    primary.header["BJDREFI"] = 2457000
    product = raw / "s0031_tp.fits"
    fits.HDUList([primary, pixels, aperture]).writeto(product)
    skipped_products = []

    with pytest.warns(UserWarning, match="unusable QUALITY column"):
        cubes = load_tpf_cubes(workspace, skipped_products=skipped_products)

    assert cubes == []
    assert skipped_products == [
        {
            "path": "data/raw/s0031_tp.fits",
            "reason": "unusable-quality-column",
        }
    ]


def test_prf_nnls_assigns_difference_flux_to_target():
    from exonym.localization import fit_difference_image_prf, gaussian_prf_kernel

    shape = (11, 11)
    yy, xx = np.indices(shape, dtype=float)
    kernel = gaussian_prf_kernel(xx, yy, 5.0, 5.0)
    difference_image = 14.0 * kernel
    pixel_mask = np.ones(shape, dtype=bool)
    amplitudes, residual, n_pixels = fit_difference_image_prf(
        difference_image, pixel_mask, [5.0, 8.0], [5.0, 5.0]
    )
    assert n_pixels > 5
    assert amplitudes[0] > 10.0 * amplitudes[1]
    assert residual is not None


def test_prf_scene_injection_recovers_detector_scale_gaussian_amplitudes():
    """The default screening template recovers a broad injected two-source scene."""
    from exonym.localization import PRF_FWHM_PIXELS, fit_difference_image_prf, gaussian_prf_kernel

    shape = (15, 15)
    yy, xx = np.indices(shape, dtype=float)
    injected = np.array([12.0, 4.0])
    difference_image = (
        injected[0] * gaussian_prf_kernel(xx, yy, 6.0, 7.0, fwhm_pixels=2.0)
        + injected[1] * gaussian_prf_kernel(xx, yy, 8.5, 7.0, fwhm_pixels=2.0)
    )

    amplitudes, residual, _ = fit_difference_image_prf(
        difference_image, np.ones(shape, dtype=bool), [6.0, 8.5], [7.0, 7.0]
    )

    assert PRF_FWHM_PIXELS == pytest.approx(2.0)
    assert amplitudes == pytest.approx(injected, rel=0.02)
    assert residual == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Scientific analysis modules: SED
# ---------------------------------------------------------------------------


def test_sed_recovers_synthetic_photometry():
    from exonym.sed import _fit_blackbody
    from tests.fixtures.synthetic_observations import _synthetic_photometry

    stellar = {
        "teff_k": 5772.0,
        "logg_cgs": 4.438,
        "feh": 0.0,
        "mass_solar": 1.0,
        "radius_solar": 1.0,
        "parallax_mas": 10.0,
        "parallax_mas_err": 0.05,
    }
    observations = _synthetic_photometry(stellar)
    result = _fit_blackbody(observations, stellar, n_walkers=24, burn_in=150, production=250)
    posterior = result["posterior"]
    assert posterior["teff_k"]["median"] == pytest.approx(5772.0, abs=250.0)
    assert posterior["radius_solar"]["median"] == pytest.approx(1.0, abs=0.35)
    assert posterior["logg_cgs"]["median"] == pytest.approx(4.438, abs=0.3)


def test_sed_blackbody_rejects_invalid_parallax_draws_before_summaries(monkeypatch):
    import exonym.sed as sed

    stellar = {
        "teff_k": 5772.0,
        "logg_cgs": 4.438,
        "feh": 0.0,
        "mass_solar": 1.0,
        "radius_solar": 1.0,
        "parallax_mas": 10.0,
        "parallax_mas_err": 0.05,
    }

    def fake_run(log_probability, start, n_walkers, burn_in, production, seed):
        samples = np.tile(np.asarray(start, dtype=float), (4, 1))
        return samples, SimpleNamespace(acceptance_fraction=np.full(n_walkers, 0.5))

    fake_rng = SimpleNamespace(
        normal=lambda mean, sigma, size: np.array([10.0, 0.0, -1.0, np.nan])
    )
    monkeypatch.setattr(sed, "_run_emcee", fake_run)
    monkeypatch.setattr(sed.np.random, "default_rng", lambda seed: fake_rng)
    monkeypatch.setattr(
        sed,
        "blackbody_model_magnitudes",
        lambda teff_k, log_radius_over_distance, av_mag, band_data: np.zeros(len(band_data)),
    )

    result = sed._fit_blackbody([("J", 0.0, 0.1)], stellar, n_walkers=4, burn_in=1, production=1)

    diagnostics = result["fit_quality"]["parallax_draws"]
    assert diagnostics == {
        "proposed_count": 4,
        "accepted_positive_finite_count": 1,
        "rejected_nonpositive_or_nonfinite_count": 3,
        "rejection_rate": pytest.approx(0.75),
        "policy": "non-positive or non-finite draws are rejected before distance-derived summaries",
    }
    posterior = result["posterior"]
    assert posterior["distance_pc"]["median"] == pytest.approx(100.0)
    for key in ("distance_pc", "radius_solar", "luminosity_solar", "logg_cgs"):
        assert np.isfinite(posterior[key]["median"])


def test_sed_blackbody_fails_closed_when_all_parallax_draws_are_invalid(monkeypatch):
    import exonym.sed as sed

    stellar = {
        "teff_k": 5772.0,
        "logg_cgs": 4.438,
        "feh": 0.0,
        "mass_solar": 1.0,
        "radius_solar": 1.0,
        "parallax_mas": 10.0,
        "parallax_mas_err": 0.05,
    }

    def fake_run(log_probability, start, n_walkers, burn_in, production, seed):
        samples = np.tile(np.asarray(start, dtype=float), (3, 1))
        return samples, SimpleNamespace(acceptance_fraction=np.full(n_walkers, 0.5))

    fake_rng = SimpleNamespace(normal=lambda mean, sigma, size: np.array([0.0, -1.0, np.nan]))
    monkeypatch.setattr(sed, "_run_emcee", fake_run)
    monkeypatch.setattr(sed.np.random, "default_rng", lambda seed: fake_rng)

    with pytest.raises(RuntimeError, match="rejected every parallax draw"):
        sed._fit_blackbody([("J", 0.0, 0.1)], stellar, n_walkers=4, burn_in=1, production=1)


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


def _mock_candidate_fit_inputs(monkeypatch):
    from tests.fixtures.synthetic_observations import _synthetic_transit_table

    ephemeris = {
        "period_days": 3.2,
        "epoch_btjd": 1.0,
        "duration_days": 0.12,
        "depth_ppm": 1200.0,
        "source": "candidate-config",
        "field_sources": {
            "period_days": "candidate-config",
            "epoch_btjd": "candidate-config",
            "duration_days": "candidate-config",
            "depth_ppm": "candidate-config",
        },
    }
    stellar = {
        "mass_solar": 1.0,
        "mass_solar_err": 0.1,
        "radius_solar": 1.0,
        "radius_solar_err": 0.05,
        "source": "candidate-data",
    }
    monkeypatch.setattr("exonym.transit_fit.load_transit_ephemeris", lambda *args, **kwargs: ephemeris)
    monkeypatch.setattr("exonym.transit_fit.load_stellar_parameters", lambda *args, **kwargs: stellar)
    monkeypatch.setattr(
        "exonym.transit_fit.load_light_curve_table",
        lambda *args, **kwargs: _synthetic_transit_table(ephemeris),
    )


def test_transit_fit_recovers_synthetic_parameters(tmp_path, monkeypatch):
    from exonym.transit_fit import run_mcmc_transit_fit
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "fit-test")
    _mock_candidate_fit_inputs(monkeypatch)
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
    assert payload["parameter_names"] == [
        "rp_rs",
        "log_rho_star",
        "impact_parameter",
        "baseline",
        "log_jitter",
        "q1",
        "q2",
    ]
    assert payload["source"] == "candidate-data"
    assert payload["likelihood"]["cadence"] == "native"
    assert payload["likelihood"]["exposure_seconds_by_sector"] == {"1": pytest.approx(120.0)}
    assert payload["density_prior"]["log10_sigma"] == pytest.approx(
        np.sqrt(0.1**2 + 0.15**2) / np.log(10.0)
    )
    jitter_prior = payload["assumptions"]["jitter_prior"]
    assert jitter_prior["distribution"] == "half-cauchy-on-jitter"
    assert jitter_prior["scale_normalized_flux"] == pytest.approx(1e-3)
    assert jitter_prior["data_dependent"] is False
    assert jitter_prior["empirical_bayes"] is False


def test_native_transit_window_preserves_sector_cadence_and_baselines():
    from exonym.transit_fit import (
        _initial_fit_parameters,
        _log_prior,
        _native_transit_window_data,
        _parameter_names,
    )

    cadence_days = 120.0 / 86400.0
    first_sector_time = np.arange(0.0, 0.4, cadence_days)
    second_sector_time = np.arange(9.0, 9.4, cadence_days)
    time = np.concatenate((first_sector_time, second_sector_time))
    table = {
        "time": time,
        "flux": np.ones_like(time),
        "flux_err": np.full_like(time, 100e-6),
        "sector": np.concatenate((
            np.full(first_sector_time.size, 701),
            np.full(second_sector_time.size, 703),
        )),
    }
    ephemeris = {"period_days": 3.0, "epoch_btjd": 0.0, "duration_days": 0.2}

    phase, flux, flux_err, sector_index, labels, exposures = _native_transit_window_data(
        table, ephemeris
    )

    assert phase.size == flux.size == flux_err.size == sector_index.size
    assert phase.size == time.size
    assert labels == [701, 703]
    assert np.allclose(exposures, 120.0)
    assert set(sector_index) == {0, 1}
    names = _parameter_names(False, n_sectors=2, sector_labels=labels)
    assert names[3:5] == ["baseline_sector_701", "baseline_sector_703"]
    theta = _initial_fit_parameters(1000.0, 1.0, 100e-6, eccentric=False, n_sectors=2)
    assert np.isfinite(_log_prior(theta, 1.0, 0.1, eccentric=False, n_sectors=2))


def test_transit_fit_eccentric_mode_runs(tmp_path, monkeypatch):
    from exonym.transit_fit import run_mcmc_transit_fit
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "fit-ecc-test")
    _mock_candidate_fit_inputs(monkeypatch)
    output = run_mcmc_transit_fit(
        workspace, n_samples=120, eccentric=True, n_walkers=20, burn_in=30
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "eccentricity" in payload["posterior"]
    assert "omega_deg" in payload["posterior"]
    assert payload["posterior"]["eccentricity"]["median"] < 0.3


def test_transit_fit_emcee_chain_is_reproducible_for_a_fixed_seed(tmp_path, monkeypatch):
    from exonym.transit_fit import run_mcmc_transit_fit
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "fit-emcee-reproducible")
    _mock_candidate_fit_inputs(monkeypatch)
    output = run_mcmc_transit_fit(workspace, n_samples=60, n_walkers=16, burn_in=20, seed=19)
    first_chain = np.load(workspace.path / "outputs" / "mcmc_transit_fit_chain.npy")
    run_mcmc_transit_fit(workspace, n_samples=60, n_walkers=16, burn_in=20, seed=19)

    payload = json.loads(output.read_text(encoding="utf-8"))
    second_chain = np.load(workspace.path / "outputs" / "mcmc_transit_fit_chain.npy")
    assert payload["mcmc"]["random_seed"] == 19
    assert payload["mcmc"]["random_generator"] == "numpy.random.RandomState (MT19937)"
    assert np.array_equal(second_chain, first_chain)


def test_transit_fit_emcee_progress_callback_uses_global_phase_counts(tmp_path, monkeypatch):
    from exonym.transit_fit import run_mcmc_transit_fit
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "fit-emcee-progress")
    _mock_candidate_fit_inputs(monkeypatch)
    events = []

    run_mcmc_transit_fit(
        workspace,
        n_samples=4,
        n_walkers=16,
        burn_in=2,
        seed=23,
        sampler="emcee",
        progress_callback=lambda done, total, **metadata: events.append((done, total, metadata)),
    )

    assert events[0][:2] == (1, 6)
    assert events[0][2]["burn_in"] == 2
    assert events[0][2]["production"] == 4
    assert events[-1][:2] == (6, 6)
    assert events[-1][2]["resumed"] is False


def test_transit_fit_emcee_resume_reports_saved_global_offset(tmp_path, monkeypatch):
    from exonym.transit_fit import run_mcmc_transit_fit
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "fit-emcee-resume-progress")
    _mock_candidate_fit_inputs(monkeypatch)
    original_unlink = Path.unlink

    def preserve_checkpoint(path, *args, **kwargs):
        if path.name.endswith(".checkpoint.npz"):
            return None
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", preserve_checkpoint)
    run_mcmc_transit_fit(
        workspace,
        n_samples=4,
        n_walkers=16,
        burn_in=2,
        seed=29,
        sampler="emcee",
        checkpoint_interval=1,
    )
    checkpoint = workspace.path / "outputs" / "mcmc_transit_fit_chain.checkpoint.npz"
    events = []

    run_mcmc_transit_fit(
        workspace,
        n_samples=8,
        n_walkers=16,
        burn_in=2,
        seed=29,
        sampler="emcee",
        resume=str(checkpoint),
        progress_callback=lambda done, total, **metadata: events.append((done, total, metadata)),
    )

    assert events[0][:2] == (7, 10)
    assert events[-1][:2] == (10, 10)
    assert all(event[2]["resumed"] is True for event in events)


def test_dynesty_fit_writes_evidence_and_reproducible_compatibility_chain(tmp_path, monkeypatch):
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
            if kwargs["print_progress"]:
                kwargs["print_func"](self.results, 4, 12)
            assert kwargs["nlive_init"] > 0
            assert kwargs["dlogz_init"] == pytest.approx(0.25)
            assert "dlogz" not in kwargs
            assert "maxcall" not in kwargs

    workspace = create_candidate(tmp_path, "fit-dynesty-test")
    _mock_candidate_fit_inputs(monkeypatch)
    fake_dynesty = SimpleNamespace(DynamicNestedSampler=FakeDynamicNestedSampler, __version__="test")
    progress_events = []

    with patch.dict(sys.modules, {"dynesty": fake_dynesty}):
        output = run_mcmc_transit_fit(
            workspace,
            n_samples=40,
            sampler="dynesty",
            seed=5,
            dlogz_tolerance=0.25,
            progress_callback=lambda done, total, **metadata: progress_events.append(
                (done, total, metadata)
            ),
        )
        first_chain = np.load(workspace.path / "outputs" / "mcmc_transit_fit_chain.npy")
        run_mcmc_transit_fit(
            workspace,
            n_samples=40,
            sampler="dynesty",
            seed=5,
            dlogz_tolerance=0.25,
        )

    payload = json.loads(output.read_text(encoding="utf-8"))
    chain = np.load(workspace.path / "outputs" / "mcmc_transit_fit_chain.npy")
    assert output.name == "mcmc_transit_fit.json"
    assert payload["sampler"] == "dynesty"
    assert payload["parameter_names"] == [
        "rp_rs",
        "log_rho_star",
        "impact_parameter",
        "baseline",
        "log_jitter",
        "q1",
        "q2",
    ]
    assert payload["evidence"]["log_z"] == pytest.approx(0.0)
    assert payload["diagnostics"]["resampling"] == "systematic equal-weight resampling"
    assert payload["diagnostics"]["resampling_seed"] == 5
    assert payload["diagnostics"]["dlogz_init_tolerance"] == pytest.approx(0.25)
    assert payload["diagnostics"]["dlogz_tolerance"] == pytest.approx(0.25)
    configuration = payload["diagnostics"]["dynesty_run_configuration"]
    assert configuration["initial_baseline"] == {
        "nlive_init": 50,
        "criterion": "dlogz_init",
        "dlogz_init": 0.25,
    }
    assert configuration["final_dynamic_stopping"]["criterion"] == (
        "dynesty-default-stopping-function"
    )
    assert configuration["final_dynamic_stopping"]["result_criterion"] is None
    assert payload["diagnostics"]["sampler_stop_criterion"] is None
    jitter_prior = payload["assumptions"]["jitter_prior"]
    assert jitter_prior["distribution"] == "half-cauchy-on-jitter"
    assert jitter_prior["scale_normalized_flux"] == pytest.approx(1e-3)
    assert jitter_prior["data_dependent"] is False
    assert jitter_prior["empirical_bayes"] is False
    assert payload["diagnostics"]["sampler_stop_criterion_status"] == (
        "not-reported-by-dynesty-results"
    )
    assert progress_events == [
        (
            4,
            None,
            {
                "sub_phase": "nested sampling",
                "log_z": 0.0,
                "log_z_error": 0.1,
                "likelihood_calls": 12,
            },
        )
    ]
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


def test_transit_likelihood_rejects_an_invalid_batman_forward_model(monkeypatch):
    from exonym.transit_fit import _initial_fit_parameters, _log_likelihood

    monkeypatch.setattr(
        "exonym.transit_fit.batman_transit_flux",
        lambda *_args, **_kwargs: None,
    )
    theta = _initial_fit_parameters(1200.0, 1.0, 1e-4, eccentric=False)
    result = _log_likelihood(
        theta,
        np.linspace(-0.1, 0.1, 32),
        np.ones(32),
        np.full(32, 1e-4),
        {"period_days": 3.2},
        eccentric=False,
    )

    assert result == -np.inf


def test_posterior_summaries_refuse_a_missing_batman_model(monkeypatch):
    from exonym.transit_fit import _initial_fit_parameters, _posterior_summaries

    monkeypatch.setattr(
        "exonym.transit_fit.batman_transit_flux",
        lambda *_args, **_kwargs: None,
    )
    chain = np.tile(_initial_fit_parameters(1200.0, 1.0, 1e-4, eccentric=False), (4, 1))

    with pytest.raises(RuntimeError, match="posterior-derived mid-transit depths"):
        _posterior_summaries(chain, {"period_days": 3.2}, eccentric=False)


def test_posterior_summaries_report_inclination_geometry_clipping(monkeypatch):
    from exonym.transit_fit import _initial_fit_parameters, _posterior_summaries

    monkeypatch.setattr(
        "exonym.transit_fit.batman_transit_flux",
        lambda time, *_args, **_kwargs: np.ones_like(time),
    )
    initial = _initial_fit_parameters(1200.0, 1.0, 1e-4, eccentric=False)
    chain = np.tile(initial, (4, 1))
    chain[2:, 2] = 100.0

    summaries = _posterior_summaries(chain, {"period_days": 3.2}, eccentric=False)

    assert summaries["inclination_deg"]["conjunction_distance_clip_fraction"] == pytest.approx(0.5)


def test_posterior_summaries_vectorize_kipping_limb_darkening_exactly(monkeypatch):
    from exonym.lightcurve import kipping_to_quadratic_limb_darkening
    from exonym.transit_fit import _initial_fit_parameters, _posterior_summaries

    monkeypatch.setattr(
        "exonym.transit_fit.batman_transit_flux",
        lambda time, *_args, **_kwargs: np.ones_like(time),
    )
    q1, q2 = 0.64, 0.20
    chain = np.tile(_initial_fit_parameters(1200.0, 1.0, 1e-4, eccentric=False), (8, 1))
    chain[:, 5] = q1
    chain[:, 6] = q2

    summaries = _posterior_summaries(chain, {"period_days": 3.2}, eccentric=False)

    expected_u1, expected_u2 = kipping_to_quadratic_limb_darkening(q1, q2)
    assert summaries["u1"]["median"] == pytest.approx(expected_u1)
    assert summaries["u2"]["median"] == pytest.approx(expected_u2)


def test_transit_fit_fails_loudly_when_batman_is_unavailable(tmp_path, monkeypatch):
    from exonym.transit_fit import run_mcmc_transit_fit
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "fit-batman-unavailable")

    def missing_batman():
        raise RuntimeError("batman-package is required for transit fitting")

    monkeypatch.setattr("exonym.transit_fit._require_batman", missing_batman)
    with pytest.raises(RuntimeError, match="batman-package"):
        run_mcmc_transit_fit(workspace)

    assert not (workspace.path / "outputs" / "mcmc_transit_fit.json").exists()
    assert not (workspace.path / "outputs" / "mcmc_transit_fit_chain.npy").exists()


def test_dynesty_stopping_configuration_is_recorded_without_a_results_stop_field(tmp_path, monkeypatch):
    """Record configured initial/dynamic stopping without inventing result metadata."""
    import json
    import sys

    import numpy as np
    from types import SimpleNamespace
    from unittest.mock import patch

    from exonym.transit_fit import run_mcmc_transit_fit
    from exonym.workspace import create_candidate

    class FakeDynamicNestedSampler:
        def __init__(self, log_likelihood, prior_transform, ndim, rstate):
            probe = prior_transform(np.full(ndim, 0.5))
            assert np.isfinite(log_likelihood(probe))
            samples = np.tile(probe, (4, 1))
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
            assert kwargs["dlogz_init"] == pytest.approx(0.5)
            assert "dlogz" not in kwargs
            assert "maxcall" not in kwargs

    workspace = create_candidate(tmp_path, "fit-dynesty-stopping")
    _mock_candidate_fit_inputs(monkeypatch)
    fake_dynesty = SimpleNamespace(DynamicNestedSampler=FakeDynamicNestedSampler, __version__="test")

    with patch.dict(sys.modules, {"dynesty": fake_dynesty}):
        output = run_mcmc_transit_fit(workspace, n_samples=40, sampler="dynesty", seed=5)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload["diagnostics"]["sampler_niter"], int), "sampler_niter must be recorded"
    assert payload["diagnostics"]["sampler_niter"] > 0, "sampler_niter must be a positive integer"
    configuration = payload["diagnostics"]["dynesty_run_configuration"]
    assert configuration["initial_baseline"]["criterion"] == "dlogz_init"
    assert configuration["initial_baseline"]["dlogz_init"] == pytest.approx(0.5)
    assert configuration["final_dynamic_stopping"] == {
        "criterion": "dynesty-default-stopping-function",
        "custom_stop_function": None,
        "custom_stop_kwargs": {},
        "use_stop": {
            "value": True,
            "source": "dynesty API default; not overridden by this run",
        },
        "configured_hard_stop_kwargs": {},
        "result_criterion": None,
        "result_criterion_status": "not-reported-by-dynesty-results",
    }
    assert payload["diagnostics"]["sampler_stop_criterion"] is None
    assert payload["diagnostics"]["sampler_stop_criterion_status"] == (
        "not-reported-by-dynesty-results"
    )

def test_stellar_density_a_rs_monotonic():
    from exonym.transit_fit import stellar_density_a_rs

    assert stellar_density_a_rs(1.0, 3.5) > stellar_density_a_rs(1.0, 1.0)
    assert stellar_density_a_rs(4.0, 3.5) > stellar_density_a_rs(1.0, 3.5)
    with pytest.raises(ValueError):
        stellar_density_a_rs(0.0, 3.5)


def test_transit_fit_initial_jitter_uses_natural_log_likelihood_units():
    from exonym.transit_fit import _initial_fit_parameters

    start = _initial_fit_parameters(1200.0, 1.0, 1e-4, eccentric=False)

    assert start[1] == pytest.approx(0.0)
    assert start[4] == pytest.approx(np.log(1e-4))
    assert np.exp(start[4]) == pytest.approx(1e-4)


def test_transit_fit_propagates_candidate_stellar_density_uncertainty():
    from exonym.transit_fit import _stellar_density_prior

    density_prior = _stellar_density_prior(
        {"mass_solar": 1.0, "mass_solar_err": 0.1, "radius_solar": 1.0, "radius_solar_err": 0.05}
    )

    assert density_prior["rho_solar"] == pytest.approx(1.0)
    assert density_prior["log10_sigma"] == pytest.approx(np.sqrt(0.1**2 + 0.15**2) / np.log(10.0))


def test_stellar_parameter_loader_retains_density_prior_uncertainties(tmp_path):
    from exonym.inputs import load_stellar_parameters
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "fit-stellar-uncertainties")
    parameter_path = workspace.path / "data" / "external" / "stellar_params.json"
    parameter_path.parent.mkdir(parents=True, exist_ok=True)
    parameter_path.write_text(
        json.dumps(
            {
                "teff_k": 5700.0,
                "logg_cgs": 4.4,
                "feh": 0.0,
                "mass_solar": 1.0,
                "mass_solar_err": 0.1,
                "radius_solar": 1.0,
                "radius_solar_err": 0.05,
            }
        ),
        encoding="utf-8",
    )

    stellar = load_stellar_parameters(workspace)

    assert stellar["source"] == "candidate-data"
    assert stellar["mass_solar_err"] == pytest.approx(0.1)
    assert stellar["radius_solar_err"] == pytest.approx(0.05)


def test_transit_fit_rejects_stellar_density_without_candidate_uncertainties():
    from exonym.transit_fit import _stellar_density_prior

    with pytest.raises(RuntimeError, match="mass_solar_err and radius_solar_err"):
        _stellar_density_prior({"mass_solar": 1.0, "radius_solar": 1.0})


def test_eccentric_transit_geometry_preserves_semimajor_axis_and_adjusts_inclination():
    from exonym.transit_fit import (
        conjunction_distance_a_rs,
        inclination_deg_from_impact_parameter,
    )

    conjunction_distance = conjunction_distance_a_rs(10.0, 0.5, 90.0)
    inclination = inclination_deg_from_impact_parameter(10.0, 1.0, 0.5, 90.0)

    assert conjunction_distance == pytest.approx(5.0)
    assert inclination == pytest.approx(np.degrees(np.arccos(1.0 / 5.0)))


# ---------------------------------------------------------------------------
# Scientific analysis modules: phase curve
# ---------------------------------------------------------------------------


def test_phase_curve_recovers_injected_reflection():
    from exonym.phasecurve import fit_phase_curve_components
    from tests.fixtures.synthetic_observations import _synthetic_phase_curve_table

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


def test_phase_curve_marks_zero_covariance_error_as_undefined(monkeypatch):
    """Do not turn an undefined covariance error into a zero-ppm limit."""
    import exonym.phasecurve as phasecurve
    from tests.fixtures.synthetic_observations import _synthetic_phase_curve_table

    table = _synthetic_phase_curve_table()
    ephemeris = {
        "period_days": table.pop("_period_days"),
        "epoch_btjd": table.pop("_epoch_btjd"),
        "duration_days": table.pop("_duration_days"),
    }
    monkeypatch.setattr(
        phasecurve,
        "cluster_sandwich_covariance",
        lambda design, *_args: (np.zeros((design.shape[1], design.shape[1])), 2),
    )

    result = phasecurve.fit_phase_curve_components(
        table["time"], table["flux"], table["flux_err"], table["sector"], ephemeris
    )

    assert result["status"] == "undefined_component_significance"
    assert result["maximum_absolute_significance_sigma"] is None
    assert all(
        component["significance_sigma"] is None
        and component["three_sigma_absolute_upper_bound_ppm"] is None
        for component in result["components"].values()
    )
    json.dumps(result, allow_nan=False)


def test_phase_curve_eccentric_secondary_control_uses_candidate_posterior(tmp_path):
    from exonym.phasecurve import (
        _secondary_eclipse_geometry_samples,
        resolve_secondary_eclipse_control,
    )
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "phase-ecc-test")
    outputs = workspace.path / "outputs"
    ephemeris = {"period_days": 3.0, "epoch_btjd": 100.0, "duration_days": 0.1}
    (outputs / "mcmc_transit_fit.json").write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "model": "batman quadratic limb darkening, stellar-density locked, eccentric orbit",
                "ephemeris": {"period_days": 3.0, "epoch_btjd": 100.0},
            }
        ),
        encoding="utf-8",
    )
    eccentricity = 0.2
    chain = np.tile(
        np.array([0.1, 0.0, 0.2, 1.0, -8.0, 0.3, 0.3, eccentricity**0.5, 0.0]),
        (16, 1),
    )
    np.save(str(outputs / "mcmc_transit_fit_chain.npy"), chain)

    arguments, report = resolve_secondary_eclipse_control(workspace, ephemeris)

    phases, duration_ratios, occulting = _secondary_eclipse_geometry_samples(
        np.array([0.0, eccentricity]),
        np.array([0.0, 0.0]),
        np.array([0.1, 0.1]),
        np.array([0.2, 0.2]),
    )
    assert occulting.tolist() == [True, True]
    assert phases[0] == pytest.approx(0.5)
    assert phases[1] == pytest.approx(0.5 + 2.0 * eccentricity / np.pi, abs=0.003)
    assert duration_ratios.tolist() == pytest.approx([1.0, 1.0])
    assert report["mode"] == "eccentric-posterior-marginalized-box-control"
    assert report["phase"]["median"] == pytest.approx(phases[1])
    assert report["duration_hours"]["median"] == pytest.approx(2.4)
    assert arguments["secondary_eclipse_template_total_samples"] == 16
    assert arguments["secondary_eclipse_phase_samples"].size == 16


def test_phase_curve_uses_declared_transit_chain_parameter_contract(tmp_path):
    """Named eccentric coordinates prevent a later chain-layout drift."""
    from exonym.phasecurve import resolve_secondary_eclipse_control
    from exonym.transit_fit import PARAMETER_NAMES_ECCENTRIC
    from exonym.workspace import create_candidate

    assert tuple(PARAMETER_NAMES_ECCENTRIC[-2:]) == ("sqe_cosw", "sqe_sinw")
    workspace = create_candidate(tmp_path, "phase-chain-contract")
    outputs = workspace.path / "outputs"
    parameter_names = list(PARAMETER_NAMES_ECCENTRIC) + ["future_trailing_parameter"]
    (outputs / "mcmc_transit_fit.json").write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "model": "batman quadratic limb darkening, stellar-density locked, eccentric orbit",
                "ephemeris": {"period_days": 3.0, "epoch_btjd": 100.0},
                "parameter_names": parameter_names,
            }
        ),
        encoding="utf-8",
    )
    chain = np.tile(
        np.array([0.1, 0.0, 0.2, 1.0, -8.0, 0.3, 0.3, 0.2**0.5, 0.0, 99.0]),
        (16, 1),
    )
    np.save(str(outputs / "mcmc_transit_fit_chain.npy"), chain)

    _, report = resolve_secondary_eclipse_control(
        workspace, {"period_days": 3.0, "epoch_btjd": 100.0, "duration_days": 0.1}
    )

    assert report["mode"] == "eccentric-posterior-marginalized-box-control"
    assert report["phase"]["median"] == pytest.approx(0.5 + 0.4 / np.pi, abs=0.003)


def test_phase_curve_run_records_eccentric_secondary_control(tmp_path, monkeypatch):
    from exonym import phasecurve
    from exonym.workspace import create_candidate
    from tests.fixtures.synthetic_observations import _synthetic_phase_curve_table

    workspace = create_candidate(tmp_path, "phase-ecc-output-test")
    outputs = workspace.path / "outputs"
    ephemeris = {
        "period_days": 3.0,
        "epoch_btjd": 100.0,
        "duration_days": 0.1,
        "time_system": "BTJD_TDB",
        "source": "candidate-config",
        "field_sources": {
            "period_days": "candidate-config",
            "epoch_btjd": "candidate-config",
            "duration_days": "candidate-config",
        },
    }
    (outputs / "mcmc_transit_fit.json").write_text(
        json.dumps(
            {
                "source": "candidate-data",
                "model": "batman quadratic limb darkening, stellar-density locked, eccentric orbit",
                "ephemeris": {"period_days": 3.0, "epoch_btjd": 100.0},
            }
        ),
        encoding="utf-8",
    )
    chain = np.tile(
        np.array([0.1, 0.0, 0.2, 1.0, -8.0, 0.3, 0.3, 0.2**0.5, 0.0]),
        (32, 1),
    )
    np.save(str(outputs / "mcmc_transit_fit_chain.npy"), chain)
    table = _synthetic_phase_curve_table()
    table.pop("_duration_days")
    table.pop("_epoch_btjd")
    table.pop("_period_days")
    monkeypatch.setattr(phasecurve, "load_light_curve_table", lambda *_args, **_kwargs: table)
    monkeypatch.setattr(phasecurve, "load_transit_ephemeris", lambda _workspace: ephemeris)

    output = phasecurve.run_phase_curve_search(workspace)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["secondary_eclipse_control"]["mode"] == "eccentric-posterior-marginalized-box-control"
    assert payload["secondary_box_template_method"] == "posterior-marginalized-eccentric-box"
    assert payload["secondary_box_phase"] == pytest.approx(0.5 + 0.4 / np.pi, abs=0.003)
    assert payload["secondary_eclipse_control"]["transit_fit"]["chain_path"] == "outputs/mcmc_transit_fit_chain.npy"


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
    if sector is not None:
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


def test_scoped_light_curve_requires_verified_or_canonical_sector(tmp_path):
    from exonym.inputs import load_light_curve_table
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "scoped-sector-resolution-test")
    raw = workspace.path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    _write_test_lightcurve(raw / "s0701_lc.fits", sector=None, n_points=200, seed=1)
    _write_test_lightcurve(raw / "unlabeled_lc.fits", sector=None, n_points=200, seed=2)

    with pytest.warns(UserWarning, match="sector cannot be verified"):
        table = load_light_curve_table(workspace, sectors=[701])

    assert table is not None
    assert set(np.unique(table["sector"])) == {701}
    assert [path.name for path in table["input_files"]] == ["s0701_lc.fits"]


def test_light_curve_table_skips_an_unusable_quality_column(tmp_path, monkeypatch):
    import types

    from exonym.inputs import load_light_curve_table
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "unusable-quality-test")
    product = workspace.path / "data" / "raw" / "s0001_lc.fits"
    product.parent.mkdir(parents=True, exist_ok=True)
    product.write_bytes(b"placeholder")

    fake_light_curve = types.SimpleNamespace(
        quality=types.SimpleNamespace(value=np.zeros(99, dtype=int)),
        time=types.SimpleNamespace(value=np.arange(100.0)),
    )
    monkeypatch.setitem(
        sys.modules,
        "lightkurve",
        types.SimpleNamespace(read=lambda _path: fake_light_curve),
    )

    with pytest.warns(UserWarning, match="unusable quality column"):
        table = load_light_curve_table(workspace)

    assert table is None


def test_light_curve_table_keeps_the_per_sector_binning_budget(tmp_path):
    from exonym.inputs import load_light_curve_table
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "per-sector-binning-test")
    raw = workspace.path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    _write_test_lightcurve(raw / "s0001_lc.fits", sector=1, n_points=4500, seed=1)
    _write_test_lightcurve(raw / "s0002_lc.fits", sector=2, n_points=4500, seed=2)

    table = load_light_curve_table(workspace, max_points=4000)

    assert table is not None
    assert len(table["time"]) == 8000
    assert set(np.unique(table["sector"])) == {1, 2}


def test_localization_ignores_unvalidated_external_gaia_csv(tmp_path):
    from exonym.localization import _load_archival_gaia_neighbors
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
    rows, metadata = _load_archival_gaia_neighbors(workspace)

    assert rows == []
    assert metadata["availability"] == "unavailable"


# ---------------------------------------------------------------------------
# Scientific analysis modules: TTV
# ---------------------------------------------------------------------------


def test_ttv_linear_ephemeris_has_small_residuals():
    from exonym.search import calculate_ttv_super_period
    from exonym.transit_fit import stellar_density_a_rs
    from exonym.ttv import (
        enumerate_companion_super_periods,
        transit_template_parameters,
        transit_timing_analysis,
    )
    from tests.fixtures.synthetic_observations import _synthetic_timing_table

    table = _synthetic_timing_table(ttv_amplitude_minutes=0.0)
    ephemeris = {
        "period_days": table.pop("_period_days"),
        "epoch_btjd": table.pop("_epoch_btjd"),
        "duration_days": table.pop("_duration_days"),
        "depth_ppm": table.pop("_depth_ppm"),
    }
    a_rs = stellar_density_a_rs(1.0, ephemeris["period_days"])
    template = transit_template_parameters(
        ephemeris, a_rs, impact_parameter=0.3, q1=0.3, q2=0.3
    )
    analysis = transit_timing_analysis(
        table["time"], table["flux"], table["flux_err"], ephemeris, template
    )
    assert analysis["n_transits_fit"] >= 5
    assert analysis["oc_rms_minutes"] < 2.0

    assert calculate_ttv_super_period(3.5, 5.0, j_resonance=2) == pytest.approx(8.75, rel=0.01)
    contexts = enumerate_companion_super_periods(5.0, [3.5, 10.0])
    inner_context = next(
        item
        for item in contexts
        if item["companion_orbital_relation"] == "inner-companion"
        and item["resonance_j"] == 2
    )
    assert inner_context["super_period_days"] == pytest.approx(8.75, rel=0.01)
    assert any(item["resonance_j"] == 5 for item in contexts)
    exact_context = next(
        item
        for item in contexts
        if item["companion_orbital_relation"] == "outer-companion"
        and item["resonance_j"] == 2
    )
    assert exact_context["super_period_days"] is None
    assert exact_context["super_period_status"] == "exact-resonance-unbounded"


def test_ttv_refits_a_weighted_linear_ephemeris():
    from exonym.ttv import fit_weighted_linear_ephemeris

    fit = fit_weighted_linear_ephemeris(
        np.array([10, 11, 12, 13]),
        np.array([100.0, 102.5, 105.0, 107.5]),
        np.full(4, 0.001),
    )

    assert fit["status"] == "fit"
    assert fit["reference_epoch"] == 12
    assert fit["period_days"] == pytest.approx(2.5)
    assert fit["period_uncertainty_days"] > 0
    assert fit["chi_square"] == pytest.approx(0.0)
    assert np.asarray(fit["covariance_matrix_days2"]).shape == (2, 2)
    assert fit["covariance_parameter_order"] == ["reference_epoch_btjd", "period_days"]


def test_ttv_injected_signal_has_nonzero_refit_residuals():
    from exonym.transit_fit import stellar_density_a_rs
    from exonym.ttv import (
        transit_template_parameters,
        transit_timing_analysis,
    )
    from tests.fixtures.synthetic_observations import _synthetic_timing_table

    table = _synthetic_timing_table(ttv_amplitude_minutes=20.0)
    ephemeris = {
        "period_days": table.pop("_period_days"),
        "epoch_btjd": table.pop("_epoch_btjd"),
        "duration_days": table.pop("_duration_days"),
        "depth_ppm": table.pop("_depth_ppm"),
    }
    a_rs = stellar_density_a_rs(1.0, ephemeris["period_days"])
    template = transit_template_parameters(
        ephemeris, a_rs, impact_parameter=0.3, q1=0.3, q2=0.3
    )
    analysis = transit_timing_analysis(
        table["time"], table["flux"], table["flux_err"], ephemeris, template
    )
    assert analysis["oc_rms_minutes"] > 5.0


def test_ttv_rejects_low_snr_epochs_and_persists_epoch_diagnostics(tmp_path, monkeypatch):
    from exonym.ttv import run_ttv_analysis
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "ttv-low-snr-test")
    (workspace.path / "outputs" / "mcmc_transit_fit.json").write_text(
        json.dumps(
            {
                "work_package": "MCMC_TRANSIT_FIT",
                "source": "candidate-data",
                "signal": None,
                "parameter_names": [
                    "rp_rs",
                    "log_rho_star",
                    "impact_parameter",
                    "baseline",
                    "log_jitter",
                    "q1",
                    "q2",
                ],
                "ephemeris": {
                    "period_days": 1.0,
                    "epoch_btjd": 1.0,
                    "source": "candidate-config",
                },
                "posterior": {
                    "impact_parameter": {"median": 0.3},
                    "q1": {"median": 0.3},
                    "q2": {"median": 0.3},
                },
            }
        ),
        encoding="utf-8",
    )
    period_days = 1.0
    epoch_btjd = 1.0
    time = np.arange(0.6, 5.4, 0.002)
    flux = np.ones_like(time)
    for transit_time in (2.0, 4.0):
        flux -= 0.01 * np.exp(-0.5 * ((time - transit_time) / 0.03) ** 2)
    table = {
        "time": time,
        "flux": flux,
        "flux_err": np.full_like(time, 0.001),
        "sector": np.ones(time.size, dtype=int),
    }
    ephemeris = {
        "period_days": period_days,
        "epoch_btjd": epoch_btjd,
        "duration_days": 0.12,
        "depth_ppm": 10000.0,
        "source": "candidate-config",
        "field_sources": {
            "period_days": "candidate-config",
            "epoch_btjd": "candidate-config",
            "duration_days": "candidate-config",
            "depth_ppm": "candidate-config",
        },
    }
    stellar = {"mass_solar": 1.0, "radius_solar": 1.0, "source": "candidate-data"}

    def fake_template_flux(_template, sample_time, t0_value):
        return 1.0 - 0.01 * np.exp(-0.5 * ((sample_time - t0_value) / 0.03) ** 2)

    monkeypatch.setattr("exonym.ttv.load_light_curve_table", lambda *_args, **_kwargs: table)
    monkeypatch.setattr("exonym.ttv.load_transit_ephemeris", lambda *_args, **_kwargs: ephemeris)
    monkeypatch.setattr("exonym.ttv.load_stellar_parameters", lambda *_args, **_kwargs: stellar)
    monkeypatch.setattr("exonym.ttv._template_flux", fake_template_flux)
    monkeypatch.setattr("exonym.ttv.plot_timing_diagram", lambda *_args, **_kwargs: None)

    output = run_ttv_analysis(workspace)
    payload = json.loads(output.read_text(encoding="utf-8"))
    timing = payload["timing"]
    rejected_by_epoch = {record["epoch"]: record for record in timing["rejected_epochs"]}

    # Epoch indices are zero-based relative to the 1.0 BTJD reference, so
    # injected transits at 2.0 and 4.0 are epochs 1 and 3. The non-injected
    # windows must be retained as rejected epochs.
    for rejected_epoch in (0, 2, 4):
        record = rejected_by_epoch[rejected_epoch]
        assert record["rejection_reason"] in {
            "non-positive-local-depth",
            "low-local-depth-snr",
        }
        assert record["local_depth_snr"] < timing["epoch_acceptance"]["minimum_local_depth_snr"]
        assert record["sigma_t0_days"] is None
    assert timing["n_transits_fit"] >= 2
    assert timing["n_rejected_epochs"] == len(timing["rejected_epochs"])
    accepted = [record for record in timing["per_epoch"] if not record["excluded_no_detection"]]
    assert all(record["local_duration_days"] is not None for record in accepted)
    assert timing["epoch_acceptance"]["local_duration_method"] == (
        "fixed candidate-derived transit-fit template duration"
    )
    assert "uncertainty_clipped_epochs" in timing
    assert "search_boundary_epochs" in timing
    json.dumps(payload, allow_nan=False)


def test_ttv_rejects_positive_depth_when_timing_curvature_is_non_positive(monkeypatch):
    from exonym.ttv import fit_transit_epoch

    time = np.linspace(0.7, 1.3, 301)
    transit_flux = 1.0 - 0.01 * np.exp(-0.5 * ((time - 1.0) / 0.03) ** 2)

    def fixed_template(_template, sample_time, _t0_value):
        return 1.0 - 0.01 * np.exp(-0.5 * ((sample_time - 1.0) / 0.03) ** 2)

    monkeypatch.setattr("exonym.ttv._template_flux", fixed_template)
    fit = fit_transit_epoch(
        time,
        transit_flux,
        np.full_like(time, 1e-4),
        {"unused": True},
        1.0,
    )

    assert fit["rejection_reason"] == "non-positive-timing-curvature"
    assert fit["excluded_no_detection"] is True
    assert fit["sigma_t0"] is None
    assert fit["depth_snr"] > 3.0
    assert fit["at_search_boundary"] is True


def test_ttv_retains_clipped_uncertainty_and_search_boundary_records(monkeypatch):
    from exonym.ttv import transit_template_parameters, transit_timing_analysis

    def fake_epoch_fit(_time, _flux, _errors, _template, t0_expected, **_kwargs):
        return {
            "t0_fit": t0_expected,
            "sigma_t0": 0.0005,
            "sigma_t0_raw": 0.0001,
            "depth_ppm": 1000.0,
            "depth_uncertainty_ppm": 100.0,
            "depth_snr": 10.0,
            "excluded_no_detection": False,
            "rejection_reason": None,
            "at_search_boundary": True,
            "sigma_t0_clipped": True,
        }

    monkeypatch.setattr("exonym.ttv.fit_transit_epoch", fake_epoch_fit)
    ephemeris = {
        "period_days": 1.0,
        "epoch_btjd": 0.0,
        "duration_days": 0.1,
        "depth_ppm": 1000.0,
    }
    template = transit_template_parameters(
        ephemeris, a_rs=10.0, impact_parameter=0.3, q1=0.3, q2=0.3
    )
    analysis = transit_timing_analysis(
        np.array([0.0, 2.0]),
        np.ones(2),
        np.full(2, 1e-4),
        ephemeris,
        template,
    )

    assert len(analysis["uncertainty_clipped_epochs"]) == 3
    assert len(analysis["search_boundary_epochs"]) == 3
    assert all(record["sigma_t0_clipped"] for record in analysis["per_epoch"])
    assert all(record["at_search_boundary"] for record in analysis["per_epoch"])


# ---------------------------------------------------------------------------
# Scientific analysis modules: stellar activity
# ---------------------------------------------------------------------------


def test_activity_recovers_rotation_period():
    from exonym.activity import (
        gls_periodogram,
        reconcile_harmonic_segment_periods,
        sampling_window_diagnostics,
        segment_harmonic_persistence,
        sinusoid_amplitude_ppm,
        sinusoid_amplitude_posterior,
        weighted_period_summary,
        weighted_percentile_summary,
    )
    from tests.fixtures.synthetic_observations import _synthetic_rotation_table

    table = _synthetic_rotation_table()
    periods, powers, fap = gls_periodogram(table["time"], table["flux"])
    best_period = float(periods[int(np.argmax(powers))])
    assert best_period == pytest.approx(5.0, abs=0.2)
    assert fap < 0.01

    amplitude = sinusoid_amplitude_ppm(table["time"], table["flux"], best_period)
    assert amplitude == pytest.approx(400.0, abs=100.0)

    summary = weighted_period_summary([5.0, 5.1], [1.0, 1.0])
    assert summary["weighted_mean_period_days"] == pytest.approx(5.05)
    period_posterior = weighted_percentile_summary([5.0, 5.1], [1.0, 1.0])
    assert period_posterior["p16"] <= period_posterior["median"] <= period_posterior["p84"]
    amplitude_posterior = sinusoid_amplitude_posterior(
        table["time"], table["flux"], table["flux_err"], best_period
    )
    assert amplitude_posterior["p16"] <= amplitude_posterior["median"] <= amplitude_posterior["p84"]
    assert np.asarray(amplitude_posterior["covariance_cos_sin_baseline"]).shape == (3, 3)

    sampling = sampling_window_diagnostics(
        table["time"], 1.0 / periods, 1.0 / best_period
    )
    assert sampling["method"] == "normalized-spectral-window-v1"
    assert sampling["frequency_resolution_days_inverse"] > 0.0
    assert sampling["top_window_peaks"]
    assert all(np.isfinite(peak["window_power"]) for peak in sampling["top_window_peaks"])

    persistence = segment_harmonic_persistence(
        [
            {"sector": 1, "best_period_days": 5.0, "baseline_days": 25.0},
            {"sector": 2, "best_period_days": 2.5, "baseline_days": 25.0},
        ]
    )
    assert persistence["status"] == "descriptive-harmonic-consistency"
    assert persistence["consistent_segment_count"] == 2
    assert persistence["segments"][1]["nearest_harmonic_frequency_factor"] == 2.0
    reconciliation = reconcile_harmonic_segment_periods(
        [
            {"sector": 1, "best_period_days": 5.0, "max_power": 1.0},
            {"sector": 2, "best_period_days": 2.5, "max_power": 1.0},
        ],
        persistence,
    )
    assert reconciliation["status"] == "harmonic-reconciled"
    assert reconciliation["periods_days"] == pytest.approx([5.0, 5.0])


def test_top_window_peaks_exclude_nonpositive_frequencies():
    from exonym.activity import _top_window_peaks

    frequency = np.array([-0.2, 0.0, 0.1, 0.25, 0.4])
    power = np.array([50.0, 40.0, 10.0, 30.0, 5.0])

    peaks = _top_window_peaks(frequency, power, frequency_resolution_days_inverse=0.05)

    assert [peak["frequency_days_inverse"] for peak in peaks] == [0.25]
    assert all(peak["frequency_days_inverse"] > 0.0 for peak in peaks)

    monotonic_power = np.array([90.0, 80.0, 7.0, 8.0, 9.0])
    fallback_peaks = _top_window_peaks(
        frequency, monotonic_power, frequency_resolution_days_inverse=0.05
    )

    assert [peak["frequency_days_inverse"] for peak in fallback_peaks] == [0.4]


def test_activity_does_not_mask_with_a_synthetic_ephemeris(tmp_path, monkeypatch):
    import exonym.activity as activity
    from exonym.workspace import create_candidate

    # Arrange
    workspace = create_candidate(tmp_path, "activity-mask-provenance-test")
    time = np.linspace(0.0, 27.0, 800)
    table = {
        "time": time,
        "flux": 1.0 + 300e-6 * np.sin(2.0 * np.pi * time / 5.0),
        "flux_err": np.full(time.size, 100e-6),
        "sector": np.full(time.size, 701),
    }
    synthetic_ephemeris = {
        "period_days": 3.5,
        "epoch_btjd": 2.0,
        "duration_days": 0.12,
        "source": "synthetic-demo",
        "field_sources": {
            "period_days": "synthetic-demo",
            "epoch_btjd": "synthetic-demo",
            "duration_days": "synthetic-demo",
        },
    }
    monkeypatch.setattr(activity, "load_light_curve_table", lambda *_args, **_kwargs: table)
    monkeypatch.setattr(activity, "load_transit_ephemeris", lambda _workspace: synthetic_ephemeris)

    # Act
    output = activity.run_stellar_activity(workspace)
    payload = json.loads(output.read_text(encoding="utf-8"))

    # Assert
    assert payload["source"] == "candidate-data"
    assert payload["transit_mask_status"] == "not-applied-no-candidate-ephemeris"
    assert payload["validation_eligible"] is False
    assert "best_analytic_white_noise_false_alarm_probability" in payload
    assert "best_false_alarm_probability" not in payload
    assert payload["segments"][0]["sampling_window"]["top_window_peaks"]
    assert payload["harmonic_persistence"]["status"] == "unresolved-insufficient-segments"


def test_activity_reconciles_harmonic_segment_peaks_before_summary(tmp_path, monkeypatch):
    """The reported period must not average a fundamental with its harmonic."""
    import exonym.activity as activity
    from exonym.workspace import create_candidate

    workspace = create_candidate(tmp_path, "activity-harmonic-reconciliation")
    first = np.linspace(0.0, 20.0, 120)
    second = np.linspace(40.0, 60.0, 120)
    time = np.concatenate((first, second))
    table = {
        "time": time,
        "flux": np.ones(time.size),
        "flux_err": np.full(time.size, 100e-6),
        "sector": np.concatenate((np.full(first.size, 1), np.full(second.size, 2))),
    }
    synthetic_ephemeris = {
        "source": "synthetic-demo",
        "field_sources": {
            "period_days": "synthetic-demo",
            "epoch_btjd": "synthetic-demo",
            "duration_days": "synthetic-demo",
        },
    }
    periodograms = iter(
        [
            (np.asarray([5.0, 4.0]), np.asarray([2.0, 1.0]), 0.01),
            (np.asarray([2.5, 4.0]), np.asarray([2.0, 1.0]), 0.01),
        ]
    )
    monkeypatch.setattr(activity, "load_light_curve_table", lambda *_args, **_kwargs: table)
    monkeypatch.setattr(activity, "load_transit_ephemeris", lambda _workspace: synthetic_ephemeris)
    monkeypatch.setattr(activity, "gls_periodogram", lambda *_args: next(periodograms))
    monkeypatch.setattr(
        activity,
        "sampling_window_diagnostics",
        lambda *_args: {"top_window_peaks": [], "frequency_resolution_days_inverse": 0.05},
    )
    monkeypatch.setattr(activity, "sinusoid_amplitude_posterior", lambda *_args: {"median": 0.0})

    output = activity.run_stellar_activity(workspace)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["rotation_period_selection"] == "harmonic-reconciled"
    assert payload["rotation_period_days"] == pytest.approx(5.0)
    assert len(payload["rotation_period_reconciled_segments"]) == 2


# ---------------------------------------------------------------------------
# Scientific analysis modules: dilution
# ---------------------------------------------------------------------------


def test_dilution_contamination_factor_sums_neighbors():
    from exonym.dilution import gaia_contamination_factor, gaia_g_to_tess_mag

    rows = [
        {"separation_arcsec": 10.0, "flux_ratio": 0.02, "is_target": False},
        {"separation_arcsec": 100.0, "flux_ratio": 0.5, "is_target": False},
        {
            "separation_arcsec": 5.0,
            "flux_ratio": None,
            "is_target": False,
            "g_mag": 14.0,
            "bp_rp_color": 1.5,
        },
    ]
    result = gaia_contamination_factor(
        rows, search_radius_arcsec=60.0, target_g_mag=10.0, target_bp_rp_color=0.8
    )
    target_t_mag = gaia_g_to_tess_mag(10.0, 0.8)
    neighbor_t_mag = gaia_g_to_tess_mag(14.0, 1.5)
    assert target_t_mag is not None
    assert neighbor_t_mag is not None
    expected = 0.02 + 10.0 ** (-0.4 * (neighbor_t_mag - target_t_mag))
    assert result["contamination_factor"] == pytest.approx(expected, abs=1e-6)
    assert result["n_neighbors_included"] == 2


def test_dilution_marks_nonfinite_neighbor_measurements_unavailable():
    from exonym.dilution import gaia_contamination_factor

    # Arrange
    rows = [
        {"separation_arcsec": 10.0, "flux_ratio": float("nan"), "is_target": False},
        {"separation_arcsec": float("inf"), "flux_ratio": 0.5, "is_target": False},
    ]

    # Act
    result = gaia_contamination_factor(rows)

    # Assert
    assert result["availability"] == "unavailable"
    assert result["contamination_factor"] is None
    assert result["n_neighbors_included"] == 0


def test_dilution_reads_validated_archival_neighbors(tmp_path):
    from exonym.dilution import _load_archival_gaia_neighbor_rows

    # Arrange
    workspace = type("Workspace", (), {"path": tmp_path, "candidate_id": "dilution-test"})()
    output = tmp_path / "outputs"
    output.mkdir()
    output.joinpath("archival_vetting_report.json").write_text(
        json.dumps(
            {
                "candidate_id": workspace.candidate_id,
                "gaia_astrometry": {
                    "validated": True,
                    "query_status": "ok",
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
    rows, target_g_mag, metadata, target_bp_rp_color = _load_archival_gaia_neighbor_rows(workspace)

    # Assert
    assert target_g_mag == pytest.approx(10.0)
    assert rows == [
        {
            "g_mag": 14.0,
            "bp_rp_color": None,
            "separation_arcsec": 8.0,
            "flux_ratio": None,
            "is_target": False,
        }
    ]
    assert metadata["availability"] == "available"
    assert metadata["target_selection"] == "reported-target-source-id"
    assert target_bp_rp_color is None


def test_dilution_rejects_unvalidated_archival_neighbors(tmp_path):
    from exonym.dilution import _load_archival_gaia_neighbor_rows

    # Arrange
    workspace = type("Workspace", (), {"path": tmp_path, "candidate_id": "dilution-test"})()
    output = tmp_path / "outputs"
    output.mkdir()
    output.joinpath("archival_vetting_report.json").write_text(
        json.dumps(
            {
                "candidate_id": workspace.candidate_id,
                "gaia_astrometry": {
                    "validated": False,
                    "query_status": "unvalidated",
                    "sources": [],
                },
            }
        ),
        encoding="utf-8",
    )

    # Act
    rows, target_g_mag, metadata, target_bp_rp_color = _load_archival_gaia_neighbor_rows(workspace)

    # Assert
    assert rows == []
    assert target_g_mag is None
    assert metadata["availability"] == "unavailable"
    assert target_bp_rp_color is None


def test_dilution_aperture_depth_decreases_with_size():
    from exonym.dilution import (
        _extract_cube_light_curves,
        aperture_depth_ppm,
    )
    from tests.fixtures.synthetic_observations import _synthetic_tpf_cube

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
