"""Unit coverage for the optional JAX/NumPyro transit-fit dispatcher."""

import numpy as np
import pytest
from pathlib import Path

from exonym.transit_fit import (
    GPU_NUTS_SAMPLES,
    GPU_NUTS_TARGET_ACCEPT_PROB,
    GPU_NUTS_WARMUP,
    _GpuBackendUnavailable,
    _load_emcee_checkpoint,
    _shutdown_worker_pool,
    _summarize_accelerated_samples,
    _validate_accelerated_transit_fit_data,
    _write_emcee_checkpoint,
    fit_transit_light_curve,
    run_mcmc_transit_fit,
)


def _fit_inputs():
    time_days = np.linspace(-0.12, 0.12, 24)
    return {
        "time_days": time_days,
        "flux": np.ones_like(time_days),
        "flux_err": np.full_like(time_days, 100e-6),
        "period_days": 3.0,
        "t0_days": 0.0,
        "rho_star_g_cm3": 1.4,
        "rho_star_sigma_g_cm3": 0.14,
    }


def test_cpu_device_bypasses_jax_and_selects_emcee(monkeypatch):
    captured = {}

    def fake_cpu(data, **kwargs):
        captured["data"] = data
        captured.update(kwargs)
        return {"backend": "emcee-cpu"}

    monkeypatch.setattr(
        "exonym.transit_fit._load_jax_gpu_stack",
        lambda: pytest.fail("CPU dispatch must not initialize JAX"),
    )
    monkeypatch.setattr("exonym.transit_fit._fit_emcee_cpu", fake_cpu)

    result = fit_transit_light_curve(**_fit_inputs(), device="cpu")

    assert result == {"backend": "emcee-cpu"}
    assert captured["fallback_reason"] == "CPU requested explicitly"
    assert captured["burn_in"] == 1000
    assert captured["production"] == 2500
    assert captured["n_walkers"] == 50


def test_auto_device_uses_numpyro_when_a_gpu_stack_is_available(monkeypatch):
    stack = {"device": "gpu-test"}
    captured = {}

    def fake_gpu(data, **kwargs):
        captured["data"] = data
        captured.update(kwargs)
        return {"backend": "jax-gpu"}

    monkeypatch.setattr("exonym.transit_fit._load_jax_gpu_stack", lambda: stack)
    monkeypatch.setattr("exonym.transit_fit._fit_numpyro_gpu", fake_gpu)
    monkeypatch.setattr(
        "exonym.transit_fit._fit_emcee_cpu",
        lambda *_args, **_kwargs: pytest.fail("GPU stack should select NumPyro"),
    )

    result = fit_transit_light_curve(**_fit_inputs(), device="auto")

    assert result == {"backend": "jax-gpu"}
    assert captured["stack"] is stack
    assert captured["num_warmup"] == GPU_NUTS_WARMUP
    assert captured["num_samples"] == GPU_NUTS_SAMPLES
    assert captured["target_accept_prob"] == GPU_NUTS_TARGET_ACCEPT_PROB


def test_unavailable_gpu_falls_back_to_emcee_with_reason(monkeypatch):
    captured = {}

    def unavailable():
        raise _GpuBackendUnavailable("JAX GPU backend unavailable: no device")

    def fake_cpu(_data, **kwargs):
        captured.update(kwargs)
        return {"backend": "emcee-cpu"}

    monkeypatch.setattr("exonym.transit_fit._load_jax_gpu_stack", unavailable)
    monkeypatch.setattr("exonym.transit_fit._fit_emcee_cpu", fake_cpu)

    result = fit_transit_light_curve(**_fit_inputs(), device="gpu")

    assert result == {"backend": "emcee-cpu"}
    assert captured["fallback_reason"] == "JAX GPU backend unavailable: no device"


def test_cpu_backend_executes_a_small_batman_emcee_run():
    result = fit_transit_light_curve(
        **_fit_inputs(),
        device="cpu",
        n_walkers=16,
        burn_in=2,
        production=3,
        seed=11,
    )

    assert result["backend"] == "emcee-cpu"
    assert result["sampler_metadata"]["flat_samples"] == 48
    assert result["rp_rstar"]["p16"] <= result["rp_rstar"]["median"]


def test_candidate_fit_auto_dispatches_to_gpu_nuts_when_available(monkeypatch):
    captured = {}

    def fake_gpu(workspace, **kwargs):
        captured["workspace"] = workspace
        captured.update(kwargs)
        return Path("fit.json")

    monkeypatch.setattr("exonym.transit_fit._load_jax_gpu_stack", lambda: {"device": "gpu-test"})
    monkeypatch.setattr("exonym.transit_fit._run_numpyro_candidate_transit_fit", fake_gpu)

    workspace = object()
    assert run_mcmc_transit_fit(workspace) == Path("fit.json")
    assert captured["workspace"] is workspace
    assert captured["num_warmup"] == GPU_NUTS_WARMUP
    assert captured["num_samples"] == GPU_NUTS_SAMPLES
    assert captured["target_accept_prob"] == GPU_NUTS_TARGET_ACCEPT_PROB


def test_emcee_checkpoint_round_trip_uses_only_non_pickle_fields(tmp_path):
    path = tmp_path / "checkpoint.npz"
    chain = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    random_state = np.random.RandomState(9).get_state()

    _write_emcee_checkpoint(path, chain, iteration=2, burn_in=1, random_state=random_state)

    with np.load(path, allow_pickle=False) as archive:
        assert "random_state" not in archive.files
        assert archive["checkpoint_schema_version"].item() == 1
    restored = _load_emcee_checkpoint(path, expected_walkers=4, expected_dimensions=3, expected_burn_in=1)

    assert np.array_equal(restored["chain"], chain)
    assert restored["iteration"] == 2
    assert restored["burn_in"] == 1
    assert restored["random_state"][0] == "MT19937"
    assert np.array_equal(restored["random_state"][1], random_state[1])


def test_emcee_checkpoint_rejects_legacy_pickle_payload(tmp_path):
    path = tmp_path / "legacy-checkpoint.npz"
    np.savez(
        path,
        chain=np.ones((2, 4, 3)),
        iteration=2,
        burn_in=1,
        random_state=np.asarray(np.random.RandomState(9).get_state(), dtype=object),
    )

    with pytest.raises(RuntimeError, match="invalid or unsafe"):
        _load_emcee_checkpoint(path, expected_walkers=4, expected_dimensions=3, expected_burn_in=1)


def test_emcee_checkpoint_rejects_corrupt_and_unrecognized_archives(tmp_path):
    path = tmp_path / "checkpoint.npz"
    chain = np.ones((2, 4, 3))
    random_state = np.random.RandomState(9).get_state()
    _write_emcee_checkpoint(path, chain, iteration=2, burn_in=1, random_state=random_state)

    with np.load(path, allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    fields["unexpected"] = np.asarray(1, dtype=np.int64)
    np.savez_compressed(path, **fields)

    with pytest.raises(RuntimeError, match="invalid or unsafe"):
        _load_emcee_checkpoint(path, expected_walkers=4, expected_dimensions=3, expected_burn_in=1)

    path.write_bytes(b"not a valid npz archive")
    with pytest.raises(RuntimeError, match="invalid or unsafe"):
        _load_emcee_checkpoint(path, expected_walkers=4, expected_dimensions=3, expected_burn_in=1)


def test_worker_pool_terminates_after_failure_and_closes_after_success():
    class FakePool:
        def __init__(self):
            self.calls = []

        def close(self):
            self.calls.append("close")

        def terminate(self):
            self.calls.append("terminate")

        def join(self):
            self.calls.append("join")

    failed_pool = FakePool()
    _shutdown_worker_pool(failed_pool, failed=True)
    assert failed_pool.calls == ["terminate", "join"]

    successful_pool = FakePool()
    _shutdown_worker_pool(successful_pool, failed=False)
    assert successful_pool.calls == ["close", "join"]


def test_normalized_output_has_every_required_posterior_summary():
    inputs = _fit_inputs()
    data = _validate_accelerated_transit_fit_data(
        **inputs,
        period_sigma_days=None,
        t0_sigma_days=None,
        exposure_seconds=120.0,
        eccentric=False,
    )
    samples = {
        "rp_rstar": np.array([0.08, 0.09, 0.10, 0.11]),
        "log_rho_star": np.log(np.array([1.3, 1.4, 1.5, 1.6])),
        "rho_star_g_cm3": np.array([1.3, 1.4, 1.5, 1.6]),
        "impact_parameter": np.array([0.2, 0.25, 0.3, 0.35]),
        "log_jitter": np.log(np.array([50e-6, 60e-6, 70e-6, 80e-6])),
        "q1": np.array([0.25, 0.36, 0.49, 0.64]),
        "q2": np.array([0.2, 0.3, 0.4, 0.5]),
    }

    result = _summarize_accelerated_samples(
        samples,
        data,
        backend="emcee-cpu",
        sampler_metadata={"sampler": "test"},
    )

    assert result["backend"] == "emcee-cpu"
    assert result["sampler_metadata"] == {"sampler": "test"}
    for name in (
        "rp_rstar",
        "a_rstar",
        "period",
        "t0",
        "inclination_deg",
        "q1",
        "q2",
        "u1",
        "u2",
        "jitter_ppm",
    ):
        assert set(result[name]) >= {"p16", "median", "p84"}
        assert result[name]["p16"] <= result[name]["median"] <= result[name]["p84"]
    assert result["period"]["median"] == pytest.approx(inputs["period_days"])
    assert result["t0"]["median"] == pytest.approx(inputs["t0_days"])
    assert result["u1"]["median"] == pytest.approx(0.46)
    assert result["u2"]["median"] == pytest.approx(0.19)
