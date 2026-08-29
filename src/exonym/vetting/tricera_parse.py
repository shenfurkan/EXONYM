"""Interface to candidate-local TRICERATOPS statistical-vetting reports.

The parser validates a retained report, extracts finite model outputs, and
applies preregistered routing criteria to the recorded false-positive terms.
It preserves runtime and observed-input provenance so an unavailable or
fallback execution cannot be mistaken for a completed Monte Carlo result.

Scientific boundary:
    A finite false-positive probability remains evidence from its recorded
    assumptions. Claim creation is disabled until provenance-bound observed
    photometry and calibrated scene constraints are integrated.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
import ssl
import tempfile
import threading
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from ..inputs import (
    EPHEMERIS_CONFIG_NAMES,
    _parse_finite_float,
    _reject_duplicate_json_keys,
    _reject_nonfinite_json_constant,
)
from ..workspace import validate_signal_suffix

# SCIENTIFIC_BOUNDARY: This threshold is retained for transparent routing; it
# does not enable a claim while the calibrated scene-model prerequisite is open.
FPP_THRESHOLD = 0.01
# SCIENTIFIC_BOUNDARY: A report-aware routing pass requires both retained
# TRICERATOPS false-positive terms; an FPP-only scalar is legacy utility input.
NFPP_THRESHOLD = 0.01
FPP_CLAIM_BLOCK_REASON = (
    "FPP claim creation is disabled until TRICERATOPS receives provenance-bound "
    "observed photometry and scene constraints."
)
DEFAULT_TRICERATOPS_SEED = 1729
_SERVER_AUTH_OID = "1.3.6.1.5.5.7.3.1"
# TRICERATOPS and its legacy dependencies require process-global CWD, NumPy RNG,
# and TLS environment changes. Serialize the complete affected region.
_TRICERATOPS_PROCESS_STATE_LOCK = threading.RLock()


def _ensure_legacy_numpy_scalars(numpy_module: Any) -> None:
    """Restore removed NumPy scalar aliases needed by legacy optional engines.

    The aliases are installed only when NumPy no longer exposes them. This is a
    compatibility bridge for third-party imports, not a replacement for their
    own NumPy-2-compatible releases.
    """
    for type_name, builtin_type in (("int", int), ("float", float), ("bool", bool)):
        if type_name not in numpy_module.__dict__:
            setattr(numpy_module, type_name, builtin_type)



def _observed_sectors(workspace: Any) -> list:
    """Return the sorted list of TESS sectors observed for this workspace.

    Sector numbers are read from the workspace data (``data/raw`` product
    filenames first, ``data/external/tess_holdings.json`` as fallback) so the
    library stays target-neutral.
    """
    sectors: set = set()
    raw = workspace.path / "data" / "raw"
    if raw.is_dir():
        for path in sorted(raw.rglob("*")):
            if path.is_file() and path.name.startswith("s") and path.suffix.lower() in (".fits", ".fz"):
                stem = path.name[1:5]
                if stem.isdigit():
                    sectors.add(int(stem))
    if not sectors:
        holdings = workspace.path / "data" / "external" / "tess_holdings.json"
        try:
            payload = json.loads(holdings.read_text(encoding="utf-8"))
            for pipeline in payload.get("pipelines", {}).values():
                for entry in pipeline:
                    sector = entry.get("sector")
                    if isinstance(sector, int):
                        sectors.add(sector)
        except Exception:
            pass
    return sorted(sectors)


def load_fpp_report(path: Path) -> Dict[str, Any]:
    """Load one strict, finite TRICERATOPS JSON object."""
    try:
        data = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_float=_parse_finite_float,
            parse_constant=_reject_nonfinite_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("TRICERATOPS report must be strict finite JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("TRICERATOPS report must be a JSON object")
    return data


def _probability_value(report: Dict[str, Any], keys: Tuple[str, ...], label: str) -> float:
    for key in keys:
        value = report.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("{0} must be a JSON number".format(label))
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("{0} must be a finite probability in [0, 1]".format(label))
        return probability
    raise ValueError("no {0} value found in report".format(label))


def extract_fpp(report: Dict[str, Any]) -> float:
    """Return the FPP value from a report, probing common key layouts."""
    return _probability_value(
        report,
        ("fpp", "FPP", "fpp_value", "fpp_specific", "FPP_specific", "fpp_specific_value"),
        "FPP",
    )


def extract_nfpp(report: Dict[str, Any]) -> float:
    """Return the nearby-false-positive probability from a retained report.

    The report-aware gate requires this value alongside FPP. A missing,
    non-finite, or out-of-range value is not evidence that the nearby-source
    contribution is small and therefore cannot pass the joint screen.
    """
    return _probability_value(report, ("nfpp", "NFPP", "nfpp_value"), "NFPP")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for a candidate-local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_ca_certificates() -> list:
    """Return Windows certificates trusted for TLS server authentication.

    Python's OpenSSL trust store does not consistently include the Windows
    root and intermediate stores.  The optional TRICERATOPS backend performs
    its own HTTP requests, so a temporary PEM bundle is the narrowest way to
    make that platform trust material available without weakening validation.
    """
    if os.name != "nt":
        return []
    certificates: list = []
    try:
        for store_name in ("ROOT", "CA"):
            for certificate, encoding, trust in ssl.enum_certificates(store_name):
                trusted_for_server_auth = trust is True or (
                    isinstance(trust, (set, tuple, list)) and _SERVER_AUTH_OID in trust
                )
                if encoding != "x509_asn" or not trusted_for_server_auth:
                    continue
                certificates.append(ssl.DER_cert_to_PEM_cert(certificate))
    except (AttributeError, OSError, ssl.SSLError):
        return []
    return certificates


@contextmanager
def _triceratops_tls_environment():
    """Provide a verified Windows CA bundle while TRICERATOPS performs I/O.

    Requests honors ``REQUESTS_CA_BUNDLE`` and standard-library clients honor
    ``SSL_CERT_FILE``.  Both are restored before returning.  The lock prevents
    concurrent TRICERATOPS runs from replacing each other's temporary bundle.
    """
    certificates = _windows_ca_certificates()
    if not certificates:
        yield "default"
        return
    previous = {
        name: os.environ.get(name) for name in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")
    }
    try:
        from requests.certs import where as requests_ca_bundle

        bundle_texts = [Path(requests_ca_bundle()).read_text(encoding="ascii")]
        for configured_bundle in set(value for value in previous.values() if value):
            bundle_texts.append(Path(configured_bundle).read_text(encoding="ascii"))
    except (ImportError, OSError, UnicodeError):
        yield "default"
        return

    bundle_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", suffix=".pem", delete=False
        ) as bundle:
            for bundle_text in bundle_texts:
                bundle.write(bundle_text)
                if not bundle_text.endswith("\n"):
                    bundle.write("\n")
            bundle.write("\n".join(certificates))
            bundle.write("\n")
            bundle_path = Path(bundle.name)
        with _TRICERATOPS_PROCESS_STATE_LOCK:
            os.environ["REQUESTS_CA_BUNDLE"] = str(bundle_path)
            os.environ["SSL_CERT_FILE"] = str(bundle_path)
            try:
                yield "windows-root-intermediate-and-operator-store"
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
    finally:
        if bundle_path is not None:
            try:
                bundle_path.unlink()
            except OSError:
                pass


def _is_tls_verification_error(error: BaseException) -> bool:
    """Return whether an optional-engine failure is TLS chain verification."""
    current: Optional[BaseException] = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError):
            return True
        message = str(current).lower()
        if "certificate_verify_failed" in message or "certificate verify failed" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _notify_progress(
    progress_callback: Optional[Callable[[str, Optional[int], Optional[int]], None]],
    step: str,
    done: Optional[int] = None,
    total: Optional[int] = None,
) -> None:
    """Report a truthful backend milestone without affecting scientific work."""
    if progress_callback is None:
        return
    try:
        progress_callback(step, done, total)
    except Exception:
        # Presentation callbacks must never change a vetting outcome.
        pass


def _parallel_calc_probs_dispatcher(
    target: Any,
    observed_input: Dict[str, Any],
    period: float,
    n_draws: int,
    n_jobs: int,
    progress_callback: Optional[Callable[[str, Optional[int], Optional[int]], None]],
) -> Dict[str, Any]:
    """Run one non-partitioned TRICERATOPS calculation with bounded native threads.

    TRICERATOPS has no public chunk/merge API: its scenario evidence uses a
    global RNG stream and one final normalization.  ``n_jobs`` therefore maps
    only to its supported in-process PyTransit vectorized mode. Its likelihood
    implementation uses Numba; this configures its native thread limit when
    that runtime honors the setting. It never creates worker processes or
    partitions Monte Carlo draws.
    """
    execution: Dict[str, Any] = {
        "requested_n_jobs": n_jobs,
        "effective_n_jobs": 1,
        "calc_probs_parallel": False,
        "parallel_backend": "serial",
        "fallback_reason": None,
    }
    previous_threads: Optional[int] = None
    numba_module: Any = None
    if n_jobs > 1:
        try:
            import numba

            numba_module = numba
            previous_threads = int(numba.get_num_threads())
            maximum_threads = int(numba.config.NUMBA_NUM_THREADS)
            effective_threads = min(n_jobs, maximum_threads)
            numba.set_num_threads(effective_threads)
            execution.update(
                {
                    "effective_n_jobs": effective_threads,
                    "calc_probs_parallel": True,
                    "parallel_backend": "triceratops-pytransit-vectorized",
                    "numba_version": str(numba.__version__),
                }
            )
            if effective_threads != n_jobs:
                execution["fallback_reason"] = "requested threads exceeded the Numba runtime limit"
        except Exception as exc:
            execution["fallback_reason"] = "native thread control unavailable: {0}: {1}".format(
                type(exc).__name__, exc
            )
            numba_module = None
            previous_threads = None

    _notify_progress(progress_callback, "TRICERATOPS Monte Carlo Vetting", 0, 1)
    try:
        target.calc_probs(
            time=observed_input["time_days"],
            flux_0=observed_input["flux"],
            flux_err_0=observed_input["flux_err"],
            P_orb=period,
            N=n_draws,
            parallel=bool(execution["calc_probs_parallel"]),
            # TRICERATOPS only prints scenario-start messages. It does not
            # expose draw counters, so verbose output cannot provide progress.
            verbose=0,
            exptime=observed_input["exposure_days"],
            nsamples=5,
        )
    finally:
        if numba_module is not None and previous_threads is not None:
            numba_module.set_num_threads(previous_threads)
    _notify_progress(progress_callback, "TRICERATOPS Monte Carlo Vetting", 1, 1)
    return execution


def _artifact_from_path(workspace: Any, path: Path) -> Dict[str, str]:
    """Hash one regular candidate-local artifact for an execution snapshot."""
    candidate_root = workspace.path.resolve()
    try:
        resolved = path.resolve()
        relative = resolved.relative_to(candidate_root)
    except (OSError, ValueError) as exc:
        raise ValueError("TRICERATOPS artifact must remain inside the candidate workspace") from exc
    if path.is_symlink() or not resolved.is_file():
        raise ValueError("TRICERATOPS artifact must be an available regular file")
    return {"path": relative.as_posix(), "sha256": _sha256(resolved)}


def _ephemeris_artifacts(workspace: Any, signal: Optional[str], source: object) -> list:
    """Bind every candidate config that could have supplied the ephemeris."""
    source_text = str(source)
    paths = []
    if "signal" in source_text and signal is not None:
        paths.append(workspace.path / "config" / "signals" / "transit_config{0}.json".format(signal))
    elif "candidate-config" in source_text:
        paths.extend(workspace.path / "config" / name for name in EPHEMERIS_CONFIG_NAMES)
    return [_artifact_from_path(workspace, path) for path in paths if path.is_file()]


def _snapshot_artifacts(workspace: Any, artifacts: Iterable[object]) -> list:
    """Normalize and rehash a complete, candidate-local execution input set."""
    snapshot = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError("TRICERATOPS input provenance contains a malformed artifact")
        record = _artifact_from_path(workspace, workspace.path / artifact["path"])
        expected = artifact.get("sha256")
        if expected is not None and expected != record["sha256"]:
            raise ValueError("TRICERATOPS input changed before execution: {0}".format(record["path"]))
        snapshot[record["path"]] = record
    return [snapshot[path] for path in sorted(snapshot)]


def _verify_snapshot(workspace: Any, artifacts: Iterable[object]) -> None:
    """Fail closed when an input changes during an optional-engine run."""
    _snapshot_artifacts(workspace, artifacts)


def _prepare_observed_transit_input(workspace: Any, signal: Optional[str]) -> Dict[str, Any]:
    """Return provenance-bound, phase-folded observed photometry for TRICERATOPS.

    The returned ``time_days`` values are measured from the nearest declared
    transit midpoint. Flux values are inverse-variance binned across phase,
    using a maximum of one hundred bins.  The central transit receives five
    equal-width bins, so its resolution is at most ``duration_days / 5`` even
    for a low-duty-cycle signal. The backend accepts only one flux uncertainty,
    so this function records and passes the mean uncertainty of the observed
    phase bins.
    """
    import numpy as np

    from ..inputs import BTJD_TIME_SYSTEM, load_light_curve_table, load_transit_ephemeris

    table = load_light_curve_table(workspace, max_points=None, require_raw_provenance=True)
    if table is None:
        raise ValueError("TRICERATOPS requires a readable candidate light curve")

    ephemeris = load_transit_ephemeris(workspace, signal=signal)
    field_sources = ephemeris.get("field_sources", {})
    required_fields = ("period_days", "epoch_btjd", "duration_days")
    if not isinstance(field_sources, dict) or any(
        field_sources.get(field) in (None, "synthetic-demo") for field in required_fields
    ):
        raise ValueError(
            "TRICERATOPS requires candidate-derived period, epoch, and duration values"
        )
    if ephemeris.get("time_system") != BTJD_TIME_SYSTEM:
        raise ValueError("TRICERATOPS requires a BTJD_TDB ephemeris")

    try:
        period_days = float(ephemeris["period_days"])
        epoch_btjd = float(ephemeris["epoch_btjd"])
        duration_days = float(ephemeris["duration_days"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("TRICERATOPS requires finite period, epoch, and duration values") from exc
    if (
        not math.isfinite(period_days)
        or not math.isfinite(epoch_btjd)
        or not math.isfinite(duration_days)
        or period_days <= 0.0
        or duration_days <= 0.0
        or duration_days >= 0.5 * period_days
    ):
        raise ValueError("TRICERATOPS requires a positive duration shorter than half the orbital period")

    flux_err_sources = table.get("flux_err_sources")
    if flux_err_sources != ["reported"]:
        raise ValueError("TRICERATOPS requires reported per-cadence flux uncertainties")

    time_btjd = np.asarray(table.get("time"), dtype=float)
    flux = np.asarray(table.get("flux"), dtype=float)
    flux_err = np.asarray(table.get("flux_err"), dtype=float)
    sectors = np.asarray(table.get("sector"), dtype=int)
    valid = np.isfinite(time_btjd) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0.0)
    time_btjd = time_btjd[valid]
    flux = flux[valid]
    flux_err = flux_err[valid]
    sectors = sectors[valid]
    if time_btjd.size < 50:
        raise ValueError("TRICERATOPS requires at least fifty finite observed cadences")

    time_days = np.remainder(time_btjd - epoch_btjd + 0.5 * period_days, period_days) - 0.5 * period_days
    max_bin_count = min(100, time_days.size)
    transit_bin_count = 5
    local_baseline_bin_count = 3
    broad_baseline_bin_count = (
        max_bin_count - transit_bin_count - 2 * local_baseline_bin_count
    ) // 2
    if broad_baseline_bin_count < 3:
        raise ValueError("TRICERATOPS requires sufficient phase coverage for transit and baseline bins")
    local_baseline_limit = min(3.0 * duration_days, 0.25 * period_days)
    # The central five bins have width duration_days / 5. Adjacent local
    # baseline bins measure depth without folding phase-curve modulation or a
    # secondary eclipse from the rest of the orbit into the transit depth.
    edges = np.concatenate(
        (
            np.linspace(-0.5 * period_days, -local_baseline_limit, broad_baseline_bin_count + 1),
            np.linspace(-local_baseline_limit, -0.5 * duration_days, local_baseline_bin_count + 1)[1:],
            np.linspace(-0.5 * duration_days, 0.5 * duration_days, transit_bin_count + 1)[1:],
            np.linspace(0.5 * duration_days, local_baseline_limit, local_baseline_bin_count + 1)[1:],
            np.linspace(local_baseline_limit, 0.5 * period_days, broad_baseline_bin_count + 1)[1:],
        )
    )
    binned_time: list = []
    binned_flux: list = []
    binned_err: list = []
    for index in range(edges.size - 1):
        if index + 1 == edges.size - 1:
            in_bin = (time_days >= edges[index]) & (time_days <= edges[index + 1])
        else:
            in_bin = (time_days >= edges[index]) & (time_days < edges[index + 1])
        if not np.any(in_bin):
            continue
        weights = np.reciprocal(np.square(flux_err[in_bin]))
        weight_sum = float(np.sum(weights))
        if not math.isfinite(weight_sum) or weight_sum <= 0.0:
            continue
        binned_time.append(float(np.sum(weights * time_days[in_bin]) / weight_sum))
        binned_flux.append(float(np.sum(weights * flux[in_bin]) / weight_sum))
        binned_err.append(float(math.sqrt(1.0 / weight_sum)))
    if len(binned_time) < 10:
        raise ValueError("TRICERATOPS phase folding produced fewer than ten populated bins")

    phase_days = np.asarray(binned_time, dtype=float)
    phase_flux = np.asarray(binned_flux, dtype=float)
    phase_err = np.asarray(binned_err, dtype=float)
    in_transit = np.abs(phase_days) <= 0.5 * duration_days
    out_of_transit = (np.abs(phase_days) >= duration_days) & (
        np.abs(phase_days) <= local_baseline_limit
    )
    if int(np.count_nonzero(in_transit)) < transit_bin_count or int(np.count_nonzero(out_of_transit)) < 3:
        raise ValueError("TRICERATOPS requires populated in-transit and out-of-transit phase bins")
    depth_ppm = float((np.median(phase_flux[out_of_transit]) - np.median(phase_flux[in_transit])) * 1e6)
    if not math.isfinite(depth_ppm) or depth_ppm <= 0.0:
        raise ValueError("TRICERATOPS could not measure a positive observed transit depth")

    exposure_steps = np.diff(np.sort(np.unique(time_btjd)))
    exposure_steps = exposure_steps[np.isfinite(exposure_steps) & (exposure_steps > 0.0)]
    if exposure_steps.size == 0:
        raise ValueError("TRICERATOPS could not determine the observed cadence")
    exposure_days = float(np.median(exposure_steps))
    flux_err_scalar = float(np.mean(phase_err))
    # NUMERICAL_GUARD: adopt the inverse-variance effective error so mixed-cadence
    # or multi-sector phase bins weight their uncertainties correctly instead of
    # over-trusting a simple arithmetic mean of reported standard errors.
    if np.all(phase_err > 0.0) and np.isfinite(phase_err).all():
        flux_err_scalar = float(np.sqrt(phase_err.size / float(np.sum(1.0 / (phase_err ** 2)))))
    if not math.isfinite(exposure_days) or exposure_days <= 0.0 or not math.isfinite(flux_err_scalar):
        raise ValueError("TRICERATOPS observed photometry has invalid cadence or uncertainty")

    candidate_root = workspace.path.resolve()
    input_files = []
    for item in table.get("input_files", []):
        path = Path(item).resolve()
        try:
            relative = path.relative_to(candidate_root)
        except ValueError as exc:
            raise ValueError("TRICERATOPS input photometry must belong to the candidate workspace") from exc
        if not path.is_file():
            raise FileNotFoundError("TRICERATOPS input photometry is unavailable: {0}".format(relative))
        input_files.append({"path": relative.as_posix(), "sha256": _sha256(path)})
    if not input_files:
        raise ValueError("TRICERATOPS requires candidate-local photometry provenance")

    return {
        "time_days": phase_days,
        "flux": phase_flux,
        "flux_err": flux_err_scalar,
        "period_days": period_days,
        "duration_hours": duration_days * 24.0,
        "depth_ppm": depth_ppm,
        "exposure_days": exposure_days,
        "sectors": sorted({int(value) for value in sectors}),
        "provenance": {
            "representation": "phase-folded observed candidate photometry",
            "time_reference": "days from nearest declared transit midpoint",
            "raw_cadence_count": int(time_btjd.size),
            "phase_bin_count": int(phase_days.size),
            "phase_binning": {
                "method": "transit-centered-nonuniform",
                "transit_bin_count": transit_bin_count,
                "transit_bin_width_days": duration_days / transit_bin_count,
                "local_baseline_limit_days": local_baseline_limit,
            },
            "flux_error_source": "reported per-cadence uncertainties",
            "flux_error_scalar": flux_err_scalar,
            "exposure_days": exposure_days,
            "input_files": input_files,
            "ephemeris_artifacts": _ephemeris_artifacts(
                workspace, signal, ephemeris.get("source")
            ),
            "ephemeris_source": ephemeris.get("source"),
            "ephemeris_field_sources": {
                field: field_sources.get(field) for field in required_fields
            },
            "observed_depth_ppm": depth_ppm,
        },
    }


def fpp_gate(
    report_or_value: Dict[str, Any],
    threshold: float = FPP_THRESHOLD,
    nfpp_threshold: float = NFPP_THRESHOLD,
) -> Tuple[bool, float]:
    """Return ``(passes, fpp)`` for an FPP or joint FPP/NFPP screen.

    A scalar preserves the historical FPP-only utility behavior. A report is
    stricter: both finite probabilities must lie in the unit interval and each
    must be below its registered threshold. The function remains a transparent
    routing helper; the global analysis claim gate is intentionally separate
    and remains closed pending calibrated scene constraints.
    """
    if isinstance(report_or_value, dict):
        fpp = extract_fpp(report_or_value)
        try:
            nfpp = extract_nfpp(report_or_value)
        except (TypeError, ValueError):
            return False, fpp
        if not (
            math.isfinite(fpp)
            and math.isfinite(nfpp)
            and 0.0 <= fpp <= 1.0
            and 0.0 <= nfpp <= 1.0
        ):
            return False, fpp
        return fpp < threshold and nfpp < nfpp_threshold, fpp
    else:
        fpp = float(report_or_value)
    return fpp < threshold, fpp


def run_triceratops_simulation(
    workspace: Any,
    n_draws: int = 2000,
    search_radius: int = 10,
    signal: Optional[str] = None,
    allow_fallback: bool = False,
    random_seed: int = DEFAULT_TRICERATOPS_SEED,
    n_jobs: int = 1,
    progress_callback: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None,
) -> Path:
    """Run TRICERATOPS Monte Carlo Bayesian false positive probability sampling target-neutrally.

    Reads candidate target metadata and either a per-signal transit config
    (``config/signals/transit_config<signal>.json`` when ``signal`` is given)
    or the BLS periodogram outputs, executes Monte Carlo sampling over
    candidate model scenarios, and writes outputs/triceratops_report.json.
    FPP claims are disabled until observed photometry and scene constraints are
    integrated and provenance-bound.

    Parameters
    ----------
    allow_fallback:
        When True, allow the report to be written even if the TRICERATOPS Monte
        Carlo could not run (e.g., the package is not installed). The report will
        carry ``source = "triceratops-failed-UNVALIDATED"`` and ``FPP = null``.
        When False (default), a RuntimeError is raised so the analysis gate is
        not silently satisfied by a placeholder value.
    random_seed:
        Seed applied only while calling the backend's legacy global NumPy RNG,
        then restored. The seed is recorded in the report. It makes this
        single realization replayable but does not quantify Monte Carlo
        variation across independent realizations.
    n_jobs:
        One preserves the canonical serial reference execution. Values above
        one opt into TRICERATOPS's in-process PyTransit vectorized mode and
        request that its Numba runtime use at most that many threads. Exonym
        never partitions scenarios or draw streams across processes.
    progress_callback:
        Optional milestone callback receiving ``(step, done, total)``. The
        optional backend does not expose per-draw progress, so only start and
        completion milestones are emitted.
    """
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")
    if random_seed < 0 or random_seed > 2**32 - 1:
        raise ValueError("random_seed must be between 0 and 2**32 - 1")
    if isinstance(n_draws, bool) or int(n_draws) < 1:
        raise ValueError("n_draws must be at least one")
    if isinstance(search_radius, bool) or int(search_radius) < 1:
        raise ValueError("search_radius must be at least one")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    signal = validate_signal_suffix(signal)
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    tic_str = workspace.metadata.get("identifiers", {}).get("tic")
    tic_id = int(tic_str) if tic_str and str(tic_str).isdigit() else None
    observed_input: Optional[Dict[str, Any]] = None
    _notify_progress(progress_callback, "Preparing observed TRICERATOPS input")
    from ..statistical_vetting import (
        triceratops_vetting_decision_path,
        write_triceratops_vetting_decision,
        _write_json_atomic,
    )

    def _existing_vetting_decision() -> Dict[str, Any]:
        path = triceratops_vetting_decision_path(workspace, signal)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    input_snapshot = []
    if tic_id is not None:
        # The public function is also an API entry point. Enforce the same
        # ordered readiness checks as the CLI before importing or invoking the
        # Monte Carlo backend, so callers cannot bypass the scientific guard.
        from ..statistical_vetting import require_vetting_readiness

        require_vetting_readiness(workspace, signal=signal)
        try:
            observed_input = _prepare_observed_transit_input(workspace, signal)
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            # The observed-input contract is part of execution, not a
            # scientific disposition. Keep the candidate unresolved and
            # replace a stale ``ready`` record with an explicit failure.
            write_triceratops_vetting_decision(
                workspace,
                signal=signal,
                execution_status="failed",
                triage_status=_existing_vetting_decision().get("triage_status", "not-run"),
                result_status="unresolved",
                fpp=None,
                nfpp=None,
                blocking_reasons=[],
                error={
                    "code": "triceratops-observed-input-failed",
                    "message": "{0}: {1}".format(type(exc).__name__, exc),
                },
            )
            raise

        previous = _existing_vetting_decision()
        try:
            input_snapshot = _snapshot_artifacts(
                workspace,
                list(previous.get("input_artifacts", []))
                + list(observed_input["provenance"].get("input_files", []))
                + list(observed_input["provenance"].get("ephemeris_artifacts", [])),
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            write_triceratops_vetting_decision(
                workspace,
                signal=signal,
                execution_status="failed",
                triage_status=previous.get("triage_status", "not-run"),
                result_status="unresolved",
                fpp=None,
                nfpp=None,
                blocking_reasons=[],
                error={
                    "code": "triceratops-input-snapshot-failed",
                    "message": "{0}: {1}".format(type(exc).__name__, exc),
                },
            )
            raise

    period, depth_ppm, duration_hrs, ephemeris_source = 2.50, 1250.0, 2.85, "defaults"
    if observed_input is not None:
        period = observed_input["period_days"]
        depth_ppm = observed_input["depth_ppm"]
        duration_hrs = observed_input["duration_hours"]
        ephemeris_source = observed_input["provenance"]["ephemeris_source"]
    else:
        # A fallback report may describe a candidate-owned ephemeris, but it
        # must use the same provenance checks as the actual observed-data path.
        try:
            from ..inputs import load_transit_ephemeris

            ephemeris = load_transit_ephemeris(workspace, signal=signal)
            field_sources = ephemeris.get("field_sources", {})
            required_fields = ("period_days", "duration_days", "depth_ppm")
            if not isinstance(field_sources, dict) or any(
                field_sources.get(field) == "synthetic-demo" for field in required_fields
            ):
                raise ValueError("ephemeris is incomplete or synthetic")
            period = float(ephemeris["period_days"])
            depth_ppm = float(ephemeris["depth_ppm"])
            duration_hrs = float(ephemeris["duration_days"]) * 24.0
            if (
                not math.isfinite(period)
                or not math.isfinite(depth_ppm)
                or not math.isfinite(duration_hrs)
                or period <= 0.0
                or depth_ppm <= 0.0
                or duration_hrs <= 0.0
            ):
                raise ValueError("ephemeris values are not physically usable")
            ephemeris_source = str(ephemeris.get("source"))
        except (KeyError, TypeError, ValueError) as exc:
            detail = (
                "could not read signal transit config transit_config{0}.json".format(signal)
                if signal is not None
                else "could not load candidate-derived transit ephemeris"
            )
            warnings.warn(
                "{0}: {1!r}".format(detail, exc),
                stacklevel=2,
            )

    # fpp is initialized to NaN so any code path that does not successfully
    # run the Monte Carlo produces an explicit sentinel rather than a
    # hardcoded value that could silently satisfy the FPP gate.
    fpp: float = float("nan")
    nfpp: float = float("nan")
    scenarios: Dict[str, float] = {}
    source = "not-run"
    triceratops_error: Optional[str] = None
    triceratops_exception_type: Optional[str] = None
    triceratops_error_code: Optional[str] = None
    backend: Optional[Dict[str, Any]] = None
    scene_artifacts = []
    runtime_compatible = True

    if tic_id is not None:
        from ..engines import check_engine

        runtime_compatible, runtime_message = check_engine("triceratops")
        if not runtime_compatible:
            triceratops_error = runtime_message
            triceratops_error_code = "triceratops-runtime-incompatible"
            source = "triceratops-failed-UNVALIDATED"

    if tic_id is not None and runtime_compatible:
        try:
            import numpy as np

            # TRICERATOPS and pytransit releases in the supported optional
            # stack can still import these aliases on NumPy 1.24+ / 2.x.
            with _TRICERATOPS_PROCESS_STATE_LOCK:
                _ensure_legacy_numpy_scalars(np)
                import triceratops.triceratops as triceratops_module

                try:
                    triceratops_version = importlib.metadata.version("triceratops")
                except importlib.metadata.PackageNotFoundError:
                    triceratops_version = "unknown"
                backend = {
                    "package": "triceratops",
                    "version": triceratops_version,
                    "numpy_version": str(np.__version__),
                }
                target_cls = triceratops_module.target
                if observed_input is None:
                    raise RuntimeError("TRICERATOPS observed input preparation did not complete")
                sectors = observed_input["sectors"]
                # TRICERATOPS writes scene inputs in its working directory. Retain
                # them under the candidate workspace for inspection and replay.
                cwd_before = os.getcwd()
                rng_state = np.random.get_state()
                scene_dir = workspace.path / "data" / "external" / "triceratops" / uuid.uuid4().hex
                scene_dir.mkdir(parents=True, exist_ok=False)
                try:
                    _verify_snapshot(workspace, input_snapshot)
                    os.chdir(scene_dir)
                    np.random.seed(random_seed)
                    _notify_progress(progress_callback, "Constructing TRICERATOPS scene")
                    with _triceratops_tls_environment() as certificate_source:
                        backend["tls_certificate_source"] = certificate_source
                        targ = target_cls(
                            ID=tic_id,
                            sectors=np.array(sectors, dtype=int),
                            search_radius=search_radius,
                            mission="TESS",
                        )
                        targ.calc_depths(depth_ppm * 1e-6)
                        backend["execution"] = _parallel_calc_probs_dispatcher(
                            targ,
                            observed_input,
                            period,
                            n_draws,
                            n_jobs,
                            progress_callback,
                        )
                    _verify_snapshot(workspace, input_snapshot)

                    fpp = float(targ.FPP)
                    nfpp = float(targ.NFPP)
                    if not (math.isfinite(fpp) and 0.0 <= fpp <= 1.0):
                        raise RuntimeError("TRICERATOPS returned an invalid FPP")
                    if not (math.isfinite(nfpp) and 0.0 <= nfpp <= 1.0):
                        raise RuntimeError("TRICERATOPS returned an invalid NFPP")
                    if hasattr(targ, "probs") and hasattr(targ.probs, "groupby"):
                        scenarios = (
                            targ.probs.groupby("scenario")["prob"]
                            .sum()
                            .sort_values(ascending=False)
                            .to_dict()
                        )
                    source = "triceratops-monte-carlo"
                finally:
                    np.random.set_state(rng_state)
                    os.chdir(cwd_before)
                    scene_artifacts = [
                        _artifact_from_path(workspace, path)
                        for path in sorted(scene_dir.rglob("*"))
                        if path.is_file()
                    ]
        except Exception as exc:
            fpp = float("nan")
            nfpp = float("nan")
            scenarios = {}
            triceratops_error = "{0}: {1}".format(type(exc).__name__, exc)
            triceratops_exception_type = type(exc).__name__
            triceratops_error_code = (
                "triceratops-tls-verification-failed"
                if _is_tls_verification_error(exc)
                else "triceratops-runtime-failed"
            )
            warnings.warn(
                "TRICERATOPS Monte Carlo failed: {0!r}. "
                "FPP will be marked UNVALIDATED.".format(exc),
                stacklevel=2,
            )
            source = "triceratops-failed-UNVALIDATED"

    if source in ("not-run", "triceratops-failed-UNVALIDATED"):
        unavailable = source == "not-run" or triceratops_exception_type in {
            "ImportError", "ModuleNotFoundError", "PackageNotFoundError",
        } or triceratops_error_code == "triceratops-runtime-incompatible"
        failure = triceratops_error or "TRICERATOPS could not run because no numeric TIC identifier is available."
        write_triceratops_vetting_decision(
            workspace,
            signal=signal,
            execution_status="unavailable" if unavailable else "failed",
            triage_status=_existing_vetting_decision().get("triage_status", "not-run"),
            blocking_reasons=[],
            input_artifacts=input_snapshot,
            error={
                "code": triceratops_error_code
                if triceratops_error_code == "triceratops-runtime-incompatible"
                else "triceratops-unavailable"
                if unavailable
                else triceratops_error_code,
                "message": failure,
            },
        )

    # Raise before writing any files when the Monte Carlo was not run and the
    # caller has not explicitly opted in to an unvalidated fallback.
    if not allow_fallback and (source in ("not-run", "triceratops-failed-UNVALIDATED")):
        raise RuntimeError(
            "TRICERATOPS Monte Carlo did not run (source={0!r}). "
            "Install the 'triceratops' package or pass allow_fallback=True "
            "to write an unvalidated placeholder report.".format(source)
        )

    fpp_rounded: Optional[float] = round(fpp, 6) if math.isfinite(fpp) else None
    nfpp_rounded: Optional[float] = round(nfpp, 6) if math.isfinite(nfpp) else None

    report = {
        "method": "TRICERATOPS",
        "candidate_id": workspace.candidate_id,
        "tic_id": tic_id,
        "signal": signal,
        "n_draws": n_draws,
        "random_seed": random_seed,
        "backend": backend,
        "ephemeris": {
            "period_days": round(period, 6),
            "depth_ppm": round(depth_ppm, 2),
            "duration_hours": round(duration_hrs, 3),
            "source": ephemeris_source,
        },
        "FPP": fpp_rounded,
        "NFPP": nfpp_rounded,
        "scenarios": scenarios,
        "source": source,
        "triceratops_error": triceratops_error,
        "input_provenance": (
            {
                **observed_input["provenance"],
                "bound_artifacts": input_snapshot,
                "scene_artifacts": scene_artifacts,
            }
            if observed_input is not None
            else None
        ),
        "audit_status": "valid" if source == "triceratops-monte-carlo" else "invalid",
        "audit_invalid_reason": (
            None
            if source == "triceratops-monte-carlo"
            else "TRICERATOPS Monte Carlo did not complete; this report is not auditable scientific evidence."
        ),
        "claim_eligible": False,
        "claim_block_reason": FPP_CLAIM_BLOCK_REASON,
    }
    suffix = f".{signal.lstrip('.')}" if signal else ""
    report_path = outputs_dir / f"triceratops_report{suffix}.json"
    _write_json_atomic(report_path, report)

    if source == "triceratops-monte-carlo":
        passes = fpp_rounded is not None and nfpp_rounded is not None and fpp_rounded < FPP_THRESHOLD and nfpp_rounded < NFPP_THRESHOLD
        previous = _existing_vetting_decision()
        triage_status = previous.get("triage_status", "not-run")
        write_triceratops_vetting_decision(
            workspace,
            signal=signal,
            execution_status="succeeded",
            triage_status=triage_status,
            result_status=("review-required" if passes and triage_status == "review-required" else "fpp-pass" if passes else "fpp-fail"),
            fpp=fpp_rounded,
            nfpp=nfpp_rounded,
            input_artifacts=input_snapshot,
            triceratops_report={"path": report_path.relative_to(workspace.path).as_posix(), "sha256": _sha256(report_path)},
            audit_status="valid",
            audit_invalid_reason=None,
        )
    elif allow_fallback:
        previous = _existing_vetting_decision()
        write_triceratops_vetting_decision(
            workspace,
            signal=signal,
            execution_status=previous.get("execution_status", "failed"),
            triage_status=previous.get("triage_status", "not-run"),
            result_status="unresolved",
            error=previous.get("error") if isinstance(previous.get("error"), dict) else None,
            input_artifacts=input_snapshot,
            triceratops_report={"path": report_path.relative_to(workspace.path).as_posix(), "sha256": _sha256(report_path)},
            audit_status="invalid",
            audit_invalid_reason=report["audit_invalid_reason"],
        )

    return report_path
