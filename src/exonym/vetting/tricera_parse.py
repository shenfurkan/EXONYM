"""Interface to TRICERATOPS FPP reports.

Parses a TRICERATOPS output JSON file and applies the statistical validation
gate: FPP below the preregistered threshold.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..workspace import validate_signal_suffix

FPP_THRESHOLD = 0.01
FPP_CLAIM_BLOCK_REASON = (
    "FPP claim creation is disabled until TRICERATOPS receives provenance-bound "
    "observed photometry and scene constraints."
)
DEFAULT_TRICERATOPS_SEED = 1729


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
    """Load a TRICERATOPS output report (JSON dict)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("TRICERATOPS report must be a JSON object")
    return data


def extract_fpp(report: Dict[str, Any]) -> float:
    """Return the FPP value from a report, probing common key layouts."""
    for key in ("fpp", "FPP", "fpp_value"):
        value = report.get(key)
        if value is not None:
            return float(value)
    for key in ("fpp_specific", "FPP_specific", "fpp_specific_value"):
        value = report.get(key)
        if value is not None:
            return float(value)
    raise ValueError("no FPP value found in report")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for a candidate-local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_observed_transit_input(workspace: Any, signal: Optional[str]) -> Dict[str, Any]:
    """Return provenance-bound, phase-folded observed photometry for TRICERATOPS.

    The returned ``time_days`` values are measured from the nearest declared
    transit midpoint. Flux values are inverse-variance binned across phase,
    using a maximum of one hundred bins as recommended by the TRICERATOPS
    tutorial. The backend accepts only one flux uncertainty, so this function
    records and passes the mean uncertainty of the observed phase bins.
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
        or duration_days >= period_days
    ):
        raise ValueError("TRICERATOPS requires a positive duration shorter than the orbital period")

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
    bin_count = min(100, time_days.size)
    edges = np.linspace(-0.5 * period_days, 0.5 * period_days, bin_count + 1)
    binned_time: list = []
    binned_flux: list = []
    binned_err: list = []
    for index in range(bin_count):
        if index + 1 == bin_count:
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
    out_of_transit = np.abs(phase_days) >= duration_days
    if int(np.count_nonzero(in_transit)) < 3 or int(np.count_nonzero(out_of_transit)) < 3:
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
            "flux_error_source": "reported per-cadence uncertainties",
            "flux_error_scalar": flux_err_scalar,
            "exposure_days": exposure_days,
            "input_files": input_files,
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
) -> Tuple[bool, float]:
    """Return (pass, fpp). Pass means FPP is below the threshold."""
    if isinstance(report_or_value, dict):
        fpp = extract_fpp(report_or_value)
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
    """
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")
    if random_seed < 0 or random_seed > 2**32 - 1:
        raise ValueError("random_seed must be between 0 and 2**32 - 1")
    if isinstance(n_draws, bool) or int(n_draws) < 1:
        raise ValueError("n_draws must be at least one")
    if isinstance(search_radius, bool) or int(search_radius) < 1:
        raise ValueError("search_radius must be at least one")
    signal = validate_signal_suffix(signal)
    outputs_dir = workspace.path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    tic_str = workspace.metadata.get("identifiers", {}).get("tic")
    tic_id = int(tic_str) if tic_str and str(tic_str).isdigit() else None
    observed_input: Optional[Dict[str, Any]] = None

    if tic_id is not None:
        # The public function is also an API entry point. Enforce the same
        # ordered readiness checks as the CLI before importing or invoking the
        # Monte Carlo backend, so callers cannot bypass the scientific guard.
        from ..statistical_vetting import require_vetting_readiness

        require_vetting_readiness(workspace, signal=signal)
        observed_input = _prepare_observed_transit_input(workspace, signal)

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
    backend: Optional[Dict[str, str]] = None
    scene_artifacts = []

    if tic_id is not None:
        try:
            import numpy as np

            # TRICERATOPS and pytransit releases in the supported optional
            # stack can still import these aliases on NumPy 1.24+ / 2.x.
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
                os.chdir(scene_dir)
                np.random.seed(random_seed)
                targ = target_cls(
                    ID=tic_id,
                    sectors=np.array(sectors, dtype=int),
                    search_radius=search_radius,
                    mission="TESS",
                )
                targ.calc_depths(depth_ppm * 1e-6)

                targ.calc_probs(
                    time=observed_input["time_days"],
                    flux_0=observed_input["flux"],
                    flux_err_0=observed_input["flux_err"],
                    P_orb=period,
                    N=n_draws,
                    parallel=False,
                    verbose=0,
                    exptime=observed_input["exposure_days"],
                    nsamples=5,
                )

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
                    {
                        "path": path.relative_to(workspace.path).as_posix(),
                        "sha256": _sha256(path),
                    }
                    for path in sorted(scene_dir.rglob("*"))
                    if path.is_file()
                ]
        except Exception as exc:
            fpp = float("nan")
            nfpp = float("nan")
            scenarios = {}
            triceratops_error = "{0}: {1}".format(type(exc).__name__, exc)
            warnings.warn(
                "TRICERATOPS Monte Carlo failed: {0!r}. "
                "FPP will be marked UNVALIDATED.".format(exc),
                stacklevel=2,
            )
            source = "triceratops-failed-UNVALIDATED"

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
            {**observed_input["provenance"], "scene_artifacts": scene_artifacts}
            if observed_input is not None
            else None
        ),
        "claim_eligible": False,
        "claim_block_reason": FPP_CLAIM_BLOCK_REASON,
    }
    suffix = f".{signal.lstrip('.')}" if signal else ""
    report_path = outputs_dir / f"triceratops_report{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    return report_path
