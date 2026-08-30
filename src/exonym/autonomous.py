"""Bounded candidate-local automation for evidence collection and vetting.

The coordinator invokes a fixed sequence of existing commands and writes a
candidate-local run record with the selected inputs and outputs. It keeps
automation operationally useful without allowing a convenience command to
change lifecycle ownership or scientific interpretation.

Scientific boundary:
    Automation is not a workflow-state shortcut. It never checks human
    checklist items, advances phases, assigns a disposition, records a decisive
    rejection, or turns a statistical output into a validation claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import __version__
from .inputs import _read_json, MINIMUM_BLS_CANDIDATE_SNR, is_manifest_bound_bls_result
from .workspace import CandidateWorkspace

_EXPECTED_AUTOMATION_FAILURES = (FileNotFoundError, ImportError, ValueError, RuntimeError)


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    """Durably replace a text record without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def record_autonomous_incident(
    repository_root: Path, command: str, exc: BaseException, exit_code: int = 2
) -> Path:
    """Atomically retain an unexpected autonomous-command failure at repo root."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    suffix = uuid.uuid4().hex[:8]
    issue_id = "ISSUE-{0}-{1}".format(now.strftime("%Y%m%d"), suffix)
    timestamp = now.isoformat().replace("+00:00", "Z")
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    content = """# {issue_id}

- UTC Timestamp: `{timestamp}`
- Exonym Version: `{version}`
- CLI Command: `{command}`
- Exit Code: `{exit_code}`

## Expected Behavior

The bounded autonomous command should retain durable progress records and return a controlled outcome.

## Observed Behavior

Unhandled `{exception_type}`: `{exception}`

## Full Python Traceback

```text
{trace}```

## Root Cause And Affected Modules

The immediate cause is the unhandled exception above. Inspect the traceback for the affected `src/exonym/` modules.

## Actionable Remediation

Reproduce the recorded command, inspect its durable run record, and fix the failing operation without changing lifecycle state, disposition, or claim eligibility.
""".format(
        issue_id=issue_id,
        timestamp=timestamp,
        version=__version__,
        command=command,
        exit_code=exit_code,
        exception_type=type(exc).__name__,
        exception=exc,
        trace=trace,
    )
    path = repository_root.resolve() / "log" / "issue-{0}-{1}.md".format(now.strftime("%Y%m%d"), suffix)
    _atomic_write_text(path, content)
    return path


def create_run_loop_journal(survey: Any, configuration: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    """Create the durable parent record for one bounded survey run-loop."""
    run_id = uuid.uuid4().hex
    path = survey.path / "runs" / "run-loop" / run_id / "run-loop.json"
    journal: Dict[str, Any] = {
        "schema_version": 1,
        "engine": "run-loop",
        "run_id": run_id,
        "survey_id": survey.survey_id,
        "status": "running",
        "started_at": _timestamp(),
        "completed_at": None,
        "configuration": configuration,
        "cycles": [],
        "claim_eligible": False,
        "workflow_advanced": False,
        "disposition_changed": False,
    }
    _atomic_write_json(path, journal)
    return path, journal


def write_run_loop_journal(path: Path, journal: Dict[str, Any]) -> None:
    """Atomically checkpoint a run-loop parent record after each batch event."""
    _atomic_write_json(path, journal)


def auto_vet_started(candidate: CandidateWorkspace) -> bool:
    """Return whether a candidate has any durable auto-vet run snapshot."""
    return any(
        path.is_file()
        for path in (candidate.path / "runs" / "auto-vet").glob("*/engine-run.json")
    )


def _artifact(candidate: CandidateWorkspace, path: Path, role: str) -> Dict[str, str]:
    return {"path": path.relative_to(candidate.path).as_posix(), "sha256": _sha256(path), "role": role}


def _write_bls_transit_config(candidate: CandidateWorkspace, result_path: Path) -> Path:
    """Persist only measured BLS values as a candidate-local downstream input."""
    try:
        result = _read_json(result_path)
        if not isinstance(result, dict) or not is_manifest_bound_bls_result(
            candidate, result_path, result, None
        ):
            raise ValueError("BLS result is not bound to raw provenance and its manifest")
        if result.get("detection_status") != "detected":
            raise ValueError("BLS result is not a detected transit signal")
        if result.get("time_system") != "BTJD_TDB":
            raise ValueError("BLS result does not declare a BTJD_TDB epoch")
        period = float(result["best_period"])
        epoch = float(result["best_epoch"])
        duration_hours = float(result["best_duration_hours"])
        depth = float(result["best_depth_ppm"])
        snr = float(result["snr"])
        event_count = int(result["n_distinct_transit_events"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("BLS output cannot provide a candidate-local transit configuration") from exc
    if not all(math.isfinite(value) for value in (period, epoch, duration_hours, depth, snr)):
        raise RuntimeError("BLS output contains non-finite transit measurements")
    if period <= 0.0 or duration_hours <= 0.0 or duration_hours / 24.0 >= period or depth <= 0.0:
        raise RuntimeError("BLS output contains unusable transit measurements")
    if snr < MINIMUM_BLS_CANDIDATE_SNR:
        raise RuntimeError("BLS output does not meet the candidate-selection threshold")
    if event_count < 2:
        raise RuntimeError("BLS output does not contain two distinct transit events")
    manifest_path = candidate.path / "outputs" / "bls_search_manifest.json"
    payload = {
        "source": "candidate-data-bls",
        "bls_provenance": {
            "result": _artifact(candidate, result_path, "bls-result"),
            "manifest": _artifact(candidate, manifest_path, "bls-manifest"),
        },
        "transit": {
            "period_days": period,
            "epoch_btjd": epoch,
            "epoch_time_system": "BTJD_TDB",
            "duration_days": duration_hours / 24.0,
            "depth_ppm": depth,
        },
    }
    path = candidate.path / "config" / "transit_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _has_raw_fits(candidate: CandidateWorkspace) -> bool:
    raw = candidate.path / "data" / "raw"
    return any(path.is_file() and path.suffix.lower() in (".fits", ".fz") for path in raw.rglob("*"))


def _normalize_requested_sectors(sectors: Optional[Sequence[int]]) -> Optional[List[int]]:
    """Return a sorted explicit sector scope or reject malformed caller values."""
    if sectors is None:
        return None
    from .ingest import _coerce_sector_value

    normalized: List[int] = []
    for value in sectors:
        sector = _coerce_sector_value(value)
        if sector is None:
            raise ValueError("sectors must contain positive integer values")
        normalized.append(sector)
    return sorted(set(normalized))


def _available_common_sectors(
    lightcurve_sequence_numbers: Iterable[object],
    tpf_sequence_numbers: Iterable[object],
) -> List[int]:
    """Return positive archive sectors represented by both SPOC product searches."""
    from .ingest import _coerce_sector_value

    lightcurve_sectors = {
        sector
        for value in lightcurve_sequence_numbers
        for sector in [_coerce_sector_value(value)]
        if sector is not None
    }
    tpf_sectors = {
        sector
        for value in tpf_sequence_numbers
        for sector in [_coerce_sector_value(value)]
        if sector is not None
    }
    return sorted(lightcurve_sectors.intersection(tpf_sectors))


def _select_download_sectors(
    available_common_sectors: Sequence[int], requested_sectors: Optional[Sequence[int]]
) -> List[int]:
    """Select the effective common archive sectors for one auto-vet download.

    The default preserves the bounded one-sector acquisition policy. Explicit
    caller selections are intersected with products available in both searches
    rather than being forwarded as an unverified archive filter.
    """
    available = sorted(set(available_common_sectors))
    if not available:
        raise RuntimeError("No common LC and TPF sectors for target")
    if requested_sectors is None:
        return [available[0]]
    selected = sorted(set(available).intersection(requested_sectors))
    if not selected:
        raise RuntimeError("No requested sectors are available in both LC and TPF searches")
    return selected


def auto_vet_candidate(
    candidate: CandidateWorkspace,
    sectors: Optional[Sequence[int]] = None,
    n_draws: int = 500,
    fit_samples: int = 2500,
    download: bool = True,
    incident_command: Optional[str] = None,
) -> Path:
    """Run a bounded evidence chain and write an engine-run manifest.

    Individual failures are retained in the manifest and do not cause later,
    independent diagnostics to be skipped. The final TRICERATOPS call continues
    to enforce ``require_vetting_readiness`` internally. The manifest records
    the effective sector scope passed to search; ``null`` denotes an unfiltered
    existing candidate-local data set.
    """
    if isinstance(n_draws, bool) or int(n_draws) < 1:
        raise ValueError("n_draws must be at least one")
    if isinstance(fit_samples, bool) or int(fit_samples) < 1:
        raise ValueError("fit_samples must be at least one")
    requested_sectors = _normalize_requested_sectors(sectors)
    sectors_used = requested_sectors
    run_id = uuid.uuid4().hex
    run_dir = candidate.path / "runs" / "auto-vet" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = _timestamp()
    artifacts: List[Dict[str, str]] = []
    steps: List[Dict[str, str]] = []
    incident_recorded = False
    manifest_path = run_dir / "engine-run.json"
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": candidate.candidate_id,
        "engine": "auto-vet",
        "run_id": run_id,
        "status": "blocked",
        "started_at": started_at,
        "completed_at": started_at,
        "runtime": {
            "kind": "direct",
            "version": __version__,
            "version_known": True,
            "executable": "exonym.autonomous",
        },
        "inputs": [_artifact(candidate, candidate.path / "candidate.json", "candidate-metadata")],
        "outputs": artifacts,
        "failure": {
            "code": "auto-vet-incomplete",
            "message": "Auto-vet run started but has not completed.",
        },
        "automation": {
            "steps": [
                {
                    "name": "initialization",
                    "status": "skipped",
                    "detail": "Durable auto-vet manifest initialized.",
                }
            ],
            "sectors_used": sectors_used,
            "claim_eligible": False,
            "disposition_changed": False,
            "workflow_advanced": False,
        },
    }

    def checkpoint_manifest() -> None:
        manifest["completed_at"] = _timestamp()
        manifest["outputs"] = list(artifacts)
        manifest["automation"]["steps"] = list(steps) or manifest["automation"]["steps"]
        manifest["automation"]["sectors_used"] = sectors_used
        _atomic_write_json(manifest_path, manifest)

    checkpoint_manifest()

    def execute(name: str, operation: Callable[[], Any]) -> Optional[Any]:
        nonlocal incident_recorded
        try:
            result = operation()
            paths = result if isinstance(result, (list, tuple)) else [result]
            for output in paths:
                if isinstance(output, Path) and output.is_file():
                    artifacts.append(_artifact(candidate, output, name))
            steps.append({"name": name, "status": "succeeded"})
            checkpoint_manifest()
            return result
        except Exception as exc:  # Preserve later independent diagnostic attempts.
            steps.append({"name": name, "status": "blocked", "detail": "{0}: {1}".format(type(exc).__name__, exc)})
            checkpoint_manifest()
            if not incident_recorded and not isinstance(exc, _EXPECTED_AUTOMATION_FAILURES):
                record_autonomous_incident(
                    candidate.repository_root,
                    incident_command or "exonym survey auto-vet {0}".format(candidate.candidate_id),
                    exc,
                )
                incident_recorded = True
            return None

    if download and not _has_raw_fits(candidate):
        def ingest() -> List[Path]:
            nonlocal sectors_used

            from contextlib import ExitStack

            from .ingest import fetch_tess_products, fetch_tess_tpfs, ingest_products
            import lightkurve as lk

            sectors_used = []
            tic = candidate.metadata["identifiers"].get("tic")
            sr_lc = lk.search_lightcurve(f"TIC {tic}", author="SPOC")
            sr_tp = lk.search_targetpixelfile(f"TIC {tic}", author="SPOC")
            if not sr_lc or not sr_tp:
                raise RuntimeError("MAST returned no requested SPOC products")
            common = _available_common_sectors(
                sr_lc.table["sequence_number"], sr_tp.table["sequence_number"]
            )
            sectors_used = _select_download_sectors(common, requested_sectors)

            with ExitStack() as staging_batches:
                products = list(
                    staging_batches.enter_context(
                        fetch_tess_products(candidate, sectors=sectors_used, provider="spoc")
                    )
                )
                products.extend(
                    staging_batches.enter_context(
                        fetch_tess_tpfs(candidate, sectors=sectors_used, provider="spoc")
                    )
                )
                if not products:
                    raise RuntimeError("MAST returned no requested SPOC products")
                return ingest_products(candidate, products, fetched_by="exonym-auto-vet/1.2.0")

        execute("ingest", ingest)
    else:
        steps.append({"name": "ingest", "status": "skipped", "detail": "candidate-local raw FITS products already exist"})
        checkpoint_manifest()

    def search() -> Path:
        from .search import run_bls_on_candidate

        result_path = run_bls_on_candidate(candidate, sectors=sectors_used)
        config_path = _write_bls_transit_config(candidate, result_path)
        artifacts.append(_artifact(candidate, config_path, "bls-transit-config"))
        return result_path

    execute("search", search)
    operations: Tuple[Tuple[str, Callable[[], Any]], ...] = (
        ("screen", lambda: __import__("exonym.screening", fromlist=["run_fixed_ephemeris_screen"]).run_fixed_ephemeris_screen(candidate)),
        ("archive", lambda: __import__("exonym.archive", fromlist=["run_archival_vetting"]).run_archival_vetting(candidate)),
        ("localization", lambda: __import__("exonym.localization", fromlist=["run_prf_localization"]).run_prf_localization(candidate)),
        ("activity", lambda: __import__("exonym.activity", fromlist=["run_stellar_activity"]).run_stellar_activity(candidate)),
        ("asteroseismology", lambda: __import__("exonym.asteroseismology", fromlist=["run_asteroseismology"]).run_asteroseismology(candidate)),
        ("sed", lambda: __import__("exonym.sed", fromlist=["run_sed_fit"]).run_sed_fit(candidate)),
        ("dilution", lambda: __import__("exonym.dilution", fromlist=["run_dilution_sensitivity"]).run_dilution_sensitivity(candidate)),
        ("fit", lambda: __import__("exonym.transit_fit", fromlist=["run_mcmc_transit_fit"]).run_mcmc_transit_fit(candidate, n_samples=int(fit_samples))),
        ("ttv", lambda: __import__("exonym.ttv", fromlist=["run_ttv_analysis"]).run_ttv_analysis(candidate)),
        ("phasecurve", lambda: __import__("exonym.phasecurve", fromlist=["run_phase_curve_search"]).run_phase_curve_search(candidate)),
        ("plot", lambda: __import__("exonym.plotting", fromlist=["generate_candidate_plots"]).generate_candidate_plots(candidate)),
        ("triage", lambda: __import__("exonym.engines", fromlist=["run_automated_triage"]).run_automated_triage(candidate)),
        ("vet", lambda: __import__("exonym.vetting.tricera_parse", fromlist=["run_triceratops_simulation"]).run_triceratops_simulation(candidate, n_draws=int(n_draws))),
    )
    for name, operation in operations:
        execute(name, operation)

    vet_step = next((step for step in steps if step["name"] == "vet"), None)
    succeeded = vet_step is not None and vet_step["status"] == "succeeded"
    manifest["status"] = "succeeded" if succeeded else "blocked"
    if not succeeded:
        detail = vet_step.get("detail") if vet_step is not None else "vet step was not attempted"
        manifest["failure"] = {"code": "vetting-not-complete", "message": str(detail)}
    else:
        manifest.pop("failure", None)
    checkpoint_manifest()
    return manifest_path
