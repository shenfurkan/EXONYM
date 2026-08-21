"""Bounded candidate-local automation for evidence collection and vetting.

This is an execution coordinator, not a workflow-state shortcut. It never
checks human checklist items, advances phases, assigns a disposition, records a
decisive rejection, or turns an FPP into a validation claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .inputs import MINIMUM_BLS_CANDIDATE_SNR
from .workspace import CandidateWorkspace


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(candidate: CandidateWorkspace, path: Path, role: str) -> Dict[str, str]:
    return {"path": path.relative_to(candidate.path).as_posix(), "sha256": _sha256(path), "role": role}


def _write_bls_transit_config(candidate: CandidateWorkspace, result_path: Path) -> Path:
    """Persist only measured BLS values as a candidate-local downstream input."""
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("detection_status") != "detected":
            raise ValueError("BLS result is not a detected transit signal")
        if result.get("time_system") != "BTJD_TDB":
            raise ValueError("BLS result does not declare a BTJD_TDB epoch")
        period = float(result["best_period"])
        epoch = float(result["best_epoch"])
        duration_hours = float(result["best_duration_hours"])
        depth = float(result["best_depth_ppm"])
        snr = float(result["snr"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("BLS output cannot provide a candidate-local transit configuration") from exc
    if not all(math.isfinite(value) for value in (period, epoch, duration_hours, depth, snr)):
        raise RuntimeError("BLS output contains non-finite transit measurements")
    if period <= 0.0 or duration_hours <= 0.0 or duration_hours / 24.0 >= period or depth <= 0.0:
        raise RuntimeError("BLS output contains unusable transit measurements")
    if snr < MINIMUM_BLS_CANDIDATE_SNR:
        raise RuntimeError("BLS output does not meet the candidate-selection threshold")
    payload = {
        "source": "candidate-data-bls",
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


def auto_vet_candidate(
    candidate: CandidateWorkspace,
    sectors: Optional[Sequence[int]] = None,
    n_draws: int = 500,
    fit_samples: int = 3000,
    download: bool = True,
) -> Path:
    """Run a bounded evidence chain and write an engine-run manifest.

    Individual failures are retained in the manifest and do not cause later,
    independent diagnostics to be skipped. The final TRICERATOPS call continues
    to enforce ``require_vetting_readiness`` internally.
    """
    if isinstance(n_draws, bool) or int(n_draws) < 1:
        raise ValueError("n_draws must be at least one")
    if isinstance(fit_samples, bool) or int(fit_samples) < 1:
        raise ValueError("fit_samples must be at least one")
    run_id = uuid.uuid4().hex
    run_dir = candidate.path / "runs" / "auto-vet" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = _timestamp()
    artifacts: List[Dict[str, str]] = []
    steps: List[Dict[str, str]] = []

    def execute(name: str, operation: Callable[[], Any]) -> Optional[Any]:
        try:
            result = operation()
            paths = result if isinstance(result, (list, tuple)) else [result]
            for output in paths:
                if isinstance(output, Path) and output.is_file():
                    artifacts.append(_artifact(candidate, output, name))
            steps.append({"name": name, "status": "succeeded"})
            return result
        except Exception as exc:  # Preserve later independent diagnostic attempts.
            steps.append({"name": name, "status": "blocked", "detail": "{0}: {1}".format(type(exc).__name__, exc)})
            return None

    if download and not _has_raw_fits(candidate):
        def ingest() -> List[Path]:
            from .ingest import fetch_tess_products, fetch_tess_tpfs, ingest_products
            import lightkurve as lk

            tic = candidate.metadata["identifiers"].get("tic")
            sr_lc = lk.search_lightcurve(f"TIC {tic}", author="SPOC")
            sr_tp = lk.search_targetpixelfile(f"TIC {tic}", author="SPOC")
            if not sr_lc or not sr_tp:
                raise RuntimeError("MAST returned no requested SPOC products")
            common = sorted(list(set(sr_lc.table['sequence_number']).intersection(set(sr_tp.table['sequence_number']))))
            if not common:
                raise RuntimeError("No common LC and TPF sectors for target")
            use_sectors = list(sectors) if sectors else [common[0]]

            products = fetch_tess_products(candidate, sectors=use_sectors, provider="spoc")
            products.extend(fetch_tess_tpfs(candidate, sectors=use_sectors, provider="spoc"))
            if not products:
                raise RuntimeError("MAST returned no requested SPOC products")
            return ingest_products(candidate, products, fetched_by="exonym-auto-vet/1.2.0")

        execute("ingest", ingest)
    else:
        steps.append({"name": "ingest", "status": "skipped", "detail": "candidate-local raw FITS products already exist"})

    def search() -> Path:
        from .search import run_bls_on_candidate

        result_path = run_bls_on_candidate(candidate, sectors=sectors)
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
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": candidate.candidate_id,
        "engine": "auto-vet",
        "run_id": run_id,
        "status": "succeeded" if succeeded else "blocked",
        "started_at": started_at,
        "completed_at": _timestamp(),
        "runtime": {"kind": "direct", "version": "1.2.0", "executable": "exonym.autonomous"},
        "inputs": [_artifact(candidate, candidate.path / "candidate.json", "candidate-metadata")],
        "outputs": artifacts,
        "automation": {
            "steps": steps,
            "claim_eligible": False,
            "disposition_changed": False,
            "workflow_advanced": False,
        },
    }
    if not succeeded:
        detail = vet_step.get("detail") if vet_step is not None else "vet step was not attempted"
        manifest["failure"] = {"code": "vetting-not-complete", "message": str(detail)}
    manifest_path = run_dir / "engine-run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
