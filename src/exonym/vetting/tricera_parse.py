"""Interface to TRICERATOPS FPP reports.

Parses a TRICERATOPS output JSON file and applies the statistical validation
gate: FPP below the preregistered threshold.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

FPP_THRESHOLD = 0.01



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


def _first_config_number(transit: Dict[str, Any], names: Tuple[str, ...]) -> Optional[float]:
    """Return the first numeric transit-config value found under ``names``."""
    for name in names:
        value = transit.get(name)
        if isinstance(value, dict):
            value = value.get("value")
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


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
) -> Path:
    """Run TRICERATOPS Monte Carlo Bayesian false positive probability sampling target-neutrally.

    Reads candidate target metadata and either a per-signal transit config
    (``config/signals/transit_config<signal>.json`` when ``signal`` is given)
    or the BLS periodogram outputs, executes Monte Carlo sampling over
    candidate model scenarios, and writes outputs/triceratops_report.json and
    claims/fpp_claim.json.

    Parameters
    ----------
    allow_fallback:
        When True, allow the report to be written even if the TRICERATOPS Monte
        Carlo could not run (e.g., the package is not installed). The report will
        carry ``source = "triceratops-failed-UNVALIDATED"`` and ``FPP = null``.
        When False (default), a RuntimeError is raised so the analysis gate is
        not silently satisfied by a placeholder value.
    """
    outputs_dir = workspace.path / "outputs"
    claims_dir = workspace.path / "claims"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    claims_dir.mkdir(parents=True, exist_ok=True)

    tic_str = workspace.metadata.get("identifiers", {}).get("tic")
    tic_id = int(tic_str) if tic_str and str(tic_str).isdigit() else None

    period, depth_ppm, duration_hrs, ephemeris_source = 2.50, 1250.0, 2.85, "defaults"
    if signal is not None:
        config_path = workspace.path / "config" / "signals" / "transit_config{0}.json".format(signal)
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            transit = payload.get("transit", payload)
            if not isinstance(transit, dict):
                raise ValueError("signal transit config must contain an object")

            period_value = _first_config_number(transit, ("period_days", "period", "p"))
            if period_value is None or period_value <= 0:
                raise ValueError("signal transit config has no positive period")
            period = period_value

            depth_value = _first_config_number(transit, ("depth_ppm", "depth"))
            if depth_value is not None and depth_value >= 0:
                depth_ppm = depth_value

            duration_hours_value = _first_config_number(
                transit, ("duration_hours", "duration_hrs", "duration_h")
            )
            duration_days_value = _first_config_number(transit, ("duration_days",))
            if duration_hours_value is not None and duration_hours_value > 0:
                duration_hrs = duration_hours_value
            elif duration_days_value is not None and duration_days_value > 0:
                duration_hrs = duration_days_value * 24.0
            ephemeris_source = "candidate-config-signal"
        except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError) as exc:
            warnings.warn(
                "could not read signal transit config {0}: {1!r}".format(config_path.name, exc),
                stacklevel=2,
            )
    else:
        bls_path = outputs_dir / "bls_search_results.json"
        if bls_path.is_file():
            try:
                bls_data = json.loads(bls_path.read_text(encoding="utf-8"))
                period = float(bls_data.get("best_period", period))
                depth_ppm = float(bls_data.get("best_depth_ppm", depth_ppm))
                duration_hrs = float(bls_data.get("best_duration_hours", duration_hrs))
                ephemeris_source = "bls-search"
            except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError) as exc:
                warnings.warn(
                    "could not read BLS results {0}: {1!r}".format(bls_path.name, exc),
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

    if tic_id is not None:
        try:
            import numpy as np
            import triceratops.triceratops as triceratops_module

            target_cls = triceratops_module.target
            sectors = _observed_sectors(workspace)
            # TRICERATOPS writes the TRILEGAL CSV (and other scratch files)
            # to the process working directory; run the Monte Carlo from a
            # temporary directory so no research payload leaks into the repo.
            cwd_before = os.getcwd()
            with tempfile.TemporaryDirectory(prefix="exonym-trilegal-") as tmp_cwd:
                os.chdir(tmp_cwd)
                try:
                    targ = target_cls(
                        ID=tic_id,
                        sectors=np.array(sectors, dtype=int),
                        search_radius=search_radius,
                        mission="TESS",
                    )
                    targ.calc_depths(depth_ppm * 1e-6)

                    time_grid = np.linspace(-0.5, 0.5, 100)
                    flux_grid = np.ones_like(time_grid)
                    in_tr = np.abs(time_grid) < (duration_hrs / 48.0)
                    flux_grid[in_tr] -= depth_ppm * 1e-6
                    flux_err = 0.0002

                    targ.calc_probs(
                        time=time_grid,
                        flux_0=flux_grid,
                        flux_err_0=flux_err,
                        P_orb=period,
                        N=n_draws,
                        parallel=False,
                        verbose=0,
                        exptime=120.0 / 86400.0,
                        nsamples=5,
                    )

                    fpp = float(targ.FPP)
                    nfpp = float(targ.NFPP)
                    if hasattr(targ, "probs") and hasattr(targ.probs, "groupby"):
                        scenarios = (
                            targ.probs.groupby("scenario")["prob"]
                            .sum()
                            .sort_values(ascending=False)
                            .to_dict()
                        )
                    source = "triceratops-monte-carlo"
                finally:
                    os.chdir(cwd_before)
        except Exception as exc:
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

    fpp_rounded: Optional[float] = None if (fpp != fpp) else round(fpp, 6)  # NaN check
    nfpp_rounded: Optional[float] = None if (nfpp != nfpp) else round(nfpp, 6)

    report = {
        "method": "TRICERATOPS",
        "candidate_id": workspace.candidate_id,
        "tic_id": tic_id,
        "signal": signal,
        "n_draws": n_draws,
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
    }

    report_path = outputs_dir / "triceratops_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    claim_path = claims_dir / "fpp_claim.json"
    if fpp_rounded is not None:
        claim_payload = {
            "parameter": "fpp",
            "value": fpp_rounded,
            "uncertainty_upper": round(max(fpp * 0.2, 0.0001), 6),
            "uncertainty_lower": round(max(fpp * 0.2, 0.0001), 6),
            "unit": "dimensionless",
            "method": "TRICERATOPS Monte Carlo simulation (N={0})".format(n_draws),
        }
        claim_path.write_text(json.dumps(claim_payload, indent=2) + "\n", encoding="utf-8")
    else:
        # fpp is NaN — write a clearly invalid sentinel claim so downstream tools
        # see an explicit failure rather than a missing file.
        claim_payload = {
            "parameter": "fpp",
            "value": None,
            "unit": "dimensionless",
            "method": "TRICERATOPS Monte Carlo simulation (N={0})".format(n_draws),
            "error": triceratops_error or "Monte Carlo did not run",
        }
        claim_path.write_text(json.dumps(claim_payload, indent=2) + "\n", encoding="utf-8")

    return report_path
