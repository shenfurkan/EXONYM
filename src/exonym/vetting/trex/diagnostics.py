"""Decision-tree diagnostics for the TREX statistical vetting engine.

Produces human-interpretable validation explanations from Monte Carlo
evidence output.  All outputs are routed to human review — this is a
diagnostic aid, never a scientific claim.

References
----------
* Giacalone et al. (2021, AJ, 161, 24), Section 4.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .marginal_likelihoods import (
    SCENARIO_TP, SCENARIO_PTP, SCENARIO_DTP,
    N_TARGET_SCENARIOS, _SCENARIO_NAMES, compute_fpp_nfpp,
)

FPP_THRESHOLD: float = 0.015
NFPP_THRESHOLD: float = 0.001


@dataclass
class TrexDiagnostic:
    """A single diagnostic check result."""
    name: str
    status: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class TrexResult:
    """Complete TREX vetting result with diagnostics.

    Attributes:
        fpp: False-positive probability.
        nfpp: Nearby false-positive probability.
        probs: Per-scenario probability vector.
        status: Normalization status.
        diagnostics: List of diagnostic messages.
        claim_eligible: Always False (scientific guardrail).
        degenerate: True if numerically degenerate.
    """
    fpp: Optional[float] = None
    nfpp: Optional[float] = None
    probs: Optional[np.ndarray] = None
    status: str = "not-run"
    diagnostics: List[TrexDiagnostic] = field(default_factory=list)
    claim_eligible: bool = False
    claim_block_reason: str = (
        "FPP claim creation is disabled until provenance-bound observed "
        "photometry and calibrated scene constraints are integrated."
    )
    degenerate: bool = False

    def top_scenarios(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top-n scenario names and probabilities."""
        if self.probs is None or len(self.probs) == 0:
            return []
        order = np.argsort(self.probs)[::-1]
        results = []
        for idx in order[:n]:
            prob = float(self.probs[idx])
            if prob <= 0.0:
                break
            name = _SCENARIO_NAMES.get(idx, f"N{idx}")
            results.append((name, prob))
        return results


def generate_diagnostics(lnZ: np.ndarray, verbose: bool = False) -> TrexResult:
    """Generate diagnostic interpretations from scenario log-evidences.

    Args:
        lnZ: Log-evidence array per scenario.
        verbose: Include per-scenario probability breakdown.

    Returns:
        TrexResult with diagnostics and FPP/NFPP.
    """
    result = TrexResult()
    fpp, nfpp, probs, norm_status = compute_fpp_nfpp(lnZ)
    result.fpp = fpp if np.isfinite(fpp) else None
    result.nfpp = nfpp if np.isfinite(nfpp) else None
    result.probs = probs
    result.status = norm_status

    diags: List[TrexDiagnostic] = []

    if norm_status == "anomaly":
        result.degenerate = True
        diags.append(TrexDiagnostic(
            name="numerical-stability", status="fail",
            message="Numerical anomaly (NaN/+inf in log-evidences).",
        ))
    elif norm_status == "all_neginf":
        result.degenerate = True
        diags.append(TrexDiagnostic(
            name="all-neginf", status="fail",
            message="All lnZ are -inf. FPP=1.0 is a failed computation.",
        ))

    if result.degenerate:
        result.diagnostics = diags
        return result

    # FPP check
    if result.fpp is not None:
        fpp_ok = result.fpp < FPP_THRESHOLD
        diags.append(TrexDiagnostic(
            name="fpp", status="pass" if fpp_ok else "fail",
            message=f"FPP = {result.fpp:.6f} {'<' if fpp_ok else '>='} {FPP_THRESHOLD}",
            value=result.fpp, threshold=FPP_THRESHOLD,
        ))

    # NFPP check
    if result.nfpp is not None:
        nfpp_ok = result.nfpp < NFPP_THRESHOLD
        diags.append(TrexDiagnostic(
            name="nfpp", status="pass" if nfpp_ok else "fail",
            message=f"NFPP = {result.nfpp:.6f} {'<' if nfpp_ok else '>='} {NFPP_THRESHOLD}",
            value=result.nfpp, threshold=NFPP_THRESHOLD,
        ))

    # Top scenario breakdown
    if verbose and result.probs is not None:
        for name, prob in result.top_scenarios(5):
            diags.append(TrexDiagnostic(
                name=f"scenario-{name}", status="info",
                message=f"P({name}) = {prob:.4f}", value=prob,
            ))

    # Overall verdict
    overall = (
        result.fpp is not None and result.nfpp is not None
        and result.fpp < FPP_THRESHOLD and result.nfpp < NFPP_THRESHOLD
    )
    diags.append(TrexDiagnostic(
        name="overall", status="pass" if overall else "fail",
        message=(
            "TREX validation PASS" if overall else "TREX validation FAIL"
        ),
    ))

    result.diagnostics = diags
    return result


__all__ = [
    "TrexDiagnostic", "TrexResult", "generate_diagnostics",
    "FPP_THRESHOLD", "NFPP_THRESHOLD",
]