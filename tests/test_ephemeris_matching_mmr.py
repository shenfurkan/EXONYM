"""Observational benchmark tests for known-signal ephemeris / MMR matching.

Every period, epoch, duration, mass, and radius used here is a published
NASA Exoplanet Archive ``pscomppars`` composite parameter (default rows)
retained with full retrieval provenance in
``tests/fixtures/nasa_exoplanet_archive_benchmarks.json``.  No synthetic
"toy planet" vectors exist in this module.

Resonance proximity is grounded in the resonant-repulsion width of
Lithwick & Wu (2012, ApJ, 756, L11, Eq. 12; doi:10.1088/2041-8205/756/1/L11;
retained PDF: ``literature/lithwick_wu_2012_resonant_repulsion.pdf``).  The
Kepler-9 pair (Holman et al. 2010, Science, 330, 51; ADS 2010Sci...330...51H)
is the empirical 2:1 near-resonance benchmark; Kepler-36 (7:6, high-order)
and Kepler-20 (non-resonant) are the exclusion controls.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict

import pytest
from astropy.constants import M_earth, M_jup, R_earth, R_jup

from exonym.ephemeris_matching import (
    HARMONIC_FACTORS,
    lithwick_wu_repulsion_width_fractional,
    match_known_signal_ephemerides,
)
from exonym.workspace import CandidateWorkspace

BENCHMARKS_PATH = Path(__file__).parent / "fixtures" / "nasa_exoplanet_archive_benchmarks.json"
BTJD_OFFSET_DAYS = 2457000.0


def _systems() -> Dict[str, Any]:
    return json.loads(BENCHMARKS_PATH.read_text(encoding="utf-8"))["systems"]


def _candidate(tmp_path: Path, name: str) -> CandidateWorkspace:
    path = tmp_path / "candidate" / name
    path.mkdir(parents=True, exist_ok=True)
    return CandidateWorkspace(path)


def _write_candidate_ephemeris(
    candidate: CandidateWorkspace, row: Dict[str, Any], with_uncertainty: bool = True
) -> None:
    """Write a candidate transit configuration from one archive benchmark row."""
    config_dir = candidate.path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "candidate-config",
        "time_system": "BTJD_TDB",
        "period_days": row["pl_orbper"],
        "epoch_btjd": row["pl_tranmid"] - BTJD_OFFSET_DAYS,
        "duration_days": row["pl_trandur"] / 24.0,
    }
    if with_uncertainty:
        payload["period_uncertainty_days"] = abs(row["pl_orbpererr1"])
    (config_dir / "transit_config.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_archive_snapshot(
    candidate: CandidateWorkspace, row: Dict[str, Any], with_uncertainty: bool = True
) -> None:
    """Retain one archive benchmark row as a fresh supported-provider snapshot."""
    run_dir = candidate.path / "runs" / "catalog" / "nasa-exoplanet-archive" / "benchmark-retrieval"
    run_dir.mkdir(parents=True, exist_ok=True)
    record = dict(row)
    if not with_uncertainty:
        record.pop("pl_orbpererr1", None)
        record.pop("pl_orbpererr2", None)
    snapshot_payload = {
        "candidate_id": candidate.candidate_id,
        "provider": "nasa-exoplanet-archive",
        "retrieval_id": "benchmark-retrieval",
        "status": "available",
        "records": [record],
    }
    (run_dir / "snapshot.json").write_text(json.dumps(snapshot_payload), encoding="utf-8")
    manifest_payload = {
        "candidate_id": candidate.candidate_id,
        "provider": "nasa-exoplanet-archive",
        "retrieval_id": "benchmark-retrieval",
        "status": "available",
        "retrieved_at": "2026-09-03T00:00:00Z",
        "expires_at": "2030-01-01T00:00:00Z",
    }
    (run_dir / "query-manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")


def test_kepler9_near_2_to_1_resonance_is_flagged(tmp_path):
    """The empirical Kepler-9 b/c pair sits ~1.3% wide of the 2:1 commensurability.

    Measurement uncertainties cannot bridge that offset, so the exact-harmonic
    test fails; the Lithwick-Wu resonant-repulsion width must flag the pair for
    review instead.
    """
    systems = _systems()
    cand = _candidate(tmp_path, "kepler9")
    _write_candidate_ephemeris(cand, systems["kepler-9"]["b"])
    _write_archive_snapshot(cand, systems["kepler-9"]["c"])

    record = json.loads(match_known_signal_ephemerides(cand).read_text(encoding="utf-8"))

    assert len(record["comparisons"]) == 1
    comp = record["comparisons"][0]
    assert comp["nearest_harmonic_factor"] == float(Fraction(2, 1))
    assert comp["period_harmonic_match"] is False
    assert comp["period_tolerance_status"] == "available"
    assert comp["period_near_resonance"] is True
    assert comp["resonance_width_fractional"] > comp["period_relative_difference"]
    assert comp["review_required"] is True
    assert record["status"] == "review-required-period-harmonic"


def test_kepler9_reverse_orientation_near_1_to_2_is_flagged(tmp_path):
    """The same pair read with the candidate as the outer member (ratio ~1/2)."""
    systems = _systems()
    cand = _candidate(tmp_path, "kepler9-reverse")
    _write_candidate_ephemeris(cand, systems["kepler-9"]["c"])
    _write_archive_snapshot(cand, systems["kepler-9"]["b"])

    record = json.loads(match_known_signal_ephemerides(cand).read_text(encoding="utf-8"))

    comp = record["comparisons"][0]
    assert comp["nearest_harmonic_factor"] == float(Fraction(1, 2))
    assert comp["period_harmonic_match"] is False
    assert comp["period_near_resonance"] is True
    assert comp["review_required"] is True
    assert record["status"] == "review-required-period-harmonic"


def test_lithwick_wu_repulsion_width_pins_published_equation():
    """The width helper reproduces Lithwick & Wu (2012) Eq. (12) verbatim.

    With the paper's fiducial factors (Q1 = 10, k2 = 0.1, t = 5 Gyr, equal-mass
    beta = 1) and no reported planet parameters, the width must equal the
    analytic evaluation of the published equation.  The Kepler-9 archive
    parameters must then place the observed 2:1 offset inside the envelope,
    while the Kepler-36 (nearest first-order 5:4) and Kepler-20 (nearest 2:1)
    offsets stay outside theirs.
    """
    systems = _systems()
    kepler9_b = systems["kepler-9"]["b"]
    kepler9_c = systems["kepler-9"]["c"]
    inner_period = kepler9_b["pl_orbper"]
    outer_period = kepler9_c["pl_orbper"]

    expected_fallback = (
        0.006
        * (inner_period / 5.0) ** (-13.0 / 9.0)
        * kepler9_b["st_mass"] ** (-8.0 / 3.0)
        * (2.0 * 1.0 + 2.0 * 1.0**2) ** (1.0 / 3.0)
    )
    fallback = lithwick_wu_repulsion_width_fractional(
        inner_period, outer_period, star_mass_solar=kepler9_b["st_mass"]
    )
    assert fallback == pytest.approx(expected_fallback, rel=1e-12)

    kepler9_width = lithwick_wu_repulsion_width_fractional(
        inner_period,
        outer_period,
        mass_inner_earth=kepler9_b["pl_bmassj"] * float((M_jup / M_earth).value),
        radius_inner_earth=kepler9_b["pl_radj"] * float((R_jup / R_earth).value),
        star_mass_solar=kepler9_b["st_mass"],
    )
    kepler9_delta = outer_period / (2.0 * inner_period) - 1.0
    assert 0.0 < kepler9_delta < kepler9_width

    kepler36_b = systems["kepler-36"]["b"]
    kepler36_c = systems["kepler-36"]["c"]
    kepler36_width = lithwick_wu_repulsion_width_fractional(
        kepler36_b["pl_orbper"],
        kepler36_c["pl_orbper"],
        mass_inner_earth=kepler36_b["pl_bmassj"] * float((M_jup / M_earth).value),
        radius_inner_earth=kepler36_b["pl_radj"] * float((R_jup / R_earth).value),
        star_mass_solar=kepler36_b["st_mass"],
    )
    kepler36_delta = (
        abs(kepler36_c["pl_orbper"] / kepler36_b["pl_orbper"] - float(Fraction(5, 4)))
        / float(Fraction(5, 4))
    )
    assert kepler36_width < kepler36_delta

    kepler20_b = systems["kepler-20"]["b"]
    kepler20_c = systems["kepler-20"]["c"]
    kepler20_width = lithwick_wu_repulsion_width_fractional(
        kepler20_b["pl_orbper"],
        kepler20_c["pl_orbper"],
        mass_inner_earth=kepler20_b["pl_bmassj"] * float((M_jup / M_earth).value),
        radius_inner_earth=kepler20_b["pl_radj"] * float((R_jup / R_earth).value),
        star_mass_solar=kepler20_b["st_mass"],
    )
    kepler20_delta = abs(kepler20_c["pl_orbper"] / kepler20_b["pl_orbper"] - 2.0) / 2.0
    assert kepler20_width < kepler20_delta


def test_kepler36_high_order_7_to_6_is_not_promoted(tmp_path):
    """Kepler-36 b/c lie near the 7:6 high-order resonance.

    The engine's factor set is strictly first-order, so the pair must not be
    flagged as a harmonic or near-resonance match.
    """
    systems = _systems()
    cand = _candidate(tmp_path, "kepler36")
    _write_candidate_ephemeris(cand, systems["kepler-36"]["b"])
    _write_archive_snapshot(cand, systems["kepler-36"]["c"])

    record = json.loads(match_known_signal_ephemerides(cand).read_text(encoding="utf-8"))

    comp = record["comparisons"][0]
    assert comp["period_harmonic_match"] is False
    assert comp["period_near_resonance"] is False
    assert comp["review_required"] is False
    assert record["status"] == "no-ephemeris-match-in-current-supported-catalog"


def test_kepler20_non_resonant_control_is_not_matched(tmp_path):
    """Kepler-20 b/c (ratio ~2.9367) is far from every first-order factor."""
    systems = _systems()
    cand = _candidate(tmp_path, "kepler20")
    _write_candidate_ephemeris(cand, systems["kepler-20"]["b"])
    _write_archive_snapshot(cand, systems["kepler-20"]["c"])

    record = json.loads(match_known_signal_ephemerides(cand).read_text(encoding="utf-8"))

    comp = record["comparisons"][0]
    assert comp["period_harmonic_match"] is False
    assert comp["period_near_resonance"] is False
    assert comp["review_required"] is False
    assert record["status"] == "no-ephemeris-match-in-current-supported-catalog"


def test_missing_period_uncertainty_requires_review(tmp_path):
    """An exact-period archive row without reported errors stays review-required."""
    systems = _systems()
    cand = _candidate(tmp_path, "kepler9-no-uncertainty")
    _write_candidate_ephemeris(cand, systems["kepler-9"]["b"])
    _write_archive_snapshot(cand, systems["kepler-9"]["b"], with_uncertainty=False)

    record = json.loads(match_known_signal_ephemerides(cand).read_text(encoding="utf-8"))

    comparison = record["comparisons"][0]
    assert comparison["period_harmonic_match"] is None
    assert comparison["period_tolerance_status"] == "unavailable"
    assert comparison["review_required"] is True
    assert record["status"] == "review-required-period-tolerance-unavailable"


def test_harmonic_factors_are_exact_first_order_rationals():
    """The factor set is exactly the first-order j:(j-1) family and reciprocals."""
    expected = {float(Fraction(1, 1))}
    for j in range(2, 6):
        expected.add(float(Fraction(j, j - 1)))
        expected.add(float(Fraction(j - 1, j)))
    assert set(HARMONIC_FACTORS) == expected
    assert 3.0 not in HARMONIC_FACTORS
    assert float(Fraction(1, 3)) not in HARMONIC_FACTORS
    assert float(Fraction(7, 6)) not in HARMONIC_FACTORS
    assert float(Fraction(6, 7)) not in HARMONIC_FACTORS
