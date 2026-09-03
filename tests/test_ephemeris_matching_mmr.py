"""Target-neutral unit tests for first-order Mean Motion Resonance (MMR) ephemeris matching."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from exonym.ephemeris_matching import (
    HARMONIC_FACTORS,
    match_known_signal_ephemerides,
)
from exonym.workspace import CandidateWorkspace


def _candidate(tmp_path: Path) -> CandidateWorkspace:
    path = tmp_path / "candidate" / "test-target"
    path.mkdir(parents=True, exist_ok=True)
    return CandidateWorkspace(path)


def _write_candidate_ephemeris(
    candidate: CandidateWorkspace, period_days: float = 10.0, epoch_btjd: float = 100.0, duration_hours: float = 2.0
) -> None:
    config_dir = candidate.path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "candidate-config",
        "time_system": "BTJD_TDB",
        "period_days": period_days,
        "period_uncertainty_days": 0.01,
        "epoch_btjd": epoch_btjd,
        "duration_days": duration_hours / 24.0,
        "field_sources": {
            "period_days": "manual-prior",
            "epoch_btjd": "manual-prior",
            "duration_days": "manual-prior",
        },
    }
    (config_dir / "transit_config.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_mock_snapshot(
    candidate: CandidateWorkspace,
    known_period: float,
    known_epoch_bjd: float = 2457100.0,
    known_duration_hours: float = 2.0,
    name: str = "Resonant Planet",
) -> None:
    run_dir = candidate.path / "runs" / "catalog" / "nasa-exoplanet-archive" / "mock-retrieval"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_payload = {
        "candidate_id": candidate.candidate_id,
        "provider": "nasa-exoplanet-archive",
        "retrieval_id": "mock-retrieval",
        "status": "available",
        "records": [
            {
                "pl_orbper": known_period,
                "pl_orbpererr1": 0.01,
                "pl_orbpererr2": -0.01,
                "pl_tranmid": known_epoch_bjd,
                "pl_trandur": known_duration_hours,
                "pl_tranmid_systemref": "BJD_TDB",
                "pl_name": name,
            }
        ],
    }
    (run_dir / "snapshot.json").write_text(json.dumps(snapshot_payload), encoding="utf-8")

    manifest_payload = {
        "candidate_id": candidate.candidate_id,
        "provider": "nasa-exoplanet-archive",
        "retrieval_id": "mock-retrieval",
        "status": "available",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "expires_at": "2030-01-01T00:00:00Z",
    }
    (run_dir / "query-manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")


@pytest.mark.parametrize(
    "ratio,expected_factor",
    [
        (float(Fraction(3, 2)), float(Fraction(3, 2))),  # 3:2 MMR
        (float(Fraction(4, 3)), float(Fraction(4, 3))),  # 4:3 MMR
        (float(Fraction(5, 4)), float(Fraction(5, 4))),  # 5:4 MMR
        (float(Fraction(2, 3)), float(Fraction(2, 3))),  # 2:3 MMR
        (float(Fraction(3, 4)), float(Fraction(3, 4))),  # 3:4 MMR
        (float(Fraction(4, 5)), float(Fraction(4, 5))),  # 4:5 MMR
    ],
)
def test_first_order_mmr_detection(tmp_path, ratio, expected_factor):
    """Verify that 1st-order MMR period ratios are recognized and flagged as review-required."""
    cand = _candidate(tmp_path / f"cand_{ratio}")
    cand_period = 10.0
    _write_candidate_ephemeris(cand, period_days=cand_period, epoch_btjd=100.0)
    
    known_period = cand_period * ratio
    _write_mock_snapshot(cand, known_period=known_period, known_epoch_bjd=2457100.0)
    
    result_path = match_known_signal_ephemerides(cand)
    record = json.loads(result_path.read_text(encoding="utf-8"))
    
    assert len(record["comparisons"]) == 1
    comp = record["comparisons"][0]
    assert comp["period_harmonic_match"] is True
    assert comp["nearest_harmonic_factor"] == expected_factor
    assert comp["review_required"] is True
    assert record["status"] in ("review-required-known-signal-match", "review-required-period-harmonic")


def test_higher_order_resonances_are_not_allowed_comparison_factors():
    assert 3.0 not in HARMONIC_FACTORS
    assert 1.0 / 3.0 not in HARMONIC_FACTORS


def test_missing_period_uncertainty_requires_review(tmp_path):
    cand = _candidate(tmp_path / "cand_missing_uncertainty")
    _write_candidate_ephemeris(cand)
    _write_mock_snapshot(cand, known_period=10.0)
    snapshot_path = cand.path / "runs" / "catalog" / "nasa-exoplanet-archive" / "mock-retrieval" / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["records"][0].pop("pl_orbpererr1")
    snapshot["records"][0].pop("pl_orbpererr2")
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    record = json.loads(match_known_signal_ephemerides(cand).read_text(encoding="utf-8"))

    comparison = record["comparisons"][0]
    assert comparison["period_harmonic_match"] is None
    assert comparison["period_tolerance_status"] == "unavailable"
    assert comparison["review_required"] is True
    assert record["status"] == "review-required-period-tolerance-unavailable"


def test_non_resonant_period_is_not_matched(tmp_path):
    """Verify that an arbitrary non-resonant ratio (e.g. 1.111) reports no harmonic match."""
    cand = _candidate(tmp_path / "cand_non_resonant")
    _write_candidate_ephemeris(cand, period_days=10.0, epoch_btjd=100.0)
    
    # 11.111 days -> ratio 1.1111 (not near 1.0, 1.25, or any MMR)
    _write_mock_snapshot(cand, known_period=11.111, known_epoch_bjd=2457100.0)
    
    result_path = match_known_signal_ephemerides(cand)
    record = json.loads(result_path.read_text(encoding="utf-8"))
    
    assert len(record["comparisons"]) == 1
    comp = record["comparisons"][0]
    assert comp["period_harmonic_match"] is False
    assert record["status"] == "no-ephemeris-match-in-current-supported-catalog"
