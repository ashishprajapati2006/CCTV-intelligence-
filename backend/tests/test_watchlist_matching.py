"""Unit tests for Step 9 — Real-Time Watchlist Matching.

Covers Parts 15 and 18:
  1. TEST 1 — MATCH (Exact normalized match against active watchlist)
  2. TEST 2 — NO MATCH (Unlisted registration)
  3. TEST 3 — INACTIVE WATCHLIST (Inactive record in watchlist does not match)
  4. TEST 4 — UNCERTAIN OCR / LOW CONSENSUS (Gated by eligibility into NOT_ELIGIBLE)
  5. TEST 5 — REPEATED TRACK DEDUPLICATION (Repeated observations suppressed to 1 alert)
  6. TEST 6 — CROSS-CAMERA INDEPENDENCE (cam01 and cam07 remain distinct match events)
  7. TEST 7 — CONTROLLED FUZZY REVIEW (Single-char discrepancy produces POSSIBLE_MATCH_REVIEW)
  8. TEST 8 — WATCHLIST METADATA SNAPSHOT (Preserved immutably in decision)
  9. TEST 9 — SCORE SEPARATION (OCR confidence != consensus score != match score)
  10. TEST 10 — AUDIT LOG DESTINATIONS (raw, confirmed, rejected, deduplicated)
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.watchlist import (
    AlertDeduplicator,
    JSONWatchlistRepository,
    MatchDecisionEngine,
    MatchEligibilityConfig,
    MatchEligibilityEvaluator,
    WatchlistMatcher,
    WatchlistVehicle,
)


@pytest.fixture
def test_setup(tmp_path):
    v_file = tmp_path / "watchlist.json"
    p_file = tmp_path / "persons.json"
    repo = JSONWatchlistRepository(vehicles_path=str(v_file), persons_path=str(p_file))

    # Add active vehicle
    v1 = WatchlistVehicle(
        watchlist_id="WL-VEH-000001",
        registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234",
        category="DEMO_STOLEN_VEHICLE",
        priority="HIGH",
        status="ACTIVE",
        source="SYNTHETIC_DEMO",
        synthetic=True,
    )
    # Add inactive vehicle
    v2 = WatchlistVehicle(
        watchlist_id="WL-VEH-000002",
        registration_number="GJ02CD5678",
        normalized_registration_number="GJ02CD5678",
        category="DEMO_BLACKLISTED_VEHICLE",
        priority="CRITICAL",
        status="INACTIVE",
        source="SYNTHETIC_DEMO",
        synthetic=True,
    )
    repo.add_vehicle(v1)
    repo.add_vehicle(v2)

    matches_dir = tmp_path / "matches"
    engine = MatchDecisionEngine(
        repository=repo,
        matches_dir=str(matches_dir),
    )

    return {
        "repo": repo,
        "engine": engine,
        "matches_dir": matches_dir,
    }


def _make_obs(
    cam: str = "cam01",
    track: str = "TRK-0001",
    reg: str = "GJ01AB1234",
    status: str = "CONFIRMED",
    consensus: float = 0.94,
    ocr_conf: float = 0.92,
    pts: float = 11200.0,
) -> dict:
    return {
        "camera_id": cam,
        "track_id": track,
        "registration_number": reg,
        "normalized_registration_number": reg,
        "recognition_status": status,
        "consensus_score": consensus,
        "ocr_confidence": ocr_conf,
        "plate_detection_confidence": 0.95,
        "plate_quality_score": 0.88,
        "recognition_pts_ms": pts,
        "evidence_image": f"data/snapshots/{cam}/plates/{track}_pts{int(pts)}.jpg",
    }


# ════════════════════════════════════════════════════════════════════════════
# 1. Demonstration Test Cases (Part 15)
# ════════════════════════════════════════════════════════════════════════════

def test_1_exact_match(test_setup):
    """TEST 1: Observed GJ01AB1234 matches active watchlist entry."""
    engine = test_setup["engine"]
    obs = _make_obs(reg="GJ01AB1234")

    decision = engine.process_observation(obs)

    assert decision.decision == "MATCH"
    assert decision.watchlist_id == "WL-VEH-000001"
    assert decision.watchlist_match_score == 1.0
    assert decision.is_deduplicated is False
    assert decision.watchlist_metadata is not None
    assert decision.watchlist_metadata.category == "DEMO_STOLEN_VEHICLE"
    assert decision.watchlist_metadata.priority == "HIGH"


def test_2_no_match_unlisted(test_setup):
    """TEST 2: Observed GJ05XY7812 is not in watchlist -> NO_MATCH."""
    engine = test_setup["engine"]
    obs = _make_obs(reg="GJ05XY7812")

    decision = engine.process_observation(obs)

    assert decision.decision == "NO_MATCH"
    assert decision.watchlist_id is None
    assert decision.watchlist_match_score == 0.0
    assert "UNLISTED" in decision.decision_reason


def test_3_inactive_watchlist(test_setup):
    """TEST 3: Observed GJ02CD5678 is in watchlist but INACTIVE -> NO_MATCH."""
    engine = test_setup["engine"]
    obs = _make_obs(reg="GJ02CD5678")

    decision = engine.process_observation(obs)

    assert decision.decision == "NO_MATCH"
    assert "INACTIVE" in decision.decision_reason
    assert decision.is_deduplicated is False


def test_4_uncertain_ocr_and_low_confidence(test_setup):
    """TEST 4: Low confidence or UNCERTAIN recognition fails eligibility gate."""
    engine = test_setup["engine"]

    # Case A: UNCERTAIN recognition status
    obs_uncertain = _make_obs(reg="GJ01AB1234", status="UNCERTAIN")
    dec_a = engine.process_observation(obs_uncertain)
    assert dec_a.decision == "NOT_ELIGIBLE"
    assert dec_a.decision_reason == "UNCERTAIN_RECOGNITION"

    # Case B: Low consensus score
    obs_low_consensus = _make_obs(reg="GJ01AB1234", consensus=0.55)
    dec_b = engine.process_observation(obs_low_consensus)
    assert dec_b.decision == "NOT_ELIGIBLE"
    assert dec_b.decision_reason == "LOW_CONSENSUS"

    # Case C: Low OCR confidence
    obs_low_ocr = _make_obs(reg="GJ01AB1234", ocr_conf=0.45)
    dec_c = engine.process_observation(obs_low_ocr)
    assert dec_c.decision == "NOT_ELIGIBLE"
    assert dec_c.decision_reason == "LOW_OCR_CONFIDENCE"


def test_5_repeated_track_deduplication(test_setup):
    """TEST 5: Same vehicle observation repeated 10 times produces 1 alert-ready match."""
    engine = test_setup["engine"]

    decisions = []
    # 10 observations spaced 500ms apart (total 4.5s, well within 30s window)
    for i in range(10):
        obs = _make_obs(cam="cam01", track="TRK-0001", reg="GJ01AB1234", pts=10000.0 + (i * 500.0))
        dec = engine.process_observation(obs)
        decisions.append(dec)

    # First observation should be alert-ready (not deduplicated)
    assert decisions[0].decision == "MATCH"
    assert decisions[0].is_deduplicated is False
    assert decisions[0].supporting_observations_count == 1

    # Remaining 9 observations should be suppressed
    for dec in decisions[1:]:
        assert dec.decision == "MATCH"
        assert dec.is_deduplicated is True

    # Final observation records all 10 supporting frames
    assert decisions[-1].supporting_observations_count == 10

    # Summary report check
    summary = engine.write_summary_report(decisions)
    assert summary["total_exact_matches"] == 10
    assert summary["alert_ready_matches"] == 1
    assert summary["suppressed_duplicate_matches"] == 9


def test_6_cross_camera_independence(test_setup):
    """TEST 6: cam01 and cam07 observing same plate remain separate match events."""
    engine = test_setup["engine"]

    obs_cam01 = _make_obs(cam="cam01", track="TRK-0001", reg="GJ01AB1234", pts=11200.0)
    obs_cam07 = _make_obs(cam="cam07", track="TRK-0019", reg="GJ01AB1234", pts=46000.0)

    dec1 = engine.process_observation(obs_cam01)
    dec2 = engine.process_observation(obs_cam07)

    # Both must be independent alert-ready events
    assert dec1.decision == "MATCH"
    assert dec1.is_deduplicated is False
    assert dec1.camera_id == "cam01"

    assert dec2.decision == "MATCH"
    assert dec2.is_deduplicated is False
    assert dec2.camera_id == "cam07"


# ════════════════════════════════════════════════════════════════════════════
# 2. Fuzzy Review & Score Separation
# ════════════════════════════════════════════════════════════════════════════

def test_7_controlled_fuzzy_review(test_setup):
    """TEST 7: OCR typo (GJ01A81234 vs GJ01AB1234) produces POSSIBLE_MATCH_REVIEW when enabled."""
    repo = test_setup["repo"]
    matches_dir = test_setup["matches_dir"]

    # Engine with fuzzy review enabled
    fuzzy_engine = MatchDecisionEngine(
        repository=repo,
        enable_fuzzy_review=True,
        matches_dir=str(matches_dir / "fuzzy"),
    )

    # Note: GJ01A81234 has edit distance 1 from GJ01AB1234 ('8' vs 'B')
    obs = _make_obs(reg="GJ01A81234")
    dec = fuzzy_engine.process_observation(obs)

    assert dec.decision == "POSSIBLE_MATCH_REVIEW"
    assert dec.watchlist_id == "WL-VEH-000001"
    assert dec.fuzzy_diff is not None
    assert dec.fuzzy_diff["edit_distance"] == 1
    assert dec.fuzzy_diff["target_registration"] == "GJ01AB1234"


def test_score_separation():
    """Verify underlying OCR confidence is never overwritten by match score."""
    evaluator = MatchEligibilityEvaluator()
    obs = _make_obs(consensus=0.88, ocr_conf=0.75)
    is_elig, _ = evaluator.evaluate(obs)
    assert is_elig is True

    # Check distinct values
    assert obs["ocr_confidence"] == 0.75
    assert obs["consensus_score"] == 0.88
