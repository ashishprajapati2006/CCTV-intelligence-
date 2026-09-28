"""Match Decision Engine for Step 9 Real-Time Watchlist Matching.

Orchestrates:
  Observation -> Eligibility Gating -> Watchlist Matcher -> Deduplicator -> Persistence & Summary.

Directories Managed:
  data/matches/
    raw/matches.jsonl
    confirmed/matches.jsonl
    rejected/rejected.jsonl
    deduplicated/dedup_summary.json
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.logging import get_logger
from src.watchlist.deduplication import AlertDeduplicator
from src.watchlist.eligibility import MatchEligibilityConfig, MatchEligibilityEvaluator
from src.watchlist.matcher import WatchlistMatcher
from src.watchlist.models import (
    MatchDecision,
    WatchlistMetadataSnapshot,
)
from src.watchlist.normalizer import normalize_watchlist_registration
from src.watchlist.repository import WatchlistRepository

logger = get_logger("decision_engine")


class MatchDecisionEngine:
    """End-to-end decision pipeline for Step 9 watchlist matching."""

    def __init__(
        self,
        repository: WatchlistRepository,
        eligibility_config: Optional[MatchEligibilityConfig] = None,
        enable_fuzzy_review: bool = False,
        suppression_window_pts_ms: float = 30000.0,
        matches_dir: str = "data/matches",
    ):
        if eligibility_config is None:
            eligibility_config = MatchEligibilityConfig(allow_fuzzy_candidates=enable_fuzzy_review)
        elif enable_fuzzy_review and not eligibility_config.allow_fuzzy_candidates:
            eligibility_config.allow_fuzzy_candidates = True
        self.eligibility_evaluator = MatchEligibilityEvaluator(config=eligibility_config)
        self.matcher = WatchlistMatcher(repository=repository, enable_fuzzy_review=enable_fuzzy_review)
        self.deduplicator = AlertDeduplicator(suppression_window_pts_ms=suppression_window_pts_ms)

        self.matches_dir = Path(matches_dir)
        self.raw_dir = self.matches_dir / "raw"
        self.confirmed_dir = self.matches_dir / "confirmed"
        self.rejected_dir = self.matches_dir / "rejected"
        self.dedup_dir = self.matches_dir / "deduplicated"

        for d in (self.raw_dir, self.confirmed_dir, self.rejected_dir, self.dedup_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.raw_file = self.raw_dir / "matches.jsonl"
        self.confirmed_file = self.confirmed_dir / "matches.jsonl"
        self.rejected_file = self.rejected_dir / "rejected.jsonl"
        self.summary_file = self.dedup_dir / "dedup_summary.json"

        self.match_counter = 1

    def _generate_match_id(self, camera_id: str, track_id: str, pts_ms: Optional[float]) -> str:
        pts_part = int(pts_ms) if pts_ms is not None else self.match_counter
        self.match_counter += 1
        return f"MATCH-{camera_id}-{track_id}-{pts_part}"

    def process_observation(self, observation: Dict[str, Any]) -> MatchDecision:
        """Process a single vehicle observation through the full matching pipeline.

        Args:
            observation: Dict representation of vehicle observation (from Step 6 or Step 7).

        Returns:
            MatchDecision object.
        """
        obs_id = observation.get("observation_id") or f"OBS-{uuid.uuid4().hex[:8]}"
        raw_reg = observation.get("registration_number") or ""
        norm_reg = observation.get("normalized_registration_number") or normalize_watchlist_registration(raw_reg)
        cam_id = observation.get("camera_id") or "unknown"
        trk_id = observation.get("track_id") or "unknown"
        first_pts = observation.get("first_seen_pts_ms") or observation.get("observed_at_pts_ms")
        rec_pts = observation.get("recognition_pts_ms") or first_pts
        last_pts = observation.get("last_seen_pts_ms") or rec_pts
        src_time = observation.get("source_time")
        src_time_status = observation.get("source_time_status") or "NOT_RESOLVED"
        ocr_conf = float(observation.get("ocr_confidence") or 0.0)
        consensus = float(observation.get("consensus_score") or 0.0)
        plate_det_conf = float(observation.get("plate_detection_confidence") or 0.0)
        plate_qual = float(observation.get("plate_quality_score") or 0.0)
        evidence_img = observation.get("evidence_image")
        filename_pts_status = observation.get("evidence_filename_pts_status") or "UNKNOWN"

        match_id = self._generate_match_id(cam_id, trk_id, rec_pts)

        # 1. Eligibility Evaluation Gate
        is_eligible, elig_reason = self.eligibility_evaluator.evaluate(observation)
        if not is_eligible:
            decision = MatchDecision(
                match_id=match_id,
                observation_id=obs_id,
                decision="NOT_ELIGIBLE",
                decision_reason=elig_reason,
                registration_number=raw_reg,
                normalized_registration_number=norm_reg,
                watchlist_id=None,
                watchlist_match_score=0.0,
                recognition_confidence=ocr_conf,
                consensus_score=consensus,
                plate_detection_confidence=plate_det_conf,
                plate_quality_score=plate_qual,
                camera_id=cam_id,
                track_id=trk_id,
                first_seen_pts_ms=first_pts,
                recognition_pts_ms=rec_pts,
                last_seen_pts_ms=last_pts,
                source_time=src_time,
                source_time_status=src_time_status,
                evidence_image=evidence_img,
                evidence_filename_pts_status=filename_pts_status,
                watchlist_metadata=None,
                is_deduplicated=False,
                supporting_observations_count=1,
            )
            self._record_decision(decision)
            return decision

        # 2. Watchlist Matching
        dec_type, dec_reason, match_score, matched_veh, fuzzy_diff = self.matcher.match(norm_reg)

        meta_snapshot = WatchlistMetadataSnapshot.from_watchlist_vehicle(matched_veh) if matched_veh else None
        wl_id = matched_veh.watchlist_id if matched_veh else None

        decision = MatchDecision(
            match_id=match_id,
            observation_id=obs_id,
            decision=dec_type,
            decision_reason=dec_reason,
            registration_number=raw_reg,
            normalized_registration_number=norm_reg,
            watchlist_id=wl_id,
            watchlist_match_score=match_score,
            recognition_confidence=ocr_conf,
            consensus_score=consensus,
            plate_detection_confidence=plate_det_conf,
            plate_quality_score=plate_qual,
            camera_id=cam_id,
            track_id=trk_id,
            first_seen_pts_ms=first_pts,
            recognition_pts_ms=rec_pts,
            last_seen_pts_ms=last_pts,
            source_time=src_time,
            source_time_status=src_time_status,
            evidence_image=evidence_img,
            evidence_filename_pts_status=filename_pts_status,
            watchlist_metadata=meta_snapshot,
            is_deduplicated=False,
            supporting_observations_count=1,
            fuzzy_diff=fuzzy_diff,
        )

        # 3. Deduplication Evaluation
        is_dup, supp_count, _ = self.deduplicator.process_decision(decision)
        decision.is_deduplicated = is_dup
        decision.supporting_observations_count = supp_count

        self._record_decision(decision)
        return decision

    def _record_decision(self, decision: MatchDecision) -> None:
        """Append decision record to appropriate storage destinations."""
        data_str = json.dumps(decision.to_dict()) + "\n"

        # 1. Raw decision log (auditable trail of every decision)
        with open(self.raw_file, "a", encoding="utf-8") as f:
            f.write(data_str)

        # 2. Confirmed alert-ready events (only non-suppressed MATCH decisions)
        if decision.decision == "MATCH" and not decision.is_deduplicated:
            with open(self.confirmed_file, "a", encoding="utf-8") as f:
                f.write(data_str)

        # 3. Rejected log (NOT_ELIGIBLE and NO_MATCH)
        if decision.decision in ("NOT_ELIGIBLE", "NO_MATCH"):
            with open(self.rejected_file, "a", encoding="utf-8") as f:
                f.write(data_str)

    def write_summary_report(self, all_decisions: List[MatchDecision]) -> Dict[str, Any]:
        """Generate and save comprehensive deduplicated summary report."""
        total = len(all_decisions)
        eligible = [d for d in all_decisions if d.decision != "NOT_ELIGIBLE"]
        exact_matches = [d for d in all_decisions if d.decision == "MATCH"]
        alert_ready_matches = [d for d in exact_matches if not d.is_deduplicated]
        suppressed_matches = [d for d in exact_matches if d.is_deduplicated]
        no_matches = [d for d in all_decisions if d.decision == "NO_MATCH"]
        possible_reviews = [d for d in all_decisions if d.decision == "POSSIBLE_MATCH_REVIEW"]
        not_eligible = [d for d in all_decisions if d.decision == "NOT_ELIGIBLE"]

        report = {
            "total_observations_processed": total,
            "eligible_observations": len(eligible),
            "not_eligible_observations": len(not_eligible),
            "total_exact_matches": len(exact_matches),
            "alert_ready_matches": len(alert_ready_matches),
            "suppressed_duplicate_matches": len(suppressed_matches),
            "no_matches": len(no_matches),
            "possible_match_reviews": len(possible_reviews),
            "alert_ready_events": [d.to_dict() for d in alert_ready_matches],
        }

        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
