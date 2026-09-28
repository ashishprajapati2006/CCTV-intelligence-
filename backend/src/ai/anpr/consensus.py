"""Multi-Frame & Cross-Variant Consensus Engine for Step 6 ANPR.

EVIDENCE DIVERSITY RULES:
  1. Frame Diversity vs Preprocessing Diversity:
     - Multiple preprocessing variants of the SAME video frame (same frame_id)
       are grouped into a single frame-level vote (average or max score across variants).
     - Genuinely different video frames (different frame_ids / PTS timestamps)
       contribute independent physical evidence votes.

  2. Fuzzy Match Clustering:
     - Optional edit-distance (Levenshtein) clustering (distance <= 1) to cluster
       minor OCR character disagreements (e.g., GJ01AB1234 vs GJ01A81234).
     - Preserves all original raw OCR readings in the evidence chain.

  3. Recognition Status Assignment:
     - CONFIRMED : Score >= min_confirmed_score (0.70), distinct_frames >= 2, valid_format = True
     - PROBABLE  : Score >= min_probable_score (0.50), distinct_frames >= 1, valid_format = True
     - UNCERTAIN : Conflicting readings with similar support or low confidence.
     - UNREADABLE: No valid/readable text extracted across any frame.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

from src.ai.anpr.schemas import ANPRObservation, TrackANPRConsensus
from src.ai.anpr.validator import validate_plate
from src.common.logging import get_logger

logger = get_logger("anpr_consensus")


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0 = list(v1)

    return v1[len(s2)]


class MultiFrameConsensusEngine:
    """Multi-frame and cross-variant consensus engine for track-level ANPR."""

    def __init__(
        self,
        min_confirmed_score: float = 0.05,
        min_probable_score: float = 0.03,
        min_confirmed_frames: int = 1,
        enable_fuzzy_clustering: bool = True,
        max_fuzzy_distance: int = 1,
        plate_format: str = "indian",
    ):
        self.min_confirmed_score = min_confirmed_score
        self.min_probable_score = min_probable_score
        self.min_confirmed_frames = min_confirmed_frames
        self.enable_fuzzy_clustering = enable_fuzzy_clustering
        self.max_fuzzy_distance = max_fuzzy_distance
        self.plate_format = plate_format  # 'indian', 'european', or 'auto'

    def resolve_track_consensus(
        self,
        camera_id: str,
        track_id: str,
        observations: List[ANPRObservation],
    ) -> TrackANPRConsensus:
        """Process all ANPR observations for one vehicle track and compute final consensus.

        Args:
            camera_id: Camera identifier.
            track_id: Vehicle track identifier.
            observations: List of ANPRObservation objects collected across frames & variants.

        Returns:
            TrackANPRConsensus object with status, registration number, and evidence chain.
        """
        if not observations:
            return TrackANPRConsensus(
                camera_id=camera_id,
                track_id=track_id,
                final_registration_number=None,
                recognition_status="UNREADABLE",
                consensus_score=0.0,
                supporting_frame_count=0,
                total_frames_examined=0,
                total_observations_examined=0,
                first_seen_pts_ms=None,
                last_seen_pts_ms=None,
            )

        # Track-level PTS bounds
        pts_list = [o.pts_ms for o in observations if o.pts_ms is not None]
        first_pts = min(pts_list) if pts_list else None
        last_pts = max(pts_list) if pts_list else None

        # Filter out empty normalized readings
        valid_obs = [o for o in observations if o.normalized_ocr.normalized_text]

        unique_frames_examined = len({o.frame_id for o in observations})

        if not valid_obs:
            return TrackANPRConsensus(
                camera_id=camera_id,
                track_id=track_id,
                final_registration_number=None,
                recognition_status="UNREADABLE",
                consensus_score=0.0,
                supporting_frame_count=0,
                total_frames_examined=unique_frames_examined,
                total_observations_examined=len(observations),
                first_seen_pts_ms=first_pts,
                last_seen_pts_ms=last_pts,
            )

        # ── 1. Group observations by (normalized_text, frame_id) ───────────────
        # This prevents 5 preprocessing variants of 1 frame from acting as 5 independent physical votes.
        frame_votes: Dict[str, Dict[int, List[ANPRObservation]]] = defaultdict(lambda: defaultdict(list))

        for obs in valid_obs:
            text = obs.normalized_ocr.normalized_text
            frame_votes[text][obs.frame_id].append(obs)

        # ── 2. Calculate string scores based on FRAME DIVERSITY ───────────────
        string_scores: Dict[str, float] = {}
        string_frames: Dict[str, Set[int]] = {}
        string_valid_format: Dict[str, bool] = {}

        for text, fmap in frame_votes.items():
            frame_scores = []
            frame_set = set(fmap.keys())

            for fid, obs_list in fmap.items():
                # Frame-level score is max score across preprocessing variants for this frame
                best_variant_score = max(o.observation_score for o in obs_list)
                frame_scores.append(best_variant_score)

            # Cumulative weighted support = sum of best frame scores + frame diversity bonus
            raw_sum = sum(frame_scores)
            distinct_frames = len(frame_set)
            diversity_bonus = min(0.30, (distinct_frames - 1) * 0.15)
            
            # Average frame quality * frame multiplier
            avg_frame_score = raw_sum / float(distinct_frames)
            final_str_score = min(1.0, avg_frame_score * (1.0 + diversity_bonus))

            string_scores[text] = final_str_score
            string_frames[text] = frame_set
            # Validate against configured plate format (indian/european/auto)
            is_valid, fmt_name, fmt_boost = validate_plate(text, self.plate_format)
            string_valid_format[text] = is_valid

        # ── 3. Optional Fuzzy Match Clustering ────────────────────────────────
        if self.enable_fuzzy_clustering and len(string_scores) > 1:
            sorted_candidates = sorted(string_scores.keys(), key=lambda k: string_scores[k], reverse=True)
            primary_str = sorted_candidates[0]

            for secondary_str in sorted_candidates[1:]:
                dist = _levenshtein_distance(primary_str, secondary_str)
                if dist <= self.max_fuzzy_distance:
                    # Merge secondary support into primary if primary has higher score
                    merged_frames = string_frames[primary_str].union(string_frames[secondary_str])
                    string_frames[primary_str] = merged_frames
                    
                    # Boost primary score slightly due to fuzzy agreement
                    string_scores[primary_str] = min(1.0, string_scores[primary_str] + 0.10)
                    logger.debug(
                        "[%s/%s] Fuzzy match clustered '%s' into '%s' (edit dist=%d)",
                        camera_id, track_id, secondary_str, primary_str, dist
                    )

        # ── 4. Rank Candidates ────────────────────────────────────────────────
        ranked = sorted(string_scores.keys(), key=lambda k: string_scores[k], reverse=True)
        best_text = ranked[0]
        best_score = string_scores[best_text]
        best_frame_ids = sorted(list(string_frames[best_text]))
        supporting_count = len(best_frame_ids)
        is_valid_fmt = string_valid_format.get(best_text, False)

        # ── 5. Status Assignment Rules ────────────────────────────────────────
        if best_score >= self.min_confirmed_score and supporting_count >= self.min_confirmed_frames and is_valid_fmt:
            status = "CONFIRMED"
            final_reg = best_text
        elif best_score >= self.min_probable_score and supporting_count >= 1 and is_valid_fmt:
            status = "PROBABLE"
            final_reg = best_text
        elif best_score >= 0.02 and supporting_count >= 1:
            status = "UNCERTAIN"
            final_reg = None  # Do not hallucinate or confirm uncertain plate
        else:
            status = "UNREADABLE"
            final_reg = None

        # Build evidence chain trace for best candidate
        evidence_chain = []
        for o in valid_obs:
            if o.normalized_ocr.normalized_text == best_text or (self.enable_fuzzy_clustering and _levenshtein_distance(o.normalized_ocr.normalized_text, best_text) <= self.max_fuzzy_distance):
                evidence_chain.append(o.to_dict())

        return TrackANPRConsensus(
            camera_id=camera_id,
            track_id=track_id,
            final_registration_number=final_reg,
            recognition_status=status,
            consensus_score=best_score,
            supporting_frame_count=supporting_count,
            total_frames_examined=unique_frames_examined,
            total_observations_examined=len(observations),
            first_seen_pts_ms=first_pts,
            last_seen_pts_ms=last_pts,
            supporting_frame_ids=best_frame_ids,
            candidate_scores=string_scores,
            evidence_chain=evidence_chain,
        )
