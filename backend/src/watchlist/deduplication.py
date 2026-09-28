"""Alert Deduplication Engine for Step 9 Watchlist Matching.

Suppresses repetitive alerts generated from consecutive video frames or tracks of the
same vehicle within a configurable time window.

Key Rules:
  1. Track-level deduplication: Multiple observations for the same (camera, track, watchlist_id)
     produce exactly ONE alert-ready match event, accumulating supporting observations.
  2. Cross-camera isolation: Sightings across distinct cameras (e.g. cam01 vs cam07)
     are NEVER merged into one match event. Each camera produces its own independent event.
  3. Configurable suppression window: Based on camera-local stream PTS (or wall-clock fallback).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.common.logging import get_logger
from src.watchlist.models import MatchDecision

logger = get_logger("watchlist_dedup")


@dataclass
class _SuppressionEntry:
    first_seen_pts_ms: Optional[float]
    last_seen_pts_ms: Optional[float]
    first_wall_clock: float
    last_wall_clock: float
    supporting_count: int
    primary_match_id: str


class AlertDeduplicator:
    """Manages suppression windows to prevent alert flooding on CCTV streams."""

    def __init__(self, suppression_window_pts_ms: float = 30000.0, suppression_window_seconds: float = 30.0):
        self.suppression_window_pts_ms = suppression_window_pts_ms
        self.suppression_window_seconds = suppression_window_seconds
        # Key: (camera_id, track_id, watchlist_id) -> _SuppressionEntry
        self._cache: Dict[Tuple[str, str, str], _SuppressionEntry] = {}

    def _make_key(self, decision: MatchDecision) -> Tuple[str, str, str]:
        """Construct deduplication key.

        Notice: camera_id is included, ensuring cross-camera observations are never merged.
        """
        cam = decision.camera_id or "unknown"
        trk = decision.track_id or "unknown"
        wl = decision.watchlist_id or decision.normalized_registration_number or "unlisted"
        return (cam, trk, wl)

    def process_decision(self, decision: MatchDecision) -> Tuple[bool, int, Optional[str]]:
        """Evaluate if match decision should be suppressed as a duplicate.

        Args:
            decision: MatchDecision to evaluate.

        Returns:
            Tuple of:
              - is_duplicate: bool (True if suppressed, False if this is the first / alert-ready match)
              - supporting_count: int (cumulative observation count for this track/window)
              - primary_match_id: Optional[str] (the ID of the first match event)
        """
        # Only successful matches and reviews require alert deduplication
        if decision.decision not in ("MATCH", "POSSIBLE_MATCH_REVIEW"):
            return False, 1, decision.match_id

        key = self._make_key(decision)
        current_wall = time.time()
        current_pts = decision.recognition_pts_ms or decision.first_seen_pts_ms

        entry = self._cache.get(key)

        if entry is None:
            # First occurrence
            self._cache[key] = _SuppressionEntry(
                first_seen_pts_ms=current_pts,
                last_seen_pts_ms=current_pts,
                first_wall_clock=current_wall,
                last_wall_clock=current_wall,
                supporting_count=1,
                primary_match_id=decision.match_id,
            )
            return False, 1, decision.match_id

        # Check suppression window
        is_suppressed = False

        if current_pts is not None and entry.first_seen_pts_ms is not None:
            # PTS-based suppression window (camera-local)
            pts_elapsed = abs(current_pts - entry.first_seen_pts_ms)
            if pts_elapsed <= self.suppression_window_pts_ms:
                is_suppressed = True
        else:
            # Fallback to wall-clock window
            wall_elapsed = current_wall - entry.first_wall_clock
            if wall_elapsed <= self.suppression_window_seconds:
                is_suppressed = True

        if is_suppressed:
            entry.supporting_count += 1
            entry.last_seen_pts_ms = current_pts
            entry.last_wall_clock = current_wall
            return True, entry.supporting_count, entry.primary_match_id
        else:
            # Window expired: start new alert sequence
            self._cache[key] = _SuppressionEntry(
                first_seen_pts_ms=current_pts,
                last_seen_pts_ms=current_pts,
                first_wall_clock=current_wall,
                last_wall_clock=current_wall,
                supporting_count=1,
                primary_match_id=decision.match_id,
            )
            return False, 1, decision.match_id

    def reset(self) -> None:
        """Clear the deduplication cache."""
        self._cache.clear()
