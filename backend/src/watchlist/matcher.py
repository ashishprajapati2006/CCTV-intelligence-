"""Watchlist Matching Engine (Step 9).

Performs exact normalized lookup and optional controlled candidate review.
Strictly adheres to:
  - ACTIVE records only for automated matching (INACTIVE/EXPIRED do not match).
  - Exact normalized equality as primary mechanism.
  - Score separation: recognition_confidence != consensus_score != watchlist_match_score.
  - Controlled fuzzy review for OCR ambiguity (disabled by default for alerts).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.common.logging import get_logger
from src.watchlist.models import (
    MatchDecision,
    WatchlistMetadataSnapshot,
    WatchlistVehicle,
)
from src.watchlist.normalizer import normalize_watchlist_registration
from src.watchlist.repository import WatchlistRepository

logger = get_logger("watchlist_matcher")


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[m][n]


class WatchlistMatcher:
    """Matches normalized vehicle registrations against active watchlist entries."""

    def __init__(
        self,
        repository: WatchlistRepository,
        enable_fuzzy_review: bool = False,
        max_fuzzy_distance: int = 1,
    ):
        self.repo = repository
        self.enable_fuzzy_review = enable_fuzzy_review
        self.max_fuzzy_distance = max_fuzzy_distance

    def match(self, normalized_reg: str) -> Tuple[str, str, float, Optional[WatchlistVehicle], Optional[Dict[str, Any]]]:
        """Perform watchlist matching for a normalized registration.

        Args:
            normalized_reg: Canonical normalized plate string.

        Returns:
            Tuple of:
              - decision: "MATCH" | "NO_MATCH" | "POSSIBLE_MATCH_REVIEW"
              - decision_reason: explanation string
              - match_score: float (1.0 for exact, ratio for fuzzy, 0.0 for none)
              - matched_vehicle: Optional WatchlistVehicle
              - fuzzy_diff: Optional dict with diff diagnostics
        """
        clean_reg = normalize_watchlist_registration(normalized_reg)

        # 1. Exact Match Lookup
        candidate = self.repo.get_vehicle(clean_reg)
        if candidate is not None:
            if candidate.status == "ACTIVE":
                return "MATCH", "EXACT_NORMALIZED_MATCH", 1.0, candidate, None
            else:
                return "NO_MATCH", f"INACTIVE_WATCHLIST_ENTRY (status={candidate.status})", 0.0, candidate, None

        # 2. Optional Fuzzy Review (Disabled by default for automated alerts)
        if self.enable_fuzzy_review:
            active_vehicles = self.repo.get_active_vehicles()
            best_diff = None
            closest_veh = None
            min_dist = 999

            for v in active_vehicles:
                dist = _levenshtein_distance(clean_reg, v.normalized_registration_number)
                if dist <= self.max_fuzzy_distance and dist < min_dist:
                    min_dist = dist
                    closest_veh = v
                    max_len = max(len(clean_reg), len(v.normalized_registration_number))
                    sim_ratio = (max_len - dist) / max_len if max_len > 0 else 0.0
                    best_diff = {
                        "target_registration": v.normalized_registration_number,
                        "observed_registration": clean_reg,
                        "edit_distance": dist,
                        "similarity_score": round(sim_ratio, 4),
                    }

            if closest_veh and best_diff:
                return (
                    "POSSIBLE_MATCH_REVIEW",
                    f"FUZZY_CANDIDATE_DETECTED (edit_distance={best_diff['edit_distance']})",
                    best_diff["similarity_score"],
                    closest_veh,
                    best_diff,
                )

        return "NO_MATCH", "UNLISTED_REGISTRATION", 0.0, None, None
