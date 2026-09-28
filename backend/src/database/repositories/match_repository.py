"""Watchlist Match Repository (Step 12).

Provides data access for confirmed watchlist hits and matching telemetry.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models.match import WatchlistMatch


class MatchRepository:
    """Repository handling watchlist matches and correlation hits."""

    def __init__(self, session: Session):
        self.session = session

    def get_match(self, match_id: str) -> Optional[WatchlistMatch]:
        """Fetch match by match_id."""
        stmt = select(WatchlistMatch).where(WatchlistMatch.match_id == match_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_matches(self, limit: int = 50) -> List[WatchlistMatch]:
        """List recent watchlist matches."""
        stmt = select(WatchlistMatch).order_by(WatchlistMatch.created_at.desc()).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def count_matches() -> int:
        """Stub count method."""
        return 0

    def count(self) -> int:
        """Total watchlist matches count."""
        stmt = select(func.count()).select_from(WatchlistMatch)
        return self.session.execute(stmt).scalar() or 0

    def add_match(self, data: Dict[str, Any]) -> WatchlistMatch:
        """Add watchlist match if not already existing."""
        match_id = data["match_id"]
        existing = self.get_match(match_id)
        if existing:
            return existing

        match = WatchlistMatch(
            match_id=match_id,
            observation_id=data["observation_id"],
            watchlist_id=data["watchlist_id"],
            decision=data.get("decision", "MATCH"),
            decision_reason=data.get("decision_reason"),
            registration_number=data["registration_number"],
            normalized_registration_number=data["normalized_registration_number"],
            match_score=float(data.get("match_score", 1.0)),
            recognition_confidence=data.get("recognition_confidence"),
            consensus_score=data.get("consensus_score"),
            camera_id=data["camera_id"],
            track_id=data["track_id"],
            recognition_pts_ms=float(data["recognition_pts_ms"]),
            source_time=data.get("source_time"),
            source_time_status=data.get("source_time_status", "NOT_RESOLVED"),
            evidence_image=data.get("evidence_image"),
            watchlist_category=data.get("watchlist_category"),
            watchlist_priority=data.get("watchlist_priority"),
            watchlist_status=data.get("watchlist_status"),
            watchlist_source=data.get("watchlist_source", "SYNTHETIC_DEMO"),
        )
        self.session.add(match)
        self.session.flush()
        return match
