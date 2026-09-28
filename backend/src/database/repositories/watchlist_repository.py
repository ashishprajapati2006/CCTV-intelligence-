"""Watchlist Repository (Step 12).

Provides data access for synthetic demonstration watchlist entities and priority targets.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models.watchlist import WatchlistEntry


class WatchlistRepository:
    """Repository handling watchlist targets and priority classifications."""

    def __init__(self, session: Session):
        self.session = session

    def get_entry(self, watchlist_id: str) -> Optional[WatchlistEntry]:
        """Fetch watchlist entry by watchlist_id."""
        stmt = select(WatchlistEntry).where(WatchlistEntry.watchlist_id == watchlist_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_registration(
        self, normalized_registration: str
    ) -> Optional[WatchlistEntry]:
        """Fetch active watchlist target by normalized license plate."""
        stmt = select(WatchlistEntry).where(
            WatchlistEntry.normalized_registration_number == normalized_registration,
            WatchlistEntry.status == "ACTIVE",
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def list_entries(self, status: Optional[str] = "ACTIVE") -> List[WatchlistEntry]:
        """List watchlist entries filtered by status."""
        stmt = select(WatchlistEntry)
        if status:
            stmt = stmt.where(WatchlistEntry.status == status)
        stmt = stmt.order_by(WatchlistEntry.priority.asc(), WatchlistEntry.watchlist_id.asc())
        return list(self.session.execute(stmt).scalars().all())

    def count_entries(self) -> int:
        """Total watchlist targets count."""
        stmt = select(func.count()).select_from(WatchlistEntry)
        return self.session.execute(stmt).scalar() or 0

    def upsert_entry(self, data: Dict[str, Any]) -> WatchlistEntry:
        """Insert or update watchlist entry idempotently."""
        wl_id = data["watchlist_id"]
        entry = self.get_entry(wl_id)

        norm_reg = data.get("normalized_registration_number") or data.get("registration_number", "")

        if entry is None:
            entry = WatchlistEntry(
                watchlist_id=wl_id,
                entity_type=data.get("entity_type", "VEHICLE"),
                registration_number=data["registration_number"],
                normalized_registration_number=norm_reg,
                category=data.get("category", "SUSPECT_VEHICLE"),
                priority=data.get("priority", "HIGH"),
                status=data.get("status", "ACTIVE"),
                source=data.get("source", "SYNTHETIC_DEMO"),
                synthetic=data.get("synthetic", True),
                description=data.get("description"),
            )
            self.session.add(entry)
        else:
            entry.category = data.get("category", entry.category)
            entry.priority = data.get("priority", entry.priority)
            entry.status = data.get("status", entry.status)
            entry.description = data.get("description", entry.description)
            entry.source = data.get("source", entry.source)

        self.session.flush()
        return entry
