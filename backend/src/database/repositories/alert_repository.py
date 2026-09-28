"""Alert Repository (Step 12).

Provides data access and workflow state transitions for incident alerts.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models.alert import AlertModel


class AlertRepository:
    """Repository handling incident alert creation and operator lifecycle workflows."""

    def __init__(self, session: Session):
        self.session = session

    def get_alert(self, alert_id: str) -> Optional[AlertModel]:
        """Fetch alert by alert_id."""
        stmt = select(AlertModel).where(AlertModel.alert_id == alert_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_alerts(self, limit: int = 50) -> List[AlertModel]:
        """List recent alerts."""
        stmt = select(AlertModel).order_by(AlertModel.created_at.desc()).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def count(self) -> int:
        """Total alert count."""
        stmt = select(func.count()).select_from(AlertModel)
        return self.session.execute(stmt).scalar() or 0

    def add_alert(self, data: Dict[str, Any]) -> AlertModel:
        """Add new alert if not existing."""
        alert_id = data["alert_id"]
        existing = self.get_alert(alert_id)
        if existing:
            return existing

        alert = AlertModel(
            alert_id=alert_id,
            match_id=data.get("match_id"),
            alert_type=data.get("alert_type", "WATCHLIST_HIT"),
            priority=data.get("priority", "HIGH"),
            status=data.get("status", "NEW"),
            registration_number=data["registration_number"],
            camera_id=data["camera_id"],
            track_id=data["track_id"],
            message=data.get("message"),
            source_time=data.get("source_time"),
            source_time_status=data.get("source_time_status", "NOT_RESOLVED"),
            evidence_image=data.get("evidence_image"),
        )
        self.session.add(alert)
        self.session.flush()
        return alert

    def update_status(self, alert_id: str, status: str) -> Optional[AlertModel]:
        """Update alert status (e.g. ACKNOWLEDGED, ESCALATED, RESOLVED)."""
        alert = self.get_alert(alert_id)
        if not alert:
            return None

        alert.status = status
        now = datetime.now(timezone.utc)
        if status in ("ACKNOWLEDGED", "ESCALATED") and not alert.acknowledged_at:
            alert.acknowledged_at = now
        elif status == "RESOLVED":
            alert.resolved_at = now

        self.session.flush()
        return alert
