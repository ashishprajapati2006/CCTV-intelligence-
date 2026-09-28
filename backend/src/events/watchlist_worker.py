"""Watchlist Matching Worker for Step 10 Kafka Event Pipeline.

Connects the Kafka event transport to Step 9 MatchDecisionEngine:
  vehicle.anpr
      ↓
  Idempotency Check
      ↓
  Step 9 Matcher (Business Logic)
      ↓
  watchlist.matches (all decisions)
      ↓
  alerts (only qualifying non-deduplicated matches)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.common.logging import get_logger
from src.events.idempotency import IdempotencyTracker, JSONIdempotencyTracker
from src.events.schemas import (
    EventEnvelope,
    build_alert_event,
    build_watchlist_match_event,
)
from src.events.topics import TOPIC_ALERTS, TOPIC_WATCHLIST_MATCHES
from src.kafka.consumer import KafkaConsumerWrapper
from src.kafka.producer import KafkaProducerWrapper
from src.watchlist.decision import MatchDecisionEngine

logger = get_logger("watchlist_worker")


class WatchlistMatchingWorker:
    """Consumes vehicle.anpr events, evaluates watchlist matches via Step 9, and publishes match/alert events."""

    def __init__(
        self,
        decision_engine: MatchDecisionEngine,
        producer: KafkaProducerWrapper,
        idempotency_tracker: Optional[IdempotencyTracker] = None,
    ):
        self.decision_engine = decision_engine
        self.producer = producer
        self.idempotency_tracker = idempotency_tracker or JSONIdempotencyTracker()

        self.metrics = {
            "anpr_events_received": 0,
            "duplicate_events_skipped": 0,
            "matches_produced": 0,
            "alerts_produced": 0,
        }

    def process_anpr_event(self, event: EventEnvelope) -> Optional[Dict[str, Any]]:
        """Process a single vehicle.anpr event envelope."""
        self.metrics["anpr_events_received"] += 1
        event_id = event.event_id

        # 1. Idempotency Check (Part 5 & 18)
        if self.idempotency_tracker.is_processed(event_id):
            self.metrics["duplicate_events_skipped"] += 1
            logger.info("Skipping duplicate event_id: %s", event_id)
            return None

        # 2. Invoke Step 9 Matcher (Strict separation: business logic in Step 9)
        observation_payload = dict(event.payload)
        # Ensure observation_id defaults to event correlation_id or event_id
        if "observation_id" not in observation_payload:
            observation_payload["observation_id"] = event.correlation_id or event.event_id

        decision = self.decision_engine.process_observation(observation_payload)

        # 3. Publish watchlist.matches Event
        meta_dict = decision.watchlist_metadata.to_dict() if decision.watchlist_metadata else None
        match_event = build_watchlist_match_event(
            match_id=decision.match_id,
            observation_id=decision.observation_id,
            watchlist_id=decision.watchlist_id,
            decision=decision.decision,
            registration_number=decision.registration_number or "",
            normalized_registration_number=decision.normalized_registration_number or "",
            camera_id=decision.camera_id,
            track_id=decision.track_id,
            consensus_score=decision.consensus_score,
            match_score=decision.watchlist_match_score,
            watchlist_metadata=meta_dict,
            evidence_image=decision.evidence_image,
            recognition_pts_ms=decision.recognition_pts_ms,
            is_deduplicated=decision.is_deduplicated,
            correlation_id=event.correlation_id,
        )
        self.producer.publish_event(TOPIC_WATCHLIST_MATCHES, match_event)
        self.metrics["matches_produced"] += 1

        # 4. If Qualifying Alert: Publish alerts Event
        alert_event = None
        if decision.decision == "MATCH" and not decision.is_deduplicated and decision.watchlist_metadata:
            alert_event = build_alert_event(
                match_id=decision.match_id,
                watchlist_id=decision.watchlist_id or "unknown",
                registration_number=decision.normalized_registration_number or "",
                priority=decision.watchlist_metadata.priority,
                category=decision.watchlist_metadata.category,
                decision=decision.decision,
                camera_id=decision.camera_id,
                track_id=decision.track_id,
                recognition_pts_ms=decision.recognition_pts_ms,
                evidence_image=decision.evidence_image,
                recognition_consensus=decision.consensus_score,
                correlation_id=event.correlation_id,
            )
            self.producer.publish_event(TOPIC_ALERTS, alert_event)
            self.metrics["alerts_produced"] += 1

        # 5. Mark as processed for idempotency
        self.idempotency_tracker.mark_processed(event_id)

        return {
            "decision": decision.to_dict(),
            "match_event": match_event.to_dict(),
            "alert_event": alert_event.to_dict() if alert_event else None,
        }
