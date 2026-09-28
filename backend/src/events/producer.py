"""High-level Event Producer for Step 10 Event Pipeline.

Provides typed event publication methods for detections, ANPR, matches, and alerts.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.common.logging import get_logger
from src.events.schemas import (
    EventEnvelope,
    build_alert_event,
    build_anpr_event,
    build_vehicle_detection_event,
    build_watchlist_match_event,
)
from src.events.topics import (
    TOPIC_ALERTS,
    TOPIC_VEHICLE_ANPR,
    TOPIC_VEHICLE_DETECTIONS,
    TOPIC_WATCHLIST_MATCHES,
)
from src.kafka.config import KafkaConfig
from src.kafka.producer import KafkaProducerWrapper

logger = get_logger("event_producer")


class EventProducer:
    """High-level facade for publishing typed domain events into Kafka."""

    def __init__(self, config: Optional[KafkaConfig] = None, producer_wrapper: Optional[KafkaProducerWrapper] = None):
        self.wrapper = producer_wrapper or KafkaProducerWrapper(config=config)

    def publish_detection(
        self,
        camera_id: str,
        track_id: str,
        vehicle_class: str,
        confidence: float,
        bbox: List[int],
        pts_ms: Optional[float] = None,
        correlation_id: Optional[str] = None,
    ) -> EventEnvelope:
        event = build_vehicle_detection_event(
            camera_id=camera_id,
            track_id=track_id,
            vehicle_class=vehicle_class,
            confidence=confidence,
            bbox=bbox,
            pts_ms=pts_ms,
            correlation_id=correlation_id,
        )
        self.wrapper.publish_event(TOPIC_VEHICLE_DETECTIONS, event)
        return event

    def publish_anpr(
        self,
        camera_id: str,
        track_id: str,
        registration_number: str,
        normalized_registration_number: str,
        recognition_status: str,
        consensus_score: float,
        ocr_confidence: float,
        plate_detection_confidence: float = 0.95,
        plate_quality_score: float = 0.88,
        recognition_pts_ms: Optional[float] = None,
        evidence_image: Optional[str] = None,
        source_time: Optional[str] = None,
        event_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> EventEnvelope:
        event = build_anpr_event(
            camera_id=camera_id,
            track_id=track_id,
            registration_number=registration_number,
            normalized_registration_number=normalized_registration_number,
            recognition_status=recognition_status,
            consensus_score=consensus_score,
            ocr_confidence=ocr_confidence,
            plate_detection_confidence=plate_detection_confidence,
            plate_quality_score=plate_quality_score,
            recognition_pts_ms=recognition_pts_ms,
            evidence_image=evidence_image,
            source_time=source_time,
            event_id=event_id,
            correlation_id=correlation_id,
        )
        self.wrapper.publish_event(TOPIC_VEHICLE_ANPR, event)
        return event

    def close(self) -> None:
        self.wrapper.close()
