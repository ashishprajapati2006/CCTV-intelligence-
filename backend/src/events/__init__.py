"""Events Backbone Package for Step 10 Kafka Event Pipeline.

Exports:
  - Topics: TOPIC_VEHICLE_DETECTIONS, TOPIC_VEHICLE_ANPR, TOPIC_WATCHLIST_MATCHES, TOPIC_ALERTS, TOPIC_DLQ
  - Schemas: EventEnvelope, EventSource, build_anpr_event, build_alert_event, etc.
  - Serialization: serialize_event, deserialize_event
  - Idempotency: IdempotencyTracker, JSONIdempotencyTracker
  - Worker: WatchlistMatchingWorker
  - Producer: EventProducer
  - Consumer: EventConsumer
"""
from src.events.consumer import EventConsumer
from src.events.idempotency import IdempotencyTracker, JSONIdempotencyTracker
from src.events.producer import EventProducer
from src.events.retry import RetryConfig, execute_with_retry
from src.events.schemas import (
    CURRENT_SCHEMA_VERSION,
    EventEnvelope,
    EventSource,
    build_alert_event,
    build_anpr_event,
    build_dlq_event,
    build_vehicle_detection_event,
    build_watchlist_match_event,
    generate_event_id,
)
from src.events.serializer import (
    EventSerializationError,
    EventValidationError,
    deserialize_event,
    serialize_event,
)
from src.events.topics import (
    ALL_STANDARD_TOPICS,
    TOPIC_ALERTS,
    TOPIC_CAMERA_HEALTH,
    TOPIC_DLQ,
    TOPIC_VEHICLE_ANPR,
    TOPIC_VEHICLE_DETECTIONS,
    TOPIC_VEHICLE_JOURNEYS,
    TOPIC_WATCHLIST_MATCHES,
)
from src.events.watchlist_worker import WatchlistMatchingWorker

__all__ = [
    "ALL_STANDARD_TOPICS",
    "TOPIC_VEHICLE_DETECTIONS",
    "TOPIC_VEHICLE_ANPR",
    "TOPIC_WATCHLIST_MATCHES",
    "TOPIC_ALERTS",
    "TOPIC_DLQ",
    "TOPIC_CAMERA_HEALTH",
    "TOPIC_VEHICLE_JOURNEYS",
    "CURRENT_SCHEMA_VERSION",
    "EventEnvelope",
    "EventSource",
    "generate_event_id",
    "build_vehicle_detection_event",
    "build_anpr_event",
    "build_watchlist_match_event",
    "build_alert_event",
    "build_dlq_event",
    "serialize_event",
    "deserialize_event",
    "EventSerializationError",
    "EventValidationError",
    "IdempotencyTracker",
    "JSONIdempotencyTracker",
    "RetryConfig",
    "execute_with_retry",
    "WatchlistMatchingWorker",
    "EventProducer",
    "EventConsumer",
]
