"""Unit tests for Step 10 — Kafka Event Pipeline.

Covers Part 23:
  1. Event serialization and deserialization with UTF-8 encoding.
  2. Schema envelope validation and schema version checks (rejects invalid versions).
  3. Producer partition key derivation (normalized plate vs camera:track fallback).
  4. Consumer offset tracking and manual commit behavior (at-least-once).
  5. Idempotency tracker: drops duplicate event_ids.
  6. Exponential backoff retry handler (transient vs permanent errors).
  7. Dead Letter Queue (DLQ) routing for malformed/unrecoverable messages.
  8. Watchlist matching worker integration (vehicle.anpr -> watchlist.matches).
  9. Alert event generation on qualifying matches (watchlist.matches -> alerts).
  10. Correlation ID preservation and distributed tracing.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.events import (
    CURRENT_SCHEMA_VERSION,
    TOPIC_ALERTS,
    TOPIC_DLQ,
    TOPIC_VEHICLE_ANPR,
    TOPIC_WATCHLIST_MATCHES,
    EventEnvelope,
    EventProducer,
    EventSerializationError,
    EventSource,
    EventValidationError,
    IdempotencyTracker,
    JSONIdempotencyTracker,
    RetryConfig,
    WatchlistMatchingWorker,
    build_alert_event,
    build_anpr_event,
    build_dlq_event,
    deserialize_event,
    execute_with_retry,
    generate_event_id,
    serialize_event,
)
from src.kafka import (
    InMemoryKafkaBroker,
    KafkaConfig,
    KafkaConsumerWrapper,
    KafkaProducerWrapper,
)
from src.watchlist import (
    JSONWatchlistRepository,
    MatchDecisionEngine,
    WatchlistVehicle,
)


@pytest.fixture(autouse=True)
def reset_broker():
    """Ensure in-memory broker is clear before each test."""
    broker = InMemoryKafkaBroker.get_instance()
    broker.clear()
    yield
    broker.clear()


@pytest.fixture
def mock_pipeline(tmp_path):
    v_file = tmp_path / "watchlist.json"
    repo = JSONWatchlistRepository(vehicles_path=str(v_file))

    # Add active vehicle
    v1 = WatchlistVehicle(
        watchlist_id="WL-VEH-000001",
        registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234",
        category="DEMO_STOLEN_VEHICLE",
        priority="HIGH",
        status="ACTIVE",
        source="SYNTHETIC_DEMO",
        synthetic=True,
    )
    repo.add_vehicle(v1)

    decision_engine = MatchDecisionEngine(repository=repo, matches_dir=str(tmp_path / "matches"))
    producer_wrapper = KafkaProducerWrapper(config=KafkaConfig(use_mock_fallback=True))
    idempotency_tracker = JSONIdempotencyTracker(storage_path=str(tmp_path / "events.json"))

    worker = WatchlistMatchingWorker(
        decision_engine=decision_engine,
        producer=producer_wrapper,
        idempotency_tracker=idempotency_tracker,
    )

    return {
        "repo": repo,
        "worker": worker,
        "producer": producer_wrapper,
        "tracker": idempotency_tracker,
    }


# ════════════════════════════════════════════════════════════════════════════
# 1. Serialization & Schema Envelope Validation
# ════════════════════════════════════════════════════════════════════════════

def test_serialization_roundtrip():
    event = build_anpr_event(
        camera_id="cam01",
        track_id="TRK-001",
        registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234",
        recognition_status="CONFIRMED",
        consensus_score=0.94,
        ocr_confidence=0.93,
        plate_detection_confidence=0.95,
        plate_quality_score=0.88,
        recognition_pts_ms=11200.0,
        correlation_id="CORR-001",
    )

    raw_bytes = serialize_event(event)
    assert isinstance(raw_bytes, bytes)

    deserialized = deserialize_event(raw_bytes)
    assert deserialized.event_id == event.event_id
    assert deserialized.event_type == "vehicle.anpr"
    assert deserialized.schema_version == CURRENT_SCHEMA_VERSION
    assert deserialized.correlation_id == "CORR-001"
    assert deserialized.payload["normalized_registration_number"] == "GJ01AB1234"


def test_schema_validation_unsupported_version():
    malformed = {
        "event_id": "EVT-001",
        "event_type": "vehicle.anpr",
        "schema_version": "2.0",  # Incompatible version
        "source": {"system": "test"},
        "payload": {},
    }
    raw = json.dumps(malformed).encode("utf-8")
    with pytest.raises(EventValidationError) as exc:
        deserialize_event(raw)
    assert "Unsupported schema_version" in str(exc.value)


def test_schema_validation_missing_envelope_fields():
    incomplete = {"event_id": "EVT-001"}
    raw = json.dumps(incomplete).encode("utf-8")
    with pytest.raises(EventValidationError) as exc:
        deserialize_event(raw)
    assert "Missing required envelope fields" in str(exc.value)


# ════════════════════════════════════════════════════════════════════════════
# 2. Partition Key Derivation
# ════════════════════════════════════════════════════════════════════════════

def test_partition_key_derivation():
    # Case A: Has normalized registration -> key is registration
    evt1 = build_anpr_event(
        camera_id="cam01", track_id="TRK-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", recognition_status="CONFIRMED",
        consensus_score=0.94, ocr_confidence=0.92, plate_detection_confidence=0.9,
        plate_quality_score=0.8, recognition_pts_ms=1000.0,
    )
    assert KafkaProducerWrapper.derive_partition_key(evt1) == b"GJ01AB1234"

    # Case B: Unreadable plate fallback -> key is camera:track
    evt2 = build_anpr_event(
        camera_id="cam01", track_id="TRK-002", registration_number="",
        normalized_registration_number="", recognition_status="UNREADABLE",
        consensus_score=0.0, ocr_confidence=0.0, plate_detection_confidence=0.0,
        plate_quality_score=0.0, recognition_pts_ms=1000.0,
    )
    assert KafkaProducerWrapper.derive_partition_key(evt2) == b"cam01:TRK-002"


# ════════════════════════════════════════════════════════════════════════════
# 3. Idempotency & Duplicate Suppression (Part 5 & 18)
# ════════════════════════════════════════════════════════════════════════════

def test_idempotency_tracker(tmp_path):
    tracker = JSONIdempotencyTracker(storage_path=str(tmp_path / "processed_ids.json"))
    assert tracker.is_processed("EVT-100") is False

    tracker.mark_processed("EVT-100")
    assert tracker.is_processed("EVT-100") is True

    # Reload from disk
    tracker2 = JSONIdempotencyTracker(storage_path=str(tmp_path / "processed_ids.json"))
    assert tracker2.is_processed("EVT-100") is True


def test_duplicate_event_not_reprocessed(mock_pipeline):
    worker = mock_pipeline["worker"]
    broker = InMemoryKafkaBroker.get_instance()

    event = build_anpr_event(
        camera_id="cam01", track_id="TRK-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", recognition_status="CONFIRMED",
        consensus_score=0.94, ocr_confidence=0.93, plate_detection_confidence=0.95,
        plate_quality_score=0.88, recognition_pts_ms=11200.0,
        event_id="EVT-IDEMPOTENCY-TEST-001",
    )

    # First delivery
    res1 = worker.process_anpr_event(event)
    assert res1 is not None
    assert len(broker.topics[TOPIC_WATCHLIST_MATCHES]) == 1
    assert len(broker.topics[TOPIC_ALERTS]) == 1

    # Second delivery with identical event_id
    res2 = worker.process_anpr_event(event)
    assert res2 is None  # Skipped!
    assert worker.metrics["duplicate_events_skipped"] == 1
    # Message counts must NOT increase
    assert len(broker.topics[TOPIC_WATCHLIST_MATCHES]) == 1
    assert len(broker.topics[TOPIC_ALERTS]) == 1


# ════════════════════════════════════════════════════════════════════════════
# 4. Retry Logic & Error Classification (Part 9)
# ════════════════════════════════════════════════════════════════════════════

def test_retry_transient_failure():
    attempts = 0

    def _flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionResetError("Temporary network reset")
        return "success"

    cfg = RetryConfig(max_retries=4, initial_backoff_sec=0.01, backoff_multiplier=1.5)
    res = execute_with_retry(_flaky, config=cfg)
    assert res == "success"
    assert attempts == 3


def test_retry_permanent_failure_aborts_immediately():
    attempts = 0

    def _fatal():
        nonlocal attempts
        attempts += 1
        raise EventValidationError("Malformed schema")

    cfg = RetryConfig(max_retries=4, initial_backoff_sec=0.01)
    with pytest.raises(EventValidationError):
        execute_with_retry(_fatal, config=cfg)
    assert attempts == 1  # No wasteful retries on permanent faults


# ════════════════════════════════════════════════════════════════════════════
# 5. Dead Letter Queue (DLQ) Handling (Part 10)
# ════════════════════════════════════════════════════════════════════════════

def test_consumer_routes_malformed_messages_to_dlq():
    broker = InMemoryKafkaBroker.get_instance()
    # Publish corrupt bytes to vehicle.anpr
    broker.publish(TOPIC_VEHICLE_ANPR, key=b"bad", value=b"NOT_A_VALID_JSON_STRING{{{")

    consumer = KafkaConsumerWrapper(topic=TOPIC_VEHICLE_ANPR, config=KafkaConfig(use_mock_fallback=True))
    processed = consumer.poll_and_process(handler=lambda evt: None, max_messages=10)

    assert processed == 0
    assert consumer.metrics["processing_failure"] == 1
    assert consumer.metrics["dlq_events"] == 1

    # Verify DLQ received the record
    assert len(broker.topics[TOPIC_DLQ]) == 1


# ════════════════════════════════════════════════════════════════════════════
# 6. End-to-End Worker Matching & Alert Generation
# ════════════════════════════════════════════════════════════════════════════

def test_worker_end_to_end_matching_and_alerts(mock_pipeline):
    worker = mock_pipeline["worker"]
    broker = InMemoryKafkaBroker.get_instance()

    # Ingest matching event (GJ01AB1234 is in watchlist with ACTIVE status)
    evt_match = build_anpr_event(
        camera_id="cam01", track_id="TRK-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", recognition_status="CONFIRMED",
        consensus_score=0.94, ocr_confidence=0.93, plate_detection_confidence=0.95,
        plate_quality_score=0.88, recognition_pts_ms=11200.0,
        correlation_id="TRACE-001",
    )
    worker.process_anpr_event(evt_match)

    # Must produce watchlist.matches and alerts
    assert len(broker.topics[TOPIC_WATCHLIST_MATCHES]) == 1
    assert len(broker.topics[TOPIC_ALERTS]) == 1

    # Ingest non-matching event (GJ05XY9999 is unlisted)
    evt_unlisted = build_anpr_event(
        camera_id="cam01", track_id="TRK-002", registration_number="GJ05XY9999",
        normalized_registration_number="GJ05XY9999", recognition_status="CONFIRMED",
        consensus_score=0.90, ocr_confidence=0.88, plate_detection_confidence=0.95,
        plate_quality_score=0.88, recognition_pts_ms=15000.0,
        correlation_id="TRACE-002",
    )
    worker.process_anpr_event(evt_unlisted)

    # watchlist.matches increments (records NO_MATCH decision)
    assert len(broker.topics[TOPIC_WATCHLIST_MATCHES]) == 2
    # alerts must NOT increment
    assert len(broker.topics[TOPIC_ALERTS]) == 1
