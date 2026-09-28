"""Kafka Producer Layer for Step 10 Event Pipeline.

Provides low-level message publication with delivery callbacks,
partition key derivation, and automatic offline test harness fallback.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

from src.common.logging import get_logger
from src.events.schemas import EventEnvelope
from src.events.serializer import serialize_event
from src.kafka.client import InMemoryKafkaBroker, check_kafka_connection
from src.kafka.config import KafkaConfig

logger = get_logger("kafka_producer")


class KafkaProducerWrapper:
    """Wrapper over KafkaProducer with transparent fallback for testing."""

    def __init__(self, config: Optional[KafkaConfig] = None):
        self.config = config or KafkaConfig()
        self._live_producer = None
        self._is_live = False
        self._broker = InMemoryKafkaBroker.get_instance()

        # Metrics
        self.metrics = {
            "events_produced": 0,
            "bytes_produced": 0,
            "delivery_failures": 0,
        }

        self._init_producer()

    def _init_producer(self) -> None:
        """Attempt to connect to real Kafka broker; fallback to in-memory if offline."""
        is_reachable = check_kafka_connection(self.config.bootstrap_servers)

        if is_reachable:
            try:
                from kafka import KafkaProducer
                self._live_producer = KafkaProducer(
                    bootstrap_servers=self.config.bootstrap_servers,
                    client_id=self.config.client_id,
                    request_timeout_ms=self.config.request_timeout_ms,
                    retries=3,
                )
                self._is_live = True
                logger.info("Connected to live Kafka broker at %s", self.config.bootstrap_servers)
                return
            except Exception as e:
                logger.warning("Failed to initialize KafkaProducer: %s. Using in-memory fallback.", e)

        if self.config.use_mock_fallback:
            logger.info("Operating in in-memory test transport mode (broker offline: %s)", self.config.bootstrap_servers)
            self._is_live = False
        else:
            raise ConnectionError(f"Kafka broker unreachable at {self.config.bootstrap_servers}")

    @staticmethod
    def derive_partition_key(event: EventEnvelope) -> bytes:
        """Derive Kafka message key for partition routing (Part 7 & 8).

        Ordering Strategy:
          - Use normalized_registration_number if available (keeps same vehicle on same partition).
          - Fallback to camera_id:track_id if plate is unreadable or not present.
        """
        payload = event.payload
        reg = payload.get("normalized_registration_number") or payload.get("registration_number")
        if reg:
            return str(reg).encode("utf-8")

        cam = payload.get("camera_id") or event.source.camera_id
        trk = payload.get("track_id")
        if cam and trk:
            return f"{cam}:{trk}".encode("utf-8")

        return event.event_id.encode("utf-8")

    def publish_event(
        self,
        topic: str,
        event: EventEnvelope,
        on_delivery: Optional[Callable[[Optional[Exception], EventEnvelope], None]] = None,
    ) -> bool:
        """Serialize and publish an EventEnvelope to a Kafka topic."""
        try:
            val_bytes = serialize_event(event)
            key_bytes = self.derive_partition_key(event)

            if self._is_live and self._live_producer:
                future = self._live_producer.send(topic, key=key_bytes, value=val_bytes)
                if on_delivery:
                    def _cb(metadata):
                        on_delivery(None, event)
                    def _err_cb(exc):
                        on_delivery(exc, event)
                    future.add_callback(_cb).add_errback(_err_cb)
                self._live_producer.flush()
            else:
                self._broker.publish(topic, key=key_bytes, value=val_bytes)
                if on_delivery:
                    on_delivery(None, event)

            self.metrics["events_produced"] += 1
            self.metrics["bytes_produced"] += len(val_bytes)
            return True

        except Exception as e:
            self.metrics["delivery_failures"] += 1
            logger.error("Failed to publish event %s to topic %s: %s", event.event_id, topic, e)
            if on_delivery:
                on_delivery(e, event)
            raise

    def close(self) -> None:
        """Flush and close producer."""
        if self._live_producer:
            try:
                self._live_producer.flush()
                self._live_producer.close()
            except Exception as e:
                logger.error("Error closing KafkaProducer: %s", e)
