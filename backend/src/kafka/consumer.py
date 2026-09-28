"""Kafka Consumer Layer for Step 10 Event Pipeline.

Provides message consumption with manual offset commits (at-least-once semantics),
graceful shutdown handling, and dead-letter queue fallback.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from src.common.logging import get_logger
from src.events.schemas import EventEnvelope, build_dlq_event
from src.events.serializer import deserialize_event
from src.events.topics import TOPIC_DLQ
from src.kafka.client import InMemoryKafkaBroker, check_kafka_connection
from src.kafka.config import KafkaConfig

logger = get_logger("kafka_consumer")


class KafkaConsumerWrapper:
    """Wrapper over KafkaConsumer supporting manual commit and in-memory test fallback."""

    def __init__(
        self,
        topic: str,
        config: Optional[KafkaConfig] = None,
        dlq_publisher: Optional[Callable[[EventEnvelope], None]] = None,
    ):
        self.topic = topic
        self.config = config or KafkaConfig()
        self.dlq_publisher = dlq_publisher

        self._running = False
        self._is_live = False
        self._live_consumer = None
        self._broker = InMemoryKafkaBroker.get_instance()
        self._current_offset = 0

        # Observability Metrics (Part 12)
        self.metrics = {
            "events_consumed": 0,
            "processing_success": 0,
            "processing_failure": 0,
            "dlq_events": 0,
            "duplicate_events": 0,
            "total_latency_ms": 0.0,
        }

        self._init_consumer()

    def _init_consumer(self) -> None:
        is_reachable = check_kafka_connection(self.config.bootstrap_servers)
        if is_reachable:
            try:
                from kafka import KafkaConsumer
                self._live_consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.config.bootstrap_servers,
                    client_id=self.config.client_id,
                    group_id=self.config.group_id,
                    auto_offset_reset=self.config.auto_offset_reset,
                    enable_auto_commit=self.config.enable_auto_commit,
                    consumer_timeout_ms=1000,
                )
                self._is_live = True
                logger.info("KafkaConsumer connected to %s on topic %s", self.config.bootstrap_servers, self.topic)
                return
            except Exception as e:
                logger.warning("Failed to initialize KafkaConsumer: %s. Using in-memory fallback.", e)

        if self.config.use_mock_fallback:
            self._is_live = False
        else:
            raise ConnectionError(f"Kafka broker unreachable at {self.config.bootstrap_servers}")

    def poll_and_process(
        self,
        handler: Callable[[EventEnvelope], None],
        max_messages: int = 10,
        timeout_ms: int = 1000,
    ) -> int:
        """Poll and process messages, routing unrecoverable errors to DLQ."""
        processed_count = 0

        if self._is_live and self._live_consumer:
            records_dict = self._live_consumer.poll(timeout_ms=timeout_ms, max_records=max_messages)
            for tp, records in records_dict.items():
                for record in records:
                    t_start = time.time()
                    self.metrics["events_consumed"] += 1
                    try:
                        event = deserialize_event(record.value)
                        handler(event)
                        self.metrics["processing_success"] += 1
                        processed_count += 1
                        # Commit offset after successful processing (at-least-once semantics)
                        self._live_consumer.commit()
                    except Exception as exc:
                        self.metrics["processing_failure"] += 1
                        logger.error("Processing failed on topic %s offset %d: %s", self.topic, record.offset, exc)
                        self._route_to_dlq(record.value, exc)
                    finally:
                        elapsed_ms = (time.time() - t_start) * 1000.0
                        self.metrics["total_latency_ms"] += elapsed_ms

        else:
            # In-memory broker mode
            records = self._broker.fetch(self.topic, offset=self._current_offset, max_messages=max_messages)
            for record in records:
                t_start = time.time()
                self.metrics["events_consumed"] += 1
                try:
                    event = deserialize_event(record["value"])
                    handler(event)
                    self.metrics["processing_success"] += 1
                    processed_count += 1
                    self._current_offset = record["offset"] + 1
                except Exception as exc:
                    self.metrics["processing_failure"] += 1
                    logger.error("Processing failed on topic %s offset %d: %s", self.topic, record["offset"], exc)
                    self._route_to_dlq(record["value"], exc)
                    self._current_offset = record["offset"] + 1
                finally:
                    elapsed_ms = (time.time() - t_start) * 1000.0
                    self.metrics["total_latency_ms"] += elapsed_ms

        return processed_count

    def _route_to_dlq(self, raw_bytes: bytes, exc: Exception) -> None:
        """Construct and publish DLQ event for unrecoverable errors."""
        self.metrics["dlq_events"] += 1
        raw_text = None
        try:
            raw_text = raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            pass

        dlq_event = build_dlq_event(
            original_event_id="UNKNOWN",
            original_topic=self.topic,
            error_code=type(exc).__name__,
            error_message=str(exc),
            retry_count=0,
            raw_payload=raw_text,
        )

        if self.dlq_publisher:
            try:
                self.dlq_publisher(dlq_event)
            except Exception as e:
                logger.error("Failed to publish DLQ event: %s", e)
        else:
            # Publish directly to in-memory DLQ topic
            self._broker.publish(TOPIC_DLQ, key=b"dlq", value=raw_bytes)

    def close(self) -> None:
        """Close consumer connection."""
        self._running = False
        if self._live_consumer:
            try:
                self._live_consumer.close()
            except Exception as e:
                logger.error("Error closing KafkaConsumer: %s", e)
