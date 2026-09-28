"""High-level Event Consumer for Step 10 Event Pipeline.

Provides typed event consumption, graceful shutdown on SIGINT/SIGTERM,
and error handling.
"""
from __future__ import annotations

import signal
import sys
from typing import Callable, Optional

from src.common.logging import get_logger
from src.events.schemas import EventEnvelope
from src.kafka.config import KafkaConfig
from src.kafka.consumer import KafkaConsumerWrapper

logger = get_logger("event_consumer")


class EventConsumer:
    """High-level consumer facade wrapping KafkaConsumerWrapper."""

    def __init__(
        self,
        topic: str,
        config: Optional[KafkaConfig] = None,
        consumer_wrapper: Optional[KafkaConsumerWrapper] = None,
    ):
        self.topic = topic
        self.wrapper = consumer_wrapper or KafkaConsumerWrapper(topic=topic, config=config)
        self._running = True
        self._setup_signals()

    def _setup_signals(self) -> None:
        """Register signal handlers for graceful shutdown (Part 20)."""
        def _sig_handler(signum, frame):
            logger.info("Shutdown signal (%d) received. Stopping consumer on %s...", signum, self.topic)
            self._running = False

        try:
            signal.signal(signal.SIGINT, _sig_handler)
            signal.signal(signal.SIGTERM, _sig_handler)
        except (ValueError, AttributeError):
            # Non-main thread or unsupported platform
            pass

    def run_loop(self, handler: Callable[[EventEnvelope], None], poll_interval_sec: float = 0.5) -> None:
        """Continuous polling loop."""
        logger.info("Starting consumer loop on topic '%s'", self.topic)
        while self._running:
            try:
                self.wrapper.poll_and_process(handler=handler, max_messages=20)
            except Exception as e:
                logger.error("Consumer loop exception on %s: %s", self.topic, e)
        self.close()

    def close(self) -> None:
        self.wrapper.close()
