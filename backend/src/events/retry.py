"""Retry and Backoff Strategies for Step 10 Kafka Event Pipeline.

Provides exponential backoff and error classification (transient vs permanent).
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Type, TypeVar

from src.common.logging import get_logger
from src.events.serializer import EventSerializationError, EventValidationError

logger = get_logger("kafka_retry")

T = TypeVar("T")

# Permanent errors that must NOT be retried (should route to DLQ immediately)
PERMANENT_ERRORS = (
    EventSerializationError,
    EventValidationError,
    KeyError,
    ValueError,
    TypeError,
)


class RetryConfig:
    """Configurable backoff parameters."""
    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff_sec: float = 1.0,
        backoff_multiplier: float = 2.0,
        max_backoff_sec: float = 30.0,
    ):
        self.max_retries = max_retries
        self.initial_backoff_sec = initial_backoff_sec
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff_sec = max_backoff_sec

    def get_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff time for a given attempt index (0-indexed)."""
        backoff = self.initial_backoff_sec * (self.backoff_multiplier ** attempt)
        return min(backoff, self.max_backoff_sec)


def is_permanent_error(exc: Exception) -> bool:
    """Determine whether an error is permanent (unrecoverable schema/type fault)."""
    return isinstance(exc, PERMANENT_ERRORS)


def execute_with_retry(
    action: Callable[[], T],
    config: Optional[RetryConfig] = None,
    action_name: str = "operation",
) -> T:
    """Execute a callable with exponential backoff on transient failures."""
    cfg = config or RetryConfig()
    attempt = 0

    while True:
        try:
            return action()
        except Exception as exc:
            if is_permanent_error(exc):
                logger.error("Permanent error in %s: %s. Not retrying.", action_name, exc)
                raise

            attempt += 1
            if attempt > cfg.max_retries:
                logger.error("Exceeded max retries (%d) in %s: %s", cfg.max_retries, action_name, exc)
                raise

            delay = cfg.get_backoff(attempt - 1)
            logger.warning(
                "Transient error in %s (attempt %d/%d): %s. Backing off for %.2fs...",
                action_name, attempt, cfg.max_retries, exc, delay
            )
            time.sleep(delay)
