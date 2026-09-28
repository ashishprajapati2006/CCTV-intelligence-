"""Idempotency and Duplicate Event Suppression for Step 10 Kafka Event Pipeline.

Prevents duplicate execution and alert flooding when Kafka delivers messages
more than once under at-least-once semantics.
"""
from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Optional, Set

from src.common.logging import get_logger

logger = get_logger("kafka_idempotency")


class IdempotencyTracker(abc.ABC):
    """Abstract interface for event idempotency and duplicate detection."""

    @abc.abstractmethod
    def is_processed(self, event_id: str) -> bool:
        """Check if an event_id has already been successfully processed."""
        pass

    @abc.abstractmethod
    def mark_processed(self, event_id: str) -> None:
        """Record an event_id as successfully processed."""
        pass

    @abc.abstractmethod
    def count(self) -> int:
        """Total count of tracked processed events."""
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear the idempotency store (primarily for testing)."""
        pass


class JSONIdempotencyTracker(IdempotencyTracker):
    """File-backed JSON idempotency store.

    Stores processed event IDs in `data/events/processed_event_ids.json`.
    Designed for seamless migration to PostgreSQL table: processed_events.
    """

    def __init__(self, storage_path: str = "data/events/processed_event_ids.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._processed_ids: Set[str] = set()
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._processed_ids = set(data)
            except Exception as e:
                logger.error("Failed to load processed event IDs from %s: %s", self.storage_path, e)

    def _flush_to_disk(self) -> None:
        try:
            tmp_file = self.storage_path.with_suffix(".tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(list(self._processed_ids), f, indent=2)
            tmp_file.replace(self.storage_path)
        except Exception as e:
            logger.error("Failed to flush processed event IDs to disk: %s", e)

    def is_processed(self, event_id: str) -> bool:
        return event_id in self._processed_ids

    def mark_processed(self, event_id: str) -> None:
        if event_id not in self._processed_ids:
            self._processed_ids.add(event_id)
            self._flush_to_disk()

    def count(self) -> int:
        return len(self._processed_ids)

    def clear(self) -> None:
        self._processed_ids.clear()
        if self.storage_path.exists():
            self.storage_path.unlink()
