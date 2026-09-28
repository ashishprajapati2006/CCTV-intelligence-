"""Event Serialization and Validation for Step 10 Kafka Event Pipeline.

Provides robust JSON encoding/decoding and envelope validation.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from src.events.schemas import CURRENT_SCHEMA_VERSION, EventEnvelope


class EventSerializationError(Exception):
    """Raised when an event fails serialization or deserialization."""
    pass


class EventValidationError(Exception):
    """Raised when an event fails schema envelope validation."""
    pass


def serialize_event(event: EventEnvelope) -> bytes:
    """Serialize an EventEnvelope into UTF-8 encoded JSON bytes."""
    try:
        data = event.to_dict()
        return json.dumps(data, ensure_ascii=False).encode("utf-8")
    except Exception as e:
        raise EventSerializationError(f"Failed to serialize event {event.event_id}: {e}") from e


def deserialize_event(raw_bytes: bytes) -> EventEnvelope:
    """Deserialize UTF-8 encoded JSON bytes into an EventEnvelope.

    Validates:
      1. Valid JSON format
      2. Required envelope fields
      3. Schema version compatibility
    """
    try:
        text = raw_bytes.decode("utf-8")
        data = json.loads(text)
    except Exception as e:
        raise EventSerializationError(f"Malformed JSON payload: {e}") from e

    if not isinstance(data, dict):
        raise EventValidationError(f"Expected JSON object, got {type(data).__name__}")

    # Validate envelope requirements
    required_fields = ["event_id", "event_type", "schema_version", "source", "payload"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise EventValidationError(f"Missing required envelope fields: {missing}")

    version = str(data.get("schema_version", ""))
    if not version.startswith("1."):
        raise EventValidationError(
            f"Unsupported schema_version '{version}'. Compatible versions: 1.x"
        )

    return EventEnvelope.from_dict(data)
