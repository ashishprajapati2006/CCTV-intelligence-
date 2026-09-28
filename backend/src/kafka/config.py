"""Kafka Configuration for Step 10 Event Pipeline.

Reads connection parameters from environment variables with sensible local defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class KafkaConfig:
    """Connection and client parameters for Kafka producers and consumers."""
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    client_id: str = os.getenv("KAFKA_CLIENT_ID", "gujarat-cctv-platform")
    group_id: str = os.getenv("KAFKA_GROUP_ID", "cctv-watchlist-consumers")
    auto_offset_reset: str = os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest")
    enable_auto_commit: bool = False  # Manual commit for at-least-once semantics
    request_timeout_ms: int = int(os.getenv("KAFKA_REQUEST_TIMEOUT_MS", "5000"))
    # Fallback to in-memory broker harness when external broker is unreachable
    use_mock_fallback: bool = os.getenv("KAFKA_USE_MOCK_FALLBACK", "true").lower() in ("true", "1", "yes")

    def to_dict(self) -> dict:
        return {
            "bootstrap_servers": self.bootstrap_servers,
            "client_id": self.client_id,
            "group_id": self.group_id,
            "auto_offset_reset": self.auto_offset_reset,
            "enable_auto_commit": self.enable_auto_commit,
            "request_timeout_ms": self.request_timeout_ms,
        }
