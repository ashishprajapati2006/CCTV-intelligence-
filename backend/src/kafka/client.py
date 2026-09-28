"""Kafka Client and Topic Management Utilities for Step 10.

Provides:
  - Broker connection check
  - Topic creation / inspection
  - In-memory test broker harness for offline execution & testing
"""
from __future__ import annotations

import collections
import socket
from typing import Any, Dict, List, Optional, Set

from src.common.logging import get_logger
from src.events.topics import ALL_STANDARD_TOPICS

logger = get_logger("kafka_client")


def check_kafka_connection(bootstrap_servers: str, timeout_sec: float = 1.0) -> bool:
    """Check if the Kafka broker host:port is reachable via TCP socket."""
    for server in bootstrap_servers.split(","):
        server = server.strip()
        if not server:
            continue
        try:
            if ":" in server:
                host, port_str = server.split(":")
                port = int(port_str)
            else:
                host = server
                port = 9092

            with socket.create_connection((host, port), timeout=timeout_sec):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            continue
    return False


class InMemoryKafkaBroker:
    """Thread-safe in-memory message broker simulating Kafka topics for testing and offline development."""
    _instance: Optional[InMemoryKafkaBroker] = None

    @classmethod
    def get_instance(cls) -> InMemoryKafkaBroker:
        if cls._instance is None:
            cls._instance = InMemoryKafkaBroker()
        return cls._instance

    def __init__(self):
        # topic -> list of dict(key, value, headers, offset)
        self.topics: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        # (group_id, topic) -> committed_offset
        self.committed_offsets: Dict[Tuple[str, str], int] = {}
        for t in ALL_STANDARD_TOPICS:
            self.topics[t] = []

    def publish(self, topic: str, key: Optional[bytes], value: bytes, headers: Optional[List] = None) -> int:
        offset = len(self.topics[topic])
        self.topics[topic].append({
            "key": key,
            "value": value,
            "headers": headers or [],
            "offset": offset,
        })
        return offset

    def fetch(self, topic: str, offset: int, max_messages: int = 100) -> List[Dict[str, Any]]:
        messages = self.topics[topic]
        if offset >= len(messages):
            return []
        return messages[offset : offset + max_messages]

    def clear(self) -> None:
        self.topics.clear()
        self.committed_offsets.clear()
        for t in ALL_STANDARD_TOPICS:
            self.topics[t] = []
