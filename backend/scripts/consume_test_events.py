#!/usr/bin/env python3
"""Step 10 - Consume Events from Kafka CLI.

Subscribes to a specified Kafka topic and displays received event envelopes.

Usage:
  python scripts/consume_test_events.py --topic alerts
  python scripts/consume_test_events.py --topic watchlist.matches
  python scripts/consume_test_events.py --topic vehicle.events.dlq
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.events import (
    TOPIC_ALERTS,
    TOPIC_DLQ,
    TOPIC_WATCHLIST_MATCHES,
    EventEnvelope,
)
from src.kafka import KafkaConfig, KafkaConsumerWrapper


def display_event(event: EventEnvelope) -> None:
    print("-" * 65)
    print(f"EVENT: {event.event_id} [{event.event_type}]")
    print(f"  Version     : {event.schema_version}")
    print(f"  Source      : {event.source.system} (Camera: {event.source.camera_id})")
    print(f"  Correlation : {event.correlation_id}")
    print(f"  Created At  : {event.created_at}")
    print("  Payload     :")
    for k, v in event.payload.items():
        if isinstance(v, dict):
            print(f"    {k}:")
            for sub_k, sub_v in v.items():
                print(f"      {sub_k}: {sub_v}")
        else:
            print(f"    {k}: {v}")


def main():
    ap = argparse.ArgumentParser(description="Step 10 - Consume Events from Kafka Topic")
    ap.add_argument("--topic", default="alerts",
                    choices=["alerts", "watchlist.matches", "vehicle.anpr", "vehicle.detections", "vehicle.events.dlq"],
                    help="Kafka topic to consume from (default: alerts)")
    ap.add_argument("--max-messages", type=int, default=10, help="Max messages to poll (default: 10)")
    ap.add_argument("--timeout", type=int, default=2000, help="Poll timeout in ms (default: 2000)")
    ap.add_argument("--bootstrap", default=None, help="Kafka bootstrap servers override")
    args = ap.parse_args()

    config = KafkaConfig()
    if args.bootstrap:
        config.bootstrap_servers = args.bootstrap

    consumer = KafkaConsumerWrapper(topic=args.topic, config=config)

    print("=" * 65)
    print(f"  SUBSCRIBED TO KAFKA TOPIC: {args.topic}")
    print("=" * 65)
    print(f"  Group ID         : {config.group_id}")
    print(f"  Bootstrap Servers: {config.bootstrap_servers}")
    print(f"  Transport Mode   : {'Live Broker' if consumer._is_live else 'In-Memory Test Bus'}")
    print("=" * 65)

    events_received = []

    def _handler(event: EventEnvelope):
        events_received.append(event)
        display_event(event)

    consumer.poll_and_process(handler=_handler, max_messages=args.max_messages, timeout_ms=args.timeout)

    print("-" * 65)
    print(f"Total events consumed from '{args.topic}': {len(events_received)}\n")


if __name__ == "__main__":
    main()
