#!/usr/bin/env python3
"""Step 10 - Inspect Kafka Topics & Pipeline Health CLI.

Checks Kafka broker connectivity, inspects topic definitions,
and prints message queue sizes.

Usage:
  python scripts/inspect_topics.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.events.topics import ALL_STANDARD_TOPICS
from src.kafka import (
    InMemoryKafkaBroker,
    KafkaConfig,
    check_kafka_connection,
)


def main():
    ap = argparse.ArgumentParser(description="Step 10 - Inspect Kafka Topics & Health")
    ap.add_argument("--bootstrap", default=None, help="Kafka bootstrap servers (default from env)")
    args = ap.parse_args()

    config = KafkaConfig()
    if args.bootstrap:
        config.bootstrap_servers = args.bootstrap

    is_live = check_kafka_connection(config.bootstrap_servers)
    broker = InMemoryKafkaBroker.get_instance()

    print("=" * 68)
    print("  KAFKA EVENT BACKBONE INSPECTION")
    print("=" * 68)
    print(f"  Target Bootstrap Servers : {config.bootstrap_servers}")
    print(f"  Broker Connectivity      : {'CONNECTED (Live Broker)' if is_live else 'OFFLINE (Using In-Memory Test Bus)'}")
    print(f"  Configured Client ID     : {config.client_id}")
    print(f"  Configured Consumer Group: {config.group_id}")
    print("-" * 68)
    print(f"{'TOPIC NAME':<28} {'PURPOSE':<28} {'QUEUED MSGS'}")
    print("-" * 68)

    purposes = {
        "vehicle.detections": "Raw vehicle detections (Step 4)",
        "vehicle.anpr": "ANPR plate recognitions (Step 6/7)",
        "watchlist.matches": "All match decisions (Step 9)",
        "alerts": "Alert-ready hits (Step 9/10)",
        "vehicle.events.dlq": "Dead letter queue (errors)",
    }

    for topic in ALL_STANDARD_TOPICS:
        purpose = purposes.get(topic, "General event bus")
        count = len(broker.topics.get(topic, []))
        print(f"{topic:<28} {purpose:<28} {count}")

    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
