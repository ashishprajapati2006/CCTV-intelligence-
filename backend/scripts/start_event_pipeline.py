#!/usr/bin/env python3
"""Step 10 - Start Kafka Event Pipeline CLI.

Orchestrates end-to-end event streaming through Kafka:
  Step 7 Observations (or synthetic feed)
      ↓
  vehicle.anpr (Publisher)
      ↓
  Watchlist Matching Consumer (Worker)
      ↓
  Step 9 Match Decision Engine (Business Logic)
      ↓
  watchlist.matches (Topic)
      ↓
  alerts (Topic)

Usage:
  python scripts/start_event_pipeline.py \
      --observations data/observed/vehicles/observations.jsonl \
      --watchlist data/watchlist/vehicles/watchlist.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.logging import get_logger
from src.events import (
    TOPIC_ALERTS,
    TOPIC_VEHICLE_ANPR,
    TOPIC_WATCHLIST_MATCHES,
    EventConsumer,
    EventEnvelope,
    EventProducer,
    IdempotencyTracker,
    JSONIdempotencyTracker,
    WatchlistMatchingWorker,
)
from src.kafka import (
    InMemoryKafkaBroker,
    KafkaConfig,
    KafkaConsumerWrapper,
    KafkaProducerWrapper,
    check_kafka_connection,
)
from src.watchlist import (
    JSONWatchlistRepository,
    MatchDecisionEngine,
)

logger = get_logger("event_pipeline")


def main():
    ap = argparse.ArgumentParser(description="Step 10 - Start Kafka Event Pipeline")
    ap.add_argument("--observations", default="data/observed/vehicles/observations.jsonl",
                    help="Input observations file to stream (default: data/observed/vehicles/observations.jsonl)")
    ap.add_argument("--watchlist", default="data/watchlist/vehicles/watchlist.json",
                    help="Path to vehicle watchlist (default: data/watchlist/vehicles/watchlist.json)")
    ap.add_argument("--bootstrap", default=None, help="Kafka bootstrap servers override")
    ap.add_argument("--limit", type=int, default=50, help="Max observations to process (default: 50)")
    args = ap.parse_args()

    obs_path = Path(args.observations)
    if not obs_path.exists():
        print(f"[ERROR] Observations file not found: {args.observations}")
        sys.exit(1)

    wl_path = Path(args.watchlist)
    if not wl_path.exists():
        print(f"[ERROR] Watchlist file not found: {args.watchlist}")
        sys.exit(1)

    config = KafkaConfig()
    if args.bootstrap:
        config.bootstrap_servers = args.bootstrap

    is_live = check_kafka_connection(config.bootstrap_servers)

    print("=" * 70)
    print("  GUJARAT POLICE CCTV - KAFKA EVENT PIPELINE (STEP 10)")
    print("=" * 70)
    print(f"  Observations Source  : {args.observations}")
    print(f"  Watchlist Target     : {args.watchlist}")
    print(f"  Bootstrap Servers    : {config.bootstrap_servers}")
    print(f"  Transport Backbone   : {'Live Kafka Cluster' if is_live else 'In-Memory Event Bus (Local Test Mode)'}")
    print("=" * 70 + "\n")

    # 1. Initialize Producer and Consumers
    producer_wrapper = KafkaProducerWrapper(config=config)
    event_producer = EventProducer(producer_wrapper=producer_wrapper)

    repo = JSONWatchlistRepository(vehicles_path=str(wl_path))
    decision_engine = MatchDecisionEngine(repository=repo)
    idempotency_tracker = JSONIdempotencyTracker()

    worker = WatchlistMatchingWorker(
        decision_engine=decision_engine,
        producer=producer_wrapper,
        idempotency_tracker=idempotency_tracker,
    )

    anpr_consumer_wrapper = KafkaConsumerWrapper(topic=TOPIC_VEHICLE_ANPR, config=config)

    # 2. Ingest observations and publish to vehicle.anpr topic
    published_events = []
    with open(obs_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(published_events) >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            obs = json.loads(line)
            evt = event_producer.publish_anpr(
                camera_id=obs.get("camera_id", "unknown"),
                track_id=obs.get("track_id", "unknown"),
                registration_number=obs.get("registration_number", ""),
                normalized_registration_number=obs.get("normalized_registration_number", ""),
                recognition_status=obs.get("status", "CONFIRMED"),
                consensus_score=float(obs.get("consensus_score", 0.94)),
                ocr_confidence=float(obs.get("ocr_confidence", 0.92)),
                recognition_pts_ms=obs.get("recognition_pts_ms"),
                evidence_image=obs.get("evidence_image"),
                source_time=obs.get("source_time"),
                event_id=f"EVT-ANPR-{obs.get('observation_id', len(published_events)+1)}",
                correlation_id=f"CORR-{obs.get('camera_id')}-{obs.get('track_id')}",
            )
            published_events.append(evt)

    print(f"[+] Published {len(published_events)} observation(s) to topic '{TOPIC_VEHICLE_ANPR}'")

    # 3. Process events through the Watchlist Matching Worker
    t0 = time.time()
    results = []

    def _anpr_handler(event: EventEnvelope):
        res = worker.process_anpr_event(event)
        if res:
            results.append(res)

    processed_count = anpr_consumer_wrapper.poll_and_process(
        handler=_anpr_handler,
        max_messages=len(published_events) + 10,
        timeout_ms=3000,
    )
    total_time_ms = (time.time() - t0) * 1000.0
    avg_latency_ms = total_time_ms / max(1, processed_count)

    # 4. Display Pipeline Execution Summary
    print("\n" + "=" * 70)
    print("  KAFKA PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Events Published to '{TOPIC_VEHICLE_ANPR}'     : {len(published_events)}")
    print(f"  Events Consumed from '{TOPIC_VEHICLE_ANPR}'      : {processed_count}")
    print(f"  Duplicate Events Skipped                         : {worker.metrics['duplicate_events_skipped']}")
    print("-" * 70)
    print(f"  Events Published to '{TOPIC_WATCHLIST_MATCHES}'  : {worker.metrics['matches_produced']}")
    print(f"  Events Published to '{TOPIC_ALERTS}'             : {worker.metrics['alerts_produced']}")
    print("-" * 70)
    print(f"  End-to-End Processing Latency                    : {total_time_ms:.2f} ms total (avg {avg_latency_ms:.2f} ms/event)")
    print("=" * 70)

    print("\n  ALERT-READY EVENTS EMITTED TO 'alerts' TOPIC:")
    for r in results:
        alert = r.get("alert_event")
        if alert:
            p = alert["payload"]
            print(f"    - [{p['camera_id']}] {p['registration_number']:<12} -> Priority: {p['priority']:<8} Category: {p['category']:<24} (Match ID: {p['match_id']})")

    print("\n[+] Step 10 Kafka Event Pipeline executed successfully.")


if __name__ == "__main__":
    main()
