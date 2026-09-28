#!/usr/bin/env python3
"""Step 10 - Publish Test ANPR Event to Kafka CLI.

Publishes test vehicle.anpr events into the Kafka event backbone to demonstrate
event streaming, watchlist matching, and duplicate event idempotency.

Usage:
  # Publish known watchlist match:
  python scripts/publish_test_event.py --plate GJ01AB1234

  # Publish non-matching plate:
  python scripts/publish_test_event.py --plate GJ05CD5678

  # Publish duplicate event to test idempotency:
  python scripts/publish_test_event.py --plate GJ01AB1234 --event-id EVT-DUP-001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.events import (
    TOPIC_VEHICLE_ANPR,
    EventProducer,
    build_anpr_event,
)
from src.kafka import KafkaConfig, KafkaProducerWrapper
from src.watchlist.normalizer import normalize_watchlist_registration


def main():
    ap = argparse.ArgumentParser(description="Step 10 - Publish Test ANPR Event to Kafka")
    ap.add_argument("--plate", default="GJ01AB1234", help="Registration plate to publish (default: GJ01AB1234)")
    ap.add_argument("--camera", default="cam01", help="Camera ID (default: cam01)")
    ap.add_argument("--track", default="TRK-0001", help="Track ID (default: TRK-0001)")
    ap.add_argument("--pts", type=float, default=11200.0, help="Recognition PTS in ms (default: 11200.0)")
    ap.add_argument("--status", default="CONFIRMED", help="Recognition status (default: CONFIRMED)")
    ap.add_argument("--consensus", type=float, default=0.94, help="Consensus score (default: 0.94)")
    ap.add_argument("--ocr-conf", type=float, default=0.93, help="OCR confidence (default: 0.93)")
    ap.add_argument("--event-id", default=None, help="Explicit event_id (useful for duplicate tests)")
    ap.add_argument("--bootstrap", default=None, help="Kafka bootstrap servers override")
    args = ap.parse_args()

    config = KafkaConfig()
    if args.bootstrap:
        config.bootstrap_servers = args.bootstrap

    producer_wrapper = KafkaProducerWrapper(config=config)
    producer = EventProducer(producer_wrapper=producer_wrapper)

    norm_reg = normalize_watchlist_registration(args.plate)

    event = producer.publish_anpr(
        camera_id=args.camera,
        track_id=args.track,
        registration_number=args.plate,
        normalized_registration_number=norm_reg,
        recognition_status=args.status,
        consensus_score=args.consensus,
        ocr_confidence=args.ocr_conf,
        recognition_pts_ms=args.pts,
        evidence_image=f"data/snapshots/{args.camera}/plates/{args.track}_pts{int(args.pts)}.jpg",
        event_id=args.event_id,
        correlation_id=f"TRK-{args.camera}-{args.track}",
    )

    print("=" * 68)
    print("  KAFKA EVENT PUBLISHED")
    print("=" * 68)
    print(f"  Topic             : {TOPIC_VEHICLE_ANPR}")
    print(f"  Event ID          : {event.event_id}")
    print(f"  Event Type        : {event.event_type}")
    print(f"  Schema Version    : {event.schema_version}")
    print(f"  Correlation ID    : {event.correlation_id}")
    print(f"  Source System     : {event.source.system} (Camera: {event.source.camera_id})")
    print("-" * 68)
    print(f"  Plate (Raw / Norm): {args.plate} / {norm_reg}")
    print(f"  Status / Consensus: {args.status} (score: {args.consensus:.2f}, ocr: {args.ocr_conf:.2f})")
    print(f"  Transport Mode    : {'Live Broker (' + config.bootstrap_servers + ')' if producer_wrapper._is_live else 'In-Memory Test Bus'}")
    print("=" * 68 + "\n")
    print(json.dumps(event.to_dict(), indent=2))


if __name__ == "__main__":
    main()
