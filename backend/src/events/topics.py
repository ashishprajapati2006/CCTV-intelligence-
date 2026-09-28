"""Centralized Kafka Topic Definitions for Step 10 Event Pipeline.

Centralizes all topic names to prevent hardcoded string discrepancies across
producers, consumers, and monitoring utilities.
"""
from __future__ import annotations

# Core Event Topics
TOPIC_VEHICLE_DETECTIONS = "vehicle.detections"
TOPIC_VEHICLE_ANPR = "vehicle.anpr"
TOPIC_WATCHLIST_MATCHES = "watchlist.matches"
TOPIC_ALERTS = "alerts"

# Error / Dead Letter Queue Topic
TOPIC_DLQ = "vehicle.events.dlq"

# Optional Extension Topics
TOPIC_CAMERA_HEALTH = "camera.health"
TOPIC_VEHICLE_JOURNEYS = "vehicle.journeys"

ALL_STANDARD_TOPICS = [
    TOPIC_VEHICLE_DETECTIONS,
    TOPIC_VEHICLE_ANPR,
    TOPIC_WATCHLIST_MATCHES,
    TOPIC_ALERTS,
    TOPIC_DLQ,
]
