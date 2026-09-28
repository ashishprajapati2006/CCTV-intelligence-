"""SourceTimeResolver for Step 11.1 — Cross-Camera Journey Hardening.

Inspects camera catalogue metadata and observation fields to determine whether
a wall-clock / calendar timestamp can be derived for a given observation.

IMPORTANT — STRICT RULES:
  1. NEVER compute: datetime.now() + pts_ms
  2. NEVER compute: ingestion_time + pts_ms
  3. If no verified anchor timestamp is present in camera metadata → NOT_RESOLVED.
  4. Only return RESOLVED when a genuine calendar-time anchor exists and
     the arithmetic is provably correct.

Current state (2026-09-01):
  All 30 Sentinel cameras have latitude=null, longitude=null, timezone=null,
  extra={} in cameras.json.  The raw catalogue provides only 'id' and 'name'.
  No wall_time, server_epoch, slot_offset, or any time anchor is available.
  Therefore SourceTimeResolver will always return NOT_RESOLVED until external
  enrichment data is provided.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Resolution result type
# ---------------------------------------------------------------------------

class SourceTimeResult:
    """Immutable result of a source-time resolution attempt."""

    __slots__ = (
        "source_time",
        "status",
        "method",
        "confidence",
        "explanation",
    )

    def __init__(
        self,
        *,
        source_time: Optional[str],
        status: str,
        method: str,
        confidence: float,
        explanation: str,
    ) -> None:
        self.source_time = source_time
        self.status = status           # "RESOLVED" | "NOT_RESOLVED" | "PARTIAL"
        self.method = method
        self.confidence = confidence   # 0.0 – 1.0
        self.explanation = explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_time": self.source_time,
            "status": self.status,
            "method": self.method,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SourceTimeResult(status={self.status!r}, method={self.method!r}, "
            f"confidence={self.confidence})"
        )


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class SourceTimeResolver:
    """Attempts to derive a verified calendar timestamp for an observation.

    Resolution hierarchy (checked in order):
      1. Observation already carries a verified source_time (status=RESOLVED)
         → pass-through; return as-is.
      2. Camera metadata contains a 'wall_time' anchor in extra/top-level fields
         → NOT implemented yet; returns NOT_RESOLVED.
      3. Camera metadata contains 'server_epoch' + 'slot_offset' or 'slot_seconds'
         → NOT implemented yet; returns NOT_RESOLVED.
      4. All other cases → NOT_RESOLVED.

    MUST NOT USE:
      - datetime.now()
      - ingestion_time + pts_ms
      - Any fabricated timestamp
    """

    # Fields inspected in camera metadata (from cameras.json / extra dict)
    _ANCHOR_FIELDS = (
        "wall_time",
        "server_epoch",
        "slot_offset",
        "slot_seconds",
        "offset",
    )

    def resolve(
        self,
        camera_metadata: Dict[str, Any],
        pts_ms: Optional[float],
        loop_instance: Optional[int] = None,
        existing_source_time: Optional[str] = None,
        existing_source_time_status: Optional[str] = None,
    ) -> SourceTimeResult:
        """Attempt to resolve a calendar timestamp for this observation.

        Parameters
        ----------
        camera_metadata:
            Dict from cameras.json for this camera (may be empty).
        pts_ms:
            Camera-local presentation timestamp in milliseconds.
        loop_instance:
            Loop/session counter if known from upstream pipeline; None otherwise.
        existing_source_time:
            source_time string already on the observation (if any).
        existing_source_time_status:
            source_time_status already on the observation (if any).

        Returns
        -------
        SourceTimeResult
        """
        # Case 1: Observation already has a verified resolved source time
        if (
            existing_source_time_status == "RESOLVED"
            and existing_source_time is not None
        ):
            return SourceTimeResult(
                source_time=existing_source_time,
                status="RESOLVED",
                method="PASSTHROUGH_EXISTING",
                confidence=1.0,
                explanation="Source time already resolved upstream; passed through unchanged.",
            )

        # Inspect available metadata fields
        available_fields = self._inspect_anchor_fields(camera_metadata)

        if not available_fields:
            return SourceTimeResult(
                source_time=None,
                status="NOT_RESOLVED",
                method="NO_ANCHOR_IN_CATALOGUE",
                confidence=0.0,
                explanation=(
                    "Camera catalogue does not contain any time anchor field "
                    f"({', '.join(self._ANCHOR_FIELDS)}). "
                    "Source/calendar time cannot be derived without a verified anchor. "
                    "Computation from datetime.now() or ingestion_time is prohibited."
                ),
            )

        # Fields exist but resolution logic for this anchor type is not yet implemented
        field_list = ", ".join(available_fields)
        return SourceTimeResult(
            source_time=None,
            status="NOT_RESOLVED",
            method="ANCHOR_PRESENT_BUT_UNIMPLEMENTED",
            confidence=0.0,
            explanation=(
                f"Camera metadata contains anchor field(s): {field_list}. "
                "Resolution logic for this anchor type is not yet implemented. "
                "Source time remains unresolved."
            ),
        )

    def inspect_camera_metadata(
        self, camera_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return a structured report of available/missing time anchor fields."""
        available = self._inspect_anchor_fields(camera_metadata)
        missing = [f for f in self._ANCHOR_FIELDS if f not in available]
        has_timezone = bool(
            camera_metadata.get("timezone")
            or (camera_metadata.get("extra") or {}).get("timezone")
        )
        has_coordinates = (
            camera_metadata.get("latitude") is not None
            and camera_metadata.get("longitude") is not None
        )
        return {
            "camera_id": camera_metadata.get("camera_id", "unknown"),
            "anchor_fields_present": available,
            "anchor_fields_missing": missing,
            "has_timezone": has_timezone,
            "has_coordinates": has_coordinates,
            "resolvable": len(available) > 0,
            "resolution_blocked_reason": (
                None if available else "No time anchor fields in camera catalogue"
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inspect_anchor_fields(self, camera_metadata: Dict[str, Any]) -> list:
        """Return list of anchor fields that are actually present and non-null."""
        found = []
        extra = camera_metadata.get("extra") or {}
        for field in self._ANCHOR_FIELDS:
            top_val = camera_metadata.get(field)
            extra_val = extra.get(field)
            if top_val is not None or extra_val is not None:
                found.append(field)
        return found
