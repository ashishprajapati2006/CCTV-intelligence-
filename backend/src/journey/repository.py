"""Repository Layer for Step 11 Vehicle Journey Persistence.

Provides:
  - JourneyRepository (ABC)
  - JSONJourneyRepository (File-backed storage in data/journeys/)
  - Documented migration path to PostgreSQL / PostGIS.
"""
from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import List, Optional

from src.common.logging import get_logger
from src.journey.models import VehicleJourney

logger = get_logger("journey_repository")


class JourneyRepository(abc.ABC):
    """Abstract interface for vehicle journey storage."""

    @abc.abstractmethod
    def save_journey(self, journey: VehicleJourney) -> None:
        """Persist a vehicle journey."""
        pass

    @abc.abstractmethod
    def get_journey(self, registration_number: str) -> Optional[VehicleJourney]:
        """Retrieve a vehicle journey by registration number."""
        pass

    @abc.abstractmethod
    def list_journeys(self) -> List[VehicleJourney]:
        """List all stored vehicle journeys."""
        pass


class JSONJourneyRepository(JourneyRepository):
    """File-backed JSON repository storing journeys in data/journeys/."""

    def __init__(self, base_dir: str = "data/journeys"):
        self.base_dir = Path(base_dir)
        self.vehicles_dir = self.base_dir / "vehicles"
        self.reports_dir = self.base_dir / "reports"
        self.sequences_dir = self.base_dir / "sequences"

        self.vehicles_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.sequences_dir.mkdir(parents=True, exist_ok=True)

    def save_journey(self, journey: VehicleJourney) -> None:
        """Save journey JSON and generate machine/text reports."""
        reg = journey.normalized_registration_number

        # 1. Save canonical vehicle journey JSON
        veh_file = self.vehicles_dir / f"{reg}.json"
        with open(veh_file, "w", encoding="utf-8") as f:
            json.dump(journey.to_dict(), f, indent=2)

        # 2. Save machine-readable report
        report_json = self.reports_dir / f"{reg}_journey.json"
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(journey.to_dict(), f, indent=2)

        # 3. Generate human-readable text report (Part 20)
        report_txt = self.reports_dir / f"{reg}_journey.txt"
        self._generate_text_report(journey, report_txt)

    def get_journey(self, registration_number: str) -> Optional[VehicleJourney]:
        veh_file = self.vehicles_dir / f"{registration_number}.json"
        if not veh_file.exists():
            return None
        try:
            with open(veh_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return VehicleJourney.from_dict(data)
        except Exception as e:
            logger.error("Failed to load journey from %s: %s", veh_file, e)
            return None

    def list_journeys(self) -> List[VehicleJourney]:
        journeys = []
        for p in self.vehicles_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    journeys.append(VehicleJourney.from_dict(data))
            except Exception as e:
                logger.error("Failed to read journey file %s: %s", p, e)
        return journeys

    def _generate_text_report(self, journey: VehicleJourney, output_path: Path) -> None:
        """Render human-readable text report matching Part 16/17/20 specifications."""
        lines = [
            "=" * 70,
            f"VEHICLE JOURNEY / OBSERVATION REPORT",
            "=" * 70,
            f"VEHICLE:               {journey.registration_number}",
            f"VEHICLE ID:            {journey.vehicle_id}",
            f"ORDERING MODE:         {journey.ordering_mode}",
            f"STATUS:                {journey.status}",
            f"TOTAL SEGMENTS:        {len(journey.segments)}",
            f"TOTAL LEGS:            {len(journey.legs)}",
        ]

        if journey.confidence_score:
            cs = journey.confidence_score
            lines.extend([
                "-" * 70,
                f"CONFIDENCE SCORE:      {cs.overall_score:.2f} (Engineering Indicator)",
                f"  - Recognition Quality:  {cs.recognition_quality:.2f}",
                f"  - Temporal Resolution:  {cs.temporal_resolution:.2f}",
                f"  - Spatial Resolution:   {cs.spatial_resolution:.2f}",
                f"  - Plausibility:         {cs.plausibility_consistency:.2f}",
            ])
            for note in cs.notes:
                lines.append(f"    * {note}")

        lines.extend([
            "-" * 70,
            "OBSERVATION SEGMENTS:",
            "-" * 70,
        ])

        for idx, seg in enumerate(journey.segments, 1):
            cam_info = f"{seg.camera_id}"
            if seg.camera_name:
                cam_info += f" ({seg.camera_name})"
            coords = "No Coordinates"
            if seg.camera_latitude is not None and seg.camera_longitude is not None:
                coords = f"{seg.camera_latitude:.5f}, {seg.camera_longitude:.5f}"

            lines.append(f"{idx}. Camera: {cam_info} | Track: {seg.track_id} | Coords: {coords}")
            if seg.source_time_status == "RESOLVED":
                lines.append(f"   Source Time:          {seg.source_time}")
            else:
                lines.append(f"   Time Basis:           CAMERA-LOCAL MEDIA PTS (not globally comparable)")
                lines.append(f"   PTS First / Rec / Last: {seg.first_seen_pts_ms or 0.0:.1f} ms / {seg.recognition_pts_ms or 0.0:.1f} ms / {seg.last_seen_pts_ms or 0.0:.1f} ms")
            lines.append(f"   Status / Consensus:   {seg.recognition_status} (Score: {seg.consensus_score:.2f}, OCR: {seg.ocr_confidence:.2f})")
            lines.append(f"   Evidence Image:       {seg.evidence_image or 'None'}")
            lines.append("")

        if journey.legs:
            lines.extend([
                "-" * 70,
                "JOURNEY LEGS (Cross-Camera Transit):",
                "-" * 70,
            ])
            for idx, leg in enumerate(journey.legs, 1):
                from_c = f"{leg.from_camera_id}" + (f" ({leg.from_camera_name})" if leg.from_camera_name else "")
                to_c = f"{leg.to_camera_id}" + (f" ({leg.to_camera_name})" if leg.to_camera_name else "")
                lines.append(f"Leg {idx}: {from_c} -> {to_c}")
                if leg.time_delta_seconds is not None:
                    mins = leg.time_delta_seconds / 60.0
                    lines.append(f"  Time Delta:             {leg.time_delta_seconds:.1f} s ({mins:.1f} mins)")
                else:
                    lines.append(f"  Time Delta:             UNRESOLVED (Media PTS not comparable)")

                if leg.straight_line_distance_m is not None:
                    lines.append(f"  Approx Distance:        {leg.straight_line_distance_m:.1f} m (straight-line)")
                else:
                    lines.append(f"  Approx Distance:        UNAVAILABLE (missing camera coordinates)")

                if leg.implied_speed_kmh is not None:
                    lines.append(f"  Implied Speed:          {leg.implied_speed_kmh:.1f} km/h (straight-line indicator)")

                lines.append(f"  Plausibility:           {leg.plausibility.value} ({leg.plausibility_reason})")
                lines.append("")

        if journey.ordering_mode == "CAMERA_LOCAL":
            lines.extend([
                "=" * 70,
                "NOTICE: Camera-local PTS values are not globally comparable across cameras.",
                "This output represents an unverified camera-local observation sequence, NOT a validated route.",
                "=" * 70,
            ])
        else:
            lines.append("=" * 70)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
