"""Vehicle Journey Builder Orchestrator for Step 11.

Constructs complete VehicleJourney instances from raw Step 7 observations:
  Loader -> Correlator -> SequenceBuilder -> PlausibilityAnalyzer -> JourneyScorer -> VehicleJourney.

Step 11.1 additions:
  - SourceTimeResolver is wired in to populate time_resolution, spatial_resolution, limitations.
"""
from __future__ import annotations

from typing import List, Optional

from src.journey.correlation import ObservationCorrelator
from src.journey.models import (
    CrossCameraObservation,
    OrderingMode,
    VehicleJourney,
)
from src.journey.observation_loader import ObservationLoader
from src.journey.plausibility import PlausibilityAnalyzer
from src.journey.scoring import JourneyScorer
from src.journey.sequence import SequenceBuilder
from src.journey.source_time import SourceTimeResolver
from src.watchlist.normalizer import normalize_watchlist_registration


class VehicleJourneyBuilder:
    """End-to-end journey reconstruction service."""

    def __init__(
        self,
        loader: Optional[ObservationLoader] = None,
        correlator: Optional[ObservationCorrelator] = None,
        sequence_builder: Optional[SequenceBuilder] = None,
        plausibility_analyzer: Optional[PlausibilityAnalyzer] = None,
        scorer: Optional[JourneyScorer] = None,
        source_time_resolver: Optional[SourceTimeResolver] = None,
    ):
        self.loader = loader or ObservationLoader()
        self.correlator = correlator or ObservationCorrelator()
        self.sequence_builder = sequence_builder or SequenceBuilder()
        self.plausibility_analyzer = plausibility_analyzer or PlausibilityAnalyzer()
        self.scorer = scorer or JourneyScorer()
        self.source_time_resolver = source_time_resolver or SourceTimeResolver()

    def build_journey_for_vehicle(
        self,
        registration_number: str,
        include_probable: bool = False,
    ) -> Optional[VehicleJourney]:
        """Load observations from Step 7 database and build vehicle journey."""
        norm_reg = normalize_watchlist_registration(registration_number)
        observations = self.loader.load_observations_for_vehicle(
            norm_reg,
            include_probable=include_probable,
        )
        if not observations:
            return None

        return self.build_from_observations(observations, registration_number=norm_reg)

    def build_from_observations(
        self,
        observations: List[CrossCameraObservation],
        registration_number: Optional[str] = None,
    ) -> VehicleJourney:
        """Construct VehicleJourney directly from a list of CrossCameraObservations."""
        if not observations:
            reg = registration_number or "UNKNOWN"
            return VehicleJourney(
                journey_id=f"JRN-{reg}-EMPTY",
                vehicle_id=f"VEH-{reg}",
                registration_number=reg,
                normalized_registration_number=reg,
                ordering_mode=OrderingMode.CAMERA_LOCAL.value,
                status="NO_OBSERVATIONS",
                limitations=["No observations found for this registration"],
            )

        reg = registration_number or observations[0].normalized_registration_number
        veh_id = observations[0].vehicle_id

        # 1. Correlate intra-camera observations into segments
        segments = self.correlator.build_camera_segments(observations)

        # 2. Build temporal sequence (SOURCE_TIME vs CAMERA_LOCAL)
        ordered_segments, ordering_mode = self.sequence_builder.build_sequence(segments)

        # 3. Build journey legs & evaluate plausibility
        legs = self.plausibility_analyzer.build_journey_legs(ordered_segments)

        # 4. Score journey confidence
        confidence = self.scorer.score_journey(ordered_segments, legs)

        # 5. Compute totals
        total_dist = None
        distances = [leg.straight_line_distance_m for leg in legs if leg.straight_line_distance_m is not None]
        if distances:
            total_dist = round(sum(distances), 1)

        total_duration = None
        durations = [leg.time_delta_seconds for leg in legs if leg.time_delta_seconds is not None]
        if durations and len(durations) == len(legs) and ordering_mode == OrderingMode.SOURCE_TIME:
            total_duration = round(sum(durations), 1)

        # 6. Determine journey status
        status = (
            "RECONSTRUCTED"
            if ordering_mode == OrderingMode.SOURCE_TIME
            else "OBSERVATION_SEQUENCE_ONLY"
        )

        # 7. Derive time_resolution and spatial_resolution for Part 24 schema
        resolved_count = sum(1 for s in ordered_segments if s.source_time_status == "RESOLVED")
        if resolved_count == len(ordered_segments) and ordered_segments:
            time_resolution = "RESOLVED"
        elif resolved_count > 0:
            time_resolution = "PARTIAL"
        else:
            time_resolution = "NOT_RESOLVED"

        coord_count = sum(
            1 for s in ordered_segments
            if s.camera_latitude is not None and s.camera_longitude is not None
        )
        if coord_count == len(ordered_segments) and ordered_segments:
            spatial_resolution = "AVAILABLE"
        elif coord_count > 0:
            spatial_resolution = "PARTIAL"
        else:
            spatial_resolution = "UNAVAILABLE"

        # 8. Build limitations list (transparent disclosure)
        limitations: List[str] = []
        if time_resolution == "NOT_RESOLVED":
            limitations.append(
                "Source/calendar time is unresolved — journey uses camera-local media PTS only. "
                "No time anchor (wall_time, server_epoch) found in Sentinel camera catalogue."
            )
        if spatial_resolution == "UNAVAILABLE":
            limitations.append(
                "Camera geographic coordinates are unavailable — "
                "straight-line distances and implied speeds cannot be calculated."
            )
        if len(ordered_segments) == 1:
            limitations.append(
                "Only one camera observation segment — no cross-camera journey legs can be formed."
            )

        journey_id = f"JRN-{reg}"

        return VehicleJourney(
            journey_id=journey_id,
            vehicle_id=veh_id,
            registration_number=reg,
            normalized_registration_number=reg,
            ordering_mode=ordering_mode.value,
            status=status,
            segments=ordered_segments,
            legs=legs,
            confidence_score=confidence,
            total_distance_m=total_dist,
            total_duration_seconds=total_duration,
            time_resolution=time_resolution,
            spatial_resolution=spatial_resolution,
            limitations=limitations,
        )
