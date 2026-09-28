"""Cross-Camera Vehicle Correlation & Journey Reconstruction Package (Step 11 / 11.1).

Exports:
  - Models: CrossCameraObservation, CameraObservationSegment, JourneyLeg,
            VehicleJourney, JourneyConfidenceScore, PlausibilityState, OrderingMode
  - Loader: ObservationLoader
  - Temporal: can_order_globally, parse_source_time, calculate_time_delta_seconds,
              validate_source_time, compare_source_times, detect_time_conflict
  - Correlator: ObservationCorrelator
  - Sequence: SequenceBuilder
  - Plausibility: PlausibilityAnalyzer, haversine_distance_meters
  - Scorer: JourneyScorer
  - Builder: VehicleJourneyBuilder
  - Repository: JourneyRepository, JSONJourneyRepository
  - Step 11.1: SourceTimeResolver, SourceTimeResult, LoopDetector, ObservationGraph
"""
from src.journey.correlation import ObservationCorrelator
from src.journey.graph import ObservationGraph
from src.journey.journey_builder import VehicleJourneyBuilder
from src.journey.loop import LoopDetector, LoopTransitionResult
from src.journey.models import (
    CameraObservationSegment,
    CrossCameraObservation,
    JourneyConfidenceScore,
    JourneyLeg,
    OrderingMode,
    PlausibilityState,
    VehicleJourney,
)
from src.journey.observation_loader import ObservationLoader
from src.journey.plausibility import PlausibilityAnalyzer, haversine_distance_meters
from src.journey.repository import JSONJourneyRepository, JourneyRepository
from src.journey.scoring import JourneyScorer
from src.journey.sequence import SequenceBuilder
from src.journey.source_time import SourceTimeResolver, SourceTimeResult
from src.journey.temporal import (
    calculate_time_delta_seconds,
    can_order_globally,
    compare_source_times,
    detect_time_conflict,
    parse_source_time,
    validate_source_time,
)

__all__ = [
    # Models
    "CrossCameraObservation",
    "CameraObservationSegment",
    "JourneyLeg",
    "VehicleJourney",
    "JourneyConfidenceScore",
    "PlausibilityState",
    "OrderingMode",
    # Loader
    "ObservationLoader",
    # Temporal
    "can_order_globally",
    "parse_source_time",
    "calculate_time_delta_seconds",
    "validate_source_time",
    "compare_source_times",
    "detect_time_conflict",
    # Pipeline
    "ObservationCorrelator",
    "SequenceBuilder",
    "PlausibilityAnalyzer",
    "haversine_distance_meters",
    "JourneyScorer",
    "VehicleJourneyBuilder",
    # Repository
    "JourneyRepository",
    "JSONJourneyRepository",
    # Step 11.1
    "SourceTimeResolver",
    "SourceTimeResult",
    "LoopDetector",
    "LoopTransitionResult",
    "ObservationGraph",
]
