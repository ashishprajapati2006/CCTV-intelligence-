"""Observed Vehicle Database Package (Step 7).

Manages persistent storage and querying of vehicles observed by CCTV.
Strictly separates observed ground-truth data from watchlists.
"""
from src.observed.models import (
    ObservedVehicle,
    SightingRecord,
    SupportingFrameRef,
    TimelineEntry,
    UncertainObservation,
    VehicleObservation,
    VehicleTrackRecord,
)
from src.observed.repository import (
    JSONObservedVehicleRepository,
    ObservedVehicleRepository,
)
from src.observed.aggregator import VehicleAggregator
from src.observed.importer import ANPRStep6Importer
from src.observed.queries import ObservedVehicleQueries

__all__ = [
    "ObservedVehicle",
    "SightingRecord",
    "SupportingFrameRef",
    "TimelineEntry",
    "UncertainObservation",
    "VehicleObservation",
    "VehicleTrackRecord",
    "ObservedVehicleRepository",
    "JSONObservedVehicleRepository",
    "VehicleAggregator",
    "ANPRStep6Importer",
    "ObservedVehicleQueries",
]
