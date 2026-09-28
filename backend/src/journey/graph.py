"""Observation Graph for Step 11.1 — Cross-Camera Correlation.

Builds a graph of camera-to-camera edges when the same vehicle registration
appears at both cameras.  Edges are unordered unless source time is resolved.

Generates camera_transition_statistics.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.logging import get_logger

logger = get_logger("journey_graph")

_STATS_PATH = Path("data/journeys/reports/camera_transition_statistics.json")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraEdge:
    """Directed or undirected edge between two cameras sharing a vehicle observation."""
    camera_a: str
    camera_b: str
    shared_vehicles: List[str] = field(default_factory=list)
    observation_id_pairs: List[Tuple[str, str]] = field(default_factory=list)
    temporal_order_known: bool = False  # True only if source_time is RESOLVED for both
    vehicle_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuple list to list of lists for JSON serialisation
        d["observation_id_pairs"] = [list(p) for p in self.observation_id_pairs]
        return d


@dataclass
class ObservationGraphReport:
    """Summary report of the observation graph."""
    total_cameras_in_graph: int
    total_edges: int
    total_shared_vehicle_instances: int
    edges: List[CameraEdge] = field(default_factory=list)
    generated_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "total_cameras_in_graph": self.total_cameras_in_graph,
            "total_edges": self.total_edges,
            "total_shared_vehicle_instances": self.total_shared_vehicle_instances,
            "edges": [e.to_dict() for e in self.edges],
            "generated_at_utc": self.generated_at_utc,
        }
        return d


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class ObservationGraph:
    """Camera adjacency graph derived from shared vehicle observations.

    Usage
    -----
    graph = ObservationGraph()
    graph.add_observation(obs)           # CrossCameraObservation or dict
    graph.build()
    shared = graph.get_shared_vehicles("cam01", "cam07")
    graph.save_statistics()
    """

    def __init__(self) -> None:
        # vehicle_reg -> {camera_id -> [observation_id, ...]}
        self._vehicle_cameras: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # vehicle_reg -> {camera_id -> source_time_status}
        self._vehicle_time_status: Dict[str, Dict[str, str]] = defaultdict(dict)
        # (cam_a, cam_b) unordered key -> CameraEdge
        self._edges: Dict[Tuple[str, str], CameraEdge] = {}
        self._built = False

    def add_observation(self, obs: Any) -> None:
        """Add a single observation (dict or CrossCameraObservation)."""
        if isinstance(obs, dict):
            reg = obs.get("normalized_registration_number") or obs.get("registration_number", "")
            cam = obs.get("camera_id", "unknown")
            obs_id = obs.get("observation_id", "")
            time_status = obs.get("source_time_status", "NOT_RESOLVED")
        else:
            reg = getattr(obs, "normalized_registration_number", "")
            cam = getattr(obs, "camera_id", "unknown")
            obs_id = getattr(obs, "observation_id", "")
            time_status = getattr(obs, "source_time_status", "NOT_RESOLVED")

        if not reg or not cam:
            return

        self._vehicle_cameras[reg][cam].append(obs_id)
        self._vehicle_time_status[reg][cam] = time_status
        self._built = False

    def add_observations(self, observations: List[Any]) -> None:
        """Add multiple observations at once."""
        for obs in observations:
            self.add_observation(obs)

    def build(self) -> None:
        """Compute camera-to-camera edges from accumulated observations."""
        self._edges = {}

        for reg, cam_map in self._vehicle_cameras.items():
            cameras = sorted(cam_map.keys())
            if len(cameras) < 2:
                continue  # Single-camera vehicle — no cross-camera edge

            # Generate all unordered pairs
            for i in range(len(cameras)):
                for j in range(i + 1, len(cameras)):
                    cam_a = cameras[i]
                    cam_b = cameras[j]
                    edge_key = (cam_a, cam_b)

                    if edge_key not in self._edges:
                        self._edges[edge_key] = CameraEdge(
                            camera_a=cam_a,
                            camera_b=cam_b,
                        )

                    edge = self._edges[edge_key]
                    if reg not in edge.shared_vehicles:
                        edge.shared_vehicles.append(reg)
                        edge.vehicle_count += 1

                    # Record one pair of observation IDs (first from each camera)
                    obs_ids_a = cam_map.get(cam_a, [])
                    obs_ids_b = cam_map.get(cam_b, [])
                    if obs_ids_a and obs_ids_b:
                        pair = (obs_ids_a[0], obs_ids_b[0])
                        if pair not in edge.observation_id_pairs:
                            edge.observation_id_pairs.append(pair)

                    # Temporal order is known only if both sides have RESOLVED time
                    status_a = self._vehicle_time_status[reg].get(cam_a, "NOT_RESOLVED")
                    status_b = self._vehicle_time_status[reg].get(cam_b, "NOT_RESOLVED")
                    if status_a == "RESOLVED" and status_b == "RESOLVED":
                        edge.temporal_order_known = True

        self._built = True
        logger.info("ObservationGraph built: %d edges", len(self._edges))

    def get_shared_vehicles(self, camera_a: str, camera_b: str) -> List[str]:
        """Return list of registration numbers seen at both cameras (unordered pair)."""
        if not self._built:
            self.build()
        key = tuple(sorted([camera_a, camera_b]))
        edge = self._edges.get(key)  # type: ignore[arg-type]
        return list(edge.shared_vehicles) if edge else []

    def get_edges(self) -> List[CameraEdge]:
        """Return all edges."""
        if not self._built:
            self.build()
        return list(self._edges.values())

    def get_cameras_in_graph(self) -> Set[str]:
        """Return all camera IDs that appear in at least one edge."""
        if not self._built:
            self.build()
        cams: Set[str] = set()
        for edge in self._edges.values():
            cams.add(edge.camera_a)
            cams.add(edge.camera_b)
        return cams

    def generate_report(self) -> ObservationGraphReport:
        """Build and return a structured graph report."""
        if not self._built:
            self.build()

        edges = self.get_edges()
        cams = self.get_cameras_in_graph()
        total_shared = sum(e.vehicle_count for e in edges)

        return ObservationGraphReport(
            total_cameras_in_graph=len(cams),
            total_edges=len(edges),
            total_shared_vehicle_instances=total_shared,
            edges=edges,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def save_statistics(
        self, output_path: Path = _STATS_PATH
    ) -> Path:
        """Save camera_transition_statistics.json."""
        report = self.generate_report()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Saved camera transition statistics -> %s", output_path)
        return output_path
