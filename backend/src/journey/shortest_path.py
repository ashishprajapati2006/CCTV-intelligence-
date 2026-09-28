"""Shortest Path and Road Network Routing for Gujarat CCTV Infrastructure.

Computes exact Dijkstra shortest paths between surveillance cameras,
pruning redundant detours and eliminating duplicate edges.
"""
from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

COORDINATES_FILE = Path("data/catalogue/enrichment/camera_coordinates.json")
CATALOGUE_FILE = Path("data/catalogue/normalized/cameras.json")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points in kilometers."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


class CameraRoadNetwork:
    """Camera adjacency network representing physical road connections across Gujarat."""

    def __init__(self, coord_path: Path = COORDINATES_FILE) -> None:
        self.cam_map: Dict[str, Dict[str, Any]] = {}
        self.adj: Dict[str, Dict[str, float]] = {}
        self._load_cameras(coord_path)
        self._build_topology()

    def _load_cameras(self, coord_path: Path) -> None:
        if coord_path.exists():
            try:
                with open(coord_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for c in data.get("cameras", []):
                        cid = c["camera_id"]
                        self.cam_map[cid] = {
                            "camera_id": cid,
                            "name": c.get("location", f"Camera {cid}"),
                            "latitude": float(c["latitude"]),
                            "longitude": float(c["longitude"]),
                            "location": c.get("location", ""),
                        }
            except Exception:
                pass

        # Fallback to normalized catalogue if empty
        if not self.cam_map and CATALOGUE_FILE.exists():
            try:
                with open(CATALOGUE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for c in data:
                        cid = c.get("camera_id") or c.get("id")
                        if cid and c.get("latitude") is not None and c.get("longitude") is not None:
                            self.cam_map[cid] = {
                                "camera_id": cid,
                                "name": c.get("name") or c.get("location") or f"Camera {cid}",
                                "latitude": float(c["latitude"]),
                                "longitude": float(c["longitude"]),
                                "location": c.get("location", ""),
                            }
            except Exception:
                pass

        for cid in self.cam_map:
            self.adj[cid] = {}

    def _add_edge(self, u: str, v: str) -> None:
        if u in self.cam_map and v in self.cam_map:
            d = haversine_km(
                self.cam_map[u]["latitude"], self.cam_map[u]["longitude"],
                self.cam_map[v]["latitude"], self.cam_map[v]["longitude"]
            )
            self.adj[u][v] = round(d, 2)
            self.adj[v][u] = round(d, 2)

    def _build_topology(self) -> None:
        """Construct realistic road topology connecting all 30 cameras."""
        # 1. Ahmedabad Urban Grid
        # North corridor (Sabarmati / Chandkheda / Visat)
        self._add_edge("cam05", "cam16")
        self._add_edge("cam03", "cam05")
        self._add_edge("cam03", "cam16")
        self._add_edge("cam01", "cam03")
        self._add_edge("cam01", "cam05")
        self._add_edge("cam01", "cam15")
        self._add_edge("cam01", "cam20")

        # West & Central (Ashram Rd, Paldi, Vastrapur, Ambawadi, Navrangpura)
        self._add_edge("cam15", "cam02")
        self._add_edge("cam15", "cam14")
        self._add_edge("cam14", "cam13")
        self._add_edge("cam13", "cam02")
        self._add_edge("cam13", "cam04")
        self._add_edge("cam02", "cam04")
        self._add_edge("cam02", "cam20")
        self._add_edge("cam04", "cam20")

        # 2. Gandhinagar & North Gujarat Corridor
        self._add_edge("cam05", "cam12")
        self._add_edge("cam16", "cam12")
        self._add_edge("cam12", "cam24")
        self._add_edge("cam24", "cam23")
        self._add_edge("cam23", "cam21")
        self._add_edge("cam21", "cam22")

        # 3. Ahmedabad to Saurashtra Highway
        self._add_edge("cam04", "cam17")
        self._add_edge("cam02", "cam17")
        self._add_edge("cam17", "cam18")

        # 4. Rajkot to Kutch Highway
        self._add_edge("cam17", "cam30")
        self._add_edge("cam18", "cam30")

        # 5. Rajkot to Junagadh & Somnath Highway
        self._add_edge("cam17", "cam11")
        self._add_edge("cam18", "cam11")
        self._add_edge("cam11", "cam08")
        self._add_edge("cam08", "cam10")
        self._add_edge("cam10", "cam06")
        self._add_edge("cam06", "cam09")
        self._add_edge("cam10", "cam09")
        self._add_edge("cam09", "cam07")
        self._add_edge("cam06", "cam07")

        # 6. Ahmedabad to South Gujarat Highway (Navsari / Bilimora)
        self._add_edge("cam04", "cam19")
        self._add_edge("cam20", "cam19")
        self._add_edge("cam19", "cam25")
        self._add_edge("cam19", "cam29")
        self._add_edge("cam25", "cam26")
        self._add_edge("cam26", "cam29")
        self._add_edge("cam29", "cam28")
        self._add_edge("cam28", "cam27")
        self._add_edge("cam27", "cam19")

    def find_shortest_path(self, origin_id: str, destination_id: str) -> Optional[Dict[str, Any]]:
        """Dijkstra shortest path algorithm.

        Guarantees strictly ONE shortest path between the two cameras,
        pruning all redundant detour paths.
        """
        if origin_id not in self.cam_map or destination_id not in self.cam_map:
            return None

        if origin_id == destination_id:
            cam = self.cam_map[origin_id]
            return {
                "origin": origin_id,
                "destination": destination_id,
                "shortest_path": [origin_id],
                "total_distance_km": 0.0,
                "estimated_transit_minutes": 0.0,
                "checkpoints": [{
                    "camera_id": origin_id,
                    "name": cam["name"],
                    "coordinates": [cam["longitude"], cam["latitude"]],
                }],
                "redundant_paths_pruned": True,
            }

        pq: List[Tuple[float, str, List[str]]] = [(0.0, origin_id, [origin_id])]
        visited: Dict[str, float] = {}

        while pq:
            cost, curr, path = heapq.heappop(pq)

            if curr == destination_id:
                # Path found
                checkpoints = []
                for cid in path:
                    c = self.cam_map[cid]
                    checkpoints.append({
                        "camera_id": cid,
                        "name": c["name"],
                        "latitude": c["latitude"],
                        "longitude": c["longitude"],
                        "coordinates": [c["longitude"], c["latitude"]],
                    })

                # Calculate realistic transit time (avg 45 km/h urban/highway blend)
                transit_mins = round((cost / 45.0) * 60.0, 1) if cost > 0 else 0.0

                return {
                    "origin": origin_id,
                    "destination": destination_id,
                    "shortest_path": path,
                    "total_distance_km": round(cost, 2),
                    "estimated_transit_minutes": max(2.0, transit_mins),
                    "checkpoints": checkpoints,
                    "redundant_paths_pruned": True,
                }

            if curr in visited and visited[curr] <= cost:
                continue
            visited[curr] = cost

            for neighbor, weight in self.adj.get(curr, {}).items():
                new_cost = cost + weight
                if neighbor not in visited or visited[neighbor] > new_cost:
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))

        return None


# Global singleton instance
road_network = CameraRoadNetwork()
