"""Camera Repository with PostGIS Spatial Operations (Step 12).

Provides decoupled data access for camera entities and geospatial queries:
- Nearby cameras within radial distance (meters)
- Distance between two camera installations
- Bounding box spatial containment
- Coordinates filtering (verified vs unverified)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from geoalchemy2 import functions as geofunc
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from src.common.logging import get_logger
from src.database.models.camera import Camera

logger = get_logger("camera_repository")


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Standard Haversine straight-line distance in meters (fallback for non-PostGIS)."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class CameraRepository:
    """Repository handling camera storage and PostGIS spatial queries."""

    def __init__(self, session: Session):
        self.session = session

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        """Fetch camera by camera_id."""
        stmt = select(Camera).where(Camera.camera_id == camera_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_cameras(self) -> List[Camera]:
        """List all cameras sorted by camera_id."""
        stmt = select(Camera).order_by(Camera.camera_id)
        return list(self.session.execute(stmt).scalars().all())

    def cameras_with_coordinates(self) -> List[Camera]:
        """Return cameras that have verified spatial coordinates."""
        stmt = (
            select(Camera)
            .where(Camera.latitude.isnot(None), Camera.longitude.isnot(None))
            .order_by(Camera.camera_id)
        )
        return list(self.session.execute(stmt).scalars().all())

    def cameras_without_coordinates(self) -> List[Camera]:
        """Return cameras that lack verified spatial coordinates."""
        stmt = (
            select(Camera)
            .where((Camera.latitude.is_(None)) | (Camera.longitude.is_(None)))
            .order_by(Camera.camera_id)
        )
        return list(self.session.execute(stmt).scalars().all())

    def upsert_camera(self, data: Dict[str, Any]) -> Camera:
        """Insert or update camera record, ensuring geom is populated only when coordinates exist."""
        camera_id = data["camera_id"]
        camera = self.get_camera(camera_id)

        lat = data.get("latitude")
        lon = data.get("longitude")

        is_postgres = self.session.bind is not None and self.session.bind.dialect.name == "postgresql"

        # Set PostGIS point geometry if coordinates exist on Postgres
        geom_val = None
        if lat is not None and lon is not None:
            if is_postgres:
                geom_val = geofunc.ST_SetSRID(geofunc.ST_MakePoint(float(lon), float(lat)), 4326)
            else:
                geom_val = f"POINT({lon} {lat})"

        if camera is None:
            camera = Camera(
                camera_id=camera_id,
                camera_number=data.get("camera_number"),
                name=data.get("name"),
                location=data.get("location"),
                status=data.get("status", "active"),
                codec=data.get("codec"),
                width=data.get("width"),
                height=data.get("height"),
                fps=data.get("fps"),
                bitrate=data.get("bitrate"),
                rtsp_url=data.get("rtsp_url"),
                hls_url=data.get("hls_url"),
                webrtc_url=data.get("webrtc_url"),
                timezone=data.get("timezone"),
                latitude=lat,
                longitude=lon,
                geom=geom_val,
                extra_metadata=data.get("metadata", {}),
            )
            self.session.add(camera)
        else:
            camera.name = data.get("name", camera.name)
            camera.location = data.get("location", camera.location)
            camera.status = data.get("status", camera.status)
            camera.codec = data.get("codec", camera.codec)
            camera.width = data.get("width", camera.width)
            camera.height = data.get("height", camera.height)
            camera.fps = data.get("fps", camera.fps)
            camera.bitrate = data.get("bitrate", camera.bitrate)
            camera.rtsp_url = data.get("rtsp_url", camera.rtsp_url)
            camera.hls_url = data.get("hls_url", camera.hls_url)
            camera.webrtc_url = data.get("webrtc_url", camera.webrtc_url)
            camera.timezone = data.get("timezone", camera.timezone)
            camera.latitude = lat
            camera.longitude = lon
            camera.geom = geom_val
            if "metadata" in data:
                camera.extra_metadata = data["metadata"]

        self.session.flush()
        return camera

    def nearby_cameras(
        self, lat: float, lon: float, radius_m: float = 5000.0
    ) -> List[Dict[str, Any]]:
        """Find cameras within radius_m meters of given (lat, lon).

        Uses PostGIS ST_DistanceSphere when available on PostgreSQL,
        otherwise computes Haversine distances in Python.
        """
        is_postgres = self.session.bind is not None and self.session.bind.dialect.name == "postgresql"

        results = []
        if is_postgres:
            try:
                # PostGIS query: ST_DistanceSphere returns distance in meters
                sql = text("""
                    SELECT camera_id, name, latitude, longitude,
                           ST_DistanceSphere(geom, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)) AS distance_m
                    FROM cameras
                    WHERE geom IS NOT NULL
                      AND ST_DistanceSphere(geom, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)) <= :radius_m
                    ORDER BY distance_m ASC
                """)
                rows = self.session.execute(sql, {"lat": lat, "lon": lon, "radius_m": radius_m}).fetchall()
                for r in rows:
                    results.append({
                        "camera_id": r[0],
                        "name": r[1],
                        "latitude": r[2],
                        "longitude": r[3],
                        "distance_m": round(float(r[4]), 2),
                    })
                return results
            except Exception as e:
                logger.warning("PostGIS ST_DistanceSphere failed, falling back to Haversine: %s", e)

        # Haversine fallback
        all_spatial = self.cameras_with_coordinates()
        for cam in all_spatial:
            dist = haversine_distance_m(lat, lon, cam.latitude, cam.longitude)  # type: ignore
            if dist <= radius_m:
                results.append({
                    "camera_id": cam.camera_id,
                    "name": cam.name,
                    "latitude": cam.latitude,
                    "longitude": cam.longitude,
                    "distance_m": round(dist, 2),
                })

        results.sort(key=lambda x: x["distance_m"])
        return results

    def camera_distance(self, camera_a_id: str, camera_b_id: str) -> Optional[float]:
        """Compute straight-line distance in meters between two cameras."""
        cam_a = self.get_camera(camera_a_id)
        cam_b = self.get_camera(camera_b_id)

        if not cam_a or not cam_b:
            return None
        if cam_a.latitude is None or cam_a.longitude is None:
            return None
        if cam_b.latitude is None or cam_b.longitude is None:
            return None

        is_postgres = self.session.bind is not None and self.session.bind.dialect.name == "postgresql"
        if is_postgres:
            try:
                sql = text("""
                    SELECT ST_DistanceSphere(
                        ST_SetSRID(ST_MakePoint(:lon1, :lat1), 4326),
                        ST_SetSRID(ST_MakePoint(:lon2, :lat2), 4326)
                    )
                """)
                val = self.session.execute(sql, {
                    "lon1": cam_a.longitude, "lat1": cam_a.latitude,
                    "lon2": cam_b.longitude, "lat2": cam_b.latitude,
                }).scalar()
                if val is not None:
                    return round(float(val), 2)
            except Exception as e:
                logger.warning("PostGIS ST_DistanceSphere error: %s", e)

        return round(haversine_distance_m(cam_a.latitude, cam_a.longitude, cam_b.latitude, cam_b.longitude), 2)

    def cameras_in_bbox(
        self, min_lon: float, min_lat: float, max_lon: float, max_lat: float
    ) -> List[Camera]:
        """Find cameras enclosed within bounding box coordinates."""
        stmt = (
            select(Camera)
            .where(
                Camera.latitude >= min_lat,
                Camera.latitude <= max_lat,
                Camera.longitude >= min_lon,
                Camera.longitude <= max_lon,
            )
            .order_by(Camera.camera_id)
        )
        return list(self.session.execute(stmt).scalars().all())
