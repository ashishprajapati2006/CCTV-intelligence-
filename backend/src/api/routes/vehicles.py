"""Vehicles API routes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import ObservedVehicleSchema, VehicleObservationSchema

router = APIRouter(prefix="/api/vehicles", tags=["vehicles"])

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent
REPO_ROOT = BACKEND_DIR.parent if (BACKEND_DIR.parent / "frontend").exists() else BACKEND_DIR
PROJECT_ROOT = BACKEND_DIR
DATA_DIR = (BACKEND_DIR / "data") if (BACKEND_DIR / "data").exists() else (REPO_ROOT / "data")
VEHICLES_FILE = DATA_DIR / "observed" / "vehicles" / "vehicles.json"
OBSERVATIONS_FILE = DATA_DIR / "observed" / "vehicles" / "observations.jsonl"
JOURNEYS_DIR = DATA_DIR / "journeys" / "vehicles"
JOURNEYS_REPORTS_DIR = DATA_DIR / "journeys" / "reports"


def _format_evidence_url(image_path: Optional[str]) -> Optional[str]:
    if not image_path:
        return None
    p = image_path.replace("\\", "/").strip()
    rel = p[5:] if p.startswith("data/") else p
    disk_file = DATA_DIR / rel
    if not disk_file.exists():
        return None
    return f"/api/evidence/{rel}"


def _load_vehicles() -> List[Dict[str, Any]]:
    if not VEHICLES_FILE.exists():
        return []
    try:
        with open(VEHICLES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            for v in data:
                for item in v.get("timeline", []):
                    item["evidence_url"] = _format_evidence_url(item.get("evidence_image"))
            return data
    except Exception:
        return []


def _load_observations() -> List[Dict[str, Any]]:
    if not OBSERVATIONS_FILE.exists():
        return []
    obs = []
    try:
        with open(OBSERVATIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    item["evidence_url"] = _format_evidence_url(item.get("evidence_image_path"))
                    obs.append(item)
    except Exception:
        pass
    return obs


@router.get("", response_model=List[ObservedVehicleSchema])
def list_vehicles(
    search: Optional[str] = Query(None, description="Search registration number"),
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    min_consensus: Optional[float] = Query(None, description="Minimum consensus score"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List observed vehicles with optional filtering and search."""
    vehicles = _load_vehicles()
    
    if search:
        s = search.strip().upper()
        vehicles = [
            v for v in vehicles
            if s in v.get("registration_number", "").upper()
            or s in v.get("normalized_registration_number", "").upper()
        ]
        
    if camera_id:
        vehicles = [v for v in vehicles if camera_id in v.get("cameras", [])]
        
    if min_consensus is not None:
        vehicles = [v for v in vehicles if v.get("best_consensus_score", 0.0) >= min_consensus]
        
    if status:
        vehicles = [v for v in vehicles if v.get("status", "").upper() == status.upper()]
        
    return vehicles[offset : offset + limit]


@router.get("/{reg}", response_model=ObservedVehicleSchema)
def get_vehicle(reg: str):
    """Get single observed vehicle detail by registration number."""
    vehicles = _load_vehicles()
    reg_clean = reg.strip().upper()
    for v in vehicles:
        if (v.get("registration_number", "").upper() == reg_clean or
            v.get("normalized_registration_number", "").upper() == reg_clean or
            v.get("vehicle_id", "").upper() == reg_clean):
            return v
    raise HTTPException(status_code=404, detail=f"Vehicle '{reg}' not found")


@router.get("/{reg}/observations")
def get_vehicle_observations(reg: str):
    """Get all raw observations for a specific vehicle registration."""
    obs = _load_observations()
    reg_clean = reg.strip().upper()
    results = [
        o for o in obs
        if (o.get("registration_number", "").upper() == reg_clean or
            o.get("normalized_registration_number", "").upper() == reg_clean)
    ]
    return results


@router.get("/{reg}/journey")
def get_vehicle_journey(reg: str):
    """Get reconstructed journey dossier for a vehicle."""
    reg_clean = reg.strip().upper()
    
    target_file = JOURNEYS_DIR / f"{reg_clean}.json"
    if not target_file.exists():
        if JOURNEYS_DIR.exists():
            for f in JOURNEYS_DIR.glob("*.json"):
                if f.stem.upper() == reg_clean:
                    target_file = f
                    break
                    
    if target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                for seg in data.get("segments", []):
                    seg["evidence_url"] = _format_evidence_url(seg.get("evidence_image"))
                return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading journey: {str(e)}")
            
    report_file = JOURNEYS_REPORTS_DIR / f"{reg_clean}_journey.json"
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as rf:
                data = json.load(rf)
                return data
        except Exception:
            pass

    raise HTTPException(status_code=404, detail=f"Journey for vehicle '{reg}' not found")


@router.get("/{reg}/geojson")
def get_vehicle_geojson(reg: str):
    """Get GeoJSON feature collection for a vehicle's journey (if coordinates exist)."""
    try:
        journey = get_vehicle_journey(reg)
    except HTTPException:
        return {"type": "FeatureCollection", "features": []}

    features = []
    coordinates = []
    
    for seg in journey.get("segments", []):
        lat = seg.get("camera_latitude")
        lon = seg.get("camera_longitude")
        if lat is not None and lon is not None:
            coordinates.append([lon, lat])
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "segment_id": seg.get("segment_id"),
                    "camera_id": seg.get("camera_id"),
                    "camera_name": seg.get("camera_name"),
                    "pts_ms": seg.get("first_seen_pts_ms"),
                    "source_time": seg.get("source_time"),
                    "consensus_score": seg.get("consensus_score")
                }
            })
            
    if len(coordinates) >= 2:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
            "properties": {
                "vehicle_id": journey.get("vehicle_id"),
                "registration_number": journey.get("registration_number")
            }
        })
        
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "spatial_resolution": journey.get("spatial_resolution", "UNAVAILABLE"),
            "coordinate_count": len(coordinates)
        }
    }
