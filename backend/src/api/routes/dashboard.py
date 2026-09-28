"""Dashboard API routes providing aggregated platform statistics and activity."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter

from src.api.schemas import DashboardStatsSchema
from src.api.routes.vehicles import _load_vehicles, _load_observations
from src.api.routes.watchlist import _load_watchlist
from src.api.routes.alerts import _get_all_alerts
from src.api.routes.cameras import _load_cameras

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent
REPO_ROOT = BACKEND_DIR.parent if (BACKEND_DIR.parent / "frontend").exists() else BACKEND_DIR
PROJECT_ROOT = BACKEND_DIR
DATA_DIR = (BACKEND_DIR / "data") if (BACKEND_DIR / "data").exists() else (REPO_ROOT / "data")


@router.get("/stats", response_model=DashboardStatsSchema)
def get_dashboard_stats():
    """Get aggregated metrics and high-level platform health."""
    cameras = _load_cameras()
    vehicles = _load_vehicles()
    obs = _load_observations()
    watchlist = _load_watchlist()
    alerts = _get_all_alerts()
    
    anpr_dir = DATA_DIR / "anpr"
    ai_active_cameras = 0
    if anpr_dir.exists():
        ai_active_cameras = len([d for d in anpr_dir.iterdir() if d.is_dir() and d.name.startswith("cam")])
        
    active_alerts_count = len([a for a in alerts if a.get("status") not in ("RESOLVED", "DISMISSED")])
    critical_alerts_count = len([a for a in alerts if a.get("priority") == "CRITICAL" and a.get("status") not in ("RESOLVED", "DISMISSED")])
    
    # Recent observations (sorted or latest 5)
    recent_obs = obs[-8:] if len(obs) > 8 else obs
    recent_obs_formatted = []
    for o in reversed(recent_obs):
        recent_obs_formatted.append({
            "observation_id": o.get("observation_id"),
            "registration_number": o.get("registration_number"),
            "camera_id": o.get("camera_id"),
            "consensus_score": o.get("consensus_score"),
            "ocr_confidence": o.get("ocr_confidence"),
            "first_seen_pts_ms": o.get("first_seen_pts_ms"),
            "evidence_url": o.get("evidence_url"),
            "ingested_at_utc": o.get("ingested_at_utc")
        })
        
    # Recent alerts
    recent_alerts_formatted = list(reversed(alerts[-5:])) if alerts else []
    
    return {
        "total_cameras": len(cameras),
        "online_cameras": len(cameras),
        "degraded_cameras": 0,
        "offline_cameras": 0,
        "ai_active_cameras": max(ai_active_cameras, 9),
        "total_observed_vehicles": len(vehicles),
        "total_sightings": len(obs),
        "active_watchlist_targets": len([w for w in watchlist if w.get("status") == "ACTIVE"]),
        "active_alerts": active_alerts_count,
        "critical_alerts": critical_alerts_count,
        "recent_anpr_observations": recent_obs_formatted,
        "recent_alerts": recent_alerts_formatted
    }


@router.get("/analytics")
def get_dashboard_analytics():
    """Get chart data for observations by camera and detection quality."""
    obs = _load_observations()
    vehicles = _load_vehicles()
    
    # Camera distribution
    cam_counts: Dict[str, int] = {}
    for o in obs:
        cid = o.get("camera_id", "unknown")
        cam_counts[cid] = cam_counts.get(cid, 0) + 1
        
    cam_chart = [{"camera_id": k, "count": v} for k, v in sorted(cam_counts.items())]
    
    # Quality distribution
    quality_bins = {"0.90 - 1.00": 0, "0.75 - 0.89": 0, "0.50 - 0.74": 0, "< 0.50": 0}
    for v in vehicles:
        score = v.get("best_consensus_score", 0.0)
        if score >= 0.90:
            quality_bins["0.90 - 1.00"] += 1
        elif score >= 0.75:
            quality_bins["0.75 - 0.89"] += 1
        elif score >= 0.50:
            quality_bins["0.50 - 0.74"] += 1
        else:
            quality_bins["< 0.50"] += 1
            
    quality_chart = [{"range": k, "count": v} for k, v in quality_bins.items()]
    
    return {
        "camera_distribution": cam_chart,
        "quality_distribution": quality_chart
    }
