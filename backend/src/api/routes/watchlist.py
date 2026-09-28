"""Watchlist API routes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.api.schemas import WatchlistEntrySchema

router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent
REPO_ROOT = BACKEND_DIR.parent if (BACKEND_DIR.parent / "frontend").exists() else BACKEND_DIR
PROJECT_ROOT = BACKEND_DIR
WATCHLIST_FILE = (BACKEND_DIR / "data" / "watchlist" / "vehicles" / "watchlist.json") if (BACKEND_DIR / "data").exists() else (REPO_ROOT / "data" / "watchlist" / "vehicles" / "watchlist.json")


class CreateWatchlistTargetRequest(BaseModel):
    registration_number: str
    category: str = "DEMO_SUSPECT_VEHICLE"
    priority: str = "HIGH"  # CRITICAL, HIGH, MEDIUM, LOW
    description: Optional[str] = "Synthetic demonstration target"
    notes: Optional[str] = None


def _load_watchlist() -> List[Dict[str, Any]]:
    if not WATCHLIST_FILE.exists():
        return []
    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_watchlist(data: List[Dict[str, Any]]) -> None:
    WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@router.get("", response_model=List[WatchlistEntrySchema])
def list_watchlist(
    search: Optional[str] = Query(None, description="Search registration number"),
    category: Optional[str] = Query(None, description="Filter by category"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    status: Optional[str] = Query(None, description="Filter by status"),
):
    """List all watchlist targets with SYNTHETIC_DEMO badging."""
    items = _load_watchlist()
    
    if search:
        s = search.strip().upper()
        items = [
            i for i in items
            if s in i.get("registration_number", "").upper()
            or s in i.get("normalized_registration_number", "").upper()
        ]
        
    if category:
        items = [i for i in items if i.get("category", "").upper() == category.upper()]
        
    if priority:
        items = [i for i in items if i.get("priority", "").upper() == priority.upper()]
        
    if status:
        items = [i for i in items if i.get("status", "").upper() == status.upper()]
        
    # Map created_at to created_at_utc if needed
    for i in items:
        if "created_at" in i and "created_at_utc" not in i:
            i["created_at_utc"] = i["created_at"]
        if "description" in i and "reason" not in i:
            i["reason"] = i["description"]
            
    return items


@router.post("", response_model=WatchlistEntrySchema)
def add_watchlist_target(req: CreateWatchlistTargetRequest):
    """Add a new synthetic demo target to the watchlist."""
    items = _load_watchlist()
    reg_clean = req.registration_number.strip().upper()
    
    # Check duplicate
    for i in items:
        if i.get("normalized_registration_number", "").upper() == reg_clean:
            raise HTTPException(status_code=400, detail=f"Target '{reg_clean}' already exists in watchlist")
            
    new_id = f"WL-VEH-{len(items) + 1:06d}"
    now_iso = datetime.now(timezone.utc).isoformat()
    
    new_entry = {
        "watchlist_id": new_id,
        "registration_number": reg_clean,
        "normalized_registration_number": reg_clean,
        "category": req.category,
        "priority": req.priority,
        "status": "ACTIVE",
        "entity_type": "vehicle",
        "source": "SYNTHETIC_DEMO",
        "synthetic": True,
        "description": req.description or "Synthetic demonstration record",
        "notes": req.notes or "Created from Command Centre",
        "created_at": now_iso,
        "created_at_utc": now_iso,
        "reason": req.description or "Synthetic demonstration record",
        "updated_at": now_iso
    }
    
    items.append(new_entry)
    _save_watchlist(items)
    try:
        from src.api.routes.synthetic import refresh_watchlist_index
        refresh_watchlist_index()
    except Exception:
        pass
    return new_entry
