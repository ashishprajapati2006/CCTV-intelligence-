"""Alerts API routes with operator action workflows and audit logs."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.api.schemas import AlertSchema, AlertActionRequest

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent
REPO_ROOT = BACKEND_DIR.parent if (BACKEND_DIR.parent / "frontend").exists() else BACKEND_DIR
PROJECT_ROOT = BACKEND_DIR
DATA_DIR = (BACKEND_DIR / "data") if (BACKEND_DIR / "data").exists() else (REPO_ROOT / "data")
MATCHES_FILE = DATA_DIR / "matches" / "confirmed" / "matches.jsonl"
ALERTS_STATE_FILE = DATA_DIR / "matches" / "alerts_state.json"

# In-memory store + disk persistence for operator actions
_alerts_state_cache: Dict[str, Dict[str, Any]] = {}
_listeners: List[asyncio.Queue] = []


def _format_evidence_url(image_path: Optional[str]) -> Optional[str]:
    if not image_path:
        return None
    p = image_path.replace("\\", "/").strip()
    rel = p[5:] if p.startswith("data/") else p
    disk_file = DATA_DIR / rel
    if not disk_file.exists():
        return None
    return f"/api/evidence/{rel}"


def _load_state():
    global _alerts_state_cache
    if ALERTS_STATE_FILE.exists():
        try:
            with open(ALERTS_STATE_FILE, "r", encoding="utf-8") as f:
                _alerts_state_cache = json.load(f)
        except Exception:
            _alerts_state_cache = {}


def _save_state():
    try:
        ALERTS_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ALERTS_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(_alerts_state_cache, f, indent=2)
    except Exception:
        pass


def _get_all_alerts() -> List[Dict[str, Any]]:
    _load_state()
    alerts = []
    
    if MATCHES_FILE.exists():
        try:
            with open(MATCHES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = json.loads(line)
                    alert_id = m.get("match_id", f"ALT-{m.get('observation_id')}")
                    
                    # Watchlist details
                    wl_meta = m.get("watchlist_metadata") or {}
                    category = wl_meta.get("category", "SUSPECT_VEHICLE")
                    priority = wl_meta.get("priority", "HIGH")
                    
                    # Merge operator state if exists
                    state = _alerts_state_cache.get(alert_id, {})
                    status = state.get("status", "NEW")
                    notes = state.get("operator_notes")
                    ack_by = state.get("acknowledged_by")
                    ack_at = state.get("acknowledged_at")
                    audit = state.get("audit_history", [
                        {
                            "timestamp": m.get("matched_at_utc", datetime.now(timezone.utc).isoformat()),
                            "action": "TRIGGERED",
                            "operator": "SYSTEM (AI Watchlist Matcher)",
                            "notes": f"Match detected with score {m.get('watchlist_match_score', 1.0):.2f}"
                        }
                    ])
                    
                    alert_obj = {
                        "alert_id": alert_id,
                        "match_id": m.get("match_id", ""),
                        "observation_id": m.get("observation_id", ""),
                        "registration_number": m.get("registration_number", ""),
                        "normalized_registration_number": m.get("normalized_registration_number", ""),
                        "watchlist_id": m.get("watchlist_id") or "",
                        "category": category,
                        "priority": priority,
                        "decision": m.get("decision", "MATCH"),
                        "camera_id": m.get("camera_id", ""),
                        "track_id": m.get("track_id", ""),
                        "status": status,
                        "recognition_confidence": m.get("recognition_confidence", 0.0),
                        "consensus_score": m.get("consensus_score", 0.0),
                        "first_seen_pts_ms": m.get("first_seen_pts_ms"),
                        "recognition_pts_ms": m.get("recognition_pts_ms"),
                        "source_time": m.get("source_time"),
                        "source_time_status": m.get("source_time_status", "NOT_RESOLVED"),
                        "evidence_image": m.get("evidence_image"),
                        "evidence_url": _format_evidence_url(m.get("evidence_image")),
                        "matched_at_utc": m.get("matched_at_utc", datetime.now(timezone.utc).isoformat()),
                        "acknowledged_by": ack_by,
                        "acknowledged_at": ack_at,
                        "operator_notes": notes,
                        "audit_history": audit
                    }
                    alerts.append(alert_obj)
        except Exception:
            pass
            
    return alerts


@router.get("", response_model=List[AlertSchema])
def list_alerts(
    status: Optional[str] = Query(None, description="Filter by status (NEW, ACKNOWLEDGED, UNDER_REVIEW, ESCALATED, RESOLVED, DISMISSED)"),
    priority: Optional[str] = Query(None, description="Filter by priority (CRITICAL, HIGH, MEDIUM, LOW)"),
    camera_id: Optional[str] = Query(None, description="Filter by camera"),
    search: Optional[str] = Query(None, description="Search vehicle registration"),
):
    """List watchlist match alerts with status and filtering."""
    alerts = _get_all_alerts()
    
    if status:
        alerts = [a for a in alerts if a["status"].upper() == status.upper()]
        
    if priority:
        alerts = [a for a in alerts if a["priority"].upper() == priority.upper()]
        
    if camera_id:
        alerts = [a for a in alerts if a["camera_id"] == camera_id]
        
    if search:
        s = search.strip().upper()
        alerts = [
            a for a in alerts
            if s in a["registration_number"].upper()
            or s in a["normalized_registration_number"].upper()
        ]
        
    return alerts


@router.get("/{alert_id}", response_model=AlertSchema)
def get_alert(alert_id: str):
    """Get alert detail by ID."""
    alerts = _get_all_alerts()
    for a in alerts:
        if a["alert_id"] == alert_id or a["match_id"] == alert_id:
            return a
    raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")


@router.post("/{alert_id}/action", response_model=AlertSchema)
async def update_alert_action(alert_id: str, req: AlertActionRequest):
    """Update alert status (acknowledge, escalate, resolve, dismiss)."""
    alerts = _get_all_alerts()
    target_alert = None
    for a in alerts:
        if a["alert_id"] == alert_id or a["match_id"] == alert_id:
            target_alert = a
            break
            
    if not target_alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
        
    act = req.action.lower()
    now_iso = datetime.now(timezone.utc).isoformat()
    
    status_map = {
        "acknowledge": "ACKNOWLEDGED",
        "escalate": "ESCALATED",
        "resolve": "RESOLVED",
        "dismiss": "DISMISSED",
        "review": "UNDER_REVIEW"
    }
    
    new_status = status_map.get(act, "ACKNOWLEDGED")
    
    cached = _alerts_state_cache.get(target_alert["alert_id"], {
        "status": target_alert["status"],
        "operator_notes": target_alert.get("operator_notes"),
        "acknowledged_by": target_alert.get("acknowledged_by"),
        "acknowledged_at": target_alert.get("acknowledged_at"),
        "audit_history": list(target_alert.get("audit_history", []))
    })
    
    cached["status"] = new_status
    if act == "acknowledge" and not cached.get("acknowledged_by"):
        cached["acknowledged_by"] = req.operator
        cached["acknowledged_at"] = now_iso
    if req.notes:
        cached["operator_notes"] = req.notes
        
    cached["audit_history"].append({
        "timestamp": now_iso,
        "action": act.upper(),
        "operator": req.operator,
        "notes": req.notes or f"Status changed to {new_status}"
    })
    
    _alerts_state_cache[target_alert["alert_id"]] = cached
    _save_state()
    
    # Broadcast to SSE listeners
    for q in _listeners:
        try:
            await q.put(json.dumps({
                "type": "ALERT_UPDATE",
                "alert_id": target_alert["alert_id"],
                "status": new_status,
                "operator": req.operator,
                "timestamp": now_iso
            }))
        except Exception:
            pass
            
    return get_alert(alert_id)


@router.get("/stream/live")
async def stream_alerts():
    """SSE endpoint for live alert notifications."""
    queue: asyncio.Queue = asyncio.Queue()
    _listeners.append(queue)
    
    async def event_generator():
        try:
            # Send initial ping
            yield f"event: connected\ndata: {json.dumps({'message': 'Connected to Alert Stream'})}\n\n"
            while True:
                data = await queue.get()
                yield f"event: alert\ndata: {data}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in _listeners:
                _listeners.remove(queue)
                
    return StreamingResponse(event_generator(), media_type="text/event-stream")
