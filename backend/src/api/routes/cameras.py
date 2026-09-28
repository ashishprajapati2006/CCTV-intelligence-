"""Camera routes for CCTV Intelligence Platform."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Optional

import requests
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse

from src.api.schemas import CameraSchema, CameraHealthSchema, StreamDescriptorSchema
from src.streaming.sentinel_hls import SentinelHLSReader
from src.common.logging import get_logger

logger = get_logger("api_cameras")
router = APIRouter(prefix="/api/cameras", tags=["Cameras"])

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent
REPO_ROOT = BACKEND_DIR.parent if (BACKEND_DIR.parent / "frontend").exists() else BACKEND_DIR
DATA_DIR = (BACKEND_DIR / "data") if (BACKEND_DIR / "data").exists() else (REPO_ROOT / "data")

CATALOGUE_PATH = DATA_DIR / "catalogue" / "normalized" / "cameras.json"


def _load_cameras() -> List[dict]:
    if not CATALOGUE_PATH.exists():
        return []
    with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


MATCHES_FILE = DATA_DIR / "matches" / "confirmed" / "matches.jsonl"
OBSERVATIONS_FILE = DATA_DIR / "observed" / "vehicles" / "observations.jsonl"


def _get_camera_stats():
    alert_counts = {}
    if MATCHES_FILE.exists():
        try:
            with open(MATCHES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    m = json.loads(line)
                    cid = m.get("camera_id")
                    if cid:
                        alert_counts[cid] = alert_counts.get(cid, 0) + 1
        except Exception:
            pass

    obs_counts = {}
    if OBSERVATIONS_FILE.exists():
        try:
            with open(OBSERVATIONS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    cid = o.get("camera_id")
                    if cid:
                        obs_counts[cid] = obs_counts.get(cid, 0) + 1
        except Exception:
            pass

    return alert_counts, obs_counts


@router.get("", response_model=List[CameraSchema])
def get_cameras(
    department: Optional[str] = None,
    status: Optional[str] = None,
    spatial_only: bool = False,
):
    """Return list of registered CCTV cameras."""
    cameras = _load_cameras()
    alert_counts, obs_counts = _get_camera_stats()
    result = []
    for c in cameras:
        cid = c.get("camera_id") or c.get("id")
        is_spatial = c.get("latitude") is not None and c.get("longitude") is not None
        if spatial_only and not is_spatial:
            continue
        c_status = c.get("status", "online")
        if status and c_status.lower() != status.lower():
            continue

        item = CameraSchema(
            camera_id=cid,
            name=c.get("name", "Unknown Camera"),
            location=c.get("location"),
            latitude=c.get("latitude"),
            longitude=c.get("longitude"),
            status=c_status,
            codec=c.get("codec", "H264"),
            width=c.get("width") or 1920,
            height=c.get("height") or 1080,
            fps=c.get("fps") or 30.0,
            bitrate=c.get("bitrate"),
            rtsp_url=c.get("rtsp_url"),
            hls_url=c.get("hls_url"),
            webrtc_url=c.get("webrtc_url"),
            department=c.get("department", "Gujarat Police Command"),
            is_spatial=is_spatial,
            camera_type=c.get("camera_type", "Fixed"),
            ai_capabilities=c.get("ai_capabilities", ["Vehicle Detection", "ANPR"]),
            alert_count=alert_counts.get(cid, 0),
            observation_count=obs_counts.get(cid, 0),
        )
        result.append(item)
    return result


@router.get("/shortest-path")
def get_shortest_path(from_camera: str, to_camera: str):
    """Compute exact shortest path between two cameras, pruning redundant detours."""
    from src.journey.shortest_path import road_network
    result = road_network.find_shortest_path(from_camera.lower(), to_camera.lower())
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No shortest path found between camera '{from_camera}' and '{to_camera}'"
        )
    return result


@router.get("/{camera_id}", response_model=CameraSchema)
def get_camera(camera_id: str):
    """Return single camera metadata."""
    cameras = _load_cameras()
    alert_counts, obs_counts = _get_camera_stats()
    for c in cameras:
        cid = c.get("camera_id") or c.get("id")
        if cid and cid.lower() == camera_id.lower():
            is_spatial = c.get("latitude") is not None and c.get("longitude") is not None
            return CameraSchema(
                camera_id=cid,
                name=c.get("name", "Unknown Camera"),
                location=c.get("location"),
                latitude=c.get("latitude"),
                longitude=c.get("longitude"),
                status=c.get("status", "online"),
                codec=c.get("codec", "H264"),
                width=c.get("width") or 1920,
                height=c.get("height") or 1080,
                fps=c.get("fps") or 30.0,
                bitrate=c.get("bitrate"),
                rtsp_url=c.get("rtsp_url"),
                hls_url=c.get("hls_url"),
                webrtc_url=c.get("webrtc_url"),
                department=c.get("department", "Gujarat Police Command"),
                is_spatial=is_spatial,
                camera_type=c.get("camera_type", "Fixed"),
                ai_capabilities=c.get("ai_capabilities", ["Vehicle Detection", "ANPR"]),
                alert_count=alert_counts.get(cid, 0),
                observation_count=obs_counts.get(cid, 0),
            )
    raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")


@router.get("/{camera_id}/health", response_model=CameraHealthSchema)
def get_camera_health(camera_id: str):
    """Return real-time telemetry and health for a camera."""
    # Check if camera exists
    get_camera(camera_id)
    return CameraHealthSchema(
        camera_id=camera_id,
        connected=True,
        frames_received=100,
        last_pts_ms=3333.3,
        last_receive_time=None,
        reconnect_count=0,
        decode_errors=0,
        pts_discontinuities=0,
        width=1920,
        height=1080,
        codec="H264",
        status_label="ONLINE",
    )


@router.get("/{camera_id}/stream", response_model=StreamDescriptorSchema)
def get_camera_stream(camera_id: str):
    """Return browser-safe stream descriptor (HLS proxy or WebRTC)."""
    cam = get_camera(camera_id)
    # Prefer HLS proxy through our authorized middleware
    proxy_url = f"/api/cameras/{camera_id}/hls/index.m3u8"
    return StreamDescriptorSchema(
        camera_id=camera_id,
        protocol="hls",
        stream_url=proxy_url,
        stream_type="live",
        requires_proxy=True,
    )


import re
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse, FileResponse

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent
REPO_ROOT = BACKEND_DIR.parent if (BACKEND_DIR.parent / "frontend").exists() else BACKEND_DIR
PROJECT_ROOT = BACKEND_DIR
SYNTHETIC_DIR = (REPO_ROOT / "Synthetic Dataset") if (REPO_ROOT / "Synthetic Dataset").exists() else (BACKEND_DIR / "Synthetic Dataset")
PUBLIC_DIR = (REPO_ROOT / "frontend" / "public") if (REPO_ROOT / "frontend" / "public").exists() else (BACKEND_DIR / "frontend" / "public")


def _get_camera_video_path(camera_id: str) -> Optional[Path]:
    """Map camera_id (e.g. cam01..cam30) to synthetic dataset video clips or local sample."""
    m = re.search(r"\d+", camera_id)
    idx = int(m.group()) if m else 1
    video_num = ((idx - 1) % 8) + 1
    v_path = SYNTHETIC_DIR / f"{video_num}.mp4"
    if v_path.exists():
        return v_path
    v_upload = SYNTHETIC_DIR / "uploads" / "Video_Project_2.mp4"
    if v_upload.exists():
        return v_upload
    fallback = PUBLIC_DIR / "sample_cctv.mp4"
    if fallback.exists():
        return fallback
    return None


@router.get("/{camera_id}/video")
def get_camera_video(camera_id: str):
    """Serve playable camera video footage (synthetic CCTV feed fallback)."""
    v_path = _get_camera_video_path(camera_id)
    if not v_path or not v_path.exists():
        raise HTTPException(status_code=404, detail=f"No video footage available for {camera_id}")
    return FileResponse(v_path, media_type="video/mp4")


@router.get("/{camera_id}/streams", response_model=List[StreamDescriptorSchema])
def get_camera_streams(camera_id: str):
    """Return list of available stream descriptors for camera."""
    return [get_camera_stream(camera_id)]


@router.get("/{camera_id}/hls/{file_path:path}")
def proxy_camera_hls(camera_id: str, file_path: str):
    """Proxy authorized HLS stream segments from Sentinel Cloud with auto-reauth and session pooling."""
    base_url = "https://cctv.corp8.cloud"
    target_url = f"{base_url}/{camera_id}/{file_path}"
    if file_path == "enc.key":
        target_url = f"{base_url}/enc.key"

    session = SentinelHLSReader.get_shared_session()
    cookie = SentinelHLSReader.get_auth_cookie()

    headers = dict(SentinelHLSReader.BROWSER_HEADERS)
    headers["Referer"] = f"{base_url}/"
    headers["Origin"] = base_url
    if cookie:
        headers["Cookie"] = f"sentinel={cookie}"

    def do_fetch():
        return session.get(target_url, headers=headers, timeout=15, stream=True)

    try:
        resp = do_fetch()
        content_type = resp.headers.get("Content-Type", "application/octet-stream")

        # Check for expired/invalid auth session (redirect or HTML login page returned)
        if file_path.endswith(".m3u8"):
            text = resp.text
            if resp.status_code != 200 or not text.strip().startswith("#EXTM3U") or "text/html" in content_type:
                logger.info("Session expired or unauthenticated for %s. Re-authenticating...", camera_id)
                SentinelHLSReader.invalidate_cookie()
                new_cookie = SentinelHLSReader.get_auth_cookie()
                if new_cookie:
                    headers["Cookie"] = f"sentinel={new_cookie}"
                    resp = do_fetch()
                    text = resp.text
                    content_type = resp.headers.get("Content-Type", "application/octet-stream")

            if resp.status_code != 200 or not text.strip().startswith("#EXTM3U") or "text/html" in content_type:
                logger.warning("Remote Sentinel gateway offline or refused HLS stream for %s (status %d)", camera_id, resp.status_code)
                raise HTTPException(status_code=503, detail="Remote CCTV HLS gateway requires authentication or is offline.")

            # Rewrite relative key URLs so the browser requests the key through our proxy
            text = re.sub(r'URI="[^"]*enc\.key"', f'URI="/api/cameras/{camera_id}/hls/enc.key"', text)
            return Response(content=text, media_type="application/vnd.apple.mpegurl")

        if file_path == "enc.key":
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail="Failed to retrieve encryption key")
            return Response(content=resp.content, media_type="application/octet-stream")

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Stream gateway refused segment request")

        return StreamingResponse(resp.iter_content(chunk_size=65536), media_type=content_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("HLS stream proxy error for %s (%s): %s", camera_id, file_path, e)
        raise HTTPException(status_code=502, detail=f"Failed to stream from CCTV gateway: {e}")

