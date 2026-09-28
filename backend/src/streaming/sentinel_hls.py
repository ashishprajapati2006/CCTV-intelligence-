"""Sentinel Cloud HLS Stream Ingestion and Recording Module."""
import os
import re
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import Generator, Optional, Tuple, Dict, Any

from dotenv import load_dotenv
import requests
import numpy as np

from src.common.logging import get_logger, mask_sensitive
from src.common.time import get_monotonic_time
from src.streaming.health import StreamHealth

import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()
logger = get_logger("sentinel_hls")


class SentinelHLSReader:
    """Reads real-time video frames or downloads clips from authenticated Sentinel HLS feeds."""

    DEFAULT_BASE_URL = "https://cctv.corp8.cloud"
    BROWSER_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Referer": "https://cctv.corp8.cloud/",
        "Origin": "https://cctv.corp8.cloud",
        "Sec-Ch-Ua": "\"Chromium\";v=\"130\", \"Google Chrome\";v=\"130\", \"Not?A_Brand\";v=\"99\"",
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": "\"Windows\"",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }

    _cached_cookie: Optional[str] = None
    _cookie_timestamp: float = 0.0
    _auth_lock = threading.Lock()
    _shared_session: Optional[requests.Session] = None

    @classmethod
    def get_shared_session(cls) -> requests.Session:
        if cls._shared_session is None:
            session = requests.Session()
            session.headers.update(cls.BROWSER_HEADERS)
            retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[500, 502, 504])
            adapter = HTTPAdapter(pool_connections=30, pool_maxsize=30, max_retries=retries)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            cls._shared_session = session
        return cls._shared_session

    @classmethod
    def invalidate_cookie(cls):
        with cls._auth_lock:
            cls._cached_cookie = None
            cls._cookie_timestamp = 0.0

    def __init__(
        self,
        camera_id: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        base_url: Optional[str] = None,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
    ):
        self.camera_id = camera_id.lower()
        self.email = email or os.getenv("SENTINEL_AUTH_EMAIL")
        self.password = password or os.getenv("SENTINEL_AUTH_PASSWORD")
        self.base_url = (base_url or os.getenv("SENTINEL_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size = self.width * self.height * 3

        self.hls_url = f"{self.base_url}/{self.camera_id}/index.m3u8"
        self.health = StreamHealth(camera_id=self.camera_id, codec="h264")
        self.proc: Optional[subprocess.Popen] = None
        self._is_running: bool = False
        self.frame_sequence: int = 0

    @classmethod
    def get_auth_cookie(cls, email: Optional[str] = None, password: Optional[str] = None, base_url: str = DEFAULT_BASE_URL) -> Optional[str]:
        """Authenticate with Sentinel portal and return session cookie."""
        # Direct cookie override from env if present
        direct_cookie = os.getenv("SENTINEL_AUTH_COOKIE")
        if direct_cookie and direct_cookie.strip():
            return direct_cookie.strip()

        with cls._auth_lock:
            now = time.time()
            # Reuse cookie if fresh (within 4 hours)
            if cls._cached_cookie and (now - cls._cookie_timestamp) < 14400:
                return cls._cached_cookie

            load_dotenv(override=True)
            email = email or os.getenv("SENTINEL_AUTH_EMAIL")
            password = password or os.getenv("SENTINEL_AUTH_PASSWORD")
            if not email or not password:
                logger.warning("No email or password provided for Sentinel authentication.")
                return None

            login_url = f"{base_url}/auth/login"
            session = cls.get_shared_session()

            try:
                logger.info("Authenticating with Sentinel portal for email: %s", email)
                resp = session.post(
                    login_url,
                    data={"email": email, "password": password},
                    timeout=15,
                    allow_redirects=False,
                )
                cookie_val = session.cookies.get("sentinel")
                if not cookie_val and resp.status_code in [200, 302, 303]:
                    cookie_val = resp.cookies.get("sentinel")

                if cookie_val:
                    cls._cached_cookie = cookie_val
                    cls._cookie_timestamp = now
                    logger.info("Successfully acquired Sentinel session cookie.")
                    return cookie_val
                else:
                    logger.error("Failed to acquire Sentinel session cookie (status %d)", resp.status_code)
                    return None
            except Exception as e:
                logger.error("Sentinel authentication request failed: %s", mask_sensitive(str(e)))
                return None

    def _build_ffmpeg_headers(self) -> str:
        """Build newline-delimited headers string for FFmpeg."""
        cookie = self.get_auth_cookie(self.email, self.password, self.base_url)
        headers = [
            f"User-Agent: {self.BROWSER_HEADERS['User-Agent']}",
            f"Referer: {self.base_url}/",
            f"Origin: {self.base_url}",
        ]
        if cookie:
            headers.append(f"Cookie: sentinel={cookie}")
        return "\r\n".join(headers) + "\r\n"

    def connect(self) -> bool:
        """Check availability of the HLS stream."""
        cookie = self.get_auth_cookie(self.email, self.password, self.base_url)
        session = requests.Session()
        session.headers.update(self.BROWSER_HEADERS)
        if cookie:
            session.cookies.set("sentinel", cookie)

        try:
            r = session.get(self.hls_url, timeout=10)
            if r.status_code == 200 and ("#EXTM3U" in r.text or "application" in r.headers.get("Content-Type", "")):
                self.health.connected = True
                logger.info("Sentinel HLS stream reachable for %s (%s)", self.camera_id, self.hls_url)
                return True
            else:
                logger.warning("Sentinel HLS stream returned status %d for %s", r.status_code, self.camera_id)
                self.health.connected = False
                return False
        except Exception as e:
            logger.error("Error probing Sentinel HLS for %s: %s", self.camera_id, mask_sensitive(str(e)))
            self.health.connected = False
            return False

    def stream_frames(
        self,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[int, np.ndarray, float, float], None, None]:
        """Yield decoded (frame_sequence, frame_bgr, media_pts_ms, monotonic_time)."""
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        headers_arg = self._build_ffmpeg_headers()

        cmd = [
            ffmpeg_bin,
            "-headers", headers_arg,
            "-i", self.hls_url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-v", "quiet",
            "-"
        ]

        logger.info("Launching FFmpeg HLS stream pipeline for '%s'...", self.camera_id)
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_size * 5,
        )
        self._is_running = True
        self.health.connected = True
        self.frame_sequence = 0

        frame_interval_ms = 1000.0 / self.fps

        try:
            while self._is_running:
                if max_frames is not None and self.frame_sequence >= max_frames:
                    logger.info("[%s] Reached target validation frame count (%d).", self.camera_id, max_frames)
                    break

                raw_bytes = self.proc.stdout.read(self.frame_size)
                if not raw_bytes or len(raw_bytes) < self.frame_size:
                    logger.info("[%s] Stream ended or disconnected.", self.camera_id)
                    break

                self.frame_sequence += 1
                local_recv_time = get_monotonic_time()
                media_pts_ms = (self.frame_sequence - 1) * frame_interval_ms

                frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((self.height, self.width, 3))
                self.health.record_frame(
                    pts_ms=media_pts_ms,
                    receive_monotonic=local_recv_time,
                    width=self.width,
                    height=self.height,
                    codec="h264",
                )

                yield self.frame_sequence, frame, media_pts_ms, local_recv_time

        finally:
            self.release()

    def download_clip(
        self,
        output_path: str,
        duration_seconds: int = 30,
    ) -> bool:
        """Download a continuous MP4 clip of the camera footage."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        headers_arg = self._build_ffmpeg_headers()

        cmd = [
            ffmpeg_bin,
            "-y",
            "-headers", headers_arg,
            "-i", self.hls_url,
            "-t", str(duration_seconds),
            "-c", "copy",
            str(out),
        ]

        logger.info("Downloading %ds clip for '%s' to %s...", duration_seconds, self.camera_id, out)
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0 and out.exists() and out.stat().st_size > 1000:
            logger.info("Successfully downloaded clip: %s (%d bytes)", out, out.stat().st_size)
            return True
        else:
            logger.error("Failed to download clip: %s", p.stderr[-300:])
            return False

    def release(self) -> None:
        """Release FFmpeg process and streams."""
        self._is_running = False
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
        self.health.connected = False
