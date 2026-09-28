"""RTSP Stream Ingestion with TCP enforcement and PTS timing."""
import os
import re
import time
from typing import Generator, Optional, Tuple, Dict, Any
import cv2
import numpy as np

from src.common.logging import get_logger
from src.common.time import get_monotonic_time
from src.streaming.health import StreamHealth
from src.streaming.reconnect import ExponentialBackoff

logger = get_logger("rtsp_stream")


class RTSPStreamReader:
    """Manages OpenCV VideoCapture connection to an RTSP stream.

    Guarantees:
    1. RTSP transport is strictly forced to TCP.
    2. Timing is anchored on media container PTS (cv2.CAP_PROP_POS_MSEC).
    3. Reconnection is managed via exponential backoff.
    4. Initial H264/H265 decoder warnings do not immediately terminate the stream.
    5. Clean resource release.
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        codec: Optional[str] = None,
        force_tcp: bool = True,
        forward_discontinuity_threshold_ms: float = 5000.0,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.codec = codec or "UNKNOWN"
        self.force_tcp = force_tcp
        self.forward_discontinuity_threshold_ms = forward_discontinuity_threshold_ms

        self.health = StreamHealth(camera_id=camera_id, codec=self.codec)
        self.backoff = ExponentialBackoff()
        self.cap: Optional[cv2.VideoCapture] = None
        self._hls_reader = None
        self._is_running: bool = False
        self.frame_sequence: int = 0

    def _configure_transport(self) -> None:
        """Enforce RTSP over TCP for OpenCV FFmpeg backend."""
        if self.force_tcp:
            # Force RTSP transport over TCP
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            logger.debug("Forced OpenCV FFmpeg capture option: rtsp_transport;tcp")

    def connect(self) -> bool:
        """Attempt to open the RTSP capture stream."""
        self._configure_transport()
        self.release()

        # Check if direct HLS stream is provided
        if ".m3u8" in self.rtsp_url or "cctv.corp8.cloud" in self.rtsp_url:
            from src.streaming.sentinel_hls import SentinelHLSReader
            self._hls_reader = SentinelHLSReader(self.camera_id)
            if self._hls_reader.connect():
                self.health.connected = True
                return True
            return False

        target_url = self.rtsp_url
        auth_user = os.getenv("RTSP_AUTH_USER", "")
        auth_pass = os.getenv("RTSP_AUTH_PASSWORD") or os.getenv("SENTINEL_AUTH_PASSWORD")
        if "@" not in target_url and target_url.startswith("rtsp://") and (auth_user or auth_pass):
            user_part = f"{auth_user}:{auth_pass}" if auth_pass else auth_user
            target_url = target_url.replace("rtsp://", f"rtsp://{user_part}@")

        masked_url = re.sub(r":([^@]+)@", ":***@", target_url)
        logger.info(
            "Connecting to RTSP stream for '%s' over TCP: %s",
            self.camera_id,
            masked_url,
        )

        try:
            self.cap = cv2.VideoCapture(target_url, cv2.CAP_FFMPEG)
            if self.cap and self.cap.isOpened():
                self.health.connected = True
                logger.info("Successfully opened RTSP stream for '%s'.", self.camera_id)
                return True
            else:
                logger.warning("Failed to open RTSP stream for '%s'.", self.camera_id)
                # If RTSP fails and Sentinel credentials exist, fall back automatically to HLS
                if os.getenv("SENTINEL_AUTH_EMAIL") and os.getenv("SENTINEL_AUTH_PASSWORD"):
                    logger.info("[%s] RTSP stream unavailable. Falling back to authorized Sentinel HLS stream...", self.camera_id)
                    from src.streaming.sentinel_hls import SentinelHLSReader
                    self._hls_reader = SentinelHLSReader(self.camera_id)
                    if self._hls_reader.connect():
                        self.health.connected = True
                        return True
                self.health.connected = False
                return False
        except Exception as e:
            logger.error("Exception opening RTSP stream for '%s': %s", self.camera_id, e)
            self.health.connected = False
            return False

    def _check_pts_discontinuity(self, current_pts_ms: Optional[float]) -> None:
        """Detect and log PTS timing discontinuities."""
        if current_pts_ms is None or current_pts_ms <= 0:
            return

        last_pts = self.health.last_pts_ms
        if last_pts is not None and last_pts > 0:
            delta = current_pts_ms - last_pts

            # 1. PTS moves backwards (stream looped or timeline reset)
            if delta < 0:
                logger.warning(
                    "[%s] PTS DISCONTINUITY (BACKWARDS): PTS jumped backwards from %.1fms to %.1fms (delta: %.1fms). "
                    "Classified as potential stream loop or timeline reset.",
                    self.camera_id,
                    last_pts,
                    current_pts_ms,
                    delta,
                )
                self.health.record_discontinuity()

            # 2. Large forward gap
            elif delta > self.forward_discontinuity_threshold_ms:
                logger.warning(
                    "[%s] PTS DISCONTINUITY (FORWARD GAP): PTS jumped forward from %.1fms to %.1fms (gap: %.1fms). "
                    "Classified as timing gap.",
                    self.camera_id,
                    last_pts,
                    current_pts_ms,
                    delta,
                )
                self.health.record_discontinuity()

    def frames(
        self,
        max_frames: Optional[int] = None,
        max_consecutive_read_failures: int = 5,
        max_reconnect_attempts: int = 3,
    ) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """Yield frames and telemetry metadata dict until stopped or EOF.

        Yields:
            Tuple of (frame_image, frame_metadata_dict)
        """
        self._is_running = True
        consecutive_read_failures = 0

        # Initial connection attempt
        if not self.connect():
            self.health.record_reconnect()
            self.backoff.wait()

        # If HLS reader was selected (e.g. RTSP 401 fallback or direct HLS URL)
        if self._hls_reader:
            for seq, frame, pts_ms, recv_mono in self._hls_reader.stream_frames(max_frames=max_frames):
                self.frame_sequence = seq
                self.health.record_frame(
                    pts_ms=pts_ms,
                    receive_monotonic=recv_mono,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    codec="h264",
                )
                meta = {
                    "camera_id": self.camera_id,
                    "frame_sequence": seq,
                    "media_pts_ms": pts_ms,
                    "local_receive_monotonic": recv_mono,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "codec": "h264",
                }
                yield frame, meta
            return

        while self._is_running:
            # Reconnect loop if capture is not opened
            while self._is_running and (self.cap is None or not self.cap.isOpened()):
                if self.backoff.attempts >= max_reconnect_attempts:
                    logger.error(
                        "[%s] Maximum reconnect attempts (%d) reached. Stream unreachable or authentication failed (HTTP 401). Terminating stream reader.",
                        self.camera_id,
                        max_reconnect_attempts,
                    )
                    self._is_running = False
                    return

                self.health.record_reconnect()
                logger.info("[%s] Attempting stream reconnection (attempt %d/%d)...", self.camera_id, self.backoff.attempts + 1, max_reconnect_attempts)
                if self.connect():
                    consecutive_read_failures = 0
                    break
                self.backoff.wait()

            if not self._is_running:
                break

            # Read frame
            ret, frame = self.cap.read()
            local_receive_monotonic = get_monotonic_time()

            if not ret or frame is None:
                # Check if this is a local video file rather than a network stream.
                # If local file, ret=False indicates End-Of-File (EOF). Do NOT reconnect.
                is_local = os.path.exists(self.rtsp_url) or not self.rtsp_url.startswith(
                    ("rtsp://", "rtsps://", "http://", "https://")
                )
                if is_local and self.frame_sequence > 0:
                    logger.info(
                        "[%s] End-of-file (EOF) reached for local video (%d frames read). Terminating reader.",
                        self.camera_id, self.frame_sequence
                    )
                    self._is_running = False
                    break

                consecutive_read_failures += 1
                self.health.record_decode_error()

                # In H.264/H.265, initial packets may generate warnings before first keyframe.
                # Allow a small tolerance before treating as fatal disconnect.
                if consecutive_read_failures <= max_consecutive_read_failures:
                    logger.debug(
                        "[%s] Transient frame read miss (%d/%d). Decoder may be awaiting keyframe...",
                        self.camera_id,
                        consecutive_read_failures,
                        max_consecutive_read_failures,
                    )
                    time.sleep(0.05)
                    continue

                logger.warning(
                    "[%s] Persistent read failure (%d consecutive misses). Reconnecting...",
                    self.camera_id,
                    consecutive_read_failures,
                )
                self.release()
                self.health.record_reconnect()
                self.backoff.wait()
                continue

            # Reset failure count on successful read
            consecutive_read_failures = 0
            self.frame_sequence += 1

            # Extract media PTS
            raw_pts_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            pts_ms = float(raw_pts_ms) if raw_pts_ms > 0 else None

            # Frame dimensions
            height, width = frame.shape[:2]

            # Check for timing discontinuities
            self._check_pts_discontinuity(pts_ms)

            # Record health telemetry
            self.health.record_frame(
                pts_ms=pts_ms,
                receive_monotonic=local_receive_monotonic,
                width=width,
                height=height,
                codec=self.codec,
            )

            # Register stable connection progress
            self.backoff.record_frame_success()

            metadata = {
                "camera_id": self.camera_id,
                "frame_sequence": self.frame_sequence,
                "media_pts_ms": pts_ms,
                "local_receive_monotonic": local_receive_monotonic,
                "width": width,
                "height": height,
                "codec": self.codec,
            }

            yield frame, metadata

            if max_frames and self.frame_sequence >= max_frames:
                logger.info(
                    "[%s] Reached target validation frame count (%d).",
                    self.camera_id,
                    max_frames,
                )
                break

    def stop(self) -> None:
        """Signal the reader to stop streaming."""
        self._is_running = False

    def release(self) -> None:
        """Release underlying OpenCV VideoCapture or HLS resources."""
        if hasattr(self, "_hls_reader") and self._hls_reader is not None:
            try:
                self._hls_reader.release()
            except Exception:
                pass
            self._hls_reader = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.error("Error releasing VideoCapture for '%s': %s", self.camera_id, e)
            self.cap = None
        self.health.connected = False
