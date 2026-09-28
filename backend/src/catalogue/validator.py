"""Validation rules for camera catalogue data."""
import re
from typing import Dict, List, Set, Tuple
from urllib.parse import urlparse
from src.catalogue.models import Camera


class CatalogueValidator:
    """Validates camera models according to Sentinel operational requirements."""

    RECOGNIZED_STATUSES = {"active", "inactive", "maintenance", "unknown"}
    RECOGNIZED_CODECS = {"H264", "H265", "UNKNOWN"}

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def _validate_url(self, url: str, scheme: str) -> bool:
        """Validate stream URL format."""
        if not url:
            return False
        try:
            parsed = urlparse(url)
            return parsed.scheme == scheme and bool(parsed.netloc)
        except Exception:
            return False

    def validate_camera(self, camera: Camera, seen_ids: Set[str]) -> Tuple[bool, List[str], List[str]]:
        """Validate an individual camera instance.

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        camera_errors = []
        camera_warnings = []

        # 1. Camera ID must exist and not be empty
        if not camera.camera_id or not camera.camera_id.strip():
            camera_errors.append("Camera ID is missing or empty.")
        else:
            # 2. Camera ID must be unique
            if camera.camera_id in seen_ids:
                camera_errors.append(f"Duplicate camera ID: '{camera.camera_id}'.")
            seen_ids.add(camera.camera_id)

        # 3. Status must be recognized
        if camera.status not in self.RECOGNIZED_STATUSES:
            camera_warnings.append(
                f"Camera '{camera.camera_id}' has unrecognized status: '{camera.status}'."
            )

        # 4. Stream URLs must be syntactically valid
        if camera.rtsp_url:
            if not self._validate_url(camera.rtsp_url, "rtsp"):
                camera_errors.append(
                    f"Camera '{camera.camera_id}' has invalid RTSP URL: '{camera.rtsp_url}'."
                )
        if camera.hls_url:
            parsed = urlparse(camera.hls_url)
            if parsed.scheme not in ["http", "https"] or not parsed.netloc:
                camera_warnings.append(
                    f"Camera '{camera.camera_id}' has invalid HLS URL: '{camera.hls_url}'."
                )
        if camera.webrtc_url:
            parsed = urlparse(camera.webrtc_url)
            if parsed.scheme not in ["http", "https"] or not parsed.netloc:
                camera_warnings.append(
                    f"Camera '{camera.camera_id}' has invalid WebRTC/WHEP URL: '{camera.webrtc_url}'."
                )

        # 5. Codec check
        if camera.codec not in self.RECOGNIZED_CODECS:
            camera_warnings.append(
                f"Camera '{camera.camera_id}' has unusual codec '{camera.codec}'. "
                f"Expected H264, H265, or UNKNOWN."
            )

        # 6. Resolution check if supplied
        if camera.width is not None or camera.height is not None:
            if camera.width is None or camera.height is None or camera.width <= 0 or camera.height <= 0:
                camera_errors.append(
                    f"Camera '{camera.camera_id}' has invalid resolution ({camera.width}x{camera.height})."
                )

        # 7. Coordinates check if supplied
        if camera.latitude is not None:
            if not (-90.0 <= camera.latitude <= 90.0):
                camera_errors.append(
                    f"Camera '{camera.camera_id}' has out-of-range latitude: {camera.latitude}."
                )
        if camera.longitude is not None:
            if not (-180.0 <= camera.longitude <= 180.0):
                camera_errors.append(
                    f"Camera '{camera.camera_id}' has out-of-range longitude: {camera.longitude}."
                )

        # 8. Timing enforcement rule: Informational FPS must not be relied upon
        if camera.fps is not None and camera.fps <= 0:
            camera_warnings.append(
                f"Camera '{camera.camera_id}' has non-positive fps ({camera.fps}). Informational only."
            )

        return (len(camera_errors) == 0, camera_errors, camera_warnings)

    def validate_catalogue(self, cameras: List[Camera]) -> Tuple[bool, List[str], List[str]]:
        """Validate entire list of cameras.

        Returns:
            Tuple of (all_valid, all_errors, all_warnings)
        """
        all_errors = []
        all_warnings = []
        seen_ids: Set[str] = set()

        if not cameras:
            all_warnings.append("Catalogue is empty (0 cameras found).")

        for cam in cameras:
            _, errs, warns = self.validate_camera(cam, seen_ids)
            all_errors.extend(errs)
            all_warnings.extend(warns)

        return (len(all_errors) == 0, all_errors, all_warnings)
