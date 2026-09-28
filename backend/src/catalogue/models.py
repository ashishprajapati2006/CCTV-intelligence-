"""Data models for camera catalogue representation."""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class Camera:
    """Normalized internal representation of a CCTV camera stream.

    NOTE:
    - fps is strictly informational metadata from the catalogue.
      It MUST NOT be used as a basis for frame timing or timestamp calculation.
    - latitude and longitude are preserved only if explicitly provided by the catalogue.
      They are never synthesized or invented.
    """
    camera_id: str
    name: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    status: str = "unknown"  # "active", "inactive", "maintenance", "unknown"
    codec: str = "UNKNOWN"   # "H264", "H265", "UNKNOWN"
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    bitrate: Optional[int] = None
    rtsp_url: Optional[str] = None
    hls_url: Optional[str] = None
    webrtc_url: Optional[str] = None
    timezone: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolution_str(self) -> str:
        """Return formatted resolution string or 'Unknown'."""
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "Unknown"

    @property
    def has_rtsp(self) -> bool:
        """Check if RTSP URL is available."""
        return bool(self.rtsp_url and self.rtsp_url.startswith("rtsp://"))

    @property
    def has_hls(self) -> bool:
        """Check if HLS URL is available."""
        return bool(self.hls_url and (self.hls_url.startswith("http://") or self.hls_url.startswith("https://")))

    def to_dict(self) -> Dict[str, Any]:
        """Convert camera to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Camera":
        """Reconstruct Camera instance from dictionary."""
        data_copy = dict(data)
        extra = data_copy.pop("extra", {})
        known_fields = {
            "camera_id", "name", "location", "latitude", "longitude",
            "status", "codec", "width", "height", "fps", "bitrate",
            "rtsp_url", "hls_url", "webrtc_url", "timezone"
        }
        filtered = {k: v for k, v in data_copy.items() if k in known_fields}
        # Anything else goes to extra
        for k, v in data_copy.items():
            if k not in known_fields and k not in extra:
                extra[k] = v
        return cls(**filtered, extra=extra)
