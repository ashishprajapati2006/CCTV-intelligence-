"""Stream health monitoring and metrics."""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from src.common.time import get_monotonic_time


@dataclass
class StreamHealth:
    """Represents real-time health and transport telemetry of a camera stream.

    NOTE:
    - last_pts_ms is the media presentation timestamp from the video container.
    - last_receive_time is the local monotonic clock timestamp upon frame delivery.
    - They are intentionally decoupled to preserve video timing fidelity.
    """
    camera_id: str
    connected: bool = False
    frames_received: int = 0
    last_pts_ms: Optional[float] = None
    last_receive_time: Optional[float] = None
    reconnect_count: int = 0
    decode_errors: int = 0
    pts_discontinuities: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None
    extra_telemetry: Dict[str, Any] = field(default_factory=dict)

    def record_frame(
        self,
        pts_ms: Optional[float],
        receive_monotonic: float,
        width: Optional[int] = None,
        height: Optional[int] = None,
        codec: Optional[str] = None,
    ) -> None:
        """Record the successful arrival of a decoded frame."""
        self.connected = True
        self.frames_received += 1
        self.last_pts_ms = pts_ms
        self.last_receive_time = receive_monotonic
        if width and height:
            self.width = width
            self.height = height
        if codec:
            self.codec = codec

    def record_discontinuity(self) -> None:
        """Record a detected PTS timing discontinuity."""
        self.pts_discontinuities += 1

    def record_reconnect(self) -> None:
        """Record a stream reconnect event."""
        self.connected = False
        self.reconnect_count += 1

    def record_decode_error(self) -> None:
        """Record a frame read or decoder fault."""
        self.decode_errors += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize health status to dictionary."""
        return asdict(self)

    def summary_str(self) -> str:
        """Return a human-readable summary of stream health."""
        res_str = f"{self.width}x{self.height}" if self.width and self.height else "Unknown"
        return (
            f"Camera: {self.camera_id} | Status: {'CONNECTED' if self.connected else 'DISCONNECTED'} | "
            f"Frames: {self.frames_received} | Resolution: {res_str} | Codec: {self.codec or 'Unknown'} | "
            f"Last PTS: {self.last_pts_ms:.1f}ms | Reconnects: {self.reconnect_count} | "
            f"Decode Errors: {self.decode_errors} | PTS Discontinuities: {self.pts_discontinuities}"
            if self.last_pts_ms is not None else
            f"Camera: {self.camera_id} | Status: {'CONNECTED' if self.connected else 'DISCONNECTED'} | "
            f"Frames: {self.frames_received} | Resolution: {res_str} | Codec: {self.codec or 'Unknown'} | "
            f"Last PTS: None | Reconnects: {self.reconnect_count} | "
            f"Decode Errors: {self.decode_errors} | PTS Discontinuities: {self.pts_discontinuities}"
        )
