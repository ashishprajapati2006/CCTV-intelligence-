"""Unit tests for StreamHealth tracking and discontinuity detection."""
import pytest
from src.streaming.health import StreamHealth
from src.streaming.rtsp import RTSPStreamReader


def test_stream_health_metrics_accumulation():
    health = StreamHealth(camera_id="cam01", codec="H264")
    assert not health.connected
    assert health.frames_received == 0

    # Record 1st frame
    health.record_frame(pts_ms=100.0, receive_monotonic=1000.5, width=1920, height=1080)
    assert health.connected
    assert health.frames_received == 1
    assert health.last_pts_ms == 100.0
    assert health.last_receive_time == 1000.5
    assert health.width == 1920
    assert health.height == 1080

    # Record reconnect event
    health.record_reconnect()
    assert not health.connected
    assert health.reconnect_count == 1

    # Record error event
    health.record_decode_error()
    assert health.decode_errors == 1

    # Record discontinuity
    health.record_discontinuity()
    assert health.pts_discontinuities == 1

    # Serialization
    data = health.to_dict()
    assert data["camera_id"] == "cam01"
    assert data["frames_received"] == 1
    assert data["reconnect_count"] == 1

    summary = health.summary_str()
    assert "cam01" in summary
    assert "1920x1080" in summary


def test_pts_backward_discontinuity_detection():
    reader = RTSPStreamReader(
        camera_id="cam01",
        rtsp_url="rtsp://dummy/test",
    )
    # 1st frame at 1000ms
    reader.health.record_frame(pts_ms=1000.0, receive_monotonic=10.0)

    # 2nd frame at 1040ms (normal)
    reader._check_pts_discontinuity(1040.0)
    assert reader.health.pts_discontinuities == 0
    reader.health.record_frame(pts_ms=1040.0, receive_monotonic=10.04)

    # 3rd frame at 40ms (backward jump / loop reset)
    reader._check_pts_discontinuity(40.0)
    assert reader.health.pts_discontinuities == 1


def test_pts_forward_gap_detection():
    reader = RTSPStreamReader(
        camera_id="cam01",
        rtsp_url="rtsp://dummy/test",
        forward_discontinuity_threshold_ms=5000.0,
    )
    reader.health.record_frame(pts_ms=2000.0, receive_monotonic=10.0)

    # Frame jumps forward by 8000ms
    reader._check_pts_discontinuity(10000.0)
    assert reader.health.pts_discontinuities == 1
