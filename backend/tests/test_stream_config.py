"""Unit tests for RTSP transport configuration and exponential backoff."""
import os
import pytest
from src.streaming.reconnect import ExponentialBackoff
from src.streaming.rtsp import RTSPStreamReader


def test_exponential_backoff_progression():
    """Verify schedule: 2 -> 4 -> 8 -> 16 -> 30 (maximum)."""
    backoff = ExponentialBackoff()

    assert backoff.current_delay == 2.0
    delay1 = backoff.record_attempt()
    assert delay1 == 2.0

    assert backoff.current_delay == 4.0
    delay2 = backoff.record_attempt()
    assert delay2 == 4.0

    assert backoff.current_delay == 8.0
    delay3 = backoff.record_attempt()
    assert delay3 == 8.0

    assert backoff.current_delay == 16.0
    delay4 = backoff.record_attempt()
    assert delay4 == 16.0

    assert backoff.current_delay == 30.0
    delay5 = backoff.record_attempt()
    assert delay5 == 30.0

    # Must stay capped at 30.0
    assert backoff.current_delay == 30.0
    delay6 = backoff.record_attempt()
    assert delay6 == 30.0


def test_exponential_backoff_reset_on_stable_connection():
    """Verify backoff resets to initial interval when stream stabilizes."""
    backoff = ExponentialBackoff(stable_frames_threshold=10)

    # Simulate 3 failures
    backoff.record_attempt()
    backoff.record_attempt()
    backoff.record_attempt()
    assert backoff.current_delay == 16.0

    # Simulate 9 successful frames (not yet threshold)
    for _ in range(9):
        did_reset = backoff.record_frame_success()
        assert not did_reset
        assert backoff.current_delay == 16.0

    # 10th frame triggers reset
    did_reset = backoff.record_frame_success()
    assert did_reset
    assert backoff.current_delay == 2.0
    assert backoff.attempts == 0


def test_rtsp_tcp_transport_enforcement():
    """Verify OPENCV_FFMPEG_CAPTURE_OPTIONS is set to rtsp_transport;tcp."""
    reader = RTSPStreamReader(
        camera_id="cam_test",
        rtsp_url="rtsp://dummy/test",
        force_tcp=True,
    )
    reader._configure_transport()
    assert os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") == "rtsp_transport;tcp"
