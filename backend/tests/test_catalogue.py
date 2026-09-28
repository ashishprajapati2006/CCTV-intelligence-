"""Unit tests for catalogue parsing and models."""
import pytest
from src.catalogue.parser import CatalogueParser
from src.catalogue.models import Camera


def test_parse_list_format():
    parser = CatalogueParser()
    raw_payload = [
        {"id": "cam01", "name": "Entrance North"},
        {"id": "cam02", "name": "Exit South"},
    ]
    cameras = parser.parse_payload(raw_payload)
    assert len(cameras) == 2
    assert cameras[0].camera_id == "cam01"
    assert cameras[0].name == "Entrance North"
    assert cameras[1].camera_id == "cam02"
    assert cameras[1].name == "Exit South"


def test_parse_nested_format():
    parser = CatalogueParser()
    raw_payload = {
        "cameras": [
            {"camera_id": "cam10", "name": "Junction 1"},
            {"cam_id": "cam11", "name": "Junction 2"},
        ]
    }
    cameras = parser.parse_payload(raw_payload)
    assert len(cameras) == 2
    assert cameras[0].camera_id == "cam10"
    assert cameras[1].camera_id == "cam11"


def test_id_normalization():
    parser = CatalogueParser()
    test_cases = [
        ({"camera_id": "camA"}, "camA"),
        ({"id": "camB"}, "camB"),
        ({"cam_id": "camC"}, "camC"),
        ({"number": "101"}, "101"),
    ]
    for raw, expected in test_cases:
        cam = parser.parse_camera_dict(raw)
        assert cam.camera_id == expected


def test_codec_normalization():
    parser = CatalogueParser()
    test_cases = [
        ("h264", "H264"),
        ("H.264", "H264"),
        ("avc", "H264"),
        ("h265", "H265"),
        ("H.265", "H265"),
        ("hevc", "H265"),
        ("VP9", "VP9"),
        (None, "UNKNOWN"),
    ]
    for raw_codec, expected in test_cases:
        cam = parser.parse_camera_dict({"id": "c1", "codec": raw_codec})
        assert cam.codec == expected


def test_resolution_extraction():
    parser = CatalogueParser()
    # Explicit fields
    cam1 = parser.parse_camera_dict({"id": "c1", "width": 1920, "height": 1080})
    assert cam1.width == 1920
    assert cam1.height == 1080
    assert cam1.resolution_str == "1920x1080"

    # Composite resolution string
    cam2 = parser.parse_camera_dict({"id": "c2", "resolution": "1280x720"})
    assert cam2.width == 1280
    assert cam2.height == 720
    assert cam2.resolution_str == "1280x720"


def test_unknown_fields_preserved_in_extra():
    parser = CatalogueParser()
    raw = {
        "id": "cam05",
        "name": "Highway Post",
        "custom_department": "Traffic Police",
        "firmware_version": "v3.2.1",
        "installation_year": 2024,
    }
    cam = parser.parse_camera_dict(raw)
    assert cam.camera_id == "cam05"
    assert "custom_department" in cam.extra
    assert cam.extra["custom_department"] == "Traffic Police"
    assert cam.extra["firmware_version"] == "v3.2.1"
    assert cam.extra["installation_year"] == 2024


def test_coordinates_not_invented():
    parser = CatalogueParser()
    # When coordinates are not provided, they must remain None
    cam_no_coords = parser.parse_camera_dict({"id": "c1"})
    assert cam_no_coords.latitude is None
    assert cam_no_coords.longitude is None

    # When coordinates are explicitly provided, they are preserved
    cam_with_coords = parser.parse_camera_dict({
        "id": "c2",
        "latitude": 23.0225,
        "longitude": 72.5714,
    })
    assert cam_with_coords.latitude == 23.0225
    assert cam_with_coords.longitude == 72.5714


def test_gateway_template_resolution():
    parser = CatalogueParser(
        gateway_rtsp_template="rtsp://gateway:8554/stream/{camera_id}",
        gateway_hls_template="https://cdn.corp/{camera_id}/index.m3u8",
    )
    cam = parser.parse_camera_dict({"id": "cam08"})
    assert cam.rtsp_url == "rtsp://gateway:8554/stream/cam08"
    assert cam.hls_url == "https://cdn.corp/cam08/index.m3u8"

    # Explicit direct URL must not be overridden
    cam_override = parser.parse_camera_dict({
        "id": "cam09",
        "rtsp_url": "rtsp://custom:8554/live",
    })
    assert cam_override.rtsp_url == "rtsp://custom:8554/live"
