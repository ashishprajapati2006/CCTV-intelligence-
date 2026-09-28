"""Unit tests for catalogue validation rules."""
import pytest
from src.catalogue.validator import CatalogueValidator
from src.catalogue.models import Camera


def test_validator_valid_camera():
    validator = CatalogueValidator()
    cam = Camera(
        camera_id="cam01",
        name="Valid Camera",
        status="active",
        codec="H264",
        width=1920,
        height=1080,
        rtsp_url="rtsp://10.0.0.1:8554/live",
    )
    is_valid, errors, warnings = validator.validate_camera(cam, seen_ids=set())
    assert is_valid is True
    assert len(errors) == 0


def test_validator_duplicate_id():
    validator = CatalogueValidator()
    cam1 = Camera(camera_id="cam01")
    cam2 = Camera(camera_id="cam01")

    is_valid, errors, warnings = validator.validate_catalogue([cam1, cam2])
    assert is_valid is False
    assert any("Duplicate camera ID" in err for err in errors)


def test_validator_missing_id():
    validator = CatalogueValidator()
    cam = Camera(camera_id="")
    is_valid, errors, warnings = validator.validate_camera(cam, seen_ids=set())
    assert is_valid is False
    assert any("Camera ID is missing" in err for err in errors)


def test_validator_invalid_rtsp_url():
    validator = CatalogueValidator()
    cam = Camera(
        camera_id="cam02",
        rtsp_url="http://not-an-rtsp-scheme.com/stream",
    )
    is_valid, errors, warnings = validator.validate_camera(cam, seen_ids=set())
    assert is_valid is False
    assert any("invalid RTSP URL" in err for err in errors)


def test_validator_invalid_resolution():
    validator = CatalogueValidator()
    cam_negative = Camera(camera_id="cam03", width=-100, height=720)
    is_valid, errors, warnings = validator.validate_camera(cam_negative, seen_ids=set())
    assert is_valid is False
    assert any("invalid resolution" in err for err in errors)

    cam_partial = Camera(camera_id="cam04", width=1920, height=None)
    is_valid, errors, warnings = validator.validate_camera(cam_partial, seen_ids=set())
    assert is_valid is False
    assert any("invalid resolution" in err for err in errors)


def test_validator_unrecognized_status_warning():
    validator = CatalogueValidator()
    cam = Camera(camera_id="cam05", status="flapping_state")
    is_valid, errors, warnings = validator.validate_camera(cam, seen_ids=set())
    assert is_valid is True  # Warning does not invalidate record
    assert any("unrecognized status" in w for w in warnings)


def test_validator_out_of_bounds_coordinates():
    validator = CatalogueValidator()
    cam_bad_lat = Camera(camera_id="cam06", latitude=120.0, longitude=50.0)
    is_valid, errors, warnings = validator.validate_camera(cam_bad_lat, seen_ids=set())
    assert is_valid is False
    assert any("out-of-range latitude" in err for err in errors)

    cam_bad_lng = Camera(camera_id="cam07", latitude=20.0, longitude=-200.0)
    is_valid, errors, warnings = validator.validate_camera(cam_bad_lng, seen_ids=set())
    assert is_valid is False
    assert any("out-of-range longitude" in err for err in errors)
