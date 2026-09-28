"""Unit and integration tests for FastAPI backend routes."""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Gujarat Police" in data["service"]


def test_list_cameras():
    response = client.get("/api/cameras")
    assert response.status_code == 200
    cameras = response.json()
    assert len(cameras) >= 30
    assert any(c["camera_id"] == "cam01" for c in cameras)


def test_get_camera_detail():
    response = client.get("/api/cameras/cam01")
    assert response.status_code == 200
    cam = response.json()
    assert cam["camera_id"] == "cam01"
    assert "Chiman" in cam["name"]


def test_get_camera_health():
    response = client.get("/api/cameras/cam01/health")
    assert response.status_code == 200
    health = response.json()
    assert health["camera_id"] == "cam01"


def test_get_camera_streams():
    response = client.get("/api/cameras/cam01/streams")
    assert response.status_code == 200
    streams = response.json()
    assert len(streams) >= 1
    assert any(s["protocol"] == "hls" for s in streams)


def test_list_vehicles():
    response = client.get("/api/vehicles")
    assert response.status_code == 200
    vehicles = response.json()
    assert len(vehicles) >= 1
    assert any(v.get("registration_number") for v in vehicles)


def test_get_vehicle_detail():
    response = client.get("/api/vehicles")
    assert response.status_code == 200
    vehicles = response.json()
    assert len(vehicles) >= 1
    target_reg = vehicles[0]["registration_number"]
    response = client.get(f"/api/vehicles/{target_reg}")
    assert response.status_code == 200
    veh = response.json()
    assert veh["registration_number"] == target_reg
    assert len(veh["timeline"]) >= 1


def test_get_vehicle_journey():
    response = client.get("/api/vehicles/CH0BHBGE/journey")
    assert response.status_code == 200
    journey = response.json()
    assert journey["registration_number"] == "CH0BHBGE"
    assert "segments" in journey
    assert "ordering_mode" in journey


def test_get_vehicle_geojson():
    response = client.get("/api/vehicles/CH0BHBGE/geojson")
    assert response.status_code == 200
    geojson = response.json()
    assert geojson["type"] == "FeatureCollection"


def test_list_watchlist():
    response = client.get("/api/watchlist")
    assert response.status_code == 200
    wl = response.json()
    assert len(wl) >= 1
    # Check synthetic tag
    assert all(item.get("synthetic") is True for item in wl)
    assert all(item.get("source") == "SYNTHETIC_DEMO" for item in wl)


def test_add_watchlist_target():
    payload = {
        "registration_number": "GJ01TEST99",
        "category": "DEMO_SUSPECT_VEHICLE",
        "priority": "HIGH",
        "description": "Automated test target"
    }
    response = client.post("/api/watchlist", json=payload)
    assert response.status_code in (200, 400)  # 200 if new, 400 if duplicate


def test_list_alerts():
    response = client.get("/api/alerts")
    assert response.status_code == 200
    alerts = response.json()
    assert len(alerts) >= 1


def test_alert_operator_action():
    alerts_resp = client.get("/api/alerts")
    alerts = alerts_resp.json()
    if alerts:
        alert_id = alerts[0]["alert_id"]
        # Test acknowledge
        ack_resp = client.post(
            f"/api/alerts/{alert_id}/action",
            json={"action": "acknowledge", "operator": "Officer Patel", "notes": "Dispatched unit 4"}
        )
        assert ack_resp.status_code == 200
        updated = ack_resp.json()
        assert updated["status"] == "ACKNOWLEDGED"
        assert updated["acknowledged_by"] == "Officer Patel"
        assert len(updated["audit_history"]) >= 2


def test_dashboard_stats():
    response = client.get("/api/dashboard/stats")
    assert response.status_code == 200
    stats = response.json()
    assert stats["total_cameras"] == 30
    assert stats["online_cameras"] == 30
    assert stats["total_observed_vehicles"] >= 6


def test_dashboard_analytics():
    response = client.get("/api/dashboard/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "camera_distribution" in data
    assert "quality_distribution" in data
