#!/usr/bin/env python3
"""generate_mongodb_dummy_cctv.py

Generates realistic multi-camera vehicle tracking & GIS dummy data for MongoDB.
Produces MongoDB-ready collections with GeoJSON Point and LineString features
indexed for 2dsphere geospatial queries ($near, $geoWithin, $geoIntersects).

Usage:
    python scripts/generate_mongodb_dummy_cctv.py
    python scripts/generate_mongodb_dummy_cctv.py --vehicles 25 --output-dir data/mongodb_export
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.journey.shortest_path import road_network

# Structured realistic surveillance corridors across Gujarat (Cameras 1-30)
SURVEILLANCE_CORRIDORS = [
    # 1. Ahmedabad Urban Corridors (East-West & North-South)
    ("cam01", "cam04"), ("cam03", "cam13"), ("cam05", "cam14"), ("cam16", "cam02"),
    ("cam20", "cam15"), ("cam12", "cam04"), ("cam14", "cam01"), ("cam04", "cam16"),
    ("cam13", "cam01"), ("cam02", "cam12"), ("cam01", "cam20"), ("cam15", "cam04"),
    ("cam03", "cam04"), ("cam16", "cam13"), ("cam05", "cam02"),
    # 2. Junagadh Urban & Bypass Corridors
    ("cam06", "cam11"), ("cam07", "cam08"), ("cam09", "cam10"), ("cam06", "cam07"),
    ("cam11", "cam07"), ("cam10", "cam08"), ("cam08", "cam09"), ("cam11", "cam06"),
    # 3. Navsari & Bilimora Coastal Highway Corridors
    ("cam27", "cam19"), ("cam28", "cam25"), ("cam29", "cam26"), ("cam19", "cam27"),
    ("cam25", "cam29"), ("cam26", "cam28"), ("cam27", "cam26"), ("cam28", "cam19"),
    # 4. North Gujarat Highway Corridors (Adalaj, Dehgam, Sabarkantha, Patan, Banaskantha)
    ("cam12", "cam22"), ("cam24", "cam21"), ("cam23", "cam12"), ("cam22", "cam24"),
    ("cam05", "cam23"), ("cam21", "cam23"), ("cam16", "cam24"),
    # 5. Rajkot Hub & Kutch Express Corridors
    ("cam17", "cam18"), ("cam17", "cam30"), ("cam18", "cam30"), ("cam30", "cam17"),
    # 6. Inter-regional State Highway Corridors
    ("cam04", "cam18"), ("cam20", "cam17"), ("cam18", "cam06"), ("cam17", "cam07"),
    ("cam04", "cam19"), ("cam20", "cam27"), ("cam01", "cam08"), ("cam12", "cam30"),
    ("cam16", "cam17"), ("cam15", "cam18"), ("cam02", "cam30"), ("cam04", "cam06"),
]

# Common Gujarat RTO codes
RTO_CODES = ["GJ01", "GJ02", "GJ03", "GJ05", "GJ06", "GJ27", "GJ18"]
LETTERS = ["AB", "CD", "EF", "GH", "JK", "MN", "PQ", "RS", "XY", "ZZ"]
VEHICLE_MODELS = [
    ("Hyundai Creta", "White", "SUV"),
    ("Maruti Suzuki Swift", "Silver", "Hatchback"),
    ("Toyota Innova Crysta", "White", "MPV"),
    ("Tata Nexon", "Blue", "Compact SUV"),
    ("Mahindra Scorpio-N", "Black", "SUV"),
    ("Honda City", "Grey", "Sedan"),
    ("Kia Seltos", "Red", "SUV"),
    ("Maruti Suzuki Baleno", "White", "Hatchback"),
    ("Royal Enfield Classic 350", "Black", "Two-Wheeler"),
    ("GSRTC Express Bus", "Orange-White", "Bus"),
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in kilometers."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def load_camera_catalogue(coord_file: Path) -> list[dict]:
    """Load camera catalogue with verified GPS coordinates."""
    if coord_file.exists():
        with open(coord_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            cams = data.get("cameras", [])
            if cams:
                return cams

    # Fallback Gujarat Police CCTV camera network
    return [
        {"camera_id": "cam01", "latitude": 23.0673, "longitude": 72.5815, "location": "Chimanbhai Bridge, Sabarmati, Ahmedabad"},
        {"camera_id": "cam02", "latitude": 23.0286, "longitude": 72.5620, "location": "Janpath Crossroad, Ashram Road, Ahmedabad"},
        {"camera_id": "cam03", "latitude": 23.0934, "longitude": 72.5878, "location": "ONGC Complex, Chandkheda, Ahmedabad"},
        {"camera_id": "cam04", "latitude": 23.0135, "longitude": 72.5625, "location": "Paldi Crossroad, Ellisbridge, Ahmedabad"},
        {"camera_id": "cam05", "latitude": 23.0371, "longitude": 72.5126, "location": "Pakwan Crossroad, SG Highway, Ahmedabad"},
        {"camera_id": "cam06", "latitude": 23.0258, "longitude": 72.5074, "location": "ISCON Crossroad, SG Highway, Ahmedabad"},
        {"camera_id": "cam07", "latitude": 23.0512, "longitude": 72.5298, "location": "Drive-In Road, Memnagar, Ahmedabad"},
        {"camera_id": "cam08", "latitude": 23.0225, "longitude": 72.5714, "location": "Income Tax Circle, Ashram Road, Ahmedabad"},
        {"camera_id": "cam09", "latitude": 23.0039, "longitude": 72.5997, "location": "Kankaria Lake Gate 1, Maninagar, Ahmedabad"},
        {"camera_id": "cam10", "latitude": 23.0489, "longitude": 72.5542, "location": "Helmet Crossroad, Gurukul, Ahmedabad"},
        {"camera_id": "cam11", "latitude": 23.0612, "longitude": 72.5342, "location": "Sola Bhagwat Vidyapith, SG Highway"},
        {"camera_id": "cam12", "latitude": 23.0333, "longitude": 72.5467, "location": "Gujarat University Circle, Navrangpura"},
        {"camera_id": "cam13", "latitude": 23.0765, "longitude": 72.5189, "location": "Gota Crossroad, SG Highway, Ahmedabad"},
        {"camera_id": "cam14", "latitude": 23.0189, "longitude": 72.5290, "location": "Shivranjani Crossroad, Satellite, Ahmedabad"},
        {"camera_id": "cam15", "latitude": 23.0315, "longitude": 72.5842, "location": "Delhi Darwaja, Old City, Ahmedabad"},
    ]


def generate_mongodb_dataset(
    num_vehicles: int = 20,
    min_cameras_per_vehicle: int = 3,
    max_cameras_per_vehicle: int = 6,
    output_dir: Path = Path("data/mongodb_export"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    coord_file = Path("data/catalogue/enrichment/camera_coordinates.json")
    cameras_raw = load_camera_catalogue(coord_file)

    # 1. MongoDB Cameras Collection (GeoJSON Point format for 2dsphere index)
    mongo_cameras = []
    for cam in cameras_raw:
        mongo_cameras.append({
            "camera_id": cam["camera_id"],
            "name": cam.get("location", f"Camera {cam['camera_id']}"),
            "status": "ACTIVE",
            "jurisdiction": "Ahmedabad City Police",
            # GeoJSON specification requires [longitude, latitude]
            "location": {
                "type": "Point",
                "coordinates": [round(cam["longitude"], 6), round(cam["latitude"], 6)],
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    # Save Cameras JSON
    cameras_out = output_dir / "mongodb_cameras.json"
    with open(cameras_out, "w", encoding="utf-8") as f:
        json.dump(mongo_cameras, f, indent=2)

    # 2. Generate Vehicles, Sightings, and Journeys
    mongo_sightings = []
    mongo_journeys = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=6)

    for i in range(num_vehicles):
        rto = random.choice(RTO_CODES)
        letters = random.choice(LETTERS)
        digits = f"{random.randint(1000, 9999)}"
        plate = f"{rto}{letters}{digits}"
        vehicle_id = f"VEH-{plate}"

        model_info = random.choice(VEHICLE_MODELS)
        make_model, color, v_class = model_info

        # Select realistic surveillance corridor and compute exact shortest path sequence
        cameras_by_id = {c["camera_id"]: c for c in cameras_raw}
        orig_cid, dest_cid = SURVEILLANCE_CORRIDORS[i % len(SURVEILLANCE_CORRIDORS)]
        path_info = road_network.find_shortest_path(orig_cid, dest_cid)
        if path_info and len(path_info["shortest_path"]) >= 2:
            selected_cameras = [cameras_by_id[cid] for cid in path_info["shortest_path"] if cid in cameras_by_id]
        else:
            selected_cameras = [cameras_by_id[orig_cid], cameras_by_id[dest_cid]]

        vehicle_start_time = base_time + timedelta(minutes=random.randint(0, 180))
        current_time = vehicle_start_time

        route_coords = []
        route_segments = []
        prev_cam = None

        for idx, cam in enumerate(selected_cameras):
            lon = cam["longitude"]
            lat = cam["latitude"]
            route_coords.append([round(lon, 6), round(lat, 6)])

            # Time advancement based on distance and speed (30-60 km/h)
            speed_kmh = random.uniform(32.0, 58.0)
            if prev_cam:
                dist_km = haversine_km(prev_cam["latitude"], prev_cam["longitude"], lat, lon)
                travel_minutes = max(2.5, (dist_km / speed_kmh) * 60.0)
                current_time += timedelta(minutes=travel_minutes)
            else:
                dist_km = 0.0

            sighting_id = f"SGT-{cam['camera_id']}-{plate}-{int(current_time.timestamp())}"
            sighting = {
                "sighting_id": sighting_id,
                "vehicle_id": vehicle_id,
                "registration_number": plate,
                "camera_id": cam["camera_id"],
                "camera_name": cam.get("location"),
                # GeoJSON Point
                "location": {
                    "type": "Point",
                    "coordinates": [round(lon, 6), round(lat, 6)],
                },
                "timestamp": current_time.isoformat(),
                "timestamp_epoch_ms": int(current_time.timestamp() * 1000),
                "speed_kmh": round(speed_kmh, 1),
                "confidence_score": round(random.uniform(0.88, 0.99), 2),
                "vehicle_type": v_class,
                "color": color,
                "model": make_model,
                "status": "CONFIRMED",
            }
            mongo_sightings.append(sighting)

            route_segments.append({
                "sequence_order": idx + 1,
                "camera_id": cam["camera_id"],
                "camera_name": cam.get("location"),
                "timestamp": current_time.isoformat(),
                "coordinates": [round(lon, 6), round(lat, 6)],
                "speed_kmh": round(speed_kmh, 1),
            })
            prev_cam = cam

        # Total distance
        total_dist_km = 0.0
        for seg_idx in range(len(selected_cameras) - 1):
            c1 = selected_cameras[seg_idx]
            c2 = selected_cameras[seg_idx + 1]
            total_dist_km += haversine_km(c1["latitude"], c1["longitude"], c2["latitude"], c2["longitude"])

        total_duration_minutes = (current_time - vehicle_start_time).total_seconds() / 60.0

        # MongoDB Journey Document with GeoJSON LineString
        journey = {
            "journey_id": f"JRN-{plate}",
            "vehicle_id": vehicle_id,
            "registration_number": plate,
            "vehicle_details": {
                "make_model": make_model,
                "color": color,
                "class": v_class,
            },
            "camera_count": len(selected_cameras),
            "camera_sequence": [c["camera_id"] for c in selected_cameras],
            "first_seen": {
                "camera_id": selected_cameras[0]["camera_id"],
                "timestamp": vehicle_start_time.isoformat(),
            },
            "last_seen": {
                "camera_id": selected_cameras[-1]["camera_id"],
                "timestamp": current_time.isoformat(),
            },
            "total_distance_km": round(total_dist_km, 2),
            "total_duration_minutes": round(total_duration_minutes, 1),
            "avg_speed_kmh": round((total_dist_km / (total_duration_minutes / 60.0)) if total_duration_minutes > 0 else 40.0, 1),
            # GeoJSON LineString trajectory for GIS map path rendering
            "trajectory": {
                "type": "LineString",
                "coordinates": route_coords,
            },
            "segments": route_segments,
            "status": "CONFIRMED",
            "is_watchlist_match": random.random() < 0.20,  # 20% match probability
        }
        mongo_journeys.append(journey)

    # Save Sightings JSON
    sightings_out = output_dir / "mongodb_vehicle_sightings.json"
    with open(sightings_out, "w", encoding="utf-8") as f:
        json.dump(mongo_sightings, f, indent=2)

    # Save Journeys JSON
    journeys_out = output_dir / "mongodb_vehicle_journeys.json"
    with open(journeys_out, "w", encoding="utf-8") as f:
        json.dump(mongo_journeys, f, indent=2)

    # 3. Create a README with MongoDB commands & 2dsphere indexes
    readme_out = output_dir / "README_MONGODB_GIS.md"
    readme_content = f"""# CCTV Vehicle Intelligence — MongoDB & GIS Import Guide

Generated dummy multi-camera vehicle tracking dataset ready for MongoDB & GIS visualization.

## Files Generated:
- `mongodb_cameras.json`: CCTV Cameras catalogue with GeoJSON Point coordinates `[longitude, latitude]`.
- `mongodb_vehicle_sightings.json`: Vehicle camera sightings with timestamps and speeds.
- `mongodb_vehicle_journeys.json`: Reconstructed multi-camera vehicle routes with GeoJSON LineString trajectories.

---

## 1. Quick Import via `mongoimport`

Run these commands in your terminal:

```bash
# 1. Import Cameras
mongoimport --uri="mongodb://localhost:27017/cctv_gis" --collection=cameras --file=mongodb_cameras.json --jsonArray

# 2. Import Sightings
mongoimport --uri="mongodb://localhost:27017/cctv_gis" --collection=vehicle_sightings --file=mongodb_vehicle_sightings.json --jsonArray

# 3. Import Journeys
mongoimport --uri="mongodb://localhost:27017/cctv_gis" --collection=vehicle_journeys --file=mongodb_vehicle_journeys.json --jsonArray
```

---

## 2. Essential MongoDB Geospatial Indexes (2dsphere)

Open `mongosh` and create 2dsphere spatial indexes:

```javascript
use cctv_gis;

// Index on camera locations
db.cameras.createIndex({{ "location": "2dsphere" }});

// Index on vehicle sightings
db.vehicle_sightings.createIndex({{ "location": "2dsphere" }});
db.vehicle_sightings.createIndex({{ "registration_number": 1, "timestamp": 1 }});

// Index on multi-camera journey paths
db.vehicle_journeys.createIndex({{ "trajectory": "2dsphere" }});
db.vehicle_journeys.createIndex({{ "registration_number": 1 }});
```

---

## 3. Useful GIS Spatial Queries in MongoDB

### Find all cameras within 2 km of a coordinate:
```javascript
db.cameras.find({{
  location: {{
    $near: {{
      $geometry: {{ type: "Point", coordinates: [72.5620, 23.0286] }},
      $maxDistance: 2000 // meters
    }}
  }}
}});
```

### Trace complete journey of a vehicle by registration number:
```javascript
db.vehicle_journeys.findOne({{ registration_number: "GJ01AB1234" }});
```
"""
    with open(readme_out, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("=" * 65)
    print("  MONGODB GIS DUMMY DATA GENERATOR COMPLETED")
    print("=" * 65)
    print(f"  Target Directory   : {output_dir}")
    print(f"  Cameras Generated  : {len(mongo_cameras)} -> mongodb_cameras.json")
    print(f"  Vehicles Generated : {num_vehicles}")
    print(f"  Sightings Generated: {len(mongo_sightings)} -> mongodb_vehicle_sightings.json")
    print(f"  Journeys Generated : {len(mongo_journeys)} -> mongodb_vehicle_journeys.json")
    print(f"  Guide Created      : {readme_out}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Generate MongoDB & GIS dummy CCTV vehicle tracking data")
    parser.add_argument("--vehicles", type=int, default=25, help="Number of vehicles to simulate (default: 25)")
    parser.add_argument("--min-cameras", type=int, default=3, help="Min cameras per vehicle (default: 3)")
    parser.add_argument("--max-cameras", type=int, default=7, help="Max cameras per vehicle (default: 7)")
    parser.add_argument("--output-dir", default="data/mongodb_export", help="Output directory")
    args = parser.parse_args()

    generate_mongodb_dataset(
        num_vehicles=args.vehicles,
        min_cameras_per_vehicle=args.min_cameras,
        max_cameras_per_vehicle=args.max_cameras,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
