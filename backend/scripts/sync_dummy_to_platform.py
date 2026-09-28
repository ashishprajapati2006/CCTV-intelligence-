#!/usr/bin/env python3
"""sync_dummy_to_platform.py

Synchronizes the generated 50 multi-camera vehicle journeys from
data/mongodb_export/ into:
  1. data/observed/vehicles/vehicles.json (Observed vehicle database)
  2. data/observed/vehicles/observations.jsonl (Observation log)
  3. data/journeys/reports/<REG>_journey.json (Journey reports)
  4. data/journeys/vehicles/<REG>.json (Journey dossiers)
  5. frontend/src/data/fallbackData.ts (Frontend static fallback for Vercel/offline)

This ensures the GIS map, vehicles page, and dashboard immediately display
these multi-camera vehicles and their full traversal paths.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "data" / "mongodb_export"
JOURNEYS_FILE = EXPORT_DIR / "mongodb_vehicle_journeys.json"
SIGHTINGS_FILE = EXPORT_DIR / "mongodb_vehicle_sightings.json"

OBS_VEHICLES_FILE = ROOT / "data" / "observed" / "vehicles" / "vehicles.json"
OBS_JSONL_FILE = ROOT / "data" / "observed" / "vehicles" / "observations.jsonl"
JOURNEYS_REP_DIR = ROOT / "data" / "journeys" / "reports"
JOURNEYS_VEH_DIR = ROOT / "data" / "journeys" / "vehicles"
FALLBACK_TS_FILE = ROOT / "frontend" / "src" / "data" / "fallbackData.ts"


def sync():
    if not JOURNEYS_FILE.exists():
        print(f"Error: {JOURNEYS_FILE} does not exist. Run generate_mongodb_dummy_cctv.py first.")
        return

    with open(JOURNEYS_FILE, "r", encoding="utf-8") as f:
        journeys = json.load(f)

    with open(SIGHTINGS_FILE, "r", encoding="utf-8") as f:
        sightings = json.load(f)

    # 1. Update data/observed/vehicles/vehicles.json
    existing_vehicles = []
    if OBS_VEHICLES_FILE.exists():
        with open(OBS_VEHICLES_FILE, "r", encoding="utf-8") as f:
            try:
                existing_vehicles = json.load(f)
            except Exception:
                existing_vehicles = []

    # Map by registration number to avoid duplicates
    existing_reg_map = {v.get("registration_number", "").upper(): idx for idx, v in enumerate(existing_vehicles)}

    new_vehicle_objects = []
    for j in journeys:
        reg = j["registration_number"].upper()
        timeline = []
        for idx, seg in enumerate(j.get("segments", [])):
            timeline.append({
                "camera_id": seg["camera_id"],
                "track_id": f"TRK-SIM-{idx+1:04d}",
                "first_seen_pts_ms": float(idx * 60000.0),
                "recognition_pts_ms": float(idx * 60000.0 + 5000.0),
                "last_seen_pts_ms": float(idx * 60000.0 + 15000.0),
                "source_time": seg.get("timestamp"),
                "source_time_status": "RESOLVED",
                "status": "CONFIRMED",
                "consensus_score": round(0.92 + (idx % 5) * 0.015, 3),
                "evidence_image": None,
                "evidence_filename_pts_status": "MATCH",
                "ingested_at_utc": seg.get("timestamp"),
            })

        v_obj = {
            "vehicle_id": j["vehicle_id"],
            "registration_number": reg,
            "normalized_registration_number": reg,
            "first_seen": {
                "camera_id": j["camera_sequence"][0],
                "pts_ms": 0.0,
                "source_time": j["first_seen"]["timestamp"],
                "source_time_status": "RESOLVED",
            },
            "last_seen": {
                "camera_id": j["camera_sequence"][-1],
                "pts_ms": float((len(j["camera_sequence"]) - 1) * 60000.0),
                "source_time": j["last_seen"]["timestamp"],
                "source_time_status": "RESOLVED",
            },
            "camera_count": j["camera_count"],
            "cameras": j["camera_sequence"],
            "observation_count": len(j["camera_sequence"]),
            "track_count": len(j["camera_sequence"]),
            "best_consensus_score": 0.98,
            "average_consensus_score": 0.94,
            "status": "CONFIRMED",
            "timeline": timeline,
            "vehicle_details": j.get("vehicle_details"),
            "total_distance_km": j.get("total_distance_km"),
            "avg_speed_kmh": j.get("avg_speed_kmh"),
            "is_watchlist_match": j.get("is_watchlist_match", False),
        }

        if reg in existing_reg_map:
            existing_vehicles[existing_reg_map[reg]] = v_obj
        else:
            existing_vehicles.insert(0, v_obj)  # put at front for prominent display
            new_vehicle_objects.append(v_obj)

    OBS_VEHICLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OBS_VEHICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_vehicles, f, indent=2)
    print(f"[+] Updated {OBS_VEHICLES_FILE} with {len(journeys)} multi-camera vehicles.")

    # 2. Append observations to observations.jsonl
    OBS_JSONL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OBS_JSONL_FILE, "a", encoding="utf-8") as f:
        for s in sightings:
            rec = {
                "observation_id": s["sighting_id"],
                "camera_id": s["camera_id"],
                "track_id": f"TRK-SIM-{s['camera_id']}",
                "registration_number": s["registration_number"],
                "normalized_registration_number": s["registration_number"],
                "consensus_score": s.get("confidence_score", 0.95),
                "ocr_confidence": s.get("confidence_score", 0.95),
                "first_seen_pts_ms": 10000.0,
                "last_seen_pts_ms": 25000.0,
                "evidence_image_path": None,
                "source_time": s.get("timestamp"),
                "status": "CONFIRMED",
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[+] Appended {len(sightings)} sightings to {OBS_JSONL_FILE}.")

    # 3. Write individual journey dossiers and reports
    JOURNEYS_REP_DIR.mkdir(parents=True, exist_ok=True)
    JOURNEYS_VEH_DIR.mkdir(parents=True, exist_ok=True)
    for j in journeys:
        reg = j["registration_number"].upper()
        # Build dossier
        dossier = {
            "journey_id": j["journey_id"],
            "vehicle_id": j["vehicle_id"],
            "registration_number": reg,
            "normalized_registration_number": reg,
            "ordering_mode": "SOURCE_TIME",
            "status": "RESOLVED_MULTI_CAMERA_JOURNEY",
            "camera_count": j["camera_count"],
            "total_distance_km": j.get("total_distance_km"),
            "avg_speed_kmh": j.get("avg_speed_kmh"),
            "segments": [
                {
                    "segment_id": f"SEG-{j['vehicle_id']}-{seg['camera_id']}",
                    "vehicle_id": j["vehicle_id"],
                    "registration_number": reg,
                    "camera_id": seg["camera_id"],
                    "camera_name": seg["camera_name"],
                    "camera_latitude": seg["coordinates"][1],
                    "camera_longitude": seg["coordinates"][0],
                    "source_time": seg["timestamp"],
                    "source_time_status": "RESOLVED",
                    "recognition_status": "CONFIRMED",
                    "consensus_score": 0.95,
                    "ocr_confidence": 0.95,
                    "speed_kmh": seg.get("speed_kmh"),
                }
                for seg in j.get("segments", [])
            ],
            "legs": [
                {
                    "from_camera_id": j["segments"][i]["camera_id"],
                    "to_camera_id": j["segments"][i+1]["camera_id"],
                    "from_camera_name": j["segments"][i]["camera_name"],
                    "to_camera_name": j["segments"][i+1]["camera_name"],
                    "plausibility": "PLAUSIBLE",
                    "speed_kmh": j["segments"][i+1].get("speed_kmh", 45.0),
                }
                for i in range(len(j.get("segments", [])) - 1)
            ],
            "confidence_score": {
                "overall_score": 0.94,
                "recognition_quality": 0.96,
                "temporal_resolution": 0.92,
                "spatial_resolution": 0.95,
                "plausibility_consistency": 0.98,
                "notes": ["Verified multi-camera GPS trajectory", "Source time synchronized"],
            },
            "trajectory": j.get("trajectory"),
        }

        with open(JOURNEYS_REP_DIR / f"{reg}_journey.json", "w", encoding="utf-8") as rf:
            json.dump(dossier, rf, indent=2)
        with open(JOURNEYS_VEH_DIR / f"{reg}.json", "w", encoding="utf-8") as vf:
            json.dump(dossier, vf, indent=2)

    print(f"[+] Written 50 journey dossiers to {JOURNEYS_REP_DIR} and {JOURNEYS_VEH_DIR}.")

    # 4. Update frontend/src/data/fallbackData.ts FALLBACK_VEHICLES
    if FALLBACK_TS_FILE.exists():
        content = FALLBACK_TS_FILE.read_text(encoding="utf-8")
        marker = "export const FALLBACK_VEHICLES: ObservedVehicle[] = ("
        if marker in content:
            idx = content.find(marker)
            prefix = content[:idx + len(marker)]
            # Find the closing of this array
            # We can re-serialize the top 35 vehicles from existing_vehicles
            top_vehicles = existing_vehicles[:45]
            json_dump = json.dumps(top_vehicles, indent=2)
            suffix = ");\n"
            # Look for the matching end
            end_marker = "export const "
            next_export_idx = content.find(end_marker, idx + len(marker))
            if next_export_idx != -1:
                # Find the closing parenthesis before next_export_idx
                closing_paren_idx = content.rfind(")", idx, next_export_idx)
                if closing_paren_idx != -1:
                    new_content = prefix + json_dump + content[closing_paren_idx:]
                    FALLBACK_TS_FILE.write_text(new_content, encoding="utf-8")
                    print(f"[+] Updated {FALLBACK_TS_FILE} with top {len(top_vehicles)} multi-camera vehicles.")
            else:
                # End of file
                closing_paren_idx = content.rfind(")")
                if closing_paren_idx != -1:
                    new_content = prefix + json_dump + content[closing_paren_idx:]
                    FALLBACK_TS_FILE.write_text(new_content, encoding="utf-8")
                    print(f"[+] Updated {FALLBACK_TS_FILE} to EOF.")


if __name__ == "__main__":
    sync()
