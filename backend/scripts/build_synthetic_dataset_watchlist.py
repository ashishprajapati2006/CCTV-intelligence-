"""Script to generate a representative, police-classified watchlist database
correlated directly with detected entities from the Synthetic Dataset.

Classifications:
  1. STOLEN_VEHICLE
  2. WANTED_PERSON (Vehicle linked to wanted suspect/fugitive)
  3. MISSING_PERSON_VEHICLE (Silver / Amber alert)
  4. BLACKLISTED_VEHICLE (Revoked permit / court impound / illegal transport)
  5. SUSPECT_VEHICLE (Surveillance vehicle of interest)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("synthetic_watchlist_builder")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_RESULT_DIR = PROJECT_ROOT / "Synthetic Dataset_result" / "anpr"
OUTPUT_PATH = PROJECT_ROOT / "data" / "watchlist" / "vehicles" / "synthetic_watchlist.json"
CENTRAL_WATCHLIST_PATH = PROJECT_ROOT / "data" / "watchlist" / "vehicles" / "watchlist.json"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Curated law-enforcement mock dossiers mapped to synthetic detections
CORRELATED_PROFILES: Dict[str, Dict[str, Any]] = {
    "VW1292": {
        "category": "STOLEN_VEHICLE",
        "priority": "CRITICAL",
        "description": "2024 White Sedan reported stolen during armed carjacking (Gujarat FIR #GJ-2026-8831).",
        "notes": "Correlated with Synthetic CCTV Dataset Video 4 (Track TRK-0001). Suspects considered armed.",
        "case_number": "FIR-GJ-2026-8831",
        "jurisdiction": "Ahmedabad City Police",
    },
    "JO8HCH": {
        "category": "WANTED_PERSON",
        "priority": "CRITICAL",
        "description": "Silver SUV registered to fugitive wanted under NBW warrant (Narcotics & Organised Crime).",
        "notes": "Correlated with Synthetic CCTV Dataset Video 4 (Track TRK-0003). Immediate interception requested.",
        "case_number": "WNT-2026-0419",
        "jurisdiction": "State Crime Branch",
    },
    "BB53567": {
        "category": "SUSPECT_VEHICLE",
        "priority": "HIGH",
        "description": "Dark Blue Sedan observed operating in multiple surveillance loops near critical transit infrastructure.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 5 (Track TRK-0023).",
        "case_number": "INTEL-2026-904",
        "jurisdiction": "Special Operations Group (SOG)",
    },
    "ABFE3975": {
        "category": "BLACKLISTED_VEHICLE",
        "priority": "HIGH",
        "description": "Commercial cargo carrier with revoked transport permit and active court impound order.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 5 (Track TRK-0031). Impound upon sighting.",
        "case_number": "RTO-BLK-2026-118",
        "jurisdiction": "Gujarat Transport Enforcement",
    },
    "ICK09": {
        "category": "MISSING_PERSON_VEHICLE",
        "priority": "MEDIUM",
        "description": "Maroon hatchback linked to reported missing vulnerable adult and child welfare alert.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 3 (Track TRK-0001). Non-tactical welfare stop.",
        "case_number": "MISS-2026-5520",
        "jurisdiction": "Surat Rural Police",
    },
    "657048": {
        "category": "STOLEN_VEHICLE",
        "priority": "HIGH",
        "description": "Delivery van reported stolen from commercial warehouse logistics yard overnight.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 5 (Track TRK-0009).",
        "case_number": "FIR-GJ-2026-9012",
        "jurisdiction": "Vadodara City Police",
    },
    "VISION": {
        "category": "SUSPECT_VEHICLE",
        "priority": "MEDIUM",
        "description": "Van bearing custom vanity plate flagged in cross-border contraband surveillance inquiry.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 7 (Track TRK-0140).",
        "case_number": "INTEL-2026-1102",
        "jurisdiction": "State CID Crime",
    },
    "C98L9U81": {
        "category": "BLACKLISTED_VEHICLE",
        "priority": "MEDIUM",
        "description": "Vehicle registered with fake tax token and flagged for repeated toll evasion and toll runner violations.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 5 (Track TRK-0001).",
        "case_number": "TOLL-BLK-2026-784",
        "jurisdiction": "Highway Safety Patrol",
    },
    "LJOSEPD": {
        "category": "WANTED_PERSON",
        "priority": "HIGH",
        "description": "Grey MPV registered to individual with active arrest warrant for financial fraud and absconding.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 7 (Track TRK-0047).",
        "case_number": "WNT-2026-1123",
        "jurisdiction": "Economic Offences Wing",
    },
    "VOVS": {
        "category": "MISSING_PERSON_VEHICLE",
        "priority": "LOW",
        "description": "Compact city car linked to missing elderly citizen overdue return notice.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 8 (Track TRK-0010).",
        "case_number": "MISS-2026-3391",
        "jurisdiction": "Rajkot City Police",
    },
    "3O28H7250": {
        "category": "STOLEN_VEHICLE",
        "priority": "CRITICAL",
        "description": "Heavy duty commercial truck stolen from national highway transport bay.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 8 (Track TRK-0022). High-value cargo.",
        "case_number": "FIR-GJ-2026-9452",
        "jurisdiction": "Highway Patrol Task Force",
    },
    "7L644344": {
        "category": "SUSPECT_VEHICLE",
        "priority": "LOW",
        "description": "Observed stationary vehicle near perimeter fence during anomalous hours.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 5 (Track TRK-0015).",
        "case_number": "INTEL-2026-1215",
        "jurisdiction": "Perimeter Security Command",
    },
    "SMH6J43": {
        "category": "BLACKLISTED_VEHICLE",
        "priority": "LOW",
        "description": "Vehicle with multiple unpaid court summons and delinquent speed camera notices.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 1 (Track TRK-0034).",
        "case_number": "TRAF-2026-4401",
        "jurisdiction": "Traffic Police Division",
    },
    "EIJ3I": {
        "category": "SUSPECT_VEHICLE",
        "priority": "LOW",
        "description": "Vehicle of interest in residential burglary ring reconnaissance.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 1 (Track TRK-0056).",
        "case_number": "INTEL-2026-1402",
        "jurisdiction": "Zone-2 Investigation Unit",
    },
    "7I3E": {
        "category": "MISSING_PERSON_VEHICLE",
        "priority": "LOW",
        "description": "Family welfare check requested for vehicle observed away from usual route.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 1 (Track TRK-0063).",
        "case_number": "MISS-2026-4019",
        "jurisdiction": "Gandhinagar Police",
    },
    "T3A3": {
        "category": "WANTED_PERSON",
        "priority": "MEDIUM",
        "description": "Vehicle identified during getaway in jewellery store smash-and-grab.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 3 (Track TRK-0009).",
        "case_number": "FIR-GJ-2026-7784",
        "jurisdiction": "Crime Branch Unit 4",
    },
    "13E3": {
        "category": "STOLEN_VEHICLE",
        "priority": "LOW",
        "description": "Motorcycle reported stolen from university campus parking.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 3 (Track TRK-0015).",
        "case_number": "FIR-GJ-2026-3310",
        "jurisdiction": "University Police Station",
    },
    "DOICC": {
        "category": "SUSPECT_VEHICLE",
        "priority": "LOW",
        "description": "Vehicle associated with scrap metal theft syndicate.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 7 (Track TRK-0095).",
        "case_number": "INTEL-2026-1509",
        "jurisdiction": "Industrial Area Police",
    },
    "033556": {
        "category": "BLACKLISTED_VEHICLE",
        "priority": "LOW",
        "description": "Repeated red-light violation delinquent with suspended driving license.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 7 (Track TRK-0263).",
        "case_number": "TRAF-2026-8902",
        "jurisdiction": "Traffic Command Centre",
    },
    "3WCUL": {
        "category": "STOLEN_VEHICLE",
        "priority": "LOW",
        "description": "Stolen rental vehicle unreturned across state border.",
        "notes": "Correlated with Synthetic CCTV Dataset Video 7 (Track TRK-0005).",
        "case_number": "FIR-GJ-2026-6641",
        "jurisdiction": "Inter-State Crime Liaison",
    },
}

# Control non-matching vehicles (To demonstrate negative control)
CONTROL_VEHICLES: List[Dict[str, Any]] = [
    {
        "plate": "GJ01AB9999",
        "category": "STOLEN_VEHICLE",
        "priority": "CRITICAL",
        "description": "Control negative target: Luxury SUV reported hijacked on express highway.",
        "notes": "Negative control vehicle — not present in Synthetic Dataset.",
    },
    {
        "plate": "MH02CD8888",
        "category": "WANTED_PERSON",
        "priority": "HIGH",
        "description": "Control negative target: Interstate fugitive vehicle.",
        "notes": "Negative control vehicle — not present in Synthetic Dataset.",
    },
    {
        "plate": "DL04EF7777",
        "category": "BLACKLISTED_VEHICLE",
        "priority": "MEDIUM",
        "description": "Control negative target: Blacklisted hazardous material carrier without clearance.",
        "notes": "Negative control vehicle — not present in Synthetic Dataset.",
    },
    {
        "plate": "RJ14GH6666",
        "category": "MISSING_PERSON_VEHICLE",
        "priority": "LOW",
        "description": "Control negative target: Missing person tourist vehicle inquiry.",
        "notes": "Negative control vehicle — not present in Synthetic Dataset.",
    },
    {
        "plate": "GJ05JK5555",
        "category": "SUSPECT_VEHICLE",
        "priority": "HIGH",
        "description": "Control negative target: Nighttime warehouse surveillance suspect.",
        "notes": "Negative control vehicle — not present in Synthetic Dataset.",
    },
]


def build_synthetic_watchlist():
    now_iso = datetime.now(timezone.utc).isoformat()
    watchlist_records = []
    idx = 1

    # 1. Add All Correlated Synthetic Detections
    for plate, profile in CORRELATED_PROFILES.items():
        record = {
            "watchlist_id": f"WL-SYN-{idx:06d}",
            "registration_number": plate,
            "normalized_registration_number": plate,
            "category": profile["category"],
            "priority": profile["priority"],
            "status": "ACTIVE",
            "entity_type": "vehicle",
            "source": "SYNTHETIC_DEMO",
            "synthetic": True,
            "description": profile["description"],
            "notes": profile["notes"],
            "case_number": profile.get("case_number", f"CASE-SYN-{idx:04d}"),
            "jurisdiction": profile.get("jurisdiction", "Gujarat Police Command Centre"),
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        watchlist_records.append(record)
        idx += 1

    # 2. Add Control Non-Matching Targets
    for ctrl in CONTROL_VEHICLES:
        record = {
            "watchlist_id": f"WL-SYN-{idx:06d}",
            "registration_number": ctrl["plate"],
            "normalized_registration_number": ctrl["plate"],
            "category": ctrl["category"],
            "priority": ctrl["priority"],
            "status": "ACTIVE",
            "entity_type": "vehicle",
            "source": "SYNTHETIC_DEMO",
            "synthetic": True,
            "description": ctrl["description"],
            "notes": ctrl["notes"],
            "case_number": f"CTRL-NEG-{idx:04d}",
            "jurisdiction": "Control Benchmark",
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        watchlist_records.append(record)
        idx += 1

    # Save to dedicated synthetic watchlist file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(watchlist_records, f, indent=2)
    logger.info(f"Generated {len(watchlist_records)} synthetic watchlist records -> {OUTPUT_PATH}")

    # Merge into Central Watchlist Path so the central backend/command centre recognizes them
    existing_records = []
    if CENTRAL_WATCHLIST_PATH.exists():
        try:
            with open(CENTRAL_WATCHLIST_PATH, "r", encoding="utf-8") as f:
                existing_records = json.load(f)
        except Exception:
            existing_records = []

    # De-duplicate by normalized_registration_number
    merged_map = {r["normalized_registration_number"]: r for r in existing_records}
    for new_r in watchlist_records:
        merged_map[new_r["normalized_registration_number"]] = new_r

    merged_list = list(merged_map.values())
    with open(CENTRAL_WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, indent=2)
    logger.info(f"Merged into central watchlist -> {CENTRAL_WATCHLIST_PATH} (Total: {len(merged_list)} records)")

    # Print Breakdown Report
    print("=" * 70)
    print("  GUJARAT POLICE - REPRESENTATIVE SYNTHETIC WATCHLIST DATABASE")
    print("=" * 70)
    print(f"  Total Watchlist Targets Generated : {len(watchlist_records)}")
    print(f"  Correlated Observed Matches       : {len(CORRELATED_PROFILES)} targets")
    print(f"  Negative Control Targets          : {len(CONTROL_VEHICLES)} targets")
    print("-" * 70)
    
    categories = {}
    priorities = {}
    for r in watchlist_records:
        categories[r["category"]] = categories.get(r["category"], 0) + 1
        priorities[r["priority"]] = priorities.get(r["priority"], 0) + 1

    print("  CLASSIFICATION BY POLICE CATEGORY:")
    for cat, count in categories.items():
        print(f"    • {cat:<25}: {count} records")

    print("\n  PRIORITY DISTRIBUTION:")
    for prio, count in priorities.items():
        print(f"    • {prio:<10}: {count} records")
    print("=" * 70)


if __name__ == "__main__":
    build_synthetic_watchlist()
