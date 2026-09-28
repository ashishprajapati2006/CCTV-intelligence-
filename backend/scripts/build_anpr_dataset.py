#!/usr/bin/env python3
"""Step 6 - ANPR Dataset Builder Tool (Part 4).

Assists in constructing curated, labeled ANPR evaluation datasets from:
  1. Actual Step 5 plate crops (data/snapshots/<camera_id>/plates/)
  2. Step 6 debug samples (data/anpr/<camera_id>/debug/)
  3. Controlled calibration plate crops

Features:
  - Discovers candidate plate crop images and parses filename metadata
    (camera_id, track_id, frame_id, pts_ms).
  - Matches against candidate JSONL metadata (plate_confidence, plate_quality) when available.
  - Supports interactive labeling via CLI prompts or batch ingestion with a mapping file.
  - Copies samples into data/evaluation/anpr/<split>/images/
  - Generates or updates data/evaluation/anpr/<split>/labels.json.
  - Automatically prevents duplicate records.
  - Records provenance metadata: source ("sentinel" or "controlled"), camera_id, track_id.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.anpr.evaluation import ALLOWED_CONDITIONS
from src.ai.anpr.normalizer import normalize_raw_text
from src.common.logging import get_logger

logger = get_logger("build_anpr_dataset")

# Filename pattern: cam01_TRK-0001_frame000006_pts1084_plate01.jpg
_FILENAME_RE = re.compile(
    r"^(?P<cam>[A-Za-z0-9_-]+)_(?P<track>TRK-[0-9]+)_frame(?P<frame>[0-9]+)_pts(?P<pts>[0-9.]+)_plate(?P<idx>[0-9]+)\.jpg$"
)


def parse_filename_metadata(filename: str) -> Dict[str, Any]:
    """Extract metadata from standard Step 5 plate crop filename."""
    m = _FILENAME_RE.match(filename)
    if m:
        return {
            "camera_id": m.group("cam"),
            "track_id": m.group("track"),
            "frame_id": int(m.group("frame")),
            "pts_ms": float(m.group("pts")),
            "plate_idx": int(m.group("idx")),
        }
    return {
        "camera_id": "unknown",
        "track_id": "unknown",
        "frame_id": 0,
        "pts_ms": 0.0,
        "plate_idx": 0,
    }


def load_candidate_metadata_index(candidates_jsonl: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Index candidate JSONL records by filename for quality & confidence enrichment."""
    index = {}
    if not candidates_jsonl:
        return index

    p = Path(candidates_jsonl)
    if not p.exists():
        return index

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            orig_path = item.get("original_crop_path", "")
            if orig_path:
                fname = Path(orig_path).name
                index[fname] = item
    return index


def build_dataset(
    source_dir: str,
    target_split: str = "development",
    output_base: str = "data/evaluation/anpr",
    candidates_jsonl: Optional[str] = None,
    interactive: bool = False,
    mapping_file: Optional[str] = None,
    source_type: str = "sentinel",
) -> None:
    src = Path(source_dir)
    if not src.exists():
        logger.error("Source directory not found: %s", source_dir)
        sys.exit(1)

    target_dir = Path(output_base) / target_split
    target_images_dir = target_dir / "images"
    target_labels_file = target_dir / "labels.json"

    target_images_dir.mkdir(parents=True, exist_ok=True)

    # Load existing labels to prevent duplicate entries
    existing_labels: List[Dict[str, Any]] = []
    seen_images: Set[str] = set()
    if target_labels_file.exists():
        try:
            with open(target_labels_file, "r", encoding="utf-8") as f:
                existing_labels = json.load(f)
                for rec in existing_labels:
                    img_name = Path(rec.get("image", "")).name
                    if img_name:
                        seen_images.add(img_name)
        except Exception as e:
            logger.warning("Could not read existing labels: %s", e)

    # Load optional candidate metadata index
    cand_index = load_candidate_metadata_index(candidates_jsonl)

    # Load batch mapping if provided: {"filename.jpg": {"ground_truth": "...", "condition": "..."}}
    mappings: Dict[str, Dict[str, str]] = {}
    if mapping_file and Path(mapping_file).exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            mappings = json.load(f)
        logger.info("Loaded %d manual mappings from %s", len(mappings), mapping_file)

    # Find candidate image files
    files = sorted(list(src.glob("*.jpg")) + list(src.glob("*.png")))
    logger.info("Found %d image files in %s", len(files), str(src))

    added_count = 0
    skipped_count = 0

    for fpath in files:
        fname = fpath.name
        if fname in seen_images:
            skipped_count += 1
            continue

        meta = parse_filename_metadata(fname)
        cand_meta = cand_index.get(fname, {})

        gt = ""
        condition = "clear"

        if fname in mappings:
            gt = mappings[fname].get("ground_truth", "")
            condition = mappings[fname].get("condition", "clear")
        elif interactive:
            print("-" * 65)
            print(f"Candidate: {fname}")
            print(f"  Camera : {meta['camera_id']} | Track: {meta['track_id']} | Frame: {meta['frame_id']} | PTS: {meta['pts_ms']}")
            if cand_meta:
                print(f"  Confidence: {cand_meta.get('plate_confidence')} | Quality: {cand_meta.get('plate_quality_score')}")

            val = input("Enter ground truth registration number (or press Enter to skip): ").strip()
            if not val:
                skipped_count += 1
                continue
            gt = normalize_raw_text(val)

            cond_in = input("Enter condition [clear/night/blur/glare/small_plate/angled/occluded] (default: clear): ").strip().lower()
            condition = cond_in if cond_in in ALLOWED_CONDITIONS else "clear"
        else:
            # Non-interactive without mapping: skip unlabeled candidate
            continue

        if not gt:
            continue

        # Copy to split images directory
        dest_img_path = target_images_dir / fname
        if not dest_img_path.exists():
            shutil.copy(str(fpath), str(dest_img_path))

        rel_img_path = f"images/{fname}"
        new_record = {
            "image": rel_img_path,
            "ground_truth": gt,
            "condition": condition,
            "source": source_type,
            "camera_id": meta["camera_id"],
            "track_id": meta["track_id"],
            "frame_id": meta["frame_id"],
            "pts_ms": meta["pts_ms"],
            "plate_confidence": cand_meta.get("plate_confidence"),
            "plate_quality_score": cand_meta.get("plate_quality_score"),
            "original_source_path": str(fpath).replace("\\", "/"),
        }

        existing_labels.append(new_record)
        seen_images.add(fname)
        added_count += 1

    # Save updated labels.json
    with open(target_labels_file, "w", encoding="utf-8") as f:
        json.dump(existing_labels, f, indent=2)

    print("\n" + "=" * 65)
    print("  ANPR DATASET BUILDER REPORT")
    print("=" * 65)
    print(f"  Target Split     : {target_split}")
    print(f"  Target Directory : {target_dir}")
    print(f"  Labels File      : {target_labels_file}")
    print(f"  Samples Added    : {added_count}")
    print(f"  Samples Skipped  : {skipped_count} (already existing or unassigned)")
    print(f"  Total In Dataset : {len(existing_labels)}")
    print("=" * 65 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Step 6 - ANPR Evaluation Dataset Builder")
    ap.add_argument("--source", required=True, help="Source directory containing plate crops")
    ap.add_argument("--split", default="development", choices=["development", "heldout"],
                    help="Target dataset split (default: development)")
    ap.add_argument("--output-base", default="data/evaluation/anpr",
                    help="Base output path (default: data/evaluation/anpr)")
    ap.add_argument("--candidates-jsonl", default=None,
                    help="Optional Step 5 plate_candidates.jsonl for quality metadata")
    ap.add_argument("--mapping-file", default=None,
                    help="Optional JSON file mapping filenames to ground_truth and condition")
    ap.add_argument("--source-type", default="sentinel", choices=["sentinel", "controlled", "unknown"],
                    help="Source provenance (default: sentinel)")
    ap.add_argument("--interactive", action="store_true", help="Interactively label samples via CLI")
    args = ap.parse_args()

    build_dataset(
        source_dir=args.source,
        target_split=args.split,
        output_base=args.output_base,
        candidates_jsonl=args.candidates_jsonl,
        interactive=args.interactive,
        mapping_file=args.mapping_file,
        source_type=args.source_type,
    )


if __name__ == "__main__":
    main()
