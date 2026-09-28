#!/usr/bin/env python3
"""Generates a representative ground-truth evaluation dataset for ANPR."""
from pathlib import Path
import json
import cv2
import numpy as np

img_dir = Path("data/evaluation/anpr/images")
img_dir.mkdir(parents=True, exist_ok=True)

plates_data = [
    ("sample_001.jpg", "GJ01AB1234", "clear", 240, 70),
    ("sample_002.jpg", "GJ05XY7812", "clear", 240, 70),
    ("sample_003.jpg", "MH12DE5678", "night", 240, 70),
    ("sample_004.jpg", "DL03CC4422", "blur", 240, 70),
    ("sample_005.jpg", "GJ27AF9001", "glare", 240, 70),
    ("sample_006.jpg", "GJ01CD3344", "small_plate", 160, 48),
    ("sample_007.jpg", "22BH1234AA", "clear", 240, 70),
    ("sample_008.jpg", "GJ06K5566", "angled", 240, 70),
]

labels = []

for fname, text, cond, w, h in plates_data:
    fpath = img_dir / fname
    img = np.full((h, w, 3), 245 if cond != "night" else 60, dtype=np.uint8)

    # Border
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), (20, 20, 20), 2)

    # Blue IND strip
    cv2.rectangle(img, (3, 3), (26, h - 3), (180, 50, 0), -1)
    cv2.putText(img, "IND", (5, h // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9 if w == 240 else 0.6
    thick = 2
    ts = cv2.getTextSize(text, font, scale, thick)[0]
    tx = 32 + (w - 36 - ts[0]) // 2
    ty = (h + ts[1]) // 2

    color = (0, 0, 0) if cond != "night" else (220, 220, 220)
    cv2.putText(img, text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)

    if cond == "blur":
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
    elif cond == "glare":
        cv2.circle(img, (tx + 40, ty - 5), 25, (255, 255, 255), -1)

    cv2.imwrite(str(fpath), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    track_name = fname.replace(".jpg", "")
    labels.append({
        "track_id": f"TRK-{track_name}",
        "image": str(fpath).replace("\\", "/"),
        "ground_truth": text,
        "condition": cond
    })

eval_json_path = Path("data/evaluation/anpr/labels.json")
eval_json_path.parent.mkdir(parents=True, exist_ok=True)
with open(eval_json_path, "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2)

print(f"Created evaluation dataset: {eval_json_path} with {len(labels)} samples.")
