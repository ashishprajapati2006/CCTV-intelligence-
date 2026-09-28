#!/usr/bin/env python3
r"""Download the YOLO License Plate Detector model (license_plate_detector.pt).

This script downloads the best publicly available YOLOv8 nano license plate
detection model. Once downloaded, the pipeline will automatically use it
instead of the classical contour detector.

Usage:
  python scripts/download_lp_model.py

The model will be saved as 'license_plate_detector.pt' in the project root.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

SOURCES = [
    (
        "GitHub - computervisioneng ANPR YOLOv8",
        "https://github.com/niconielsen32/YOLOv8-LicensePlateDetector/raw/main/runs/detect/train2/weights/best.pt",
    ),
    (
        "GitHub - Winter2121 LP Detector",
        "https://github.com/Winter2121/License-Plate-Recognition/raw/main/weights/license_plate_detector.pt",
    ),
    (
        "GitHub - AntonioConsiglio LP YOLOv8",
        "https://github.com/AntonioConsiglio/license_plate_detection/raw/main/weights/best.pt",
    ),
]

TARGET = Path("license_plate_detector.pt")


def try_download(name: str, url: str) -> bool:
    print(f"  Trying: {name}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 100_000:   # too small to be a real model
            print(f"    [SKIP] File too small ({len(data)} bytes), probably not a valid model.")
            return False
        TARGET.write_bytes(data)
        size_mb = round(len(data) / 1024 / 1024, 2)
        print(f"    [OK] Downloaded {size_mb} MB -> {TARGET}")
        return True
    except Exception as e:
        print(f"    [FAIL] {e}")
        return False


def verify_model() -> bool:
    try:
        from ultralytics import YOLO
        import numpy as np
        m = YOLO(str(TARGET))
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        m.predict(dummy, verbose=False, conf=0.5)
        nc = len(m.names)
        print(f"  [OK] Model verified. Classes: {nc} ({list(m.names.values())[:5]})")
        return True
    except Exception as e:
        print(f"  [FAIL] Model verification failed: {e}")
        return False


def main():
    print("=" * 65)
    print("  GUJARAT POLICE CCTV -- YOLO LP MODEL DOWNLOADER")
    print("=" * 65)

    if TARGET.exists() and TARGET.stat().st_size > 1_000_000:
        print(f"Model already exists: {TARGET} ({round(TARGET.stat().st_size/1024/1024, 2)} MB)")
        print("Verifying existing model...")
        if verify_model():
            print("\n[DONE] Existing model is valid. No download needed.")
            return
        else:
            print("Existing model invalid, re-downloading...")
            TARGET.unlink()

    print("\nDownloading license plate detector model...")
    for name, url in SOURCES:
        if try_download(name, url):
            print("\nVerifying downloaded model...")
            if verify_model():
                print("\n[SUCCESS] License plate detector is ready!")
                print(f"  File: {TARGET.resolve()}")
                print("\nThe pipeline will now automatically use YOLO LP detection")
                print("instead of the classical contour detector.")
                return
            else:
                TARGET.unlink(missing_ok=True)

    # Fallback: manual instruction
    print("\n" + "=" * 65)
    print("[MANUAL DOWNLOAD REQUIRED]")
    print("=" * 65)
    print("Automatic download failed. Please manually download the model:")
    print()
    print("  Option A (Recommended - keremberke YOLOv8 LP):")
    print("  1. Go to: https://huggingface.co/keremberke/yolov8n-license-plate")
    print("  2. Click 'Files and versions' tab")
    print("  3. Download 'best.pt'")
    print("  4. Rename it to 'license_plate_detector.pt'")
    print("  5. Place it in your project root folder")
    print()
    print("  Option B (Direct GitHub):")
    print("  1. Go to: https://github.com/ultralytics/ultralytics")
    print("  2. In issues/examples, find ANPR demo 'license_plate_detector.pt'")
    print()
    print("Once placed, re-run your pipeline commands.")
    sys.exit(1)


if __name__ == "__main__":
    main()
