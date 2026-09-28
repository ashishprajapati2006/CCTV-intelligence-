#!/usr/bin/env python3
"""Step 6 - Isolated OCR Sanity Test Tool.

Validates the OCR engine independently of the video pipeline.

Usage:
  # Test with an existing plate image:
  python scripts/test_ocr.py --image data/snapshots/cam01/plates/sample.jpg

  # Generate and test a synthetic Indian license plate image:
  python scripts/test_ocr.py --generate-synthetic

Features:
  1. Loads and validates image dimensions.
  2. Runs EasyOCR with allowlist.
  3. Displays raw OCR detections, confidence, and latency.
  4. Runs text normalization & character correction audit.
  5. Validates Indian registration format pattern.
  6. Reports clear pass/fail status.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.anpr.ocr import EasyOCREngine, OCREngineFacade
from src.ai.anpr.normalizer import normalize_with_audit
from src.ai.anpr.validator import validate_indian_plate
from src.common.logging import get_logger

logger = get_logger("test_ocr")


def generate_synthetic_plate_image(
    text: str = "GJ01AB1234",
    output_path: str = "data/evaluation/anpr/images/synthetic_plate.jpg",
    width: int = 240,
    height: int = 70,
) -> str:
    """Generate a clean synthetic Indian license plate image for sanity testing."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # White plate background
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Black border
    cv2.rectangle(img, (2, 2), (width - 3, height - 3), (0, 0, 0), 2)

    # Blue IND strip on left (standard HSRP plate)
    cv2.rectangle(img, (4, 4), (28, height - 4), (180, 50, 0), -1)
    cv2.putText(img, "IND", (6, height // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Plate text (Black)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.05
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = 36 + (width - 40 - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logger.info("Saved synthetic plate image to: %s", str(p))
    return str(p)


def run_ocr_sanity_test(image_path: str, engine_name: str = "easyocr") -> bool:
    print("=" * 65)
    print("  GUJARAT POLICE CCTV - OCR ENGINE SANITY TEST")
    print("=" * 65)
    print(f"  Target image : {image_path}")
    print(f"  OCR Engine   : {engine_name}")
    print("=" * 65 + "\n")

    p = Path(image_path)
    if not p.exists():
        logger.error("Image file not found: %s", image_path)
        return False

    img = cv2.imread(str(p))
    if img is None or img.size == 0:
        logger.error("Could not decode image at: %s", image_path)
        return False

    h, w = img.shape[:2]
    print(f"[*] Image loaded successfully: {w}x{h} px ({p.stat().st_size} bytes)")

    # 1. Initialize Engine
    ocr = OCREngineFacade(preferred_engine=engine_name)
    print(f"[*] OCR Engine initialized: {ocr.engine_name} ({ocr.engine_version})")

    # 2. Run Recognition
    print("[*] Running OCR inference...")
    ocr_res = ocr.recognize(img, variant_name="original")

    print("\n" + "-" * 65)
    print("  RAW OCR DETECTION")
    print("-" * 65)
    print(f"  Raw text          : '{ocr_res.raw_text}'")
    print(f"  Confidence        : {ocr_res.confidence:.4f}")
    print(f"  Processing time   : {ocr_res.processing_time_ms:.1f} ms")

    if not ocr_res.raw_text:
        print("\n[!] OCR result is EMPTY. The image contains no readable text.")
        return False

    # 3. Normalization & Audit
    norm_res = normalize_with_audit(ocr_res.raw_text, apply_contextual_corrections=True)
    is_valid, fmt_name, _ = validate_indian_plate(norm_res.normalized_text)

    print("\n" + "-" * 65)
    print("  NORMALIZATION & FORMAT VALIDATION")
    print("-" * 65)
    print(f"  Normalized text   : '{norm_res.normalized_text}'")
    print(f"  Corrections       : {len(norm_res.corrections)} audit log(s)")
    for c in norm_res.corrections:
        print(f"    - pos {c.position}: '{c.original_char}' -> '{c.corrected_char}' ({c.reason})")
    print(f"  Format valid      : {is_valid} ({fmt_name or 'unrecognized'})")

    print("\n" + "=" * 65)
    if ocr_res.raw_text:
        print("  STATUS: OCR TEST PASSED")
    else:
        print("  STATUS: OCR TEST FAILED (EMPTY)")
    print("=" * 65 + "\n")

    return bool(ocr_res.raw_text)


def main():
    ap = argparse.ArgumentParser(description="Step 6 - Isolated OCR Sanity Test Tool")
    ap.add_argument("--image", default=None, help="Path to plate crop image")
    ap.add_argument("--engine", default="easyocr", choices=["easyocr", "opencv"], help="OCR engine to test")
    ap.add_argument("--generate-synthetic", action="store_true", help="Generate and test a synthetic plate image")
    args = ap.parse_args()

    target_image = args.image
    if args.generate_synthetic or not target_image:
        target_image = generate_synthetic_plate_image(text="GJ01AB1234")

    success = run_ocr_sanity_test(target_image, engine_name=args.engine)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
