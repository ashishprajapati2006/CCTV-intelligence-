# Video Project 2 — Dedicated Vehicle Detection & ANPR Pipeline

## Folder Structure

```
video_project2_pipeline/
├── run_pipeline.py          # Main entry point — runs all steps sequentially
├── annotate_video.py        # Step 4 — generates annotated output video + reports
├── README.md                # This file
├── input/                   # (optional) copy of source video
├── processing/              # Intermediate AI pipeline artifacts
│   ├── camera_catalogue.json
│   ├── detections/vp2/      # YOLOv8 detection JSONL + track summaries
│   ├── snapshots/vp2/       # Best-frame snapshots per track
│   └── anpr/                # ANPR/OCR consensus results
└── results/                 # Final outputs
    ├── Video_Project_2_annotated.mp4   # Output video with bounding boxes + labels
    ├── annotated_frames/               # Key frames saved as JPG
    ├── vehicle_crops/                  # Cropped vehicle images per track
    ├── plate_crops/                    # Cropped license plate images
    └── reports/
        ├── vehicles_detected.json      # Full JSON report
        └── vehicles_detected.csv       # CSV summary
```

## How to Run

From the **project root** (`CCTV Platform/`):

```bash
python video_project2_pipeline/run_pipeline.py
```

## Pipeline Steps

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `run_vehicle_detection.py` | YOLOv8 detection every frame (interval=1), confidence≥0.35, top-10 best frames per track |
| 2 | `run_plate_detection.py` | License plate region detection, low threshold to catch all candidates |
| 3 | `run_anpr.py` | EasyOCR / PaddleOCR + Indian plate format validation + multi-frame consensus |
| 4 | `annotate_video.py` | Produces annotated MP4 + frame exports + vehicle crops + JSON/CSV reports |

## Output Label Format

Matches the reference image exactly:
```
[CAR] GJ02HE1110 | Hyundai Creta | SUV | White
```

## Accuracy Settings

- **Detector interval**: 1 (every single frame processed)
- **Detection confidence**: ≥ 0.35
- **Best frames kept**: 10 per track
- **Plate format**: Indian (GJ-XX-XX-XXXX)
- **OCR min confidence**: 0.05 (very low floor to not miss anything)
- **Consensus threshold**: 0.10
