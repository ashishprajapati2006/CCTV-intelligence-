#!/usr/bin/env python3
import subprocess
import time
import glob

# Focus on 3 key cameras and process a much longer duration (2000 frames)
CAMERAS = ["cam01", "cam07", "cam02"]
FRAMES = 2000

def run_cmd(cmd):
    print(f">>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)

def main():
    print("========================================")
    print("  DEEP SCANNING CAMERAS (2000 frames)")
    print("========================================")
    
    for cam in CAMERAS:
        # 1. Vehicle Detection (optimized by checking every 4th frame)
        run_cmd([
            "python", "scripts/run_vehicle_detection.py",
            "--camera-id", cam,
            "--frames", str(FRAMES),
            "--no-display"
        ])
        
        # 2. Plate Cropping
        run_cmd([
            "python", "scripts/run_plate_detection.py",
            "--camera-id", cam,
            "--tracks", f"data/detections/{cam}/tracks.json",
            "--no-display"
        ])
        
        # 3. OCR / ANPR
        run_cmd([
            "python", "scripts/run_anpr.py",
            "--camera-id", cam,
            "--no-display"
        ])

    print("========================================")
    print("  REBUILDING DATABASES")
    print("========================================")
    
    # Grab all consensus files dynamically
    consensus_files = glob.glob("data/anpr/cam*/consensus/anpr_consensus.json")
    
    db_cmd = ["python", "scripts/build_observed_vehicle_db.py", "--rebuild"]
    for f in consensus_files:
        db_cmd.extend(["--input", f])
        
    run_cmd(db_cmd)
    
    # Rebuild Journeys
    run_cmd(["python", "scripts/build_vehicle_journey.py", "--all"])
    
    # Generate updated reports
    run_cmd(["python", "scripts/report_observation_dataset.py"])

if __name__ == "__main__":
    main()
