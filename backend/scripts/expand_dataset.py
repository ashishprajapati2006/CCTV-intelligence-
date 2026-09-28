#!/usr/bin/env python3
import subprocess
import sys
import time

CAMERAS_TO_PROCESS = [f"cam{i:02d}" for i in range(2, 21)]  # cam02 to cam20
FRAMES = 300

def run_cmd(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}")
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with code {e.returncode}")
    print(f"<<< Finished in {time.time() - start:.1f}s")

def main():
    print("========================================")
    print("  EXPANDING DATASET (cam02 to cam20)")
    print("========================================")
    
    for cam in CAMERAS_TO_PROCESS:
        print(f"\n--- Processing {cam} ---")
        # Step 4: Vehicle Detection
        run_cmd([
            "python", "scripts/run_vehicle_detection.py",
            "--camera-id", cam,
            "--frames", str(FRAMES),
            "--no-display"
        ])
        
        # Step 5: Plate Detection
        run_cmd([
            "python", "scripts/run_plate_detection.py",
            "--camera-id", cam,
            "--tracks", f"data/detections/{cam}/tracks.json",
            "--no-display"
        ])
        
        # Step 6: ANPR
        run_cmd([
            "python", "scripts/run_anpr.py",
            "--camera-id", cam,
            "--no-display"
        ])

    print("\n========================================")
    print("  BUILDING AGGREGATED DATABASES")
    print("========================================")
    # Step 7: Observed DB
    run_cmd(["python", "scripts/build_observed_vehicle_db.py"])
    
    # Step 11: Journeys
    run_cmd(["python", "scripts/build_vehicle_journey.py", "--all", "--include-probable"])
    
    # Step 11.1 Reports
    run_cmd(["python", "scripts/report_observation_dataset.py", "--include-uncertain"])
    run_cmd(["python", "scripts/find_multicamera_vehicles.py", "--include-probable"])

if __name__ == "__main__":
    main()
