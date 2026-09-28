# Unified AI-Powered CCTV Intelligence Platform

> **From fragmented CCTV infrastructure to unified, AI-driven operational intelligence.**

A modular, enterprise-grade CCTV intelligence platform designed for integrating heterogeneous cameras, NVRs, and VMS systems into a centralized, situational-awareness command centre. The platform unifies CCTV registry and GIS mapping, real-time stream ingestion, AI-based vehicle detection and multi-object tracking, high-precision ANPR/OCR, watchlist correlation, automated alerts, cross-camera identity tracking, and vehicle journey reconstruction.

---

## 📌 Executive Summary

### Problem
Surveillance and traffic CCTV infrastructure across administrative zones and municipal departments is heavily fragmented across disparate camera vendors, proprietary VMS/NVR systems, varying network topologies, legacy protocols, and isolated monitoring rooms. This fragmentation severely impedes:
- Unified situational awareness across jurisdictions.
- Timely vehicle identification and suspect interception.
- Cross-camera tracking and journey reconstruction.
- Rapid emergency response and automated law-enforcement dispatch.

### Proposed Solution — Hybrid Model 5 Architecture
The platform implements a **Hybrid Model 5 Architecture**:
- **Model 1 — CCTV Registry + GIS:** Centralized camera inventory, ownership, geographic coordinates, hardware metadata, and live operational health.
- **Model 3 — Federation & Integration:** Standardized abstraction adapters for heterogeneous VMS, NVR, and camera systems using RTSP, ONVIF, REST APIs, and vendor SDKs.
- **Selective Model 2 — Direct Stream Integration:** Direct low-latency RTSP/ONVIF processing where edge camera access is provisioned.

> **Core Architectural Principle:** *Centralize intelligence — not necessarily every raw video stream.*

---

## ⚡ Key Features

- **Centralized CCTV Registry & GIS Mapping:** Interactive geospatial map showing real-time camera locations, status, operational health, and field-of-view coverage.
- **Heterogeneous VMS / NVR Federation:** Protocol adapters unifying RTSP, ONVIF, HLS, and WebRTC across diverse manufacturers (Hikvision, Dahua, Axis, CP Plus, etc.).
- **Live Video Ingestion Pipeline:** Resilient RTSP ingestion with TCP transport, automatic exponential reconnect, and wall-clock media PTS synchronization.
- **AI Vehicle Detection & Tracking:** High-speed YOLOv8 detection and lightweight centroid/IoU tracking for cars, motorcycles, buses, and trucks at 20–25+ FPS.
- **Automatic Number Plate Recognition (ANPR / OCR):** Dedicated license plate localization, adaptive contrast enhancement, and deep learning OCR (EasyOCR / CRNN).
- **Multi-Frame Consensus Engine:** Filters optical noise, character jitter, and transient occlusion across multiple consecutive frames before locking registration numbers.
- **Classified Watchlist Management:** Centralized registry for critical targets (*Stolen Vehicles, Wanted Persons, Missing Persons, Blacklisted Vehicles, Suspect Vehicles*).
- **Real-Time Automated Alerting:** Sub-second watchlist correlation, flashing HUD banners, targeted red alert bounding boxes, and 1-click operator acknowledgement.
- **Cross-Camera Vehicle Correlation:** Associates sightings across multiple camera locations using normalized registration numbers and PTS temporal sequences.
- **Vehicle Journey Reconstruction:** Reconstructs chronological movement dossiers with interactive GIS route polylines and speed/timestamp telemetry.
- **Role-Based Access Control (RBAC) & Audit Logs:** Immutable audit history capturing operator actions, alert dispatches, and evidence chain of custody.

---

## 🔄 End-to-End Workflow

```
 CCTV Cameras / NVRs / VMS
            ↓
 Integration & Federation Layer (RTSP / ONVIF / APIs / SDKs)
            ↓
 Stream Ingestion & Pre-processing (PTS Sync & Bilinear Scale)
            ↓
 Vehicle Detection & Multi-Object Tracking (YOLOv8)
            ↓
 Best-Frame Selection & Plate Region Localization
            ↓
 Deep Learning ANPR / OCR
            ↓
 Multi-Frame Consensus Normalization
            ↓
 Observed Vehicle Database
            ↓
 Watchlist Database Correlation
            ↓
 Automated Real-Time Alert (Red Reticle + HUD Banner)
            ↓
 Cross-Camera Correlation
            ↓
 Vehicle Journey Analysis
            ↓
 Unified Command Centre Dashboard
```

---

## 🧠 AI & Computer Vision Pipeline

```
       [Video Frame Ingestion]
                  ↓
       [Vehicle Detection (YOLOv8)]
                  ↓
       [Centroid & IoU Tracking]
                  ↓
       [Best-Frame Crop Selection]
                  ↓
    [Plate Region Localization (YOLO)]
                  ↓
    [Image Pre-processing & Contrast Boost]
                  ↓
         [ANPR / Deep OCR]
                  ↓
      [Multi-Frame Consensus Filter]
                  ↓
   [Normalized Registration & Confidence]
                  ↓
      [Vehicle Observation Dossier]
```

### Traceable Confidence Scoring
The recognition pipeline calculates independent confidence metrics to guarantee evidentiary integrity:
1. **Vehicle Detection Confidence:** YOLO classification score ($C_{\text{veh}} \ge 0.25$).
2. **Plate Localization Confidence:** Bounding box accuracy of the plate detector.
3. **OCR Recognition Score:** Character-level probability score from the neural OCR engine.
4. **Multi-Frame Consensus Score:** Ratio of identical character sequences across $N$ continuous frames.

---

## 🚨 Watchlist Correlation & Alerting Engine

```
      [Recognized Registration Plate]
                     ↓
      [Alphanumeric Normalization (Regex)]
                     ↓
     [Hash-Indexed Watchlist Database Lookup]
                     ↓
                Match Found?
               /            \
             Yes             No
             /                \
   [Validate Threshold]   [Store Standard Observation]
            ↓
   [Generate Alert Event]
            ↓
   [Attach Evidence Snapshots & FIR Case Info]
            ↓
   [Push to Command Centre HUD & WebSocket Feed]
```

### Visual Alerting Standard
- **Matched Vehicles:** Highlighted with a high-visibility **Red Bounding Box**, corner reticle brackets, FIR case details, and a pulsating top HUD banner.
- **Benign Traffic:** Processed cleanly without screen-cluttering bounding boxes, keeping the operator focused strictly on high-priority security events.
- **Strict Deduplication:** Enforces **one alert per vehicle/plate per session** to prevent alert fatigue.

---

## 🌐 Cross-Camera Intelligence & Journey Analysis

The platform indexes normalized vehicle registration numbers as primary correlation keys across distributed camera feeds:

```
                  ┌── [CAM-01: Ring Road North] ── 14:02:15 UTC
                  │
[GJ 01 AB 1234] ──┼── [CAM-07: Highway Toll Plaza] ── 14:18:40 UTC
                  │
                  └── [CAM-12: City Centre Junction] ── 14:32:05 UTC
```

- **Identity Correlation vs. Physical Journey:** The system distinguishes verified temporal detections from continuous trajectories. Missing intermediate camera sightings are represented as dotted hypothesis corridors rather than confirmed routes.
- **Dossier Generation:** Generates comprehensive PDF/JSON investigation dossiers containing chronological timestamps, camera IDs, geo-coordinates, and high-resolution plate crops.

---

## 🏗️ System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │           Physical CCTV Ecosystem            │
                       │     (PTZ, Bullet, ANPR Cameras, NVRs)       │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │        Integration & Federation Layer        │
                       │         (RTSP / ONVIF / HLS / WebRTC)        │
                       └──────────────┬────────────────┬──────────────┘
                                      │                │
            ┌─────────────────────────┘                └─────────────────────────┐
            ▼                                                                    ▼
┌───────────────────────┐                                            ┌───────────────────────┐
│  CCTV Registry & GIS  │                                            │ Stream & AI Engine    │
│  - Camera Inventory   │                                            │ - YOLOv8 Vehicle Det  │
│  - GeoJSON Mapping    │                                            │ - Centroid Tracker    │
│  - Hardware Telemetry │                                            │ - Plate Localizer     │
│  - Stream Health      │                                            │ - EasyOCR Inference   │
└───────────┬───────────┘                                            └───────────┬───────────┘
            │                                                                    │
            └─────────────────────────┬──────────────────────────────────────────┘
                                      ▼
                       ┌──────────────────────────────────────────────┐
                       │     Intelligence & Correlation Backbone      │
                       │   - Watchlist Matching & Fuzzy Lookup        │
                       │   - Real-Time Alert Dispatcher               │
                       │   - Cross-Camera Tracking & Journey Analysis │
                       └──────────────────────┬───────────────────────┘
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │           Data & Event Persistence           │
                       │      - PostgreSQL + PostGIS (Spatial)        │
                       │      - Apache Kafka / JSONL Event Logs       │
                       │      - Snapshot & Crop Storage Cache         │
                       └──────────────────────┬───────────────────────┘
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │       Unified React Command Centre           │
                       │  (Live Video Wall, GIS Map, Studio, Alerts)  │
                       └──────────────────────────────────────────────┘
```

---

## 💻 Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend / Command Centre** | React 18, TypeScript, Vite, Tailwind CSS, Lucide React, Leaflet GIS |
| **Backend & REST APIs** | Python 3.10+, FastAPI, Uvicorn, Pydantic, WebSockets, AsyncIO |
| **Video & Stream Ingestion** | OpenCV (`cv2`), FFmpeg, RTSP over TCP, MJPEG Streaming |
| **AI & Computer Vision** | PyTorch, Ultralytics YOLOv8, EasyOCR (Deep CRNN/ResNet), NumPy |
| **Database & Geospatial** | PostgreSQL, PostGIS, SQLAlchemy, JSONL audit stores |
| **Security & Auditing** | JWT Bearer Auth, Role-Based Access Control (RBAC), TLS/HTTPS, Audit Trail |
| **Deployment & DevOps** | Docker, Docker Compose, Nginx, Vercel (Frontend), Cloud Run / VM (Backend) |

---

## 📁 Repository Directory Structure

```
cctv-intelligence-platform/
├── README.md                           # Master system documentation
├── .gitignore                          # Git exclude rules (ignores large videos/caches)
├── .env.example                        # Environment variables template
├── requirements.txt                    # Python dependencies
├── weights/
│   ├── yolov8n.pt                      # YOLOv8 vehicle detection weights
│   └── license_plate_detector.pt       # Custom YOLO license plate localization weights
├── data/
│   ├── watchlist/                      # Classified law-enforcement watchlists
│   │   └── vehicles/
│   │       ├── watchlist.json
│   │       └── synthetic_watchlist.json
│   ├── matches/                        # Confirmed alert matches & audit history
│   │   ├── alerts_state.json
│   │   └── confirmed/matches.jsonl
│   └── synthetic_cache/                # Cached thumbnails and plate crops
├── Synthetic Dataset/                  # Operational test video feeds (1.mp4 - 8.mp4)
├── src/
│   ├── api/                            # FastAPI REST API & Streaming Engine
│   │   ├── app.py                      # FastAPI application factory & middleware
│   │   └── routes/
│   │       ├── synthetic.py            # Live MJPEG streaming, AI worker & alert engine
│   │       ├── cameras.py              # CCTV registry and camera telemetry
│   │       ├── vehicles.py             # Vehicle detection records & dossiers
│   │       ├── watchlist.py            # Watchlist CRUD and target classification
│   │       ├── alerts.py               # Alert acknowledgement and audit dispatch
│   │       └── dashboard.py            # System analytics and operational metrics
│   ├── catalogue/                      # Camera discovery and inventory management
│   ├── streaming/                      # RTSP / TCP ingestion pipelines
│   └── common/                         # Shared logging, security, and time utilities
└── frontend/                           # React Command Centre Web Application
    ├── package.json
    ├── vite.config.ts
    ├── vercel.json                     # Vercel deployment configuration
    ├── src/
    │   ├── App.tsx                     # React Router routes
    │   ├── pages/
    │   │   ├── Dashboard.tsx           # High-level command overview
    │   │   ├── SyntheticStudio.tsx     # CCTV live AI stream, video shelf & upload
    │   │   ├── Watchlist.tsx           # Law-enforcement target management
    │   │   ├── Alerts.tsx              # Alert audit and operator acknowledgement
    │   │   ├── CameraMap.tsx           # GIS interactive camera map
    │   │   └── VideoWall.tsx           # Multi-grid surveillance monitor
    │   └── services/
    │       └── api.ts                  # Centralized HTTP & evidence resolution client
    └── public/
        └── synthetic-standalone.html   # Ultra-lightweight standalone CCTV console
```

---

## 🚀 Getting Started (Local Development)

### Prerequisites
- **Python 3.10+** (64-bit)
- **Node.js 18+** & `npm`
- **Git**
- **FFmpeg** installed and added to system PATH

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/cctv-intelligence-platform.git
cd cctv-intelligence-platform
```

### 2. Backend Setup
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI Backend Server
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
The backend will be live at `http://localhost:8000` (Swagger API documentation available at `http://localhost:8000/docs`).

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The Command Centre interface will open at `http://localhost:5173`.

---

## ☁️ Deployment Guide (Connecting Deployed Vercel Link to Backend)

The frontend is deployed on Vercel at:  
👉 **[CCTV Intelligence Platform (Vercel)](https://cctv-intelligence.vercel.app/)**

### Why is a Cloud Backend Required?
Vercel hosts static frontend files serverlessly. Real-time computer vision inference (YOLOv8, OpenCV video streaming, and EasyOCR) requires a persistent Python backend runtime.

### Option 1: Deploy Backend to Cloud (Render / Railway / Cloud Run)
1. Deploy the root repository to [Render](https://render.com) or [Railway](https://railway.app) as a Web Service:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`
2. Once deployed, copy your backend URL (e.g., `https://cctv-backend.onrender.com`).
3. In your **Vercel Dashboard** under Project Settings → **Environment Variables**, add:
   ```env
   VITE_API_URL=https://cctv-backend.onrender.com
   ```
4. Trigger a Redeploy on Vercel. All synthetic video streams, uploads, and AI alerts will now run seamlessly on your live Vercel URL!

### Option 2: Live Local Tunnel to Vercel (Instant & Free via ngrok)
If you wish to demonstrate the platform on the deployed Vercel link using your local GPU/CPU:
1. Start your local backend:
   ```bash
   python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```
2. In a separate terminal, expose your local port:
   ```bash
   ngrok http 8000
   ```
3. Copy the secure public URL generated by ngrok (`https://xxxx.ngrok-free.app`).
4. Set `VITE_API_URL=https://xxxx.ngrok-free.app` in Vercel Environment Variables and redeploy.

---

## 📈 Scalability & Large-Scale Deployment

Designed for statewide and multi-city surveillance scaling (up to 80,000+ cameras):
- **Edge Analytics Nodes:** Lightweight YOLOv8 inference deployed at regional substations/servers, sending only lightweight JSON metadata and evidence snapshots over the WAN.
- **Centralized Event Backbone:** Apache Kafka distributed topics handle thousands of concurrent vehicle detection messages per second.
- **Selective Full-Stream Ingestion:** High-bandwidth video streams are decoded on-demand or during active priority alerts.
- **Tiered Evidence Storage:** High-resolution snapshots are retained for 30–90 days with cold-tier automated archiving.

---

## 🔒 Security & Responsible Use

- **Data Privacy:** Synthetic demonstration hotlists are utilized for validation to protect operational law-enforcement data.
- **Human-in-the-Loop:** All AI detections are treated as investigative decision-support cues requiring certified operator acknowledgement and manual review.
- **Role-Based Access Control:** Strict permission tiers separating field officers, command centre analysts, and system administrators.
- **Tamper-Evident Audit Logging:** Every plate lookup, match confirmation, and export event is logged with operator ID and UTC timestamps.

---

## 🚗 Step 11 — Cross-Camera Vehicle Correlation & Journey Reconstruction

Step 11 correlates recognized vehicle sightings across the integrated CCTV network into an observation history and journey timeline, adhering to the **Common-Clock Rule**:

### Key Principles
- **No Global PTS Assumption:** Camera-local PTS values are never subtracted across different cameras to fabricate travel durations or speeds.
- **Strict Timing Separation:**
  - `source_time_status == "RESOLVED"`: Chronological timeline with Haversine distance, travel time delta, and straight-line implied speed.
  - `source_time_status == "NOT_RESOLVED"`: Output is explicitly labeled as **`CAMERA-LOCAL OBSERVATION SEQUENCE`** (not a validated route). Cross-camera time deltas are marked unresolved.
- **Plausibility State Machine:**
  - `PLAUSIBLE`: Implied speed $\le 120\text{ km/h}$.
  - `POSSIBLE`: Implied speed between $120\text{ km/h}$ and $200\text{ km/h}$ (warning).
  - `ANOMALOUS`: Implied speed $> 200\text{ km/h}$ or negative time delta (flagged for review).
  - `UNKNOWN`: Insufficient temporal or spatial data to compute speed.
- **Confidence Scoring:** Transparent multi-factor score: Recognition Quality (35%), Temporal Resolution (25%), Spatial Resolution (20%), Plausibility (20%).

### Usage CLI
```bash
# Build journeys for all observed vehicles
python scripts/build_vehicle_journey.py --all

# Query journey for a specific vehicle registration
python scripts/query_vehicle_journey.py --registration CHME

# Audit journey integrity and Common-Clock rule compliance
python scripts/validate_journey.py --registration CHME

# Find vehicles observed across multiple distinct cameras
python scripts/find_multicamera_vehicles.py --min-cameras 2
```

---

## 🗺️ Step 12 — PostgreSQL + PostGIS + GIS Data Layer

Step 12 establishes PostgreSQL + PostGIS as the authoritative spatial and relational data store for the platform.

### Architecture Highlights
- **Spatial Indexing:** PostGIS `geometry(Point, 4326)` with GIST indexing on cameras.
- **Radial & Distance Queries:** High-performance spatial lookups using `ST_DistanceSphere` and `ST_MakeEnvelope`.
- **Zero Coordinate Fabrication:** Unverified cameras have `geom = NULL`.
- **Media Timing Integrity:** PTS (`recognition_pts_ms`) and wall-clock `source_time` remain strictly separated.
- **Evidence Provenance:** Raw OCR text and evidence image crop paths are preserved in `anpr_observations`.
- **Alembic Migrations:** Versioned schema management with automatic PostGIS extension provisioning.

### Database Setup & Operations
```bash
# 1. Initialize database and enable PostGIS extension
python scripts/init_database.py

# 2. Execute idempotent JSON/JSONL migration
python scripts/migrate_json_to_postgres.py

# 3. Audit database health and row counts
python scripts/verify_database.py

# 4. Run database & spatial test suite
pytest tests/test_database.py -v
```

### Environment Configuration (.env)
```env
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=cctv_platform
DATABASE_USER=cctv_user
DATABASE_PASSWORD=your_password
```

---

## 📄 License
This project is distributed under the **MIT License**. See `LICENSE` for details.

```
CONNECT ──► PROCESS ──► UNDERSTAND ──► CORRELATE ──► ALERT ──► INVESTIGATE ──► ACT
```
