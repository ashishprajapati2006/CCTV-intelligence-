# Gujarat Police CCTV Intelligence Platform — Backend Service

High-performance, container-ready backend API and real-time AI processing pipeline for the CCTV Intelligence Platform.

---

## Architecture Overview

```
backend/
├── src/
│   ├── api/                 # FastAPI routes, schemas, and endpoints
│   │   ├── routes/
│   │   │   ├── cameras.py   # Camera catalogue & authenticated streaming proxy
│   │   │   ├── vehicles.py  # Observed vehicles & journey analytics
│   │   │   ├── watchlist.py # Watchlist management & CRUD
│   │   │   ├── alerts.py    # Match alerts & operator acknowledgement
│   │   │   ├── synthetic.py # Real-time YOLOv8 + ANPR pipeline & frame analyzer
│   │   │   └── dashboard.py # Aggregated metrics & activity feeds
│   │   └── app.py           # FastAPI application & lifespan pre-warming
│   ├── ai/                  # AI core, plate localization, and OCR normalization
│   ├── anpr/                # ANPR consensus & evaluation
│   ├── catalogue/           # Camera catalogue parser
│   ├── common/              # Logging, metrics, time utilities
│   ├── database/            # PostgreSQL / SQLite async database connection
│   ├── detection/           # YOLOv8 object detector wrappers
│   ├── events/              # Event serialization & retry handling
│   ├── journey/             # Multi-camera vehicle trajectory graph reconstruction
│   ├── kafka/               # Kafka producers & consumers
│   ├── observed/            # Observation aggregation repository
│   ├── streaming/           # RTSP & Sentinel HLS stream readers
│   ├── tracking/            # Centroid & IoU vehicle tracker
│   └── watchlist/           # Watchlist validation & decision engine
├── config/                  # Configuration YAMLs
├── data/                    # Data stores (catalogue, observed, watchlist, matches)
├── database/                # SQL schemas
├── migrations/              # Alembic database migrations
├── scripts/                 # Utility, seeding, and migration scripts
├── tests/                   # Pytest test suite (100% passing)
├── tools/                   # Batch processing & offline video annotation tools
├── weights/                 # YOLOv8 & license plate detector weights
├── alembic.ini              # Alembic migration configuration
├── pytest.ini               # Pytest configuration
└── requirements.txt         # Production Python dependencies
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Run Local Development Server
From the `backend/` directory:
```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Or from the repository root:
```bash
python -m uvicorn backend.src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Run Automated Tests
```bash
pytest tests/test_api.py -v
```
