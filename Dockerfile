# ==============================================================================
# Gujarat Police CCTV Intelligence Platform — Production Multi-Stage Dockerfile
# ==============================================================================

# --- Stage 1: Build Frontend Assets ---
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# --- Stage 2: Production Python Backend Runtime ---
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

# Install system dependencies required for OpenCV, FFmpeg video decoding, and ANPR
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only build first to prevent bulky CUDA wheels
# (Conserves memory and ensures smooth operation under constrained cloud containers)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache EasyOCR model weights into ~/.EasyOCR so no download occurs at runtime
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Copy model files and weights
COPY yolov8n.pt /app/yolov8n.pt
COPY license_plate_detector.pt /app/license_plate_detector.pt
COPY weights/ /app/weights/

# Pre-warm YOLO weights
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('license_plate_detector.pt')"

# Copy core source and configuration
COPY src/ /app/src/
COPY config/ /app/config/
COPY data/ /app/data/

# Copy built frontend assets so FastAPI can serve both API & Web UI
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist
COPY frontend/public/ /app/frontend/public/

# Ensure runtime directories exist
RUN mkdir -p /app/data/synthetic_cache/thumbnails \
             /app/data/synthetic_cache/snapshots \
             /app/Synthetic\ Dataset/uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

CMD ["sh", "-c", "python -m uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
