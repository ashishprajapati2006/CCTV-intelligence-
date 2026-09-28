"""FastAPI Backend Application for Gujarat Police CCTV Command Centre."""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure backend directory is in sys.path before internal imports
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = BACKEND_DIR.parent if (BACKEND_DIR.parent / "frontend").exists() else BACKEND_DIR
PROJECT_ROOT = BACKEND_DIR
DATA_DIR = (BACKEND_DIR / "data") if (BACKEND_DIR / "data").exists() else (REPO_ROOT / "data")
FRONTEND_DIST = (REPO_ROOT / "frontend" / "dist") if (REPO_ROOT / "frontend" / "dist").exists() else (BACKEND_DIR / "frontend" / "dist")

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# PEP 366: Support direct script execution without package resolution errors
if __name__ == "__main__" and not __package__:
    __package__ = "src.api"

try:
    from .routes.cameras import router as cameras_router
    from .routes.vehicles import router as vehicles_router
    from .routes.watchlist import router as watchlist_router
    from .routes.alerts import router as alerts_router
    from .routes.dashboard import router as dashboard_router
    from .routes.synthetic import router as synthetic_router
except (ImportError, ModuleNotFoundError):
    from src.api.routes.cameras import router as cameras_router
    from src.api.routes.vehicles import router as vehicles_router
    from src.api.routes.watchlist import router as watchlist_router
    from src.api.routes.alerts import router as alerts_router
    from src.api.routes.dashboard import router as dashboard_router
    from src.api.routes.synthetic import router as synthetic_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cctv_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan. Models are lazy-loaded on demand to conserve container RAM and avoid OOM crashes."""
    logger.info("CCTV Intelligence Platform API started. AI models configured for on-demand lazy loading.")
    yield


app = FastAPI(
    title="Gujarat Police CCTV Intelligence Platform API",
    description="Backend API powering the Operator Command Centre (Step 14). Integrates CCTV catalogue, authenticated HLS proxying, ANPR results, watchlist matching, and journey reconstruction.",
    version="1.0.0",
    lifespan=lifespan,
)


# CORS configuration for production Vercel frontend, Render backend, and local development
cors_origins = [
    "https://cctv-intelligence.vercel.app",
    "https://cctv-intelligence-platform.vercel.app",
    "https://cctv-intelligence.onrender.com",
    "https://cctv-intelligence-platform-backend.onrender.com",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

env_cors = os.getenv("CORS_ORIGINS", "")
if env_cors:
    cors_origins.extend([origin.strip() for origin in env_cors.split(",") if origin.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"https?://.*\.vercel\.app|https?://.*\.onrender\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount data directory as static evidence endpoint
if DATA_DIR.exists():
    app.mount("/api/evidence", StaticFiles(directory=str(DATA_DIR)), name="evidence")

# Include API routers first
app.include_router(cameras_router)
app.include_router(vehicles_router)
app.include_router(watchlist_router)
app.include_router(alerts_router)
app.include_router(dashboard_router)
app.include_router(synthetic_router)


@app.get("/api/health")
def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "Gujarat Police CCTV Intelligence Platform API",
        "version": "1.0.0",
        "steps_supported": "1 through 14"
    }


STANDALONE_HTML = (REPO_ROOT / "frontend" / "public" / "synthetic-standalone.html") if (REPO_ROOT / "frontend" / "public" / "synthetic-standalone.html").exists() else (BACKEND_DIR / "frontend" / "public" / "synthetic-standalone.html")

@app.get("/synthetic-standalone")
def synthetic_standalone():
    """Direct standalone web UI for synthetic video AI vehicle & plate detection."""
    if STANDALONE_HTML.exists():
        return FileResponse(STANDALONE_HTML)
    return {"error": "Standalone HTML template not found"}


# Mount built React frontend if available
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    app_module = "backend.src.api.app:app" if (REPO_ROOT / "backend").exists() else "src.api.app:app"
    app_dir = str(REPO_ROOT) if (REPO_ROOT / "backend").exists() else str(BACKEND_DIR)
    uvicorn.run(app_module, host="0.0.0.0", port=port, reload=True, app_dir=app_dir)

