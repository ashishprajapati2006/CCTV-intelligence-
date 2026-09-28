"""Root entrypoint forwarding to backend.src.api.app for 100% backward compatibility."""
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Import FastAPI app from the backend source
from backend.src.api.app import app

__all__ = ["app"]
