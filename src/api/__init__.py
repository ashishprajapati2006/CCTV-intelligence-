"""Root API compatibility package proxying to backend/src/api."""
import sys
from pathlib import Path

_BACKEND_API = Path(__file__).resolve().parent.parent.parent / "backend" / "src" / "api"

if str(_BACKEND_API) not in __path__:
    __path__.append(str(_BACKEND_API))
