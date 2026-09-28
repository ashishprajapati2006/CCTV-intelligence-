"""Root compatibility package proxying to backend/src."""
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
_BACKEND_SRC = _BACKEND_DIR / "src"

if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Extend __path__ so that any subpackage inside backend/src is resolved by Python
if str(_BACKEND_SRC) not in __path__:
    __path__.append(str(_BACKEND_SRC))
