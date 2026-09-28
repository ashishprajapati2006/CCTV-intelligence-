"""Run script for Gujarat Police CCTV Command Centre (Step 14)."""
import argparse
import sys
import uvicorn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Start Gujarat Police CCTV Command Centre (Step 14)")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    args = parser.parse_args()

    print("=" * 70)
    print("  GUJARAT POLICE INNOVATION HACKATHON 2026 — STEP 14 COMMAND CENTRE")
    print("=" * 70)
    print(f"[*] Command Centre UI  : http://localhost:{args.port}/")
    print(f"[*] Interactive API Docs: http://localhost:{args.port}/docs")
    print(f"[*] Active Streams Proxy: http://localhost:{args.port}/api/cameras/cam01/hls/index.m3u8")
    print(f"[*] Live Alert Stream   : http://localhost:{args.port}/api/alerts/stream/live")
    print("-" * 70)
    print("Architecture: Hybrid Model 5 (Catalogue + Federation + Live HLS)")
    print("Press Ctrl+C to stop the server.\n")

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
