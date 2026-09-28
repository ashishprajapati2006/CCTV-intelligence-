"""Catalogue Synchronization Module.

Fetches, validates, normalizes, and locally stores the Sentinel camera catalogue.
Can be executed as: python -m src.catalogue.sync
"""
import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from src.catalogue.client import CatalogueClient, AuthenticationRequiredError
from src.catalogue.parser import CatalogueParser
from src.catalogue.validator import CatalogueValidator
from src.catalogue.models import Camera
from src.common.logging import get_logger, mask_sensitive

# Load environment variables
load_dotenv()

logger = get_logger("catalogue_sync")


def sync_catalogue(
    source: Optional[str] = None,
    local_file: Optional[str] = None,
    output_dir: str = "data/catalogue",
) -> List[Camera]:
    """Synchronize camera catalogue from live Sentinel or local development file."""
    source = (source or os.getenv("CATALOGUE_SOURCE", "live")).strip().lower()
    raw_dir = Path(output_dir) / "raw"
    normalized_dir = Path(output_dir) / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    parser = CatalogueParser()
    validator = CatalogueValidator()
    raw_json_data = None

    if source == "local":
        print("\n========================================================")
        print(" [!] WARNING: LOCAL DEVELOPMENT CATALOGUE IN USE")
        print("========================================================\n")
        logger.warning("Operating in LOCAL development catalogue mode.")

        file_path = Path(local_file or os.getenv("LOCAL_CATALOGUE_PATH", "data/catalogue/normalized/cameras.json"))
        if not file_path.exists():
            raise FileNotFoundError(
                f"Local catalogue file not found at: {file_path}. "
                "Provide a valid file or switch CATALOGUE_SOURCE=live."
            )

        with open(file_path, "r", encoding="utf-8") as f:
            raw_json_data = json.load(f)

    else:
        print("\n========================================================")
        print(" [*] CONNECTING TO LIVE SENTINEL CAMERA CATALOGUE")
        print("========================================================\n")
        logger.info("Attempting live catalogue sync from Sentinel...")

        client = CatalogueClient()
        try:
            raw_json_data, raw_text = client.fetch_catalogue()

            # Save raw untouched response with timestamp ONLY when valid JSON
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            raw_filename = raw_dir / f"cameras_{timestamp_str}.json"
            with open(raw_filename, "w", encoding="utf-8") as f:
                f.write(raw_text)
            logger.info("Raw catalogue snapshot saved to: %s", raw_filename)

        except AuthenticationRequiredError as auth_err:
            logger.error("AUTHENTICATION REQUIRED: %s", auth_err)
            print("\n--------------------------------------------------------")
            print(" RESULT: Catalogue requires authorized access.")
            print(" Provide valid credentials in .env (SENTINEL_AUTH_PASSWORD,")
            print(" SENTINEL_AUTH_COOKIE, or SENTINEL_AUTH_TOKEN)")
            print(" or set CATALOGUE_SOURCE=local for local development.")
            print("--------------------------------------------------------\n")
            raise
        except Exception as e:
            logger.error("Failed to retrieve live catalogue: %s", mask_sensitive(str(e)))
            raise

    # Parse payload
    cameras = parser.parse_payload(raw_json_data)
    logger.info("Parsed %d cameras from payload.", len(cameras))

    # Validate
    is_valid, errors, warnings = validator.validate_catalogue(cameras)
    for warn in warnings:
        logger.warning("Catalogue Validation Warning: %s", warn)

    if not is_valid:
        for err in errors:
            logger.error("Catalogue Validation Error: %s", err)
        raise ValueError(f"Catalogue validation failed with {len(errors)} error(s).")

    # Save normalized catalogue
    normalized_path = normalized_dir / "cameras.json"
    normalized_data = [cam.to_dict() for cam in cameras]
    with open(normalized_path, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, indent=2, ensure_ascii=False)

    logger.info("Normalized catalogue successfully written to: %s", normalized_path)
    print(f"\n[+] Catalogue synchronization complete: {len(cameras)} cameras normalized.")
    print(f"[+] Saved to: {normalized_path}\n")

    return cameras


def main():
    """CLI entrypoint for catalogue sync."""
    arg_parser = argparse.ArgumentParser(description="Synchronize Sentinel Camera Catalogue.")
    arg_parser.add_argument(
        "--source",
        choices=["live", "local"],
        default=None,
        help="Catalogue source: 'live' or 'local' (defaults to CATALOGUE_SOURCE env var)",
    )
    arg_parser.add_argument(
        "--file",
        dest="local_file",
        default=None,
        help="Path to local catalogue file (used when source is 'local')",
    )

    args = arg_parser.parse_args()

    try:
        sync_catalogue(source=args.source, local_file=args.local_file)
    except AuthenticationRequiredError:
        sys.exit(2)
    except Exception as e:
        logger.error("Sync terminated with error: %s", mask_sensitive(str(e)))
        sys.exit(1)


if __name__ == "__main__":
    main()
