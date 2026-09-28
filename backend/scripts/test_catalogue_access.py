#!/usr/bin/env python3
"""Diagnostic script to test access to the Sentinel camera catalogue.

Usage:
    python scripts/test_catalogue_access.py

Windows CMD Alternative:
    curl -i -L https://cctv.corp8.cloud/cameras.json
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import requests

from src.common.logging import mask_sensitive

load_dotenv()


def run_diagnostic():
    catalogue_url = os.getenv("SENTINEL_CATALOGUE_URL", "https://cctv.corp8.cloud/cameras.json")
    timeout = 15

    print("============================================================")
    print("           SENTINEL CATALOGUE DIAGNOSTIC TOOL               ")
    print("============================================================")
    print(f"Configured URL:\n  {catalogue_url}\n")

    http_status = None
    final_url = catalogue_url
    content_type = "Unknown"
    response_size = 0
    result_text = ""

    try:
        # Step 1: Probe without auto-redirect to capture initial 301/302 status code
        initial_resp = requests.get(catalogue_url, timeout=timeout, allow_redirects=False)
        http_status = initial_resp.status_code
        content_type = initial_resp.headers.get("Content-Type", "Unknown")
        response_size = len(initial_resp.content)

        if initial_resp.is_redirect:
            redirect_target = initial_resp.headers.get("Location", "")
            final_url = redirect_target
            if "login" in redirect_target or "auth" in redirect_target:
                result_text = "AUTHENTICATION REQUIRED (Redirected to login endpoint)"
            else:
                result_text = f"HTTP REDIRECT ({http_status}) -> {redirect_target}"

        # Step 2: Follow redirects to report final landing state
        full_resp = requests.get(catalogue_url, timeout=timeout, allow_redirects=True)
        final_status = full_resp.status_code
        final_url = full_resp.url
        content_type = full_resp.headers.get("Content-Type", "Unknown")
        response_size = len(full_resp.content)

        if http_status is None:
            http_status = final_status

        # Evaluate final status
        if final_status in [401, 403]:
            result_text = "AUTHENTICATION REQUIRED (HTTP 401/403 Unauthorized)"
        elif final_status == 404:
            result_text = "ENDPOINT NOT FOUND (HTTP 404)"
        elif "login" in final_url or "Sign in" in full_resp.text:
            result_text = "AUTHENTICATION REQUIRED (Landed on login page)"
        elif final_status == 200:
            if "application/json" in content_type or full_resp.text.strip().startswith(("{", "[")):
                result_text = "SUCCESS: Catalogue accessible as valid JSON"
            else:
                result_text = f"UNEXPECTED CONTENT: Received HTTP 200 with non-JSON ({content_type})"
        else:
            result_text = f"HTTP STATUS {final_status}"

    except requests.exceptions.SSLError as e:
        result_text = "TLS/SSL ERROR: Certificate verification failed"
    except requests.exceptions.ConnectionError as e:
        result_text = "NETWORK/DNS ERROR: Host unreachable or DNS resolution failed"
    except requests.exceptions.Timeout:
        result_text = f"TIMEOUT: Request timed out after {timeout} seconds"
    except Exception as e:
        result_text = f"FAILURE: {mask_sensitive(str(e))}"

    print(f"HTTP Status:\n  {http_status}")
    print(f"Final URL:\n  {final_url}")
    print(f"Content-Type:\n  {content_type}")
    print(f"Response Size:\n  {response_size} bytes\n")
    print(f"Result:\n  {result_text}")
    print("------------------------------------------------------------")
    print("Windows CMD Manual Diagnostic Command:")
    print(f"  curl -i -L {catalogue_url}")
    print("============================================================\n")

    if "AUTHENTICATION REQUIRED" in result_text:
        print("[!] NOTICE: Catalogue requires authorized access.")
        print("    Per Sentinel guidelines, do not attempt to bypass authentication.")
        print("    Ensure authorized credentials are set in .env or switch to local mode.\n")


if __name__ == "__main__":
    run_diagnostic()
