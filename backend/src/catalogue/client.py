"""HTTP Client for authorized access to Sentinel camera catalogue."""
import os
import json
from typing import Any, Dict, Optional, Tuple, Union
import requests
from src.common.logging import get_logger, mask_sensitive

logger = get_logger("catalogue_client")


class AuthenticationRequiredError(Exception):
    """Raised when the Sentinel catalogue requires authorized credentials."""
    pass


class CatalogueClient:
    """Client for retrieving camera catalogue from Sentinel."""

    DEFAULT_CATALOGUE_URL = "https://cctv.corp8.cloud/cameras.json"
    DEFAULT_LOGIN_URL = "https://cctv.corp8.cloud/auth/login"

    def __init__(
        self,
        catalogue_url: Optional[str] = None,
        timeout: int = 15,
        auth_token: Optional[str] = None,
        auth_cookie: Optional[str] = None,
        auth_password: Optional[str] = None,
        auth_email: Optional[str] = None,
    ):
        self.catalogue_url = catalogue_url or os.getenv(
            "SENTINEL_CATALOGUE_URL", self.DEFAULT_CATALOGUE_URL
        )
        self.timeout = timeout
        self.auth_token = auth_token or os.getenv("SENTINEL_AUTH_TOKEN")
        self.auth_cookie = auth_cookie or os.getenv("SENTINEL_AUTH_COOKIE")
        self.auth_password = auth_password or os.getenv("SENTINEL_AUTH_PASSWORD")
        self.auth_email = auth_email or os.getenv("SENTINEL_AUTH_EMAIL")
        self.session = requests.Session()

    def _setup_auth(self) -> None:
        """Configure session with authorized credentials without exposing secrets."""
        # 1. Bearer Token
        if self.auth_token:
            self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})
            logger.debug("Configured session with Bearer authorization.")

        # 2. Pre-configured session cookie
        if self.auth_cookie:
            self.session.headers.update({"Cookie": self.auth_cookie})
            logger.debug("Configured session with authorized cookie.")

        # 3. Form-based password login if password provided and no session cookie set
        elif self.auth_password:
            login_url = os.getenv("SENTINEL_LOGIN_URL", self.DEFAULT_LOGIN_URL)
            try:
                logger.info("Authenticating with Sentinel via authorized password credentials...")
                post_data = {"password": self.auth_password}
                if self.auth_email:
                    post_data["email"] = self.auth_email
                resp = self.session.post(
                    login_url,
                    data=post_data,
                    timeout=self.timeout,
                    allow_redirects=True,
                )
                if resp.status_code in [200, 302]:
                    logger.info("Authentication request dispatched successfully.")
                else:
                    logger.warning("Authentication response returned status %d", resp.status_code)
            except Exception as e:
                logger.error("Authentication handshake failed: %s", mask_sensitive(str(e)))

    def test_connectivity(self) -> Dict[str, Any]:
        """Perform diagnostic check of catalogue access without attempting bypass.

        Returns:
            Dictionary with diagnostic results:
            - configured_url
            - status_code
            - final_url
            - content_type
            - response_size
            - is_json
            - is_auth_required
            - result_description
        """
        result = {
            "configured_url": self.catalogue_url,
            "status_code": None,
            "final_url": None,
            "content_type": None,
            "response_size": 0,
            "is_json": False,
            "is_auth_required": False,
            "result_description": "",
            "error": None,
        }

        try:
            # We perform a GET request without following redirects first to detect 301/302 redirects cleanly
            response = requests.get(
                self.catalogue_url,
                timeout=self.timeout,
                allow_redirects=False,
            )

            result["status_code"] = response.status_code
            result["content_type"] = response.headers.get("Content-Type", "")
            result["response_size"] = len(response.content)

            # Check if redirected (301, 302, 307, 308)
            if response.status_code in [301, 302, 303, 307, 308]:
                location = response.headers.get("Location", "")
                result["final_url"] = location
                if "login" in location or "auth" in location:
                    result["is_auth_required"] = True
                    result["result_description"] = "AUTHENTICATION REQUIRED (Redirected to login)"
                else:
                    result["result_description"] = f"HTTP REDIRECT ({response.status_code}) -> {location}"
                return result

            # Follow redirects to examine final landing page if not immediately redirected
            full_response = requests.get(
                self.catalogue_url,
                timeout=self.timeout,
                allow_redirects=True,
            )
            result["status_code"] = full_response.status_code
            result["final_url"] = full_response.url
            result["content_type"] = full_response.headers.get("Content-Type", "")
            result["response_size"] = len(full_response.content)

            if full_response.status_code in [401, 403]:
                result["is_auth_required"] = True
                result["result_description"] = "AUTHENTICATION REQUIRED (HTTP 401/403 Unauthorized)"
            elif full_response.status_code == 404:
                result["result_description"] = "NOT FOUND (HTTP 404)"
            elif "application/json" in result["content_type"] or full_response.text.strip().startswith(("{", "[")):
                try:
                    full_response.json()
                    result["is_json"] = True
                    result["result_description"] = "SUCCESS (Catalogue accessible as valid JSON)"
                except ValueError:
                    result["result_description"] = "INVALID JSON CONTENT"
            elif "/auth/login" in full_response.url or "Sign in" in full_response.text:
                result["is_auth_required"] = True
                result["result_description"] = "AUTHENTICATION REQUIRED (Landed on login page)"
            else:
                result["result_description"] = f"HTTP STATUS {full_response.status_code} ({result['content_type']})"

        except requests.exceptions.SSLError as e:
            result["error"] = "TLS/SSL Certificate Verification Failed"
            result["result_description"] = "TLS ERROR: Verification failed"
        except requests.exceptions.ConnectionError as e:
            result["error"] = "Connection refused or DNS resolution failure"
            result["result_description"] = "NETWORK/DNS ERROR: Could not resolve or reach host"
        except requests.exceptions.Timeout:
            result["error"] = f"Request timed out after {self.timeout}s"
            result["result_description"] = "TIMEOUT: Sentinel endpoint did not respond"
        except Exception as e:
            result["error"] = mask_sensitive(str(e))
            result["result_description"] = f"ERROR: {mask_sensitive(str(e))}"

        return result

    def fetch_catalogue(self) -> Tuple[Union[Dict[str, Any], list], str]:
        """Fetch the catalogue JSON payload using configured credentials.

        Returns:
            Tuple of (parsed_json, raw_text_response)

        Raises:
            AuthenticationRequiredError: If unauthorized or redirected to login.
            ValueError: If response is not valid JSON.
            requests.RequestException: On network or transport failures.
        """
        self._setup_auth()

        resp = self.session.get(
            self.catalogue_url,
            timeout=self.timeout,
            allow_redirects=True,
        )

        # Check for authentication redirect or unauthorized status
        if resp.status_code in [401, 403]:
            raise AuthenticationRequiredError(
                f"Catalogue requires authorized access (HTTP {resp.status_code})."
            )

        if "/auth/login" in resp.url or "<html" in resp.text.lower() and "sign in" in resp.text.lower():
            raise AuthenticationRequiredError(
                f"Catalogue requires authorized access (Redirected to login: {resp.url})."
            )

        resp.raise_for_status()

        # Enforce that response must be genuine JSON
        try:
            data = resp.json()
            return data, resp.text
        except ValueError as err:
            raise ValueError(
                f"Sentinel response was not valid JSON. Content-Type: {resp.headers.get('Content-Type')}"
            ) from err
