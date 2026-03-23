"""OAuth 2.0 authentication for Google services."""
import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import requests
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)


class OAuth2Handler:
    """Handle OAuth 2.0 authentication flow."""

    def __init__(self):
        """Initialize OAuth handler."""
        self.client_id = config.oauth.client_id
        self.client_secret = config.oauth.client_secret
        self.redirect_uri = config.oauth.redirect_uri
        self.scopes = config.oauth.scopes
        self.token_cache_file = Path(".token_cache.json")
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None

        # Load cached token if exists
        self._load_cached_token()

    def get_authorization_url(self) -> str:
        """
        Generate authorization URL for user to visit.

        Returns:
            Authorization URL
        """
        auth_endpoint = "https://accounts.google.com/o/oauth2/v2/auth"

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.scopes),
            "access_type": "offline",
            "state": "security_token"
        }

        from urllib.parse import urlencode
        url = f"{auth_endpoint}?{urlencode(params)}"
        return url

    def exchange_code_for_token(self, authorization_code: str) -> bool:
        """
        Exchange authorization code for access token.

        Args:
            authorization_code: Code from OAuth callback

        Returns:
            True if successful
        """
        try:
            token_endpoint = "https://oauth2.googleapis.com/token"

            payload = {
                "code": authorization_code,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code"
            }

            response = requests.post(token_endpoint, data=payload)
            response.raise_for_status()
            token_data = response.json()

            self._store_token(token_data)
            logger.info("Successfully exchanged authorization code for tokens")
            return True

        except Exception as e:
            logger.error(f"Failed to exchange authorization code: {e}")
            return False

    def get_access_token(self) -> Optional[str]:
        """
        Get valid access token (refresh if needed).

        Returns:
            Access token or None
        """
        # Check if token is expired
        if self._is_token_expired():
            if not self.refresh_token:
                logger.error("No refresh token available")
                return None

            if not self._refresh_access_token():
                return None

        return self.access_token

    def _is_token_expired(self) -> bool:
        """Check if access token is expired."""
        if self.token_expiry is None:
            return True

        # Consider expired if less than 5 minutes left
        return datetime.utcnow() >= (self.token_expiry - timedelta(minutes=5))

    def _refresh_access_token(self) -> bool:
        """Refresh access token using refresh token."""
        try:
            token_endpoint = "https://oauth2.googleapis.com/token"

            payload = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "grant_type": "refresh_token"
            }

            response = requests.post(token_endpoint, data=payload)
            response.raise_for_status()
            token_data = response.json()

            self._store_token(token_data)
            logger.info("Successfully refreshed access token")
            return True

        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            return False

    def _store_token(self, token_data: Dict[str, Any]):
        """
        Store token data.

        Args:
            token_data: Token response from Google
        """
        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token", self.refresh_token)

        # Calculate expiry
        expires_in = token_data.get("expires_in", 3600)
        self.token_expiry = datetime.utcnow() + timedelta(seconds=expires_in)

        # Cache to file
        self._save_cached_token(token_data)

    def _save_cached_token(self, token_data: Dict[str, Any]):
        """Save token to cache file."""
        try:
            cache_data = {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "expiry": self.token_expiry.isoformat() if self.token_expiry else None,
            }
            with open(self.token_cache_file, "w") as f:
                json.dump(cache_data, f)
            logger.debug(f"Token cached to {self.token_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache token: {e}")

    def _load_cached_token(self):
        """Load token from cache file."""
        try:
            if not self.token_cache_file.exists():
                return

            with open(self.token_cache_file) as f:
                cache_data = json.load(f)

            self.access_token = cache_data.get("access_token")
            self.refresh_token = cache_data.get("refresh_token")

            expiry_str = cache_data.get("expiry")
            if expiry_str:
                self.token_expiry = datetime.fromisoformat(expiry_str)

            logger.info("Loaded token from cache")
        except Exception as e:
            logger.warning(f"Failed to load cached token: {e}")

    def clear_cache(self):
        """Clear cached tokens."""
        try:
            if self.token_cache_file.exists():
                self.token_cache_file.unlink()
                logger.info("Cleared token cache")
        except Exception as e:
            logger.error(f"Failed to clear token cache: {e}")
