"""Spotify API Client (2026 schema compatible).

Handles:
- OAuth 2.0 Client Credentials token management (with caching & auto-refresh).
- User OAuth token usage for playlist creation and track additions.
- Reading playlist items via GET /v1/playlists/{id}/items (nested under `item['track']`).
- Resolving single track URLs/IDs via GET /v1/tracks/{id}.
- Searching final winning tracks via GET /v1/search (enforcing limit <= 10).
- Creating user playlists via POST /v1/me/playlists and adding items.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SpotifyTrack:
    """Normalized Spotify track metadata."""

    id: str
    name: str
    artist: str
    uri: str
    album: str = ""
    image_url: str = ""


class SpotifyClient:
    """Clean Spotify client implementing 2026 API rules."""

    TOKEN_URL = "https://accounts.spotify.com/api/token"
    BASE_URL = "https://api.spotify.com/v1"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_uri: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET", "")
        self.redirect_uri = redirect_uri or os.getenv("SPOTIFY_REDIRECT_URI", "")

        self._session = session or requests.Session()
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0

    @staticmethod
    def extract_id(url_or_id: str, kind: str = "track") -> str:
        """Extract a clean 22-character Spotify ID from a URL, URI, or bare ID."""
        cleaned = url_or_id.strip()
        if not cleaned:
            raise ValueError(f"Empty {kind} identifier provided.")

        # Match spotify:<kind>:<id>
        uri_pattern = rf"^spotify:{kind}:([a-zA-Z0-9_-]+)$"
        uri_match = re.match(uri_pattern, cleaned)
        if uri_match:
            return uri_match.group(1)

        # Match https://open.spotify.com/<kind>/<id>?...
        if "open.spotify.com" in cleaned or cleaned.startswith("http"):
            parsed = urlparse(cleaned)
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] == kind:
                return parts[1]
            if len(parts) == 1:
                return parts[0]

        # Raw ID (alphanumeric, typically 22 characters, or test identifiers)
        raw_pattern = r"^[a-zA-Z0-9_-]{6,64}$"
        if re.match(raw_pattern, cleaned):
            return cleaned

        raise ValueError(f"Could not parse valid Spotify {kind} ID from: '{url_or_id}'")

    def get_client_token(self) -> str:
        """Obtain or return cached Client Credentials access token."""
        now = time.time()
        if self._access_token and now < self._token_expires_at:
            return self._access_token

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be configured in environment or passed to constructor."
            )

        payload = {"grant_type": "client_credentials"}
        response = self._session.post(
            self.TOKEN_URL,
            data=payload,
            auth=(self.client_id, self.client_secret),
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        self._access_token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        # Refresh 60 seconds early to avoid race conditions
        self._token_expires_at = now + max(0, expires_in - 60)
        return self._access_token

    def _auth_headers(self, user_token: str | None = None) -> dict[str, str]:
        token = user_token or self.get_client_token()
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def get_track(self, track_id_or_url: str) -> SpotifyTrack:
        """Resolve a single track by URL or ID via GET /v1/tracks/{id}."""
        track_id = self.extract_id(track_id_or_url, kind="track")
        url = f"{self.BASE_URL}/tracks/{track_id}"
        response = self._session.get(url, headers=self._auth_headers(), timeout=10)
        response.raise_for_status()
        data = response.json()
        return self._parse_track(data)

    def get_playlist_items(
        self, playlist_id_or_url: str, limit: int = 100
    ) -> list[SpotifyTrack]:
        """Fetch playlist tracks via GET /v1/playlists/{id}/items (2026 schema).

        In Spotify 2026 schema, track objects are nested under item['track'].
        Null or unplayable tracks are cleanly omitted.
        """
        playlist_id = self.extract_id(playlist_id_or_url, kind="playlist")
        limit = max(1, min(limit, 100))
        url = f"{self.BASE_URL}/playlists/{playlist_id}/items"
        params = {"limit": limit}

        response = self._session.get(
            url, headers=self._auth_headers(), params=params, timeout=10
        )
        response.raise_for_status()
        data = response.json()

        tracks: list[SpotifyTrack] = []
        for item in data.get("items", []):
            if not isinstance(item, dict):
                continue
            raw_track = item.get("track")
            if not raw_track or not isinstance(raw_track, dict):
                continue
            # Filter out local or invalid tracks without IDs
            if not raw_track.get("id"):
                continue
            tracks.append(self._parse_track(raw_track))

        return tracks

    def search_track(
        self, title: str, artist: str | None = None, limit: int = 5
    ) -> SpotifyTrack | None:
        """Search Spotify for winning track. Enforces limit <= 10 per 2026 rules."""
        limit = max(1, min(limit, 10))
        query = f"track:{title}"
        if artist:
            query += f" artist:{artist}"

        url = f"{self.BASE_URL}/search"
        params = {"q": query, "type": "track", "limit": limit}

        response = self._session.get(
            url, headers=self._auth_headers(), params=params, timeout=10
        )
        response.raise_for_status()
        data = response.json()

        items = data.get("tracks", {}).get("items", [])
        if not items:
            # Fallback to loose text search if field-specific search missed
            loose_q = f"{title} {artist}" if artist else title
            fallback_res = self._session.get(
                url,
                headers=self._auth_headers(),
                params={"q": loose_q, "type": "track", "limit": limit},
                timeout=10,
            )
            fallback_res.raise_for_status()
            items = fallback_res.json().get("tracks", {}).get("items", [])

        if items and items[0].get("id"):
            return self._parse_track(items[0])
        return None

    def create_playlist(
        self,
        user_token: str,
        name: str,
        description: str = "Created by BSideRecs",
        track_uris: list[str] | None = None,
        public: bool = False,
    ) -> dict[str, Any]:
        """Create a new playlist for the current user and populate tracks."""
        create_url = f"{self.BASE_URL}/me/playlists"
        payload = {"name": name, "description": description, "public": public}

        headers = self._auth_headers(user_token=user_token)
        response = self._session.post(
            create_url, json=payload, headers=headers, timeout=10
        )
        response.raise_for_status()
        playlist_data = response.json()
        playlist_id = playlist_data["id"]

        if track_uris:
            add_url = f"{self.BASE_URL}/playlists/{playlist_id}/items"
            # Add in chunks of 100
            for i in range(0, len(track_uris), 100):
                chunk = track_uris[i : i + 100]
                add_res = self._session.post(
                    add_url, json={"uris": chunk}, headers=headers, timeout=10
                )
                add_res.raise_for_status()

        return playlist_data

    @staticmethod
    def _parse_track(track_dict: dict[str, Any]) -> SpotifyTrack:
        """Parse Spotify raw track dict into SpotifyTrack dataclass."""
        track_id = track_dict.get("id", "")
        name = track_dict.get("name", "Unknown Track")
        uri = track_dict.get("uri", f"spotify:track:{track_id}")

        artists = track_dict.get("artists", [])
        artist_name = (
            ", ".join(a.get("name", "") for a in artists if a.get("name"))
            if artists
            else "Unknown Artist"
        )

        album = track_dict.get("album", {})
        album_name = album.get("name", "")
        images = album.get("images", [])
        image_url = images[0].get("url", "") if images else ""

        return SpotifyTrack(
            id=track_id,
            name=name,
            artist=artist_name,
            uri=uri,
            album=album_name,
            image_url=image_url,
        )
