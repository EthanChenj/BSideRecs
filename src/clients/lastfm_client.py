"""Last.fm API Client.

Handles:
- Regex normalization of song titles:
  stripping (feat. ...), [feat. ...], - Remastered, - Twin Ver., - Radio Edit, etc.
- Candidate discovery via track.getSimilar with autocorrect=1.
- Safe response parsing (handling single-dict vs list edge cases and API errors).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class LastFMTrack:
    """Candidate track discovered via Last.fm."""

    name: str
    artist: str
    match: float


class LastFMClient:
    """Client for Last.fm API candidate discovery."""

    BASE_URL = "https://ws.audioscrobbler.com/2.0/"

    # Pre-compiled regex patterns for title normalization
    # 1. Feature tags like (feat. X), [ft. X], {featuring X}
    FEAT_PAREN_RE = re.compile(
        r"[\(\[\{]\s*(?:feat\.?|ft\.?|featuring)\s+[^)\]\}]+[\)\]\}]",
        re.IGNORECASE,
    )
    # 2. Trailing feature tags like ' feat. X' or ' ft. X'
    FEAT_TRAILING_RE = re.compile(
        r"\s+(?:feat\.?|ft\.?|featuring)\s+.+$",
        re.IGNORECASE,
    )
    # 3. Remaster patterns like '- 2011 Remaster', '- Remastered 2020', '(Remastered)'
    REMASTER_DASH_RE = re.compile(
        r"\s*-\s*(?:\d{4}\s*)?remaster(?:ed)?(?:\s*\d{4})?.*$",
        re.IGNORECASE,
    )
    REMASTER_PAREN_RE = re.compile(
        r"\s*[\(\[](?:(?:\d{4}\s*)?remaster(?:ed)?(?:\s*\d{4})?)[^\]\)]*[\)\]]",
        re.IGNORECASE,
    )
    # 4. Version tags like '- Twin Ver.', '(Twin Ver.)'
    TWIN_VER_DASH_RE = re.compile(
        r"\s*-\s*twin\s+ver(?:\.|sion)?.*$",
        re.IGNORECASE,
    )
    TWIN_VER_PAREN_RE = re.compile(
        r"\s*[\(\[]twin\s+ver(?:\.|sion)?[^\]\)]*[\)\]]",
        re.IGNORECASE,
    )
    # 5. Common suffixes: Radio Edit, Single Version, Original Mix, Bonus Track, Live
    SUFFIX_DASH_RE = re.compile(
        r"\s*-\s*(?:radio\s+edit|single\s+version|original\s+mix|bonus\s+track|live(?:\s+at.*)?).*$",
        re.IGNORECASE,
    )
    SUFFIX_PAREN_RE = re.compile(
        r"\s*[\(\[](?:radio\s+edit|single\s+version|original\s+mix|bonus\s+track|live(?:\s+at.*)?)[^\]\)]*[\)\]]",
        re.IGNORECASE,
    )
    # 6. Dangling punctuation & trailing dash cleanup
    TRAILING_CLEAN_RE = re.compile(r"[\s\-_]+$")

    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("LASTFM_API_KEY", "")
        self._session = session or requests.Session()

    @classmethod
    def normalize_title(cls, title: str) -> str:
        """Strip featured artists, remaster suffixes, and version tags from title."""
        if not title:
            return ""

        cleaned = title
        # Strip feature tags
        cleaned = cls.FEAT_PAREN_RE.sub("", cleaned)
        cleaned = cls.FEAT_TRAILING_RE.sub("", cleaned)

        # Strip remaster suffixes
        cleaned = cls.REMASTER_DASH_RE.sub("", cleaned)
        cleaned = cls.REMASTER_PAREN_RE.sub("", cleaned)

        # Strip Twin Ver tags
        cleaned = cls.TWIN_VER_DASH_RE.sub("", cleaned)
        cleaned = cls.TWIN_VER_PAREN_RE.sub("", cleaned)

        # Strip other standard suffixes
        cleaned = cls.SUFFIX_DASH_RE.sub("", cleaned)
        cleaned = cls.SUFFIX_PAREN_RE.sub("", cleaned)

        # Strip trailing hyphens and whitespace
        cleaned = cls.TRAILING_CLEAN_RE.sub("", cleaned).strip()

        # Fallback to original if stripping emptied the string
        return cleaned or title.strip()

    def get_similar_tracks(
        self, artist: str, track: str, limit: int = 50
    ) -> list[LastFMTrack]:
        """Fetch similar candidate tracks using track.getSimilar with autocorrect=1.

        Returns an empty list on missing tracks or API errors without crashing.
        """
        if not self.api_key:
            raise ValueError(
                "LASTFM_API_KEY must be set in environment or passed to constructor."
            )

        cleaned_title = self.normalize_title(track)
        params: dict[str, Any] = {
            "method": "track.getsimilar",
            "artist": artist,
            "track": cleaned_title,
            "api_key": self.api_key,
            "autocorrect": 1,
            "format": "json",
            "limit": max(1, limit),
        }

        try:
            response = self._session.get(self.BASE_URL, params=params, timeout=10)
            if response.status_code != 200:
                logger.warning(
                    "Last.fm returned HTTP %d for %s - %s",
                    response.status_code,
                    artist,
                    cleaned_title,
                )
                return []

            data = response.json()
        except (requests.RequestException, ValueError) as err:
            logger.warning("Error calling Last.fm: %s", err)
            return []

        # Handle Last.fm error responses (e.g., {"error": 6, "message": "Track not found"})
        if "error" in data:
            logger.info(
                "Last.fm error response: code=%s, msg=%s",
                data.get("error"),
                data.get("message"),
            )
            return []

        similar_tracks_data = data.get("similartracks", {})
        raw_tracks = similar_tracks_data.get("track", [])

        # Last.fm returns a single dict instead of a list when only one item matches
        if isinstance(raw_tracks, dict):
            raw_tracks = [raw_tracks]
        elif not isinstance(raw_tracks, list):
            return []

        results: list[LastFMTrack] = []
        for item in raw_tracks:
            if not isinstance(item, dict):
                continue

            name = item.get("name", "").strip()
            artist_data = item.get("artist", {})
            if isinstance(artist_data, dict):
                artist_name = artist_data.get("name", "").strip()
            elif isinstance(artist_data, str):
                artist_name = artist_data.strip()
            else:
                artist_name = ""

            if not name or not artist_name:
                continue

            # Safely parse match score (float between 0.0 and 1.0)
            try:
                match_score = float(item.get("match", 0.0))
            except (ValueError, TypeError):
                match_score = 0.0

            results.append(
                LastFMTrack(name=name, artist=artist_name, match=match_score)
            )

        return results
