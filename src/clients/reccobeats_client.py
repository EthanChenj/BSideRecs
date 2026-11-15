"""ReccoBeats API Client.

Handles:
- Searching candidate track IDs.
- Batch-fetching the 8 acoustic features via GET https://api.reccobeats.com/v1/audio-features?ids=...
- Slicing batch requests into chunks of <= 40 IDs.
- Graceful handling of 404 misses and missing tracks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

ACOUSTIC_FEATURES: tuple[str, ...] = (
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "valence",
    "tempo",
)


@dataclass
class AudioFeatures:
    """The 8 acoustic features enriched via ReccoBeats."""

    id: str
    danceability: float
    energy: float
    loudness: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    valence: float
    tempo: float

    def to_vector(self) -> list[float]:
        """Return the 8 features in canonical order for ML clustering and PCA."""
        return [
            self.danceability,
            self.energy,
            self.loudness,
            self.speechiness,
            self.acousticness,
            self.instrumentalness,
            self.valence,
            self.tempo,
        ]


class ReccoBeatsClient:
    """Client for ReccoBeats audio feature enrichment and track lookup."""

    BASE_URL = "https://api.reccobeats.com/v1"
    MAX_BATCH_SIZE = 40  # ReccoBeats maximum IDs per audio-features request

    def __init__(
        self,
        base_url: str = BASE_URL,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = session or requests.Session()

    def search_track_id(self, title: str, artist: str) -> str | None:
        """Search ReccoBeats for a candidate track ID.

        Handles 404 or empty results gracefully by returning None.
        """
        url = f"{self.base_url}/track"
        query = f"{title} {artist}".strip()
        params = {"query": query}

        try:
            response = self._session.get(url, params=params, timeout=10)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as err:
            logger.debug("ReccoBeats search miss for %s: %s", query, err)
            return None

        # Parse track ID from array or wrapped response
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                return first.get("id")
        elif isinstance(data, dict):
            if "id" in data:
                return data["id"]
            tracks = data.get("tracks") or data.get("items") or []
            if isinstance(tracks, list) and tracks and isinstance(tracks[0], dict):
                return tracks[0].get("id")

        return None

    def get_audio_features(self, track_ids: list[str]) -> dict[str, AudioFeatures]:
        """Batch-fetch 8 acoustic features for given track IDs.

        Slices requests into chunks of <= 40 IDs.
        Gracefully handles 404 misses and skips unresolvable tracks.
        """
        # Deduplicate while preserving order
        unique_ids = [tid for tid in dict.fromkeys(track_ids) if tid]
        if not unique_ids:
            return {}

        results: dict[str, AudioFeatures] = {}

        for i in range(0, len(unique_ids), self.MAX_BATCH_SIZE):
            chunk = unique_ids[i : i + self.MAX_BATCH_SIZE]
            chunk_results = self._fetch_features_chunk(chunk)
            results.update(chunk_results)

        return results

    def _fetch_features_chunk(self, chunk: list[str]) -> dict[str, AudioFeatures]:
        """Fetch a single chunk of <= 40 track IDs."""
        url = f"{self.base_url}/audio-features"
        params = {"ids": ",".join(chunk)}

        try:
            response = self._session.get(url, params=params, timeout=10)
            if response.status_code == 404:
                logger.debug("ReccoBeats audio-features returned 404 for chunk: %s", chunk)
                return {}
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as err:
            logger.warning("ReccoBeats audio-features chunk error: %s", err)
            return {}

        # The response can be a list of feature dicts or wrapped in an object
        raw_items: list[dict[str, Any]] = []
        if isinstance(data, list):
            raw_items = [d for d in data if isinstance(d, dict)]
        elif isinstance(data, dict):
            if "audio_features" in data and isinstance(data["audio_features"], list):
                raw_items = [d for d in data["audio_features"] if isinstance(d, dict)]
            elif "items" in data and isinstance(data["items"], list):
                raw_items = [d for d in data["items"] if isinstance(d, dict)]
            elif "id" in data:
                raw_items = [data]

        chunk_features: dict[str, AudioFeatures] = {}
        for item in raw_items:
            track_id = item.get("id") or item.get("track_id")
            if not track_id:
                continue

            try:
                features = AudioFeatures(
                    id=str(track_id),
                    danceability=float(item.get("danceability", 0.0)),
                    energy=float(item.get("energy", 0.0)),
                    loudness=float(item.get("loudness", 0.0)),
                    speechiness=float(item.get("speechiness", 0.0)),
                    acousticness=float(item.get("acousticness", 0.0)),
                    instrumentalness=float(item.get("instrumentalness", 0.0)),
                    valence=float(item.get("valence", 0.0)),
                    tempo=float(item.get("tempo", 0.0)),
                )
                chunk_features[str(track_id)] = features
            except (ValueError, TypeError) as parse_err:
                logger.debug("Failed parsing features for track %s: %s", track_id, parse_err)

        return chunk_features
