# AGENTS.md — BSideRecs

## Project Overview
BSideRecs is a focused music discovery web app. A user enters a single song or a Spotify playlist URL, and the app returns similar tracks alongside a clean interactive visualization showing how the input tracks and recommended tracks relate in audio feature space.

## Core User Flow
1. **Input:** User provides either a single track (`Artist - Song` or Spotify track URL) or a single Spotify playlist URL.
2. **Candidate Discovery & Enrichment:**
   - Pull seed tracks from Spotify (for playlists) or resolve the single seed track.
   - Fetch similar candidate tracks via Last.fm (`track.getSimilar`).
   - Enrich seed and candidate tracks with 8 acoustic features via ReccoBeats (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `valence`, `tempo`).
3. **Similarity & Clustering:**
   - Scale the 8 features with `StandardScaler`, group seed tracks via K-Means (or use the single track's feature vector as the target), and rank candidate tracks by Euclidean proximity combined with Last.fm similarity.
   - Project seed tracks, cluster centroids, and recommended tracks into 2D coordinates via PCA.
4. **Output & Web Visualization:**
   - Render a clean, dark-mode web app displaying the recommended tracklist (with optional one-click export to a new Spotify playlist) and an interactive 2D Plotly visual (styled as a bathymetric contour map or constellation chart) illustrating cluster regions, centroids, and recommended neighbors.

## Directory Layout
- `src/clients/`: API wrappers (`spotify_client.py`, `lastfm_client.py`, `reccobeats_client.py`)
- `src/ml/`: Clustering, similarity ranking, and 2D PCA projection (`similarity.py`)
- `tests/`: Pytest unit tests with mocked HTTP responses
- `app.py`: Streamlit web application and Plotly visualization

## Engineering & API Rules
- **Keep It Simple:** Prioritize clean, readable, well-typed Python 3.11+ functions over heavy abstractions or unnecessary multi-agent frameworks.
- **Secrets:** Load `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI`, and `LASTFM_API_KEY` from `.env` via `python-dotenv`. Never hardcode credentials.
- **Spotify Endpoints (2026 Schema):**
  - Use `GET /v1/playlists/{id}/items` (note: playlist objects nest track data under `items` rather than `tracks`) and `POST /v1/me/playlists`.
  - Keep `GET /v1/search` `limit <= 10` (Spotify's max search limit) and only resolve final winning tracks to Spotify URIs.
  - Never call deprecated Spotify `/audio-features` or `/recommendations` endpoints.
- **Last.fm Matching:** Strip featured-artist and remaster suffixes (e.g., `(feat. ...)`, `- Remastered`, `- Twin Ver.`) via regex and pass `autocorrect=1` on `track.getSimilar`.
- **ReccoBeats Lookup:** Search candidate track IDs via ReccoBeats track search before batch-fetching `GET https://api.reccobeats.com/v1/audio-features?ids=...`, handling `404` misses gracefully.
- **Verification:** Run `pytest` and `ruff check .` after every multi-file change and fix any errors before finishing.