"""Unit tests for Spotify, Last.fm, and ReccoBeats API clients.

All external HTTP calls are mocked to test response parsing, regex title cleanup,
2026 Spotify schema handling, and edge cases (404s, empty results, single-dict Last.fm responses).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.clients.lastfm_client import LastFMClient, LastFMTrack
from src.clients.reccobeats_client import ACOUSTIC_FEATURES, AudioFeatures, ReccoBeatsClient
from src.clients.spotify_client import SpotifyClient, SpotifyTrack

# ==============================================================================
# SpotifyClient Tests
# ==============================================================================


class TestSpotifyClient:
    @pytest.fixture
    def client(self) -> SpotifyClient:
        return SpotifyClient(
            client_id="test_client_id",
            client_secret="test_client_secret",
            redirect_uri="http://localhost:8501/callback",
        )

    def test_extract_id_track(self, client: SpotifyClient) -> None:
        url = "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT?si=12345"
        assert client.extract_id(url, kind="track") == "4cOdK2wGLETKBW3PvgPWqT"

        uri = "spotify:track:4cOdK2wGLETKBW3PvgPWqT"
        assert client.extract_id(uri, kind="track") == "4cOdK2wGLETKBW3PvgPWqT"

        raw = "4cOdK2wGLETKBW3PvgPWqT"
        assert client.extract_id(raw, kind="track") == "4cOdK2wGLETKBW3PvgPWqT"

        with pytest.raises(ValueError):
            client.extract_id("", kind="track")

        with pytest.raises(ValueError):
            client.extract_id("invalid-format!!!", kind="track")

    def test_extract_id_playlist(self, client: SpotifyClient) -> None:
        url = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=abcd"
        assert client.extract_id(url, kind="playlist") == "37i9dQZF1DXcBWIGoYBM5M"

        uri = "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M"
        assert client.extract_id(uri, kind="playlist") == "37i9dQZF1DXcBWIGoYBM5M"

    def test_client_credentials_token_caching(self, client: SpotifyClient) -> None:
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "token_abc123",
            "expires_in": 3600,
        }

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            token1 = client.get_client_token()
            assert token1 == "token_abc123"
            assert mock_post.call_count == 1

            # Second call should use cached token without extra POST
            token2 = client.get_client_token()
            assert token2 == "token_abc123"
            assert mock_post.call_count == 1

            # Simulate expired token
            client._token_expires_at = time.time() - 10
            token3 = client.get_client_token()
            assert token3 == "token_abc123"
            assert mock_post.call_count == 2

    def test_get_track_resolution(self, client: SpotifyClient) -> None:
        client._access_token = "valid_token"
        client._token_expires_at = time.time() + 3600

        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "track_123",
            "name": "Karma Police",
            "uri": "spotify:track:track_123",
            "artists": [{"name": "Radiohead"}],
            "album": {
                "name": "OK Computer",
                "images": [{"url": "https://image.url/okc.jpg"}],
            },
        }

        with patch.object(client._session, "get", return_value=mock_response) as mock_get:
            track = client.get_track("track_123")
            assert isinstance(track, SpotifyTrack)
            assert track.id == "track_123"
            assert track.name == "Karma Police"
            assert track.artist == "Radiohead"
            assert track.album == "OK Computer"
            assert track.image_url == "https://image.url/okc.jpg"
            assert mock_get.called

    def test_get_playlist_items_2026_schema(self, client: SpotifyClient) -> None:
        client._access_token = "valid_token"
        client._token_expires_at = time.time() + 3600

        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        # 2026 schema: tracks are nested under item['track']
        mock_response.json.return_value = {
            "items": [
                {
                    "track": {
                        "id": "t1",
                        "name": "Track One",
                        "uri": "spotify:track:t1",
                        "artists": [{"name": "Artist A"}],
                        "album": {"name": "Album 1", "images": []},
                    }
                },
                # Edge case: null track item (e.g., unavailable episode)
                {"track": None},
                # Edge case: track missing ID
                {"track": {"id": None, "name": "Corrupt Track"}},
                {
                    "track": {
                        "id": "t2",
                        "name": "Track Two",
                        "uri": "spotify:track:t2",
                        "artists": [{"name": "Artist B"}, {"name": "Artist C"}],
                        "album": {"name": "Album 2", "images": [{"url": "http://img.jpg"}]},
                    }
                },
            ]
        }

        with patch.object(client._session, "get", return_value=mock_response) as mock_get:
            tracks = client.get_playlist_items("37i9dQZF1DXcBWIGoYBM5M")
            assert len(tracks) == 2
            assert tracks[0].name == "Track One"
            assert tracks[0].artist == "Artist A"
            assert tracks[1].name == "Track Two"
            assert tracks[1].artist == "Artist B, Artist C"

            # Check that URL called ends in /playlists/{id}/items
            called_url = mock_get.call_args[0][0]
            assert called_url.endswith("/playlists/37i9dQZF1DXcBWIGoYBM5M/items")

    def test_search_track_enforces_limit_and_parsing(self, client: SpotifyClient) -> None:
        client._access_token = "valid_token"
        client._token_expires_at = time.time() + 3600

        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tracks": {
                "items": [
                    {
                        "id": "win_1",
                        "name": "Winning Song",
                        "uri": "spotify:track:win_1",
                        "artists": [{"name": "Winning Artist"}],
                        "album": {"name": "Victory", "images": []},
                    }
                ]
            }
        }

        with patch.object(client._session, "get", return_value=mock_response) as mock_get:
            # Pass limit > 10 to test clamping
            result = client.search_track("Winning Song", "Winning Artist", limit=25)
            assert result is not None
            assert result.id == "win_1"
            assert result.name == "Winning Song"

            # Verify limit in params <= 10
            called_params = mock_get.call_args[1]["params"]
            assert called_params["limit"] <= 10

    def test_search_track_returns_none_when_empty(self, client: SpotifyClient) -> None:
        client._access_token = "valid_token"
        client._token_expires_at = time.time() + 3600

        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"tracks": {"items": []}}

        with patch.object(client._session, "get", return_value=mock_response):
            result = client.search_track("Nonexistent Song")
            assert result is None

    def test_create_playlist_and_add_items(self, client: SpotifyClient) -> None:
        create_res = MagicMock(spec=requests.Response)
        create_res.status_code = 201
        create_res.json.return_value = {
            "id": "new_pl_id",
            "name": "BSide Recommendations",
            "external_urls": {"spotify": "https://open.spotify.com/playlist/new_pl_id"},
        }

        add_res = MagicMock(spec=requests.Response)
        add_res.status_code = 201
        add_res.json.return_value = {"snapshot_id": "snap_123"}

        def mock_post(url: str, *args, **kwargs):
            if "/me/playlists" in url:
                return create_res
            if "/playlists/new_pl_id/items" in url:
                return add_res
            return MagicMock(status_code=400)

        with patch.object(client._session, "post", side_effect=mock_post):
            res = client.create_playlist(
                user_token="user_token_xyz",
                name="BSide Recommendations",
                track_uris=["spotify:track:1", "spotify:track:2"],
            )
            assert res["id"] == "new_pl_id"


# ==============================================================================
# LastFMClient Tests
# ==============================================================================


class TestLastFMClient:
    @pytest.mark.parametrize(
        ("input_title", "expected"),
        [
            ("Starboy (feat. Daft Punk)", "Starboy"),
            ("Starboy [feat. Daft Punk]", "Starboy"),
            ("Starboy (ft. Daft Punk)", "Starboy"),
            ("Starboy ft. Daft Punk", "Starboy"),
            ("Starboy featuring Daft Punk", "Starboy"),
            ("Cupid - Twin Ver.", "Cupid"),
            ("Cupid (Twin Ver.)", "Cupid"),
            ("Let It Be - Remastered 2009", "Let It Be"),
            ("Let It Be - 2009 Remaster", "Let It Be"),
            ("Let It Be (Remastered 2009)", "Let It Be"),
            ("Let It Be (2009 Remaster)", "Let It Be"),
            ("Blinding Lights - Radio Edit", "Blinding Lights"),
            ("Blinding Lights (Radio Edit)", "Blinding Lights"),
            ("Song - Single Version", "Song"),
            ("Song - Original Mix", "Song"),
            ("Song - Live", "Song"),
            ("Song (Live at Wembley)", "Song"),
            ("Song (feat. Artist) - 2021 Remaster", "Song"),
            ("Post-Modern", "Post-Modern"),  # Hyphen preserved
            ("Spider-Man", "Spider-Man"),    # Hyphen preserved
            ("", ""),
        ],
    )
    def test_normalize_title_regex(self, input_title: str, expected: str) -> None:
        assert LastFMClient.normalize_title(input_title) == expected

    def test_get_similar_tracks_success(self) -> None:
        client = LastFMClient(api_key="test_lastfm_key")

        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "similartracks": {
                "track": [
                    {
                        "name": "Similar Track 1",
                        "artist": {"name": "Artist 1"},
                        "match": "0.925",
                    },
                    {
                        "name": "Similar Track 2",
                        "artist": {"name": "Artist 2"},
                        "match": "0.710",
                    },
                ]
            }
        }

        with patch.object(client._session, "get", return_value=mock_response) as mock_get:
            candidates = client.get_similar_tracks(
                artist="Artist Seed", track="Song (feat. X) - Remastered"
            )
            assert len(candidates) == 2
            assert candidates[0] == LastFMTrack(
                name="Similar Track 1", artist="Artist 1", match=0.925
            )
            assert candidates[1] == LastFMTrack(
                name="Similar Track 2", artist="Artist 2", match=0.710
            )

            # Check params: track normalized and autocorrect=1 sent
            params = mock_get.call_args[1]["params"]
            assert params["autocorrect"] == 1
            assert params["track"] == "Song"
            assert params["method"] == "track.getsimilar"

    def test_get_similar_tracks_single_dict_handling(self) -> None:
        client = LastFMClient(api_key="test_lastfm_key")

        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        # Edge case: Last.fm JSON returns single dict when only 1 track found
        mock_response.json.return_value = {
            "similartracks": {
                "track": {
                    "name": "Lone Similar Track",
                    "artist": {"name": "Lone Artist"},
                    "match": "0.88",
                }
            }
        }

        with patch.object(client._session, "get", return_value=mock_response):
            candidates = client.get_similar_tracks(artist="Seed", track="Song")
            assert len(candidates) == 1
            assert candidates[0].name == "Lone Similar Track"
            assert candidates[0].match == 0.88

    def test_get_similar_tracks_error_and_404_graceful(self) -> None:
        client = LastFMClient(api_key="test_lastfm_key")

        # Last.fm API error object (e.g., error 6: Track not found)
        error_res = MagicMock(spec=requests.Response)
        error_res.status_code = 200
        error_res.json.return_value = {"error": 6, "message": "Track not found"}

        with patch.object(client._session, "get", return_value=error_res):
            assert client.get_similar_tracks(artist="Unknown", track="Unknown") == []

        # HTTP 404 response
        http_404_res = MagicMock(spec=requests.Response)
        http_404_res.status_code = 404

        with patch.object(client._session, "get", return_value=http_404_res):
            assert client.get_similar_tracks(artist="Missing", track="Missing") == []


# ==============================================================================
# ReccoBeatsClient Tests
# ==============================================================================


class TestReccoBeatsClient:
    @pytest.fixture
    def client(self) -> ReccoBeatsClient:
        return ReccoBeatsClient()

    def test_search_track_id_success_and_miss(self, client: ReccoBeatsClient) -> None:
        # Success response
        success_res = MagicMock(spec=requests.Response)
        success_res.status_code = 200
        success_res.json.return_value = [{"id": "recco_uuid_123", "title": "Song"}]

        with patch.object(client._session, "get", return_value=success_res):
            track_id = client.search_track_id("Song", "Artist")
            assert track_id == "recco_uuid_123"

        # 404 miss
        miss_res = MagicMock(spec=requests.Response)
        miss_res.status_code = 404

        with patch.object(client._session, "get", return_value=miss_res):
            assert client.search_track_id("Missing", "Missing") is None

    def test_get_audio_features_8_metrics_and_vector(self, client: ReccoBeatsClient) -> None:
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "track_1",
                "danceability": 0.75,
                "energy": 0.80,
                "loudness": -5.2,
                "speechiness": 0.05,
                "acousticness": 0.12,
                "instrumentalness": 0.01,
                "valence": 0.65,
                "tempo": 120.0,
            }
        ]

        with patch.object(client._session, "get", return_value=mock_response):
            features = client.get_audio_features(["track_1"])
            assert "track_1" in features
            feat = features["track_1"]
            assert isinstance(feat, AudioFeatures)
            assert feat.danceability == 0.75
            assert feat.energy == 0.80
            assert feat.loudness == -5.2
            assert feat.speechiness == 0.05
            assert feat.acousticness == 0.12
            assert feat.instrumentalness == 0.01
            assert feat.valence == 0.65
            assert feat.tempo == 120.0

            # Vector has length 8 in exact ACOUSTIC_FEATURES order
            vec = feat.to_vector()
            assert len(vec) == 8
            assert len(ACOUSTIC_FEATURES) == 8
            assert vec[0] == 0.75  # danceability
            assert vec[7] == 120.0  # tempo

    def test_get_audio_features_batch_slicing(self, client: ReccoBeatsClient) -> None:
        # Generate 75 unique dummy IDs (should trigger 2 chunk calls: 40 + 35)
        dummy_ids = [f"track_{i}" for i in range(75)]

        def mock_chunk(url: str, params: dict[str, str], **kwargs):
            res = MagicMock(spec=requests.Response)
            res.status_code = 200
            requested_ids = params["ids"].split(",")
            res.json.return_value = [
                {
                    "id": tid,
                    "danceability": 0.5,
                    "energy": 0.5,
                    "loudness": -6.0,
                    "speechiness": 0.1,
                    "acousticness": 0.2,
                    "instrumentalness": 0.0,
                    "valence": 0.5,
                    "tempo": 100.0,
                }
                for tid in requested_ids
            ]
            return res

        with patch.object(client._session, "get", side_effect=mock_chunk) as mock_get:
            results = client.get_audio_features(dummy_ids)
            assert len(results) == 75
            assert mock_get.call_count == 2
            # First call has 40 IDs, second call has 35 IDs
            call_1_ids = mock_get.call_args_list[0][1]["params"]["ids"].split(",")
            call_2_ids = mock_get.call_args_list[1][1]["params"]["ids"].split(",")
            assert len(call_1_ids) == 40
            assert len(call_2_ids) == 35

    def test_get_audio_features_empty_and_404_graceful(self, client: ReccoBeatsClient) -> None:
        # Empty input should not make any HTTP call
        with patch.object(client._session, "get") as mock_get:
            assert client.get_audio_features([]) == {}
            assert not mock_get.called

        # 404 response handled gracefully
        mock_404 = MagicMock(spec=requests.Response)
        mock_404.status_code = 404

        with patch.object(client._session, "get", return_value=mock_404):
            assert client.get_audio_features(["missing_id"]) == {}
