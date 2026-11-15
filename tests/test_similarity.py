"""Unit tests for ML similarity, clustering, and 2D PCA projection engine."""

from __future__ import annotations

import math

import pytest

from src.clients.reccobeats_client import AudioFeatures
from src.ml.similarity import (
    Point2D,
    ScoredRecommendation,
    SimilarityResult,
    TrackItem,
    calculate_dynamic_k,
    rank_and_project,
)


def make_features(
    track_id: str,
    danceability: float = 0.5,
    energy: float = 0.5,
    loudness: float = -6.0,
    speechiness: float = 0.05,
    acousticness: float = 0.2,
    instrumentalness: float = 0.1,
    valence: float = 0.5,
    tempo: float = 120.0,
) -> AudioFeatures:
    """Helper to create dummy AudioFeatures with defaults."""
    return AudioFeatures(
        id=track_id,
        danceability=danceability,
        energy=energy,
        loudness=loudness,
        speechiness=speechiness,
        acousticness=acousticness,
        instrumentalness=instrumentalness,
        valence=valence,
        tempo=tempo,
    )


# ==============================================================================
# Dynamic K Calculation Tests
# ==============================================================================


class TestDynamicK:
    @pytest.mark.parametrize(
        ("n_seeds", "expected_k"),
        [
            (1, 1),
            (2, 2),
            (3, 2),
            (5, 2),
            (10, 3),    # ceil(ln(10)) = 3
            (25, 4),    # ceil(ln(25)) = 4
            (50, 4),    # ceil(ln(50)) = 4
            (100, 5),   # ceil(ln(100)) = 5
            (200, 6),   # ceil(ln(200)) = 6
            (500, 6),   # clamped at 6
            (1000, 6),  # clamped at 6
        ],
    )
    def test_dynamic_k_formula(self, n_seeds: int, expected_k: int) -> None:
        assert calculate_dynamic_k(n_seeds) == expected_k


# ==============================================================================
# Single-Track Mode Tests
# ==============================================================================


class TestSingleTrackMode:
    def test_single_track_clustering_and_ranking(self) -> None:
        seed = TrackItem(
            id="seed_1",
            name="Radiohead - Karma Police",
            artist="Radiohead",
            features=make_features("seed_1", danceability=0.3, energy=0.4, tempo=90.0),
        )

        candidates = [
            TrackItem(
                id="cand_1",
                name="Thom Yorke - Harrowdown Hill",
                artist="Thom Yorke",
                features=make_features("cand_1", danceability=0.32, energy=0.42, tempo=92.0),
                lastfm_similarity=0.9,
            ),
            TrackItem(
                id="cand_2",
                name="Skrillex - Bangarang",
                artist="Skrillex",
                features=make_features("cand_2", danceability=0.9, energy=0.95, tempo=140.0),
                lastfm_similarity=0.1,
            ),
            TrackItem(
                id="cand_3",
                name="Portishead - Glory Box",
                artist="Portishead",
                features=make_features("cand_3", danceability=0.4, energy=0.45, tempo=95.0),
                lastfm_similarity=0.75,
            ),
        ]

        result = rank_and_project(
            seed_tracks=[seed],
            candidate_tracks=candidates,
            top_m=3,
            alpha=0.5,
        )

        assert isinstance(result, SimilarityResult)
        assert result.k == 1
        assert result.seed_clusters == [0]
        assert len(result.recommendations) == 3

        # All recommendations should point to cluster 0 (since k=1)
        for rec in result.recommendations:
            assert isinstance(rec, ScoredRecommendation)
            assert rec.nearest_cluster == 0

        # Thom Yorke should rank 1st (high acoustic similarity + high Last.fm score)
        assert result.recommendations[0].id == "cand_1"
        # Skrillex should rank last (distant acoustic + low Last.fm score)
        assert result.recommendations[-1].id == "cand_2"

        # Verify 2D points output
        seed_pts = [p for p in result.points_2d if p.point_type == "seed"]
        centroid_pts = [p for p in result.points_2d if p.point_type == "centroid"]
        rec_pts = [p for p in result.points_2d if p.point_type == "recommendation"]

        assert len(seed_pts) == 1
        assert len(centroid_pts) == 1
        assert len(rec_pts) == 3

        for pt in result.points_2d:
            assert isinstance(pt, Point2D)
            assert not math.isnan(pt.x)
            assert not math.isnan(pt.y)


# ==============================================================================
# Playlist Mode & Diversity Tests
# ==============================================================================


class TestPlaylistMode:
    def test_playlist_kmeans_clustering_and_diversity(self) -> None:
        # Create 12 seeds representing three distinct musical profiles:
        # Profile A: Low energy, acoustic (4 tracks)
        # Profile B: High energy, electronic (4 tracks)
        # Profile C: Rock / driving beat (4 tracks)
        seeds: list[TrackItem] = []
        for i in range(4):
            seeds.append(
                TrackItem(
                    id=f"seed_acoustic_{i}",
                    name=f"Acoustic Song {i}",
                    artist="Folk Artist",
                    features=make_features(
                        f"seed_acoustic_{i}",
                        energy=0.2 + 0.02 * i,
                        acousticness=0.85 + 0.02 * i,
                        danceability=0.3 + 0.01 * i,
                        tempo=80.0 + i,
                    ),
                )
            )
        for i in range(4):
            seeds.append(
                TrackItem(
                    id=f"seed_electronic_{i}",
                    name=f"Electronic Song {i}",
                    artist="EDM Artist",
                    features=make_features(
                        f"seed_electronic_{i}",
                        energy=0.9 + 0.01 * i,
                        acousticness=0.05 + 0.01 * i,
                        danceability=0.85 + 0.01 * i,
                        tempo=130.0 + i,
                    ),
                )
            )
        for i in range(4):
            seeds.append(
                TrackItem(
                    id=f"seed_rock_{i}",
                    name=f"Rock Song {i}",
                    artist="Rock Artist",
                    features=make_features(
                        f"seed_rock_{i}",
                        energy=0.7 + 0.02 * i,
                        acousticness=0.3 + 0.02 * i,
                        danceability=0.5 + 0.01 * i,
                        tempo=110.0 + i,
                    ),
                )
            )

        # Create candidates in acoustic, electronic, and rock spaces
        candidates: list[TrackItem] = []
        for i in range(3):
            candidates.append(
                TrackItem(
                    id=f"cand_acoustic_{i}",
                    name=f"Cand Acoustic {i}",
                    artist="Folk Artist",
                    features=make_features(
                        f"cand_acoustic_{i}",
                        energy=0.22 + 0.01 * i,
                        acousticness=0.88,
                        danceability=0.32,
                        tempo=82.0,
                    ),
                    lastfm_similarity=0.8,
                )
            )
        for i in range(3):
            candidates.append(
                TrackItem(
                    id=f"cand_electronic_{i}",
                    name=f"Cand Electronic {i}",
                    artist="EDM Artist",
                    features=make_features(
                        f"cand_electronic_{i}",
                        energy=0.92,
                        acousticness=0.08,
                        danceability=0.82,
                        tempo=128.0,
                    ),
                    lastfm_similarity=0.8,
                )
            )
        for i in range(3):
            candidates.append(
                TrackItem(
                    id=f"cand_rock_{i}",
                    name=f"Cand Rock {i}",
                    artist="Rock Artist",
                    features=make_features(
                        f"cand_rock_{i}",
                        energy=0.72,
                        acousticness=0.28,
                        danceability=0.52,
                        tempo=112.0,
                    ),
                    lastfm_similarity=0.8,
                )
            )

        # 12 tracks -> ceil(ln(12)) = 3 clusters
        result = rank_and_project(
            seed_tracks=seeds,
            candidate_tracks=candidates,
            top_m=6,
            alpha=0.5,
        )

        assert result.k == 3
        assert len(result.seed_clusters) == 12
        assert len(result.recommendations) == 6

        # Check diversity: both clusters should have winning recommendations
        cluster_set = {rec.nearest_cluster for rec in result.recommendations}
        assert len(cluster_set) >= 2


# ==============================================================================
# Ranking Order and Weight Tests
# ==============================================================================


class TestRankingOrderAndWeights:
    def test_alpha_weight_extremes(self) -> None:
        seed = TrackItem(
            id="seed",
            name="Seed",
            artist="Artist",
            features=make_features("seed", danceability=0.5, energy=0.5),
        )

        cand_audio_match = TrackItem(
            id="cand_audio",
            name="Acoustic Match",
            artist="Artist A",
            features=make_features("cand_audio", danceability=0.5, energy=0.5),  # exact match
            lastfm_similarity=0.1,  # low Last.fm score
        )
        cand_lastfm_match = TrackItem(
            id="cand_lastfm",
            name="Collaborative Match",
            artist="Artist B",
            features=make_features("cand_lastfm", danceability=0.99, energy=0.99),  # distant audio
            lastfm_similarity=0.95,  # high Last.fm score
        )

        candidates = [cand_audio_match, cand_lastfm_match]

        # Case 1: Pure audio weighting (alpha = 1.0)
        res_audio = rank_and_project(
            seed_tracks=[seed],
            candidate_tracks=candidates,
            top_m=2,
            alpha=1.0,
        )
        assert res_audio.recommendations[0].id == "cand_audio"
        assert res_audio.recommendations[1].id == "cand_lastfm"

        # Case 2: Pure collaborative weighting (alpha = 0.0)
        res_lastfm = rank_and_project(
            seed_tracks=[seed],
            candidate_tracks=candidates,
            top_m=2,
            alpha=0.0,
        )
        assert res_lastfm.recommendations[0].id == "cand_lastfm"
        assert res_lastfm.recommendations[1].id == "cand_audio"


# ==============================================================================
# PCA Coordinate Output Tests
# ==============================================================================


class TestPCAProjection:
    def test_pca_shapes_and_types(self) -> None:
        seeds = [
            TrackItem(
                id=f"seed_{i}",
                name=f"Seed {i}",
                artist="Artist",
                features=make_features(f"seed_{i}", danceability=0.1 * i, energy=0.1 * i),
            )
            for i in range(4)
        ]
        candidates = [
            TrackItem(
                id=f"cand_{i}",
                name=f"Cand {i}",
                artist="Artist",
                features=make_features(f"cand_{i}", danceability=0.2 * i, energy=0.2 * i),
                lastfm_similarity=0.5,
            )
            for i in range(5)
        ]

        result = rank_and_project(seed_tracks=seeds, candidate_tracks=candidates, top_m=3)

        # Expected total points: 4 seeds + k centroids (2) + 3 recommendations = 9
        expected_total = len(seeds) + result.k + len(result.recommendations)
        assert len(result.points_2d) == expected_total

        # Verify PCA coordinates are finite floats
        for pt in result.points_2d:
            assert isinstance(pt.x, float)
            assert isinstance(pt.y, float)
            assert not math.isinf(pt.x)
            assert not math.isinf(pt.y)


# ==============================================================================
# Edge Cases Tests
# ==============================================================================


class TestEdgeCases:
    def test_seed_deduplication(self) -> None:
        seed = TrackItem(
            id="duplicate_id",
            name="Exact Title",
            artist="Exact Artist",
            features=make_features("duplicate_id", danceability=0.2, energy=0.3),
        )
        cand_duplicate = TrackItem(
            id="duplicate_id",
            name="Exact Title",
            artist="Exact Artist",
            features=make_features("duplicate_id", danceability=0.2, energy=0.3),
            lastfm_similarity=1.0,
        )
        cand_fresh = TrackItem(
            id="fresh_id",
            name="New Song",
            artist="New Artist",
            features=make_features("fresh_id", danceability=0.7, energy=0.8),
            lastfm_similarity=0.8,
        )

        result = rank_and_project(
            seed_tracks=[seed],
            candidate_tracks=[cand_duplicate, cand_fresh],
            top_m=5,
        )

        # The duplicate candidate should be filtered out
        assert len(result.recommendations) == 1
        assert result.recommendations[0].id == "fresh_id"

    def test_candidates_fewer_than_top_m(self) -> None:
        seed = TrackItem(
            id="s1", name="S1", artist="A1", features=make_features("s1", danceability=0.1, energy=0.2)
        )
        cand = TrackItem(
            id="c1",
            name="C1",
            artist="A2",
            features=make_features("c1", danceability=0.8, energy=0.9),
            lastfm_similarity=0.5,
        )

        result = rank_and_project(seed_tracks=[seed], candidate_tracks=[cand], top_m=10)
        assert len(result.recommendations) == 1

    def test_empty_inputs_raise_value_error(self) -> None:
        valid_track = TrackItem(
            id="t1", name="T1", artist="A1", features=make_features("t1")
        )

        with pytest.raises(ValueError):
            rank_and_project(seed_tracks=[], candidate_tracks=[valid_track])

        with pytest.raises(ValueError):
            rank_and_project(seed_tracks=[valid_track], candidate_tracks=[])
