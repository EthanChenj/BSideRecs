"""Music similarity, clustering, and 2D PCA projection engine.

Handles:
- Standard scaling of the 8 acoustic features across seeds and candidates.
- Dynamic K-Means clustering for playlists (k = max(2, min(6, ceil(ln(N))))).
- Single-song mode: uses the single seed track vector directly as the target centroid (k=1).
- Composite ranking blending normalized Euclidean proximity and Last.fm similarity.
- Diversity selection across clusters for balanced recommendations.
- PCA projection to 2D coordinates (pc1, pc2) for interactive bathymetric/constellation visuals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.clients.reccobeats_client import AudioFeatures


@dataclass
class TrackItem:
    """Represents a seed or candidate track with its acoustic features."""

    id: str
    name: str
    artist: str
    features: AudioFeatures
    lastfm_similarity: float = 0.0


@dataclass
class ScoredRecommendation:
    """Winning candidate track with similarity breakdown."""

    id: str
    name: str
    artist: str
    features: AudioFeatures
    audio_similarity: float
    lastfm_similarity: float
    composite_score: float
    nearest_cluster: int


@dataclass
class Point2D:
    """Projected 2D coordinate for interactive Plotly visual."""

    id: str
    name: str
    artist: str
    x: float
    y: float
    point_type: Literal["seed", "centroid", "recommendation"]
    cluster_id: int


@dataclass
class SimilarityResult:
    """Complete output containing ranked recommendations, 2D coordinates, and cluster count."""

    recommendations: list[ScoredRecommendation]
    points_2d: list[Point2D]
    k: int
    seed_clusters: list[int]


def calculate_dynamic_k(n_seeds: int) -> int:
    """Compute dynamic cluster count k for playlist clustering.

    Formula: k = max(2, min(6, ceil(ln(N)))), clamped so k <= N.
    For single track (N=1), returns 1.
    """
    if n_seeds <= 1:
        return 1
    raw_k = math.ceil(math.log(n_seeds))
    k = max(2, min(6, raw_k))
    return min(k, n_seeds)


def rank_and_project(
    seed_tracks: list[TrackItem],
    candidate_tracks: list[TrackItem],
    top_m: int = 10,
    alpha: float = 0.5,
    random_state: int = 42,
) -> SimilarityResult:
    """Cluster seed tracks, rank candidate tracks, and project into 2D PCA space.

    Args:
        seed_tracks: Input track(s) with AudioFeatures.
        candidate_tracks: Discovered candidate tracks with AudioFeatures and Last.fm scores.
        top_m: Number of winning recommendations to return.
        alpha: Weight for acoustic similarity vs Last.fm match (alpha in [0, 1]).
        random_state: Seed for reproducibility in K-Means and PCA.

    Returns:
        SimilarityResult with recommendations, 2D coordinates, k, and seed cluster assignments.
    """
    if not seed_tracks:
        raise ValueError("seed_tracks cannot be empty.")
    if not candidate_tracks:
        raise ValueError("candidate_tracks cannot be empty.")

    # 1. Deduplicate candidates against seeds (by id or exact name+artist)
    seed_ids = {s.id for s in seed_tracks}
    seed_signatures = {(s.name.strip().lower(), s.artist.strip().lower()) for s in seed_tracks}

    filtered_candidates: list[TrackItem] = []
    seen_cand_ids: set[str] = set()
    for c in candidate_tracks:
        if c.id in seed_ids or c.id in seen_cand_ids:
            continue
        sig = (c.name.strip().lower(), c.artist.strip().lower())
        if sig in seed_signatures:
            continue
        seen_cand_ids.add(c.id)
        filtered_candidates.append(c)

    # If all candidates filtered, fallback to original candidates to prevent empty result
    active_candidates = filtered_candidates or candidate_tracks

    # 2. Extract 8-feature vectors and scale via StandardScaler
    x_seeds = np.array([s.features.to_vector() for s in seed_tracks], dtype=np.float64)
    x_cands = np.array([c.features.to_vector() for c in active_candidates], dtype=np.float64)

    scaler = StandardScaler()
    # Fit scaler on combined feature pool to prevent zero-variance when N=1
    combined_features = np.vstack([x_seeds, x_cands])
    scaler.fit(combined_features)

    scaled_seeds = scaler.transform(x_seeds)
    scaled_cands = scaler.transform(x_cands)

    # 3. Determine cluster centroids (Single-song vs Playlist K-Means)
    n_seeds = len(seed_tracks)
    if n_seeds == 1:
        # Single track mode: target centroid is directly the scaled seed vector
        k = 1
        centroids = scaled_seeds.copy()  # shape (1, 8)
        seed_clusters = [0]
    else:
        # Playlist mode: dynamic K-Means
        k = calculate_dynamic_k(n_seeds)
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        seed_clusters = kmeans.fit_predict(scaled_seeds).tolist()
        centroids = kmeans.cluster_centers_  # shape (k, 8)

    # 4. Compute Euclidean distances from candidates to nearest cluster centroid
    # Shape: (n_candidates, k)
    diff = scaled_cands[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)  # Euclidean norm along 8D axis

    nearest_clusters = np.argmin(distances, axis=1)  # shape (n_candidates,)
    min_distances = np.min(distances, axis=1)        # shape (n_candidates,)

    # Normalize Euclidean distances to [0, 1] proximity
    d_min = float(np.min(min_distances))
    d_max = float(np.max(min_distances))
    range_d = d_max - d_min

    if range_d > 1e-9:
        audio_similarities = 1.0 - ((min_distances - d_min) / range_d)
    else:
        # If all candidates are equidistant, give equal normalized proximity
        audio_similarities = np.full_like(min_distances, 1.0)

    # 5. Composite score ranking
    scored_candidates: list[ScoredRecommendation] = []
    alpha = max(0.0, min(1.0, alpha))

    for idx, cand in enumerate(active_candidates):
        a_sim = float(audio_similarities[idx])
        l_sim = float(max(0.0, min(1.0, cand.lastfm_similarity)))
        comp_score = (alpha * a_sim) + ((1.0 - alpha) * l_sim)

        scored_candidates.append(
            ScoredRecommendation(
                id=cand.id,
                name=cand.name,
                artist=cand.artist,
                features=cand.features,
                audio_similarity=a_sim,
                lastfm_similarity=l_sim,
                composite_score=comp_score,
                nearest_cluster=int(nearest_clusters[idx]),
            )
        )

    # Sort descending by composite score
    scored_candidates.sort(key=lambda s: s.composite_score, reverse=True)

    # 6. Select Top M with cluster diversity when k > 1
    top_m = max(1, min(top_m, len(scored_candidates)))
    if k > 1:
        # Group candidates by assigned cluster
        cluster_buckets: dict[int, list[ScoredRecommendation]] = {c_id: [] for c_id in range(k)}
        for sc in scored_candidates:
            cluster_buckets[sc.nearest_cluster].append(sc)

        winning_recommendations: list[ScoredRecommendation] = []
        # Round-robin selection across clusters
        while len(winning_recommendations) < top_m:
            added_in_round = False
            for c_id in range(k):
                if cluster_buckets[c_id] and len(winning_recommendations) < top_m:
                    winning_recommendations.append(cluster_buckets[c_id].pop(0))
                    added_in_round = True
            if not added_in_round:
                break
    else:
        winning_recommendations = scored_candidates[:top_m]

    # 7. Project active points (seeds, centroids, winning recommendations) into 2D via PCA
    rec_indices = [
        active_candidates.index(
            next(c for c in active_candidates if c.id == rec.id)
        )
        for rec in winning_recommendations
    ]
    scaled_recs = scaled_cands[rec_indices]

    # Combine all vectors to fit PCA
    pca_input = np.vstack([scaled_seeds, centroids, scaled_recs])
    # Handle zero-variance degenerate case gracefully
    if float(np.sum(np.var(pca_input, axis=0))) < 1e-9:
        seeds_2d = np.zeros((scaled_seeds.shape[0], 2))
        centroids_2d = np.zeros((centroids.shape[0], 2))
        recs_2d = np.zeros((scaled_recs.shape[0], 2))
    else:
        n_components = min(2, pca_input.shape[0], pca_input.shape[1])
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(pca_input)

        seeds_2d = pca.transform(scaled_seeds)
        centroids_2d = pca.transform(centroids)
        recs_2d = pca.transform(scaled_recs)

    points_2d: list[Point2D] = []

    # Helper to extract (x, y) safely even if n_components < 2 (fallback to 0.0)
    def get_xy(coords_row: np.ndarray) -> tuple[float, float]:
        px = float(coords_row[0]) if coords_row.shape[0] > 0 else 0.0
        py = float(coords_row[1]) if coords_row.shape[0] > 1 else 0.0
        return px, py

    # Add seed points
    for idx, seed in enumerate(seed_tracks):
        px, py = get_xy(seeds_2d[idx])
        points_2d.append(
            Point2D(
                id=seed.id,
                name=seed.name,
                artist=seed.artist,
                x=px,
                y=py,
                point_type="seed",
                cluster_id=seed_clusters[idx],
            )
        )

    # Add centroid points
    for c_id in range(k):
        px, py = get_xy(centroids_2d[c_id])
        points_2d.append(
            Point2D(
                id=f"centroid_{c_id}",
                name=f"Cluster {c_id + 1} Centroid",
                artist="Cluster Centroid",
                x=px,
                y=py,
                point_type="centroid",
                cluster_id=c_id,
            )
        )

    # Add recommended points
    for idx, rec in enumerate(winning_recommendations):
        px, py = get_xy(recs_2d[idx])
        points_2d.append(
            Point2D(
                id=rec.id,
                name=rec.name,
                artist=rec.artist,
                x=px,
                y=py,
                point_type="recommendation",
                cluster_id=rec.nearest_cluster,
            )
        )

    return SimilarityResult(
        recommendations=winning_recommendations,
        points_2d=points_2d,
        k=k,
        seed_clusters=seed_clusters,
    )
