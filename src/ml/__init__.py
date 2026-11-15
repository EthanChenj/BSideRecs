"""Machine learning modules for clustering, similarity, and dimensionality reduction."""

from src.ml.similarity import (
    Point2D,
    ScoredRecommendation,
    SimilarityResult,
    TrackItem,
    calculate_dynamic_k,
    rank_and_project,
)

__all__ = [
    "Point2D",
    "ScoredRecommendation",
    "SimilarityResult",
    "TrackItem",
    "calculate_dynamic_k",
    "rank_and_project",
]
