"""Rank-based fusion for independent retrieval channels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def reciprocal_rank_fusion(
    rankings: Sequence[np.ndarray],
    article_ids: np.ndarray,
    weights: Sequence[float] | None = None,
    rank_constant: int = 60,
    top_k: int = 10,
) -> np.ndarray:
    """Fuse aligned article rankings without relying on incomparable score scales."""
    if not rankings:
        raise ValueError("at least one ranking channel is required")
    if rank_constant < 0:
        raise ValueError("rank_constant must be non-negative")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(rankings) != len(weights):
        raise ValueError("weights must have one value per ranking channel")

    article_ids = np.asarray(article_ids, dtype=np.int64)
    article_positions = {article_id: position for position, article_id in enumerate(article_ids)}
    query_count = len(rankings[0])
    scores = np.zeros((query_count, len(article_ids)), dtype=np.float32)
    for ranking, weight in zip(rankings, weights, strict=True):
        if ranking.ndim != 2 or len(ranking) != query_count:
            raise ValueError("rankings must be aligned two-dimensional arrays")
        for rank, ranked_ids in enumerate(ranking.T, start=1):
            positions = np.asarray([article_positions[article_id] for article_id in ranked_ids], dtype=np.intp)
            scores[np.arange(query_count), positions] += weight / (rank_constant + rank)

    order = np.lexsort((np.broadcast_to(article_ids, scores.shape), -scores), axis=1)
    return article_ids[order[:, :top_k]]
