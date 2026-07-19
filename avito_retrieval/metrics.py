"""Ranking metrics for local validation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def average_precision_at_k(
    predicted_ids: Sequence[int], relevant_ids: Iterable[int], k: int = 10
) -> float:
    """Calculate AP@k, ignoring repeated predictions after their first occurrence."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen: set[int] = set()
    for rank, article_id in enumerate(predicted_ids[:k], start=1):
        if article_id in seen:
            continue
        seen.add(article_id)
        if article_id in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(len(relevant), k)


def mean_average_precision_at_k(
    predictions: Iterable[Sequence[int]], relevant_sets: Iterable[Iterable[int]], k: int = 10
) -> float:
    """Calculate MAP@k over aligned ranked predictions and target sets."""
    values = [
        average_precision_at_k(predicted_ids, relevant_ids, k=k)
        for predicted_ids, relevant_ids in zip(predictions, relevant_sets, strict=True)
    ]
    if not values:
        raise ValueError("MAP@k requires at least one query")
    return sum(values) / len(values)
