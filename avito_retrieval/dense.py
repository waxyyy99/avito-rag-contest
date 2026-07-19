"""Dense retrieval utilities for help-article ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .metrics import mean_average_precision_at_k


Aggregation = Literal["max", "sum"]


@dataclass(frozen=True)
class DenseCVResult:
    """Cross-validation MAP@k for one dense query-to-query configuration."""

    fold_scores: tuple[float, ...]

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.fold_scores))


@dataclass(frozen=True)
class ArticleChunks:
    """Chunk texts together with the article represented by each chunk."""

    article_ids: np.ndarray
    texts: list[str]


def chunk_articles(
    articles: pd.DataFrame,
    chunk_size_words: int = 384,
    overlap_words: int = 64,
) -> ArticleChunks:
    """Split article bodies into overlapping word chunks, retaining title chunks."""
    if chunk_size_words <= 0 or not 0 <= overlap_words < chunk_size_words:
        raise ValueError("chunk_size_words must be positive and exceed overlap_words")

    article_ids: list[int] = []
    texts: list[str] = []
    step = chunk_size_words - overlap_words
    for article in articles.itertuples(index=False):
        article_id = int(article.article_id)
        title = str(article.clean_title).strip()
        body_words = str(article.clean_body).split()
        if title:
            article_ids.append(article_id)
            texts.append(title)
        if not body_words:
            continue
        for start in range(0, len(body_words), step):
            chunk = " ".join(body_words[start:start + chunk_size_words])
            if chunk:
                article_ids.append(article_id)
                texts.append(chunk)
            if start + chunk_size_words >= len(body_words):
                break

    return ArticleChunks(article_ids=np.asarray(article_ids, dtype=np.int64), texts=texts)


def evaluate_dense_query_to_query_cv(
    embeddings: np.ndarray,
    calibration: pd.DataFrame,
    aggregation: Aggregation = "max",
    n_splits: int = 5,
    random_state: int = 42,
) -> DenseCVResult:
    """Evaluate pretrained embeddings without fitting on validation examples."""
    if embeddings.ndim != 2 or len(embeddings) != len(calibration):
        raise ValueError("embeddings must have one two-dimensional row per calibration query")

    embeddings = _l2_normalize(embeddings.astype(np.float32, copy=False))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: list[float] = []

    for train_indices, validation_indices in splitter.split(calibration):
        train_targets = calibration.iloc[train_indices]["ground_truth_ids"].tolist()
        predictions = [
            _rank_by_neighbors(
                embeddings[query_index] @ embeddings[train_indices].T,
                train_targets,
                aggregation,
            )
            for query_index in validation_indices
        ]
        validation_targets = calibration.iloc[validation_indices]["ground_truth_ids"]
        fold_scores.append(mean_average_precision_at_k(predictions, validation_targets, k=10))

    return DenseCVResult(fold_scores=tuple(fold_scores))


def score_query_to_article(
    query_embeddings: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_article_ids: np.ndarray,
    article_ids: np.ndarray,
) -> np.ndarray:
    """Max-pool chunk cosine similarities into one score per article."""
    if query_embeddings.ndim != 2 or chunk_embeddings.ndim != 2:
        raise ValueError("query_embeddings and chunk_embeddings must be two-dimensional")
    if query_embeddings.shape[1] != chunk_embeddings.shape[1]:
        raise ValueError("query and chunk embedding dimensions must match")
    if len(chunk_embeddings) != len(chunk_article_ids):
        raise ValueError("chunk_article_ids must align with chunk_embeddings")

    article_ids = np.asarray(article_ids, dtype=np.int64)
    chunk_article_ids = np.asarray(chunk_article_ids, dtype=np.int64)
    article_positions = {article_id: position for position, article_id in enumerate(article_ids)}
    try:
        chunk_positions = np.asarray(
            [article_positions[article_id] for article_id in chunk_article_ids], dtype=np.intp
        )
    except KeyError as error:
        raise ValueError(f"Unknown chunk article id: {error.args[0]}") from error

    similarities = _l2_normalize(query_embeddings.astype(np.float32, copy=False)) @ (
        _l2_normalize(chunk_embeddings.astype(np.float32, copy=False)).T
    )
    scores = np.full((len(query_embeddings), len(article_ids)), -np.inf, dtype=np.float32)
    for query_index, row in enumerate(similarities):
        np.maximum.at(scores[query_index], chunk_positions, row)
    return scores


def score_query_to_query(
    query_embeddings: np.ndarray,
    neighbor_embeddings: np.ndarray,
    neighbor_targets: list[tuple[int, ...]],
    article_ids: np.ndarray,
    aggregation: Aggregation = "max",
) -> np.ndarray:
    """Aggregate query-neighbor similarities into one score per article."""
    if len(neighbor_embeddings) != len(neighbor_targets):
        raise ValueError("neighbor_targets must align with neighbor_embeddings")
    if query_embeddings.ndim != 2 or neighbor_embeddings.ndim != 2:
        raise ValueError("query_embeddings and neighbor_embeddings must be two-dimensional")
    if query_embeddings.shape[1] != neighbor_embeddings.shape[1]:
        raise ValueError("query and neighbor embedding dimensions must match")

    article_ids = np.asarray(article_ids, dtype=np.int64)
    article_positions = {article_id: position for position, article_id in enumerate(article_ids)}
    scores = np.full(
        (len(query_embeddings), len(article_ids)),
        -np.inf if aggregation == "max" else 0.0,
        dtype=np.float32,
    )
    similarities = _l2_normalize(query_embeddings.astype(np.float32, copy=False)) @ (
        _l2_normalize(neighbor_embeddings.astype(np.float32, copy=False)).T
    )
    for neighbor_index, target_ids in enumerate(neighbor_targets):
        for article_id in target_ids:
            position = article_positions[article_id]
            if aggregation == "max":
                scores[:, position] = np.maximum(scores[:, position], similarities[:, neighbor_index])
            elif aggregation == "sum":
                scores[:, position] += similarities[:, neighbor_index]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
    return scores


def rank_article_scores(scores: np.ndarray, article_ids: np.ndarray, top_k: int = 10) -> np.ndarray:
    """Rank article scores with article ID as a deterministic tie breaker."""
    article_ids = np.asarray(article_ids, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] != len(article_ids):
        raise ValueError("scores must have one column per article")
    order = np.lexsort((np.broadcast_to(article_ids, scores.shape), -scores), axis=1)
    return article_ids[order[:, :top_k]]


def evaluate_article_scores_cv(
    scores: np.ndarray,
    calibration: pd.DataFrame,
    article_ids: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> DenseCVResult:
    """Report article-retrieval MAP@10 on the same folds as other experiments."""
    if scores.shape != (len(calibration), len(article_ids)):
        raise ValueError("scores must have one row per calibration query and article")

    predictions = rank_article_scores(scores, article_ids)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = [
        mean_average_precision_at_k(
            predictions[validation_indices],
            calibration.iloc[validation_indices]["ground_truth_ids"],
            k=10,
        )
        for _, validation_indices in splitter.split(calibration)
    ]
    return DenseCVResult(fold_scores=tuple(fold_scores))


def load_or_create_embeddings(
    cache_path: str | Path,
    texts: list[str],
    encode: Callable[[list[str]], np.ndarray],
) -> np.ndarray:
    """Reuse cached vectors only when their count matches the requested texts."""
    cache_path = Path(cache_path)
    if cache_path.is_file():
        cached = np.load(cache_path)["embeddings"]
        if len(cached) == len(texts):
            return cached.astype(np.float32)

    embeddings = np.asarray(encode(texts), dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings)
    return embeddings


def _rank_by_neighbors(
    similarities: np.ndarray,
    target_ids: list[tuple[int, ...]],
    aggregation: Aggregation,
) -> list[int]:
    scores: dict[int, float] = {}
    for similarity, article_ids in zip(similarities, target_ids, strict=True):
        for article_id in article_ids:
            if aggregation == "max":
                scores[article_id] = max(scores.get(article_id, -np.inf), float(similarity))
            elif aggregation == "sum":
                scores[article_id] = scores.get(article_id, 0.0) + float(similarity)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

    return [article_id for article_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:10]]


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding vectors must be non-zero")
    return embeddings / norms
