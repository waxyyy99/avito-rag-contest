"""Utilities for the Avito help-article retrieval task."""

from .data import Dataset, clean_html, load_dataset
from .dense import (
    ArticleChunks,
    DenseCVResult,
    chunk_articles,
    evaluate_article_scores_cv,
    evaluate_dense_query_to_query_cv,
    load_or_create_embeddings,
    rank_article_scores,
    score_query_to_article,
    score_query_to_query,
)
from .embeddings import FridaEmbedder
from .fusion import reciprocal_rank_fusion
from .lexical import LexicalCVResult, LexicalRetriever, evaluate_lexical_cv
from .metrics import average_precision_at_k, mean_average_precision_at_k

__all__ = [
    "Dataset",
    "ArticleChunks",
    "DenseCVResult",
    "FridaEmbedder",
    "LexicalCVResult",
    "LexicalRetriever",
    "average_precision_at_k",
    "clean_html",
    "chunk_articles",
    "evaluate_article_scores_cv",
    "evaluate_lexical_cv",
    "evaluate_dense_query_to_query_cv",
    "load_or_create_embeddings",
    "load_dataset",
    "mean_average_precision_at_k",
    "rank_article_scores",
    "reciprocal_rank_fusion",
    "score_query_to_article",
    "score_query_to_query",
]
