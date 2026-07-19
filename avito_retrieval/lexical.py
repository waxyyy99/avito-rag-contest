"""TF-IDF retrieval baselines for help-article ranking."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

from .metrics import mean_average_precision_at_k


@dataclass(frozen=True)
class LexicalCVResult:
    """Cross-validation MAP@k scores for a lexical retrieval configuration."""

    fold_scores: tuple[float, ...]

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.fold_scores))


class LexicalRetriever:
    """Combine query-to-query and query-to-article TF-IDF signals."""

    def __init__(self) -> None:
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
        )

    def fit(self, calibration: pd.DataFrame, articles: pd.DataFrame) -> "LexicalRetriever":
        """Build indices from a calibration training split and the article corpus."""
        self.article_ids = articles["article_id"].to_numpy(dtype=np.int64)
        self.article_index = {article_id: index for index,
                              article_id in enumerate(self.article_ids)}
        self.training_targets = calibration["ground_truth_ids"].tolist()

        query_texts = calibration["clean_query"].tolist()
        document_texts = articles["document_text"].tolist()
        self.word_vectorizer.fit(query_texts + document_texts)
        self.char_vectorizer.fit(query_texts)
        self.word_queries = self.word_vectorizer.transform(query_texts)
        self.char_queries = self.char_vectorizer.transform(query_texts)
        self.word_documents = self.word_vectorizer.transform(document_texts)
        return self

    def score(self, query: str) -> dict[str, np.ndarray]:
        """Return one score per article from every lexical retrieval channel."""
        word_query = self.word_vectorizer.transform([query])
        char_query = self.char_vectorizer.transform([query])
        word_neighbors = (word_query @ self.word_queries.T).toarray().ravel()
        char_neighbors = (char_query @ self.char_queries.T).toarray().ravel()

        word_scores = np.zeros(len(self.article_ids), dtype=np.float32)
        char_scores = np.zeros(len(self.article_ids), dtype=np.float32)
        for row_index, target_ids in enumerate(self.training_targets):
            for article_id in target_ids:
                article_index = self.article_index[article_id]
                word_scores[article_index] += word_neighbors[row_index]
                char_scores[article_index] = max(
                    char_scores[article_index], char_neighbors[row_index])

        document_scores = (
            word_query @ self.word_documents.T).toarray().ravel().astype(np.float32)
        return {
            "query_word": word_scores,
            "query_char": char_scores,
            "document_word": document_scores,
        }

    def rank(
        self,
        query: str,
        weights: tuple[float, float, float] = (0.45, 0.40, 0.15),
        top_k: int = 10,
    ) -> list[int]:
        """Return article ids ordered by normalized weighted lexical score."""
        scores = self.score(query)
        combined = sum(
            weight * _normalize(scores[channel])
            for weight, channel in zip(weights, ("query_word", "query_char", "document_word"), strict=True)
        )
        order = np.lexsort((self.article_ids, -combined))
        return self.article_ids[order[:top_k]].tolist()


def evaluate_lexical_cv(
    calibration: pd.DataFrame,
    articles: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    weights: tuple[float, float, float] = (0.45, 0.40, 0.15),
) -> LexicalCVResult:
    """Evaluate the lexical ranker without fitting on validation queries."""
    splitter = KFold(n_splits=n_splits, shuffle=True,
                     random_state=random_state)
    fold_scores: list[float] = []

    for train_indices, validation_indices in splitter.split(calibration):
        train = calibration.iloc[train_indices]
        validation = calibration.iloc[validation_indices]
        retriever = LexicalRetriever().fit(train, articles)
        predictions = [retriever.rank(query, weights=weights)
                       for query in validation["clean_query"]]
        fold_scores.append(
            mean_average_precision_at_k(
                predictions, validation["ground_truth_ids"], k=10)
        )

    return LexicalCVResult(fold_scores=tuple(fold_scores))


def _normalize(scores: np.ndarray) -> np.ndarray:
    maximum = scores.max(initial=0.0)
    return scores / maximum if maximum > 0 else scores
