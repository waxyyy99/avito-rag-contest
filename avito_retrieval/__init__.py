"""Utilities for the Avito help-article retrieval task."""

from .data import Dataset, clean_html, load_dataset
from .lexical import LexicalCVResult, LexicalRetriever, evaluate_lexical_cv
from .metrics import average_precision_at_k, mean_average_precision_at_k

__all__ = [
    "Dataset",
    "LexicalCVResult",
    "LexicalRetriever",
    "average_precision_at_k",
    "clean_html",
    "evaluate_lexical_cv",
    "load_dataset",
    "mean_average_precision_at_k",
]
