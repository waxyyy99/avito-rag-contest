"""Data loading and deterministic text preparation."""

from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
import re

import pandas as pd


_SCRIPT_OR_STYLE_RE = re.compile(r"<(?:script|style)\b[^>]*>.*?</(?:script|style)\s*>", re.IGNORECASE | re.DOTALL)
_PLACEHOLDER_RE = re.compile(r"<([A-Z][A-Z0-9_]{1,})>")
_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Dataset:
    """Prepared task tables without using the test labels."""

    articles: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def clean_html(value: object) -> str:
    """Convert an article HTML field into a normalized text string."""
    if value is None or pd.isna(value):
        return ""

    text = html.    unescape(str(value))
    text = _SCRIPT_OR_STYLE_RE.sub(" ", text)
    # Preserve dataset placeholders before treating angle-bracket text as HTML.
    text = _PLACEHOLDER_RE.sub(r" \1 ", text)
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def parse_ground_truth(value: object) -> tuple[int, ...]:
    """Parse the space-separated article identifiers from calibration data."""
    if value is None or pd.isna(value) or not str(value).strip():
        return ()
    return tuple(int(article_id) for article_id in str(value).split())


def load_dataset(data_dir: str | Path) -> Dataset:
    """Load task data and add prepared fields used by retrieval modules."""
    data_dir = Path(data_dir)
    articles = pd.read_feather(data_dir / "articles.f").copy()
    calibration = pd.read_feather(data_dir / "calibration.f").copy()
    test = pd.read_feather(data_dir / "test.f").copy()

    _validate_columns(articles, {"article_id", "title", "body"}, "articles")
    _validate_columns(calibration, {"query_id", "query_text", "ground_truth"}, "calibration")
    _validate_columns(test, {"query_id", "query_text"}, "test")
    _validate_identifiers(articles, "article_id", "articles")
    _validate_identifiers(calibration, "query_id", "calibration")
    _validate_identifiers(test, "query_id", "test")

    articles["clean_title"] = articles["title"].map(clean_html)
    articles["clean_body"] = articles["body"].map(clean_html)
    articles["document_text"] = (
        articles["clean_title"] + " " + articles["clean_title"] + " " + articles["clean_body"]
    ).str.strip()
    calibration["clean_query"] = calibration["query_text"].map(clean_html)
    calibration["ground_truth_ids"] = calibration["ground_truth"].map(parse_ground_truth)
    test["clean_query"] = test["query_text"].map(clean_html)

    known_article_ids = set(articles["article_id"])
    unknown_ids = {
        article_id
        for targets in calibration["ground_truth_ids"]
        for article_id in targets
        if article_id not in known_article_ids
    }
    if unknown_ids:
        raise ValueError(f"Calibration references unknown article ids: {sorted(unknown_ids)}")

    return Dataset(articles=articles, calibration=calibration, test=test)


def _validate_columns(frame: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{table_name} is missing columns: {sorted(missing)}")


def _validate_identifiers(frame: pd.DataFrame, column: str, table_name: str) -> None:
    if frame[column].isna().any() or frame[column].duplicated().any():
        raise ValueError(f"{table_name}.{column} must contain unique non-null values")
