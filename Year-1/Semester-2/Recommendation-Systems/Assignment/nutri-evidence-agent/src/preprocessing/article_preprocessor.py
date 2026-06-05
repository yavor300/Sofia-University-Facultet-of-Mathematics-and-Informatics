"""Preprocessing helpers for normalized article records."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace and trim surrounding spaces."""
    if text is None:
        return ""

    return _WHITESPACE_RE.sub(" ", str(text)).strip()


def build_document_text(article: dict) -> str:
    """Build the text used by recommenders from article title and abstract."""
    title = normalize_text(article.get("title", ""))
    abstract = normalize_text(article.get("abstract", ""))

    if title and abstract:
        return f"{title}. {abstract}"
    if title:
        return f"{title}."
    return abstract


def filter_valid_articles(articles: list[dict]) -> list[dict]:
    """Return normalized article copies with document text and required fields."""
    valid_articles: list[dict] = []

    for article in articles:
        if not isinstance(article, dict):
            continue

        normalized = _normalize_article(article)
        if not normalized["title"] and not normalized["abstract"]:
            continue

        normalized["document_text"] = build_document_text(normalized)
        valid_articles.append(normalized)

    return valid_articles


def articles_to_dataframe(articles: list[dict]):
    """Convert normalized article dictionaries to a pandas DataFrame."""
    return pd.DataFrame(filter_valid_articles(articles))


def _normalize_article(article: dict[str, Any]) -> dict:
    normalized = dict(article)

    normalized["pmid"] = normalize_text(normalized.get("pmid", ""))
    normalized["title"] = normalize_text(normalized.get("title", ""))
    normalized["abstract"] = normalize_text(normalized.get("abstract", ""))
    normalized["journal"] = normalize_text(normalized.get("journal", ""))
    normalized["doi"] = normalize_text(normalized.get("doi", ""))
    normalized["source_query"] = normalize_text(normalized.get("source_query", ""))
    normalized["authors"] = _ensure_list(normalized.get("authors", []))
    normalized["publication_types"] = _ensure_list(normalized.get("publication_types", []))
    normalized["mesh_terms"] = _ensure_list(normalized.get("mesh_terms", []))

    return normalized


def _ensure_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item is not None]
    if isinstance(value, tuple | set):
        return [item for item in value if item is not None]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [value]
