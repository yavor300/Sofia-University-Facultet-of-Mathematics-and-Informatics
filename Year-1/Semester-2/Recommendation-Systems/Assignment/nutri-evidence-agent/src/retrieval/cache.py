"""JSON article cache utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_articles(path: str) -> list[dict]:
    """Load article dictionaries from a UTF-8 JSON file."""
    cache_path = Path(path)
    if not cache_path.exists():
        return []

    with cache_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Article cache must contain a JSON list: {cache_path}")

    return [item for item in data if isinstance(item, dict)]


def save_articles(articles: list[dict], path: str) -> None:
    """Save article dictionaries as pretty UTF-8 JSON, creating parents."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(articles, file, ensure_ascii=False, indent=2)
        file.write("\n")


def merge_articles(existing: list[dict], new_articles: list[dict]) -> list[dict]:
    """Merge article lists by PMID while preserving existing duplicates first."""
    merged: list[dict] = []
    seen_pmids: set[str] = set()

    for article in [*existing, *new_articles]:
        if not isinstance(article, dict):
            continue

        pmid = _pmid(article)
        if pmid and pmid in seen_pmids:
            continue

        if pmid:
            seen_pmids.add(pmid)

        merged.append(article)

    return merged


def _pmid(article: dict[str, Any]) -> str:
    return str(article.get("pmid", "")).strip()
