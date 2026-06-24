"""Dictionary and rule-based NER baseline."""

from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from gutbrainie.data.annotations import load_entities_csv
from gutbrainie.data.articles import load_articles_csv
from gutbrainie.submission.export_t611 import entities_to_t611_json

TRAILING_PUNCTUATION = string.punctuation + "”’"


def normalize_entity_key(text: str) -> str:
    """Normalize a mention string for dictionary lookup."""
    return str(text).strip().rstrip(TRAILING_PUNCTUATION).lower()


def build_entity_dictionary(train_entities: pd.DataFrame) -> dict[str, set[str]]:
    """Build a normalized mention-to-label-set dictionary from training entities."""
    _require_columns(train_entities, ["text_span", "label"], "train_entities")
    entity_dictionary: dict[str, set[str]] = defaultdict(set)
    for _, row in train_entities.iterrows():
        key = normalize_entity_key(row["text_span"])
        if key:
            entity_dictionary[key].add(str(row["label"]))
    return dict(entity_dictionary)


def build_dictionary_statistics(train_entities: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Count normalized mention/label frequencies for tie-breaking."""
    _require_columns(train_entities, ["text_span", "label"], "train_entities")
    counts: Counter[tuple[str, str]] = Counter()
    for _, row in train_entities.iterrows():
        key = normalize_entity_key(row["text_span"])
        label = str(row["label"])
        if key:
            counts[(key, label)] += 1
    return dict(counts)


def predict_dictionary_entities(articles: pd.DataFrame, train_entities: pd.DataFrame) -> pd.DataFrame:
    """Predict entity mentions by exact case-insensitive dictionary matching."""
    _require_columns(articles, ["pmid", "title", "abstract"], "articles")
    entity_dictionary = build_entity_dictionary(train_entities)
    frequency = build_dictionary_statistics(train_entities)
    terms = sorted(entity_dictionary, key=lambda value: (-len(value), value))
    term_patterns = [
        (term, re.compile(rf"(?<!\w){re.escape(term)}(?!\w)"))
        for term in terms
        if term
    ]

    predictions: list[dict[str, Any]] = []
    for _, article in articles.iterrows():
        pmid = str(article["pmid"])
        for location in ("title", "abstract"):
            text = str(article[location])
            predictions.extend(
                _predict_in_text(
                    pmid=pmid,
                    location=location,
                    text=text,
                    term_patterns=term_patterns,
                    entity_dictionary=entity_dictionary,
                    frequency=frequency,
                )
            )

    return pd.DataFrame(
        predictions,
        columns=["pmid", "start_idx", "end_idx", "location", "text_span", "label"],
    )


def predict_dictionary_to_json(
    train_entities_path: str | Path,
    articles_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Run dictionary NER and write T611-compatible JSON predictions."""
    train_entities = load_entities_csv(train_entities_path)
    articles = load_articles_csv(articles_path)
    predictions = predict_dictionary_entities(articles, train_entities)
    payload = entities_to_t611_json(predictions)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return predictions


def _predict_in_text(
    pmid: str,
    location: str,
    text: str,
    term_patterns: list[tuple[str, re.Pattern[str]]],
    entity_dictionary: dict[str, set[str]],
    frequency: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    lowered_text = text.lower()

    for term, pattern in term_patterns:
        for match in pattern.finditer(lowered_text):
            start_idx = match.start()
            exclusive_end_idx = match.end()
            text_span = text[start_idx:exclusive_end_idx]
            normalized_span = normalize_entity_key(text_span)
            if normalized_span != term:
                continue
            for label in entity_dictionary[term]:
                candidates.append(
                    {
                        "pmid": pmid,
                        "start_idx": start_idx,
                        "end_idx": exclusive_end_idx - 1,
                        "location": location,
                        "text_span": text_span,
                        "label": label,
                        "_term": term,
                        "_frequency": frequency.get((term, label), 0),
                        "_exclusive_end_idx": exclusive_end_idx,
                    }
                )

    ranked = sorted(
        candidates,
        key=lambda row: (
            -(row["end_idx"] - row["start_idx"]),
            -row["_frequency"],
            row["label"],
            row["start_idx"],
            row["end_idx"],
        ),
    )
    selected: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for candidate in ranked:
        span = (int(candidate["start_idx"]), int(candidate["end_idx"]))
        span = (int(candidate["start_idx"]), int(candidate["_exclusive_end_idx"]))
        if any(_overlaps(span, kept) for kept in occupied):
            continue
        occupied.append(span)
        selected.append({key: value for key, value in candidate.items() if not key.startswith("_")})

    return sorted(selected, key=lambda row: (row["start_idx"], row["end_idx"], row["label"]))


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
