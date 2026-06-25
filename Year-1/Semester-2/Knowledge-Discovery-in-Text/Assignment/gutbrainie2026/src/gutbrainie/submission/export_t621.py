"""T621 JSON export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

T621_RELATION_COLUMNS = [
    "pmid",
    "subject_text_span",
    "subject_label",
    "predicate",
    "object_text_span",
    "object_label",
]


def mention_relations_to_t621_json(relations: pd.DataFrame) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Convert mention-level relation predictions to the T621 JSON shape."""
    _require_columns(relations, T621_RELATION_COLUMNS, "relations")
    payload: dict[str, dict[str, list[dict[str, Any]]]] = {}
    sort_columns = ["pmid", "subject_label", "subject_text_span", "predicate", "object_label", "object_text_span"]
    for pmid, group in relations.sort_values(sort_columns).groupby("pmid", sort=False):
        payload[str(pmid)] = {
            "mention_level_relations": [
                {
                    "subject_text_span": str(row["subject_text_span"]),
                    "subject_label": str(row["subject_label"]),
                    "predicate": str(row["predicate"]),
                    "object_text_span": str(row["object_text_span"]),
                    "object_label": str(row["object_label"]),
                }
                for _, row in group.iterrows()
            ]
        }
    return payload


def load_t621_json(path: str | Path) -> pd.DataFrame:
    """Load T621 JSON predictions into the internal relation DataFrame shape."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for pmid, article_payload in payload.items():
        for relation in article_payload.get("mention_level_relations", []):
            rows.append(
                {
                    "pmid": str(pmid),
                    "subject_text_span": str(relation["subject_text_span"]),
                    "subject_label": str(relation["subject_label"]),
                    "predicate": str(relation["predicate"]),
                    "object_text_span": str(relation["object_text_span"]),
                    "object_label": str(relation["object_label"]),
                }
            )
    return pd.DataFrame(rows, columns=T621_RELATION_COLUMNS)


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
