"""T611 JSON export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

T611_ENTITY_COLUMNS = ["pmid", "start_idx", "end_idx", "location", "text_span", "label"]


def entities_to_t611_json(entities: pd.DataFrame) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Convert entity predictions to the T611 JSON submission shape."""
    _require_columns(entities, T611_ENTITY_COLUMNS, "entities")
    payload: dict[str, dict[str, list[dict[str, Any]]]] = {}

    sort_columns = ["pmid", "location", "start_idx", "end_idx", "label"]
    for pmid, group in entities.sort_values(sort_columns).groupby("pmid", sort=False):
        payload[str(pmid)] = {
            "entities": [
                {
                    "start_idx": int(row["start_idx"]),
                    "end_idx": int(row["end_idx"]),
                    "location": str(row["location"]),
                    "text_span": str(row["text_span"]),
                    "label": str(row["label"]),
                }
                for _, row in group.iterrows()
            ]
        }

    return payload


def load_t611_json(path: str | Path) -> pd.DataFrame:
    """Load T611 JSON predictions into the internal entity DataFrame shape."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for pmid, article_payload in payload.items():
        for entity in article_payload.get("entities", []):
            rows.append(
                {
                    "pmid": str(pmid),
                    "start_idx": int(entity["start_idx"]),
                    "end_idx": int(entity["end_idx"]),
                    "location": str(entity["location"]),
                    "text_span": str(entity["text_span"]),
                    "label": str(entity["label"]),
                }
            )
    return pd.DataFrame(rows, columns=T611_ENTITY_COLUMNS)


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
