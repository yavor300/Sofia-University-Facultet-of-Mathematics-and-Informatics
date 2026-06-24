"""Relation extraction metric helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gutbrainie.evaluation.ner_metrics import _classification_report, _require_columns

MENTION_RELATION_KEY_COLUMNS = [
    "pmid",
    "subject_text_span",
    "subject_label",
    "predicate",
    "object_text_span",
    "object_label",
]


def evaluate_mention_relations(gold_relations: pd.DataFrame, pred_relations: pd.DataFrame) -> dict[str, Any]:
    """Evaluate exact-match mention-level relation predictions."""
    _require_columns(gold_relations, MENTION_RELATION_KEY_COLUMNS, "gold_relations")
    _require_columns(pred_relations, MENTION_RELATION_KEY_COLUMNS, "pred_relations")

    from collections import Counter

    gold_counter = Counter(_mention_relation_key(row) for _, row in gold_relations.iterrows())
    pred_counter = Counter(_mention_relation_key(row) for _, row in pred_relations.iterrows())
    labels = sorted({_relation_class(key) for key in gold_counter} | {_relation_class(key) for key in pred_counter})

    return _classification_report(gold_counter, pred_counter, labels, _relation_class)


def _mention_relation_key(row: pd.Series) -> tuple[str, str, str, str, str, str]:
    return (
        str(row["pmid"]),
        str(row["subject_text_span"]),
        str(row["subject_label"]),
        str(row["predicate"]),
        str(row["object_text_span"]),
        str(row["object_label"]),
    )


def _relation_class(key: tuple[str, str, str, str, str, str]) -> str:
    return f"{key[2]}|{key[3]}|{key[5]}"
