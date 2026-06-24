"""NER metric helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable
from typing import Any

import pandas as pd

NER_KEY_COLUMNS = ["pmid", "location", "start_idx", "end_idx", "label"]


def evaluate_ner(gold_entities: pd.DataFrame, pred_entities: pd.DataFrame) -> dict[str, Any]:
    """Evaluate exact-match NER predictions.

    A prediction is correct when ``pmid``, ``location``, ``start_idx``, ``end_idx``,
    and ``label`` all match. ``text_span`` is intentionally not part of the key.
    """
    _require_columns(gold_entities, NER_KEY_COLUMNS, "gold_entities")
    _require_columns(pred_entities, NER_KEY_COLUMNS, "pred_entities")

    gold_counter = Counter(_ner_key(row) for _, row in gold_entities.iterrows())
    pred_counter = Counter(_ner_key(row) for _, row in pred_entities.iterrows())
    labels = sorted({key[-1] for key in gold_counter} | {key[-1] for key in pred_counter})

    return _classification_report(gold_counter, pred_counter, labels, lambda key: key[-1])


def _ner_key(row: pd.Series) -> tuple[str, str, int, int, str]:
    return (
        str(row["pmid"]),
        str(row["location"]),
        int(row["start_idx"]),
        int(row["end_idx"]),
        str(row["label"]),
    )


def _classification_report(
    gold_counter: Counter[Hashable],
    pred_counter: Counter[Hashable],
    labels: list[str],
    label_for_key,
) -> dict[str, Any]:
    tp = sum((gold_counter & pred_counter).values())
    fp = sum((pred_counter - gold_counter).values())
    fn = sum((gold_counter - pred_counter).values())

    micro_precision, micro_recall, micro_f1 = _prf(tp, fp, fn)

    per_label_precision: dict[str, float] = {}
    per_label_recall: dict[str, float] = {}
    per_label_f1: dict[str, float] = {}
    per_label_counts: dict[str, dict[str, int]] = {}

    for label in labels:
        label_gold = Counter({key: count for key, count in gold_counter.items() if label_for_key(key) == label})
        label_pred = Counter({key: count for key, count in pred_counter.items() if label_for_key(key) == label})
        label_tp = sum((label_gold & label_pred).values())
        label_fp = sum((label_pred - label_gold).values())
        label_fn = sum((label_gold - label_pred).values())
        precision, recall, f1 = _prf(label_tp, label_fp, label_fn)
        per_label_precision[label] = precision
        per_label_recall[label] = recall
        per_label_f1[label] = f1
        per_label_counts[label] = {"tp": label_tp, "fp": label_fp, "fn": label_fn}

    macro_precision = _mean(per_label_precision.values())
    macro_recall = _mean(per_label_recall.values())
    macro_f1 = _mean(per_label_f1.values())

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_label_precision": per_label_precision,
        "per_label_recall": per_label_recall,
        "per_label_f1": per_label_f1,
        "per_label_counts": per_label_counts,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gold_total": sum(gold_counter.values()),
        "pred_total": sum(pred_counter.values()),
    }


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
