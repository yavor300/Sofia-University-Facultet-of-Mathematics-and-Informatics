"""Relation comparison and display helpers for the Streamlit demo."""

from __future__ import annotations

import html
from collections import Counter
from typing import Any

import pandas as pd

RELATION_COLUMNS = [
    "pmid",
    "subject_text_span",
    "subject_label",
    "predicate",
    "object_text_span",
    "object_label",
]


def relation_key(row: pd.Series | dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row["pmid"]),
        str(row["subject_text_span"]),
        str(row["subject_label"]),
        str(row["predicate"]),
        str(row["object_text_span"]),
        str(row["object_label"]),
    )


def compare_relations(gold: pd.DataFrame | None, predicted: pd.DataFrame | None) -> pd.DataFrame:
    """Return predicted TP/FP rows plus gold FN rows."""
    predicted = _ensure_relations(predicted)
    if gold is None or gold.empty:
        out = predicted.copy()
        out["status"] = "Prediction"
        out["source"] = "prediction"
        return out

    gold = _ensure_relations(gold)
    gold_counter = Counter(relation_key(row) for _, row in gold.iterrows())
    pred_counter = Counter(relation_key(row) for _, row in predicted.iterrows())
    matched = gold_counter & pred_counter

    rows: list[dict[str, Any]] = []
    matched_used: Counter = Counter()
    for _, row in predicted.iterrows():
        key = relation_key(row)
        status = "True Positive" if matched_used[key] < matched[key] else "False Positive"
        matched_used[key] += 1
        rows.append({**row.to_dict(), "status": status, "source": "prediction"})

    missed_counter = gold_counter - pred_counter
    missed_used: Counter = Counter()
    for _, row in gold.iterrows():
        key = relation_key(row)
        if missed_used[key] >= missed_counter[key]:
            continue
        missed_used[key] += 1
        rows.append({**row.to_dict(), "status": "False Negative", "source": "gold"})

    return pd.DataFrame(rows, columns=[*RELATION_COLUMNS, "status", "source"])


def relation_cards_html(relations: pd.DataFrame, limit: int = 30) -> str:
    """Render relation triples as compact HTML cards."""
    if relations.empty:
        return "<p>No relations for this article.</p>"
    pieces = []
    for _, row in relations.head(limit).iterrows():
        status = str(row.get("status", "Prediction"))
        css_status = status.lower().replace(" ", "-")
        pieces.append(
            f'<div class="relation-card status-{html.escape(css_status)}">'
            f'<div><strong>{html.escape(str(row["subject_text_span"]))}</strong> '
            f'<span style="color:#6b7280">[{html.escape(str(row["subject_label"]))}]</span></div>'
            f'<div style="font-weight:700; margin:.2rem 0;">&rarr; {html.escape(str(row["predicate"]))} &rarr;</div>'
            f'<div><strong>{html.escape(str(row["object_text_span"]))}</strong> '
            f'<span style="color:#6b7280">[{html.escape(str(row["object_label"]))}]</span></div>'
            f'<div style="font-size:.8rem; color:#4b5563; margin-top:.25rem;">{html.escape(status)}</div>'
            "</div>"
        )
    if len(relations) > limit:
        pieces.append(f"<p>Showing {limit} of {len(relations)} relations.</p>")
    return "".join(pieces)


def _ensure_relations(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=RELATION_COLUMNS)
    out = df.copy()
    for column in RELATION_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    out["pmid"] = out["pmid"].astype(str)
    return out[RELATION_COLUMNS]
