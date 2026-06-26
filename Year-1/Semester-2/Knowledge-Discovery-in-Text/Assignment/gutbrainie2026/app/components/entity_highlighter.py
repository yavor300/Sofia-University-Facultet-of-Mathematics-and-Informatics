"""Entity highlighting and NER comparison helpers for the Streamlit demo."""

from __future__ import annotations

import html
from collections import Counter
from typing import Any

import pandas as pd

from gutbrainie.data.offsets import resolve_exclusive_end

LABEL_COLORS = {
    "DDF": "#f6c85f",
    "anatomical location": "#7ccba2",
    "animal": "#90b7e8",
    "bacteria": "#f08a84",
    "biomedical technique": "#b7a3e3",
    "chemical": "#f3a65a",
    "dietary supplement": "#a8d08d",
    "drug": "#d7aefb",
    "food": "#ffe08a",
    "gene": "#8fd3c7",
    "human": "#a9c4eb",
    "microbiome": "#ffb3ba",
    "statistical technique": "#c6c6c6",
}

STATUS_COLORS = {
    "True Positive": "#b7e4c7",
    "False Positive": "#ffccd5",
    "False Negative": "#ffd6a5",
}

ENTITY_COLUMNS = ["pmid", "start_idx", "end_idx", "location", "text_span", "label"]


def ner_key(row: pd.Series | dict[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(row["pmid"]),
        str(row["location"]),
        int(row["start_idx"]),
        int(row["end_idx"]),
        str(row["label"]),
    )


def compare_entities(gold: pd.DataFrame | None, predicted: pd.DataFrame | None) -> pd.DataFrame:
    """Return predicted TP/FP rows plus gold FN rows."""
    predicted = _ensure_entities(predicted)
    if gold is None or gold.empty:
        out = predicted.copy()
        out["status"] = "Prediction"
        out["source"] = "prediction"
        return out

    gold = _ensure_entities(gold)
    gold_counter = Counter(ner_key(row) for _, row in gold.iterrows())
    pred_counter = Counter(ner_key(row) for _, row in predicted.iterrows())
    matched = gold_counter & pred_counter

    rows: list[dict[str, Any]] = []
    matched_used: Counter = Counter()
    for _, row in predicted.iterrows():
        key = ner_key(row)
        status = "True Positive" if matched_used[key] < matched[key] else "False Positive"
        matched_used[key] += 1
        rows.append({**row.to_dict(), "status": status, "source": "prediction"})

    missed_counter = gold_counter - pred_counter
    missed_used: Counter = Counter()
    for _, row in gold.iterrows():
        key = ner_key(row)
        if missed_used[key] >= missed_counter[key]:
            continue
        missed_used[key] += 1
        rows.append({**row.to_dict(), "status": "False Negative", "source": "gold"})

    return pd.DataFrame(rows, columns=[*ENTITY_COLUMNS, "status", "source"])


def highlight_entities(text: str, entities: pd.DataFrame | list[dict[str, Any]], use_status_colors: bool = False) -> str:
    """Return safe HTML with entity spans highlighted."""
    if isinstance(entities, list):
        entity_df = pd.DataFrame(entities)
    else:
        entity_df = entities.copy()
    if entity_df.empty:
        return f'<div class="article-text">{html.escape(text)}</div>'

    spans = []
    for _, row in entity_df.iterrows():
        try:
            start = int(row["start_idx"])
            raw_end = int(row["end_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or start >= len(text):
            continue
        span_text = str(row.get("text_span", ""))
        end = resolve_exclusive_end(text, start, raw_end, span_text)
        if end <= start or end > len(text):
            end = min(len(text), max(start + 1, raw_end + 1))
        spans.append(
            {
                "start": start,
                "end": end,
                "label": str(row.get("label", "")),
                "status": str(row.get("status", "")),
                "text_span": span_text or text[start:end],
            }
        )

    spans = sorted(spans, key=lambda item: (item["start"], -(item["end"] - item["start"]), item["label"]))
    selected: list[dict[str, Any]] = []
    occupied_until = -1
    for span in spans:
        if span["start"] < occupied_until:
            continue
        selected.append(span)
        occupied_until = span["end"]

    pieces: list[str] = ['<div class="article-text">']
    cursor = 0
    for span in selected:
        pieces.append(html.escape(text[cursor : span["start"]]))
        label = span["label"]
        status = span["status"]
        color = STATUS_COLORS.get(status) if use_status_colors else LABEL_COLORS.get(label)
        color = color or LABEL_COLORS.get(label) or "#e9ecef"
        title = html.escape(f"{span['text_span']} | {label}" + (f" | {status}" if status else ""))
        pieces.append(
            '<span class="entity-span" '
            f'style="background:{color}; border-color:{_border_color(status)};" '
            f'title="{title}">'
            f'{html.escape(text[span["start"] : span["end"]])}'
            f'<span class="entity-label">{html.escape(label)}</span>'
            "</span>"
        )
        cursor = span["end"]
    pieces.append(html.escape(text[cursor:]))
    pieces.append("</div>")
    return "".join(pieces)


def legend_html(labels: list[str], status_mode: bool = False) -> str:
    if status_mode:
        items = STATUS_COLORS.items()
    else:
        items = [(label, LABEL_COLORS.get(label, "#e9ecef")) for label in labels]
    chips = [
        f'<span class="legend-chip"><span class="legend-dot" style="background:{color}"></span>{html.escape(label)}</span>'
        for label, color in items
    ]
    return '<div class="legend">' + "".join(chips) + "</div>"


def style_block() -> str:
    return """
<style>
.article-text {
  font-size: 0.98rem;
  line-height: 1.85;
  color: #1f2937;
}
.entity-span {
  border: 1px solid rgba(31,41,55,.25);
  border-radius: 4px;
  padding: 0.05rem 0.18rem;
  margin: 0 0.04rem;
  display: inline;
}
.entity-label {
  font-size: 0.67rem;
  font-weight: 700;
  margin-left: 0.22rem;
  color: #111827;
  opacity: .78;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: .4rem .65rem;
  margin: .5rem 0 .9rem 0;
}
.legend-chip {
  display: inline-flex;
  align-items: center;
  gap: .28rem;
  font-size: .82rem;
  color: #374151;
}
.legend-dot {
  width: .75rem;
  height: .75rem;
  border-radius: 50%;
  border: 1px solid rgba(0,0,0,.18);
}
.relation-card {
  border: 1px solid #d1d5db;
  border-radius: 6px;
  padding: .65rem .75rem;
  margin-bottom: .55rem;
  background: #ffffff;
}
.status-true-positive { border-left: 5px solid #2f9e44; }
.status-false-positive { border-left: 5px solid #e03131; }
.status-false-negative { border-left: 5px solid #f08c00; }
.status-prediction { border-left: 5px solid #6c757d; }
</style>
"""


def _border_color(status: str) -> str:
    return {
        "True Positive": "#2f9e44",
        "False Positive": "#e03131",
        "False Negative": "#f08c00",
    }.get(status, "rgba(31,41,55,.25)")


def _ensure_entities(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=ENTITY_COLUMNS)
    out = df.copy()
    for column in ENTITY_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    out["pmid"] = out["pmid"].astype(str)
    out["start_idx"] = out["start_idx"].astype(int)
    out["end_idx"] = out["end_idx"].astype(int)
    return out[ENTITY_COLUMNS]
