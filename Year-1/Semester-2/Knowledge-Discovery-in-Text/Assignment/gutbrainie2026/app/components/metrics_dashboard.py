"""Metrics loading and reshaping helpers for the Streamlit demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

SUMMARY_KEYS = [
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "tp",
    "fp",
    "fn",
    "gold_total",
    "pred_total",
]


def discover_metric_files(reports_dir: str | Path = "outputs/reports") -> list[Path]:
    root = Path(reports_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*.json") if "metrics" in path.name or path.name.startswith("pipeline_"))


def load_metric_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "metrics" in payload and isinstance(payload["metrics"], dict) and payload["metrics"].get("evaluated"):
        return payload["metrics"]
    return payload


def metric_sections(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "ner" in report or "re" in report:
        return {key: value for key, value in report.items() if key in {"ner", "re"} and isinstance(value, dict)}
    return {"metrics": report}


def summary_dataframe(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        try:
            report = load_metric_report(path)
        except (json.JSONDecodeError, OSError):
            continue
        for section, metrics in metric_sections(report).items():
            row = {"file": path.name, "section": section}
            row.update({key: metrics.get(key) for key in SUMMARY_KEYS})
            rows.append(row)
    return pd.DataFrame(rows)


def per_label_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    labels = sorted(
        set(metrics.get("per_label_precision", {}))
        | set(metrics.get("per_label_recall", {}))
        | set(metrics.get("per_label_f1", {}))
        | set(metrics.get("per_label_counts", {}))
    )
    rows = []
    for label in labels:
        counts = metrics.get("per_label_counts", {}).get(label, {})
        rows.append(
            {
                "label": label,
                "precision": metrics.get("per_label_precision", {}).get(label, 0.0),
                "recall": metrics.get("per_label_recall", {}).get(label, 0.0),
                "f1": metrics.get("per_label_f1", {}).get(label, 0.0),
                "tp": counts.get("tp", 0),
                "fp": counts.get("fp", 0),
                "fn": counts.get("fn", 0),
                "support": counts.get("tp", 0) + counts.get("fn", 0),
            }
        )
    return pd.DataFrame(rows)
