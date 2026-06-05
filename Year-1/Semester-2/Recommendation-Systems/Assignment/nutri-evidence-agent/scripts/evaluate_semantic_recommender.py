"""Compute semantic recommender metrics from human or OpenAI judge labels."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import ndcg_at_k, precision_at_k, reciprocal_rank


DEFAULT_ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "evaluation_annotations_openai.csv"
METHOD_DISPLAY_NAMES = {
    "semantic": "Semantic Similarity",
    "semantic+llm_rerank": "Semantic + Ollama LLM Reranking",
    "semantic_llm_rerank": "Semantic + Ollama LLM Reranking",
}
METHOD_ORDER = ["semantic", "semantic+llm_rerank"]


def main() -> int:
    try:
        args = parse_args()
        rows = load_labeled_rows(args.annotations)
        grouped = group_rows(rows)

        if not grouped:
            print("No valid relevance labels found.")
            print(f"Read 0 labeled rows from {args.annotations}")
            return 0

        print("## Per-Query Metrics\n")
        print(metrics_table(per_query_metrics(grouped, threshold=args.threshold), include_query=True))
        print("\n## Average Metrics\n")
        print(metrics_table(average_metrics(grouped, threshold=args.threshold), include_query=False))
        print(f"\nEvaluated {len(rows)} labeled rows from {args.annotations}")
        return 0
    except Exception as exc:
        print(f"Error evaluating semantic recommender: {exc}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="Minimum relevance value treated as relevant for Precision and MRR.",
    )
    return parser.parse_args()


def load_labeled_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        validate_columns(reader.fieldnames or [], path)
        rows = []
        for row in reader:
            relevance = relevance_value(row)
            if relevance is None:
                continue

            normalized = normalize_row(row)
            normalized["relevance"] = relevance
            rows.append(normalized)

    return rows


def validate_columns(fieldnames: list[str], path: Path) -> None:
    required = ["query_id", "method", "rank"]
    missing = [column for column in required if column not in fieldnames]
    if not any(column in fieldnames for column in ["human_relevance", "judge_relevance", "relevance"]):
        missing.append("human_relevance, judge_relevance, or relevance")
    if missing:
        raise ValueError(f"Annotation file {path} is missing required columns: {', '.join(missing)}")


def normalize_row(row: dict[str, Any]) -> dict:
    return {
        "query_id": clean(row.get("query_id")),
        "query": clean(row.get("query")),
        "method": normalize_method(clean(row.get("method"))),
        "rank": safe_int(row.get("rank")),
    }


def relevance_value(row: dict[str, Any]) -> int | None:
    for column in ["human_relevance", "relevance", "judge_relevance"]:
        value = clean(row.get(column))
        if not value:
            continue

        try:
            relevance = int(value)
        except ValueError:
            return None

        if 0 <= relevance <= 3:
            return relevance
        return None

    return None


def group_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if not row["method"] or row["rank"] <= 0:
            continue
        grouped[(row["method"], row["query_id"])].append(row)

    return grouped


def per_query_metrics(
    grouped: dict[tuple[str, str], list[dict]],
    threshold: int,
) -> list[dict]:
    metrics: list[dict] = []
    for method, query_id in sorted(grouped, key=lambda item: (_method_sort_key(item[0][0]), item[0][1])):
        ranked_rows = sorted(grouped[(method, query_id)], key=lambda row: row["rank"])
        relevances = [row["relevance"] for row in ranked_rows]
        query = next((row["query"] for row in ranked_rows if row["query"]), "")
        metrics.append(metric_row(method, relevances, threshold, query_id=query_id, query=query))

    return metrics


def average_metrics(
    grouped: dict[tuple[str, str], list[dict]],
    threshold: int,
) -> list[dict]:
    by_method: dict[str, list[list[int]]] = defaultdict(list)
    for method, query_id in grouped:
        ranked_rows = sorted(grouped[(method, query_id)], key=lambda row: row["rank"])
        by_method[method].append([row["relevance"] for row in ranked_rows])

    rows: list[dict] = []
    for method in sorted(by_method, key=_method_sort_key):
        relevance_lists = by_method[method]
        rows.append(
            {
                "method": display_method(method),
                "precision_5": mean([precision_at_k(values, k=5, threshold=threshold) for values in relevance_lists]),
                "precision_10": mean([precision_at_k(values, k=10, threshold=threshold) for values in relevance_lists]),
                "ndcg_5": mean([ndcg_at_k(values, k=5) for values in relevance_lists]),
                "ndcg_10": mean([ndcg_at_k(values, k=10) for values in relevance_lists]),
                "mrr": mean([reciprocal_rank(values, threshold=threshold) for values in relevance_lists]),
            }
        )

    return rows


def metric_row(method: str, relevances: list[int], threshold: int, query_id: str, query: str) -> dict:
    return {
        "query_id": query_id,
        "query": query,
        "method": display_method(method),
        "precision_5": precision_at_k(relevances, k=5, threshold=threshold),
        "precision_10": precision_at_k(relevances, k=10, threshold=threshold),
        "ndcg_5": ndcg_at_k(relevances, k=5),
        "ndcg_10": ndcg_at_k(relevances, k=10),
        "mrr": reciprocal_rank(relevances, threshold=threshold),
    }


def metrics_table(rows: list[dict], include_query: bool) -> str:
    if include_query:
        lines = [
            "| Query ID | Method | Precision@5 | Precision@10 | nDCG@5 | nDCG@10 | MRR |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {query_id} | {method} | {precision_5:.3f} | {precision_10:.3f} | "
                "{ndcg_5:.3f} | {ndcg_10:.3f} | {mrr:.3f} |".format(**row)
            )
        return "\n".join(lines)

    lines = [
        "| Method | Precision@5 | Precision@10 | nDCG@5 | nDCG@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {precision_5:.3f} | {precision_10:.3f} | "
            "{ndcg_5:.3f} | {ndcg_10:.3f} | {mrr:.3f} |".format(**row)
        )
    return "\n".join(lines)


def normalize_method(method: str) -> str:
    normalized = method.strip().lower().replace(" ", "_")
    if normalized in {"semantic_llm_rerank", "semantic+ollama_llm_reranking"}:
        return "semantic+llm_rerank"
    return normalized


def display_method(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method.replace("_", " ").title())


def _method_sort_key(method: str) -> tuple[int, str]:
    try:
        return METHOD_ORDER.index(method), method
    except ValueError:
        return len(METHOD_ORDER), method


def clean(value: Any) -> str:
    return str(value or "").strip()


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
