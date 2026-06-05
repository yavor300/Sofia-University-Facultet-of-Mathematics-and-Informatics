"""Evaluate recommender annotations and print a Markdown metrics table."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k


DEFAULT_ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "evaluation_annotations.csv"
METHOD_DISPLAY_NAMES = {
    "mesh": "MeSH Overlap",
    "mesh_overlap": "MeSH Overlap",
    "semantic": "Semantic",
    "semantic_llm_rerank": "Semantic + Ollama LLM Reranking",
    "graph": "Graph node2vec",
    "graph_node2vec": "Graph node2vec",
    "graph_llm_rerank": "Graph node2vec + Ollama LLM Reranking",
    "hybrid": "Hybrid",
    "hybrid_llm_rerank": "Hybrid + Ollama LLM Reranking",
}
DEFAULT_METHOD_ORDER = [
    "mesh_overlap",
    "semantic",
    "semantic_llm_rerank",
    "graph",
    "graph_llm_rerank",
    "hybrid",
    "hybrid_llm_rerank",
]


def main() -> int:
    try:
        args = parse_args()
        rows = load_annotations(args.annotations)
        grouped_relevances = group_relevances(rows)
        print(markdown_table(grouped_relevances, k=args.k, threshold=args.threshold))
        print(f"\nEvaluated {len(rows)} annotated rows from {args.annotations}")
        return 0
    except Exception as exc:
        print(f"Error evaluating recommenders: {exc}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="Minimum relevance value treated as relevant for Precision and MRR.",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        _validate_columns(reader.fieldnames or [], path)
        return [row for row in reader if _has_annotation(row)]


def group_relevances(rows: list[dict]) -> dict[str, list[list[int]]]:
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)

    for row in rows:
        method = _normalize_method(row["method"])
        query_key = _query_key(row)
        grouped[(method, query_key)].append((_safe_int(row["rank"]), _safe_int(_relevance_value(row))))

    method_relevances: dict[str, list[list[int]]] = defaultdict(list)
    for method, query_key in sorted(grouped):
        ranked_rows = sorted(grouped[(method, query_key)], key=lambda item: item[0])
        method_relevances[method].append([relevance for _, relevance in ranked_rows])

    return method_relevances


def markdown_table(grouped_relevances: dict[str, list[list[int]]], k: int = 5, threshold: int = 1) -> str:
    lines = [
        f"| Method | Precision@{k} | nDCG@{k} | MRR |",
        "|---|---:|---:|---:|",
    ]

    for method in DEFAULT_METHOD_ORDER:
        relevance_lists = grouped_relevances.get(method, [])
        precision = _mean([precision_at_k(relevances, k=k, threshold=threshold) for relevances in relevance_lists])
        ndcg = _mean([ndcg_at_k(relevances, k=k) for relevances in relevance_lists])
        mrr = mean_reciprocal_rank(relevance_lists, threshold=threshold)
        lines.append(f"| {METHOD_DISPLAY_NAMES[method]} | {precision:.3f} | {ndcg:.3f} | {mrr:.3f} |")

    return "\n".join(lines)


def _validate_columns(fieldnames: list[str], path: Path) -> None:
    expected = ["method", "rank"]
    missing = [column for column in expected if column not in fieldnames]
    if "query_id" not in fieldnames and "seed_pmid" not in fieldnames:
        missing.append("query_id or seed_pmid")
    if not any(column in fieldnames for column in ["relevance", "judge_relevance", "human_relevance"]):
        missing.append("relevance, judge_relevance, or human_relevance")
    if missing:
        raise ValueError(f"Annotation file {path} is missing required columns: {', '.join(missing)}")


def _has_annotation(row: dict) -> bool:
    return bool(row.get("method", "").strip() and row.get("rank", "").strip() and _relevance_value(row).strip())


def _relevance_value(row: dict) -> str:
    """Prefer human labels, then manual relevance, then OpenAI judge labels."""
    for column in ["human_relevance", "relevance", "judge_relevance"]:
        value = row.get(column, "")
        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def _normalize_method(method: str) -> str:
    normalized = method.strip().lower().replace("-", "_").replace(" ", "_").replace("+", "_")
    if normalized in {"mesh", "mesh_overlap", "meshoverlap"}:
        return "mesh_overlap"
    if normalized in {"graph", "graph_node2vec", "node2vec"}:
        return "graph"
    if normalized in {
        "semantic_llm_rerank",
        "semantic_ollama_llm_reranking",
        "semantic_llm_reranking",
        "semantic_ollama_rerank",
    }:
        return "semantic_llm_rerank"
    if normalized in {
        "graph_llm_rerank",
        "graph_node2vec_llm_rerank",
        "graph_ollama_llm_reranking",
        "graph_llm_reranking",
        "graph_ollama_rerank",
    }:
        return "graph_llm_rerank"
    if normalized in {
        "hybrid_llm_rerank",
        "hybrid_ollama_llm_reranking",
        "hybrid_llm_reranking",
        "hybrid_ollama_rerank",
    }:
        return "hybrid_llm_rerank"
    if normalized in {"semantic", "hybrid"}:
        return normalized

    return normalized


def _query_key(row: dict) -> str:
    query_id = row.get("query_id", "").strip()
    if query_id:
        return query_id

    return row.get("seed_pmid", "").strip()


def _safe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
