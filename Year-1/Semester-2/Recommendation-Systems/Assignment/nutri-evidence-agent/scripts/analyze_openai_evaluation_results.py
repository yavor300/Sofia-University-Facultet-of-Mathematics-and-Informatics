"""Analyze OpenAI-judged evaluation files and generate CSV/SVG report artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "evaluation"
DEFAULT_INPUTS = [
    PROJECT_ROOT / "data" / "evaluation_annotations_openai_q2.csv",
    PROJECT_ROOT / "data" / "evaluation_annotations_openai_q3.csv",
    PROJECT_ROOT / "data" / "evaluation_annotations_openai_q4.csv",
    PROJECT_ROOT / "data" / "evaluation_annotations_openai_q5.csv",
    PROJECT_ROOT / "data" / "evaluation_annotations_openai_q6.csv",
]

METHOD_LABELS = {
    "mesh_overlap": "MeSH overlap baseline",
    "semantic": "Semantic Similarity",
    "semantic+llm_rerank": "Semantic + LLM rerank",
    "graph": "Graph node2vec",
    "graph+llm_rerank": "Graph + LLM rerank",
    "hybrid": "Hybrid",
    "hybrid+llm_rerank": "Hybrid + LLM rerank",
}
METHOD_ORDER = [
    "mesh_overlap",
    "semantic",
    "semantic+llm_rerank",
    "graph",
    "graph+llm_rerank",
    "hybrid",
    "hybrid+llm_rerank",
]
METHOD_COLORS = {
    "mesh_overlap": "#64748b",
    "semantic": "#3b82f6",
    "semantic+llm_rerank": "#10b981",
    "graph": "#8b5cf6",
    "graph+llm_rerank": "#f59e0b",
    "hybrid": "#06b6d4",
    "hybrid+llm_rerank": "#ef4444",
}
METRICS = ["precision_5", "precision_10", "ndcg_5", "ndcg_10", "mrr"]
METRIC_LABELS = {
    "precision_5": "Precision@5",
    "precision_10": "Precision@10",
    "ndcg_5": "nDCG@5",
    "ndcg_10": "nDCG@10",
    "mrr": "MRR",
}


def main() -> int:
    args = parse_args()
    rows = load_rows(args.inputs)
    if not rows:
        raise SystemExit("No evaluated rows were loaded.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query = compute_per_query_metrics(rows, threshold=args.threshold)
    averages = compute_average_metrics(per_query)
    rank_movements = compute_rank_movements(rows)
    distributions = compute_relevance_distribution(rows)
    query_summary = compute_query_summary(rows)

    write_csv(args.output_dir / "openai_evaluation_per_query_metrics.csv", per_query)
    write_csv(args.output_dir / "openai_evaluation_average_metrics.csv", averages)
    write_csv(args.output_dir / "openai_evaluation_rank_movements.csv", rank_movements)
    write_csv(args.output_dir / "openai_evaluation_relevance_distribution.csv", distributions)
    write_csv(args.output_dir / "openai_evaluation_query_summary.csv", query_summary)

    write_average_metrics_svg(args.output_dir / "openai_evaluation_average_metrics.svg", averages)
    write_per_query_ndcg_svg(args.output_dir / "openai_evaluation_per_query_ndcg5.svg", per_query)
    write_relevance_distribution_svg(args.output_dir / "openai_evaluation_relevance_distribution.svg", distributions)

    print(f"Loaded {len(rows)} judged rows from {len(args.inputs)} files.")
    print(f"Wrote analysis artifacts to {args.output_dir}")
    print(markdown_table(averages))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=int, default=2)
    return parser.parse_args()


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for raw in reader:
                relevance = relevance_value(raw)
                if relevance is None:
                    continue
                rows.append(
                    {
                        "source_file": path.name,
                        "query_id": clean(raw.get("query_id")),
                        "query": clean(raw.get("query")),
                        "method": normalize_method(clean(raw.get("method"))),
                        "rank": safe_int(raw.get("rank")),
                        "pmid": clean(raw.get("pmid")),
                        "title": clean(raw.get("title")),
                        "year": clean(raw.get("year")),
                        "score": safe_float(raw.get("semantic_score")),
                        "relevance": relevance,
                        "judge_reason": clean(raw.get("judge_reason")),
                    }
                )
    return rows


def compute_per_query_metrics(rows: list[dict], threshold: int) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["query_id"], row["method"])].append(row)

    metrics: list[dict] = []
    for query_id, method in sorted(grouped, key=lambda item: (item[0][0], method_sort_key(item[0][1]))):
        ranked = sorted(grouped[(query_id, method)], key=lambda row: row["rank"])
        relevances = [row["relevance"] for row in ranked]
        query = ranked[0]["query"] if ranked else ""
        metrics.append(
            {
                "query_id": query_id,
                "query": query,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "precision_5": precision_at_k(relevances, 5, threshold),
                "precision_10": precision_at_k(relevances, 10, threshold),
                "ndcg_5": ndcg_at_k(relevances, 5),
                "ndcg_10": ndcg_at_k(relevances, 10),
                "mrr": reciprocal_rank(relevances, threshold),
                "mean_relevance": sum(relevances) / len(relevances) if relevances else 0.0,
                "relevant_count": sum(1 for value in relevances if value >= threshold),
                "evaluated_rows": len(relevances),
            }
        )

    return metrics


def compute_average_metrics(per_query: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in per_query:
        grouped[row["method"]].append(row)

    averages: list[dict] = []
    for method in sorted(grouped, key=method_sort_key):
        rows = grouped[method]
        averages.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "precision_5": mean([row["precision_5"] for row in rows]),
                "precision_10": mean([row["precision_10"] for row in rows]),
                "ndcg_5": mean([row["ndcg_5"] for row in rows]),
                "ndcg_10": mean([row["ndcg_10"] for row in rows]),
                "mrr": mean([row["mrr"] for row in rows]),
                "mean_relevance": mean([row["mean_relevance"] for row in rows]),
                "queries": len(rows),
            }
        )

    return averages


def compute_rank_movements(rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["query_id"]][row["method"]].append(row)

    movements: list[dict] = []
    for query_id, by_method in sorted(grouped.items()):
        for rerank_method, rerank_rows in by_method.items():
            if not rerank_method.endswith("+llm_rerank"):
                continue

            base_method = rerank_method.removesuffix("+llm_rerank")
            base_by_pmid = {row["pmid"]: row for row in by_method.get(base_method, [])}
            for llm_row in sorted(rerank_rows, key=lambda row: row["rank"]):
                base_row = base_by_pmid.get(llm_row["pmid"])
                if not base_row:
                    continue
                movements.append(
                    {
                        "query_id": query_id,
                        "base_method": base_method,
                        "rerank_method": rerank_method,
                        "pmid": llm_row["pmid"],
                        "title": llm_row["title"],
                        "base_rank": base_row["rank"],
                        "llm_rank": llm_row["rank"],
                        "rank_change": base_row["rank"] - llm_row["rank"],
                        "judge_relevance": llm_row["relevance"],
                        "algorithmic_score": llm_row["score"],
                    }
                )

    return movements


def compute_relevance_distribution(rows: list[dict]) -> list[dict]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        grouped[row["method"]][row["relevance"]] += 1

    output: list[dict] = []
    for method in sorted(grouped, key=method_sort_key):
        total = sum(grouped[method].values())
        for relevance in range(4):
            count = grouped[method][relevance]
            output.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "relevance": relevance,
                    "count": count,
                    "share": count / total if total else 0.0,
                }
            )

    return output


def compute_query_summary(rows: list[dict]) -> list[dict]:
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)

    output: list[dict] = []
    for query_id, query_rows in sorted(by_query.items()):
        query = query_rows[0]["query"]
        output.append(
            {
                "query_id": query_id,
                "query": query,
                "rows": len(query_rows),
                "methods": ", ".join(sorted({row["method"] for row in query_rows}, key=method_sort_key)),
                "mean_relevance": sum(row["relevance"] for row in query_rows) / len(query_rows),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(format_row(row) for row in rows)


def write_average_metrics_svg(path: Path, averages: list[dict]) -> None:
    width, height = 920, 580
    margin_left, margin_bottom, margin_top = 90, 90, 55
    chart_w = width - margin_left - 40
    chart_h = height - margin_top - margin_bottom
    metrics = METRICS
    methods = [row["method"] for row in sorted(averages, key=lambda row: method_sort_key(row["method"]))]
    group_w = chart_w / len(metrics)
    bar_w = group_w / (len(methods) + 1)
    by_method = {row["method"]: row for row in averages}

    parts = svg_header(width, height)
    parts.append(text(width / 2, 28, "Средни evaluation метрики", 20, "middle", bold=True))
    draw_axes(parts, margin_left, margin_top, chart_w, chart_h)
    draw_y_grid(parts, margin_left, margin_top, chart_w, chart_h)

    for i, metric in enumerate(metrics):
        x0 = margin_left + i * group_w
        for j, method in enumerate(methods):
            value = float(by_method.get(method, {}).get(metric, 0.0))
            bar_h = value * chart_h
            x = x0 + group_w / 2 - (len(methods) * bar_w / 2) + j * bar_w
            y = margin_top + chart_h - bar_h
            parts.append(rect(x, y, bar_w * 0.85, bar_h, METHOD_COLORS.get(method, "#9ca3af")))
            if value > 0:
                parts.append(text(x + bar_w * 0.42, y - 6, f"{value:.2f}", 9, "middle"))
        parts.append(text(x0 + group_w / 2, height - 45, METRIC_LABELS[metric], 13, "middle"))

    legend_y = height - 68
    for i, method in enumerate(methods):
        x = margin_left + (i % 3) * 280
        y = legend_y + (i // 3) * 18
        parts.append(rect(x, y - 12, 16, 12, METHOD_COLORS.get(method, "#9ca3af")))
        parts.append(text(x + 22, y - 2, METHOD_LABELS.get(method, method), 12, "start"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_per_query_ndcg_svg(path: Path, per_query: list[dict]) -> None:
    width, height = 980, 520
    margin_left, margin_bottom, margin_top = 90, 85, 55
    chart_w = width - margin_left - 40
    chart_h = height - margin_top - margin_bottom
    query_ids = sorted({row["query_id"] for row in per_query})
    methods = sorted({row["method"] for row in per_query}, key=method_sort_key)
    group_w = chart_w / len(query_ids)
    bar_w = group_w / (len(methods) + 1)
    lookup = {(row["query_id"], row["method"]): row for row in per_query}

    parts = svg_header(width, height)
    parts.append(text(width / 2, 28, "nDCG@5 по query", 20, "middle", bold=True))
    draw_axes(parts, margin_left, margin_top, chart_w, chart_h)
    draw_y_grid(parts, margin_left, margin_top, chart_w, chart_h)
    for i, query_id in enumerate(query_ids):
        x0 = margin_left + i * group_w
        for j, method in enumerate(methods):
            value = float(lookup.get((query_id, method), {}).get("ndcg_5", 0.0))
            bar_h = value * chart_h
            x = x0 + group_w / 2 - (len(methods) * bar_w / 2) + j * bar_w
            y = margin_top + chart_h - bar_h
            parts.append(rect(x, y, bar_w * 0.85, bar_h, METHOD_COLORS.get(method, "#9ca3af")))
            if value > 0:
                parts.append(text(x + bar_w * 0.42, y - 6, f"{value:.2f}", 9, "middle"))
        parts.append(text(x0 + group_w / 2, height - 45, query_id, 14, "middle", bold=True))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_relevance_distribution_svg(path: Path, distributions: list[dict]) -> None:
    width, height = 860, 460
    margin_left, margin_top = 80, 55
    chart_w, chart_h = width - margin_left - 40, height - margin_top - 80
    colors = {0: "#ef4444", 1: "#f59e0b", 2: "#60a5fa", 3: "#22c55e"}
    by_method: dict[str, dict[int, int]] = defaultdict(dict)
    max_count = 1
    for row in distributions:
        count = int(row["count"])
        by_method[row["method"]][int(row["relevance"])] = count
        max_count = max(max_count, count)

    parts = svg_header(width, height)
    parts.append(text(width / 2, 28, "Разпределение на OpenAI Judge relevance labels", 20, "middle", bold=True))
    draw_axes(parts, margin_left, margin_top, chart_w, chart_h, max_y=max_count)
    methods = sorted(by_method, key=method_sort_key)
    group_w = chart_w / len(methods)
    bar_w = group_w / 5
    for i, method in enumerate(methods):
        x0 = margin_left + i * group_w
        for relevance in range(4):
            count = by_method[method].get(relevance, 0)
            bar_h = count / max_count * chart_h
            x = x0 + group_w / 2 - 2 * bar_w + relevance * bar_w
            y = margin_top + chart_h - bar_h
            parts.append(rect(x, y, bar_w * 0.82, bar_h, colors[relevance]))
            parts.append(text(x + bar_w * 0.4, y - 5, str(count), 11, "middle"))
        parts.append(text(x0 + group_w / 2, height - 50, METHOD_LABELS[method], 13, "middle"))
    legend_y = height - 20
    for relevance in range(4):
        x = margin_left + relevance * 150
        parts.append(rect(x, legend_y - 12, 16, 12, colors[relevance]))
        parts.append(text(x + 22, legend_y - 2, f"label {relevance}", 13, "start"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def precision_at_k(relevances: list[int], k: int, threshold: int) -> float:
    values = relevances[:k]
    return sum(1 for value in values if value >= threshold) / len(values) if values else 0.0


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(((2**value) - 1) / math.log2(rank + 1) for rank, value in enumerate(relevances[:k], start=1))


def ndcg_at_k(relevances: list[int], k: int) -> float:
    values = relevances[:k]
    if not values:
        return 0.0
    ideal = sorted(values, reverse=True)
    ideal_dcg = dcg_at_k(ideal, len(ideal))
    return dcg_at_k(values, len(values)) / ideal_dcg if ideal_dcg else 0.0


def reciprocal_rank(relevances: list[int], threshold: int) -> float:
    for index, relevance in enumerate(relevances, start=1):
        if relevance >= threshold:
            return 1 / index
    return 0.0


def relevance_value(row: dict[str, Any]) -> int | None:
    for column in ["human_relevance", "relevance", "judge_relevance"]:
        raw = clean(row.get(column))
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if 0 <= value <= 3 else None
    return None


def normalize_method(method: str) -> str:
    normalized = method.strip().lower().replace(" ", "_")
    if normalized in {"semantic_llm_rerank", "semantic+ollama_llm_reranking"}:
        return "semantic+llm_rerank"
    return normalized


def method_sort_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def clean(value: Any) -> str:
    return str(value or "").strip()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_row(row: dict) -> dict:
    formatted = {}
    for key, value in row.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.4f}"
        else:
            formatted[key] = value
    return formatted


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Method | Precision@5 | Precision@10 | nDCG@5 | nDCG@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_label']} | {row['precision_5']:.3f} | {row['precision_10']:.3f} | "
            f"{row['ndcg_5']:.3f} | {row['ndcg_10']:.3f} | {row['mrr']:.3f} |"
        )
    return "\n".join(lines)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#374151;stroke-width:1.3}</style>',
    ]


def draw_axes(parts: list[str], x: float, y: float, width: float, height: float, max_y: int | float = 1.0) -> None:
    parts.append(f'<line class="axis" x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y + height:.1f}"/>')
    parts.append(f'<line class="axis" x1="{x:.1f}" y1="{y + height:.1f}" x2="{x + width:.1f}" y2="{y + height:.1f}"/>')
    for i in range(6):
        value = max_y * i / 5
        y_pos = y + height - height * i / 5
        label = f"{value:.1f}" if max_y <= 1 else str(round(value))
        parts.append(text(x - 10, y_pos + 4, label, 11, "end"))


def draw_y_grid(parts: list[str], x: float, y: float, width: float, height: float) -> None:
    for i in range(1, 6):
        y_pos = y + height - height * i / 5
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{y_pos:.1f}" x2="{x + width:.1f}" y2="{y_pos:.1f}"/>')


def rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="2" fill="{fill}"/>'


def text(x: float, y: float, content: str, size: int, anchor: str = "start", bold: bool = False) -> str:
    weight = " font-weight=\"700\"" if bold else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"{weight}>{html.escape(content)}</text>'


if __name__ == "__main__":
    raise SystemExit(main())
