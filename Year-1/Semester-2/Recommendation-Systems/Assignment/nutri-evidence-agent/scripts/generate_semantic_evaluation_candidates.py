"""Generate semantic evaluation candidate rows.

The default mode builds Top 10 semantic recommendations from
data/evaluation_queries.json. The export mode converts Streamlit-exported
"before reranking" and "after reranking" tables into the same evaluation CSV
shape so they can be judged side by side.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ARTICLES_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"
DEFAULT_EVALUATION_QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation_queries.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation_candidates_semantic.csv"
DEFAULT_SEMANTIC_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embeddings.npy"
DEFAULT_SEMANTIC_INDEX_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embedding_index.json"
FIELDNAMES = [
    "query_id",
    "query",
    "method",
    "rank",
    "pmid",
    "title",
    "year",
    "journal",
    "semantic_score",
    "relevance",
]


def main() -> int:
    try:
        args = parse_args()
        query = resolve_query(args)
        if args.exports:
            rows = rows_from_exports(args.exports, query_id=args.query_id, query=query, top_k=args.top_k)
        else:
            rows = rows_from_semantic_recommender(args)

        if not rows:
            raise ValueError("No evaluation candidate rows were generated.")

        write_rows(rows, args.output, overwrite=args.overwrite)
        print(f"Wrote {len(rows)} evaluation candidate rows to {args.output}")
        print("Fill the relevance column with 0, 1, 2, or 3 before computing metrics.")
        return 0
    except Exception as exc:
        print(f"Error generating semantic evaluation candidates: {exc}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_EVALUATION_QUERIES_PATH)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--semantic-embeddings", type=Path, default=DEFAULT_SEMANTIC_EMBEDDINGS_PATH)
    parser.add_argument("--semantic-index", type=Path, default=DEFAULT_SEMANTIC_INDEX_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true", help="Replace the output CSV instead of appending.")
    parser.add_argument(
        "--exports",
        type=Path,
        nargs="*",
        help="Streamlit recommendation export CSVs to normalize instead of generating new recommendations.",
    )
    parser.add_argument("--query-id", default="q1", help="Query id used when importing Streamlit exports.")
    parser.add_argument(
        "--query",
        default=None,
        help="Original user query used when importing Streamlit exports.",
    )
    parser.add_argument(
        "--seed-pmid",
        help="Seed article PMID for article-based recommendation exports. Builds an evaluation query from cached metadata.",
    )
    return parser.parse_args()


def resolve_query(args: argparse.Namespace) -> str:
    if args.query:
        return str(args.query).strip()

    if args.seed_pmid:
        seed_article = load_article_by_pmid(args.articles, args.seed_pmid)
        title = str(seed_article.get("title", "")).strip()
        if title:
            return f"Find articles similar to PMID {args.seed_pmid}: {title}"
        return f"Find articles similar to PMID {args.seed_pmid}."

    return "What is the evidence linking gut microbiome and Parkinson's disease?"


def load_article_by_pmid(path: Path, pmid: str) -> dict:
    from src.retrieval.cache import load_articles

    normalized_pmid = str(pmid).strip()
    for article in load_articles(str(path)):
        if str(article.get("pmid", "")).strip() == normalized_pmid:
            return article

    raise ValueError(f"Seed PMID {pmid} was not found in {path}")


def rows_from_semantic_recommender(args: argparse.Namespace) -> list[dict]:
    from src.recommenders.semantic_recommender import SemanticRecommender
    from src.retrieval.cache import load_articles

    articles = load_articles(str(args.articles))
    if not articles:
        raise ValueError(f"No cached articles found in {args.articles}")

    queries = load_evaluation_queries(args.queries)
    recommender = SemanticRecommender()
    if args.semantic_embeddings.exists() and args.semantic_index.exists():
        recommender.load_artifacts(articles, str(args.semantic_embeddings), str(args.semantic_index))
    else:
        recommender.fit(articles)
        recommender.save_artifacts(str(args.semantic_embeddings), str(args.semantic_index))

    rows: list[dict] = []
    for query_item in queries:
        query_id = str(query_item["query_id"]).strip()
        query = str(query_item["query"]).strip()
        recommendations = recommender.recommend_by_query(query, top_k=args.top_k)
        for rank, article in enumerate(recommendations, start=1):
            rows.append(candidate_row(query_id, query, article, rank, method="semantic"))

    return rows


def load_evaluation_queries(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation query file not found: {path}. "
            "Create data/evaluation_queries.json or use --exports with Streamlit CSV exports."
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        queries = data
    elif isinstance(data, dict) and isinstance(data.get("queries"), list):
        queries = data["queries"]
    else:
        raise ValueError("Evaluation queries must be a list or an object with a queries list.")

    normalized: list[dict] = []
    for index, item in enumerate(queries, start=1):
        if isinstance(item, str):
            query = item.strip()
            query_id = f"q{index}"
        elif isinstance(item, dict):
            query = str(item.get("query") or item.get("question") or "").strip()
            query_id = str(item.get("query_id") or item.get("id") or f"q{index}").strip()
        else:
            continue

        if query:
            normalized.append({"query_id": query_id, "query": query})

    if not normalized:
        raise ValueError(f"No usable evaluation queries found in {path}")

    return normalized


def rows_from_exports(exports: list[Path], query_id: str, query: str, top_k: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for export in exports:
        with export.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                method = normalize_method(cell(row, "Method"))
                rank = safe_int(cell(row, "Rank"))
                pmid = cell(row, "PMID")
                if not method or not pmid or rank <= 0 or rank > top_k:
                    continue

                key = (query_id, method, pmid)
                if key in seen:
                    continue
                seen.add(key)

                rows.append(
                    {
                        "query_id": query_id,
                        "query": query,
                        "method": method,
                        "rank": rank,
                        "pmid": pmid,
                        "title": cell(row, "Title"),
                        "year": cell(row, "Year"),
                        "journal": cell(row, "Journal"),
                        "semantic_score": cell(row, "Algorithmic Score"),
                        "relevance": "",
                    }
                )

    return sorted(rows, key=lambda item: (item["method"], int(item["rank"])))


def candidate_row(query_id: str, query: str, article: dict[str, Any], rank: int, method: str) -> dict:
    return {
        "query_id": query_id,
        "query": query,
        "method": method,
        "rank": rank,
        "pmid": str(article.get("pmid", "")).strip(),
        "title": str(article.get("title", "")).strip(),
        "year": article.get("year", ""),
        "journal": str(article.get("journal", "")).strip(),
        "semantic_score": _format_score(article.get("score")),
        "relevance": "",
    }


def write_rows(rows: list[dict], output: Path, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    existing_keys: set[tuple[str, str, str, str]] = set()

    if output.exists() and not overwrite:
        with output.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            existing_rows = [{field: row.get(field, "") for field in FIELDNAMES} for row in reader]
            existing_keys = {_row_key(row) for row in existing_rows}

    new_rows = [row for row in rows if _row_key(row) not in existing_keys]
    output_rows = [] if overwrite else existing_rows
    output_rows.extend(new_rows)

    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)


def cell(row: dict[str, Any], column: str) -> str:
    for key, value in row.items():
        if key.lstrip("\ufeff").strip() == column:
            return str(value or "").strip()
    return ""


def normalize_method(method: str) -> str:
    normalized = method.strip().lower().replace(" ", "_")
    if normalized in {"semantic", "semantic+llm_rerank"}:
        return normalized
    if normalized in {"semantic_llm_rerank", "semantic+ollama_llm_reranking"}:
        return "semantic+llm_rerank"
    return normalized


def safe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_score(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("query_id", "")),
        str(row.get("method", "")),
        str(row.get("rank", "")),
        str(row.get("pmid", "")),
    )


if __name__ == "__main__":
    raise SystemExit(main())
