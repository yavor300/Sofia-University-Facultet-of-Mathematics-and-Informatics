"""Generate blank relevance annotation rows from recommender outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.llm_recommender import LLMRecommendationReranker
from src.graph.graph_builder import ArticleMeshGraphBuilder
from src.graph.node2vec_trainer import Node2VecTrainer
from src.llm.ollama_client import OllamaClient
from src.recommenders.graph_recommender import GraphRecommender
from src.recommenders.hybrid_recommender import HybridRecommender
from src.recommenders.mesh_overlap_recommender import MeshOverlapRecommender
from src.recommenders.semantic_recommender import SemanticRecommender
from src.retrieval.cache import load_articles
from src.utils.config import load_settings


DEFAULT_ARTICLES_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"
DEFAULT_ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "evaluation_annotations.csv"
DEFAULT_SEMANTIC_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embeddings.npy"
DEFAULT_SEMANTIC_INDEX_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embedding_index.json"
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_mesh_graph.gpickle"
DEFAULT_NODE2VEC_PATH = PROJECT_ROOT / "data" / "artifacts" / "node2vec_embeddings.kv"
FIELDNAMES = ["query_id", "seed_pmid", "recommended_pmid", "method", "rank", "relevance"]


def main() -> int:
    try:
        args = parse_args()
        articles = load_articles(str(args.articles))
        if not articles:
            raise ValueError(f"No articles found in {args.articles}")

        rows = build_annotation_rows(articles, args)
        if not rows:
            raise ValueError("No annotation rows were generated.")

        write_rows(rows, args.output, overwrite=args.overwrite)
        print(f"Wrote {len(rows)} annotation template rows to {args.output}")
        print("Fill the relevance column with 0, 1, 2, or 3 before running evaluation.")
        return 0
    except Exception as exc:
        print(f"Error generating annotation template: {exc}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-pmid", required=True, help="Seed PMID for article-based recommenders.")
    parser.add_argument("--query-id", help="Evaluation query id. Defaults to seed_<PMID>.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["mesh_overlap", "semantic", "graph", "hybrid", "hybrid+llm_rerank"],
        default=["mesh_overlap", "semantic", "graph", "hybrid"],
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output CSV instead of appending new rows.")
    parser.add_argument("--use-llm", action="store_true", help="Call Ollama for hybrid+llm_rerank rows.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--semantic-embeddings", type=Path, default=DEFAULT_SEMANTIC_EMBEDDINGS_PATH)
    parser.add_argument("--semantic-index", type=Path, default=DEFAULT_SEMANTIC_INDEX_PATH)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--node2vec", type=Path, default=DEFAULT_NODE2VEC_PATH)
    return parser.parse_args()


def build_annotation_rows(articles: list[dict], args: argparse.Namespace) -> list[dict]:
    query_id = args.query_id or f"seed_{args.seed_pmid}"
    rows: list[dict] = []

    semantic = None
    graph = None
    for method in args.methods:
        if method in {"semantic", "hybrid", "hybrid+llm_rerank"} and semantic is None:
            semantic = load_semantic_recommender(articles, args)
        if method in {"graph", "hybrid", "hybrid+llm_rerank"} and graph is None:
            graph = load_graph_recommender_if_available(articles, args)

        results = recommend(method, args.seed_pmid, args.top_k, articles, semantic, graph, args)
        for rank, result in enumerate(results, start=1):
            rows.append(
                {
                    "query_id": query_id,
                    "seed_pmid": args.seed_pmid,
                    "recommended_pmid": result.get("pmid", ""),
                    "method": method,
                    "rank": rank,
                    "relevance": "",
                }
            )

    return rows


def recommend(
    method: str,
    seed_pmid: str,
    top_k: int,
    articles: list[dict],
    semantic,
    graph,
    args: argparse.Namespace,
) -> list[dict]:
    if method == "mesh_overlap":
        recommender = MeshOverlapRecommender()
        recommender.fit(articles)
        return recommender.recommend_by_article(seed_pmid, top_k=top_k)

    if method == "semantic":
        return semantic.recommend_by_article(seed_pmid, top_k=top_k)

    if method == "graph":
        if graph is None:
            print("Skipping graph rows because graph artifacts are unavailable.", file=sys.stderr)
            return []
        return graph.recommend_by_article(seed_pmid, top_k=top_k)

    hybrid = HybridRecommender(semantic, graph)
    hybrid_results = hybrid.recommend_by_article(seed_pmid, top_k=max(top_k, 10))
    if method == "hybrid":
        return hybrid_results[:top_k]

    settings = load_settings()
    client = None
    if args.use_llm:
        client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout=args.timeout,
        )

    return LLMRecommendationReranker(client).rerank(
        user_question=f"Find papers similar to PMID {seed_pmid}.",
        candidate_articles=hybrid_results,
        top_k=top_k,
    )


def load_semantic_recommender(articles: list[dict], args: argparse.Namespace) -> SemanticRecommender:
    recommender = SemanticRecommender()
    recommender.load_artifacts(
        articles,
        str(args.semantic_embeddings),
        str(args.semantic_index),
    )
    return recommender


def load_graph_recommender_if_available(articles: list[dict], args: argparse.Namespace) -> GraphRecommender | None:
    if not args.graph.exists() or not args.node2vec.exists():
        return None

    graph = ArticleMeshGraphBuilder().load(str(args.graph))
    node2vec_model = Node2VecTrainer().load(str(args.node2vec))
    recommender = GraphRecommender()
    recommender.fit(articles, graph, node2vec_model)
    return recommender


def write_rows(rows: list[dict], output: Path, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = set()
    existing_rows: list[dict] = []

    if output.exists() and not overwrite:
        with output.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            existing_rows = [row for row in reader if row]
            existing_keys = {_row_key(row) for row in existing_rows}

    new_rows = [row for row in rows if _row_key(row) not in existing_keys]
    mode_rows = [] if overwrite else existing_rows
    mode_rows.extend(new_rows)

    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(mode_rows)


def _row_key(row: dict) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("query_id", "")),
        str(row.get("seed_pmid", "")),
        str(row.get("recommended_pmid", "")),
        str(row.get("method", "")),
        str(row.get("rank", "")),
    )


if __name__ == "__main__":
    raise SystemExit(main())
