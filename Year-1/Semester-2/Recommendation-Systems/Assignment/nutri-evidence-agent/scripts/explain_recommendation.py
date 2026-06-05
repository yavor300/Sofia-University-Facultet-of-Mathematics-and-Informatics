"""Explain a cached article recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.graph_builder import ArticleMeshGraphBuilder
from src.graph.node2vec_trainer import Node2VecTrainer
from src.agents.recommendation_explainer import RecommendationExplainer
from src.llm.ollama_client import OllamaClient
from src.recommenders.hybrid_recommender import HybridRecommender
from src.recommenders.mesh_overlap_recommender import MeshOverlapRecommender
from src.recommenders.graph_recommender import GraphRecommender
from src.recommenders.semantic_recommender import SemanticRecommender
from src.utils.config import load_settings


DEFAULT_ARTICLES_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"


def main() -> None:
    args = parse_args()
    articles = load_articles(args.articles)
    seed_article = find_article(articles, args.pmid)
    recommendations = recommend(articles, args)
    if not recommendations:
        print(json.dumps({"error": "No recommendation available to explain."}, indent=2))
        return

    rank_index = max(0, args.rank - 1)
    if rank_index >= len(recommendations):
        raise ValueError(f"Rank {args.rank} is outside the available {len(recommendations)} recommendations.")

    recommended_article = recommendations[rank_index]
    settings = load_settings()
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=args.timeout,
    )
    explainer = RecommendationExplainer(client, use_llm=not args.no_llm and settings.use_llm)
    explanation = explainer.explain(seed_article, recommended_article)

    print(
        json.dumps(
            {
                "seed_pmid": args.pmid,
                "recommended_pmid": recommended_article.get("pmid"),
                "method": recommended_article.get("method"),
                "rank": args.rank,
                "explanation": explanation,
                "recommendation": recommended_article,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmid", required=True, help="Seed article PMID.")
    parser.add_argument("--method", choices=["semantic", "mesh", "graph", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rank", type=int, default=1, help="1-based recommendation rank to explain.")
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-llm", action="store_true", help="Use fallback explanation without calling Ollama.")
    parser.add_argument("--semantic-embeddings", type=Path, default=PROJECT_ROOT / "data" / "artifacts" / "article_embeddings.npy")
    parser.add_argument("--semantic-index", type=Path, default=PROJECT_ROOT / "data" / "artifacts" / "article_embedding_index.json")
    parser.add_argument("--graph", type=Path, default=PROJECT_ROOT / "data" / "artifacts" / "article_mesh_graph.gpickle")
    parser.add_argument("--node2vec", type=Path, default=PROJECT_ROOT / "data" / "artifacts" / "node2vec_embeddings.kv")
    return parser.parse_args()


def recommend(articles: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.method == "semantic":
        recommender = load_semantic_recommender(articles, args)
        return recommender.recommend_by_article(args.pmid, top_k=args.top_k)
    if args.method == "mesh":
        recommender = MeshOverlapRecommender()
        recommender.fit(articles)
        return recommender.recommend_by_article(args.pmid, top_k=args.top_k)
    if args.method == "graph":
        recommender = load_graph_recommender(articles, args)
        return recommender.recommend_by_article(args.pmid, top_k=args.top_k)

    semantic = load_semantic_recommender(articles, args)
    graph = load_graph_recommender(articles, args) if args.graph.exists() and args.node2vec.exists() else None
    recommender = HybridRecommender(semantic, graph)
    return recommender.recommend_by_article(args.pmid, top_k=args.top_k)


def find_article(articles: list[dict], pmid: str) -> dict | None:
    requested_pmid = str(pmid).strip()
    for article in articles:
        if str(article.get("pmid", "")).strip() == requested_pmid:
            return article

    return None


def load_articles(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Article cache must contain a JSON list: {path}")

    return [article for article in data if isinstance(article, dict)]


def load_semantic_recommender(articles: list[dict], args: argparse.Namespace) -> SemanticRecommender:
    recommender = SemanticRecommender()
    recommender.load_artifacts(
        articles,
        str(args.semantic_embeddings),
        str(args.semantic_index),
    )
    return recommender


def load_graph_recommender(articles: list[dict], args: argparse.Namespace) -> GraphRecommender:
    graph = ArticleMeshGraphBuilder().load(str(args.graph))
    node2vec_model = Node2VecTrainer().load(str(args.node2vec))
    recommender = GraphRecommender()
    recommender.fit(articles, graph, node2vec_model)
    return recommender


if __name__ == "__main__":
    main()
