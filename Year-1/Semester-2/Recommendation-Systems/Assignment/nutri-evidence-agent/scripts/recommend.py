"""Run article recommendations from cached NutriEvidence artifacts."""

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
from src.recommenders.graph_recommender import GraphRecommender
from src.recommenders.hybrid_recommender import HybridRecommender
from src.recommenders.mesh_overlap_recommender import MeshOverlapRecommender
from src.recommenders.semantic_recommender import SemanticRecommender


DEFAULT_ARTICLES_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"
DEFAULT_SEMANTIC_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embeddings.npy"
DEFAULT_SEMANTIC_INDEX_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embedding_index.json"
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_mesh_graph.gpickle"
DEFAULT_NODE2VEC_PATH = PROJECT_ROOT / "data" / "artifacts" / "node2vec_embeddings.kv"


def main() -> None:
    args = parse_args()
    articles = load_articles(args.articles)

    if args.method == "semantic":
        recommender = load_semantic_recommender(articles, args)
        results = recommender.recommend_by_article(args.pmid, top_k=args.top_k)
    elif args.method == "mesh":
        recommender = MeshOverlapRecommender()
        recommender.fit(articles)
        results = recommender.recommend_by_article(args.pmid, top_k=args.top_k)
    elif args.method == "graph":
        recommender = load_graph_recommender(articles, args)
        results = recommender.recommend_by_article(args.pmid, top_k=args.top_k)
    else:
        semantic = load_semantic_recommender(articles, args)
        graph = None
        if args.graph.exists() and args.node2vec.exists():
            graph = load_graph_recommender(articles, args)
        recommender = HybridRecommender(semantic, graph)
        results = recommender.recommend_by_article(args.pmid, top_k=args.top_k)

    print(json.dumps(results, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmid", required=True, help="Seed article PMID.")
    parser.add_argument("--method", choices=["semantic", "mesh", "graph", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--semantic-embeddings", type=Path, default=DEFAULT_SEMANTIC_EMBEDDINGS_PATH)
    parser.add_argument("--semantic-index", type=Path, default=DEFAULT_SEMANTIC_INDEX_PATH)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--node2vec", type=Path, default=DEFAULT_NODE2VEC_PATH)
    return parser.parse_args()


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
