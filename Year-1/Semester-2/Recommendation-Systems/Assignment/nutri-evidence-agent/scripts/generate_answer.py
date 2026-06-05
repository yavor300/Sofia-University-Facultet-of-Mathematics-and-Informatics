"""Generate a final Markdown answer from cached recommendations and evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.answer_generator import AnswerGenerator
from src.agents.evidence_extractor import EvidenceExtractionAgent
from src.graph.graph_builder import ArticleMeshGraphBuilder
from src.graph.node2vec_trainer import Node2VecTrainer
from src.llm.ollama_client import OllamaClient
from src.recommenders.graph_recommender import GraphRecommender
from src.recommenders.hybrid_recommender import HybridRecommender
from src.recommenders.mesh_overlap_recommender import MeshOverlapRecommender
from src.recommenders.semantic_recommender import SemanticRecommender
from src.utils.config import load_settings


DEFAULT_ARTICLES_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"
DEFAULT_SEMANTIC_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embeddings.npy"
DEFAULT_SEMANTIC_INDEX_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embedding_index.json"
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_mesh_graph.gpickle"
DEFAULT_NODE2VEC_PATH = PROJECT_ROOT / "data" / "artifacts" / "node2vec_embeddings.kv"


def main() -> None:
    args = parse_args()
    articles = load_articles(args.articles)
    settings = load_settings()
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=args.timeout,
    )
    recommendations = recommend(articles, args)
    evidence_items = []

    if args.extract_evidence:
        extractor = EvidenceExtractionAgent(client, use_llm=not args.no_llm and settings.use_llm)
        evidence_items = [extractor.extract(article) for article in recommendations]

    generator = AnswerGenerator(client, use_llm=not args.no_llm and settings.use_llm)
    answer = generator.generate(args.question, recommendations, evidence_items)
    print(answer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="User question to answer from cached evidence.")
    parser.add_argument("--pmid", required=True, help="Seed article PMID for article-based recommendations.")
    parser.add_argument("--method", choices=["semantic", "mesh", "graph", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic fallback without calling Ollama.")
    parser.add_argument("--extract-evidence", action="store_true", help="Extract evidence for each recommended article before answering.")
    parser.add_argument("--semantic-embeddings", type=Path, default=DEFAULT_SEMANTIC_EMBEDDINGS_PATH)
    parser.add_argument("--semantic-index", type=Path, default=DEFAULT_SEMANTIC_INDEX_PATH)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--node2vec", type=Path, default=DEFAULT_NODE2VEC_PATH)
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
