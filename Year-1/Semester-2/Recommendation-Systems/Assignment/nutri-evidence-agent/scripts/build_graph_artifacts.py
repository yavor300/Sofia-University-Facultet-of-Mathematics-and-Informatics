"""Build Article-MeSH graph and node2vec embedding artifacts."""

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


DEFAULT_ARTICLES_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_mesh_graph.gpickle"
DEFAULT_NODE2VEC_PATH = PROJECT_ROOT / "data" / "artifacts" / "node2vec_embeddings.kv"


def main() -> None:
    args = parse_args()
    articles = load_articles(args.articles)

    builder = ArticleMeshGraphBuilder()
    graph = builder.build(articles)
    builder.save(graph, str(args.graph_output))

    article_nodes = sum(1 for _, data in graph.nodes(data=True) if data.get("node_type") == "article")
    mesh_nodes = sum(1 for _, data in graph.nodes(data=True) if data.get("node_type") == "mesh_term")
    print(
        f"Saved graph to {args.graph_output} "
        f"({article_nodes} article nodes, {mesh_nodes} MeSH nodes, {graph.number_of_edges()} edges)"
    )

    if args.skip_node2vec:
        return

    trainer = Node2VecTrainer(
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        workers=args.workers,
        window=args.window,
    )
    embeddings = trainer.train(graph)
    trainer.save(embeddings, str(args.node2vec_output))
    print(
        f"Saved node2vec embeddings to {args.node2vec_output} "
        f"({len(embeddings)} nodes, {embeddings.vector_size} dimensions)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--graph-output", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--node2vec-output", type=Path, default=DEFAULT_NODE2VEC_PATH)
    parser.add_argument("--skip-node2vec", action="store_true")
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--walk-length", type=int, default=15)
    parser.add_argument("--num-walks", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--window", type=int, default=8)
    return parser.parse_args()


def load_articles(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Article cache must contain a JSON list: {path}")

    return [article for article in data if isinstance(article, dict)]


if __name__ == "__main__":
    main()
