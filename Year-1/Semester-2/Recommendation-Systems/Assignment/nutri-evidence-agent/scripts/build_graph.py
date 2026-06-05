"""Build and save the Article-MeSH graph artifact from cached PubMed articles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARTICLES_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_mesh_graph.gpickle"


def main() -> int:
    try:
        args = parse_args()

        from src.graph.graph_builder import ArticleMeshGraphBuilder
        from src.retrieval.cache import load_articles

        articles = load_articles(str(args.articles))
        if not articles:
            raise ValueError(f"No articles found in {args.articles}")

        builder = ArticleMeshGraphBuilder()
        graph = builder.build(articles)
        builder.save(graph, str(args.graph_output))

        article_nodes = sum(1 for _, data in graph.nodes(data=True) if data.get("node_type") == "article")
        mesh_nodes = sum(1 for _, data in graph.nodes(data=True) if data.get("node_type") == "mesh_term")
        print(
            "Article-MeSH graph saved successfully: "
            f"{article_nodes} article nodes, {mesh_nodes} MeSH nodes, "
            f"{graph.number_of_edges()} edges, path={args.graph_output}"
        )
        return 0
    except Exception as exc:
        print(f"Error building Article-MeSH graph: {exc}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--graph-output", type=Path, default=DEFAULT_GRAPH_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
