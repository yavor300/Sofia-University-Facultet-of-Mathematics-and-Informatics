"""Train and save node2vec embeddings for the Article-MeSH graph artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_mesh_graph.gpickle"
DEFAULT_NODE2VEC_PATH = PROJECT_ROOT / "data" / "artifacts" / "node2vec_embeddings.kv"


def main() -> int:
    try:
        args = parse_args()

        from src.graph.graph_builder import ArticleMeshGraphBuilder
        from src.graph.node2vec_trainer import Node2VecTrainer

        builder = ArticleMeshGraphBuilder()
        graph = builder.load(str(args.graph))
        if graph.number_of_nodes() == 0:
            raise ValueError(f"Graph has no nodes: {args.graph}")

        trainer = Node2VecTrainer(
            dimensions=args.dimensions,
            walk_length=args.walk_length,
            num_walks=args.num_walks,
            workers=args.workers,
            window=args.window,
            min_count=args.min_count,
        )
        embeddings = trainer.train(graph)
        trainer.save(embeddings, str(args.output))

        print(
            "node2vec embeddings saved successfully: "
            f"{len(embeddings)} nodes, {embeddings.vector_size} dimensions, path={args.output}"
        )
        return 0
    except Exception as exc:
        print(f"Error training node2vec embeddings: {exc}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_NODE2VEC_PATH)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--num-walks", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-count", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
