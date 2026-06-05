"""Build and save semantic embedding artifacts for cached PubMed articles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARTICLES_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embeddings.npy"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embedding_index.json"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    try:
        args = parse_args()

        from src.recommenders.semantic_recommender import SemanticRecommender
        from src.retrieval.cache import load_articles

        articles = load_articles(str(args.articles))
        if not articles:
            raise ValueError(f"No articles found in {args.articles}")

        recommender = SemanticRecommender(model_name=args.model_name)
        recommender.fit(articles)
        recommender.save_artifacts(str(args.embeddings_output), str(args.index_output))

        article_count = len(recommender.articles)
        embedding_shape = tuple(recommender.embeddings.shape) if recommender.embeddings is not None else (0, 0)
        print(
            "Semantic embeddings saved successfully: "
            f"{article_count} articles, shape={embedding_shape}, "
            f"embeddings={args.embeddings_output}, index={args.index_output}"
        )
        return 0
    except Exception as exc:
        print(f"Error building semantic embeddings: {exc}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--embeddings-output", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--index-output", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
