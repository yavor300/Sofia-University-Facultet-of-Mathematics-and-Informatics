"""Build or update the local PubMed article cache from sample queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_QUERY_PATH = PROJECT_ROOT / "data" / "sample_queries.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"


def main() -> int:
    try:
        args = parse_args()

        from src.retrieval.cache import load_articles, merge_articles, save_articles
        from src.retrieval.pubmed_client import PubMedClient
        from src.utils.config import load_settings, require_ncbi_email

        settings = load_settings()
        client = PubMedClient(
            email=require_ncbi_email(settings),
            api_key=settings.ncbi_api_key,
        )

        queries = load_queries(args.queries)
        existing_articles = load_articles(str(args.output))
        all_new_articles: list[dict] = []

        for query in queries:
            print(f"Retrieving up to {args.max_results} articles for: {query}")
            articles = client.search_and_fetch(query=query, max_results=args.max_results)
            all_new_articles.extend(articles)
            print(f"  fetched {len(articles)} normalized articles")

        merged_articles = merge_articles(existing_articles, all_new_articles)
        save_articles(merged_articles, str(args.output))

        print(
            "PubMed dataset saved successfully: "
            f"{len(merged_articles)} total articles "
            f"({len(all_new_articles)} fetched this run), path={args.output}"
        )
        return 0
    except Exception as exc:
        print(f"Error building PubMed dataset: {exc}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERY_PATH,
        help="Path to a JSON list of PubMed search query strings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the article cache JSON file.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Maximum PubMed results to retrieve per query.",
    )
    return parser.parse_args()


def load_queries(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Sample query file must contain a JSON list: {path}")

    queries = [str(item).strip() for item in data if str(item).strip()]
    if not queries:
        raise ValueError(f"No sample queries found in {path}")

    return queries


if __name__ == "__main__":
    raise SystemExit(main())
