"""Rerank semantic recommendation candidates with local Ollama."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.llm_recommender import LLMRecommendationReranker
from src.llm.ollama_client import OllamaClient
from src.retrieval.cache import load_articles
from src.utils.config import load_settings


DEFAULT_ARTICLES_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"
DEFAULT_SEMANTIC_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embeddings.npy"
DEFAULT_SEMANTIC_INDEX_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embedding_index.json"


def main() -> int:
    try:
        args = parse_args()
        articles = load_articles(str(args.articles))
        if not articles:
            raise ValueError(f"No articles found in {args.articles}")

        if args.candidates_csv:
            candidates = load_candidates_from_csv(
                args.candidates_csv,
                articles=articles,
                method=args.candidate_method,
                max_candidates=args.max_candidates,
            )
        else:
            candidates = load_semantic_candidates(args, articles)

        reranker = LLMRecommendationReranker(
            ollama_client=None,
            max_candidates=args.max_candidates,
            abstract_char_limit=args.abstract_char_limit,
        )
        if args.dry_run:
            print_dry_run_prompt(reranker, args.question, candidates, top_k=args.top_k)
            return 0

        settings = load_settings()
        ollama_client = None
        if not args.no_llm:
            ollama_client = OllamaClient(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                timeout=args.timeout,
            )
        reranker.ollama_client = ollama_client

        results = reranker.rerank(
            user_question=args.question,
            candidate_articles=candidates,
            top_k=args.top_k,
        )

        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Error reranking recommendations: {exc}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Research question used for semantic candidate retrieval and LLM reranking.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--abstract-char-limit", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic fallback order without calling Ollama.")
    parser.add_argument("--dry-run", action="store_true", help="Print the reranker prompt without calling Ollama.")
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        help="Use an existing candidate CSV instead of running semantic retrieval.",
    )
    parser.add_argument(
        "--candidate-method",
        default="semantic",
        help="Method rows to load from --candidates-csv, e.g. semantic or hybrid.",
    )
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--semantic-embeddings", type=Path, default=DEFAULT_SEMANTIC_EMBEDDINGS_PATH)
    parser.add_argument("--semantic-index", type=Path, default=DEFAULT_SEMANTIC_INDEX_PATH)
    return parser.parse_args()


def load_semantic_candidates(args: argparse.Namespace, articles: list[dict]) -> list[dict]:
    from src.recommenders.semantic_recommender import SemanticRecommender

    semantic = SemanticRecommender()
    semantic.load_artifacts(
        articles,
        str(args.semantic_embeddings),
        str(args.semantic_index),
    )
    return semantic.recommend_by_query(args.question, top_k=args.max_candidates)


def load_candidates_from_csv(
    path: Path,
    articles: list[dict],
    method: str,
    max_candidates: int,
) -> list[dict]:
    article_by_pmid = {
        str(article.get("pmid", "")).strip(): article
        for article in articles
        if str(article.get("pmid", "")).strip()
    }
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if clean(row.get("method")) != method:
                continue

            pmid = clean(row.get("pmid"))
            if not pmid:
                continue

            article = dict(article_by_pmid.get(pmid, {}))
            article.update(
                {
                    "pmid": pmid,
                    "title": clean(row.get("title")) or article.get("title", ""),
                    "year": clean(row.get("year")) or article.get("year"),
                    "journal": clean(row.get("journal")) or article.get("journal", ""),
                    "method": method,
                    "score": safe_float(row.get("semantic_score")),
                    "rank": safe_int(row.get("rank")),
                }
            )
            rows.append(article)

    rows.sort(key=lambda article: safe_int(article.get("rank")))
    return rows[: max(0, int(max_candidates))]


def print_dry_run_prompt(
    reranker: LLMRecommendationReranker,
    question: str,
    candidates: list[dict],
    top_k: int,
) -> None:
    preview = reranker.preview_prompt(
        user_question=question,
        candidate_articles=candidates,
        top_k=top_k,
    )
    print("Dry run only. No Ollama call was made.")
    print(f"Initial algorithmic candidates: {len(candidates)}")
    print(f"Candidate articles sent to LLM: {preview['candidate_count']}")
    print(f"Candidate PMIDs: {', '.join(preview['candidate_pmids'])}")
    print("\n--- System Prompt ---")
    print(preview["system"])
    print("\n--- User Prompt ---")
    print(preview["user"])


def clean(value: object) -> str:
    return str(value or "").strip()


def safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
