"""Extract structured evidence from a cached PubMed article."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.evidence_extractor import EvidenceExtractionAgent
from src.llm.ollama_client import OllamaClient
from src.utils.config import load_settings


DEFAULT_ARTICLES_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"


def main() -> None:
    args = parse_args()
    article = find_article(load_articles(args.articles), args.pmid)
    settings = load_settings()
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=args.timeout,
    )
    extractor = EvidenceExtractionAgent(client, use_llm=not args.no_llm and settings.use_llm)
    evidence = extractor.extract(article)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmid", required=True, help="PMID from the cached article dataset.")
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-llm", action="store_true", help="Use fallback extraction without calling Ollama.")
    return parser.parse_args()


def load_articles(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Article cache must contain a JSON list: {path}")

    return [article for article in data if isinstance(article, dict)]


def find_article(articles: list[dict], pmid: str) -> dict:
    requested_pmid = str(pmid).strip()
    for article in articles:
        if str(article.get("pmid", "")).strip() == requested_pmid:
            return article

    raise ValueError(f"PMID not found in cached articles: {requested_pmid}")


if __name__ == "__main__":
    main()
