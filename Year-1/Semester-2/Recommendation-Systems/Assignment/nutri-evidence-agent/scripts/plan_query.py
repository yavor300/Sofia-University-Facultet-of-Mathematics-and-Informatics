"""Plan a PubMed query from a biomedical question using local Ollama."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.query_planner import QueryPlannerAgent
from src.llm.ollama_client import OllamaClient
from src.utils.config import load_settings


def main() -> None:
    args = parse_args()
    settings = load_settings()
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=args.timeout,
    )
    planner = QueryPlannerAgent(client, use_llm=not args.no_llm and settings.use_llm)
    plan = planner.plan(args.question)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Biomedical question to transform into a PubMed search plan.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-llm", action="store_true", help="Use fallback planning without calling Ollama.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
