"""Apply deterministic safety checks to generated text."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.safety_checker import SafetyChecker
from src.llm.ollama_client import OllamaClient
from src.utils.config import load_settings


def main() -> None:
    args = parse_args()
    text = args.text
    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif text is None:
        text = sys.stdin.read()

    settings = load_settings()
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=args.timeout,
    )
    checker = SafetyChecker(client, use_llm=not args.no_llm and settings.use_llm)
    print(checker.validate(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Text to validate. Reads stdin if omitted.")
    parser.add_argument("--file", type=Path, help="UTF-8 text file to validate.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic rules without asking Ollama.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
