"""Query the retrieval index with BM25, dense, or hybrid retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.config import load_yaml_config
from openqa_textual.prediction import write_json
from openqa_textual.retrieval import build_retriever, load_retrieval_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrieval.yaml", help="Retrieval config YAML.")
    parser.add_argument("--index", type=Path, help="Retrieval index JSONL path.")
    parser.add_argument("--query", required=True, help="Question text to retrieve against.")
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid"],
        default="bm25",
        help="Retrieval method.",
    )
    parser.add_argument("--top-k", type=int, help="Number of matches to return.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    index_path = args.index or Path(config.get("index", {}).get("path", "data/processed/train_retrieval_index.jsonl"))
    if not index_path.exists():
        raise SystemExit(f"Retrieval index does not exist: {index_path}")

    records = load_retrieval_index(index_path)
    retriever = build_retriever(args.method, records, config=config)
    top_k = args.top_k or int(config.get(args.method, {}).get("top_k", 5))
    results = retriever.search(args.query, top_k=top_k)

    if args.output:
        write_json(args.output, results)

    for item in results:
        score = item.get(f"{args.method}_score", item.get("hybrid_score", 0.0))
        print(f"{item['rank']}. {item['question_id']} score={score:.4f}")
        print(f"   Q: {item.get('ocr_question', '')}")
        print(f"   A: {item.get('gold_answer', '')}")
    if args.output:
        print(f"Wrote retrieval results to {args.output}")


if __name__ == "__main__":
    main()
