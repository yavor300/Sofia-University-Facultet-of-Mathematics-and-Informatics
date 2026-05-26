"""Build a train retrieval memory index from OCR questions and gold answers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.config import load_yaml_config
from openqa_textual.data import load_dataset_splits
from openqa_textual.retrieval import (
    build_retrieval_index_from_dataset,
    build_retrieval_index_from_ocr_rows,
    gold_answers_from_dataset_split,
    load_ocr_rows,
    write_retrieval_index,
)
from scripts.inspect_dataset import resolve_split_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrieval.yaml", help="Retrieval config YAML.")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Data config YAML.")
    parser.add_argument("--ocr-config", default="configs/ocr.yaml", help="OCR config YAML.")
    parser.add_argument(
        "--ocr-jsonl",
        type=Path,
        help="Existing OCR JSONL for train rows. Preferred when available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output retrieval index JSONL. Defaults to retrieval config index.path.",
    )
    parser.add_argument("--dataset-name", help="Override dataset name from data config.")
    parser.add_argument("--cache-dir", help="Override Hugging Face dataset cache directory.")
    parser.add_argument("--split", default="train", help="Dataset split or configured alias.")
    parser.add_argument("--ocr-cache-dir", type=Path, help="Override OCR cache directory.")
    parser.add_argument("--engine", help="OCR engine cache name. Defaults to OCR config default_engine.")
    parser.add_argument(
        "--preprocess-variant",
        help="OCR cache preprocessing variant. Defaults to OCR config default_preprocess_variant.",
    )
    parser.add_argument("--text-field", default="clean_question", help="OCR JSONL question field.")
    parser.add_argument("--limit", type=int, help="Limit number of dataset rows.")
    parser.add_argument(
        "--no-dataset-gold",
        action="store_true",
        help="Do not load dataset gold answers when --ocr-jsonl is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_config = load_yaml_config(args.config)
    data_config = load_yaml_config(args.data_config)
    ocr_config = load_yaml_config(args.ocr_config)

    output = args.output or Path(retrieval_config.get("index", {}).get("path", "data/processed/train_retrieval_index.jsonl"))

    if args.ocr_jsonl:
        if not args.ocr_jsonl.exists():
            raise SystemExit(f"OCR JSONL does not exist: {args.ocr_jsonl}")
        ocr_rows = load_ocr_rows(args.ocr_jsonl)
        if args.limit is not None:
            ocr_rows = ocr_rows[: max(args.limit, 0)]
        if args.no_dataset_gold:
            gold_by_id = {}
        else:
            _, dataset_split = load_split(args, data_config)
            gold_by_id = gold_answers_from_dataset_split(dataset_split)
        records = build_retrieval_index_from_ocr_rows(
            ocr_rows,
            gold_by_id=gold_by_id,
            text_field=args.text_field,
        )
    else:
        split_name, dataset_split = load_split(args, data_config)
        engine = args.engine or ocr_config.get("default_engine", "tesseract")
        preprocess_variant = args.preprocess_variant or ocr_config.get("default_preprocess_variant", "resize_only")
        ocr_cache_dir = args.ocr_cache_dir or Path(ocr_config.get("cache_dir", "data/ocr_cache"))
        records = build_retrieval_index_from_dataset(
            dataset_split=dataset_split,
            split_name=split_name,
            cache_dir=ocr_cache_dir,
            engine=engine,
            preprocess_variant=preprocess_variant,
            limit=args.limit,
        )

    write_retrieval_index(output, records)
    with_questions = sum(1 for record in records if record.get("ocr_question"))
    with_answers = sum(1 for record in records if record.get("gold_answer"))
    print(f"Wrote {len(records)} retrieval records to {output}")
    print(f"Records with OCR question: {with_questions}")
    print(f"Records with gold answer: {with_answers}")


def load_split(args: argparse.Namespace, data_config: dict):
    dataset_config = data_config.get("dataset", {})
    dataset_name = args.dataset_name or dataset_config.get("name")
    cache_dir = args.cache_dir or dataset_config.get("cache_dir")
    split_aliases = dataset_config.get("splits", {})

    if not dataset_name:
        raise SystemExit("Dataset name is required via --dataset-name or configs/data.yaml.")

    dataset = load_dataset_splits(dataset_name, cache_dir=cache_dir)
    split_name = resolve_split_name(args.split, dataset, split_aliases)
    return split_name, dataset[split_name]


if __name__ == "__main__":
    main()
