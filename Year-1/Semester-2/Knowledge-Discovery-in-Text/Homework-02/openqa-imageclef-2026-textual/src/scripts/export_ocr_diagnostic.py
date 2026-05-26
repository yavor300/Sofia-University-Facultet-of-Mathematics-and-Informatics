"""Export Baseline 0 OCR-only diagnostic rows for dev analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.config import load_yaml_config
from openqa_textual.data import load_dataset_splits
from openqa_textual.prediction import (
    build_ocr_diagnostic_rows,
    gold_answers_from_dataset_split,
    gold_answers_from_jsonl,
    read_jsonl,
    write_jsonl,
)
from scripts.inspect_dataset import resolve_split_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-jsonl", type=Path, required=True, help="OCR JSONL input path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/dev_ocr_outputs.jsonl"),
        help="Diagnostic JSONL output path.",
    )
    parser.add_argument(
        "--text-field",
        default="clean_question",
        help="OCR row field to use as diagnostic text. Falls back to ocr_text.",
    )
    parser.add_argument("--gold-jsonl", type=Path, help="Optional gold answers JSONL path.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config YAML.")
    parser.add_argument("--dataset-name", help="Override dataset name from config.")
    parser.add_argument("--cache-dir", help="Override Hugging Face dataset cache directory.")
    parser.add_argument(
        "--split",
        default="dev",
        help="Dataset split or configured split alias to load gold answers from.",
    )
    parser.add_argument(
        "--no-dataset-gold",
        action="store_true",
        help="Do not try to load dataset gold answers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ocr_jsonl.exists():
        raise SystemExit(
            f"OCR JSONL does not exist: {args.ocr_jsonl}\n"
            "Create it first by running OCR and postprocessing, for example:\n"
            "  .venv/bin/python -m src.scripts.run_ocr --split dev --engine tesseract "
            "--preprocess-variant resize_only --output data/processed/dev_ocr_outputs.jsonl\n"
            "  .venv/bin/python -m src.scripts.postprocess_ocr "
            "--input data/processed/dev_ocr_outputs.jsonl "
            "--output data/processed/dev_ocr_outputs_cleaned.jsonl"
        )
    if args.gold_jsonl and not args.gold_jsonl.exists():
        raise SystemExit(f"Gold JSONL does not exist: {args.gold_jsonl}")

    ocr_rows = read_jsonl(args.ocr_jsonl)
    gold_by_id = {}

    if args.gold_jsonl:
        gold_by_id.update(gold_answers_from_jsonl(args.gold_jsonl))
    elif not args.no_dataset_gold:
        gold_by_id.update(load_dataset_gold_answers(args))

    diagnostic_rows = build_ocr_diagnostic_rows(
        ocr_rows,
        gold_by_id=gold_by_id,
        text_field=args.text_field,
    )
    write_jsonl(args.output, diagnostic_rows)
    with_gold = sum(1 for row in diagnostic_rows if row["gold_answer"])
    print(f"Wrote {len(diagnostic_rows)} OCR diagnostic rows to {args.output}")
    print(f"Rows with gold answers: {with_gold}")


def load_dataset_gold_answers(args: argparse.Namespace) -> dict[str, str]:
    config = load_yaml_config(args.config)
    dataset_config = config.get("dataset", {})
    dataset_name = args.dataset_name or dataset_config.get("name")
    cache_dir = args.cache_dir or dataset_config.get("cache_dir")
    split_aliases = dataset_config.get("splits", {})

    if not dataset_name:
        return {}

    dataset = load_dataset_splits(dataset_name, cache_dir=cache_dir)
    split_name = resolve_split_name(args.split, dataset, split_aliases)
    return gold_answers_from_dataset_split(dataset[split_name])


if __name__ == "__main__":
    main()
