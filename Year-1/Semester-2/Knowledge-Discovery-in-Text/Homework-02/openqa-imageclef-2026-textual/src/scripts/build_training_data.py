"""Build chat-format SFT records from train OCR outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.config import load_yaml_config
from openqa_textual.data import load_dataset_splits
from openqa_textual.prediction import read_jsonl, write_jsonl
from openqa_textual.training_data import (
    build_clean_question_training_records,
    build_ocr_training_records,
    gold_answers_from_dataset_split,
)
from scripts.inspect_dataset import resolve_split_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml", help="Data config YAML.")
    parser.add_argument("--finetune-config", default="configs/finetune.yaml", help="Finetune config YAML.")
    parser.add_argument(
        "--ocr-jsonl",
        type=Path,
        default=Path("data/processed/train_ocr_outputs_cleaned.jsonl"),
        help="Train OCR JSONL input path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output training JSONL. Defaults to finetune data.train_path.",
    )
    parser.add_argument(
        "--clean-output",
        type=Path,
        help="Optional output for clean-question upper-bound training records.",
    )
    parser.add_argument("--dataset-name", help="Override dataset name from data config.")
    parser.add_argument("--cache-dir", help="Override Hugging Face dataset cache directory.")
    parser.add_argument("--split", default="train", help="Dataset split or configured alias. Defaults to train.")
    parser.add_argument("--text-field", default="clean_question", help="OCR JSONL question field.")
    parser.add_argument(
        "--variant",
        choices=["ocr", "clean", "both"],
        default="ocr",
        help="Training data variant to build. Use clean only for upper-bound comparison.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of OCR rows or dataset samples.")
    parser.add_argument(
        "--allow-non-train",
        action="store_true",
        help="Allow building records from a non-train split. Not recommended for fine-tuning.",
    )
    parser.add_argument(
        "--no-dataset-gold",
        action="store_true",
        help="Use only gold_answer fields already present in --ocr-jsonl.",
    )
    parser.add_argument(
        "--keep-missing-answer",
        action="store_true",
        help="Keep records with empty assistant answers.",
    )
    parser.add_argument(
        "--keep-missing-question",
        action="store_true",
        help="Keep records with empty user questions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = load_yaml_config(args.data_config)
    finetune_config = load_yaml_config(args.finetune_config)

    split_name, dataset_split = load_split(args, data_config)
    if not args.allow_non_train and args.split != "train" and split_name != "train":
        raise SystemExit(
            f"Refusing to build training data from split '{split_name}'. "
            "Use --allow-non-train only for diagnostics."
        )

    train_path = Path(
        args.output
        or finetune_config.get("data", {}).get("train_path")
        or "data/processed/train_sft_ocr.jsonl"
    )

    if args.variant in {"ocr", "both"}:
        if not args.ocr_jsonl.exists():
            raise SystemExit(
                f"OCR JSONL does not exist: {args.ocr_jsonl}\n"
                "Create it first with src.scripts.run_ocr and src.scripts.postprocess_ocr."
            )
        ocr_rows = read_jsonl(args.ocr_jsonl)
        if args.limit is not None:
            ocr_rows = ocr_rows[: max(args.limit, 0)]
        gold_by_id = {} if args.no_dataset_gold else gold_answers_from_dataset_split(dataset_split)
        ocr_records = build_ocr_training_records(
            ocr_rows,
            gold_by_id=gold_by_id,
            text_field=args.text_field,
            skip_missing_answer=not args.keep_missing_answer,
            skip_missing_question=not args.keep_missing_question,
        )
        write_jsonl(train_path, ocr_records)
        print(f"Wrote {len(ocr_records)} OCR training records to {train_path}")

    if args.variant in {"clean", "both"}:
        clean_path = Path(
            args.clean_output
            or finetune_config.get("data", {}).get("clean_train_path")
            or "data/processed/train_sft_clean_upper_bound.jsonl"
        )
        clean_split = dataset_split
        if args.limit is not None:
            clean_split = _LimitedSplit(dataset_split, max(args.limit, 0))
        clean_records = build_clean_question_training_records(
            clean_split,
            skip_missing_answer=not args.keep_missing_answer,
            skip_missing_question=not args.keep_missing_question,
        )
        write_jsonl(clean_path, clean_records)
        print(f"Wrote {len(clean_records)} clean-question training records to {clean_path}")


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


class _LimitedSplit:
    def __init__(self, split, limit: int) -> None:
        self.split = split
        self.limit = min(len(split), limit)

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int):
        if index >= self.limit:
            raise IndexError(index)
        return self.split[index]


if __name__ == "__main__":
    main()
