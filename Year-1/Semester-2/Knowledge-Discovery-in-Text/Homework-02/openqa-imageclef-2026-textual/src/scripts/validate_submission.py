"""Validate a final submission JSON or ZIP file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.config import load_yaml_config
from openqa_textual.data import load_dataset_splits
from openqa_textual.submission import expected_ids_from_split, validate_submission_file
from scripts.inspect_dataset import resolve_split_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True, help="Submission JSON or ZIP path.")
    parser.add_argument(
        "--expected-size",
        type=int,
        help="Expected number of submission records.",
    )
    parser.add_argument(
        "--expected-size-from-split",
        help="Dataset split or configured alias used to validate expected size and IDs.",
    )
    parser.add_argument("--data-config", default="configs/data.yaml", help="Data config YAML.")
    parser.add_argument("--dataset-name", help="Override dataset name from data config.")
    parser.add_argument("--cache-dir", help="Override Hugging Face dataset cache directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.submission.exists():
        raise SystemExit(f"Submission file does not exist: {args.submission}")

    expected_size = args.expected_size
    expected_ids = None
    if args.expected_size_from_split:
        split = load_expected_split(args)
        expected_size = len(split)
        expected_ids = expected_ids_from_split(split)

    result = validate_submission_file(
        args.submission,
        expected_size=expected_size,
        expected_ids=expected_ids,
    )
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    if not result.valid:
        for error in result.errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"Submission is valid: {result.total} records")


def load_expected_split(args: argparse.Namespace):
    config = load_yaml_config(args.data_config)
    dataset_config = config.get("dataset", {})
    dataset_name = args.dataset_name or dataset_config.get("name")
    cache_dir = args.cache_dir or dataset_config.get("cache_dir")
    split_aliases = dataset_config.get("splits", {})
    if not dataset_name:
        raise SystemExit("Dataset name is required via --dataset-name or configs/data.yaml.")
    dataset = load_dataset_splits(dataset_name, cache_dir=cache_dir)
    split_name = resolve_split_name(args.expected_size_from_split, dataset, split_aliases)
    return dataset[split_name]


if __name__ == "__main__":
    main()
