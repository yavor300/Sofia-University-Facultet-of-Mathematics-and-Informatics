"""Inspect the ImageCLEF OpenQA Textual dataset and save debug image samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.config import load_yaml_config
from openqa_textual.data import load_dataset_splits, save_debug_image_samples
from openqa_textual.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config YAML.")
    parser.add_argument("--dataset-name", help="Override dataset name from config.")
    parser.add_argument("--cache-dir", help="Override Hugging Face dataset cache directory.")
    parser.add_argument(
        "--split",
        default="train",
        help="Split or configured split alias to inspect, for example train/dev/test.",
    )
    parser.add_argument(
        "--save-samples",
        type=Path,
        help="Directory where decoded sample PNG files should be written.",
    )
    parser.add_argument("--n", type=int, default=30, help="Number of samples to save.")
    parser.add_argument(
        "--manifest-name",
        default="manifest.jsonl",
        help="Manifest filename written inside --save-samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    config = load_yaml_config(args.config)
    dataset_config = config.get("dataset", {})
    dataset_name = args.dataset_name or dataset_config.get("name")
    cache_dir = args.cache_dir or dataset_config.get("cache_dir")
    split_aliases = dataset_config.get("splits", {})

    if not dataset_name:
        raise SystemExit("Dataset name is required via --dataset-name or configs/data.yaml.")

    dataset = load_dataset_splits(dataset_name, cache_dir=cache_dir)
    split_name = resolve_split_name(args.split, dataset, split_aliases)
    split = dataset[split_name]

    print(f"Dataset: {dataset_name}")
    print(f"Available splits: {', '.join(dataset.keys())}")
    print(f"Selected split: {split_name}")
    print(f"Rows: {len(split)}")

    if args.save_samples:
        manifest = save_debug_image_samples(
            split,
            output_dir=args.save_samples,
            n=args.n,
            split_name=split_name,
        )
        manifest_path = args.save_samples / args.manifest_name
        write_jsonl(manifest_path, manifest)
        saved_count = sum(1 for row in manifest if row.get("saved"))
        print(f"Saved {saved_count}/{len(manifest)} images to {args.save_samples}")
        print(f"Manifest: {manifest_path}")


def resolve_split_name(
    requested: str,
    dataset: Any,
    split_aliases: dict[str, str],
) -> str:
    if requested in dataset:
        return requested

    alias = split_aliases.get(requested)
    if alias and alias in dataset:
        return alias

    available = ", ".join(dataset.keys())
    aliases = ", ".join(f"{key}->{value}" for key, value in split_aliases.items())
    raise SystemExit(
        f"Split '{requested}' was not found. Available splits: {available}. "
        f"Configured aliases: {aliases or 'none'}."
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
