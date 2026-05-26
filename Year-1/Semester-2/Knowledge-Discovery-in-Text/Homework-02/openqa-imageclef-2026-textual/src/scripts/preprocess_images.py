"""Save OCR preprocessing debug outputs for sample images."""

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
from openqa_textual.data import get_sample_image, load_dataset_splits
from openqa_textual.image_utils import (
    preprocess_variants_for_ocr,
    save_preprocessed_debug_outputs,
)
from openqa_textual.logging_utils import configure_logging
from scripts.inspect_dataset import resolve_split_name, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ocr.yaml", help="Path to OCR config YAML.")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Path to data config YAML.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Preprocess images from an existing directory instead of loading the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/preprocessed_debug_images"),
        help="Directory where preprocessed variants should be written.",
    )
    parser.add_argument("--dataset-name", help="Override dataset name from data config.")
    parser.add_argument("--cache-dir", help="Override Hugging Face dataset cache directory.")
    parser.add_argument("--split", default="train", help="Dataset split or configured alias.")
    parser.add_argument("--n", type=int, default=30, help="Number of source images to process.")
    parser.add_argument(
        "--manifest-name",
        default="manifest.jsonl",
        help="Manifest filename written inside --output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    ocr_config = load_yaml_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir:
        manifest = save_preprocessed_debug_outputs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config=ocr_config,
            n=args.n,
        )
    else:
        manifest = save_dataset_preprocessed_outputs(args, ocr_config)

    manifest_path = args.output_dir / args.manifest_name
    write_jsonl(manifest_path, manifest)
    saved_count = sum(1 for row in manifest if row.get("saved"))
    print(f"Saved {saved_count}/{len(manifest)} preprocessed images to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


def save_dataset_preprocessed_outputs(
    args: argparse.Namespace,
    ocr_config: dict[str, Any],
) -> list[dict[str, Any]]:
    data_config = load_yaml_config(args.data_config)
    dataset_config = data_config.get("dataset", {})
    dataset_name = args.dataset_name or dataset_config.get("name")
    cache_dir = args.cache_dir or dataset_config.get("cache_dir")
    split_aliases = dataset_config.get("splits", {})

    if not dataset_name:
        raise SystemExit("Dataset name is required via --dataset-name or configs/data.yaml.")

    dataset = load_dataset_splits(dataset_name, cache_dir=cache_dir)
    split_name = resolve_split_name(args.split, dataset, split_aliases)
    split = dataset[split_name]

    manifest: list[dict[str, Any]] = []
    limit = min(max(args.n, 0), len(split))
    for index in range(limit):
        sample = split[index]
        try:
            image = get_sample_image(sample)
            variants = preprocess_variants_for_ocr(image, config=ocr_config)
            for variant, processed in variants.items():
                variant_dir = args.output_dir / split_name / variant
                variant_dir.mkdir(parents=True, exist_ok=True)
                output_path = variant_dir / f"{index:05d}.png"
                processed.save(output_path, format="PNG")
                manifest.append(
                    {
                        "index": index,
                        "split": split_name,
                        "variant": variant,
                        "path": str(output_path),
                        "width": processed.width,
                        "height": processed.height,
                        "mode": processed.mode,
                        "saved": True,
                    }
                )
        except Exception as exc:
            manifest.append(
                {
                    "index": index,
                    "split": split_name,
                    "saved": False,
                    "error": str(exc),
                }
            )
    return manifest


if __name__ == "__main__":
    main()
