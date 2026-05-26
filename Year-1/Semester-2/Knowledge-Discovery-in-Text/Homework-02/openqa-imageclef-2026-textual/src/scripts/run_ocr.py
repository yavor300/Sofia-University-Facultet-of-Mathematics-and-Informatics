"""Run a configured OCR engine on dataset samples or saved image files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterator

from PIL import Image

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.config import load_yaml_config
from openqa_textual.data import get_sample_id, get_sample_image, get_sample_language, load_dataset_splits
from openqa_textual.image_utils import preprocess_for_ocr
from openqa_textual.logging_utils import configure_logging
from openqa_textual.ocr import (
    build_ocr_engine,
    enabled_ocr_engines,
    load_ocr_cache_record,
    ocr_options_for_language,
    safe_extract_ocr,
    select_best_ocr_result,
    write_ocr_cache_record,
)
from scripts.inspect_dataset import resolve_split_name, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ocr.yaml", help="Path to OCR config YAML.")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Path to data config YAML.")
    parser.add_argument(
        "--engine",
        help="OCR engine name. Use 'ensemble' to run enabled engines/variants. Defaults to config default_engine.",
    )
    parser.add_argument(
        "--preprocess-variant",
        help="Override OCR preprocessing variant, e.g. raw, resize_only, contrast, binarized.",
    )
    parser.add_argument("--input-dir", type=Path, help="Run OCR on images from this directory.")
    parser.add_argument("--dataset-name", help="Override dataset name from data config.")
    parser.add_argument("--cache-dir", help="Override Hugging Face dataset cache directory.")
    parser.add_argument("--split", default="train", help="Dataset split or configured alias.")
    parser.add_argument("--n", type=int, default=30, help="Number of images/samples to process.")
    parser.add_argument("--ocr-cache-dir", type=Path, help="Override OCR cache directory from config.")
    parser.add_argument("--no-cache", action="store_true", help="Do not read or write OCR cache files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ocr_outputs.jsonl"),
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    ocr_config = load_yaml_config(args.config)
    engine_name = args.engine or ocr_config.get("default_engine", "tesseract")
    cache_dir = args.ocr_cache_dir or Path(ocr_config.get("cache_dir", "data/ocr_cache"))
    engine_cache = {}

    rows: list[dict[str, Any]] = []
    for item in iter_inputs(args):
        language = item.get("language", "English")
        split = item.get("split", args.split or "split")
        question_id = str(item.get("question_id"))
        if str(engine_name).lower() == "ensemble":
            result, cached, candidates = run_ensemble_ocr(
                image=item["image"],
                language=language,
                split=split,
                question_id=question_id,
                args=args,
                ocr_config=ocr_config,
                cache_dir=cache_dir,
                engine_cache=engine_cache,
            )
            preprocess_variant = result.metadata.get("preprocess_variant", "ensemble")
        else:
            result, cached, candidates = run_single_ocr(
                image=item["image"],
                engine_name=str(engine_name),
                language=language,
                split=split,
                question_id=question_id,
                args=args,
                ocr_config=ocr_config,
                cache_dir=cache_dir,
                engine_cache=engine_cache,
            )
            preprocess_variant = result.metadata.get("preprocess_variant")

        rows.append(
            {
                "source": item.get("source"),
                "index": item.get("index"),
                "split": split,
                "question_id": question_id,
                "language": language,
                "ocr_engine": result.engine,
                "preprocess_variant": preprocess_variant,
                "ocr_text": result.text,
                "confidence": result.confidence,
                "cached": cached,
                "metadata": result.metadata,
                "candidates": candidates,
            }
        )

    write_jsonl(args.output, rows)
    failures = sum(1 for row in rows if row["metadata"].get("failed"))
    cache_hits = sum(1 for row in rows if row.get("cached"))
    print(f"Wrote {len(rows)} OCR rows to {args.output}")
    if not args.no_cache:
        print(f"Cache hits: {cache_hits}; cache dir: {cache_dir}")
    if failures:
        print(f"Recorded {failures} OCR failures as empty fallback results")


def run_single_ocr(
    image,
    engine_name: str,
    language: str,
    split: str,
    question_id: str,
    args: argparse.Namespace,
    ocr_config: dict[str, Any],
    cache_dir: Path,
    engine_cache: dict[Any, Any],
) -> tuple[Any, bool, list[dict[str, Any]]]:
    preprocess_variant = selected_preprocess_variants(args, ocr_config)[0]
    result, cached = run_ocr_candidate(
        image=image,
        engine_name=engine_name,
        preprocess_variant=preprocess_variant,
        language=language,
        split=split,
        question_id=question_id,
        args=args,
        ocr_config=ocr_config,
        cache_dir=cache_dir,
        engine_cache=engine_cache,
    )
    return result, cached, []


def run_ensemble_ocr(
    image,
    language: str,
    split: str,
    question_id: str,
    args: argparse.Namespace,
    ocr_config: dict[str, Any],
    cache_dir: Path,
    engine_cache: dict[Any, Any],
) -> tuple[Any, bool, list[dict[str, Any]]]:
    candidates = []
    cached_flags = []
    for engine_name, _ in ensemble_engine_configs(ocr_config):
        for preprocess_variant in selected_preprocess_variants(args, ocr_config, ensemble=True):
            result, cached = run_ocr_candidate(
                image=image,
                engine_name=engine_name,
                preprocess_variant=preprocess_variant,
                language=language,
                split=split,
                question_id=question_id,
                args=args,
                ocr_config=ocr_config,
                cache_dir=cache_dir,
                engine_cache=engine_cache,
            )
            candidates.append(result)
            cached_flags.append(cached)

    selected = select_best_ocr_result(candidates)
    candidate_rows = selected.metadata.get("ensemble_candidates", [])
    return selected, bool(cached_flags) and all(cached_flags), candidate_rows


def run_ocr_candidate(
    image,
    engine_name: str,
    preprocess_variant: str,
    language: str,
    split: str,
    question_id: str,
    args: argparse.Namespace,
    ocr_config: dict[str, Any],
    cache_dir: Path,
    engine_cache: dict[Any, Any],
):
    preprocess_config = preprocess_config_for_variant(ocr_config, preprocess_variant)
    processed = preprocess_for_ocr(image, preprocess_config)
    base_engine_options = dict(ocr_config.get("engines", {}).get(engine_name, {}))
    base_engine_options.pop("enabled", None)
    engine_options = ocr_options_for_language(
        engine_name,
        base_engine_options,
        language,
        ocr_config,
    )
    engine_key = (engine_name, tuple(sorted((key, _freeze_value(value)) for key, value in engine_options.items())))
    if engine_key not in engine_cache:
        engine_cache[engine_key] = build_ocr_engine(engine_name, engine_options)
    engine = engine_cache[engine_key]

    cache_record = None
    if not args.no_cache:
        cache_record = load_ocr_cache_record(
            cache_dir,
            split,
            engine.name,
            preprocess_variant,
            question_id,
        )

    if cache_record is not None:
        result = cache_record.to_result()
        cached = True
    else:
        result = safe_extract_ocr(engine, processed)
        cached = False
        result.metadata["preprocess_variant"] = preprocess_variant
        if not args.no_cache:
            write_ocr_cache_record(
                cache_dir=cache_dir,
                split=split,
                question_id=question_id,
                language=language,
                preprocess_variant=preprocess_variant,
                result=result,
            )

    result.metadata.setdefault("preprocess_variant", preprocess_variant)
    result.metadata.setdefault("cache_hit", cached)
    return result, cached


def preprocess_config_for_variant(ocr_config: dict[str, Any], variant: str) -> dict[str, Any]:
    config = dict(ocr_config)
    preprocessing = dict(config.get("preprocessing", {}))
    preprocessing["variant"] = variant
    config["preprocessing"] = preprocessing
    return config


def selected_preprocess_variants(
    args: argparse.Namespace,
    ocr_config: dict[str, Any],
    ensemble: bool = False,
) -> list[str]:
    if args.preprocess_variant and args.preprocess_variant != "ensemble":
        return [args.preprocess_variant]

    if ensemble:
        variants = ocr_config.get("preprocessing", {}).get("variants") or []
        return [str(variant) for variant in variants if variant != "ensemble"] or ["resize_only"]

    return [str(args.preprocess_variant or ocr_config.get("default_preprocess_variant", "resize_only"))]


def ensemble_engine_configs(ocr_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    engines = enabled_ocr_engines(ocr_config)
    if engines:
        return engines

    default_engine = str(ocr_config.get("default_engine", "tesseract"))
    default_options = dict(ocr_config.get("engines", {}).get(default_engine, {}))
    default_options.pop("enabled", None)
    return [(default_engine, default_options)]


def iter_inputs(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    if args.input_dir:
        yield from iter_input_dir(args.input_dir, args.n)
        return

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
    limit = min(max(args.n, 0), len(split))

    for index in range(limit):
        sample = split[index]
        yield {
            "source": f"{split_name}:{index}",
            "index": index,
            "split": split_name,
            "question_id": _safe_sample_id(sample, index),
            "language": get_sample_language(sample),
            "image": get_sample_image(sample),
        }


def iter_input_dir(input_dir: Path, n: int) -> Iterator[dict[str, Any]]:
    manifest = load_input_manifest(input_dir)
    image_paths = [
        path
        for path in sorted(input_dir.iterdir())
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ][: max(n, 0)]

    for index, path in enumerate(image_paths):
        metadata = manifest.get(str(path.resolve()), manifest.get(path.stem, {}))
        with Image.open(path) as image:
            yield {
                "source": str(path),
                "index": metadata.get("index", index),
                "split": metadata.get("split", "input_dir"),
                "question_id": metadata.get("question_id", path.stem),
                "language": metadata.get("language", "English"),
                "image": image.copy(),
            }


def load_input_manifest(input_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = input_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return {}

    manifest: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            path = row.get("path")
            if path:
                manifest[str(Path(path).resolve())] = row
            question_id = row.get("question_id")
            if question_id:
                manifest[str(question_id)] = row
            if path:
                manifest[Path(path).stem] = row
    return manifest


def _freeze_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_value(item)) for key, item in value.items()))
    return value


def _safe_sample_id(sample: dict[str, Any], index: int) -> str:
    try:
        return get_sample_id(sample)
    except KeyError:
        return f"sample-{index:05d}"


if __name__ == "__main__":
    main()
