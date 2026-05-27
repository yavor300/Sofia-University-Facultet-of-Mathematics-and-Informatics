"""Run the end-to-end image OCR -> answer prediction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.config import load_yaml_config
from openqa_textual.data import (
    get_sample_id,
    get_sample_image,
    get_sample_language,
    load_dataset_splits,
)
from openqa_textual.generation import DEFAULT_GENERATION_KWARGS, HeuristicQAGenerator, LocalLLMGenerator
from openqa_textual.ocr_postprocess import clean_ocr_question
from openqa_textual.prediction import create_prediction_record, write_json, write_jsonl
from openqa_textual.retrieval import build_retriever, load_retrieval_index
from scripts.inspect_dataset import resolve_split_name
from scripts.predict_llm import _debug_retrieved_examples
from scripts.run_ocr import run_ensemble_ocr, run_single_ocr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", help="Dataset split or configured alias.")
    parser.add_argument("--config", default="configs/generation.yaml", help="Generation config YAML.")
    parser.add_argument("--ocr-config", default="configs/ocr.yaml", help="OCR config YAML.")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Data config YAML.")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml", help="Retrieval config YAML.")
    parser.add_argument("--dataset-name", help="Override dataset name from data config.")
    parser.add_argument("--cache-dir", help="Override Hugging Face dataset cache directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/dev_predictions.json"),
        help="Output prediction path.",
    )
    parser.add_argument("--jsonl", action="store_true", help="Write JSONL instead of one JSON list.")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Write JSONL incrementally after each prediction. Implies --jsonl.",
    )
    parser.add_argument(
        "--submission",
        action="store_true",
        help="Strip debug fields for final submission-style output.",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N samples.")
    parser.add_argument(
        "--generator",
        choices=["llm", "heuristic"],
        default="llm",
        help="Answer generator to use.",
    )
    parser.add_argument("--model-name", help="Override model name from generation config.")
    parser.add_argument("--model-cache-dir", help="Override model cache directory.")
    parser.add_argument("--adapter-path", help="Optional LoRA/QLoRA adapter path.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model with 4-bit quantization.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--torch-dtype", default="auto", help="Transformers torch_dtype value.")
    parser.add_argument("--max-new-tokens", type=int, help="Override max_new_tokens.")
    parser.add_argument(
        "--engine",
        help="OCR engine name. Use 'ensemble' to run enabled engines/variants. Defaults to OCR config.",
    )
    parser.add_argument("--preprocess-variant", help="Override OCR preprocessing variant.")
    parser.add_argument("--ocr-cache-dir", type=Path, help="Override OCR cache directory.")
    parser.add_argument("--no-cache", action="store_true", help="Do not read or write OCR cache files.")
    parser.add_argument("--retrieval-index", type=Path, help="Train retrieval index for RAG.")
    parser.add_argument("--retrieval-method", choices=["bm25", "dense", "hybrid"], help="RAG retrieval method.")
    parser.add_argument("--rag-k", type=int, default=0, help="Number of retrieved examples for RAG.")
    parser.add_argument("--preview-chars", type=int, default=100, help="Progress preview characters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = load_yaml_config(args.data_config)
    generation_config = load_yaml_config(args.config)
    ocr_config = load_yaml_config(args.ocr_config)

    split_name, dataset_split = load_split(args, data_config)
    total = len(dataset_split) if args.limit is None else min(len(dataset_split), max(args.limit, 0))

    generator = build_generator(args, generation_config)
    retriever = build_optional_retriever(args)
    engine_name = str(args.engine or ocr_config.get("default_engine", "tesseract"))
    ocr_cache_dir = args.ocr_cache_dir or Path(ocr_config.get("cache_dir", "data/ocr_cache"))
    engine_cache: dict[Any, Any] = {}

    output_handle = None
    predictions: list[dict[str, Any]] = []
    if args.stream:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("w", encoding="utf-8")

    try:
        for index in range(total):
            sample = dataset_split[index]
            prediction = predict_sample(
                sample=sample,
                index=index,
                split_name=split_name,
                engine_name=engine_name,
                generator=generator,
                retriever=retriever,
                args=args,
                ocr_config=ocr_config,
                ocr_cache_dir=ocr_cache_dir,
                engine_cache=engine_cache,
            )
            if args.submission:
                prediction.pop("debug", None)
            predictions.append(prediction)

            preview = prediction.get("debug", {}).get("clean_question", "")
            preview = str(preview).replace("\n", " ")
            if len(preview) > args.preview_chars >= 0:
                preview = preview[: args.preview_chars] + "..."
            answer_preview = prediction["answers"][0] if prediction.get("answers") else ""
            print(f"[{index + 1}/{total}] {prediction['question_id']}: {preview}", flush=True)
            print(f"  -> {answer_preview}", flush=True)

            if output_handle is not None:
                import json

                output_handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()

    if not args.stream:
        if args.jsonl:
            write_jsonl(args.output, predictions)
        else:
            write_json(args.output, predictions)

    print(f"Wrote {len(predictions)} predictions to {args.output}")


def predict_sample(
    sample: dict[str, Any],
    index: int,
    split_name: str,
    engine_name: str,
    generator: Any,
    retriever: Any | None,
    args: argparse.Namespace,
    ocr_config: dict[str, Any],
    ocr_cache_dir: Path,
    engine_cache: dict[Any, Any],
) -> dict[str, Any]:
    question_id = safe_sample_id(sample, index)
    language = get_sample_language(sample)
    image = get_sample_image(sample)

    if engine_name.lower() == "ensemble":
        ocr_result, cached, candidates = run_ensemble_ocr(
            image=image,
            language=language,
            split=split_name,
            question_id=question_id,
            args=args,
            ocr_config=ocr_config,
            cache_dir=ocr_cache_dir,
            engine_cache=engine_cache,
        )
    else:
        ocr_result, cached, candidates = run_single_ocr(
            image=image,
            engine_name=engine_name,
            language=language,
            split=split_name,
            question_id=question_id,
            args=args,
            ocr_config=ocr_config,
            cache_dir=ocr_cache_dir,
            engine_cache=engine_cache,
        )

    clean_question = clean_ocr_question(ocr_result.text, language=language)
    retrieved_examples = []
    if retriever is not None and args.rag_k > 0:
        retrieved_examples = [
            item
            for item in retriever.search(clean_question, top_k=args.rag_k + 1)
            if str(item.get("question_id", "")) != question_id
        ][: args.rag_k]

    if isinstance(generator, LocalLLMGenerator):
        result = generator.generate(
            clean_question,
            language=language,
            retrieved_examples=retrieved_examples,
        )
    else:
        result = generator.generate(clean_question, language=language)

    answers = [postprocess_answer(answer) for answer in result.answers]
    if not answers:
        answers = [""]

    return create_prediction_record(
        question_id=question_id,
        answers=answers,
        language=language,
        debug={
            "ocr_text": ocr_result.text,
            "clean_question": clean_question,
            "ocr_engine": ocr_result.engine,
            "preprocess_variant": ocr_result.metadata.get("preprocess_variant"),
            "ocr_cached": cached,
            "ocr_confidence": ocr_result.confidence,
            "ocr_candidates": candidates,
            "model": getattr(generator, "model_name", getattr(generator, "name", "")),
            "adapter_path": getattr(generator, "adapter_path", None),
            "rag_k": args.rag_k,
            "retrieved_examples": _debug_retrieved_examples(retrieved_examples),
            **result.metadata,
        },
    )


def build_generator(args: argparse.Namespace, config: dict[str, Any]):
    if args.generator == "heuristic":
        return HeuristicQAGenerator()

    model_config = config.get("model", {})
    generation_kwargs = dict(DEFAULT_GENERATION_KWARGS)
    generation_kwargs.update(config.get("generation", {}))
    if args.max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = args.max_new_tokens

    model_name = args.model_name or model_config.get("name") or "Qwen/Qwen2.5-7B-Instruct"
    cache_dir = args.model_cache_dir or model_config.get("cache_dir")
    load_in_4bit = args.load_in_4bit or bool(model_config.get("load_in_4bit", False))

    print(f"Loading model: {model_name}", flush=True)
    return LocalLLMGenerator.from_pretrained(
        model_name=model_name,
        cache_dir=cache_dir,
        load_in_4bit=load_in_4bit,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        adapter_path=args.adapter_path,
        generation_kwargs=generation_kwargs,
    )


def build_optional_retriever(args: argparse.Namespace):
    if args.rag_k <= 0:
        return None
    retrieval_config = load_yaml_config(args.retrieval_config)
    rag_config = retrieval_config.get("rag", {})
    index_path = args.retrieval_index or Path(
        retrieval_config.get("index", {}).get("path", "data/processed/train_retrieval_index.jsonl")
    )
    if not index_path.exists():
        raise SystemExit(f"Retrieval index does not exist: {index_path}")
    method = args.retrieval_method or rag_config.get("method") or "bm25"
    records = load_retrieval_index(index_path)
    print(f"Loaded {len(records)} retrieval records from {index_path} using {method}", flush=True)
    return build_retriever(method, records, config=retrieval_config)


def load_split(args: argparse.Namespace, data_config: dict[str, Any]):
    dataset_config = data_config.get("dataset", {})
    dataset_name = args.dataset_name or dataset_config.get("name")
    cache_dir = args.cache_dir or dataset_config.get("cache_dir")
    split_aliases = dataset_config.get("splits", {})

    if not dataset_name:
        raise SystemExit("Dataset name is required via --dataset-name or configs/data.yaml.")

    dataset = load_dataset_splits(dataset_name, cache_dir=cache_dir)
    split_name = resolve_split_name(args.split, dataset, split_aliases)
    return split_name, dataset[split_name]


def postprocess_answer(answer: str) -> str:
    return str(answer or "").strip()


def safe_sample_id(sample: dict[str, Any], index: int) -> str:
    try:
        return get_sample_id(sample)
    except KeyError:
        return f"sample-{index:05d}"


if __name__ == "__main__":
    main()
