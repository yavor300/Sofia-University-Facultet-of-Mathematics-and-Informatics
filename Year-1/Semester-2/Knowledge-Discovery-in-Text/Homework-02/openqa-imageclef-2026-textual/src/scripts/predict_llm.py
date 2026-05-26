"""Generate Baseline 2 OCR + prompted local LLM predictions."""

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
from openqa_textual.generation import DEFAULT_GENERATION_KWARGS, LocalLLMGenerator
from openqa_textual.prediction import create_prediction_record, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-jsonl", type=Path, required=True, help="Input OCR JSONL path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/llm_predictions.json"),
        help="Output prediction path.",
    )
    parser.add_argument("--config", default="configs/generation.yaml", help="Generation config YAML.")
    parser.add_argument(
        "--model-name",
        choices=["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
        help="Local instruction model to use.",
    )
    parser.add_argument("--model-cache-dir", help="Override model cache directory.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model with 4-bit quantization.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--torch-dtype", default="auto", help="Transformers torch_dtype value.")
    parser.add_argument("--max-new-tokens", type=int, help="Override max_new_tokens.")
    parser.add_argument("--text-field", default="clean_question", help="OCR field to answer from.")
    parser.add_argument("--limit", type=int, help="Only process the first N rows.")
    parser.add_argument("--jsonl", action="store_true", help="Write JSONL rows instead of one JSON list.")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Write JSONL incrementally after each prediction. Implies --jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ocr_jsonl.exists():
        raise SystemExit(f"OCR JSONL does not exist: {args.ocr_jsonl}")

    config = load_yaml_config(args.config)
    model_config = config.get("model", {})
    generation_config = dict(DEFAULT_GENERATION_KWARGS)
    generation_config.update(config.get("generation", {}))
    if args.max_new_tokens is not None:
        generation_config["max_new_tokens"] = args.max_new_tokens

    model_name = args.model_name or model_config.get("name") or "Qwen/Qwen2.5-7B-Instruct"
    cache_dir = args.model_cache_dir or model_config.get("cache_dir")
    load_in_4bit = args.load_in_4bit or bool(model_config.get("load_in_4bit", False))

    print(f"Loading model: {model_name}", flush=True)
    print(f"Generation config: {generation_config}", flush=True)
    generator = LocalLLMGenerator.from_pretrained(
        model_name=model_name,
        cache_dir=cache_dir,
        load_in_4bit=load_in_4bit,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        generation_kwargs=generation_config,
    )
    rows = read_jsonl(args.ocr_jsonl)
    if args.limit is not None:
        rows = rows[: max(args.limit, 0)]
    print(f"Loaded {len(rows)} OCR rows from {args.ocr_jsonl}", flush=True)

    predictions = build_llm_predictions(
        rows,
        generator=generator,
        text_field=args.text_field,
        output=args.output if args.stream else None,
    )
    if args.stream:
        pass
    elif args.jsonl:
        write_jsonl(args.output, predictions)
    else:
        write_json(args.output, predictions)

    answered = sum(1 for prediction in predictions if prediction["answers"] != [""])
    print(f"Wrote {len(predictions)} LLM predictions to {args.output}")
    print(f"Non-empty answers: {answered}")
    print(f"Model: {model_name}")


def build_llm_predictions(
    rows: list[dict[str, Any]],
    generator: LocalLLMGenerator,
    text_field: str,
    output: Path | None = None,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    output_handle = None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output.open("w", encoding="utf-8")

    try:
        for index, row in enumerate(rows, start=1):
            question = row.get(text_field)
            if question is None and text_field != "ocr_text":
                question = row.get("ocr_text", "")
            language = str(row.get("language") or "English")
            question_id = str(row.get("question_id", ""))
            preview = str(question or "").replace("\n", " ")[:100]
            print(f"[{index}/{len(rows)}] {question_id}: {preview}", flush=True)

            result = generator.generate(str(question or ""), language=language)
            prediction = create_prediction_record(
                question_id=question_id,
                answers=result.answers,
                language=language,
                debug={
                    "baseline": generator.name,
                    "ocr_text": row.get("ocr_text", ""),
                    "clean_question": str(question or ""),
                    "model": generator.model_name,
                    **result.metadata,
                },
            )
            predictions.append(prediction)

            answer_preview = prediction["answers"][0] if prediction["answers"] else ""
            print(f"  -> {answer_preview}", flush=True)
            if output_handle is not None:
                output_handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()

    return predictions


if __name__ == "__main__":
    main()
