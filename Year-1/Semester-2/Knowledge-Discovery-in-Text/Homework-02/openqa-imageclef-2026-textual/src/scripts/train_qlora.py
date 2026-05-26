"""Fine-tune an instruction model with QLoRA on OCR-derived SFT records."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.config import load_yaml_config
from openqa_textual.finetune import (
    SUPPORTED_QLORA_BASE_MODELS,
    create_sft_trainer,
    load_json_sft_dataset,
    load_model_for_qlora,
    load_tokenizer,
    normalize_finetune_config,
    training_arguments_kwargs,
    validate_sft_records,
)
from openqa_textual.prediction import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune.yaml", help="Finetune config YAML.")
    parser.add_argument("--train-path", type=Path, help="Chat-format SFT JSONL path.")
    parser.add_argument(
        "--base-model",
        choices=SUPPORTED_QLORA_BASE_MODELS,
        help="Base instruction model.",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory for LoRA adapter.")
    parser.add_argument("--model-cache-dir", help="Hugging Face model cache directory.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map.")
    parser.add_argument("--limit", type=int, help="Limit number of training records.")
    parser.add_argument("--max-seq-length", type=int, help="Override max sequence length.")
    parser.add_argument("--num-train-epochs", type=float, help="Override number of epochs.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument("--per-device-train-batch-size", type=int, help="Override train batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Override grad accumulation.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantized loading.")
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Checkpoint path or true/false value accepted by transformers Trainer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/data and print resolved settings without loading the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = load_yaml_config(args.config)
    config = normalize_finetune_config(
        raw_config,
        overrides={
            "base_model": args.base_model,
            "output_dir": str(args.output_dir) if args.output_dir else None,
            "train_path": str(args.train_path) if args.train_path else None,
            "max_seq_length": args.max_seq_length,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "load_in_4bit": False if args.no_4bit else None,
        },
    )

    train_path = Path(config.get("data", {}).get("train_path", "data/processed/train_sft_ocr.jsonl"))
    if not train_path.exists():
        raise SystemExit(
            f"Training JSONL does not exist: {train_path}\n"
            "Create it first with src.scripts.build_training_data."
        )

    rows = read_jsonl(train_path)
    if args.limit is not None:
        rows = rows[: max(args.limit, 0)]
    validation = validate_sft_records(rows)
    print(f"Training data: {train_path}")
    print(f"Records: {validation['total']} valid={validation['valid']} invalid={validation['invalid']}")
    if validation["invalid"]:
        print(f"Invalid examples: {validation['invalid_examples']}")
        raise SystemExit("Refusing to train with invalid SFT records.")

    print(f"Base model: {config['base_model']}")
    print(f"Output dir: {config['output_dir']}")
    print(f"QLoRA 4-bit: {config.get('qlora', {}).get('load_in_4bit', True)}")
    print(f"Training args: {training_arguments_kwargs(config)}")
    if args.dry_run:
        print("Dry run complete; model was not loaded.")
        return

    tokenizer = load_tokenizer(config["base_model"], cache_dir=args.model_cache_dir)
    model = load_model_for_qlora(config, cache_dir=args.model_cache_dir, device_map=args.device_map)
    dataset = load_json_sft_dataset(train_path, limit=args.limit)

    trainer = create_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        config=config,
    )
    trainer.train(resume_from_checkpoint=_resume_value(args.resume_from_checkpoint))
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"Saved LoRA adapter and tokenizer to {config['output_dir']}")


def _resume_value(value: str | None):
    if value is None:
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


if __name__ == "__main__":
    main()
