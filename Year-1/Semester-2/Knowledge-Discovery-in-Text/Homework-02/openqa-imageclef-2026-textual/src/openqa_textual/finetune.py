"""QLoRA fine-tuning helpers."""

from __future__ import annotations

from pathlib import Path
import inspect
from typing import Any, Callable


SUPPORTED_QLORA_BASE_MODELS = (
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DEFAULT_FINETUNE_CONFIG = {
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "output_dir": "experiments/runs/qwen25_lora",
    "data": {"train_path": "data/processed/train_sft_ocr.jsonl"},
    "qlora": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": list(DEFAULT_TARGET_MODULES),
    },
    "training": {
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 1024,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "bf16": True,
        "fp16": False,
        "optim": "paged_adamw_8bit",
        "report_to": "none",
    },
}


def normalize_finetune_config(
    config: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge user config with QLoRA defaults and optional flat overrides."""

    merged = _deep_merge(DEFAULT_FINETUNE_CONFIG, config or {})
    overrides = overrides or {}
    for key, value in overrides.items():
        if value is None:
            continue
        if key in {"base_model", "output_dir"}:
            merged[key] = value
        elif key == "train_path":
            merged.setdefault("data", {})["train_path"] = value
        elif key == "load_in_4bit":
            merged.setdefault("qlora", {})["load_in_4bit"] = bool(value)
        elif key == "max_seq_length":
            merged.setdefault("training", {})["max_seq_length"] = int(value)
        elif key in merged.get("training", {}):
            merged.setdefault("training", {})[key] = value
        elif key in {"lora_r", "r"}:
            merged.setdefault("lora", {})["r"] = int(value)
        elif key in {"lora_alpha", "alpha"}:
            merged.setdefault("lora", {})["alpha"] = int(value)
        elif key in {"lora_dropout", "dropout"}:
            merged.setdefault("lora", {})["dropout"] = float(value)
    return merged


def validate_sft_record(record: dict[str, Any]) -> list[str]:
    """Return validation errors for one chat-format SFT record."""

    errors = []
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        return ["messages must contain exactly system, user, assistant entries"]

    expected_roles = ["system", "user", "assistant"]
    for index, (message, expected_role) in enumerate(zip(messages, expected_roles, strict=True)):
        if not isinstance(message, dict):
            errors.append(f"messages[{index}] must be an object")
            continue
        if message.get("role") != expected_role:
            errors.append(f"messages[{index}].role must be {expected_role}")
        if not str(message.get("content") or "").strip():
            errors.append(f"messages[{index}].content must be non-empty")
    return errors


def validate_sft_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize validation status for a list of SFT records."""

    invalid = []
    for index, record in enumerate(records):
        errors = validate_sft_record(record)
        if errors:
            invalid.append({"index": index, "errors": errors})
    return {
        "total": len(records),
        "valid": len(records) - len(invalid),
        "invalid": len(invalid),
        "invalid_examples": invalid[:10],
    }


def format_sft_example(example: dict[str, Any], tokenizer: Any) -> str:
    """Format one chat record using the model tokenizer chat template."""

    messages = example.get("messages", [])
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    return _plain_chat_format(messages)


def training_arguments_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Build kwargs for transformers.TrainingArguments without importing transformers."""

    training = config.get("training", {})
    kwargs = {
        "output_dir": str(config.get("output_dir", "experiments/runs/qwen25_lora")),
        "learning_rate": float(training.get("learning_rate", 2e-4)),
        "num_train_epochs": float(training.get("num_train_epochs", 3)),
        "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 8)),
        "logging_steps": int(training.get("logging_steps", 5)),
        "save_strategy": str(training.get("save_strategy", "epoch")),
        "bf16": bool(training.get("bf16", True)),
        "fp16": bool(training.get("fp16", False)),
        "optim": str(training.get("optim", "paged_adamw_8bit")),
        "report_to": training.get("report_to", "none"),
    }
    optional_fields = (
        "save_total_limit",
        "warmup_ratio",
        "weight_decay",
        "lr_scheduler_type",
        "gradient_checkpointing",
        "max_grad_norm",
    )
    for field in optional_fields:
        if field in training:
            kwargs[field] = training[field]
    return kwargs


def build_bitsandbytes_config(config: dict[str, Any]):
    """Create a Transformers BitsAndBytesConfig from qlora settings."""

    qlora = config.get("qlora", {})
    if not bool(qlora.get("load_in_4bit", True)):
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required for BitsAndBytesConfig.") from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch_dtype_from_name(
            qlora.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
        bnb_4bit_use_double_quant=bool(qlora.get("bnb_4bit_use_double_quant", False)),
    )


def build_lora_config(config: dict[str, Any]):
    """Create a PEFT LoRA config for causal language model fine-tuning."""

    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise RuntimeError("peft is required for LoRA fine-tuning.") from exc

    lora = config.get("lora", {})
    return LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora.get("target_modules") or DEFAULT_TARGET_MODULES),
        bias=str(lora.get("bias", "none")),
        task_type=TaskType.CAUSAL_LM,
    )


def load_tokenizer(model_name: str, cache_dir: str | None = None):
    """Load tokenizer and ensure a pad token exists."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to load the tokenizer.") from exc

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_for_qlora(
    config: dict[str, Any],
    cache_dir: str | None = None,
    device_map: str | None = "auto",
):
    """Load a causal LM with optional 4-bit quantization for QLoRA."""

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers is required to load the model.") from exc

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype_from_name(config.get("model_dtype", "auto")),
    }
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    if device_map:
        model_kwargs["device_map"] = device_map
    quantization_config = build_bitsandbytes_config(config)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(config["base_model"], **model_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if config.get("qlora", {}).get("load_in_4bit", True):
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("peft is required to prepare a k-bit model.") from exc
        model = prepare_model_for_kbit_training(model)
    return model


def load_json_sft_dataset(path: str | Path, limit: int | None = None):
    """Load chat-format JSONL records with datasets."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load SFT training data.") from exc

    dataset = load_dataset("json", data_files=str(path), split="train")
    if limit is not None:
        dataset = dataset.select(range(min(len(dataset), max(limit, 0))))
    return dataset


def create_sft_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    config: dict[str, Any],
):
    """Create a TRL SFTTrainer across common TRL versions."""

    try:
        from transformers import TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError("transformers and trl are required for QLoRA training.") from exc

    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None

    max_seq_length = int(config.get("training", {}).get("max_seq_length", 1024))
    args_class = SFTConfig or TrainingArguments
    args_kwargs = training_arguments_kwargs(config)
    if SFTConfig is not None:
        args_kwargs["max_length"] = max_seq_length
    args_signature = inspect.signature(args_class.__init__)
    args_kwargs = {
        key: value for key, value in args_kwargs.items() if key in args_signature.parameters
    }
    training_args = args_class(**args_kwargs)
    lora_config = build_lora_config(config)
    formatting_func = formatting_func_for_tokenizer(tokenizer)

    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "peft_config": lora_config,
        "formatting_func": formatting_func,
    }

    signature = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer

    if "max_seq_length" in signature.parameters:
        kwargs["max_seq_length"] = max_seq_length
    elif "max_length" in signature.parameters:
        kwargs["max_length"] = max_seq_length

    return SFTTrainer(**kwargs)


def formatting_func_for_tokenizer(tokenizer: Any) -> Callable[[dict[str, Any]], str | list[str]]:
    """Return a TRL formatting function for chat-format records."""

    def _format(example: dict[str, Any]) -> str | list[str]:
        messages = example.get("messages")
        if isinstance(messages, list) and messages and isinstance(messages[0], list):
            return [format_sft_example({"messages": item}, tokenizer) for item in messages]
        return format_sft_example(example, tokenizer)

    return _format


def torch_dtype_from_name(name: str | None):
    """Resolve a torch dtype string, leaving 'auto' untouched for Transformers."""

    if name is None or str(name).lower() == "auto":
        return "auto"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to resolve dtype names.") from exc

    normalized = str(name).lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[normalized]


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in base.items():
        result[key] = _deep_merge(value, {}) if isinstance(value, dict) else value
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _plain_chat_format(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role") or "user").upper()
        content = str(message.get("content") or "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
