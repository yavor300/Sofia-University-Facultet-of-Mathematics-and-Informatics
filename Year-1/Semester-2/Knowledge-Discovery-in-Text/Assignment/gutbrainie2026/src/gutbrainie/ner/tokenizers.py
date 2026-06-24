"""Tokenizer loading helpers for offset-aware NER."""

from __future__ import annotations

from typing import Any


def load_fast_tokenizer(transformers: Any, model_name_or_path: str):
    """Load a fast tokenizer, with a BERT-specific fallback for older checkpoints."""
    errors: list[str] = []
    loaders = []
    model_key = str(model_name_or_path).lower()
    is_bert_like = any(marker in model_key for marker in ("bert", "scibert", "biobert", "biomedbert"))
    if is_bert_like and hasattr(transformers, "BertTokenizerFast"):
        loaders.append(("BertTokenizerFast", lambda: transformers.BertTokenizerFast.from_pretrained(model_name_or_path)))
    loaders.append(("AutoTokenizer", lambda: transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)))
    if not is_bert_like and hasattr(transformers, "BertTokenizerFast"):
        loaders.append(("BertTokenizerFast", lambda: transformers.BertTokenizerFast.from_pretrained(model_name_or_path)))

    for loader_name, loader in loaders:
        try:
            tokenizer = loader()
        except Exception as exc:  # noqa: BLE001 - preserve the upstream tokenizer failure details.
            errors.append(f"{loader_name}: {exc}")
            continue
        if getattr(tokenizer, "is_fast", False):
            return tokenizer
        errors.append(f"{loader_name}: loaded tokenizer is not fast")

    details = "\n".join(f"- {error}" for error in errors)
    raise ValueError(
        "Token-classification NER requires a fast tokenizer with offset mappings. "
        f"Could not load one for '{model_name_or_path}'.\n{details}"
    )
