"""Model loading helpers for token-classification NER."""

from __future__ import annotations

from typing import Any


def load_token_classification_model(
    transformers: Any,
    model_name_or_path: str,
    num_labels: int | None = None,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    ignore_mismatched_sizes: bool = False,
):
    """Load a token-classification model with a BERT fallback for old configs."""
    kwargs: dict[str, Any] = {"ignore_mismatched_sizes": ignore_mismatched_sizes}
    if num_labels is not None:
        kwargs["num_labels"] = num_labels
    if id2label is not None:
        kwargs["id2label"] = id2label
    if label2id is not None:
        kwargs["label2id"] = label2id

    try:
        return transformers.AutoModelForTokenClassification.from_pretrained(model_name_or_path, **kwargs)
    except ValueError as exc:
        if not _should_try_bert_fallback(model_name_or_path, exc):
            raise
        if not hasattr(transformers, "BertConfig") or not hasattr(transformers, "BertForTokenClassification"):
            raise

        config = transformers.BertConfig.from_pretrained(model_name_or_path)
        if num_labels is not None:
            config.num_labels = num_labels
        if id2label is not None:
            config.id2label = id2label
        if label2id is not None:
            config.label2id = label2id
        return transformers.BertForTokenClassification.from_pretrained(
            model_name_or_path,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )


def load_sequence_classification_model(
    transformers: Any,
    model_name_or_path: str,
    num_labels: int | None = None,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    ignore_mismatched_sizes: bool = False,
):
    """Load a sequence-classification model with a BERT fallback for old configs."""
    kwargs: dict[str, Any] = {"ignore_mismatched_sizes": ignore_mismatched_sizes}
    if num_labels is not None:
        kwargs["num_labels"] = num_labels
    if id2label is not None:
        kwargs["id2label"] = id2label
    if label2id is not None:
        kwargs["label2id"] = label2id

    try:
        return transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
    except ValueError as exc:
        if not _should_try_bert_fallback(model_name_or_path, exc):
            raise
        if not hasattr(transformers, "BertConfig") or not hasattr(transformers, "BertForSequenceClassification"):
            raise

        config = transformers.BertConfig.from_pretrained(model_name_or_path)
        if num_labels is not None:
            config.num_labels = num_labels
        if id2label is not None:
            config.id2label = id2label
        if label2id is not None:
            config.label2id = label2id
        return transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )


def _should_try_bert_fallback(model_name_or_path: str, exc: ValueError) -> bool:
    model_key = str(model_name_or_path).lower()
    error = str(exc).lower()
    is_bert_like = any(marker in model_key for marker in ("bert", "scibert", "biobert", "biomedbert"))
    missing_model_type = "model_type" in error or "unrecognized model" in error
    return is_bert_like and missing_model_type
