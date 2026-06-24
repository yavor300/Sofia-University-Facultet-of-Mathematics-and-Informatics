"""Token-classification NER training entry points."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd

from gutbrainie.config import load_yaml
from gutbrainie.data.dataset import load_split
from gutbrainie.ner.bio_tags import (
    articles_entities_to_token_features,
    build_bio_label_list,
    build_label_maps,
    entity_labels_from_dataframe,
)
from gutbrainie.ner.gliner_runner import EXPERIMENT_QUALITIES, split_articles_by_pmid
from gutbrainie.ner.tokenizers import load_fast_tokenizer


class TokenClassificationDataset:
    """Minimal list-backed dataset for Hugging Face Trainer."""

    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        feature = self.features[index]
        return {
            key: value
            for key, value in feature.items()
            if key not in {"pmid", "location"}
        }


def train_token_classifier_experiment(
    data_root: str | Path,
    experiment: str,
    output_dir: str | Path,
    config_path: str | Path,
    validation_fraction: float = 0.15,
    seed: int = 13,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Fine-tune a Hugging Face token-classification model for T611."""
    if experiment not in EXPERIMENT_QUALITIES:
        raise ValueError(f"Unknown token-classifier experiment '{experiment}'. Expected one of {sorted(EXPERIMENT_QUALITIES)}.")

    transformers = _import_transformers()
    config = load_yaml(config_path)
    model_name = model_name or str(config.get("model_name", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"))
    max_length = int(config.get("max_length", 512))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_fast_tokenizer(transformers, model_name)

    gold = load_split(data_root, "gold")
    gold_train_articles, gold_val_articles = split_articles_by_pmid(gold["articles"], validation_fraction, seed)
    gold_train_entities = _entities_for_pmids(gold["entities"], set(gold_train_articles["pmid"].astype(str)))
    gold_val_entities = _entities_for_pmids(gold["entities"], set(gold_val_articles["pmid"].astype(str)))

    train_articles = [gold_train_articles]
    train_entities = [gold_train_entities]
    included_qualities = ["gold"]

    for quality in EXPERIMENT_QUALITIES[experiment]:
        if quality == "gold":
            continue
        split = load_split(data_root, quality)
        train_articles.append(split["articles"])
        train_entities.append(split["entities"])
        included_qualities.append(quality)

    all_train_articles = pd.concat(train_articles, ignore_index=True)
    all_train_entities = pd.concat(train_entities, ignore_index=True)
    label_list = build_bio_label_list(entity_labels_from_dataframe(all_train_entities))
    label_to_id, id_to_label = build_label_maps(label_list)

    train_features = articles_entities_to_token_features(
        all_train_articles,
        all_train_entities,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=max_length,
    )
    validation_features = articles_entities_to_token_features(
        gold_val_articles,
        gold_val_entities,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=max_length,
    )

    model = transformers.AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=bool(config.get("ignore_mismatched_sizes", False)),
    )
    data_collator = transformers.DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = _build_training_arguments(transformers.TrainingArguments, output_dir, config)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": TokenClassificationDataset(train_features),
        "eval_dataset": TokenClassificationDataset(validation_features),
        "data_collator": data_collator,
        "compute_metrics": _compute_token_metrics,
    }
    trainer_signature = inspect.signature(transformers.Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = transformers.Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    label_map_path = output_dir / "label_map.json"
    label_map_path.write_text(
        json.dumps(
            {
                "label_list": label_list,
                "label_to_id": label_to_id,
                "id_to_label": {str(key): value for key, value in id_to_label.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "model_name": model_name,
        "experiment": experiment,
        "included_train_qualities": included_qualities,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "max_length": max_length,
        "train_examples": len(train_features),
        "validation_examples": len(validation_features),
        "gold_train_articles": len(gold_train_articles),
        "gold_validation_articles": len(gold_val_articles),
        "label_count": len(label_list),
        "output_dir": str(output_dir),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def _entities_for_pmids(entities: pd.DataFrame, pmids: set[str]) -> pd.DataFrame:
    return entities.loc[entities["pmid"].astype(str).isin(pmids)].reset_index(drop=True)


def _build_training_arguments(training_arguments_cls: Any, output_dir: Path, config: dict[str, Any]) -> Any:
    signature = inspect.signature(training_arguments_cls.__init__)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(config.get("learning_rate", 2e-5)),
        "per_device_train_batch_size": int(config.get("batch_size", 8)),
        "per_device_eval_batch_size": int(config.get("eval_batch_size", config.get("batch_size", 8))),
        "num_train_epochs": float(config.get("epochs", 3)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "logging_steps": int(config.get("logging_steps", 50)),
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "report_to": config.get("report_to", "none"),
    }
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = config.get("eval_strategy", "epoch")
    elif "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = config.get("eval_strategy", "epoch")
    if "save_strategy" in signature.parameters:
        kwargs["save_strategy"] = config.get("save_strategy", "epoch")
    if "load_best_model_at_end" in signature.parameters:
        kwargs["load_best_model_at_end"] = bool(config.get("load_best_model_at_end", False))
    if "metric_for_best_model" in signature.parameters and config.get("metric_for_best_model"):
        kwargs["metric_for_best_model"] = config["metric_for_best_model"]
    if "greater_is_better" in signature.parameters and config.get("greater_is_better") is not None:
        kwargs["greater_is_better"] = bool(config["greater_is_better"])
    if "max_steps" in signature.parameters and config.get("max_steps") is not None:
        kwargs["max_steps"] = int(config["max_steps"])
    if "gradient_accumulation_steps" in signature.parameters:
        kwargs["gradient_accumulation_steps"] = int(config.get("gradient_accumulation_steps", 1))
    if "use_cpu" in signature.parameters and config.get("use_cpu") is not None:
        kwargs["use_cpu"] = bool(config.get("use_cpu"))
    elif "no_cuda" in signature.parameters and config.get("use_cpu") is not None:
        kwargs["no_cuda"] = bool(config.get("use_cpu"))
    if "fp16" in signature.parameters:
        kwargs["fp16"] = bool(config.get("fp16", False))
    if "dataloader_num_workers" in signature.parameters:
        kwargs["dataloader_num_workers"] = int(config.get("dataloader_num_workers", 0))
    if "dataloader_pin_memory" in signature.parameters:
        kwargs["dataloader_pin_memory"] = bool(config.get("dataloader_pin_memory", False))

    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return training_arguments_cls(**filtered)


def _compute_token_metrics(eval_prediction: Any) -> dict[str, float]:
    import numpy as np

    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    if not mask.any():
        return {"token_accuracy": 0.0}
    return {"token_accuracy": float((predictions[mask] == labels[mask]).mean())}


def _import_transformers():
    try:
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'transformers'. Install project dependencies with: make install"
        ) from exc
    return transformers
