"""Pair classifier training entry points for mention-level relation extraction."""

from __future__ import annotations

import inspect
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from gutbrainie.config import load_yaml
from gutbrainie.data.annotations import RELATION_PREDICATES
from gutbrainie.data.dataset import load_split
from gutbrainie.ner.gliner_runner import EXPERIMENT_QUALITIES, split_articles_by_pmid
from gutbrainie.ner.model_loading import load_sequence_classification_model
from gutbrainie.ner.tokenizers import load_fast_tokenizer
from gutbrainie.re.candidates import generate_relation_candidates

NO_RELATION = "no_relation"


class PairClassificationDataset:
    """Minimal list-backed dataset for Hugging Face Trainer."""

    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


def train_pair_classifier_experiment(
    data_root: str | Path,
    experiment: str,
    output_dir: str | Path,
    config_path: str | Path,
    validation_fraction: float = 0.15,
    seed: int = 13,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Fine-tune a sequence classifier for T621 mention-level RE."""
    if experiment not in EXPERIMENT_QUALITIES:
        raise ValueError(f"Unknown RE pair-classifier experiment '{experiment}'. Expected one of {sorted(EXPERIMENT_QUALITIES)}.")

    transformers = _import_transformers()
    config = load_yaml(config_path)
    model_name = model_name or str(config.get("model_name", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"))
    max_length = int(config.get("max_length", 512))
    negative_sampling_ratio = float(config.get("negative_sampling_ratio", 3))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_fast_tokenizer(transformers, model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": marker_tokens()})

    gold = load_split(data_root, "gold")
    gold_train_articles, gold_val_articles = split_articles_by_pmid(gold["articles"], validation_fraction, seed)
    gold_train_entities = _rows_for_pmids(gold["entities"], set(gold_train_articles["pmid"].astype(str)))
    gold_val_entities = _rows_for_pmids(gold["entities"], set(gold_val_articles["pmid"].astype(str)))
    gold_train_relations = _rows_for_pmids(gold["mention_relations"], set(gold_train_articles["pmid"].astype(str)))
    gold_val_relations = _rows_for_pmids(gold["mention_relations"], set(gold_val_articles["pmid"].astype(str)))

    train_articles = [gold_train_articles]
    train_entities = [gold_train_entities]
    train_relations = [gold_train_relations]
    included_qualities = ["gold"]

    for quality in EXPERIMENT_QUALITIES[experiment]:
        if quality == "gold":
            continue
        split = load_split(data_root, quality)
        train_articles.append(split["articles"])
        train_entities.append(split["entities"])
        train_relations.append(split["mention_relations"])
        included_qualities.append(quality)

    all_train_articles = pd.concat(train_articles, ignore_index=True)
    all_train_entities = pd.concat(train_entities, ignore_index=True)
    all_train_relations = pd.concat(train_relations, ignore_index=True)

    train_examples = build_pair_examples(
        all_train_articles,
        all_train_entities,
        all_train_relations,
        negative_sampling_ratio=negative_sampling_ratio,
        seed=seed,
    )
    validation_examples = build_pair_examples(
        gold_val_articles,
        gold_val_entities,
        gold_val_relations,
        negative_sampling_ratio=negative_sampling_ratio,
        seed=seed + 1,
    )

    label_list = build_relation_label_list(all_train_relations)
    label_to_id = {label: index for index, label in enumerate(label_list)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    train_features = tokenize_pair_examples(train_examples, tokenizer, label_to_id, max_length)
    validation_features = tokenize_pair_examples(validation_examples, tokenizer, label_to_id, max_length)

    model = load_sequence_classification_model(
        transformers,
        model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=bool(config.get("ignore_mismatched_sizes", False)),
    )
    model.resize_token_embeddings(len(tokenizer))

    training_args = _build_training_arguments(transformers.TrainingArguments, output_dir, config)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": PairClassificationDataset(train_features),
        "eval_dataset": PairClassificationDataset(validation_features),
        "data_collator": data_collator,
        "compute_metrics": _compute_pair_metrics(label_list),
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

    label_map = {
        "label_list": label_list,
        "label_to_id": label_to_id,
        "id_to_label": {str(key): value for key, value in id_to_label.items()},
    }
    (output_dir / "relation_label_map.json").write_text(json.dumps(label_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    metadata = {
        "model_name": model_name,
        "experiment": experiment,
        "included_train_qualities": included_qualities,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "negative_sampling_ratio": negative_sampling_ratio,
        "max_length": max_length,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "positive_train_examples": sum(example["label"] != NO_RELATION for example in train_examples),
        "output_dir": str(output_dir),
        "label_count": len(label_list),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def build_pair_examples(
    articles: pd.DataFrame,
    entities: pd.DataFrame,
    relations: pd.DataFrame,
    negative_sampling_ratio: float = 3,
    seed: int = 13,
) -> list[dict[str, Any]]:
    """Build marked-text relation examples from candidate pairs."""
    candidates = generate_relation_candidates(articles, entities, gold_relations=relations)
    article_lookup = {str(row["pmid"]): row for _, row in articles.iterrows()}
    examples: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        article = article_lookup.get(str(candidate["pmid"]))
        if article is None:
            continue
        text = candidate_to_marked_text(article, candidate)
        if not text:
            continue
        examples.append({"text": text, "label": str(candidate["predicate"]), "candidate": candidate.to_dict()})
    return sample_negative_examples(examples, negative_sampling_ratio=negative_sampling_ratio, seed=seed)


def build_relation_label_list(relations: pd.DataFrame) -> list[str]:
    """Return relation classifier labels with no_relation first."""
    labels = sorted(set(RELATION_PREDICATES) | (set(str(label) for label in relations["predicate"].dropna().unique()) if not relations.empty else set()))
    return [NO_RELATION, *[label for label in labels if label != NO_RELATION]]


def sample_negative_examples(
    examples: list[dict[str, Any]],
    negative_sampling_ratio: float = 3,
    seed: int = 13,
) -> list[dict[str, Any]]:
    """Keep all positive examples and sample a bounded number of no_relation examples."""
    positives = [example for example in examples if example["label"] != NO_RELATION]
    negatives = [example for example in examples if example["label"] == NO_RELATION]
    max_negatives = int(round(len(positives) * negative_sampling_ratio))
    if max_negatives <= 0 or len(negatives) <= max_negatives:
        sampled_negatives = negatives
    else:
        rng = random.Random(seed)
        sampled_negatives = rng.sample(negatives, max_negatives)
    return sorted([*positives, *sampled_negatives], key=lambda item: _example_sort_key(item))


def tokenize_pair_examples(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    label_to_id: dict[str, int],
    max_length: int,
) -> list[dict[str, Any]]:
    """Tokenize marked-text examples for sequence classification."""
    features: list[dict[str, Any]] = []
    for example in examples:
        tokenized = tokenizer(example["text"], truncation=True, max_length=max_length)
        tokenized["labels"] = label_to_id[str(example["label"])]
        features.append(tokenized)
    return features


def candidate_to_marked_text(article: pd.Series | dict[str, Any], candidate: pd.Series | dict[str, Any]) -> str | None:
    """Create marked pair-classification text for one candidate."""
    row = candidate if isinstance(candidate, dict) else candidate.to_dict()
    article_row = article if isinstance(article, dict) else article.to_dict()
    title = str(article_row.get("title", ""))
    abstract = str(article_row.get("abstract", ""))
    subject = _span_from_candidate(row, "subject")
    obj = _span_from_candidate(row, "object")

    if row["subject_location"] == row["object_location"]:
        location = str(row["subject_location"])
        text = title if location == "title" else abstract
        marked = insert_entity_markers(text, subject, obj)
        return marked

    title_span = subject if row["subject_location"] == "title" else obj
    abstract_span = subject if row["subject_location"] == "abstract" else obj
    marked_title = insert_single_entity_marker(title, title_span)
    marked_abstract = insert_single_entity_marker(abstract, abstract_span)
    if marked_title is None or marked_abstract is None:
        return None
    return f"{marked_title} [SEP] {marked_abstract}"


def insert_entity_markers(
    text: str,
    subject: dict[str, Any],
    obj: dict[str, Any],
) -> str | None:
    """Insert subject and object markers into one text segment."""
    subject_start, subject_end = int(subject["start_idx"]), int(subject["end_idx"]) + 1
    object_start, object_end = int(obj["start_idx"]), int(obj["end_idx"]) + 1
    if subject_start < object_end and object_start < subject_end:
        return None

    inserts = [
        (subject_start, _open_marker("SUBJ", subject["label"])),
        (subject_end, _close_marker("SUBJ", subject["label"])),
        (object_start, _open_marker("OBJ", obj["label"])),
        (object_end, _close_marker("OBJ", obj["label"])),
    ]
    return _apply_inserts(text, inserts)


def insert_single_entity_marker(text: str, span: dict[str, Any]) -> str | None:
    start_idx = int(span["start_idx"])
    end_idx = int(span["end_idx"]) + 1
    if start_idx < 0 or end_idx > len(text) or end_idx <= start_idx:
        return None
    return _apply_inserts(
        text,
        [
            (start_idx, _open_marker(str(span["role"]), span["label"])),
            (end_idx, _close_marker(str(span["role"]), span["label"])),
        ],
    )


def marker_tokens() -> list[str]:
    labels = [
        "anatomical location",
        "animal",
        "bacteria",
        "biomedical technique",
        "chemical",
        "DDF",
        "dietary supplement",
        "drug",
        "food",
        "gene",
        "human",
        "microbiome",
        "statistical technique",
    ]
    tokens: list[str] = []
    for role in ("SUBJ", "OBJ"):
        for label in labels:
            tokens.append(_open_marker(role, label))
            tokens.append(_close_marker(role, label))
    return tokens


def _span_from_candidate(candidate: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "role": "SUBJ" if role == "subject" else "OBJ",
        "start_idx": int(candidate[f"{role}_start_idx"]),
        "end_idx": int(candidate[f"{role}_end_idx"]),
        "label": str(candidate[f"{role}_label"]),
        "text_span": str(candidate[f"{role}_text_span"]),
    }


def _open_marker(role: str, label: str) -> str:
    return f"[{role}_{_marker_label(label)}]"


def _close_marker(role: str, label: str) -> str:
    return f"[/{role}_{_marker_label(label)}]"


def _marker_label(label: str) -> str:
    return str(label).upper().replace(" ", "_").replace("-", "_")


def _apply_inserts(text: str, inserts: list[tuple[int, str]]) -> str | None:
    if any(position < 0 or position > len(text) for position, _ in inserts):
        return None
    result = text
    for position, marker in sorted(inserts, key=lambda item: item[0], reverse=True):
        result = f"{result[:position]} {marker} {result[position:]}"
    return " ".join(result.split())


def _rows_for_pmids(df: pd.DataFrame, pmids: set[str]) -> pd.DataFrame:
    return df.loc[df["pmid"].astype(str).isin(pmids)].reset_index(drop=True)


def _example_sort_key(example: dict[str, Any]) -> tuple[Any, ...]:
    candidate = example.get("candidate", {})
    return (
        str(candidate.get("pmid", "")),
        str(candidate.get("subject_label", "")),
        str(candidate.get("subject_text_span", "")),
        str(example.get("label", "")),
        str(candidate.get("object_label", "")),
        str(candidate.get("object_text_span", "")),
    )


def _build_training_arguments(training_arguments_cls: Any, output_dir: Path, config: dict[str, Any]) -> Any:
    signature = inspect.signature(training_arguments_cls.__init__)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(config.get("learning_rate", 2e-5)),
        "per_device_train_batch_size": int(config.get("batch_size", 8)),
        "per_device_eval_batch_size": int(config.get("eval_batch_size", config.get("batch_size", 8))),
        "num_train_epochs": float(config.get("epochs", 5)),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.0)),
        "logging_steps": int(config.get("logging_steps", 50)),
        "save_total_limit": int(config.get("save_total_limit", 1)),
        "report_to": config.get("report_to", "none"),
    }
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = config.get("eval_strategy", "epoch")
    elif "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = config.get("eval_strategy", "epoch")
    if "save_strategy" in signature.parameters:
        kwargs["save_strategy"] = config.get("save_strategy", "epoch")
    if "metric_for_best_model" in signature.parameters and config.get("metric_for_best_model"):
        kwargs["metric_for_best_model"] = config["metric_for_best_model"]
    if "load_best_model_at_end" in signature.parameters:
        kwargs["load_best_model_at_end"] = bool(config.get("load_best_model_at_end", False))
    if "max_steps" in signature.parameters and config.get("max_steps") is not None:
        kwargs["max_steps"] = int(config["max_steps"])
    if "gradient_accumulation_steps" in signature.parameters:
        kwargs["gradient_accumulation_steps"] = int(config.get("gradient_accumulation_steps", 1))
    if "use_cpu" in signature.parameters and config.get("use_cpu") is not None:
        kwargs["use_cpu"] = bool(config.get("use_cpu"))
    elif "no_cuda" in signature.parameters and config.get("use_cpu") is not None:
        kwargs["no_cuda"] = bool(config.get("use_cpu"))
    if "dataloader_num_workers" in signature.parameters:
        kwargs["dataloader_num_workers"] = int(config.get("dataloader_num_workers", 0))
    if "dataloader_pin_memory" in signature.parameters:
        kwargs["dataloader_pin_memory"] = bool(config.get("dataloader_pin_memory", False))
    return training_arguments_cls(**{key: value for key, value in kwargs.items() if key in signature.parameters})


def _compute_pair_metrics(label_list: list[str]):
    no_relation_id = label_list.index(NO_RELATION)

    def compute(eval_prediction: Any) -> dict[str, float]:
        import numpy as np

        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        tp = int(((predictions == labels) & (labels != no_relation_id)).sum())
        fp = int(((predictions != no_relation_id) & (predictions != labels)).sum())
        fn = int(((labels != no_relation_id) & (predictions != labels)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = float((predictions == labels).mean()) if len(labels) else 0.0
        return {
            "micro_precision_without_no_relation": precision,
            "micro_recall_without_no_relation": recall,
            "micro_f1_without_no_relation": f1,
            "accuracy": accuracy,
        }

    return compute


def _import_transformers():
    try:
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'transformers'. Install project dependencies with: make install"
        ) from exc
    return transformers
