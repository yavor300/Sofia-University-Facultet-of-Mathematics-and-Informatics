"""Pair classifier prediction entry points for mention-level relation extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from gutbrainie.data.articles import load_articles_csv
from gutbrainie.ner.model_loading import load_sequence_classification_model
from gutbrainie.ner.tokenizers import load_fast_tokenizer
from gutbrainie.re.candidates import generate_relation_candidates
from gutbrainie.re.relation_schema import valid_predicates
from gutbrainie.re.rule_baseline import MENTION_RELATION_COLUMNS, deduplicate_mention_relations, load_entities
from gutbrainie.re.train_pair_classifier import NO_RELATION, candidate_to_marked_text
from gutbrainie.submission.export_t621 import mention_relations_to_t621_json


def predict_pair_classifier_to_json(
    model_path: str | Path,
    articles_path: str | Path,
    entities_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.5,
    batch_size: int = 8,
    max_length: int = 512,
    use_cpu: bool = False,
) -> pd.DataFrame:
    """Run a trained pair classifier and write T621 JSON."""
    transformers, torch = _import_runtime()
    model_path = _resolve_model_path(Path(model_path))
    tokenizer = load_fast_tokenizer(transformers, str(model_path))
    model = load_sequence_classification_model(transformers, str(model_path))
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    model.eval()
    id_to_label = _load_id_to_label(model_path, model)

    articles = load_articles_csv(articles_path)
    entities = load_entities(entities_path)
    candidates = generate_relation_candidates(articles, entities)
    article_lookup = {str(row["pmid"]): row for _, row in articles.iterrows()}

    examples: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        article = article_lookup.get(str(candidate["pmid"]))
        if article is None:
            continue
        text = candidate_to_marked_text(article, candidate)
        if not text:
            continue
        examples.append({"text": text, "candidate": candidate.to_dict()})

    rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokenized = tokenizer(
            [example["text"] for example in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu()
        scores, predicted_ids = probabilities.max(dim=-1)
        for example, score, predicted_id in zip(batch, scores.tolist(), predicted_ids.tolist(), strict=False):
            predicate = id_to_label[int(predicted_id)]
            if predicate == NO_RELATION or float(score) < threshold:
                continue
            candidate = example["candidate"]
            if predicate not in valid_predicates(candidate["subject_label"], candidate["object_label"]):
                continue
            rows.append(
                {
                    "pmid": str(candidate["pmid"]),
                    "subject_text_span": str(candidate["subject_text_span"]),
                    "subject_label": str(candidate["subject_label"]),
                    "predicate": predicate,
                    "object_text_span": str(candidate["object_text_span"]),
                    "object_label": str(candidate["object_label"]),
                }
            )

    predictions = deduplicate_mention_relations(pd.DataFrame(rows, columns=MENTION_RELATION_COLUMNS))
    payload = mention_relations_to_t621_json(predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return predictions


def _load_id_to_label(model_path: Path, model: Any) -> dict[int, str]:
    label_map_path = model_path / "relation_label_map.json"
    if label_map_path.exists():
        payload = json.loads(label_map_path.read_text(encoding="utf-8"))
        return {int(key): value for key, value in payload["id_to_label"].items()}
    return {int(key): value for key, value in model.config.id2label.items()}


def _resolve_model_path(model_path: Path) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(
            f"RE pair-classifier model directory does not exist: {model_path}\n"
            "Train it first, for example:\n"
            "  make train-re-pair-classifier RE_PAIR_EXPERIMENT=gold"
        )
    if _looks_like_saved_model(model_path):
        return model_path
    checkpoints = sorted(
        [path for path in model_path.glob("checkpoint-*") if path.is_dir()],
        key=_checkpoint_sort_key,
    )
    for checkpoint in reversed(checkpoints):
        if _looks_like_saved_model(checkpoint):
            return checkpoint
    raise FileNotFoundError(
        f"No saved Hugging Face model files found in {model_path} or checkpoint-* subdirectories."
    )


def _looks_like_saved_model(path: Path) -> bool:
    has_config = (path / "config.json").exists()
    has_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    return has_config and has_weights


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("checkpoint-")
    return (int(suffix) if suffix.isdigit() else -1, path.name)


def _import_runtime():
    try:
        import torch
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency '{exc.name}'. Install project dependencies with: make install"
        ) from exc
    return transformers, torch
