"""GLiNER experiment runner."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from gutbrainie.config import load_yaml
from gutbrainie.data.annotations import deduplicate_entities, load_entities_csv
from gutbrainie.data.articles import load_articles_csv
from gutbrainie.data.dataset import load_split
from gutbrainie.data.offsets import resolve_exclusive_end
from gutbrainie.submission.export_t611 import entities_to_t611_json

DEFAULT_LABELS = [
    "anatomical location",
    "animal",
    "biomedical technique",
    "bacteria",
    "chemical",
    "dietary supplement",
    "DDF",
    "drug",
    "food",
    "gene",
    "human",
    "microbiome",
    "statistical technique",
]
EXPERIMENT_QUALITIES = {
    "gold": ("gold",),
    "gold_silver": ("gold", "silver"),
    "gold_silver_silver_2025": ("gold", "silver", "silver_2025"),
}


def convert_to_gliner_examples(articles: pd.DataFrame, entities: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert article/entity annotations to GLiNER examples.

    Titles and abstracts are emitted as separate examples so offsets remain local
    to the original GutBrainIE fields.
    """
    _require_columns(articles, ["pmid", "title", "abstract"], "articles")
    _require_columns(entities, ["pmid", "start_idx", "end_idx", "location", "label"], "entities")

    entities = deduplicate_entities(entities)
    article_texts = {
        (str(row["pmid"]), location): str(row[location])
        for _, row in articles.iterrows()
        for location in ("title", "abstract")
    }
    grouped: dict[tuple[str, str], list[list[Any]]] = defaultdict(list)
    for _, row in entities.iterrows():
        pmid = str(row["pmid"])
        location = str(row["location"])
        text_span = str(row.get("text_span", ""))
        text = article_texts.get((pmid, location), "")
        start_idx = int(row["start_idx"])
        end_idx = resolve_exclusive_end(text, start_idx, int(row["end_idx"]), text_span)
        grouped[(str(row["pmid"]), str(row["location"]))].append(
            [start_idx, end_idx, str(row["label"])]
        )

    examples: list[dict[str, Any]] = []
    for _, article in articles.iterrows():
        pmid = str(article["pmid"])
        for location in ("title", "abstract"):
            spans = sorted(grouped.get((pmid, location), []), key=lambda span: (span[0], span[1], span[2]))
            examples.append(
                {
                    "pmid": pmid,
                    "location": location,
                    "text": str(article[location]),
                    "label": spans,
                }
            )
    return examples


def prepare_gliner_experiment_data(
    data_root: str | Path,
    experiment: str,
    output_dir: str | Path,
    validation_fraction: float = 0.15,
    seed: int = 13,
) -> dict[str, Any]:
    """Create train/validation JSONL files for a GLiNER experiment."""
    if experiment not in EXPERIMENT_QUALITIES:
        raise ValueError(f"Unknown GLiNER experiment '{experiment}'. Expected one of {sorted(EXPERIMENT_QUALITIES)}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold = load_split(data_root, "gold")
    gold_train_articles, gold_val_articles = split_articles_by_pmid(gold["articles"], validation_fraction, seed)
    gold_train_entities = _entities_for_pmids(gold["entities"], set(gold_train_articles["pmid"]))
    gold_val_entities = _entities_for_pmids(gold["entities"], set(gold_val_articles["pmid"]))

    train_examples = convert_to_gliner_examples(gold_train_articles, gold_train_entities)
    validation_examples = convert_to_gliner_examples(gold_val_articles, gold_val_entities)
    included_qualities = ["gold"]

    for quality in EXPERIMENT_QUALITIES[experiment]:
        if quality == "gold":
            continue
        split = load_split(data_root, quality)
        train_examples.extend(convert_to_gliner_examples(split["articles"], split["entities"]))
        included_qualities.append(quality)

    train_path = output_dir / f"gliner_{experiment}_train.jsonl"
    validation_path = output_dir / f"gliner_{experiment}_validation.jsonl"
    metadata_path = output_dir / f"gliner_{experiment}_metadata.json"
    write_jsonl(train_examples, train_path)
    write_jsonl(validation_examples, validation_path)

    metadata = {
        "experiment": experiment,
        "included_train_qualities": included_qualities,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "gold_train_articles": len(gold_train_articles),
        "gold_validation_articles": len(gold_val_articles),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def split_articles_by_pmid(
    articles: pd.DataFrame,
    validation_fraction: float = 0.15,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically split articles by PMID into train/validation partitions."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    pmids = sorted(str(pmid) for pmid in articles["pmid"].unique())
    rng = random.Random(seed)
    rng.shuffle(pmids)
    validation_size = max(1, round(len(pmids) * validation_fraction))
    validation_pmids = set(pmids[:validation_size])
    is_validation = articles["pmid"].astype(str).isin(validation_pmids)
    return articles.loc[~is_validation].reset_index(drop=True), articles.loc[is_validation].reset_index(drop=True)


def predict_gliner_to_json(
    model_name_or_path: str,
    articles_path: str | Path,
    output_path: str | Path,
    labels: list[str] | None = None,
    threshold: float = 0.5,
    batch_size: int = 8,
    max_len: int | None = None,
) -> pd.DataFrame:
    """Run GLiNER prediction over article titles/abstracts and write T611 JSON."""
    GLiNER = _import_gliner_model()
    from_pretrained_kwargs = {"max_length": max_len} if max_len else {}
    model = GLiNER.from_pretrained(model_name_or_path, **from_pretrained_kwargs)
    articles = load_articles_csv(articles_path)
    labels = labels or DEFAULT_LABELS

    rows: list[dict[str, Any]] = []
    for _, article in articles.iterrows():
        pmid = str(article["pmid"])
        for location in ("title", "abstract"):
            text = str(article[location])
            predictions = model.predict_entities(text, labels, threshold=threshold)
            rows.extend(_gliner_predictions_to_rows(pmid, location, text, predictions))

    prediction_df = pd.DataFrame(rows, columns=["pmid", "start_idx", "end_idx", "location", "text_span", "label"])
    payload = entities_to_t611_json(prediction_df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return prediction_df


def train_gliner_model(
    model_name: str,
    train_path: str | Path,
    validation_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> Path:
    """Fine-tune GLiNER if the installed package exposes a compatible train API."""
    GLiNER = _import_gliner_model()
    config = load_yaml(config_path) if config_path else {}
    max_len = config.get("max_len") or config.get("max_length")
    from_pretrained_kwargs = {"max_length": int(max_len)} if max_len else {}
    model = GLiNER.from_pretrained(model_name, **from_pretrained_kwargs)
    train_examples = gliner_examples_to_training_dataset(read_jsonl(train_path), keep_empty=False)
    validation_examples = gliner_examples_to_training_dataset(read_jsonl(validation_path), keep_empty=False)
    if not train_examples:
        raise ValueError(f"No non-empty GLiNER training examples after span alignment: {train_path}")
    if not validation_examples:
        raise ValueError(f"No non-empty GLiNER validation examples after span alignment: {validation_path}")

    training_kwargs = {
        "learning_rate": config.get("learning_rate", 2e-5),
        "per_device_train_batch_size": config.get("batch_size", 8),
        "per_device_eval_batch_size": config.get("batch_size", 8),
        "num_train_epochs": config.get("epochs", 5),
        "max_steps": config.get("max_steps", -1),
        "save_steps": config.get("save_steps", 500),
        "logging_steps": config.get("logging_steps", 50),
        "use_cpu": config.get("use_cpu", False),
        "dataloader_num_workers": config.get("dataloader_num_workers", 0),
        "dataloader_pin_memory": config.get("dataloader_pin_memory", False),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
        "save_total_limit": config.get("save_total_limit", 2),
        "warmup_ratio": config.get("warmup_ratio", None),
        "warmup_steps": config.get("warmup_steps", None),
        "report_to": config.get("report_to", "none"),
    }
    training_kwargs = {key: value for key, value in training_kwargs.items() if value is not None}

    if hasattr(model, "train_model"):
        model.train_model(
            train_examples,
            validation_examples,
            output_dir=output_dir,
            **training_kwargs,
        )
    else:
        raise RuntimeError(
            "The installed gliner package does not expose model.train_model(...). "
            "Use a GLiNER version with fine-tuning support, or train with the official baseline repo."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(output_dir))
    else:
        raise RuntimeError("The trained GLiNER model does not expose save_pretrained(...).")
    return output_dir


def gliner_examples_to_training_dataset(
    examples: list[dict[str, Any]],
    keep_empty: bool = False,
) -> list[dict[str, Any]]:
    """Convert char-offset GLiNER examples to this package's word-level training shape."""
    dataset: list[dict[str, Any]] = []
    for example in examples:
        tokens, token_spans = _whitespace_tokenize_with_offsets(str(example["text"]))
        ner: list[tuple[int, int, str]] = []
        for start_idx, end_idx, label in example.get("label", []):
            word_span = _char_span_to_word_span(int(start_idx), int(end_idx), token_spans)
            if word_span is None:
                continue
            ner.append((word_span[0], word_span[1], str(label)))
        if not ner and not keep_empty:
            continue
        dataset.append({"tokenized_text": tokens, "ner": ner})
    return dataset


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _gliner_predictions_to_rows(
    pmid: str,
    location: str,
    text: str,
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ranked = sorted(
        predictions,
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -(int(item.get("end", item.get("end_idx"))) - int(item.get("start", item.get("start_idx")))),
            str(item.get("label", "")),
        ),
    )
    occupied: list[tuple[int, int]] = []
    for prediction in ranked:
        start_idx = int(prediction.get("start", prediction.get("start_idx")))
        end_idx = int(prediction.get("end", prediction.get("end_idx")))
        if start_idx < 0 or end_idx <= start_idx or end_idx > len(text):
            continue
        span = (start_idx, end_idx)
        if any(_overlaps(span, kept) for kept in occupied):
            continue
        occupied.append(span)
        rows.append(
            {
                "pmid": pmid,
                "start_idx": start_idx,
                "end_idx": end_idx - 1,
                "location": location,
                "text_span": text[start_idx:end_idx],
                "label": str(prediction["label"]),
            }
        )
    return sorted(rows, key=lambda row: (row["start_idx"], row["end_idx"], row["label"]))


def _entities_for_pmids(entities: pd.DataFrame, pmids: set[str]) -> pd.DataFrame:
    return entities.loc[entities["pmid"].astype(str).isin(pmids)].reset_index(drop=True)


def _whitespace_tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    matches = list(re.finditer(r"\S+", text))
    return [match.group(0) for match in matches], [(match.start(), match.end()) for match in matches]


def _char_span_to_word_span(
    start_idx: int,
    end_idx: int,
    token_spans: list[tuple[int, int]],
) -> tuple[int, int] | None:
    start_word = None
    end_word = None
    for index, (token_start, token_end) in enumerate(token_spans):
        if token_start == start_idx:
            start_word = index
        if token_end == end_idx:
            end_word = index
    if start_word is None or end_word is None or end_word < start_word:
        return None
    return start_word, end_word


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _import_gliner_model():
    try:
        from gliner import GLiNER
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing optional dependency 'gliner'. Install it with: make install-gliner"
        ) from exc
    return GLiNER


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
