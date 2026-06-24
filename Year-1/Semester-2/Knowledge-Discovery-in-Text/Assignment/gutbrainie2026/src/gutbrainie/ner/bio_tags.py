"""BIO tag conversion helpers for token-classification NER."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from gutbrainie.data.annotations import deduplicate_entities
from gutbrainie.data.offsets import resolve_exclusive_end

IGNORE_INDEX = -100

DEFAULT_ENTITY_LABELS = [
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


def build_bio_label_list(entity_labels: list[str] | None = None) -> list[str]:
    """Return the canonical BIO label list for a set of entity labels."""
    labels = entity_labels or DEFAULT_ENTITY_LABELS
    ordered_labels = sorted(dict.fromkeys(str(label) for label in labels))
    bio_labels = ["O"]
    for label in ordered_labels:
        bio_labels.append(f"B-{label}")
        bio_labels.append(f"I-{label}")
    return bio_labels


def build_label_maps(label_list: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Build forward and reverse label maps."""
    label_to_id = {label: index for index, label in enumerate(label_list)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return label_to_id, id_to_label


def entity_labels_from_dataframe(entities: pd.DataFrame) -> list[str]:
    """Return deterministic entity labels observed in an annotation frame."""
    if entities.empty or "label" not in entities.columns:
        return DEFAULT_ENTITY_LABELS
    observed = sorted(str(label) for label in entities["label"].dropna().unique())
    return observed or DEFAULT_ENTITY_LABELS


def articles_entities_to_token_features(
    articles: pd.DataFrame,
    entities: pd.DataFrame,
    tokenizer: Any,
    label_to_id: Mapping[str, int],
    max_length: int = 512,
) -> list[dict[str, Any]]:
    """Convert articles and char-level annotations to token-classification features.

    Title and abstract are emitted separately so offsets remain local to their
    original article field.
    """
    entities = deduplicate_entities(entities)
    grouped = {
        key: group.reset_index(drop=True)
        for key, group in entities.groupby([entities["pmid"].astype(str), entities["location"].astype(str)])
    }

    features: list[dict[str, Any]] = []
    for _, article in articles.iterrows():
        pmid = str(article["pmid"])
        for location in ("title", "abstract"):
            text = str(article[location])
            location_entities = grouped.get((pmid, location), pd.DataFrame(columns=entities.columns))
            feature = tokenize_text_with_bio_labels(
                text=text,
                entities=location_entities,
                tokenizer=tokenizer,
                label_to_id=label_to_id,
                max_length=max_length,
            )
            feature["pmid"] = pmid
            feature["location"] = location
            features.append(feature)
    return features


def tokenize_text_with_bio_labels(
    text: str,
    entities: pd.DataFrame,
    tokenizer: Any,
    label_to_id: Mapping[str, int],
    max_length: int = 512,
) -> dict[str, Any]:
    """Tokenize one text and align local entity offsets to BIO label IDs."""
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = _offsets_as_tuples(tokenized["offset_mapping"])
    label_ids = align_entities_to_bio_ids(text, offsets, entities, label_to_id)

    feature = {
        key: value
        for key, value in tokenized.items()
        if key != "offset_mapping"
    }
    feature["labels"] = label_ids
    return feature


def align_entities_to_bio_ids(
    text: str,
    offsets: list[tuple[int, int]],
    entities: pd.DataFrame,
    label_to_id: Mapping[str, int],
) -> list[int]:
    """Align character spans to token offsets and return token-level label IDs."""
    labels = [IGNORE_INDEX if _is_special_offset(offset) else label_to_id["O"] for offset in offsets]
    if entities.empty:
        return labels

    spans = []
    for _, entity in entities.iterrows():
        start_idx = int(entity["start_idx"])
        text_span = str(entity.get("text_span", ""))
        end_idx = resolve_exclusive_end(text, start_idx, int(entity["end_idx"]), text_span)
        label = str(entity["label"])
        if f"B-{label}" not in label_to_id or f"I-{label}" not in label_to_id:
            continue
        if start_idx < 0 or end_idx <= start_idx or end_idx > len(text):
            continue
        spans.append((start_idx, end_idx, label))

    spans.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2]))
    for start_idx, end_idx, label in spans:
        token_indices = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if not _is_special_offset((token_start, token_end))
            and token_start < end_idx
            and start_idx < token_end
        ]
        if not token_indices:
            continue
        if any(labels[index] not in {label_to_id["O"], IGNORE_INDEX} for index in token_indices):
            continue
        labels[token_indices[0]] = label_to_id[f"B-{label}"]
        for index in token_indices[1:]:
            labels[index] = label_to_id[f"I-{label}"]
    return labels


def decode_bio_spans(
    text: str,
    offsets: list[tuple[int, int]],
    label_ids: list[int],
    id_to_label: Mapping[int, str] | Mapping[str, str],
) -> list[dict[str, Any]]:
    """Decode token BIO predictions to inclusive character-span entities."""
    normalized_id_to_label = {int(key): value for key, value in id_to_label.items()}
    spans: list[dict[str, Any]] = []
    active_label: str | None = None
    active_start: int | None = None
    active_end: int | None = None

    def close_active() -> None:
        nonlocal active_label, active_start, active_end
        if active_label is not None and active_start is not None and active_end is not None and active_end > active_start:
            spans.append(
                {
                    "start_idx": active_start,
                    "end_idx": active_end - 1,
                    "text_span": text[active_start:active_end],
                    "label": active_label,
                }
            )
        active_label = None
        active_start = None
        active_end = None

    for offset, label_id in zip(offsets, label_ids, strict=False):
        token_start, token_end = offset
        if _is_special_offset(offset) or label_id == IGNORE_INDEX:
            close_active()
            continue

        label_name = normalized_id_to_label.get(int(label_id), "O")
        if label_name == "O" or "-" not in label_name:
            close_active()
            continue

        prefix, entity_label = label_name.split("-", 1)
        starts_new = prefix == "B" or active_label != entity_label
        if prefix not in {"B", "I"}:
            close_active()
            continue
        if starts_new:
            close_active()
            active_label = entity_label
            active_start = token_start
        active_end = token_end

    close_active()
    return spans


def _offsets_as_tuples(offsets: Any) -> list[tuple[int, int]]:
    return [(int(start), int(end)) for start, end in offsets]


def _is_special_offset(offset: tuple[int, int]) -> bool:
    return offset == (0, 0)
