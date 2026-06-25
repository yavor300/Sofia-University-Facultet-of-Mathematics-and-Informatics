"""Convert GutBrainIE annotations to the official ATLOP/DocRED-like format."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from gutbrainie.data.annotations import deduplicate_entities
from gutbrainie.data.dataset import load_split
from gutbrainie.data.offsets import resolve_exclusive_end
from gutbrainie.submission.export_t611 import load_t611_json

ATLOP_QUALITIES = ("gold", "silver", "silver_2025", "bronze", "dev")
ATLOP_FILE_NAMES = {
    "gold": "train_gold.json",
    "silver": "train_silver.json",
    "silver_2025": "train_silver_2025.json",
    "bronze": "train_bronze.json",
    "dev": "dev.json",
}


def prepare_official_atlop_data(
    data_root: str | Path,
    official_repo: str | Path,
    predicted_entities: str | Path | None = None,
) -> dict[str, Any]:
    """Write ATLOP JSON files into the official baseline `Train/RE/data` folder."""
    output_dir = Path(official_repo) / "Train" / "RE" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {"output_dir": str(output_dir), "files": {}}
    for quality in ATLOP_QUALITIES:
        loaded = load_split(data_root, quality)
        examples = split_to_atlop_examples(
            loaded["articles"],
            loaded["entities"],
            loaded["full_relations"],
            include_labels=True,
        )
        output_path = output_dir / ATLOP_FILE_NAMES[quality]
        _write_json(output_path, examples)
        outputs["files"][quality] = {"path": str(output_path), "documents": len(examples)}

    if predicted_entities is not None:
        dev = load_split(data_root, "dev")
        pred_entities = load_t611_json(predicted_entities) if str(predicted_entities).endswith(".json") else pd.read_csv(predicted_entities, sep="|")
        pred_examples = split_to_atlop_examples(
            dev["articles"],
            pred_entities,
            pd.DataFrame(),
            include_labels=False,
        )
        predicted_output = output_dir / "predicted_entities_dev_atlop_format.json"
        _write_json(predicted_output, pred_examples)
        outputs["predicted_entities"] = {"path": str(predicted_output), "documents": len(pred_examples)}

    return outputs


def split_to_atlop_examples(
    articles: pd.DataFrame,
    entities: pd.DataFrame,
    full_relations: pd.DataFrame,
    include_labels: bool = True,
) -> list[dict[str, Any]]:
    """Convert one loaded split to ATLOP examples."""
    entities = _normalize_entities(entities)
    relations = _normalize_relations(full_relations)
    entities_by_pmid = {str(pmid): group for pmid, group in entities.groupby("pmid", sort=False)}
    relations_by_pmid = {str(pmid): group for pmid, group in relations.groupby("pmid", sort=False)} if not relations.empty else {}

    examples: list[dict[str, Any]] = []
    for _, article in articles.iterrows():
        pmid = str(article["pmid"])
        article_entities = entities_by_pmid.get(pmid, entities.iloc[0:0])
        article_relations = relations_by_pmid.get(pmid, relations.iloc[0:0])
        examples.append(_article_to_atlop_example(article, article_entities, article_relations, include_labels))
    return examples


def _article_to_atlop_example(
    article: pd.Series,
    entities: pd.DataFrame,
    relations: pd.DataFrame,
    include_labels: bool,
) -> dict[str, Any]:
    title = str(article.get("title", ""))
    abstract = str(article.get("abstract", ""))
    tokenized = {
        "title": _tokenize_with_offsets(title),
        "abstract": _tokenize_with_offsets(abstract),
    }
    sents = [tokenized["title"]["tokens"], tokenized["abstract"]["tokens"]]

    mention_rows = [_entity_row_to_mention(row, title, abstract, tokenized) for _, row in entities.iterrows()]
    mention_rows = [row for row in mention_rows if row is not None]

    if include_labels and not relations.empty:
        for _, relation in relations.iterrows():
            for role in ("subject", "object"):
                mention = _relation_endpoint_to_mention(relation, role, title, abstract, tokenized)
                if mention is not None:
                    mention_rows.append(mention)

    vertex_set, mention_to_index = _build_vertex_set(mention_rows)
    example: dict[str, Any] = {
        "title": str(article["pmid"]),
        "pmid": str(article["pmid"]),
        "sents": sents,
        "vertexSet": vertex_set,
    }

    if include_labels:
        labels = _build_labels(relations, mention_to_index, title, abstract, tokenized)
        example["labels"] = labels
    return example


def _normalize_entities(entities: pd.DataFrame) -> pd.DataFrame:
    if entities.empty:
        return pd.DataFrame(columns=["pmid", "start_idx", "end_idx", "location", "text_span", "label"])
    df = entities.copy()
    if "annotator" not in df.columns:
        df["annotator"] = ""
    df = deduplicate_entities(df)
    for column in ("start_idx", "end_idx"):
        df[column] = pd.to_numeric(df[column], errors="raise").astype("int64")
    df["pmid"] = df["pmid"].astype(str)
    return df


def _normalize_relations(relations: pd.DataFrame) -> pd.DataFrame:
    if relations.empty:
        return pd.DataFrame(
            columns=[
                "pmid",
                "subject_start_idx",
                "subject_end_idx",
                "subject_location",
                "subject_text_span",
                "subject_label",
                "predicate",
                "object_start_idx",
                "object_end_idx",
                "object_location",
                "object_text_span",
                "object_label",
            ]
        )
    df = relations.copy()
    for column in ("subject_start_idx", "subject_end_idx", "object_start_idx", "object_end_idx"):
        df[column] = pd.to_numeric(df[column], errors="raise").astype("int64")
    df["pmid"] = df["pmid"].astype(str)
    return df


def _entity_row_to_mention(
    row: pd.Series,
    title: str,
    abstract: str,
    tokenized: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    location = str(row["location"])
    text = title if location == "title" else abstract
    return _make_mention(
        location=location,
        text=text,
        tokenized=tokenized[location],
        start_idx=int(row["start_idx"]),
        end_idx=int(row["end_idx"]),
        text_span=str(row["text_span"]),
        label=str(row["label"]),
    )


def _relation_endpoint_to_mention(
    relation: pd.Series,
    role: str,
    title: str,
    abstract: str,
    tokenized: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    location = str(relation[f"{role}_location"])
    text = title if location == "title" else abstract
    return _make_mention(
        location=location,
        text=text,
        tokenized=tokenized[location],
        start_idx=int(relation[f"{role}_start_idx"]),
        end_idx=int(relation[f"{role}_end_idx"]),
        text_span=str(relation[f"{role}_text_span"]),
        label=str(relation[f"{role}_label"]),
    )


def _make_mention(
    location: str,
    text: str,
    tokenized: dict[str, Any],
    start_idx: int,
    end_idx: int,
    text_span: str,
    label: str,
) -> dict[str, Any] | None:
    if location not in {"title", "abstract"}:
        return None
    exclusive_end = resolve_exclusive_end(text, start_idx, end_idx, text_span)
    token_span = _char_span_to_token_span(tokenized["offsets"], start_idx, exclusive_end)
    if token_span is None:
        return None
    sent_id = 0 if location == "title" else 1
    return {
        "name": text_span,
        "sent_id": sent_id,
        "pos": [token_span[0], token_span[1]],
        "type": label,
        "location": location,
        "start_idx": start_idx,
        "end_idx": exclusive_end,
        "label": label,
    }


def _build_vertex_set(mentions: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], dict[tuple[Any, ...], int]]:
    seen: set[tuple[Any, ...]] = set()
    vertex_set: list[list[dict[str, Any]]] = []
    mention_to_index: dict[tuple[Any, ...], int] = {}
    for mention in sorted(mentions, key=_mention_sort_key):
        key = _mention_key(mention)
        if key in seen:
            continue
        seen.add(key)
        vertex_set.append(
            [
                {
                    "name": mention["name"],
                    "sent_id": mention["sent_id"],
                    "pos": mention["pos"],
                    "type": mention["type"],
                }
            ]
        )
        mention_to_index[key] = len(vertex_set) - 1
    return vertex_set, mention_to_index


def _build_labels(
    relations: pd.DataFrame,
    mention_to_index: dict[tuple[Any, ...], int],
    title: str,
    abstract: str,
    tokenized: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for _, relation in relations.iterrows():
        subject = _relation_endpoint_to_mention(relation, "subject", title, abstract, tokenized)
        obj = _relation_endpoint_to_mention(relation, "object", title, abstract, tokenized)
        if subject is None or obj is None:
            continue
        h = mention_to_index.get(_mention_key(subject))
        t = mention_to_index.get(_mention_key(obj))
        if h is None or t is None or h == t:
            continue
        predicate = str(relation["predicate"]).upper()
        label_key = (h, t, predicate)
        if label_key in seen:
            continue
        seen.add(label_key)
        evidence = sorted({subject["sent_id"], obj["sent_id"]})
        labels.append({"h": h, "t": t, "r": predicate, "evidence": evidence})
    return sorted(labels, key=lambda item: (item["h"], item["t"], item["r"]))


def _tokenize_with_offsets(text: str) -> dict[str, Any]:
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return {"tokens": tokens, "offsets": offsets}


def _char_span_to_token_span(offsets: list[tuple[int, int]], start_idx: int, end_idx: int) -> tuple[int, int] | None:
    overlapping = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_start < end_idx and token_end > start_idx
    ]
    if not overlapping:
        return None
    return overlapping[0], overlapping[-1] + 1


def _mention_key(mention: dict[str, Any]) -> tuple[Any, ...]:
    return (
        mention["sent_id"],
        mention["pos"][0],
        mention["pos"][1],
        mention["type"],
        mention["name"],
    )


def _mention_sort_key(mention: dict[str, Any]) -> tuple[Any, ...]:
    return (
        mention["sent_id"],
        mention["pos"][0],
        mention["pos"][1],
        mention["type"],
        mention["name"],
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
