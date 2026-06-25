"""Mention-pair candidate generation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from gutbrainie.data.annotations import deduplicate_entities
from gutbrainie.data.offsets import resolve_exclusive_end
from gutbrainie.re.relation_schema import VALID_RELATIONS, valid_predicates

CANDIDATE_COLUMNS = [
    "pmid",
    "subject_text_span",
    "subject_label",
    "object_text_span",
    "object_label",
    "subject_location",
    "object_location",
    "subject_start_idx",
    "subject_end_idx",
    "object_start_idx",
    "object_end_idx",
    "text_between",
    "sentence_distance",
    "candidate_key",
]


def generate_relation_candidates(
    articles: pd.DataFrame,
    entities: pd.DataFrame,
    valid_schema: Mapping[tuple[str, str], list[str]] | None = None,
    max_distance: int | None = None,
    gold_relations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate ordered subject-object mention-pair candidates.

    Candidate offsets use the same inclusive end-index convention as the local
    GutBrainIE CSV files. If ``gold_relations`` is passed, ``predicate`` is set
    to the gold predicate for matching mention-level pairs; otherwise it is set
    to ``no_relation``.
    """
    _require_columns(articles, ["pmid", "title", "abstract"], "articles")
    _require_columns(entities, ["pmid", "start_idx", "end_idx", "location", "text_span", "label"], "entities")
    schema = valid_schema or VALID_RELATIONS
    entities = _normalize_entities(articles, deduplicate_entities(entities))
    gold_lookup = _build_gold_relation_lookup(gold_relations) if gold_relations is not None else {}

    rows: list[dict[str, Any]] = []
    for pmid, group in entities.groupby("pmid", sort=False):
        mentions = group.sort_values(["location", "start_idx", "end_idx", "label", "text_span"]).to_dict("records")
        for subject_index, subject in enumerate(mentions):
            for object_index, obj in enumerate(mentions):
                if subject_index == object_index:
                    continue
                predicates = valid_predicates(subject["label"], obj["label"], schema)
                if not predicates:
                    continue
                sentence_distance = _sentence_distance(subject, obj)
                if max_distance is not None and sentence_distance > max_distance:
                    continue
                key = _candidate_key_from_mentions(subject, obj)
                row = {
                    "pmid": str(pmid),
                    "subject_text_span": str(subject["text_span"]),
                    "subject_label": str(subject["label"]),
                    "object_text_span": str(obj["text_span"]),
                    "object_label": str(obj["label"]),
                    "subject_location": str(subject["location"]),
                    "object_location": str(obj["location"]),
                    "subject_start_idx": int(subject["start_idx"]),
                    "subject_end_idx": int(subject["end_idx"]),
                    "object_start_idx": int(obj["start_idx"]),
                    "object_end_idx": int(obj["end_idx"]),
                    "text_between": _text_between(subject, obj),
                    "sentence_distance": sentence_distance,
                    "candidate_key": "|".join(key),
                }
                row["predicate"] = gold_lookup.get((str(pmid), *key), "no_relation")
                rows.append(row)

    columns = [*CANDIDATE_COLUMNS, "predicate"]
    return pd.DataFrame(rows, columns=columns)


def _normalize_entities(articles: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    article_texts = {
        (str(row["pmid"]), location): str(row[location])
        for _, row in articles.iterrows()
        for location in ("title", "abstract")
    }
    rows: list[dict[str, Any]] = []
    for _, entity in entities.iterrows():
        row = entity.to_dict()
        pmid = str(row["pmid"])
        location = str(row["location"])
        text = article_texts.get((pmid, location), "")
        start_idx = int(row["start_idx"])
        text_span = str(row.get("text_span", ""))
        exclusive_end = resolve_exclusive_end(text, start_idx, int(row["end_idx"]), text_span)
        row["pmid"] = pmid
        row["location"] = location
        row["start_idx"] = start_idx
        row["end_idx"] = exclusive_end - 1
        row["_exclusive_end"] = exclusive_end
        row["_text"] = text
        rows.append(row)
    return pd.DataFrame(rows)


def _build_gold_relation_lookup(gold_relations: pd.DataFrame) -> dict[tuple[str, str, str, str, str], str]:
    if gold_relations.empty:
        return {}
    _require_columns(
        gold_relations,
        ["pmid", "subject_text_span", "subject_label", "predicate", "object_text_span", "object_label"],
        "gold_relations",
    )
    lookup: dict[tuple[str, str, str, str, str], str] = {}
    for _, relation in gold_relations.iterrows():
        key = (
            str(relation["pmid"]),
            str(relation["subject_text_span"]),
            str(relation["subject_label"]),
            str(relation["object_text_span"]),
            str(relation["object_label"]),
        )
        lookup.setdefault(key, str(relation["predicate"]))
    return lookup


def _candidate_key_from_mentions(subject: dict[str, Any], obj: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(subject["text_span"]),
        str(subject["label"]),
        str(obj["text_span"]),
        str(obj["label"]),
    )


def _text_between(subject: dict[str, Any], obj: dict[str, Any]) -> str:
    if subject["location"] != obj["location"]:
        return ""
    if int(subject["start_idx"]) <= int(obj["start_idx"]):
        start = int(subject["_exclusive_end"])
        end = int(obj["start_idx"])
    else:
        start = int(obj["_exclusive_end"])
        end = int(subject["start_idx"])
    if end <= start:
        return ""
    return str(subject["_text"])[start:end].strip()


def _sentence_distance(subject: dict[str, Any], obj: dict[str, Any]) -> int:
    if subject["location"] != obj["location"]:
        return 1
    between = _text_between(subject, obj)
    if not between:
        return 0
    return len(re.findall(r"[.!?]+", between))


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
