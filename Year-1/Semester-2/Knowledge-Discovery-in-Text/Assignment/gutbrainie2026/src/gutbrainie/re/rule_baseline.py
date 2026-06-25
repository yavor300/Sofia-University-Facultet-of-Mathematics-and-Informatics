"""Rule and prior-based relation extraction baseline."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from gutbrainie.data.annotations import load_entities_csv, load_mention_relations_csv
from gutbrainie.data.articles import load_articles_csv
from gutbrainie.re.candidates import generate_relation_candidates
from gutbrainie.re.relation_schema import VALID_RELATIONS, valid_predicates
from gutbrainie.submission.export_t611 import load_t611_json
from gutbrainie.submission.export_t621 import mention_relations_to_t621_json

MENTION_RELATION_COLUMNS = [
    "pmid",
    "subject_text_span",
    "subject_label",
    "predicate",
    "object_text_span",
    "object_label",
]


class RelationPriorBaseline:
    """Predict the most frequent predicate for each subject/object label pair."""

    def __init__(
        self,
        threshold: float = 0.5,
        valid_schema: dict[tuple[str, str], list[str]] | None = None,
    ) -> None:
        self.threshold = threshold
        self.valid_schema = valid_schema or VALID_RELATIONS
        self.predicate_counts_: dict[tuple[str, str], Counter[str]] = {}
        self.predicate_priors_: dict[tuple[str, str], dict[str, float]] = {}

    def fit(self, train_relations: pd.DataFrame) -> "RelationPriorBaseline":
        _require_columns(train_relations, MENTION_RELATION_COLUMNS, "train_relations")
        counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for _, relation in train_relations.iterrows():
            pair = (str(relation["subject_label"]), str(relation["object_label"]))
            predicate = str(relation["predicate"])
            if predicate in valid_predicates(pair[0], pair[1], self.valid_schema):
                counts[pair][predicate] += 1

        self.predicate_counts_ = dict(counts)
        self.predicate_priors_ = {
            pair: {predicate: count / sum(counter.values()) for predicate, count in counter.items()}
            for pair, counter in counts.items()
        }
        return self

    def predict(self, candidates: pd.DataFrame) -> pd.DataFrame:
        _require_columns(
            candidates,
            ["pmid", "subject_text_span", "subject_label", "object_text_span", "object_label"],
            "candidates",
        )
        rows: list[dict[str, Any]] = []
        for _, candidate in candidates.iterrows():
            pair = (str(candidate["subject_label"]), str(candidate["object_label"]))
            predicate, prior = self._best_predicate(pair)
            if predicate is None or prior < self.threshold:
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
        return deduplicate_mention_relations(pd.DataFrame(rows, columns=MENTION_RELATION_COLUMNS))

    def _best_predicate(self, pair: tuple[str, str]) -> tuple[str | None, float]:
        counter = self.predicate_counts_.get(pair)
        if not counter:
            return None, 0.0
        predicate, count = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0]
        return predicate, count / sum(counter.values())


def predict_re_rule_to_json(
    articles_path: str | Path,
    entities_path: str | Path,
    train_relations_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.5,
    max_distance: int | None = None,
) -> pd.DataFrame:
    """Generate RE rule-baseline predictions and write T621 JSON."""
    articles = load_articles_csv(articles_path)
    entities = load_entities(entities_path)
    train_relations = load_mention_relations_csv(train_relations_path)
    candidates = generate_relation_candidates(
        articles=articles,
        entities=entities,
        valid_schema=VALID_RELATIONS,
        max_distance=max_distance,
    )
    predictions = RelationPriorBaseline(threshold=threshold).fit(train_relations).predict(candidates)

    payload = mention_relations_to_t621_json(predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return predictions


def load_entities(path: str | Path) -> pd.DataFrame:
    """Load entity mentions from either CSV annotations or T611 JSON predictions."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_t611_json(path)
    return load_entities_csv(path)


def deduplicate_mention_relations(relations: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by the official mention-level relation fields."""
    if relations.empty:
        return pd.DataFrame(columns=MENTION_RELATION_COLUMNS)
    return relations.drop_duplicates(subset=MENTION_RELATION_COLUMNS).reset_index(drop=True)


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
