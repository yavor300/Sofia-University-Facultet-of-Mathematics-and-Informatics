"""Dataset assembly and validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gutbrainie.data.annotations import (
    deduplicate_entities,
    load_entities_csv,
    load_full_relations_csv,
    load_mention_relations_csv,
)
from gutbrainie.data.articles import load_articles_csv
from gutbrainie.data.offsets import validate_entity_offsets
from gutbrainie.data.splits import SplitPaths, resolve_split_paths


def load_split(data_root: str | Path, quality: str) -> dict[str, Any]:
    """Load articles, entities, mention-level relations, and full relations."""
    paths = resolve_split_paths(data_root, quality)
    return {
        "paths": paths,
        "articles": load_articles_csv(paths.articles),
        "entities": load_entities_csv(paths.entities),
        "mention_relations": load_mention_relations_csv(paths.mention_relations),
        "full_relations": load_full_relations_csv(paths.full_relations),
    }


def build_validation_report(data_root: str | Path, quality: str) -> dict[str, Any]:
    """Load a split and summarize entity offset validity."""
    loaded = load_split(data_root, quality)
    paths: SplitPaths = loaded["paths"]
    articles = loaded["articles"]
    raw_entities = loaded["entities"]
    entities = deduplicate_entities(raw_entities)
    mention_relations = loaded["mention_relations"]
    full_relations = loaded["full_relations"]

    article_by_pmid = articles.set_index("pmid", drop=False)
    offset_checks_passed = 0
    offset_checks_failed = 0

    for _, entity in entities.iterrows():
        pmid = entity["pmid"]
        if pmid not in article_by_pmid.index:
            offset_checks_failed += 1
            continue
        if validate_entity_offsets(article_by_pmid.loc[pmid], entity):
            offset_checks_passed += 1
        else:
            offset_checks_failed += 1

    annotated_pmids = set(entities["pmid"]) | set(mention_relations["pmid"]) | set(full_relations["pmid"])
    missing_article_pmids = sorted(annotated_pmids - set(articles["pmid"]))

    return {
        "quality": quality,
        "paths": {
            "articles": str(paths.articles),
            "entities": str(paths.entities),
            "mention_relations": str(paths.mention_relations),
            "full_relations": str(paths.full_relations),
        },
        "articles": int(len(articles)),
        "entities": int(len(entities)),
        "raw_entities": int(len(raw_entities)),
        "duplicate_entities_removed": int(len(raw_entities) - len(entities)),
        "relations": int(len(mention_relations)),
        "full_relations": int(len(full_relations)),
        "offset_checks_passed": int(offset_checks_passed),
        "offset_checks_failed": int(offset_checks_failed),
        "missing_articles": int(len(missing_article_pmids)),
        "missing_article_pmids": missing_article_pmids,
    }


def write_validation_report(data_root: str | Path, quality: str, output: str | Path) -> dict[str, Any]:
    """Build and write a validation report as formatted JSON."""
    report = build_validation_report(data_root, quality)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report
