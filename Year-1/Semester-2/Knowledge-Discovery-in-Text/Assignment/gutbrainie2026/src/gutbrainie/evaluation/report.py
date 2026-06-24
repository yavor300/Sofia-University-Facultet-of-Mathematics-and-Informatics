"""Dataset statistics and report helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

Path("outputs/.matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "outputs/.matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from gutbrainie.data.annotations import deduplicate_entities
from gutbrainie.data.dataset import load_split

DEFAULT_EDA_QUALITIES = ("gold", "dev")


def generate_data_statistics(
    data_root: str | Path,
    output_dir: str | Path = "outputs/reports",
    qualities: tuple[str, ...] | list[str] = DEFAULT_EDA_QUALITIES,
) -> dict[str, Any]:
    """Generate split statistics, distribution CSVs, plots, and an imbalance note."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_stats: list[dict[str, Any]] = []
    entity_distributions: list[pd.DataFrame] = []
    relation_distributions: list[pd.DataFrame] = []
    triple_distributions: list[pd.DataFrame] = []

    for quality in qualities:
        loaded = load_split(data_root, quality)
        articles = loaded["articles"]
        entities = deduplicate_entities(loaded["entities"])
        mention_relations = loaded["mention_relations"]
        full_relations = loaded["full_relations"]

        stats = _split_stats(quality, articles, entities, mention_relations, full_relations)
        split_stats.append(stats)
        pd.DataFrame([stats]).to_csv(output_path / f"data_stats_{quality}.csv", index=False)

        entity_distributions.append(_entity_label_distribution(quality, entities))
        relation_distributions.append(_relation_predicate_distribution(quality, mention_relations))
        triple_distributions.append(_relation_triple_distribution(quality, mention_relations))

    entity_distribution = _concat_or_empty(entity_distributions)
    relation_distribution = _concat_or_empty(relation_distributions)
    triple_distribution = _concat_or_empty(triple_distributions)

    entity_distribution.to_csv(output_path / "entity_label_distribution.csv", index=False)
    relation_distribution.to_csv(output_path / "relation_label_distribution.csv", index=False)
    triple_distribution.to_csv(output_path / "relation_triple_distribution.csv", index=False)

    _plot_distribution(
        entity_distribution,
        category_column="label",
        output_path=output_path / "entity_distribution.png",
        title="Entity label distribution",
        ylabel="Entities",
    )
    _plot_distribution(
        relation_distribution,
        category_column="predicate",
        output_path=output_path / "relation_distribution.png",
        title="Relation predicate distribution",
        ylabel="Relations",
    )
    _write_imbalance_summary(split_stats, output_path / "imbalance_summary.md")

    return {
        "qualities": list(qualities),
        "output_dir": str(output_path),
        "split_stats": split_stats,
        "files": [
            *(str(output_path / f"data_stats_{quality}.csv") for quality in qualities),
            str(output_path / "entity_label_distribution.csv"),
            str(output_path / "relation_label_distribution.csv"),
            str(output_path / "relation_triple_distribution.csv"),
            str(output_path / "entity_distribution.png"),
            str(output_path / "relation_distribution.png"),
            str(output_path / "imbalance_summary.md"),
        ],
    }


def _split_stats(
    quality: str,
    articles: pd.DataFrame,
    entities: pd.DataFrame,
    mention_relations: pd.DataFrame,
    full_relations: pd.DataFrame,
) -> dict[str, Any]:
    article_count = len(articles)
    entity_counts = entities["label"].value_counts()
    relation_counts = mention_relations["predicate"].value_counts()
    entity_imbalance = _imbalance_stats(entity_counts, "entity")
    relation_imbalance = _imbalance_stats(relation_counts, "relation")

    return {
        "split": quality,
        "documents": int(article_count),
        "entities": int(len(entities)),
        "mention_level_relations": int(len(mention_relations)),
        "full_relations": int(len(full_relations)),
        "entity_labels": int(entities["label"].nunique()),
        "relation_predicates": int(mention_relations["predicate"].nunique()),
        "avg_title_length_chars": _mean_string_length(articles["title"]),
        "avg_abstract_length_chars": _mean_string_length(articles["abstract"]),
        "avg_title_length_tokens": _mean_token_length(articles["title"]),
        "avg_abstract_length_tokens": _mean_token_length(articles["abstract"]),
        "avg_entities_per_article": _safe_ratio(len(entities), article_count),
        "avg_relations_per_article": _safe_ratio(len(mention_relations), article_count),
        **entity_imbalance,
        **relation_imbalance,
    }


def _entity_label_distribution(quality: str, entities: pd.DataFrame) -> pd.DataFrame:
    counts = entities["label"].value_counts().rename_axis("label").reset_index(name="count")
    counts.insert(0, "split", quality)
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts


def _relation_predicate_distribution(quality: str, relations: pd.DataFrame) -> pd.DataFrame:
    counts = relations["predicate"].value_counts().rename_axis("predicate").reset_index(name="count")
    counts.insert(0, "split", quality)
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts


def _relation_triple_distribution(quality: str, relations: pd.DataFrame) -> pd.DataFrame:
    counts = (
        relations.groupby(["subject_label", "predicate", "object_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    counts.insert(0, "split", quality)
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts


def _plot_distribution(
    distribution: pd.DataFrame,
    category_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    top_n: int = 30,
) -> None:
    plt.figure(figsize=(12, 7))
    if distribution.empty:
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks([])
    else:
        plot_df = (
            distribution.groupby(category_column, as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
            .head(top_n)
        )
        plt.bar(plot_df[category_column], plot_df["count"], color="#2f6f73")
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _write_imbalance_summary(split_stats: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Class Imbalance Summary",
        "",
        "Micro-F1 is important for this project because entity labels and relation predicates are imbalanced. "
        "The majority classes account for a large share of annotations, so macro-F1 should be read alongside "
        "micro-F1 to avoid hiding minority-class failures.",
        "",
    ]
    for stats in split_stats:
        lines.extend(
            [
                f"## {stats['split']}",
                "",
                (
                    f"- Entity majority label: `{stats['entity_majority_label']}` "
                    f"({stats['entity_majority_share']:.3f} share, "
                    f"imbalance ratio {stats['entity_imbalance_ratio']:.3f})."
                ),
                (
                    f"- Relation majority predicate: `{stats['relation_majority_label']}` "
                    f"({stats['relation_majority_share']:.3f} share, "
                    f"imbalance ratio {stats['relation_imbalance_ratio']:.3f})."
                ),
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _imbalance_stats(counts: pd.Series, prefix: str) -> dict[str, Any]:
    if counts.empty:
        return {
            f"{prefix}_majority_label": "",
            f"{prefix}_majority_count": 0,
            f"{prefix}_majority_share": 0.0,
            f"{prefix}_minority_label": "",
            f"{prefix}_minority_count": 0,
            f"{prefix}_imbalance_ratio": 0.0,
        }

    majority_label = str(counts.idxmax())
    minority_label = str(counts.idxmin())
    majority_count = int(counts.max())
    minority_count = int(counts.min())
    total = int(counts.sum())
    return {
        f"{prefix}_majority_label": majority_label,
        f"{prefix}_majority_count": majority_count,
        f"{prefix}_majority_share": _safe_ratio(majority_count, total),
        f"{prefix}_minority_label": minority_label,
        f"{prefix}_minority_count": minority_count,
        f"{prefix}_imbalance_ratio": _safe_ratio(majority_count, minority_count),
    }


def _mean_string_length(values: pd.Series) -> float:
    return float(values.astype(str).str.len().mean())


def _mean_token_length(values: pd.Series) -> float:
    return float(values.astype(str).str.split().str.len().mean())


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
