"""Dataset split resolution utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_QUALITIES = ("gold", "silver", "silver_2025", "bronze", "dev")


@dataclass(frozen=True)
class SplitPaths:
    articles: Path
    entities: Path
    mention_relations: Path
    full_relations: Path


def resolve_split_paths(data_root: str | Path, quality: str) -> SplitPaths:
    """Resolve article and annotation CSV paths for a train quality or dev split."""
    if quality not in SUPPORTED_QUALITIES:
        raise ValueError(f"Unsupported quality '{quality}'. Expected one of {SUPPORTED_QUALITIES}.")

    root = Path(data_root)
    if quality == "dev":
        return SplitPaths(
            articles=root / "Articles" / "csv_format" / "articles_dev.csv",
            entities=root / "Annotations" / "Dev" / "csv_format" / "dev_entities.csv",
            mention_relations=root / "Annotations" / "Dev" / "csv_format" / "dev_mention_level_relations.csv",
            full_relations=root / "Annotations" / "Dev" / "csv_format" / "dev_relations.csv",
        )

    quality_dir = "silver_quality" if quality == "silver_2025" else f"{quality}_quality"
    prefix = f"train_{quality}"
    annotation_root = root / "Annotations" / "Train" / quality_dir / "csv_format"

    return SplitPaths(
        articles=root / "Articles" / "csv_format" / f"articles_train_{quality}.csv",
        entities=annotation_root / f"{prefix}_entities.csv",
        mention_relations=annotation_root / f"{prefix}_mention_level_relations.csv",
        full_relations=annotation_root / f"{prefix}_relations.csv",
    )
