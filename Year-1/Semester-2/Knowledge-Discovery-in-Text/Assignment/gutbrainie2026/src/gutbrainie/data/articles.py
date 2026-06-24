"""Article loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ARTICLE_COLUMNS = ["pmid", "title", "authors", "journal", "year", "abstract"]


def load_articles_csv(path: str | Path) -> pd.DataFrame:
    """Load a GutBrainIE pipe-separated article CSV file."""
    df = pd.read_csv(Path(path), sep="|", dtype=str, keep_default_na=False)
    missing = [column for column in ARTICLE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing article columns in {path}: {missing}")

    df = df[ARTICLE_COLUMNS].copy()
    df["pmid"] = df["pmid"].astype(str)
    return df
