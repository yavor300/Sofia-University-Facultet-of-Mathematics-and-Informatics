"""Article filtering and summary helpers for the Streamlit demo."""

from __future__ import annotations

import pandas as pd


def filter_articles(articles: pd.DataFrame, query: str = "") -> pd.DataFrame:
    if not query:
        return articles
    q = query.strip().lower()
    if not q:
        return articles
    mask = (
        articles["pmid"].astype(str).str.lower().str.contains(q, regex=False)
        | articles["title"].astype(str).str.lower().str.contains(q, regex=False)
        | articles["abstract"].astype(str).str.lower().str.contains(q, regex=False)
    )
    return articles.loc[mask].reset_index(drop=True)


def article_summary_table(
    articles: pd.DataFrame,
    entities: pd.DataFrame | None = None,
    relations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    table = articles[["pmid", "title", "year"]].copy()
    table["pmid"] = table["pmid"].astype(str)
    if entities is not None and not entities.empty:
        counts = entities.groupby("pmid").size().rename("entities")
        table = table.join(counts, on="pmid")
    else:
        table["entities"] = 0
    if relations is not None and not relations.empty:
        counts = relations.groupby("pmid").size().rename("relations")
        table = table.join(counts, on="pmid")
    else:
        table["relations"] = 0
    table[["entities", "relations"]] = table[["entities", "relations"]].fillna(0).astype(int)
    return table
