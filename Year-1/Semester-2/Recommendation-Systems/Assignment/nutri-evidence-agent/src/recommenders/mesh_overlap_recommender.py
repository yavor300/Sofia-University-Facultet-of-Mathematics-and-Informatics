"""Baseline recommender using Jaccard overlap of MeSH terms."""

from __future__ import annotations

from typing import Any

from src.preprocessing.article_preprocessor import filter_valid_articles


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Return Jaccard similarity for two sets."""
    if not a and not b:
        return 0.0

    union = a | b
    if not union:
        return 0.0

    return len(a & b) / len(union)


class MeshOverlapRecommender:
    """Recommend similar articles by overlap in normalized MeSH terms."""

    def __init__(self):
        self.articles: list[dict] = []
        self.pmid_to_index: dict[str, int] = {}
        self.mesh_sets: list[set[str]] = []
        self.mesh_display_maps: list[dict[str, str]] = []

    def fit(self, articles: list[dict]) -> None:
        """Preprocess and index article MeSH terms."""
        self.articles = filter_valid_articles(articles)
        self.pmid_to_index = {
            str(article.get("pmid", "")).strip(): index
            for index, article in enumerate(self.articles)
            if str(article.get("pmid", "")).strip()
        }
        self.mesh_display_maps = [
            _mesh_term_map(article.get("mesh_terms", []))
            for article in self.articles
        ]
        self.mesh_sets = [set(term_map) for term_map in self.mesh_display_maps]

    def recommend_by_article(self, pmid: str, top_k: int = 5) -> list[dict]:
        """Return top articles ranked by MeSH-term Jaccard similarity."""
        if not self.articles:
            raise RuntimeError("MeshOverlapRecommender is not fitted. Call fit() first.")

        seed_index = self.pmid_to_index.get(str(pmid).strip())
        if seed_index is None:
            raise ValueError(f"PMID not found in MeSH overlap index: {pmid}")

        seed_terms = self.mesh_sets[seed_index]
        if not seed_terms:
            return []

        top_k = max(0, int(top_k))
        if top_k == 0:
            return []

        candidates: list[tuple[int, float, list[str]]] = []
        for index, terms in enumerate(self.mesh_sets):
            if index == seed_index:
                continue

            shared_terms = sorted(
                self.mesh_display_maps[seed_index][term]
                for term in seed_terms & terms
            )
            score = jaccard_similarity(seed_terms, terms)
            candidates.append((index, score, shared_terms))

        candidates.sort(
            key=lambda item: (
                item[1],
                len(item[2]),
                str(self.articles[item[0]].get("year") or ""),
            ),
            reverse=True,
        )

        return [
            self._format_result(self.articles[index], score, shared_terms)
            for index, score, shared_terms in candidates[:top_k]
        ]

    def _format_result(self, article: dict[str, Any], score: float, shared_terms: list[str]) -> dict:
        return {
            "pmid": article.get("pmid", ""),
            "title": article.get("title", ""),
            "abstract": article.get("abstract", ""),
            "year": article.get("year"),
            "journal": article.get("journal", ""),
            "publication_types": article.get("publication_types", []),
            "mesh_terms": article.get("mesh_terms", []),
            "shared_mesh_terms": shared_terms,
            "score": score,
            "method": "mesh_overlap",
        }


def _mesh_term_map(mesh_terms: Any) -> dict[str, str]:
    if mesh_terms is None:
        return {}

    if isinstance(mesh_terms, str):
        values = [mesh_terms]
    else:
        try:
            values = list(mesh_terms)
        except TypeError:
            values = [mesh_terms]

    term_map: dict[str, str] = {}
    for term in values:
        display = str(term).strip()
        if not display:
            continue
        term_map.setdefault(display.lower(), display)

    return term_map
