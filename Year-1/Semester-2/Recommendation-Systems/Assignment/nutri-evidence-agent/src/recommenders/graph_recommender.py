"""Graph embedding recommender backed by Article-MeSH node2vec vectors."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.graph.graph_builder import article_node_id
from src.preprocessing.article_preprocessor import filter_valid_articles


class GraphRecommender:
    def __init__(self):
        self.articles: list[dict] = []
        self.graph = None
        self.node_vectors = None
        self.pmid_to_article: dict[str, dict] = {}
        self.article_node_to_pmid: dict[str, str] = {}
        self.article_vectors: dict[str, np.ndarray] = {}
        self.article_mesh_terms: dict[str, set[str]] = {}
        self.article_mesh_display_maps: dict[str, dict[str, str]] = {}

    def fit(self, articles: list[dict], graph, node2vec_model) -> None:
        """Index article metadata and available article-node embeddings."""
        self.articles = filter_valid_articles(articles)
        self.graph = graph
        self.node_vectors = _keyed_vectors(node2vec_model)
        self.pmid_to_article = {
            str(article.get("pmid", "")).strip(): article
            for article in self.articles
            if str(article.get("pmid", "")).strip()
        }
        self.article_node_to_pmid = {
            article_node_id(pmid): pmid
            for pmid in self.pmid_to_article
        }
        self.article_mesh_display_maps = {
            pmid: _mesh_term_map(article.get("mesh_terms", []))
            for pmid, article in self.pmid_to_article.items()
        }
        self.article_mesh_terms = {
            pmid: set(term_map)
            for pmid, term_map in self.article_mesh_display_maps.items()
        }
        self.article_vectors = {}

        for node_id, pmid in self.article_node_to_pmid.items():
            if node_id in self.node_vectors:
                self.article_vectors[node_id] = _normalize_vector(self.node_vectors[node_id])

    def recommend_by_article(self, pmid: str, top_k: int = 5) -> list[dict]:
        """Recommend articles near the seed article in graph embedding space."""
        pmid = str(pmid).strip()
        seed_node = article_node_id(pmid)
        seed_vector = self.article_vectors.get(seed_node)
        if seed_vector is None:
            return []

        return self._recommend_from_vector(
            query_vector=seed_vector,
            top_k=top_k,
            exclude_pmids={pmid},
            reference_pmids=[pmid],
        )

    def recommend_from_liked_articles(self, pmids: list[str], top_k: int = 5) -> list[dict]:
        """Recommend from the average vector of multiple liked articles."""
        clean_pmids = [str(pmid).strip() for pmid in pmids if str(pmid).strip()]
        vectors = [
            self.article_vectors[article_node_id(pmid)]
            for pmid in clean_pmids
            if article_node_id(pmid) in self.article_vectors
        ]
        if not vectors:
            return []

        query_vector = _normalize_vector(np.mean(vectors, axis=0))
        return self._recommend_from_vector(
            query_vector=query_vector,
            top_k=top_k,
            exclude_pmids=set(clean_pmids),
            reference_pmids=clean_pmids,
        )

    def _recommend_from_vector(
        self,
        query_vector: np.ndarray,
        top_k: int,
        exclude_pmids: set[str],
        reference_pmids: list[str],
    ) -> list[dict]:
        top_k = max(0, int(top_k))
        if top_k == 0:
            return []

        candidates: list[tuple[str, float, list[str]]] = []
        reference_terms = set().union(
            *(self.article_mesh_terms.get(pmid, set()) for pmid in reference_pmids)
        )
        reference_display = {
            key: value
            for pmid in reference_pmids
            for key, value in self.article_mesh_display_maps.get(pmid, {}).items()
        }

        for node_id, candidate_vector in self.article_vectors.items():
            pmid = self.article_node_to_pmid.get(node_id)
            if not pmid or pmid in exclude_pmids:
                continue

            score = float(np.dot(query_vector, candidate_vector))
            shared_keys = reference_terms & self.article_mesh_terms.get(pmid, set())
            shared_terms = sorted(reference_display.get(key, key) for key in shared_keys)
            candidates.append((pmid, score, shared_terms))

        candidates.sort(key=lambda item: (item[1], len(item[2])), reverse=True)
        return [
            self._format_result(self.pmid_to_article[pmid], score, shared_terms)
            for pmid, score, shared_terms in candidates[:top_k]
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
            "method": "graph",
        }


def _keyed_vectors(model):
    return model.wv if hasattr(model, "wv") else model


def _normalize_vector(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0:
        return array
    return array / norm


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
