"""Hybrid recommender that combines semantic and graph scores."""

from __future__ import annotations

from typing import Any


class HybridRecommender:
    def __init__(
        self,
        semantic_recommender,
        graph_recommender=None,
        semantic_weight: float = 0.6,
        graph_weight: float = 0.4,
    ):
        if semantic_recommender is None:
            raise ValueError("semantic_recommender is required.")

        self.semantic_recommender = semantic_recommender
        self.graph_recommender = graph_recommender
        self.semantic_weight = float(semantic_weight)
        self.graph_weight = float(graph_weight)

    def recommend_by_article(self, pmid: str, top_k: int = 5) -> list[dict]:
        """Merge semantic and graph article recommendations into one ranking."""
        top_k = max(0, int(top_k))
        if top_k == 0:
            return []

        semantic_results = self.semantic_recommender.recommend_by_article(pmid, top_k=top_k)
        graph_results = []
        if self.graph_recommender is not None:
            graph_results = self.graph_recommender.recommend_by_article(pmid, top_k=top_k)

        semantic_scores = _normalized_scores(semantic_results)
        graph_scores = _normalized_scores(graph_results)
        merged: dict[str, dict] = {}

        for result in semantic_results:
            pmid_key = _pmid(result)
            if not pmid_key:
                continue
            merged[pmid_key] = _base_result(result)

        for result in graph_results:
            pmid_key = _pmid(result)
            if not pmid_key:
                continue
            if pmid_key not in merged:
                merged[pmid_key] = _base_result(result)
            elif result.get("shared_mesh_terms"):
                merged[pmid_key]["shared_mesh_terms"] = result.get("shared_mesh_terms", [])

        ranked: list[dict] = []
        for pmid_key, result in merged.items():
            semantic_score = semantic_scores.get(pmid_key, 0.0)
            graph_score = graph_scores.get(pmid_key, 0.0)
            final_score = (
                self.semantic_weight * semantic_score
                + self.graph_weight * graph_score
            )
            result.update(
                {
                    "semantic_score": semantic_score,
                    "graph_score": graph_score,
                    "final_score": final_score,
                    "score": final_score,
                    "method": "hybrid",
                }
            )
            ranked.append(result)

        ranked.sort(key=lambda item: item["final_score"], reverse=True)
        return ranked[:top_k]


def _normalized_scores(results: list[dict]) -> dict[str, float]:
    scores = {_pmid(result): float(result.get("score", 0.0)) for result in results if _pmid(result)}
    if not scores:
        return {}

    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)

    if max_score == min_score:
        return {pmid: 1.0 if score > 0 else 0.0 for pmid, score in scores.items()}

    return {
        pmid: (score - min_score) / (max_score - min_score)
        for pmid, score in scores.items()
    }


def _base_result(result: dict[str, Any]) -> dict:
    return {
        "pmid": result.get("pmid", ""),
        "title": result.get("title", ""),
        "abstract": result.get("abstract", ""),
        "year": result.get("year"),
        "journal": result.get("journal", ""),
        "publication_types": result.get("publication_types", []),
        "mesh_terms": result.get("mesh_terms", []),
        "shared_mesh_terms": result.get("shared_mesh_terms", []),
    }


def _pmid(result: dict[str, Any]) -> str:
    return str(result.get("pmid", "")).strip()
