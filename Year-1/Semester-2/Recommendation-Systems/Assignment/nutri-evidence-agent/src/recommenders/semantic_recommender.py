"""Semantic article recommender using sentence-transformer embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing.article_preprocessor import filter_valid_articles


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticRecommender:
    """Recommend articles by cosine similarity over normalized text embeddings."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.articles: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.pmid_to_index: dict[str, int] = {}

    def fit(self, articles: list[dict]) -> None:
        """Preprocess articles and compute normalized document embeddings."""
        self.articles = filter_valid_articles(articles)
        self.pmid_to_index = self._build_pmid_index(self.articles)

        documents = [article["document_text"] for article in self.articles]
        if not documents:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            return

        model = self._get_model()
        self.embeddings = self._normalize_embeddings(
            model.encode(
                documents,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )

    def recommend_by_query(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top semantic matches for a free-text query."""
        self._ensure_ready()
        query = str(query or "").strip()
        if not query:
            return []

        model = self._get_model()
        query_embedding = self._normalize_embeddings(
            model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )[0]
        scores = self.embeddings @ query_embedding
        return self._rank(scores=scores, top_k=top_k)

    def recommend_by_article(self, pmid: str, top_k: int = 5) -> list[dict]:
        """Return top semantic matches for a cached article, excluding itself."""
        self._ensure_ready()
        seed_index = self.pmid_to_index.get(str(pmid).strip())
        if seed_index is None:
            raise ValueError(f"PMID not found in semantic index: {pmid}")

        scores = self.embeddings @ self.embeddings[seed_index]
        return self._rank(scores=scores, top_k=top_k, exclude_indices={seed_index})

    def save_artifacts(self, embeddings_path: str, index_path: str) -> None:
        """Save embeddings as .npy and article index metadata as JSON."""
        self._ensure_ready()

        embeddings_file = Path(embeddings_path)
        index_file = Path(index_path)
        embeddings_file.parent.mkdir(parents=True, exist_ok=True)
        index_file.parent.mkdir(parents=True, exist_ok=True)

        np.save(embeddings_file, self.embeddings)
        with index_file.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "model_name": self.model_name,
                    "pmids": [article.get("pmid", "") for article in self.articles],
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

    def load_artifacts(self, articles: list[dict], embeddings_path: str, index_path: str) -> None:
        """Load precomputed embeddings and align the supplied articles by PMID."""
        embeddings = np.load(embeddings_path)
        with Path(index_path).open("r", encoding="utf-8") as file:
            index_data = json.load(file)

        pmids = index_data.get("pmids")
        if not isinstance(pmids, list):
            raise ValueError(f"Article index JSON is missing a pmids list: {index_path}")

        normalized_articles = filter_valid_articles(articles)
        article_by_pmid = {
            str(article.get("pmid", "")).strip(): article
            for article in normalized_articles
            if str(article.get("pmid", "")).strip()
        }
        ordered_articles: list[dict] = []

        for pmid in pmids:
            pmid = str(pmid).strip()
            article = article_by_pmid.get(pmid)
            if article is None:
                raise ValueError(f"Artifact index PMID is missing from supplied articles: {pmid}")
            ordered_articles.append(article)

        if embeddings.shape[0] != len(ordered_articles):
            raise ValueError(
                "Embedding row count does not match article index length: "
                f"{embeddings.shape[0]} != {len(ordered_articles)}"
            )

        self.articles = ordered_articles
        self.embeddings = self._normalize_embeddings(embeddings)
        self.pmid_to_index = self._build_pmid_index(self.articles)

    def _rank(
        self,
        scores: np.ndarray,
        top_k: int,
        exclude_indices: set[int] | None = None,
    ) -> list[dict]:
        exclude_indices = exclude_indices or set()
        top_k = max(0, int(top_k))
        if top_k == 0:
            return []

        candidates = [
            (index, float(score))
            for index, score in enumerate(scores)
            if index not in exclude_indices
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        return [
            self._format_result(self.articles[index], score)
            for index, score in candidates[:top_k]
        ]

    def _format_result(self, article: dict[str, Any], score: float) -> dict:
        return {
            "pmid": article.get("pmid", ""),
            "title": article.get("title", ""),
            "abstract": article.get("abstract", ""),
            "year": article.get("year"),
            "journal": article.get("journal", ""),
            "publication_types": article.get("publication_types", []),
            "mesh_terms": article.get("mesh_terms", []),
            "score": score,
            "method": "semantic",
        }

    def _ensure_ready(self) -> None:
        if self.embeddings is None:
            raise RuntimeError("SemanticRecommender is not fitted. Call fit() or load_artifacts() first.")
        if len(self.articles) != self.embeddings.shape[0]:
            raise RuntimeError("SemanticRecommender article index and embeddings are out of sync.")

    def _get_model(self):
        if self.model is None:
            self.model = self._load_model(self.model_name)
        return self.model

    def _load_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SemanticRecommender. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from exc

        return SentenceTransformer(model_name)

    @staticmethod
    def _build_pmid_index(articles: list[dict]) -> dict[str, int]:
        return {
            str(article.get("pmid", "")).strip(): index
            for index, article in enumerate(articles)
            if str(article.get("pmid", "")).strip()
        }

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.size == 0:
            return embeddings

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
