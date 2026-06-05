"""Recommendation explanation agent with local Ollama and safe fallback."""

from __future__ import annotations

from typing import Any

from src.preprocessing.article_preprocessor import normalize_text


FALLBACK_EXPLANATION = (
    "This article is recommended because it is similar to the query or selected article "
    "based on semantic similarity, shared MeSH terms, or graph embedding proximity."
)

SYSTEM_PROMPT = """You explain biomedical literature recommendations.
Use only the provided article metadata, recommendation method, scores, and shared MeSH terms.
Do not make clinical claims.
Do not provide diagnosis, treatment advice, dosage advice, or personalized medical advice.
Keep the explanation concise."""


class RecommendationExplainer:
    def __init__(self, ollama_client, use_llm: bool = True):
        self.ollama_client = ollama_client
        self.use_llm = use_llm

    def explain(self, seed_article: dict | None, recommended_article: dict) -> str:
        """Explain why a recommendation was returned."""
        if not self.use_llm or self.ollama_client is None:
            return _fallback_explanation(recommended_article)

        response = self.ollama_client.generate(
            prompt=_build_prompt(seed_article, recommended_article),
            system_prompt=SYSTEM_PROMPT,
        )
        response = normalize_text(response)

        if not response or _looks_like_ollama_error(response):
            return _fallback_explanation(recommended_article)

        return response


def _build_prompt(seed_article: dict | None, recommended_article: dict) -> str:
    seed_text = "No selected seed article was provided."
    if seed_article:
        seed_text = (
            f"Seed PMID: {seed_article.get('pmid', 'not_available')}\n"
            f"Seed title: {seed_article.get('title', 'not_available')}"
        )

    return f"""Explain this recommendation using only the data below.

{seed_text}

Recommended PMID: {recommended_article.get("pmid", "not_available")}
Recommended title: {recommended_article.get("title", "not_available")}
Recommended year: {recommended_article.get("year", "not_available")}
Recommended journal: {recommended_article.get("journal", "not_available")}
Method: {_display_method(recommended_article.get("method"))}
Score: {recommended_article.get("score", recommended_article.get("final_score", "not_available"))}
Semantic score: {recommended_article.get("semantic_score", "not_available")}
Graph score: {recommended_article.get("graph_score", "not_available")}
Shared MeSH terms: {_shared_terms_text(recommended_article)}

Requirements:
- Mention the recommendation method.
- Mention shared MeSH terms if they are available.
- Do not claim clinical effectiveness, diagnosis, treatment benefit, or causality.
- Return two concise sentences at most."""


def _fallback_explanation(recommended_article: dict) -> str:
    method = _display_method(recommended_article.get("method"))
    explanation = f"{FALLBACK_EXPLANATION} Method: {method}."
    shared_terms = _shared_terms_text(recommended_article)
    if shared_terms != "not_available":
        explanation += f" Shared MeSH terms: {shared_terms}."

    return explanation


def _display_method(method: Any) -> str:
    method_text = normalize_text(method).lower()
    if method_text == "mesh_overlap":
        return "MeSH overlap"
    if method_text in {"semantic", "graph", "hybrid"}:
        return method_text
    return method_text or "not_available"


def _shared_terms_text(recommended_article: dict) -> str:
    terms = recommended_article.get("shared_mesh_terms") or []
    if isinstance(terms, str):
        terms = [terms]

    cleaned_terms = [normalize_text(term) for term in terms if normalize_text(term)]
    return ", ".join(cleaned_terms) if cleaned_terms else "not_available"


def _looks_like_ollama_error(text: str) -> bool:
    lowered = text.lower()
    return "ollama is unavailable" in lowered or "ollama pull llama3.1:8b" in lowered
