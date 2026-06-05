"""Evidence extraction agent backed by local Ollama with safe fallback."""

from __future__ import annotations

from typing import Any

from src.preprocessing.article_preprocessor import normalize_text


EVIDENCE_KEYS = [
    "pmid",
    "title",
    "population",
    "exposure_or_intervention",
    "outcome",
    "main_finding",
    "limitations",
]

SYSTEM_PROMPT = """You are a biomedical evidence extraction assistant.
Extract only information explicitly present in the provided article title and abstract.
Return strict JSON only.
Do not use outside knowledge.
Do not infer clinical recommendations.
Do not provide diagnosis, treatment advice, dosage advice, or personalized medical advice."""


class EvidenceExtractionAgent:
    def __init__(self, ollama_client, use_llm: bool = True):
        self.ollama_client = ollama_client
        self.use_llm = use_llm

    def extract(self, article: dict) -> dict:
        """Extract structured evidence fields from one article title/abstract."""
        normalized_article = _normalize_article(article)
        if not self.use_llm or self.ollama_client is None:
            return _fallback_extraction(normalized_article)

        response = self.ollama_client.generate_json(
            prompt=_build_prompt(normalized_article),
            system_prompt=SYSTEM_PROMPT,
        )

        if response.get("error"):
            return _fallback_extraction(normalized_article, error=response.get("error"))

        return _normalize_extraction(response, normalized_article)


def _build_prompt(article: dict) -> str:
    return f"""Extract evidence from this article only.

PMID: {article["pmid"]}
Title: {article["title"]}
Abstract: {article["abstract"]}

Return exactly one JSON object with these keys:
- pmid
- population
- exposure_or_intervention
- outcome
- main_finding
- limitations

Rules:
- Use only the title and abstract above.
- Use "not_available" when a field cannot be extracted.
- Do not invent missing study details.
- Do not answer any medical question.
- Do not recommend diagnosis, treatment, dosing, or clinical action.
- Return JSON only."""


def _normalize_article(article: dict[str, Any]) -> dict:
    return {
        "pmid": normalize_text(article.get("pmid", "")),
        "title": normalize_text(article.get("title", "")),
        "abstract": normalize_text(article.get("abstract", "")),
    }


def _normalize_extraction(raw_extraction: dict[str, Any], article: dict) -> dict:
    extraction = {
        "pmid": article["pmid"],
        "title": article["title"],
        "population": _clean_value(raw_extraction.get("population")),
        "exposure_or_intervention": _clean_value(raw_extraction.get("exposure_or_intervention")),
        "outcome": _clean_value(raw_extraction.get("outcome")),
        "main_finding": _clean_value(raw_extraction.get("main_finding")),
        "limitations": _clean_value(raw_extraction.get("limitations")),
    }
    return extraction


def _fallback_extraction(article: dict, error: str | None = None) -> dict:
    extraction = {
        "pmid": article["pmid"],
        "title": article["title"],
        "population": "not_available",
        "exposure_or_intervention": "not_available",
        "outcome": "not_available",
        "main_finding": "not_available",
        "limitations": "not_available",
    }
    if error:
        extraction["llm_error"] = error

    return extraction


def _clean_value(value: Any) -> str:
    if value is None:
        return "not_available"

    text = normalize_text(value)
    if not text or text.lower() in {"null", "none", "n/a", "not specified", "not available"}:
        return "not_available"

    return text
