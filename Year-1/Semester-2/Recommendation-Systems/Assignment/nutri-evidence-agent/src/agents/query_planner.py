"""Query planning agent backed by local Ollama with safe fallback."""

from __future__ import annotations

from typing import Any


PLANNER_KEYS = [
    "population",
    "exposure",
    "intervention",
    "outcome",
    "question_type",
    "pubmed_query",
]

SYSTEM_PROMPT = """You are a biomedical literature search planner.
Extract search-planning fields from the user's question.
Return strict JSON only.
Do not answer the medical question.
Do not provide diagnosis, treatment advice, dosage advice, or personalized medical advice."""


class QueryPlannerAgent:
    def __init__(self, ollama_client, use_llm: bool = True):
        self.ollama_client = ollama_client
        self.use_llm = use_llm

    def plan(self, user_question: str) -> dict:
        """Return PICO-style planning fields and a PubMed search query."""
        question = str(user_question or "").strip()
        if not question or not self.use_llm or self.ollama_client is None:
            return _fallback_plan(question)

        response = self.ollama_client.generate_json(
            prompt=_build_prompt(question),
            system_prompt=SYSTEM_PROMPT,
        )

        if response.get("error"):
            return _fallback_plan(question, error=response.get("error"))

        return _normalize_plan(response, question)


def _build_prompt(user_question: str) -> str:
    return f"""Analyze this biomedical literature-search question:

{user_question}

Return exactly one JSON object with these keys:
- population
- exposure
- intervention
- outcome
- question_type
- pubmed_query

Rules:
- Use null when a field is not present.
- Generate pubmed_query as a concise PubMed keyword query.
- Do not answer the question.
- Do not recommend diagnosis, treatment, dosing, or clinical action.
- Return JSON only."""


def _normalize_plan(raw_plan: dict[str, Any], original_question: str) -> dict:
    plan = {key: _clean_value(raw_plan.get(key)) for key in PLANNER_KEYS}
    if not plan["pubmed_query"]:
        plan["pubmed_query"] = original_question

    return plan


def _fallback_plan(user_question: str, error: str | None = None) -> dict:
    plan = {
        "population": None,
        "exposure": None,
        "intervention": None,
        "outcome": None,
        "question_type": None,
        "pubmed_query": user_question,
    }
    if error:
        plan["llm_error"] = error

    return plan


def _clean_value(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "not specified"}:
        return None

    return text
