"""Optional OpenAI API relevance judge for evaluation annotations."""

from __future__ import annotations

import json
from typing import Any

import requests

from src.preprocessing.article_preprocessor import normalize_text


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

SYSTEM_PROMPT = """You are evaluating a biomedical literature recommender system.

Your task is to judge how relevant a recommended PubMed article is to the user's research query.

Important rules:
- Judge only relevance to the research query.
- Do not provide medical advice.
- Do not evaluate treatment recommendations.
- Do not answer the biomedical question.
- Do not invent information that is not present in the provided title, abstract, MeSH terms, or publication types.
Return strict JSON only."""

RELEVANCE_SCALE = """Use this relevance scale:
0 = not relevant
1 = somewhat relevant
2 = relevant
3 = highly relevant"""

JUDGING_CRITERIA = """Judging criteria:
- Match with the main topic of the query.
- Match with the population, if present.
- Match with the exposure or intervention, if present.
- Match with the outcome, if present.
- Biomedical/nutrition relevance.
- Prefer direct relevance over broad topical similarity."""

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relevance": {
            "type": "integer",
            "minimum": 0,
            "maximum": 3,
            "description": "Integer relevance grade from 0 to 3.",
        },
        "reason": {
            "type": "string",
            "description": "Short explanation for the relevance score.",
        },
    },
    "required": ["relevance", "reason"],
}


class OpenAIRelevanceJudge:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_abstract_chars: int = 1200,
        timeout: int = 60,
    ):
        self.api_key = str(api_key or "").strip()
        self.model = model
        self.max_abstract_chars = max(0, int(max_abstract_chars))
        self.timeout = int(timeout)

    def judge(self, query: str, article: dict) -> dict:
        """Return a 0-3 relevance judgment, or a safe fallback on failure."""
        if not self.api_key:
            return _fallback(self.model, "OPENAI_API_KEY is missing.")

        payload = self._build_payload(query=query, article=article)
        try:
            response = requests.post(
                OPENAI_RESPONSES_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as exc:
            return _fallback(self.model, f"OpenAI request failed: {exc}")
        except ValueError as exc:
            return _fallback(self.model, f"OpenAI returned invalid JSON: {exc}")

        raw_output = _extract_response_text(data)
        return validate_judge_output(raw_output=raw_output, judge_model=self.model)

    def preview_prompt(self, query: str, article: dict) -> dict[str, str]:
        """Return the system and user prompts without calling OpenAI."""
        return {
            "system": SYSTEM_PROMPT,
            "user": _build_user_prompt(
                query=query,
                article=article,
                max_abstract_chars=self.max_abstract_chars,
            ),
        }

    def _build_payload(self, query: str, article: dict) -> dict:
        return {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        query=query,
                        article=article,
                        max_abstract_chars=self.max_abstract_chars,
                    ),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "relevance_judgment",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                }
            },
        }


def validate_judge_output(raw_output: Any, judge_model: str) -> dict:
    """Validate raw judge JSON text and return a safe normalized object."""
    raw_text = normalize_text(raw_output)
    if not raw_text:
        return _fallback(
            judge_model,
            "Judge output could not be parsed.",
            judge_error="Missing output text",
            raw_judge_output=raw_output,
        )

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return _fallback(
            judge_model,
            "Judge output could not be parsed.",
            judge_error=f"Invalid JSON: {exc}",
            raw_judge_output=raw_output,
        )

    if not isinstance(parsed, dict):
        return _fallback(
            judge_model,
            "Judge output could not be parsed.",
            judge_error="Parsed JSON was not an object.",
            raw_judge_output=raw_output,
        )

    relevance = _normalize_relevance(parsed.get("relevance"))
    if relevance is None:
        return _fallback(
            judge_model,
            "Judge output could not be parsed.",
            judge_error="Missing or non-numeric relevance.",
            raw_judge_output=raw_output,
        )

    reason = parsed.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    return {
        "relevance": relevance,
        "reason": normalize_text(reason) or "No reason provided.",
        "judge_model": judge_model,
        "raw_judge_output": raw_output,
    }


def _build_user_prompt(query: str, article: dict, max_abstract_chars: int) -> str:
    pmid = normalize_text(article.get("pmid", ""))
    title = normalize_text(article.get("title", ""))
    year = article.get("year")
    journal = normalize_text(article.get("journal", ""))
    publication_types = ", ".join(_clean_list(article.get("publication_types", []))) or "not_available"
    mesh_terms = ", ".join(_clean_list(article.get("mesh_terms", []))) or "not_available"
    abstract_snippet = _truncate(article.get("abstract", ""), max_abstract_chars) or "not_available"

    return f"""You are evaluating a biomedical literature recommender system.

Your task is to judge how relevant a recommended PubMed article is to the user's research query.

{RELEVANCE_SCALE}

{JUDGING_CRITERIA}

Important rules:
- Judge only relevance to the research query.
- Do not provide medical advice.
- Do not evaluate treatment recommendations.
- Do not invent information that is not present in the provided title, abstract, MeSH terms, or publication types.
- Return strict JSON only.

User query:
{normalize_text(query)}

Recommended PubMed article:
PMID: {pmid}
Title: {title}
Year: {year}
Journal: {journal}
Publication types: {publication_types}
MeSH terms: {mesh_terms}
Abstract snippet:
{abstract_snippet}

Return JSON only:
{{
  "relevance": 0,
  "reason": "short explanation"
}}"""


def _extract_response_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []
    for output_item in data.get("output", []) if isinstance(data.get("output"), list) else []:
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content", []) if isinstance(output_item.get("content"), list) else []:
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

    return "\n".join(texts).strip()


def _normalize_relevance(value: Any) -> int | None:
    if isinstance(value, bool):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    relevance = int(round(numeric))
    return max(0, min(3, relevance))


def _fallback(
    model: str,
    reason: str,
    judge_error: str | None = None,
    raw_judge_output: Any | None = None,
) -> dict:
    return {
        "relevance": None,
        "reason": reason,
        "judge_model": model,
        "judge_error": judge_error or reason,
        "raw_judge_output": raw_judge_output,
    }


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]

    return [text for item in values if (text := normalize_text(item))]


def _truncate(value: Any, max_chars: int) -> str:
    text = normalize_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    return f"{text[:max_chars].rstrip()}..."
