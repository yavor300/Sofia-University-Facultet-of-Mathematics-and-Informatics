"""Final answer generator backed by local Ollama with safe fallback."""

from __future__ import annotations

from typing import Any

from src.agents.safety_checker import REQUIRED_DISCLAIMER, SafetyChecker
from src.preprocessing.article_preprocessor import normalize_text


REQUIRED_HEADINGS = [
    "## Short Answer",
    "## Evidence Summary",
    "## Recommended Papers",
    "## Limitations",
    "## Safety Note",
]

SAFETY_NOTE = (
    REQUIRED_DISCLAIMER
)

SYSTEM_PROMPT = """You generate cautious biomedical literature summaries.
Use only the supplied recommendations, abstracts, and extracted evidence.
Do not use outside knowledge.
Do not provide diagnosis, treatment instructions, dosing guidance, or personalized medical advice.
Always include a safety note."""


class AnswerGenerator:
    def __init__(self, ollama_client, use_llm: bool = True):
        self.ollama_client = ollama_client
        self.use_llm = use_llm
        self.safety_checker = SafetyChecker(ollama_client, use_llm=use_llm)

    def generate(self, user_question: str, recommendations: list[dict], evidence_items: list[dict]) -> str:
        """Generate the final Markdown answer for the user question."""
        question = normalize_text(user_question)
        safe_recommendations = [_normalize_recommendation(item) for item in recommendations]
        safe_evidence = [_normalize_evidence(item) for item in evidence_items]

        if not self.use_llm or self.ollama_client is None:
            return self.safety_checker.validate(_template_answer(question, safe_recommendations, safe_evidence))

        response = self.ollama_client.generate(
            prompt=_build_prompt(question, safe_recommendations, safe_evidence),
            system_prompt=SYSTEM_PROMPT,
        )
        response = normalize_text(response).replace(" ## ", "\n\n## ")

        if not response or _looks_like_ollama_error(response) or not _has_required_sections(response):
            return self.safety_checker.validate(_template_answer(question, safe_recommendations, safe_evidence))

        if "not medical advice" not in response.lower():
            response = f"{response.rstrip()}\n\n{SAFETY_NOTE}"

        return self.safety_checker.validate(response)


def _build_prompt(question: str, recommendations: list[dict], evidence_items: list[dict]) -> str:
    return f"""User question:
{question}

Recommendations:
{_format_recommendations_for_prompt(recommendations)}

Extracted evidence:
{_format_evidence_for_prompt(evidence_items)}

Write a Markdown answer using exactly these section headings:
## Short Answer
## Evidence Summary
## Recommended Papers
## Limitations
## Safety Note

Rules:
- Summarize only the supplied recommendations, abstracts, and evidence items.
- Include PMID and title for recommended papers.
- Do not claim clinical effectiveness unless stated in the supplied evidence.
- Do not provide diagnosis, treatment instructions, dosing guidance, or personalized medical advice.
- In the Safety Note, state that this is educational/research support only and not medical advice."""


def _template_answer(question: str, recommendations: list[dict], evidence_items: list[dict]) -> str:
    evidence_lines = _evidence_summary_lines(evidence_items)
    paper_lines = _recommended_paper_lines(recommendations)

    return "\n\n".join(
        [
            "## Short Answer\n\n"
            f"For the question: \"{question}\", the cached literature recommendations below may help with research-oriented review. "
            "A local LLM summary was not available, so this answer lists supplied evidence without drawing new clinical conclusions.",
            "## Evidence Summary\n\n" + ("\n".join(evidence_lines) if evidence_lines else "- No extracted evidence items were available."),
            "## Recommended Papers\n\n" + ("\n".join(paper_lines) if paper_lines else "- No recommended papers were provided."),
            "## Limitations\n\n"
            "- This answer is based only on the provided cached recommendations and extracted evidence.\n"
            "- Missing abstracts, incomplete extraction, or unavailable local Ollama output may limit the summary.\n"
            "- The listed papers may require full-text review and domain expert interpretation.",
            f"## Safety Note\n\n{SAFETY_NOTE}",
        ]
    )


def _evidence_summary_lines(evidence_items: list[dict]) -> list[str]:
    lines: list[str] = []
    for item in evidence_items:
        pmid = item.get("pmid") or "not_available"
        title = item.get("title") or "Untitled"
        finding = item.get("main_finding") or "not_available"
        outcome = item.get("outcome") or "not_available"
        limitations = item.get("limitations") or "not_available"
        lines.append(
            f"- PMID {pmid}: {title}. Main finding: {finding}. Outcome: {outcome}. Limitations: {limitations}."
        )

    return lines


def _recommended_paper_lines(recommendations: list[dict]) -> list[str]:
    lines: list[str] = []
    for article in recommendations:
        pmid = article.get("pmid") or "not_available"
        title = article.get("title") or "Untitled"
        method = article.get("method") or "not_available"
        score = _score_value(article)
        lines.append(f"- PMID {pmid}: {title} (method: {method}, score: {score}).")

    return lines


def _format_recommendations_for_prompt(recommendations: list[dict]) -> str:
    if not recommendations:
        return "No recommendations provided."

    rows = []
    for article in recommendations:
        rows.append(
            {
                "pmid": article.get("pmid", ""),
                "title": article.get("title", ""),
                "abstract": _truncate(article.get("abstract", ""), 1200),
                "year": article.get("year"),
                "journal": article.get("journal", ""),
                "method": article.get("method", ""),
                "score": article.get("final_score", article.get("score", "")),
                "shared_mesh_terms": article.get("shared_mesh_terms", []),
            }
        )
    return str(rows)


def _format_evidence_for_prompt(evidence_items: list[dict]) -> str:
    if not evidence_items:
        return "No extracted evidence provided."

    return str(evidence_items)


def _normalize_recommendation(article: dict[str, Any]) -> dict:
    return {
        "pmid": normalize_text(article.get("pmid", "")),
        "title": normalize_text(article.get("title", "")),
        "abstract": normalize_text(article.get("abstract", "")),
        "year": article.get("year"),
        "journal": normalize_text(article.get("journal", "")),
        "method": normalize_text(article.get("method", "")),
        "score": article.get("score"),
        "final_score": article.get("final_score"),
        "shared_mesh_terms": article.get("shared_mesh_terms", []),
    }


def _normalize_evidence(item: dict[str, Any]) -> dict:
    return {
        "pmid": normalize_text(item.get("pmid", "")),
        "title": normalize_text(item.get("title", "")),
        "population": normalize_text(item.get("population", "not_available")) or "not_available",
        "exposure_or_intervention": normalize_text(item.get("exposure_or_intervention", "not_available")) or "not_available",
        "outcome": normalize_text(item.get("outcome", "not_available")) or "not_available",
        "main_finding": normalize_text(item.get("main_finding", "not_available")) or "not_available",
        "limitations": normalize_text(item.get("limitations", "not_available")) or "not_available",
    }


def _has_required_sections(text: str) -> bool:
    return all(heading in text for heading in REQUIRED_HEADINGS)


def _score_value(article: dict) -> Any:
    final_score = article.get("final_score")
    if final_score is not None:
        return final_score

    score = article.get("score")
    return score if score is not None else "not_available"


def _looks_like_ollama_error(text: str) -> bool:
    lowered = text.lower()
    return "ollama is unavailable" in lowered or "ollama pull llama3.1:8b" in lowered


def _truncate(value: Any, max_chars: int) -> str:
    text = normalize_text(value)
    if len(text) <= max_chars:
        return text

    return f"{text[:max_chars].rstrip()}..."
