"""Safety checks for generated biomedical text."""

from __future__ import annotations

import re
from typing import Any


REQUIRED_DISCLAIMER = (
    "This output is for educational and research purposes only and should not be "
    "interpreted as medical advice, diagnosis, or treatment recommendation."
)

_RISKY_REPLACEMENTS = {
    r"\byou should take\b": "the literature discusses",
    r"\bthis treatment is recommended\b": "some studies evaluate this intervention",
}


class SafetyChecker:
    def __init__(self, ollama_client=None, use_llm: bool = True):
        self.ollama_client = ollama_client
        self.use_llm = use_llm

    def validate(self, text: str) -> str:
        """Return text with LLM-reviewed and deterministic safety guardrails."""
        checked_text = str(text or "").strip()
        if self.use_llm and self.ollama_client is not None and checked_text:
            checked_text = _llm_review_or_original(self.ollama_client, checked_text)

        checked_text = _replace_risky_phrases(checked_text)
        checked_text = _ensure_required_disclaimer(checked_text)
        return checked_text


def _llm_review_or_original(ollama_client, text: str) -> str:
    response = ollama_client.generate_json(
        prompt=_build_safety_review_prompt(text),
        system_prompt=(
            "You are a safety reviewer for biomedical education software. "
            "Return strict JSON only. Preserve the user's content structure. "
            "Do not add new biomedical facts."
        ),
    )
    if response.get("error"):
        return text

    return _reviewed_text(response, text)


def _build_safety_review_prompt(text: str) -> str:
    return f"""Review this generated biomedical text for safety:

{text}

Return exactly one JSON object with these keys:
- is_safe: boolean
- issues: array of short strings
- rewritten_text: string

Rules:
- If the text contains direct patient-level advice, rewrite it into research-support language.
- Replace diagnosis, treatment, dosing, or imperative phrasing with neutral literature-summary phrasing.
- Keep all Markdown headings and PMID/title references where possible.
- Do not add new evidence, claims, recommendations, or citations.
- Ensure the rewritten text is educational/research-oriented only."""


def _reviewed_text(response: dict[str, Any], original_text: str) -> str:
    rewritten_text = str(response.get("rewritten_text") or "").strip()
    is_safe = response.get("is_safe")

    if rewritten_text:
        return rewritten_text
    if is_safe is True:
        return original_text

    return original_text


def _replace_risky_phrases(text: str) -> str:
    checked_text = text
    for pattern, replacement in _RISKY_REPLACEMENTS.items():
        checked_text = re.sub(pattern, replacement, checked_text, flags=re.IGNORECASE)

    return checked_text


def _ensure_required_disclaimer(text: str) -> str:
    if not text:
        return f"## Safety Note\n\n{REQUIRED_DISCLAIMER}"

    if text.rstrip().endswith(REQUIRED_DISCLAIMER):
        return text.rstrip()

    if "## Safety Note" in text:
        return f"{text.rstrip()}\n\n{REQUIRED_DISCLAIMER}"

    return f"{text.rstrip()}\n\n## Safety Note\n\n{REQUIRED_DISCLAIMER}"
