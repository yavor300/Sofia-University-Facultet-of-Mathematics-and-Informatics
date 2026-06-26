"""Prompt templates for optional LLM relation verification."""

from __future__ import annotations

from typing import Any


def build_relation_verification_prompt(
    article_text: str,
    candidate: dict[str, Any],
    allowed_predicates: list[str],
) -> str:
    """Build a strict JSON-only prompt for GutBrainIE relation verification."""
    allowed = ", ".join([*allowed_predicates, "no_relation"])
    return f"""You are a biomedical relation extraction assistant.
Given the article text and two entity mentions, decide whether one of the allowed GutBrainIE predicates holds.
Return only JSON.

Allowed predicates: {allowed}.

Article:
{article_text}

Subject: {candidate["subject_text_span"]} [{candidate["subject_label"]}]
Object: {candidate["object_text_span"]} [{candidate["object_label"]}]

Return:
{{"predicate": "...", "confidence": 0.0}}"""
