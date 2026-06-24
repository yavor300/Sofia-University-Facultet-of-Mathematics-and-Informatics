"""Submission format validation helpers."""

from __future__ import annotations

from typing import Any


def validate_t611_payload(payload: dict[str, Any]) -> bool:
    """Validate the minimal T611 JSON shape used by the project."""
    if not isinstance(payload, dict):
        return False
    for pmid, article_payload in payload.items():
        if not isinstance(pmid, str):
            return False
        if not isinstance(article_payload, dict) or "entities" not in article_payload:
            return False
        if not isinstance(article_payload["entities"], list):
            return False
        for entity in article_payload["entities"]:
            if not _valid_t611_entity(entity):
                return False
    return True


def _valid_t611_entity(entity: Any) -> bool:
    if not isinstance(entity, dict):
        return False
    required = {"start_idx", "end_idx", "location", "text_span", "label"}
    if set(entity) != required:
        return False
    return (
        isinstance(entity["start_idx"], int)
        and isinstance(entity["end_idx"], int)
        and entity["location"] in {"title", "abstract"}
        and isinstance(entity["text_span"], str)
        and isinstance(entity["label"], str)
        and entity["start_idx"] <= entity["end_idx"]
    )
