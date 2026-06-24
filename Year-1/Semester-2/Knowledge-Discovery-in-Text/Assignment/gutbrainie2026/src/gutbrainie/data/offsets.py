"""Entity offset validation utilities."""

from __future__ import annotations

from typing import Any


def validate_entity_offsets(article_row: Any, entity_row: Any) -> bool:
    """Check whether text_span equals text[start_idx:end_idx]."""
    location = _row_value(entity_row, "location")
    if location not in {"title", "abstract"}:
        return False

    text = str(_row_value(article_row, location))
    text_span = str(_row_value(entity_row, "text_span"))

    try:
        start_idx = int(_row_value(entity_row, "start_idx"))
        end_idx = int(_row_value(entity_row, "end_idx"))
    except (TypeError, ValueError):
        return False

    if start_idx < 0 or end_idx < start_idx or end_idx > len(text):
        return False

    return text[start_idx:end_idx] == text_span or text[start_idx : end_idx + 1] == text_span


def resolve_exclusive_end(text: str, start_idx: int, end_idx: int, text_span: str) -> int:
    """Resolve dataset end offsets to Python-exclusive end offsets.

    GutBrainIE CSV annotations in the local release use inclusive end offsets,
    while Python slicing and GLiNER training use exclusive end offsets. This
    helper accepts either convention and returns the exclusive end.
    """
    if text[start_idx:end_idx] == text_span:
        return end_idx
    if text[start_idx : end_idx + 1] == text_span:
        return end_idx + 1
    return end_idx


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row[key]
    return row[key]
