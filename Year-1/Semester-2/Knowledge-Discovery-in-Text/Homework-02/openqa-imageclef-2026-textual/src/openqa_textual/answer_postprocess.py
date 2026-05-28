"""Answer postprocessing utilities."""

from __future__ import annotations

import re


EXPLANATION_MARKERS = (
    "explanation:",
    "because:",
    "reasoning:",
    "solution:",
    "обяснение:",
    "пояснение:",
    "защото:",
    "spiegazione:",
    "objašnjenje:",
    "објашњење:",
    "解释:",
    "说明:",
)

ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|ans|a|отговор|odgovor|risposta|答案|答)\s*[:：]\s*",
    flags=re.IGNORECASE,
)

NUMBER_WORDS_EN = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def clean_answer(answer: str, language: str | None = None) -> str:
    """Clean a model answer while preserving useful math notation."""

    text = str(answer or "").strip()
    if not text:
        return ""

    text = _strip_wrapping_quotes(text)
    text = _remove_prefixes(text)
    text = _remove_explanations(text)
    text = _first_meaningful_line(text)
    text = _remove_prefixes(text)
    text = _deduplicate_repeated_phrase(text)
    text = _safe_number_word_to_digit(text, language=language)
    text = _normalize_decimal_separator(text, language=language)
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = text.strip(" \t\"'")
    return str(text or "")


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1].strip()
    return stripped


def _remove_prefixes(text: str) -> str:
    previous = None
    cleaned = text
    while previous != cleaned:
        previous = cleaned
        cleaned = ANSWER_PREFIX_RE.sub("", cleaned).strip()
    return cleaned


def _remove_explanations(text: str) -> str:
    lowered = text.casefold()
    cut_positions = []
    for marker in EXPLANATION_MARKERS:
        index = lowered.find(marker.casefold())
        if index > 0:
            cut_positions.append(index)
    if not cut_positions:
        return text
    return text[: min(cut_positions)].strip()


def _first_meaningful_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]

    if len(lines) > 1 and _line_looks_like_intro(first):
        return lines[1].strip()
    return first


def _line_looks_like_intro(line: str) -> bool:
    normalized = line.strip().casefold()
    return normalized in {
        "final answer:",
        "answer:",
        "отговор:",
        "odgovor:",
        "risposta:",
        "答案:",
    }


def _deduplicate_repeated_phrase(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""

    parts = re.split(r"\s*(?:;|,|\|)\s*", normalized)
    if len(parts) > 1 and all(part == parts[0] for part in parts if part):
        separator = ";" if ";" in normalized else "," if "," in normalized else "|"
        return parts[0].strip() if separator else normalized

    tokens = normalized.split()
    if len(tokens) % 2 == 0:
        middle = len(tokens) // 2
        if tokens[:middle] == tokens[middle:]:
            return " ".join(tokens[:middle])
    return normalized


def _safe_number_word_to_digit(text: str, language: str | None = None) -> str:
    if language and str(language).casefold() not in {"english", "en", "eng"}:
        return text
    normalized = text.strip().casefold()
    return NUMBER_WORDS_EN.get(normalized, text)


def _normalize_decimal_separator(text: str, language: str | None = None) -> str:
    if not language or str(language).casefold() not in {
        "bulgarian",
        "bg",
        "croatian",
        "hr",
        "italian",
        "it",
        "serbian",
        "sr",
    }:
        return text

    # Preserve comma decimal answers when the whole answer is a simple number.
    if re.fullmatch(r"-?\d+\.\d+", text.strip()):
        return text.replace(".", ",")
    return text
