"""Conservative OCR text cleanup for image-rendered questions."""

from __future__ import annotations

import re


QUESTION_STARTERS = {
    "English": (
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "is",
        "are",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
    ),
    "Bulgarian": (
        "как",
        "какво",
        "какъв",
        "каква",
        "какви",
        "кой",
        "коя",
        "кое",
        "кои",
        "къде",
        "кога",
        "защо",
        "дали",
        "колко",
        "чий",
        "чия",
        "чие",
        "чии",
        "за какво",
    ),
}

GENERIC_QUESTION_STARTERS = tuple(
    sorted({starter for starters in QUESTION_STARTERS.values() for starter in starters}, key=len, reverse=True)
)

SYMBOL_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u200b": "",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00d7": "x",
        "\u2217": "*",
        "\u00f7": "/",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2260": "!=",
        "\uFFFD": "",
    }
)


def normalize_whitespace(text: str) -> str:
    """Collapse duplicated spaces/newlines while preserving normal word spacing."""

    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t\f\v]+", " ", cleaned)
    cleaned = re.sub(r" *\n+ *", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def fix_common_ocr_errors(text: str, language: str | None = None) -> str:
    """Apply conservative OCR typo fixes without rewriting domain-specific terms."""

    del language
    cleaned = str(text or "")
    cleaned = re.sub(r"[□■▯]", "", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "o", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z])1(?=[A-Za-z])", "l", cleaned)
    cleaned = re.sub(r"(?<=\d)[Oo](?=\d)", "0", cleaned)
    cleaned = re.sub(r"(?<=\d)[Il](?=\d)", "1", cleaned)
    cleaned = re.sub(r"\b([A-Za-z]{2,})rn(?=[A-Za-z]{2,}\b)", r"\1m", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([({\[])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)}\]])", r"\1", cleaned)
    return cleaned


def normalize_quotes_and_symbols(text: str) -> str:
    """Normalize quote, dash, math, and invisible OCR artifacts."""

    cleaned = str(text or "").translate(SYMBOL_TRANSLATION)
    cleaned = re.sub(r"\.{3,}", "...", cleaned)
    cleaned = re.sub(r"([!?]){2,}", r"\1", cleaned)
    return cleaned


def restore_question_mark(text: str) -> str:
    """Add or restore a question mark when the cleaned text looks like a question."""

    cleaned = str(text or "").strip()
    if not cleaned or cleaned.endswith("?"):
        return cleaned

    if not _looks_like_question(cleaned):
        return cleaned

    if cleaned.endswith((".", "!", ";", ":")):
        return cleaned[:-1].rstrip() + "?"
    return cleaned + "?"


def clean_ocr_question(text: str, language: str | None = None) -> str:
    """Run the full OCR question cleanup pipeline."""

    cleaned = normalize_quotes_and_symbols(text)
    cleaned = _remove_repeated_lines(cleaned)
    cleaned = _fix_broken_hyphenation(cleaned, language=language)
    cleaned = normalize_whitespace(cleaned)
    cleaned = _strip_edge_artifacts(cleaned)
    cleaned = fix_common_ocr_errors(cleaned, language=language)
    cleaned = normalize_whitespace(cleaned)
    cleaned = restore_question_mark(cleaned)
    return cleaned


def _fix_broken_hyphenation(text: str, language: str | None = None) -> str:
    cleaned = str(text or "")

    if language and str(language).casefold().strip() == "bulgarian":
        cleaned = re.sub(r"\b(най)-\s*\n\s*(?=\w)", r"\1-", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])", "", cleaned)
    cleaned = re.sub(r"(?<=[А-Яа-я])-\s*\n\s*(?=[А-Яа-я])", "-", cleaned)
    return cleaned


def _remove_repeated_lines(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines()]
    if len(lines) < 3:
        return str(text or "")

    seen_counts: dict[str, int] = {}
    for line in lines:
        key = line.casefold()
        if key:
            seen_counts[key] = seen_counts.get(key, 0) + 1

    kept = []
    for line in lines:
        key = line.casefold()
        if key and len(line) <= 80 and seen_counts.get(key, 0) > 1:
            if key in {existing.casefold() for existing in kept}:
                continue
        kept.append(line)
    return "\n".join(kept)


def _strip_edge_artifacts(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^[^\wА-Яа-я\d]+", "", cleaned)
    cleaned = re.sub(r"[|~^_`]+$", "", cleaned).strip()
    return cleaned


def _looks_like_question(text: str) -> bool:
    lowered = text.casefold().strip()
    starters = GENERIC_QUESTION_STARTERS
    return any(lowered == starter or lowered.startswith(starter + " ") for starter in starters)

