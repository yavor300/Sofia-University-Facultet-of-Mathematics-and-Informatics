"""Training data construction for supervised instruction tuning."""

from __future__ import annotations

from typing import Any

from openqa_textual.data import (
    get_sample_gold_answer,
    get_sample_id,
    get_sample_language,
    get_sample_question_text,
)
from openqa_textual.ocr_postprocess import clean_ocr_question


SYSTEM_PROMPT = "Answer exam-style open questions extracted from images by OCR. Return only the answer."


def build_training_record(
    question: str,
    answer: str,
    language: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Build one chat-format supervised training record."""

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Language: {language or 'English'}\nQuestion: {str(question or '').strip()}",
            },
            {"role": "assistant", "content": str(answer or "").strip()},
        ]
    }


def build_ocr_training_records(
    ocr_rows: list[dict[str, Any]],
    gold_by_id: dict[str, str] | None = None,
    text_field: str = "clean_question",
    skip_missing_answer: bool = True,
    skip_missing_question: bool = True,
) -> list[dict[str, Any]]:
    """Build OCR-question -> answer SFT records from OCR JSONL rows."""

    gold_by_id = gold_by_id or {}
    records = []
    for row in ocr_rows:
        question_id = str(row.get("question_id", ""))
        language = str(row.get("language") or "English")
        question = row.get(text_field)
        if question is None and text_field != "ocr_text":
            question = row.get("ocr_text", "")
        question = clean_ocr_question(str(question or ""), language=language)
        answer = str(gold_by_id.get(question_id) or row.get("gold_answer") or "").strip()

        if skip_missing_question and not question:
            continue
        if skip_missing_answer and not answer:
            continue
        records.append(build_training_record(question=question, answer=answer, language=language))
    return records


def build_clean_question_training_records(
    dataset_split: Any,
    skip_missing_answer: bool = True,
    skip_missing_question: bool = True,
) -> list[dict[str, Any]]:
    """Build clean-question -> answer records for upper-bound comparison only."""

    records = []
    for index in range(len(dataset_split)):
        sample = dataset_split[index]
        question = get_sample_question_text(sample)
        answer = get_sample_gold_answer(sample)
        language = get_sample_language(sample)

        if skip_missing_question and not question:
            continue
        if skip_missing_answer and not answer:
            continue
        records.append(build_training_record(question=question, answer=answer, language=language))
    return records


def gold_answers_from_dataset_split(dataset_split: Any) -> dict[str, str]:
    """Return question_id -> gold answer for train split samples."""

    gold_by_id: dict[str, str] = {}
    for index in range(len(dataset_split)):
        sample = dataset_split[index]
        try:
            question_id = get_sample_id(sample)
        except KeyError:
            question_id = f"sample-{index:05d}"
        gold_by_id[question_id] = get_sample_gold_answer(sample)
    return gold_by_id
