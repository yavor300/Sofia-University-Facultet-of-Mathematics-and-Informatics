"""Prediction and diagnostic report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openqa_textual.data import get_sample_gold_answer, get_sample_id
from openqa_textual.generation import HeuristicQAGenerator, LocalLLMGenerator


def create_ocr_diagnostic_record(
    ocr_row: dict[str, Any],
    gold_answer: str = "",
    text_field: str = "clean_question",
) -> dict[str, str]:
    """Create one Baseline 0 OCR diagnostic record."""

    ocr_text = ocr_row.get(text_field)
    if ocr_text is None and text_field != "ocr_text":
        ocr_text = ocr_row.get("ocr_text", "")

    return {
        "question_id": str(ocr_row.get("question_id", "")),
        "language": str(ocr_row.get("language") or "English"),
        "ocr_text": str(ocr_text or ""),
        "gold_answer": str(gold_answer or ""),
    }


def build_ocr_diagnostic_rows(
    ocr_rows: list[dict[str, Any]],
    gold_by_id: dict[str, str] | None = None,
    text_field: str = "clean_question",
) -> list[dict[str, str]]:
    """Build Baseline 0 rows from OCR output rows and optional gold answers."""

    gold_by_id = gold_by_id or {}
    return [
        create_ocr_diagnostic_record(
            row,
            gold_answer=gold_by_id.get(str(row.get("question_id", "")), ""),
            text_field=text_field,
        )
        for row in ocr_rows
    ]


def gold_answers_from_dataset_split(dataset_split: Any) -> dict[str, str]:
    """Extract gold answers from a Hugging Face split or list-like dataset."""

    answers: dict[str, str] = {}
    for index in range(len(dataset_split)):
        sample = dataset_split[index]
        try:
            question_id = get_sample_id(sample)
        except KeyError:
            question_id = f"sample-{index:05d}"
        answers[question_id] = get_sample_gold_answer(sample)
    return answers


def gold_answers_from_jsonl(path: str | Path) -> dict[str, str]:
    """Load question_id -> gold answer from a JSONL file."""

    answers: dict[str, str] = {}
    for row in read_jsonl(path):
        question_id = str(row.get("question_id") or row.get("id") or "")
        if not question_id:
            continue
        answers[question_id] = (
            str(row.get("gold_answer") or row.get("answer") or "")
            if not isinstance(row.get("answers"), list)
            else " | ".join(str(answer) for answer in row["answers"])
        ).strip()
    return answers


def create_prediction_record(
    question_id: str,
    answers: list[str],
    language: str = "English",
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an internal prediction record."""

    record: dict[str, Any] = {
        "question_id": str(question_id),
        "answers": [str(answer) for answer in (answers or [""])],
        "language": str(language or "English"),
    }
    if debug is not None:
        record["debug"] = debug
    return record


def build_heuristic_predictions_from_ocr_rows(
    ocr_rows: list[dict[str, Any]],
    text_field: str = "clean_question",
) -> list[dict[str, Any]]:
    """Build Baseline 1 heuristic QA predictions from OCR rows."""

    generator = HeuristicQAGenerator()
    predictions: list[dict[str, Any]] = []
    for row in ocr_rows:
        question = row.get(text_field)
        if question is None and text_field != "ocr_text":
            question = row.get("ocr_text", "")
        language = str(row.get("language") or "English")
        result = generator.generate(str(question or ""), language=language)
        predictions.append(
            create_prediction_record(
                question_id=str(row.get("question_id", "")),
                answers=result.answers,
                language=language,
                debug={
                    "baseline": generator.name,
                    "ocr_text": row.get("ocr_text", ""),
                    "clean_question": str(question or ""),
                    **result.metadata,
                },
            )
        )
    return predictions


def build_llm_predictions_from_ocr_rows(
    ocr_rows: list[dict[str, Any]],
    generator: LocalLLMGenerator,
    text_field: str = "clean_question",
) -> list[dict[str, Any]]:
    """Build Baseline 2 prompted LLM predictions from OCR rows."""

    predictions: list[dict[str, Any]] = []
    for row in ocr_rows:
        question = row.get(text_field)
        if question is None and text_field != "ocr_text":
            question = row.get("ocr_text", "")
        language = str(row.get("language") or "English")
        result = generator.generate(str(question or ""), language=language)
        predictions.append(
            create_prediction_record(
                question_id=str(row.get("question_id", "")),
                answers=result.answers,
                language=language,
                debug={
                    "baseline": generator.name,
                    "ocr_text": row.get("ocr_text", ""),
                    "clean_question": str(question or ""),
                    "model": generator.model_name,
                    **result.metadata,
                },
            )
        )
    return predictions


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
