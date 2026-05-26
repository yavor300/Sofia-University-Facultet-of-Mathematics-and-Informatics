"""Retrieval index construction utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openqa_textual.data import get_sample_gold_answer, get_sample_id, get_sample_language
from openqa_textual.ocr import OCRResult, load_ocr_cache_record
from openqa_textual.ocr_postprocess import clean_ocr_question
from openqa_textual.prediction import read_jsonl, write_jsonl


@dataclass(slots=True)
class RetrievalIndexRecord:
    question_id: str
    language: str
    ocr_question: str
    gold_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_retrieval_index_from_ocr_rows(
    ocr_rows: list[dict[str, Any]],
    gold_by_id: dict[str, str] | None = None,
    text_field: str = "clean_question",
) -> list[dict[str, Any]]:
    """Build retrieval index records from OCR JSONL rows."""

    gold_by_id = gold_by_id or {}
    records = []
    for row in ocr_rows:
        question_id = str(row.get("question_id", ""))
        language = str(row.get("language") or "English")
        question = row.get(text_field)
        if question is None and text_field != "ocr_text":
            question = row.get("ocr_text", "")
        question = clean_ocr_question(str(question or ""), language=language)
        records.append(
            RetrievalIndexRecord(
                question_id=question_id,
                language=language,
                ocr_question=question,
                gold_answer=gold_by_id.get(question_id, str(row.get("gold_answer") or "")),
                metadata={
                    "source": row.get("source"),
                    "split": row.get("split"),
                    "ocr_engine": row.get("ocr_engine"),
                    "preprocess_variant": row.get("preprocess_variant"),
                    "confidence": row.get("confidence"),
                },
            ).to_dict()
        )
    return records


def build_retrieval_index_from_dataset(
    dataset_split: Any,
    split_name: str,
    cache_dir: str | Path,
    engine: str,
    preprocess_variant: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build retrieval records from a dataset split using existing OCR cache only."""

    records = []
    total = len(dataset_split) if limit is None else min(len(dataset_split), max(limit, 0))
    for index in range(total):
        sample = dataset_split[index]
        try:
            question_id = get_sample_id(sample)
        except KeyError:
            question_id = f"sample-{index:05d}"
        language = get_sample_language(sample)
        gold_answer = get_sample_gold_answer(sample)

        cache_record = load_ocr_cache_record(
            cache_dir=cache_dir,
            split=split_name,
            engine=engine,
            preprocess_variant=preprocess_variant,
            question_id=question_id,
        )
        ocr_result = cache_record.to_result() if cache_record else OCRResult("", None, engine, {})
        ocr_question = clean_ocr_question(ocr_result.text, language=language)

        records.append(
            RetrievalIndexRecord(
                question_id=question_id,
                language=language,
                ocr_question=ocr_question,
                gold_answer=gold_answer,
                metadata={
                    "source": f"{split_name}:{index}",
                    "split": split_name,
                    "ocr_engine": engine,
                    "preprocess_variant": preprocess_variant,
                    "confidence": ocr_result.confidence,
                    "cache_hit": cache_record is not None,
                },
            ).to_dict()
        )
    return records


def gold_answers_from_dataset_split(dataset_split: Any) -> dict[str, str]:
    """Return question_id -> gold answer for a split."""

    gold_by_id: dict[str, str] = {}
    for index in range(len(dataset_split)):
        sample = dataset_split[index]
        try:
            question_id = get_sample_id(sample)
        except KeyError:
            question_id = f"sample-{index:05d}"
        gold_by_id[question_id] = get_sample_gold_answer(sample)
    return gold_by_id


def load_ocr_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def write_retrieval_index(path: str | Path, records: list[dict[str, Any]]) -> None:
    write_jsonl(path, records)

