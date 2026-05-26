from openqa_textual.ocr import OCRResult, write_ocr_cache_record
from openqa_textual.retrieval import (
    build_retrieval_index_from_dataset,
    build_retrieval_index_from_ocr_rows,
    gold_answers_from_dataset_split,
)


def test_build_retrieval_index_from_ocr_rows_attaches_gold_and_cleans_text() -> None:
    records = build_retrieval_index_from_ocr_rows(
        [
            {
                "question_id": "q-1",
                "language": "Bulgarian",
                "clean_question": "Кои отпадъци се разпадат най-\nтрудно?",
                "ocr_engine": "tesseract",
                "preprocess_variant": "resize_only",
            }
        ],
        gold_by_id={"q-1": "пластмасови"},
    )

    assert records[0]["question_id"] == "q-1"
    assert records[0]["ocr_question"] == "Кои отпадъци се разпадат най-трудно?"
    assert records[0]["gold_answer"] == "пластмасови"
    assert records[0]["metadata"]["ocr_engine"] == "tesseract"


def test_build_retrieval_index_from_dataset_uses_ocr_cache(tmp_path) -> None:
    write_ocr_cache_record(
        cache_dir=tmp_path,
        split="train",
        question_id="q-1",
        language="English",
        preprocess_variant="contrast",
        result=OCRResult("What is 2 + 2?", 0.9, "easyocr", {"failed": False}),
    )
    split = [{"question_id": "q-1", "language": "English", "answer": "4"}]

    records = build_retrieval_index_from_dataset(
        split,
        split_name="train",
        cache_dir=tmp_path,
        engine="easyocr",
        preprocess_variant="contrast",
    )

    assert records[0]["ocr_question"] == "What is 2 + 2?"
    assert records[0]["gold_answer"] == "4"
    assert records[0]["metadata"]["cache_hit"] is True


def test_gold_answers_from_dataset_split() -> None:
    assert gold_answers_from_dataset_split(
        [{"question_id": "q-1", "answers": ["A", "B"]}]
    ) == {"q-1": "A | B"}

