from openqa_textual.data import get_sample_gold_answer
from openqa_textual.generation import OCRDiagnosticGenerator
from openqa_textual.prediction import (
    build_ocr_diagnostic_rows,
    create_ocr_diagnostic_record,
    gold_answers_from_dataset_split,
)


def test_get_sample_gold_answer_supports_common_fields() -> None:
    assert get_sample_gold_answer({"answer": "A"}) == "A"
    assert get_sample_gold_answer({"answers": ["A", "B"]}) == "A | B"
    assert get_sample_gold_answer({"gold_answers": [{"text": "A"}]}) == "A"
    assert get_sample_gold_answer({"question_id": "q-1"}) == ""


def test_ocr_diagnostic_generator_produces_no_answer() -> None:
    result = OCRDiagnosticGenerator().generate("What is this?", language="English")
    assert result.answers == []
    assert result.metadata["baseline"] == "ocr_only_diagnostic"


def test_create_ocr_diagnostic_record_prefers_clean_question() -> None:
    row = {
        "question_id": "q-1",
        "language": "Bulgarian",
        "ocr_text": "raw",
        "clean_question": "clean",
    }
    assert create_ocr_diagnostic_record(row, gold_answer="gold") == {
        "question_id": "q-1",
        "language": "Bulgarian",
        "ocr_text": "clean",
        "gold_answer": "gold",
    }


def test_build_ocr_diagnostic_rows_attaches_gold_by_id() -> None:
    rows = build_ocr_diagnostic_rows(
        [{"question_id": "q-1", "language": "English", "ocr_text": "question"}],
        gold_by_id={"q-1": "answer"},
        text_field="ocr_text",
    )
    assert rows == [
        {
            "question_id": "q-1",
            "language": "English",
            "ocr_text": "question",
            "gold_answer": "answer",
        }
    ]


def test_gold_answers_from_dataset_split() -> None:
    split = [
        {"question_id": "q-1", "answer": "A"},
        {"id": "q-2", "answers": ["B", "C"]},
    ]
    assert gold_answers_from_dataset_split(split) == {"q-1": "A", "q-2": "B | C"}

