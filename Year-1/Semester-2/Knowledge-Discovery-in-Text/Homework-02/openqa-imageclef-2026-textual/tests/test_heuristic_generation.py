from openqa_textual.generation import (
    HeuristicQAGenerator,
    answer_arithmetic_comparison,
    answer_arithmetic_expression,
    is_yes_no_question,
)
from openqa_textual.prediction import build_heuristic_predictions_from_ocr_rows


def test_answer_arithmetic_expression() -> None:
    assert answer_arithmetic_expression("What is 2 + 2?") == "4"
    assert answer_arithmetic_expression("Колко е 10 / 4?") == "2.5"


def test_answer_arithmetic_comparison() -> None:
    assert answer_arithmetic_comparison("Is 2 + 2 = 4?", "English") == "yes"
    assert answer_arithmetic_comparison("Вярно ли е 2 + 2 = 5?", "Bulgarian") == "не"


def test_yes_no_detection_without_guessing_answer() -> None:
    assert is_yes_no_question("Is water wet?", "English")
    assert is_yes_no_question("Дали рибите плуват?", "Bulgarian")

    result = HeuristicQAGenerator().generate("Is water wet?", language="English")
    assert result.answers == [""]
    assert result.metadata["rule"] == "yes_no_detected_unanswered"


def test_heuristic_generator_fallback() -> None:
    result = HeuristicQAGenerator().generate("Name a living organism.", language="English")
    assert result.answers == [""]
    assert result.metadata["rule"] == "fallback_empty"


def test_build_heuristic_predictions_from_ocr_rows() -> None:
    predictions = build_heuristic_predictions_from_ocr_rows(
        [
            {
                "question_id": "q-1",
                "language": "English",
                "ocr_text": "raw",
                "clean_question": "What is 3 * 3?",
            }
        ]
    )

    assert predictions[0]["question_id"] == "q-1"
    assert predictions[0]["answers"] == ["9"]
    assert predictions[0]["debug"]["rule"] == "arithmetic_expression"

