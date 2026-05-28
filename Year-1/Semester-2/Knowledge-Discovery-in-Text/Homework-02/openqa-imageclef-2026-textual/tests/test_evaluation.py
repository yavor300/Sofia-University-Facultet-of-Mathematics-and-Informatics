from openqa_textual.evaluation import (
    answer_candidates,
    compare_systems,
    evaluate_predictions,
    evaluate_dev_report,
    gold_answers_from_records,
    levenshtein_distance,
    normalize_answer,
    ocr_character_error_rate,
    score_answer,
)


def test_normalize_answer_is_case_and_punctuation_insensitive() -> None:
    assert normalize_answer("  The Answer! ") == "the answer"


def test_answer_candidates_splits_common_gold_format() -> None:
    assert answer_candidates("A | B; C") == ["A", "B", "C"]


def test_score_answer_uses_best_gold_candidate() -> None:
    score = score_answer("second answer", "first answer | second answer")

    assert score["normalized_exact_match"] == 1.0
    assert score["token_f1"] == 1.0


def test_evaluate_predictions_summarizes_metrics_and_overfit_signals() -> None:
    report = evaluate_predictions(
        [
            {"question_id": "q-1", "language": "English", "answers": ["Paris"]},
            {"question_id": "q-2", "language": "English", "answers": ["Paris"]},
        ],
        gold_by_id={"q-1": "Paris", "q-2": "London"},
        train_gold_answers=["Paris"],
    )

    assert report["summary"]["total"] == 2
    assert report["summary"]["normalized_exact_match"] == 0.5
    assert report["summary"]["train_answer_copy_rate"] == 1.0
    assert report["summary"]["repeated_answer_rate"] == 1.0
    assert report["by_language"]["English"]["total"] == 2


def test_compare_systems_ranks_by_normalized_exact_match() -> None:
    report = compare_systems(
        {
            "bad": [{"question_id": "q-1", "answers": ["wrong"]}],
            "good": [{"question_id": "q-1", "answers": ["answer"]}],
        },
        gold_by_id={"q-1": "answer"},
    )

    assert report["ranking"][0]["system"] == "good"
    assert report["pairwise"][0]["left"] == "bad"


def test_gold_answers_from_records_supports_answers_list() -> None:
    assert gold_answers_from_records([{"question_id": "q-1", "answers": ["A", "B"]}]) == {
        "q-1": "A | B"
    }


def test_levenshtein_distance() -> None:
    assert levenshtein_distance("kitten", "sitting") == 3


def test_ocr_character_error_rate_uses_debug_clean_question() -> None:
    cer = ocr_character_error_rate(
        {"q-1": {"debug": {"clean_question": "what is two"}}},
        {"q-1": "what is 2"},
    )

    assert cer > 0


def test_evaluate_dev_report_contains_phase_14_metrics() -> None:
    report = evaluate_dev_report(
        predictions=[
            {
                "question_id": "q-1",
                "language": "English",
                "answers": ["Paris"],
                "debug": {"clean_question": "capital of france"},
            }
        ],
        gold_records=[
            {
                "question_id": "q-1",
                "language": "English",
                "gold_answer": "Paris",
                "clean_question": "capital of France",
            }
        ],
        experiment_name="test_exp",
        ocr_engine="tesseract",
        preprocessing="resize_only",
        generation_model="Qwen",
        retrieval="bm25",
        notes="unit test",
    )

    assert report["experiment_name"] == "test_exp"
    assert report["ocr_engine"] == "tesseract"
    assert report["metrics"]["exact_match"] == 1.0
    assert report["metrics"]["token_f1"] == 1.0
    assert "bleu" in report["metrics"]
    assert "rouge_l" in report["metrics"]
    assert report["metrics"]["ocr_character_error_rate"] == 0.0
    assert report["notes"] == "unit test"
