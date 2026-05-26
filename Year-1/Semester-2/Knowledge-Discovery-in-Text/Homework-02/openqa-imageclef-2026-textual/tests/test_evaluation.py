from openqa_textual.evaluation import (
    answer_candidates,
    compare_systems,
    evaluate_predictions,
    normalize_answer,
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
