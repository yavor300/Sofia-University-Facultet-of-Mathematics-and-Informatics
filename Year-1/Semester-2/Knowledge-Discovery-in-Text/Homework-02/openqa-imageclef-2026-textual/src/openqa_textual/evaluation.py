"""Evaluation helpers for OpenQA prediction comparisons."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import string
from typing import Any

from openqa_textual.prediction import read_jsonl


def normalize_answer(text: str) -> str:
    """Normalize answers for lightweight exact-match and overlap metrics."""

    value = str(text or "").casefold()
    value = value.replace("’", "'").replace("`", "'")
    punctuation = string.punctuation + "„“”‚‘«»…"
    value = value.translate(str.maketrans({char: " " for char in punctuation}))
    value = re.sub(r"\s+", " ", value, flags=re.UNICODE)
    return value.strip()


def answer_candidates(gold_answer: str | list[str]) -> list[str]:
    """Split common multi-answer gold formats into candidate answers."""

    if isinstance(gold_answer, list):
        return [str(answer).strip() for answer in gold_answer if str(answer).strip()]
    text = str(gold_answer or "").strip()
    if not text:
        return []
    candidates = [part.strip() for part in re.split(r"\s+\|\s+|\s*;\s*", text) if part.strip()]
    return candidates or [text]


def prediction_answer(prediction: dict[str, Any]) -> str:
    """Return the primary answer from an internal prediction record."""

    answers = prediction.get("answers")
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return str(prediction.get("answer") or "")


def score_answer(predicted: str, gold_answer: str | list[str]) -> dict[str, float]:
    """Score one predicted answer against one or more gold answer candidates."""

    candidates = answer_candidates(gold_answer)
    if not candidates:
        return {"exact_match": 0.0, "normalized_exact_match": 0.0, "token_f1": 0.0, "char_similarity": 0.0}

    scores = [_score_against_candidate(predicted, candidate) for candidate in candidates]
    return max(scores, key=lambda item: (item["normalized_exact_match"], item["token_f1"], item["char_similarity"]))


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    gold_by_id: dict[str, str],
    system_name: str = "system",
    train_gold_answers: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate one system and include simple overfit/copying indicators."""

    rows = []
    for prediction in predictions:
        question_id = str(prediction.get("question_id", ""))
        gold = gold_by_id.get(question_id, "")
        answer = prediction_answer(prediction)
        score = score_answer(answer, gold)
        rows.append(
            {
                "question_id": question_id,
                "language": str(prediction.get("language") or ""),
                "prediction": answer,
                "gold_answer": gold,
                **score,
            }
        )

    summary = _summarize_rows(rows)
    summary["system"] = system_name
    summary.update(overfit_indicators(rows, train_gold_answers=train_gold_answers))
    by_language = {
        language: _summarize_rows(language_rows)
        for language, language_rows in _group_by_language(rows).items()
    }
    return {"system": system_name, "summary": summary, "by_language": by_language, "rows": rows}


def compare_systems(
    systems: dict[str, list[dict[str, Any]]],
    gold_by_id: dict[str, str],
    train_gold_by_id: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate and compare multiple prediction files."""

    train_gold_answers = list((train_gold_by_id or {}).values())
    evaluations = {
        name: evaluate_predictions(
            predictions,
            gold_by_id=gold_by_id,
            system_name=name,
            train_gold_answers=train_gold_answers,
        )
        for name, predictions in systems.items()
    }
    ranking = sorted(
        (
            {
                "system": name,
                **evaluation["summary"],
            }
            for name, evaluation in evaluations.items()
        ),
        key=lambda item: (
            item["normalized_exact_match"],
            item["token_f1"],
            item["char_similarity"],
        ),
        reverse=True,
    )
    return {
        "ranking": ranking,
        "systems": evaluations,
        "pairwise": pairwise_deltas(evaluations),
    }


def pairwise_deltas(evaluations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact pairwise metric differences between systems."""

    names = list(evaluations)
    rows = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = evaluations[left_name]["summary"]
            right = evaluations[right_name]["summary"]
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "normalized_exact_match_delta": round(
                        left["normalized_exact_match"] - right["normalized_exact_match"],
                        6,
                    ),
                    "token_f1_delta": round(left["token_f1"] - right["token_f1"], 6),
                    "char_similarity_delta": round(
                        left["char_similarity"] - right["char_similarity"],
                        6,
                    ),
                }
            )
    return rows


def overfit_indicators(
    rows: list[dict[str, Any]],
    train_gold_answers: list[str] | None = None,
) -> dict[str, Any]:
    """Compute simple indicators that may suggest train-style overfitting."""

    predictions = [normalize_answer(row["prediction"]) for row in rows if normalize_answer(row["prediction"])]
    train_gold = {normalize_answer(answer) for answer in (train_gold_answers or []) if normalize_answer(answer)}
    counts = Counter(predictions)
    copied = sum(1 for answer in predictions if answer in train_gold)
    repeated = sum(count for _, count in counts.items() if count > 1)
    total = len(rows) or 1
    non_empty = len(predictions) or 1
    return {
        "train_answer_copy_rate": round(copied / total, 6),
        "repeated_answer_rate": round(repeated / non_empty, 6),
        "unique_answer_count": len(counts),
        "top_repeated_answers": [
            {"answer": answer, "count": count} for answer, count in counts.most_common(5)
        ],
    }


def load_prediction_file(path: str | Path) -> list[dict[str, Any]]:
    """Load predictions from either JSON list or JSONL."""

    prediction_path = Path(path)
    if prediction_path.suffix == ".jsonl":
        return read_jsonl(prediction_path)
    with prediction_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("predictions"), list):
        return data["predictions"]
    raise ValueError(f"Unsupported prediction file shape: {prediction_path}")


def _score_against_candidate(predicted: str, gold: str) -> dict[str, float]:
    normalized_predicted = normalize_answer(predicted)
    normalized_gold = normalize_answer(gold)
    return {
        "exact_match": 1.0 if str(predicted).strip() == str(gold).strip() else 0.0,
        "normalized_exact_match": 1.0 if normalized_predicted and normalized_predicted == normalized_gold else 0.0,
        "token_f1": _token_f1(normalized_predicted, normalized_gold),
        "char_similarity": _char_similarity(normalized_predicted, normalized_gold),
    }


def _token_f1(predicted: str, gold: str) -> float:
    predicted_tokens = predicted.split()
    gold_tokens = gold.split()
    if not predicted_tokens or not gold_tokens:
        return 0.0
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return round((2 * precision * recall) / (precision + recall), 6)


def _char_similarity(predicted: str, gold: str) -> float:
    if not predicted or not gold:
        return 0.0
    try:
        from rapidfuzz.fuzz import ratio

        return round(float(ratio(predicted, gold)) / 100.0, 6)
    except ImportError:
        from difflib import SequenceMatcher

        return round(SequenceMatcher(None, predicted, gold).ratio(), 6)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    denominator = total or 1
    non_empty = sum(1 for row in rows if str(row["prediction"]).strip())
    return {
        "total": total,
        "with_gold": sum(1 for row in rows if str(row["gold_answer"]).strip()),
        "non_empty_rate": round(non_empty / denominator, 6),
        "exact_match": _average(rows, "exact_match"),
        "normalized_exact_match": _average(rows, "normalized_exact_match"),
        "token_f1": _average(rows, "token_f1"),
        "char_similarity": _average(rows, "char_similarity"),
    }


def _average(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get(field, 0.0)) for row in rows) / len(rows), 6)


def _group_by_language(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("language") or "Unknown"].append(row)
    return dict(grouped)
