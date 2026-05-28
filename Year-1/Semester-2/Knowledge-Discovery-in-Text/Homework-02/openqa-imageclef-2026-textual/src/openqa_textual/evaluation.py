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


def evaluate_dev_report(
    predictions: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    experiment_name: str = "experiment",
    ocr_engine: str = "",
    preprocessing: str = "",
    generation_model: str = "",
    retrieval: str = "",
    notes: str = "",
    include_bertscore: bool = False,
) -> dict[str, Any]:
    """Build a Phase 14 single-system evaluation report."""

    gold_by_id = gold_answers_from_records(gold_records)
    clean_questions_by_id = clean_questions_from_records(gold_records)
    evaluation = evaluate_predictions(predictions, gold_by_id=gold_by_id, system_name=experiment_name)
    rows = evaluation["rows"]
    predictions_by_id = {str(row.get("question_id", "")): row for row in predictions}

    pred_texts = [row["prediction"] for row in rows]
    gold_texts = [str(row["gold_answer"] or "") for row in rows]
    metrics = dict(evaluation["summary"])
    metrics.pop("system", None)
    metrics.update(corpus_generation_metrics(pred_texts, gold_texts, include_bertscore=include_bertscore))

    ocr_cer = ocr_character_error_rate(predictions_by_id, clean_questions_by_id)
    if ocr_cer is not None:
        metrics["ocr_character_error_rate"] = ocr_cer

    return {
        "experiment_name": experiment_name,
        "ocr_engine": ocr_engine,
        "preprocessing": preprocessing,
        "generation_model": generation_model,
        "retrieval": retrieval,
        "metrics": metrics,
        "by_language": evaluation["by_language"],
        "rows": rows,
        "notes": notes,
    }


def corpus_generation_metrics(
    predictions: list[str],
    references: list[str],
    include_bertscore: bool = False,
) -> dict[str, Any]:
    """Compute corpus-level BLEU/ROUGE-L/METEOR and optional BERTScore."""

    metrics: dict[str, Any] = {
        "bleu": corpus_bleu(predictions, references),
        "rouge_l": rouge_l(predictions, references),
    }
    meteor = meteor_score(predictions, references)
    if meteor is not None:
        metrics["meteor"] = meteor
    else:
        metrics["meteor"] = None

    if include_bertscore:
        metrics["bertscore_f1"] = bertscore_f1(predictions, references)
    return metrics


def corpus_bleu(predictions: list[str], references: list[str]) -> float:
    """Return sacreBLEU score normalized to 0..1."""

    if not predictions:
        return 0.0
    try:
        import sacrebleu

        return round(float(sacrebleu.corpus_bleu(predictions, [references]).score) / 100.0, 6)
    except Exception:
        return 0.0


def rouge_l(predictions: list[str], references: list[str]) -> float:
    """Return average ROUGE-L F1."""

    if not predictions:
        return 0.0
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = [
            scorer.score(str(reference), str(prediction))["rougeL"].fmeasure
            for prediction, reference in zip(predictions, references, strict=False)
        ]
        return round(sum(scores) / len(scores), 6) if scores else 0.0
    except Exception:
        scores = [
            _lcs_f1(normalize_answer(prediction), normalize_answer(reference))
            for prediction, reference in zip(predictions, references, strict=False)
        ]
        return round(sum(scores) / len(scores), 6) if scores else 0.0


def meteor_score(predictions: list[str], references: list[str]) -> float | None:
    """Return average METEOR if NLTK resources are available."""

    if not predictions:
        return 0.0
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor_score

        scores = [
            float(nltk_meteor_score([normalize_answer(reference).split()], normalize_answer(prediction).split()))
            for prediction, reference in zip(predictions, references, strict=False)
        ]
        return round(sum(scores) / len(scores), 6) if scores else 0.0
    except Exception:
        return None


def bertscore_f1(predictions: list[str], references: list[str]) -> float | None:
    """Return average BERTScore F1 if bert-score and model weights are available."""

    if not predictions:
        return 0.0
    try:
        from bert_score import score

        _, _, f1 = score(predictions, references, lang="multilingual", verbose=False)
        return round(float(f1.mean().item()), 6)
    except Exception:
        return None


def ocr_character_error_rate(
    predictions_by_id: dict[str, dict[str, Any]],
    clean_questions_by_id: dict[str, str],
) -> float | None:
    """Compute OCR CER from prediction debug text when clean questions are available."""

    distances = 0
    lengths = 0
    for question_id, clean_question in clean_questions_by_id.items():
        if not clean_question:
            continue
        prediction = predictions_by_id.get(question_id)
        if not prediction:
            continue
        debug = prediction.get("debug") or {}
        ocr_question = str(debug.get("clean_question") or debug.get("ocr_text") or "")
        if not ocr_question:
            continue
        normalized_ocr = normalize_answer(ocr_question)
        normalized_clean = normalize_answer(clean_question)
        distances += levenshtein_distance(normalized_ocr, normalized_clean)
        lengths += len(normalized_clean)
    if lengths == 0:
        return None
    return round(distances / lengths, 6)


def gold_answers_from_records(records: list[dict[str, Any]]) -> dict[str, str]:
    """Extract question_id -> gold answer from JSON records."""

    gold_by_id = {}
    for record in records:
        question_id = str(record.get("question_id") or record.get("id") or "")
        if not question_id:
            continue
        if isinstance(record.get("answers"), list):
            answer = " | ".join(str(answer) for answer in record["answers"])
        else:
            answer = str(record.get("gold_answer") or record.get("answer") or record.get("target") or "")
        gold_by_id[question_id] = answer.strip()
    return gold_by_id


def clean_questions_from_records(records: list[dict[str, Any]]) -> dict[str, str]:
    """Extract question_id -> clean question text when present."""

    clean_by_id = {}
    for record in records:
        question_id = str(record.get("question_id") or record.get("id") or "")
        if not question_id:
            continue
        clean_question = (
            record.get("clean_question")
            or record.get("question")
            or record.get("question_text")
            or record.get("text")
            or ""
        )
        if clean_question:
            clean_by_id[question_id] = str(clean_question)
    return clean_by_id


def load_gold_records(path: str | Path) -> list[dict[str, Any]]:
    """Load gold records from JSON list/dict or JSONL."""

    gold_path = Path(path)
    if gold_path.suffix == ".jsonl":
        return read_jsonl(gold_path)
    with gold_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("gold", "references", "data", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported gold file shape: {gold_path}")


def levenshtein_distance(left: str, right: str) -> int:
    """Small dynamic-programming edit distance for CER."""

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _lcs_f1(predicted: str, gold: str) -> float:
    """Token-level LCS F1 fallback for ROUGE-L."""

    predicted_tokens = predicted.split()
    gold_tokens = gold.split()
    if not predicted_tokens or not gold_tokens:
        return 0.0

    previous = [0] * (len(gold_tokens) + 1)
    for predicted_token in predicted_tokens:
        current = [0]
        for index, gold_token in enumerate(gold_tokens, start=1):
            if predicted_token == gold_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current

    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(predicted_tokens)
    recall = lcs / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


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
