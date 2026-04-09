from __future__ import annotations

from typing import Dict, List

import pandas as pd


def _as_set(labels: List[str]) -> set:
    return {str(label).lower().strip() for label in labels if str(label).strip()}


def evaluate_predictions(
    questions_df: pd.DataFrame,
    predictions: Dict[str, List[str]],
) -> dict:
    total = 0
    exact_matches = 0
    top1_matches = 0

    for _, row in questions_df.iterrows():
        qid = row["id"]
        gold = [str(label).lower() for label in row["gold_letters"]]
        pred = [str(label).lower() for label in predictions.get(qid, [])]
        if not gold:
            continue
        total += 1
        if _as_set(gold) == _as_set(pred):
            exact_matches += 1
        if pred and gold and pred[0] == gold[0]:
            top1_matches += 1

    if total == 0:
        return {"evaluated_questions": 0, "exact_match_accuracy": 0.0, "top1_accuracy": 0.0}

    return {
        "evaluated_questions": total,
        "exact_match_accuracy": exact_matches / total,
        "top1_accuracy": top1_matches / total,
    }

