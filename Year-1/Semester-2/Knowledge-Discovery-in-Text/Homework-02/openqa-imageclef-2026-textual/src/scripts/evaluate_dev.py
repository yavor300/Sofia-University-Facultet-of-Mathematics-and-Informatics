"""Evaluate one dev prediction file against a gold/reference file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.evaluation import (
    evaluate_dev_report,
    load_gold_records,
    load_prediction_file,
)
from openqa_textual.prediction import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="Prediction JSON or JSONL path.")
    parser.add_argument("--gold", type=Path, required=True, help="Gold/reference JSON or JSONL path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/dev_metrics.json"),
        help="Output evaluation report path.",
    )
    parser.add_argument("--experiment-name", default="experiment")
    parser.add_argument("--ocr-engine", default="")
    parser.add_argument("--preprocessing", default="")
    parser.add_argument("--generation-model", default="")
    parser.add_argument("--retrieval", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="Also compute BERTScore F1 if bert-score can load its model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pred.exists():
        raise SystemExit(f"Prediction file does not exist: {args.pred}")
    if not args.gold.exists():
        raise SystemExit(f"Gold file does not exist: {args.gold}")

    predictions = load_prediction_file(args.pred)
    gold_records = load_gold_records(args.gold)
    report = evaluate_dev_report(
        predictions=predictions,
        gold_records=gold_records,
        experiment_name=args.experiment_name,
        ocr_engine=args.ocr_engine,
        preprocessing=args.preprocessing,
        generation_model=args.generation_model,
        retrieval=args.retrieval,
        notes=args.notes,
        include_bertscore=args.bertscore,
    )
    write_json(args.output, report)
    _print_metrics(report)
    print(f"Wrote evaluation report to {args.output}")


def _print_metrics(report: dict) -> None:
    metrics = report["metrics"]
    print("metric\tvalue")
    for key in (
        "exact_match",
        "normalized_exact_match",
        "token_f1",
        "bleu",
        "rouge_l",
        "meteor",
        "bertscore_f1",
        "ocr_character_error_rate",
    ):
        if key in metrics:
            print(f"{key}\t{metrics[key]}")


if __name__ == "__main__":
    main()
