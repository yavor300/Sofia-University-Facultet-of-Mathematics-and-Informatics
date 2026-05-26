"""Generate Baseline 1 heuristic QA predictions from OCR JSONL rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.prediction import (
    build_heuristic_predictions_from_ocr_rows,
    read_jsonl,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-jsonl", type=Path, required=True, help="Input OCR JSONL path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/heuristic_predictions.json"),
        help="Output prediction path.",
    )
    parser.add_argument(
        "--text-field",
        default="clean_question",
        help="OCR field to answer from. Falls back to ocr_text.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Write JSONL rows instead of one JSON list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ocr_jsonl.exists():
        raise SystemExit(f"OCR JSONL does not exist: {args.ocr_jsonl}")

    rows = read_jsonl(args.ocr_jsonl)
    predictions = build_heuristic_predictions_from_ocr_rows(rows, text_field=args.text_field)
    if args.jsonl:
        write_jsonl(args.output, predictions)
    else:
        write_json(args.output, predictions)

    answered = sum(1 for prediction in predictions if prediction["answers"] != [""])
    print(f"Wrote {len(predictions)} heuristic predictions to {args.output}")
    print(f"Non-empty answers: {answered}")


if __name__ == "__main__":
    main()

