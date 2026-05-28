"""Append one row to experiments/experiment_log.md."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.experiment_log import append_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("experiments/experiment_log.md"))
    parser.add_argument("--id", required=True, help="Experiment ID, for example E08.")
    parser.add_argument("--ocr", required=True)
    parser.add_argument("--preprocess", required=True)
    parser.add_argument("--ocr-correction", default="no")
    parser.add_argument("--retrieval", default="none")
    parser.add_argument("--llm", default="none")
    parser.add_argument("--fine-tuned", default="no")
    parser.add_argument("--dev-score", default="-")
    parser.add_argument("--notes", default="-")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = append_experiment(
        path=args.log,
        experiment_id=args.id,
        ocr=args.ocr,
        preprocess=args.preprocess,
        ocr_correction=args.ocr_correction,
        retrieval=args.retrieval,
        llm=args.llm,
        fine_tuned=args.fine_tuned,
        dev_score=args.dev_score,
        notes=args.notes,
    )
    print(f"Appended experiment to {args.log}")
    print(row)


if __name__ == "__main__":
    main()
