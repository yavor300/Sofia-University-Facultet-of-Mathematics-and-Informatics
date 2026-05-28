"""Create final submission JSON from internal prediction records."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.submission import make_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="Internal prediction JSON or JSONL path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/submissions/predictions.json"),
        help="Output final submission JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pred.exists():
        raise SystemExit(f"Prediction file does not exist: {args.pred}")
    submission = make_submission(args.pred, args.output)
    print(f"Wrote {len(submission)} submission records to {args.output}")


if __name__ == "__main__":
    main()
