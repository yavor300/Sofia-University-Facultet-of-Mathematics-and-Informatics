"""Create a submission ZIP containing exactly one JSON file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.submission import create_submission_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("data/submissions/predictions.json"),
        help="Final submission JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/submissions/openqa_textual_submission.zip"),
        help="Output ZIP path.",
    )
    parser.add_argument(
        "--arcname",
        default="predictions.json",
        help="JSON filename inside the ZIP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.submission.exists():
        raise SystemExit(f"Submission file does not exist: {args.submission}")
    output = create_submission_zip(args.submission, args.output, arcname=args.arcname)
    print(f"Wrote submission ZIP to {output}")
    with zipfile.ZipFile(output) as archive:
        print("ZIP contents:")
        for name in archive.namelist():
            print(f"  {name}")


if __name__ == "__main__":
    main()
