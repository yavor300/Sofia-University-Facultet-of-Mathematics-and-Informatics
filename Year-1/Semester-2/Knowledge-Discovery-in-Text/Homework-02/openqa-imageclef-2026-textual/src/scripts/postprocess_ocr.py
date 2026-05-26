"""Postprocess OCR JSONL outputs into cleaned question text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.ocr_postprocess import clean_ocr_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input OCR JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Output cleaned JSONL path.")
    parser.add_argument(
        "--text-field",
        default="ocr_text",
        help="Input field containing OCR text.",
    )
    parser.add_argument(
        "--output-field",
        default="clean_question",
        help="Output field for cleaned OCR question.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row[args.output_field] = clean_ocr_question(
                row.get(args.text_field, ""),
                language=row.get("language"),
            )
            rows.append(row)

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} cleaned OCR rows to {args.output}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

