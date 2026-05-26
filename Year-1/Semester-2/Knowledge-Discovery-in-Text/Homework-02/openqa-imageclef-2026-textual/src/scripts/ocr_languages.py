"""Show OCR language mappings for configured engines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.config import load_yaml_config
from openqa_textual.ocr import normalize_language, ocr_language_for_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ocr.yaml", help="Path to OCR config YAML.")
    parser.add_argument(
        "--language",
        action="append",
        help="Language label/code to inspect. Can be used multiple times.",
    )
    parser.add_argument(
        "--engine",
        action="append",
        choices=["easyocr", "tesseract", "paddleocr"],
        help="Engine to inspect. Can be used multiple times.",
    )
    parser.add_argument(
        "--check-tesseract",
        action="store_true",
        help="Also print Tesseract languages installed on this machine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    languages = args.language or ["English", "Bulgarian", "German", "French", "Spanish", None]
    engines = args.engine or ["easyocr", "tesseract", "paddleocr"]

    rows: list[dict[str, Any]] = []
    for language in languages:
        for engine in engines:
            rows.append(
                {
                    "input_language": language,
                    "normalized_language": normalize_language(language),
                    "engine": engine,
                    "ocr_language": ocr_language_for_engine(engine, language, config),
                }
            )

    print(json.dumps(rows, ensure_ascii=False, indent=2))

    if args.check_tesseract:
        print("\nTesseract installed languages:")
        try:
            result = subprocess.run(
                ["tesseract", "--list-langs"],
                check=False,
                text=True,
                capture_output=True,
            )
            print((result.stdout or result.stderr).strip())
        except FileNotFoundError:
            print("tesseract executable was not found on PATH")


if __name__ == "__main__":
    main()
