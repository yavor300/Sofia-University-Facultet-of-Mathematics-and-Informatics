"""Compare OpenQA prediction files against gold answers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from openqa_textual.config import load_yaml_config
from openqa_textual.data import load_dataset_splits
from openqa_textual.evaluation import compare_systems, load_prediction_file
from openqa_textual.prediction import gold_answers_from_dataset_split, gold_answers_from_jsonl, write_json
from scripts.inspect_dataset import resolve_split_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Prediction system to evaluate. Repeat for each system.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/reports/evaluation_report.json"))
    parser.add_argument("--data-config", default="configs/data.yaml", help="Data config YAML.")
    parser.add_argument("--dataset-name", help="Override dataset name from data config.")
    parser.add_argument("--cache-dir", help="Override Hugging Face dataset cache directory.")
    parser.add_argument("--split", default="dev", help="Gold split or configured alias. Defaults to dev.")
    parser.add_argument("--gold-jsonl", type=Path, help="Optional gold answers JSONL.")
    parser.add_argument(
        "--train-gold-jsonl",
        type=Path,
        help="Optional train gold answers JSONL for overfit/copy indicators.",
    )
    parser.add_argument(
        "--no-dataset-gold",
        action="store_true",
        help="Do not load dataset gold answers. Requires --gold-jsonl for useful metrics.",
    )
    parser.add_argument(
        "--no-train-gold",
        action="store_true",
        help="Do not load train gold answers for overfit/copy indicators.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    systems = _load_systems(args.system)
    gold_by_id = {}
    train_gold_by_id = {}

    if args.gold_jsonl:
        gold_by_id.update(gold_answers_from_jsonl(args.gold_jsonl))
    elif not args.no_dataset_gold:
        gold_by_id.update(load_dataset_gold_answers(args, split=args.split))

    if args.train_gold_jsonl:
        train_gold_by_id.update(gold_answers_from_jsonl(args.train_gold_jsonl))
    elif not args.no_train_gold and not args.no_dataset_gold:
        train_gold_by_id.update(load_dataset_gold_answers(args, split="train"))

    if not gold_by_id:
        raise SystemExit("No gold answers loaded. Provide --gold-jsonl or enable dataset gold loading.")

    report = compare_systems(systems, gold_by_id=gold_by_id, train_gold_by_id=train_gold_by_id)
    write_json(args.output, report)
    _print_ranking(report)
    print(f"Wrote evaluation report to {args.output}")


def load_dataset_gold_answers(args: argparse.Namespace, split: str) -> dict[str, str]:
    data_config = load_yaml_config(args.data_config)
    dataset_config = data_config.get("dataset", {})
    dataset_name = args.dataset_name or dataset_config.get("name")
    cache_dir = args.cache_dir or dataset_config.get("cache_dir")
    split_aliases = dataset_config.get("splits", {})

    if not dataset_name:
        raise SystemExit("Dataset name is required via --dataset-name or configs/data.yaml.")
    dataset = load_dataset_splits(dataset_name, cache_dir=cache_dir)
    split_name = resolve_split_name(split, dataset, split_aliases)
    return gold_answers_from_dataset_split(dataset[split_name])


def _load_systems(system_args: list[str]) -> dict[str, list[dict]]:
    systems = {}
    for item in system_args:
        if "=" not in item:
            raise SystemExit(f"Expected --system NAME=PATH, got: {item}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not path.exists():
            raise SystemExit(f"Prediction file does not exist for system '{name}': {path}")
        systems[name] = load_prediction_file(path)
    return systems


def _print_ranking(report: dict) -> None:
    print("system\tn\tEM\tnorm_EM\ttoken_F1\tchar_sim\tnon_empty\ttrain_copy\trepeated")
    for row in report["ranking"]:
        print(
            "\t".join(
                [
                    str(row["system"]),
                    str(row["total"]),
                    f"{row['exact_match']:.4f}",
                    f"{row['normalized_exact_match']:.4f}",
                    f"{row['token_f1']:.4f}",
                    f"{row['char_similarity']:.4f}",
                    f"{row['non_empty_rate']:.4f}",
                    f"{row['train_answer_copy_rate']:.4f}",
                    f"{row['repeated_answer_rate']:.4f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
