from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import (
    labels_to_answer_string,
    load_questions,
    load_questions_jsonl,
    save_questions_jsonl,
)
from .evaluation import evaluate_predictions
from .modeling import (
    LexicalOverlapRanker,
    OptionPairClassifier,
    TransformerOptionPairClassifier,
)


def _resolve_bbf_token(use_bbf_token: bool) -> bool | None:
    return True if use_bbf_token else None


def _load_for_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    input_path = getattr(args, "input", None)
    if input_path and input_path.endswith(".jsonl"):
        return load_questions_jsonl(input_path)
    return load_questions(
        input_path=input_path,
        cache_dir=args.cache_dir,
        english_only=True,
        source=args.source,
        bbf_language=args.bbf_language,
        bbf_split=args.bbf_split,
        bbf_question_type=args.bbf_question_type,
        bbf_token=_resolve_bbf_token(args.bbf_use_token),
    )


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        type=str,
        default="finmmeval",
        choices=["finmmeval", "bbf", "both"],
        help="Data source: FinMMEval, BBF, or both.",
    )
    parser.add_argument(
        "--bbf-language",
        type=str,
        default="English",
        help="BhashaBench-Finance language config (for example: English).",
    )
    parser.add_argument(
        "--bbf-split",
        type=str,
        default="test",
        help="BhashaBench-Finance split (for example: test).",
    )
    parser.add_argument(
        "--bbf-question-type",
        type=str,
        default="MCQ",
        help="Only BBF rows with matching question_type are kept (default: MCQ).",
    )
    parser.add_argument(
        "--bbf-use-token",
        action="store_true",
        help="Pass token=True to datasets.load_dataset for gated BBF access.",
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-type",
        type=str,
        default="linear",
        choices=["linear", "transformer"],
        help="Model used for supervised prediction.",
    )
    parser.add_argument("--max-features", type=int, default=40000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument(
        "--transformer-model-name",
        type=str,
        default="distilbert-base-uncased",
        help="HF model name for transformer cross-encoder.",
    )
    parser.add_argument("--transformer-epochs", type=int, default=2)
    parser.add_argument("--transformer-batch-size", type=int, default=16)
    parser.add_argument("--transformer-lr", type=float, default=2e-5)
    parser.add_argument("--transformer-max-length", type=int, default=256)
    parser.add_argument("--transformer-weight-decay", type=float, default=0.01)
    parser.add_argument("--transformer-grad-clip", type=float, default=1.0)


def _build_linear_model(args: argparse.Namespace) -> OptionPairClassifier:
    return OptionPairClassifier(
        max_features=args.max_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        c_value=args.c_value,
    )


def _build_transformer_model(args: argparse.Namespace) -> TransformerOptionPairClassifier:
    return TransformerOptionPairClassifier(
        model_name=args.transformer_model_name,
        max_length=args.transformer_max_length,
        learning_rate=args.transformer_lr,
        num_train_epochs=args.transformer_epochs,
        batch_size=args.transformer_batch_size,
        weight_decay=args.transformer_weight_decay,
        grad_clip_norm=args.transformer_grad_clip,
        seed=args.seed,
    )


def _resolve_model_path(model_type: str, model_path: str) -> str:
    if model_type != "transformer":
        return model_path
    path = Path(model_path)
    if path.suffix:
        path = path.with_suffix("")
    return str(path)


def _make_train_dev_split(
    questions_df: pd.DataFrame,
    dev_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    eligible = questions_df[questions_df["gold_letters"].map(len) > 0].copy()
    if len(eligible) < 5:
        raise ValueError("Not enough labeled rows to create train/dev split.")

    stratify_labels = eligible["gold_letters"].map(lambda x: x[0] if x else "none")
    try:
        train_ids, dev_ids = train_test_split(
            eligible["id"],
            test_size=dev_size,
            random_state=seed,
            stratify=stratify_labels,
        )
    except ValueError:
        train_ids, dev_ids = train_test_split(
            eligible["id"],
            test_size=dev_size,
            random_state=seed,
            shuffle=True,
        )

    train_df = questions_df[questions_df["id"].isin(train_ids)].reset_index(drop=True)
    dev_df = questions_df[questions_df["id"].isin(dev_ids)].reset_index(drop=True)
    return train_df, dev_df


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)


def cmd_prepare(args: argparse.Namespace) -> None:
    questions_df = _load_for_pipeline(args)
    save_questions_jsonl(questions_df, args.output)
    source_counts = questions_df["dataset_source"].value_counts().to_dict()
    print(f"Saved {len(questions_df)} English questions to: {args.output}")
    print(f"Source distribution: {source_counts}")


def cmd_train(args: argparse.Namespace) -> None:
    questions_df = _load_for_pipeline(args)
    train_df, dev_df = _make_train_dev_split(
        questions_df,
        dev_size=args.dev_size,
        seed=args.seed,
    )

    linear_model = _build_linear_model(args)
    linear_model.fit(train_df)
    linear_dev_preds = linear_model.predict(dev_df)
    lexical_dev_preds = LexicalOverlapRanker().predict(dev_df)

    selected_model_name = "tfidf_logreg"
    selected_dev_preds = linear_dev_preds
    model_out_resolved = _resolve_model_path(args.model_type, args.model_out)

    if args.model_type == "transformer":
        model = _build_transformer_model(args)
        model.fit(train_df)
        model.save(model_out_resolved)
        selected_model_name = "transformer_cross_encoder"
        selected_dev_preds = model.predict(dev_df)
    else:
        linear_model.save(model_out_resolved)

    metrics = {
        "train_size": len(train_df),
        "dev_size": len(dev_df),
        "selected_model": selected_model_name,
        "supervised": evaluate_predictions(dev_df, selected_dev_preds),
        "tfidf_logreg": evaluate_predictions(dev_df, linear_dev_preds),
        "lexical_baseline": evaluate_predictions(dev_df, lexical_dev_preds),
    }
    if args.model_type == "transformer":
        metrics["transformer_cross_encoder"] = evaluate_predictions(dev_df, selected_dev_preds)

    _write_json(args.metrics_out, metrics)

    split_payload = {
        "seed": args.seed,
        "dev_size": args.dev_size,
        "train_ids": train_df["id"].tolist(),
        "dev_ids": dev_df["id"].tolist(),
    }
    _write_json(args.split_out, split_payload)

    print(f"Model saved to: {model_out_resolved}")
    print(f"Metrics saved to: {args.metrics_out}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    questions_df = _load_for_pipeline(args)
    train_df, dev_df = _make_train_dev_split(
        questions_df,
        dev_size=args.dev_size,
        seed=args.seed,
    )

    linear_model = _build_linear_model(args)
    linear_model.fit(train_df)
    linear_dev_preds = linear_model.predict(dev_df)
    lexical_dev_preds = LexicalOverlapRanker().predict(dev_df)

    model_path = _resolve_model_path(args.model_type, args.model)
    if args.model_type == "transformer":
        if args.model and Path(model_path).exists():
            model = TransformerOptionPairClassifier.load(model_path)
        else:
            model = _build_transformer_model(args)
            model.fit(train_df)
        selected_model_name = "transformer_cross_encoder"
        selected_dev_preds = model.predict(dev_df)
    else:
        selected_model_name = "tfidf_logreg"
        selected_dev_preds = linear_dev_preds

    metrics = {
        "train_size": len(train_df),
        "dev_size": len(dev_df),
        "selected_model": selected_model_name,
        "supervised": evaluate_predictions(dev_df, selected_dev_preds),
        "tfidf_logreg": evaluate_predictions(dev_df, linear_dev_preds),
        "lexical_baseline": evaluate_predictions(dev_df, lexical_dev_preds),
    }
    if args.model_type == "transformer":
        metrics["transformer_cross_encoder"] = evaluate_predictions(dev_df, selected_dev_preds)
    _write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    questions_df = _load_for_pipeline(args)
    model_path = _resolve_model_path(args.model_type, args.model)
    if args.model_type == "transformer":
        model = TransformerOptionPairClassifier.load(model_path)
    else:
        model = OptionPairClassifier.load(model_path)
    predictions = model.predict(questions_df)

    rows = []
    for _, row in questions_df.iterrows():
        labels = predictions.get(row["id"], [])
        rows.append(
            {
                "id": row["id"],
                "answer": labels_to_answer_string(labels),
            }
        )
    submission_df = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")

    if questions_df["gold_letters"].map(len).sum() > 0:
        metrics = evaluate_predictions(questions_df, predictions)
        print("Local labeled-data score (for sanity check):")
        print(json.dumps(metrics, indent=2))


def cmd_run_all(args: argparse.Namespace) -> None:
    prepare_args = argparse.Namespace(
        input=None,
        cache_dir=args.cache_dir,
        output=args.prepared,
        source=args.source,
        bbf_language=args.bbf_language,
        bbf_split=args.bbf_split,
        bbf_question_type=args.bbf_question_type,
        bbf_use_token=args.bbf_use_token,
    )
    cmd_prepare(prepare_args)

    train_args = argparse.Namespace(
        input=args.prepared,
        cache_dir=args.cache_dir,
        source=args.source,
        bbf_language=args.bbf_language,
        bbf_split=args.bbf_split,
        bbf_question_type=args.bbf_question_type,
        bbf_use_token=args.bbf_use_token,
        dev_size=args.dev_size,
        seed=args.seed,
        max_features=args.max_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        c_value=args.c_value,
        model_type=args.model_type,
        transformer_model_name=args.transformer_model_name,
        transformer_epochs=args.transformer_epochs,
        transformer_batch_size=args.transformer_batch_size,
        transformer_lr=args.transformer_lr,
        transformer_max_length=args.transformer_max_length,
        transformer_weight_decay=args.transformer_weight_decay,
        transformer_grad_clip=args.transformer_grad_clip,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        split_out=args.split_out,
    )
    cmd_train(train_args)

    predict_args = argparse.Namespace(
        input=args.prepared,
        cache_dir=args.cache_dir,
        source=args.source,
        bbf_language=args.bbf_language,
        bbf_split=args.bbf_split,
        bbf_question_type=args.bbf_question_type,
        bbf_use_token=args.bbf_use_token,
        model_type=args.model_type,
        model=args.model_out,
        output=args.submission_out,
    )
    cmd_predict(predict_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FinMMEval Homework-01 pipeline (English-only)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Download/read data and save normalized JSONL.")
    prepare.add_argument("--input", type=str, default=None, help="Input file (parquet/csv/jsonl).")
    prepare.add_argument("--cache-dir", type=str, default="data/raw/hf_cache")
    prepare.add_argument("--output", type=str, default="data/processed/english_questions.jsonl")
    _add_source_args(prepare)
    prepare.set_defaults(func=cmd_prepare)

    train = sub.add_parser("train", help="Train the supervised baseline.")
    train.add_argument("--input", type=str, default="data/processed/english_questions.jsonl")
    train.add_argument("--cache-dir", type=str, default="data/raw/hf_cache")
    _add_source_args(train)
    train.add_argument("--dev-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    _add_model_args(train)
    train.add_argument("--model-out", type=str, default="models/option_pair_classifier.joblib")
    train.add_argument("--metrics-out", type=str, default="results/dev_metrics.json")
    train.add_argument("--split-out", type=str, default="results/split.json")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="Evaluate lexical and supervised models.")
    evaluate.add_argument("--input", type=str, default="data/processed/english_questions.jsonl")
    evaluate.add_argument("--cache-dir", type=str, default="data/raw/hf_cache")
    _add_source_args(evaluate)
    evaluate.add_argument("--dev-size", type=float, default=0.2)
    evaluate.add_argument("--seed", type=int, default=42)
    _add_model_args(evaluate)
    evaluate.add_argument("--model", type=str, default="models/option_pair_classifier.joblib")
    evaluate.add_argument("--output", type=str, default="results/evaluation.json")
    evaluate.set_defaults(func=cmd_evaluate)

    predict = sub.add_parser("predict", help="Generate submission CSV.")
    predict.add_argument("--input", type=str, default="data/processed/english_questions.jsonl")
    predict.add_argument("--cache-dir", type=str, default="data/raw/hf_cache")
    _add_source_args(predict)
    predict.add_argument(
        "--model-type",
        type=str,
        default="linear",
        choices=["linear", "transformer"],
        help="Model used for prediction loading.",
    )
    predict.add_argument("--model", type=str, default="models/option_pair_classifier.joblib")
    predict.add_argument("--output", type=str, default="results/submission.csv")
    predict.set_defaults(func=cmd_predict)

    run_all = sub.add_parser("run-all", help="Run prepare -> train -> predict.")
    run_all.add_argument("--cache-dir", type=str, default="data/raw/hf_cache")
    run_all.add_argument("--prepared", type=str, default="data/processed/english_questions.jsonl")
    _add_source_args(run_all)
    run_all.add_argument("--dev-size", type=float, default=0.2)
    run_all.add_argument("--seed", type=int, default=42)
    _add_model_args(run_all)
    run_all.add_argument("--model-out", type=str, default="models/option_pair_classifier.joblib")
    run_all.add_argument("--metrics-out", type=str, default="results/dev_metrics.json")
    run_all.add_argument("--split-out", type=str, default="results/split.json")
    run_all.add_argument("--submission-out", type=str, default="results/submission.csv")
    run_all.set_defaults(func=cmd_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
