"""Command line interface for GutBrainIE experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gutbrainie",
        description="GutBrainIE 2026 T611/T621 experiment runner.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="gutbrainie 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    prepare = subparsers.add_parser(
        "prepare-data",
        help="Load and validate GutBrainIE article and annotation files.",
    )
    prepare.add_argument("--data-root", default="data/gutbrainie2026")
    prepare.add_argument("--quality", default="gold", choices=["gold", "silver", "silver_2025", "bronze", "dev"])
    prepare.add_argument("--output", default="outputs/reports/data_validation_gold.json")
    prepare.set_defaults(handler=_prepare_data)

    eda = subparsers.add_parser("eda", help="Generate exploratory data analysis CSVs and plots.")
    eda.add_argument("--data-root", default="data/gutbrainie2026")
    eda.add_argument("--output-dir", default="outputs/reports")
    eda.add_argument("--qualities", nargs="+", default=["gold", "dev"], choices=["gold", "silver", "silver_2025", "bronze", "dev"])
    eda.set_defaults(handler=_eda)

    ner = subparsers.add_parser("run-ner-baseline", help="Run the dictionary/rule NER baseline.")
    ner.add_argument(
        "--train-entities",
        default="data/gutbrainie2026/Annotations/Train/gold_quality/csv_format/train_gold_entities.csv",
    )
    ner.add_argument("--articles", default="data/gutbrainie2026/Articles/csv_format/articles_dev.csv")
    ner.add_argument("--output", default="outputs/predictions/dev_t611_dictionary.json")
    ner.set_defaults(handler=_predict_ner_dictionary)

    ner_dict = subparsers.add_parser("predict-ner-dictionary", help="Predict T611 entities with a train-derived dictionary.")
    ner_dict.add_argument("--train-entities", required=True)
    ner_dict.add_argument("--articles", required=True)
    ner_dict.add_argument("--output", required=True)
    ner_dict.set_defaults(handler=_predict_ner_dictionary)

    ner_transformer = subparsers.add_parser("run-ner-transformer", help="Run the transformer NER model.")
    ner_transformer.add_argument("--config", default="configs/ner_gliner.yaml")
    ner_transformer.add_argument("--model", default=None)
    ner_transformer.add_argument("--articles", default="data/gutbrainie2026/Articles/csv_format/articles_dev.csv")
    ner_transformer.add_argument("--output", default="outputs/predictions/dev_t611_gliner_gold.json")
    ner_transformer.set_defaults(handler=_predict_gliner)

    gliner_prepare = subparsers.add_parser("prepare-gliner-data", help="Create GLiNER train/validation JSONL files.")
    gliner_prepare.add_argument("--data-root", default="data/gutbrainie2026")
    gliner_prepare.add_argument(
        "--experiment",
        default="gold",
        choices=["gold", "gold_silver", "gold_silver_silver_2025"],
    )
    gliner_prepare.add_argument("--output-dir", default="outputs/gliner")
    gliner_prepare.add_argument("--validation-fraction", type=float, default=0.15)
    gliner_prepare.add_argument("--seed", type=int, default=13)
    gliner_prepare.set_defaults(handler=_prepare_gliner_data)

    gliner_train = subparsers.add_parser("train-gliner", help="Fine-tune a GLiNER model from prepared JSONL data.")
    gliner_train.add_argument("--config", default="configs/ner_gliner.yaml")
    gliner_train.add_argument("--model", default=None)
    gliner_train.add_argument("--train-data", required=True)
    gliner_train.add_argument("--validation-data", required=True)
    gliner_train.add_argument("--output-dir", default="outputs/models/gliner")
    gliner_train.set_defaults(handler=_train_gliner)

    gliner_predict = subparsers.add_parser("predict-gliner", help="Predict T611 entities with GLiNER.")
    gliner_predict.add_argument("--config", default="configs/ner_gliner.yaml")
    gliner_predict.add_argument("--model", default=None)
    gliner_predict.add_argument("--articles", required=True)
    gliner_predict.add_argument("--output", required=True)
    gliner_predict.set_defaults(handler=_predict_gliner)

    token_train = subparsers.add_parser("train-token-classifier", help="Fine-tune a PubMedBERT/BioBERT/SciBERT token classifier.")
    token_train.add_argument("--config", default="configs/ner_transformer.yaml")
    token_train.add_argument("--data-root", default="data/gutbrainie2026")
    token_train.add_argument(
        "--experiment",
        default="gold",
        choices=["gold", "gold_silver", "gold_silver_silver_2025"],
    )
    token_train.add_argument("--model", default=None)
    token_train.add_argument("--output-dir", default="outputs/models/token_classifier_gold")
    token_train.add_argument("--validation-fraction", type=float, default=0.15)
    token_train.add_argument("--seed", type=int, default=13)
    token_train.set_defaults(handler=_train_token_classifier)

    token_predict = subparsers.add_parser("predict-token-classifier", help="Predict T611 entities with a trained token classifier.")
    token_predict.add_argument("--model", required=True, help="Trained model directory.")
    token_predict.add_argument("--articles", default="data/gutbrainie2026/Articles/csv_format/articles_dev.csv")
    token_predict.add_argument("--output", default="outputs/predictions/dev_t611_token_classifier.json")
    token_predict.add_argument("--batch-size", type=int, default=8)
    token_predict.add_argument("--max-length", type=int, default=512)
    token_predict.add_argument("--use-cpu", action="store_true")
    token_predict.set_defaults(handler=_predict_token_classifier)

    re_base = subparsers.add_parser("run-re-baseline", help="Run the mention-level relation baseline.")
    re_base.add_argument("--config", default="configs/paths.yaml")
    re_base.add_argument("--articles", default="data/gutbrainie2026/Articles/csv_format/articles_dev.csv")
    re_base.add_argument("--entities", default="data/gutbrainie2026/Annotations/Dev/csv_format/dev_entities.csv")
    re_base.add_argument(
        "--train-relations",
        default="data/gutbrainie2026/Annotations/Train/gold_quality/csv_format/train_gold_mention_level_relations.csv",
    )
    re_base.add_argument("--output", default="outputs/predictions/dev_t621_rule.json")
    re_base.add_argument("--threshold", type=float, default=0.5)
    re_base.add_argument("--max-distance", type=int)
    re_base.set_defaults(handler=_predict_re_rule)

    re_rule = subparsers.add_parser("predict-re-rule", help="Predict T621 mention-level relations with the prior baseline.")
    re_rule.add_argument("--articles", required=True)
    re_rule.add_argument("--entities", required=True, help="Gold/predicted entities as CSV or T611 JSON.")
    re_rule.add_argument("--train-relations", required=True)
    re_rule.add_argument("--output", required=True)
    re_rule.add_argument("--threshold", type=float, default=0.5)
    re_rule.add_argument("--max-distance", type=int)
    re_rule.set_defaults(handler=_predict_re_rule)

    re_transformer = subparsers.add_parser("run-re-transformer", help="Run the transformer relation model.")
    re_transformer.add_argument("--config", default="configs/re_transformer.yaml")
    re_transformer.set_defaults(handler=_placeholder_handler)

    re_pair_train = subparsers.add_parser("train-re-pair-classifier", help="Fine-tune a mention-level RE pair classifier.")
    re_pair_train.add_argument("--config", default="configs/re_transformer.yaml")
    re_pair_train.add_argument("--data-root", default="data/gutbrainie2026")
    re_pair_train.add_argument(
        "--experiment",
        default="gold",
        choices=["gold", "gold_silver", "gold_silver_silver_2025"],
    )
    re_pair_train.add_argument("--model", default=None)
    re_pair_train.add_argument("--output-dir", default="outputs/models/re_pair_classifier_gold")
    re_pair_train.add_argument("--validation-fraction", type=float, default=0.15)
    re_pair_train.add_argument("--seed", type=int, default=13)
    re_pair_train.set_defaults(handler=_train_re_pair_classifier)

    re_pair_predict = subparsers.add_parser("predict-re-pair-classifier", help="Predict T621 relations with a trained RE pair classifier.")
    re_pair_predict.add_argument("--model", required=True)
    re_pair_predict.add_argument("--articles", default="data/gutbrainie2026/Articles/csv_format/articles_dev.csv")
    re_pair_predict.add_argument("--entities", required=True, help="Gold/predicted entities as CSV or T611 JSON.")
    re_pair_predict.add_argument("--output", default="outputs/predictions/dev_t621_pair_classifier.json")
    re_pair_predict.add_argument("--threshold", type=float, default=0.5)
    re_pair_predict.add_argument("--batch-size", type=int, default=8)
    re_pair_predict.add_argument("--max-length", type=int, default=512)
    re_pair_predict.add_argument("--use-cpu", action="store_true")
    re_pair_predict.set_defaults(handler=_predict_re_pair_classifier)

    re_ollama = subparsers.add_parser("predict-re-ollama", help="Verify T621 candidates with a local Ollama LLM.")
    re_ollama.add_argument("--articles", default="data/gutbrainie2026/Articles/csv_format/articles_dev.csv")
    re_ollama.add_argument("--entities", required=True, help="Gold/predicted entities as CSV or T611 JSON.")
    re_ollama.add_argument("--model", default="llama3.1:8b")
    re_ollama.add_argument("--base-url", default="http://localhost:11434")
    re_ollama.add_argument("--output", default="outputs/predictions/dev_t621_ollama_llama31.json")
    re_ollama.add_argument("--threshold", type=float, default=0.5)
    re_ollama.add_argument("--max-candidates", type=int, default=200)
    re_ollama.add_argument("--max-distance", type=int, default=0)
    re_ollama.add_argument("--timeout", type=float, default=120)
    re_ollama.add_argument("--temperature", type=float, default=0.0)
    re_ollama.add_argument("--decisions-output", help="Optional JSONL audit log of LLM decisions.")
    re_ollama.set_defaults(handler=_predict_re_ollama)

    re_gpt = subparsers.add_parser("predict-re-gpt", help="Verify T621 candidates with OpenAI GPT.")
    re_gpt.add_argument("--articles", default="data/gutbrainie2026/Articles/csv_format/articles_dev.csv")
    re_gpt.add_argument("--entities", required=True, help="Gold/predicted entities as CSV or T611 JSON.")
    re_gpt.add_argument("--model", help="OpenAI model. Defaults to OPENAI_JUDGE_MODEL from .env, then gpt-4o-mini.")
    re_gpt.add_argument("--env-file", default=".env", help="Local dotenv file containing OPENAI_API_KEY and optional OPENAI_JUDGE_MODEL.")
    re_gpt.add_argument("--api-key-env", default="OPENAI_API_KEY")
    re_gpt.add_argument("--output", default="outputs/predictions/dev_t621_gpt.json")
    re_gpt.add_argument("--threshold", type=float, default=0.5)
    re_gpt.add_argument("--max-candidates", type=int, default=50)
    re_gpt.add_argument("--max-distance", type=int, default=0)
    re_gpt.add_argument("--timeout", type=float, default=120)
    re_gpt.add_argument("--temperature", type=float, default=0.0)
    re_gpt.add_argument("--decisions-output", help="Optional JSONL audit log of GPT decisions.")
    re_gpt.set_defaults(handler=_predict_re_gpt)

    pipeline = subparsers.add_parser("run-pipeline", help="Run end-to-end T611 NER plus T621 RE prediction.")
    pipeline.add_argument("--data-root", default="data/gutbrainie2026")
    pipeline.add_argument("--split", default="dev", choices=["dev", "test", "gold", "silver", "silver_2025", "bronze"])
    pipeline.add_argument("--articles", help="Optional article CSV override.")
    pipeline.add_argument("--ner-model", required=True)
    pipeline.add_argument("--re-model", required=True)
    pipeline.add_argument("--output-dir", default="outputs/predictions/pipeline_dev")
    pipeline.add_argument("--metrics-output", help="Optional combined metrics/config JSON output path.")
    pipeline.add_argument("--config-output", help="Optional pipeline config JSON output path.")
    pipeline.add_argument("--ner-backend", default="token-classifier", choices=["token-classifier", "gliner", "dictionary"])
    pipeline.add_argument("--re-backend", default="pair-classifier", choices=["pair-classifier", "rule"])
    pipeline.add_argument("--ner-config", default="configs/ner_gliner.yaml")
    pipeline.add_argument("--train-entities", help="Training entities CSV for dictionary NER.")
    pipeline.add_argument("--train-relations", help="Training mention-level relations CSV for rule RE.")
    pipeline.add_argument("--ner-batch-size", type=int, default=8)
    pipeline.add_argument("--ner-max-length", type=int, default=512)
    pipeline.add_argument("--re-threshold", type=float, default=0.5)
    pipeline.add_argument("--re-batch-size", type=int, default=8)
    pipeline.add_argument("--re-max-length", type=int, default=512)
    pipeline.add_argument("--use-cpu", action="store_true")
    pipeline.set_defaults(handler=_run_pipeline)

    atlop_notes = subparsers.add_parser("atlop-notes", help="Inspect official ATLOP setup and write reproduction notes.")
    atlop_notes.add_argument("--official-repo", default="external/GutBrainIE_2026_Baseline")
    atlop_notes.add_argument("--data-root", default="data/gutbrainie2026")
    atlop_notes.add_argument("--output", default="outputs/reports/atlop_notes.md")
    atlop_notes.set_defaults(handler=_atlop_notes)

    prepare_atlop = subparsers.add_parser("prepare-atlop-data", help="Convert local GutBrainIE data to official ATLOP JSON files.")
    prepare_atlop.add_argument("--data-root", default="data/gutbrainie2026")
    prepare_atlop.add_argument("--official-repo", default="external/GutBrainIE_2026_Baseline")
    prepare_atlop.add_argument("--predicted-entities", help="Optional dev T611 JSON/CSV to convert for ATLOP prediction.")
    prepare_atlop.set_defaults(handler=_prepare_atlop_data)

    run_atlop = subparsers.add_parser("run-atlop", help="Run one official ATLOP reproduction step.")
    run_atlop.add_argument("--official-repo", default="external/GutBrainIE_2026_Baseline")
    run_atlop.add_argument("--action", required=True, choices=["compose", "finetune", "predict"])
    run_atlop.add_argument("--output", default="outputs/predictions/atlop_predicted_relations_raw.json")
    run_atlop.add_argument("--log", help="Optional log path. Defaults to outputs/reports/atlop_<action>.log.")
    run_atlop.add_argument("--dry-run", action="store_true")
    run_atlop.set_defaults(handler=_run_atlop)

    evaluate = subparsers.add_parser("evaluate", help="Run internal exact-match evaluation.")
    evaluate.add_argument("--task", required=True, choices=["ner", "re"])
    evaluate.add_argument("--gold", required=True, help="Gold CSV path.")
    evaluate.add_argument("--prediction", required=True, help="Prediction CSV path.")
    evaluate.add_argument("--output", help="Optional JSON metrics output path.")
    evaluate.set_defaults(handler=_evaluate_internal)

    official = subparsers.add_parser("evaluate-official", help="Run a local official evaluation wrapper.")
    official.add_argument("--official-repo", default="external/GutBrainIE_2026_Baseline")
    official.add_argument("--prediction", required=True)
    official.add_argument("--entrypoint", help="Evaluation script path relative to the official repo.")
    official.add_argument("--gold", help="Optional gold file path passed to the official script.")
    official.add_argument("--task", choices=["T611", "T621", "ner", "re"], help="Optional task passed to the official script.")
    official.add_argument("official_args", nargs=argparse.REMAINDER, help="Extra args after -- are passed through.")
    official.set_defaults(handler=_evaluate_official)

    export_t611 = subparsers.add_parser("export-t611", help="Export NER predictions in T611 JSON format.")
    export_t611.add_argument("--output", default="outputs/submissions/t611_predictions.json")
    export_t611.set_defaults(handler=_placeholder_handler)

    export_t621 = subparsers.add_parser("export-t621", help="Export mention-level RE predictions in T621 JSON format.")
    export_t621.add_argument("--output", default="outputs/submissions/t621_predictions.json")
    export_t621.set_defaults(handler=_placeholder_handler)

    return parser


def _placeholder_handler(args: argparse.Namespace) -> int:
    command = args.command or "help"
    print(f"Command '{command}' is registered. Implementation follows in later phases.")
    return 0


def _prepare_data(args: argparse.Namespace) -> int:
    try:
        from gutbrainie.data.dataset import write_validation_report
    except ModuleNotFoundError as exc:
        if exc.name == "pandas":
            raise SystemExit("Missing dependency 'pandas'. Run: pip install -r requirements.txt") from exc
        raise

    report = write_validation_report(args.data_root, args.quality, args.output)
    print(
        "Validation report written to "
        f"{args.output}: articles={report['articles']}, "
        f"entities={report['entities']}, relations={report['relations']}, "
        f"offset_failures={report['offset_checks_failed']}, "
        f"missing_articles={report['missing_articles']}"
    )
    return 0


def _eda(args: argparse.Namespace) -> int:
    try:
        from gutbrainie.evaluation.report import generate_data_statistics
    except ModuleNotFoundError as exc:
        if exc.name in {"pandas", "matplotlib"}:
            raise SystemExit(f"Missing dependency '{exc.name}'. Run: pip install -r requirements.txt") from exc
        raise

    result = generate_data_statistics(args.data_root, args.output_dir, tuple(args.qualities))
    print(f"EDA reports written to {result['output_dir']}")
    for file_path in result["files"]:
        print(f"- {file_path}")
    return 0


def _evaluate_internal(args: argparse.Namespace) -> int:
    try:
        if args.task == "ner":
            from gutbrainie.data.annotations import load_entities_csv
            from gutbrainie.evaluation.ner_metrics import evaluate_ner
            from gutbrainie.submission.export_t611 import load_t611_json

            gold = load_entities_csv(args.gold)
            prediction = load_t611_json(args.prediction) if str(args.prediction).endswith(".json") else load_entities_csv(args.prediction)
            metrics = evaluate_ner(gold, prediction)
        else:
            from gutbrainie.data.annotations import load_mention_relations_csv
            from gutbrainie.evaluation.re_metrics import evaluate_mention_relations
            from gutbrainie.submission.export_t621 import load_t621_json

            gold = load_mention_relations_csv(args.gold)
            prediction = load_t621_json(args.prediction) if str(args.prediction).endswith(".json") else load_mention_relations_csv(args.prediction)
            metrics = evaluate_mention_relations(gold, prediction)
    except ModuleNotFoundError as exc:
        if exc.name == "pandas":
            raise SystemExit("Missing dependency 'pandas'. Run: pip install -r requirements.txt") from exc
        raise

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Internal {args.task} metrics written to {output_path}")
    else:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def _evaluate_official(args: argparse.Namespace) -> int:
    from gutbrainie.evaluation.official_eval_wrapper import run_official_evaluation

    extra_args = args.official_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    result = run_official_evaluation(
        official_repo=args.official_repo,
        prediction=args.prediction,
        entrypoint=args.entrypoint,
        gold=args.gold,
        task=args.task,
        extra_args=extra_args,
    )
    print("Command:", " ".join(result["command"]))
    if result["stdout"]:
        print(result["stdout"], end="")
    if result["stderr"]:
        print(result["stderr"], end="")
    return int(result["returncode"])


def _predict_ner_dictionary(args: argparse.Namespace) -> int:
    try:
        from gutbrainie.ner.dictionary_baseline import predict_dictionary_to_json
    except ModuleNotFoundError as exc:
        if exc.name == "pandas":
            raise SystemExit("Missing dependency 'pandas'. Run: pip install -r requirements.txt") from exc
        raise

    predictions = predict_dictionary_to_json(
        train_entities_path=args.train_entities,
        articles_path=args.articles,
        output_path=args.output,
    )
    print(f"Dictionary NER predictions written to {args.output}: entities={len(predictions)}")
    return 0


def _prepare_gliner_data(args: argparse.Namespace) -> int:
    from gutbrainie.ner.gliner_runner import prepare_gliner_experiment_data

    metadata = prepare_gliner_experiment_data(
        data_root=args.data_root,
        experiment=args.experiment,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(f"GLiNER data prepared for experiment '{args.experiment}'")
    print(f"- train: {metadata['train_path']} ({metadata['train_examples']} examples)")
    print(f"- validation: {metadata['validation_path']} ({metadata['validation_examples']} examples)")
    return 0


def _train_gliner(args: argparse.Namespace) -> int:
    from gutbrainie.config import load_yaml
    from gutbrainie.ner.gliner_runner import train_gliner_model

    config = load_yaml(args.config)
    model_name = args.model or config.get("model_name")
    if not model_name:
        raise SystemExit("Missing GLiNER model name. Pass --model or set model_name in the config.")
    try:
        output_dir = train_gliner_model(
            model_name=model_name,
            train_path=args.train_data,
            validation_path=args.validation_data,
            output_dir=args.output_dir,
            config_path=args.config,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"GLiNER model written to {output_dir}")
    return 0


def _predict_gliner(args: argparse.Namespace) -> int:
    from gutbrainie.config import load_yaml
    from gutbrainie.ner.gliner_runner import predict_gliner_to_json

    config = load_yaml(args.config)
    model_name = args.model or config.get("model_name")
    if not model_name:
        raise SystemExit("Missing GLiNER model name. Pass --model or set model_name in the config.")
    try:
        predictions = predict_gliner_to_json(
            model_name_or_path=model_name,
            articles_path=args.articles,
            output_path=args.output,
            labels=config.get("labels"),
            threshold=float(config.get("threshold", 0.5)),
            batch_size=int(config.get("batch_size", 8)),
            max_len=int(config["max_len"]) if config.get("max_len") else None,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"GLiNER predictions written to {args.output}: entities={len(predictions)}")
    return 0


def _train_token_classifier(args: argparse.Namespace) -> int:
    from gutbrainie.ner.train_token_classifier import train_token_classifier_experiment

    try:
        metadata = train_token_classifier_experiment(
            data_root=args.data_root,
            experiment=args.experiment,
            output_dir=args.output_dir,
            config_path=args.config,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            model_name=args.model,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Token classifier written to {metadata['output_dir']}")
    print(f"- train examples: {metadata['train_examples']}")
    print(f"- validation examples: {metadata['validation_examples']}")
    print(f"- labels: {metadata['label_count']}")
    return 0


def _predict_token_classifier(args: argparse.Namespace) -> int:
    from gutbrainie.ner.predict_token_classifier import predict_token_classifier_to_json

    try:
        predictions = predict_token_classifier_to_json(
            model_path=args.model,
            articles_path=args.articles,
            output_path=args.output,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_cpu=args.use_cpu,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Token classifier predictions written to {args.output}: entities={len(predictions)}")
    return 0


def _predict_re_rule(args: argparse.Namespace) -> int:
    from gutbrainie.re.rule_baseline import predict_re_rule_to_json

    try:
        predictions = predict_re_rule_to_json(
            articles_path=args.articles,
            entities_path=args.entities,
            train_relations_path=args.train_relations,
            output_path=args.output,
            threshold=args.threshold,
            max_distance=args.max_distance,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "pandas":
            raise SystemExit("Missing dependency 'pandas'. Run: pip install -r requirements.txt") from exc
        raise
    print(f"Rule RE predictions written to {args.output}: relations={len(predictions)}")
    return 0


def _train_re_pair_classifier(args: argparse.Namespace) -> int:
    from gutbrainie.re.train_pair_classifier import train_pair_classifier_experiment

    try:
        metadata = train_pair_classifier_experiment(
            data_root=args.data_root,
            experiment=args.experiment,
            output_dir=args.output_dir,
            config_path=args.config,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            model_name=args.model,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"RE pair classifier written to {metadata['output_dir']}")
    print(f"- train examples: {metadata['train_examples']}")
    print(f"- validation examples: {metadata['validation_examples']}")
    print(f"- positive train examples: {metadata['positive_train_examples']}")
    print(f"- labels: {metadata['label_count']}")
    return 0


def _predict_re_pair_classifier(args: argparse.Namespace) -> int:
    from gutbrainie.re.predict_pair_classifier import predict_pair_classifier_to_json

    try:
        predictions = predict_pair_classifier_to_json(
            model_path=args.model,
            articles_path=args.articles,
            entities_path=args.entities,
            output_path=args.output,
            threshold=args.threshold,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_cpu=args.use_cpu,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"RE pair classifier predictions written to {args.output}: relations={len(predictions)}")
    return 0


def _predict_re_ollama(args: argparse.Namespace) -> int:
    from gutbrainie.llm.ollama_relation_verifier import predict_re_ollama_to_json

    try:
        predictions = predict_re_ollama_to_json(
            articles_path=args.articles,
            entities_path=args.entities,
            output_path=args.output,
            model=args.model,
            base_url=args.base_url,
            threshold=args.threshold,
            max_candidates=args.max_candidates,
            max_distance=args.max_distance,
            timeout=args.timeout,
            temperature=args.temperature,
            decisions_output=args.decisions_output,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Ollama RE predictions written to {args.output}: relations={len(predictions)}")
    if args.decisions_output:
        print(f"- decisions: {args.decisions_output}")
    return 0


def _predict_re_gpt(args: argparse.Namespace) -> int:
    from gutbrainie.llm.gpt_relation_verifier import predict_re_gpt_to_json

    try:
        predictions = predict_re_gpt_to_json(
            articles_path=args.articles,
            entities_path=args.entities,
            output_path=args.output,
            model=args.model,
            env_path=args.env_file,
            api_key_env=args.api_key_env,
            threshold=args.threshold,
            max_candidates=args.max_candidates,
            max_distance=args.max_distance,
            timeout=args.timeout,
            temperature=args.temperature,
            decisions_output=args.decisions_output,
        )
    except (RuntimeError, ModuleNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"GPT RE predictions written to {args.output}: relations={len(predictions)}")
    if args.decisions_output:
        print(f"- decisions: {args.decisions_output}")
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    from gutbrainie.pipeline import run_prediction_pipeline

    try:
        result = run_prediction_pipeline(
            data_root=args.data_root,
            split=args.split,
            articles_path=args.articles,
            ner_model=args.ner_model,
            re_model=args.re_model,
            output_dir=args.output_dir,
            metrics_output=args.metrics_output,
            config_output=args.config_output,
            ner_backend=args.ner_backend,
            re_backend=args.re_backend,
            ner_config=args.ner_config,
            train_entities=args.train_entities,
            train_relations=args.train_relations,
            ner_batch_size=args.ner_batch_size,
            ner_max_length=args.ner_max_length,
            re_threshold=args.re_threshold,
            re_batch_size=args.re_batch_size,
            re_max_length=args.re_max_length,
            use_cpu=args.use_cpu,
        )
    except (FileNotFoundError, ValueError, ModuleNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Pipeline completed for split '{args.split}'")
    print(f"- entities: {result['outputs']['entities']} ({result['counts']['entities']})")
    print(f"- mention relations: {result['outputs']['mention_relations']} ({result['counts']['mention_relations']})")
    print(f"- config: {result['outputs']['config']}")
    print(f"- metrics: {result['outputs']['metrics']}")
    return 0


def _atlop_notes(args: argparse.Namespace) -> int:
    from gutbrainie.re.atlop_wrapper import write_atlop_notes

    status = write_atlop_notes(
        official_repo=args.official_repo,
        output_path=args.output,
        data_root=args.data_root,
    )
    print(f"ATLOP reproduction notes written to {status['output']}")
    print(f"- official repo exists: {status['official_repo_exists']}")
    print(f"- can compose: {status['can_compose']}")
    print(f"- can fine-tune: {status['can_finetune']}")
    print(f"- can predict: {status['can_predict']}")
    return 0


def _prepare_atlop_data(args: argparse.Namespace) -> int:
    from gutbrainie.re.atlop_converter import prepare_official_atlop_data

    try:
        result = prepare_official_atlop_data(
            data_root=args.data_root,
            official_repo=args.official_repo,
            predicted_entities=args.predicted_entities,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "pandas":
            raise SystemExit("Missing dependency 'pandas'. Run: pip install -r requirements.txt") from exc
        raise

    print(f"ATLOP data written to {result['output_dir']}")
    for quality, info in result["files"].items():
        print(f"- {quality}: {info['path']} documents={info['documents']}")
    if "predicted_entities" in result:
        info = result["predicted_entities"]
        print(f"- predicted entities: {info['path']} documents={info['documents']}")
    return 0


def _run_atlop(args: argparse.Namespace) -> int:
    from gutbrainie.re.atlop_wrapper import run_atlop_action

    log_path = args.log or f"outputs/reports/atlop_{args.action}.log"
    try:
        result = run_atlop_action(
            official_repo=args.official_repo,
            action=args.action,
            output_path=args.output,
            log_path=log_path,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    command = " ".join(result["command"])
    print(f"ATLOP {args.action} command: {command}")
    print(f"- cwd: {result['cwd']}")
    if args.dry_run:
        print("- dry run only; no official script was executed")
        return 0
    print(f"- return code: {result['returncode']}")
    if result.get("log_path"):
        print(f"- log: {result['log_path']}")
    if result.get("output_path"):
        print(f"- copied prediction output: {result['output_path']}")
    return int(result["returncode"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
