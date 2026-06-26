"""End-to-end T611/T621 pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gutbrainie.data.splits import resolve_split_paths


def run_prediction_pipeline(
    data_root: str | Path,
    split: str,
    ner_model: str | Path,
    re_model: str | Path,
    output_dir: str | Path,
    ner_backend: str = "token-classifier",
    re_backend: str = "pair-classifier",
    articles_path: str | Path | None = None,
    metrics_output: str | Path | None = None,
    config_output: str | Path | None = None,
    ner_config: str | Path = "configs/ner_gliner.yaml",
    train_entities: str | Path | None = None,
    train_relations: str | Path | None = None,
    ner_batch_size: int = 8,
    ner_max_length: int = 512,
    re_threshold: float = 0.5,
    re_batch_size: int = 8,
    re_max_length: int = 512,
    use_cpu: bool = False,
) -> dict[str, Any]:
    """Run NER, then RE, then optional dev evaluation."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_articles = Path(articles_path) if articles_path else resolve_pipeline_articles_path(data_root, split)
    entities_output = output_dir / f"{split}_t611_entities.json"
    relations_output = output_dir / f"{split}_t621_mention_relations.json"
    metrics_path = Path(metrics_output) if metrics_output else default_pipeline_metrics_path(output_dir, split)
    config_path = Path(config_output) if config_output else output_dir / "pipeline_config.json"

    config: dict[str, Any] = {
        "data_root": str(data_root),
        "split": split,
        "articles": str(resolved_articles),
        "output_dir": str(output_dir),
        "entities_output": str(entities_output),
        "relations_output": str(relations_output),
        "metrics_output": str(metrics_path),
        "config_output": str(config_path),
        "ner_backend": ner_backend,
        "ner_model": str(ner_model),
        "ner_config": str(ner_config),
        "ner_batch_size": ner_batch_size,
        "ner_max_length": ner_max_length,
        "re_backend": re_backend,
        "re_model": str(re_model),
        "re_threshold": re_threshold,
        "re_batch_size": re_batch_size,
        "re_max_length": re_max_length,
        "use_cpu": use_cpu,
    }
    if train_entities is not None:
        config["train_entities"] = str(train_entities)
    if train_relations is not None:
        config["train_relations"] = str(train_relations)

    ner_predictions = _run_ner(
        backend=ner_backend,
        model=ner_model,
        articles_path=resolved_articles,
        output_path=entities_output,
        config_path=ner_config,
        train_entities=train_entities,
        batch_size=ner_batch_size,
        max_length=ner_max_length,
        use_cpu=use_cpu,
    )
    re_predictions = _run_re(
        backend=re_backend,
        model=re_model,
        articles_path=resolved_articles,
        entities_path=entities_output,
        output_path=relations_output,
        train_relations=train_relations,
        threshold=re_threshold,
        batch_size=re_batch_size,
        max_length=re_max_length,
        use_cpu=use_cpu,
    )

    metrics = _evaluate_if_available(
        data_root=data_root,
        split=split,
        entities_prediction=entities_output,
        relations_prediction=relations_output,
    )
    result: dict[str, Any] = {
        "config": config,
        "outputs": {
            "entities": str(entities_output),
            "mention_relations": str(relations_output),
            "metrics": str(metrics_path),
            "config": str(config_path),
        },
        "counts": {
            "entities": int(len(ner_predictions)),
            "mention_relations": int(len(re_predictions)),
        },
        "metrics": metrics,
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def resolve_pipeline_articles_path(data_root: str | Path, split: str) -> Path:
    """Resolve article CSV path for pipeline splits."""
    root = Path(data_root)
    if split == "test":
        return root / "Test_Data" / "articles_test.csv"
    if split == "dev":
        return resolve_split_paths(root, "dev").articles
    if split in {"gold", "silver", "silver_2025", "bronze"}:
        return resolve_split_paths(root, split).articles
    candidate = root / "Articles" / "csv_format" / f"articles_{split}.csv"
    if candidate.exists():
        return candidate
    raise ValueError("Unsupported split for pipeline. Use dev, test, gold, silver, silver_2025, or bronze.")


def default_pipeline_metrics_path(output_dir: str | Path, split: str) -> Path:
    output_dir = Path(output_dir)
    reports_root = output_dir.parent.parent / "reports" if output_dir.parent.name == "predictions" else Path("outputs/reports")
    return reports_root / f"pipeline_{split}_metrics.json"


def _run_ner(
    backend: str,
    model: str | Path,
    articles_path: Path,
    output_path: Path,
    config_path: str | Path,
    train_entities: str | Path | None,
    batch_size: int,
    max_length: int,
    use_cpu: bool,
):
    if backend == "token-classifier":
        from gutbrainie.ner.predict_token_classifier import predict_token_classifier_to_json

        return predict_token_classifier_to_json(
            model_path=model,
            articles_path=articles_path,
            output_path=output_path,
            batch_size=batch_size,
            max_length=max_length,
            use_cpu=use_cpu,
        )
    if backend == "gliner":
        from gutbrainie.config import load_yaml
        from gutbrainie.ner.gliner_runner import predict_gliner_to_json

        config = load_yaml(config_path)
        return predict_gliner_to_json(
            model_name_or_path=str(model),
            articles_path=articles_path,
            output_path=output_path,
            labels=config.get("labels"),
            threshold=float(config.get("threshold", 0.5)),
            batch_size=int(config.get("batch_size", batch_size)),
            max_len=int(config["max_len"]) if config.get("max_len") else None,
        )
    if backend == "dictionary":
        from gutbrainie.ner.dictionary_baseline import predict_dictionary_to_json

        train_entities_path = train_entities or model
        return predict_dictionary_to_json(
            train_entities_path=train_entities_path,
            articles_path=articles_path,
            output_path=output_path,
        )
    raise ValueError(f"Unsupported NER backend: {backend}")


def _run_re(
    backend: str,
    model: str | Path,
    articles_path: Path,
    entities_path: Path,
    output_path: Path,
    train_relations: str | Path | None,
    threshold: float,
    batch_size: int,
    max_length: int,
    use_cpu: bool,
):
    if backend == "pair-classifier":
        from gutbrainie.re.predict_pair_classifier import predict_pair_classifier_to_json

        return predict_pair_classifier_to_json(
            model_path=model,
            articles_path=articles_path,
            entities_path=entities_path,
            output_path=output_path,
            threshold=threshold,
            batch_size=batch_size,
            max_length=max_length,
            use_cpu=use_cpu,
        )
    if backend == "rule":
        from gutbrainie.re.rule_baseline import predict_re_rule_to_json

        if train_relations is None:
            raise ValueError("--train-relations is required when --re-backend rule is used.")
        return predict_re_rule_to_json(
            articles_path=articles_path,
            entities_path=entities_path,
            train_relations_path=train_relations,
            output_path=output_path,
            threshold=threshold,
        )
    raise ValueError(f"Unsupported RE backend: {backend}")


def _evaluate_if_available(
    data_root: Path,
    split: str,
    entities_prediction: Path,
    relations_prediction: Path,
) -> dict[str, Any]:
    if split != "dev":
        return {"evaluated": False, "reason": f"No gold annotations are configured for split '{split}'."}

    from gutbrainie.data.annotations import load_entities_csv, load_mention_relations_csv
    from gutbrainie.evaluation.ner_metrics import evaluate_ner
    from gutbrainie.evaluation.re_metrics import evaluate_mention_relations
    from gutbrainie.submission.export_t611 import load_t611_json
    from gutbrainie.submission.export_t621 import load_t621_json

    paths = resolve_split_paths(data_root, "dev")
    if not paths.entities.exists() or not paths.mention_relations.exists():
        return {"evaluated": False, "reason": "Dev gold annotation files are missing."}

    return {
        "evaluated": True,
        "ner": evaluate_ner(load_entities_csv(paths.entities), load_t611_json(entities_prediction)),
        "re": evaluate_mention_relations(
            load_mention_relations_csv(paths.mention_relations),
            load_t621_json(relations_prediction),
        ),
    }
