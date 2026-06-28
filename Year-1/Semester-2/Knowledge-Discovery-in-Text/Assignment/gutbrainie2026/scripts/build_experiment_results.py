from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PUBMEDBERT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
BIOMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
SCIBERT = "allenai/scibert_scivocab_uncased"
BIOBERT = "dmis-lab/biobert-base-cased-v1.1"
BIOLINKBERT = "michiyasunaga/BioLinkBERT-base"
GLINER = "urchade/gliner_medium-v2.1"


FIELDNAMES = [
    "experiment_id",
    "run_date",
    "task",
    "split",
    "evaluation_type",
    "status",
    "include_in_report",
    "model_family",
    "model_role",
    "encoder",
    "re_encoder",
    "ner_encoder",
    "train_quality",
    "re_train_quality",
    "ner_train_quality",
    "entity_source",
    "entity_source_detail",
    "max_candidates",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "tp",
    "fp",
    "fn",
    "gold_total",
    "pred_total",
    "rank_by_task_micro_f1",
    "rank_by_status_micro_f1",
    "metrics_path",
    "prediction_path",
    "notes",
]


@dataclass
class Metadata:
    task: str
    evaluation_type: str = "internal_dev_exact"
    status: str = "canonical"
    include_in_report: bool = True
    model_family: str = ""
    model_role: str = ""
    encoder: str = ""
    re_encoder: str = ""
    ner_encoder: str = ""
    train_quality: str = ""
    re_train_quality: str = ""
    ner_train_quality: str = ""
    entity_source: str = ""
    entity_source_detail: str = ""
    max_candidates: str = ""
    prediction_path: str = ""
    notes: str = ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a report-friendly GutBrainIE experiment table from metric JSON files.")
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment_results.csv"))
    parser.add_argument("--copy-output", type=Path, default=Path("outputs/experiment_results_gutbrainie.csv"))
    parser.add_argument(
        "--backup-existing",
        type=Path,
        default=Path("outputs/experiment_results_legacy_segmentation_backup.csv"),
        help="Backup path used if the existing output looks like the older segmentation CSV.",
    )
    args = parser.parse_args()

    rows = build_rows(args.reports_dir, args.predictions_dir)
    rows = add_ranks(rows)

    maybe_backup_legacy_output(args.output, args.backup_existing)
    write_csv(args.output, rows)
    if args.copy_output and args.copy_output != args.output:
        write_csv(args.copy_output, rows)

    print(f"Wrote {len(rows)} experiments to {args.output}")
    if args.copy_output and args.copy_output != args.output:
        print(f"Wrote copy to {args.copy_output}")
    return 0


def build_rows(reports_dir: Path, predictions_dir: Path) -> list[dict[str, Any]]:
    run_date = datetime.now(timezone.utc).date().isoformat()
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(reports_dir.glob("metrics_dev_*.json")):
        metadata = parse_metadata(metrics_path, predictions_dir)
        metrics = load_json(metrics_path)
        row = {
            "experiment_id": experiment_id(metrics_path),
            "run_date": run_date,
            "task": metadata.task,
            "split": "dev",
            "evaluation_type": metadata.evaluation_type,
            "status": metadata.status,
            "include_in_report": str(metadata.include_in_report).lower(),
            "model_family": metadata.model_family,
            "model_role": metadata.model_role,
            "encoder": metadata.encoder,
            "re_encoder": metadata.re_encoder,
            "ner_encoder": metadata.ner_encoder,
            "train_quality": metadata.train_quality,
            "re_train_quality": metadata.re_train_quality,
            "ner_train_quality": metadata.ner_train_quality,
            "entity_source": metadata.entity_source,
            "entity_source_detail": metadata.entity_source_detail,
            "max_candidates": metadata.max_candidates,
            "micro_precision": value(metrics, "micro_precision"),
            "micro_recall": value(metrics, "micro_recall"),
            "micro_f1": value(metrics, "micro_f1"),
            "macro_precision": value(metrics, "macro_precision"),
            "macro_recall": value(metrics, "macro_recall"),
            "macro_f1": value(metrics, "macro_f1"),
            "tp": value(metrics, "tp", "true_positives"),
            "fp": value(metrics, "fp", "false_positives"),
            "fn": value(metrics, "fn", "false_negatives"),
            "gold_total": value(metrics, "gold_total"),
            "pred_total": value(metrics, "pred_total"),
            "rank_by_task_micro_f1": "",
            "rank_by_status_micro_f1": "",
            "metrics_path": rel(metrics_path),
            "prediction_path": metadata.prediction_path,
            "notes": metadata.notes,
        }
        rows.append(row)
    return rows


def parse_metadata(metrics_path: Path, predictions_dir: Path) -> Metadata:
    stem = metrics_path.stem
    if stem.startswith("metrics_dev_ner_"):
        return parse_ner_metadata(stem.removeprefix("metrics_dev_ner_"), predictions_dir)
    if stem.startswith("metrics_dev_re_"):
        return parse_re_metadata(stem.removeprefix("metrics_dev_re_"), predictions_dir)
    return Metadata(task="unknown", status="unparsed", include_in_report=False, notes="Unrecognized metric filename.")


def parse_ner_metadata(suffix: str, predictions_dir: Path) -> Metadata:
    prediction = existing_prediction(predictions_dir / f"dev_t611_{suffix}.json")
    base = Metadata(task="T611_NER", model_role="entity_extractor", entity_source="n/a")

    if suffix == "self":
        base.status = "sanity_check"
        base.include_in_report = False
        base.model_family = "gold_as_prediction"
        base.encoder = "none"
        base.notes = "Gold entities evaluated against themselves; useful only as an evaluator sanity check."
    elif suffix == "dictionary":
        base.model_family = "dictionary_baseline"
        base.encoder = "none"
        base.train_quality = "gold"
        base.notes = "Exact-match dictionary baseline built from train gold entities."
    elif suffix.startswith("gliner_"):
        base.model_family = "gliner"
        base.encoder = GLINER
        base.train_quality = suffix.removeprefix("gliner_")
    elif suffix.startswith("pubmedbert_"):
        base.model_family = "token_classifier"
        base.encoder = PUBMEDBERT
        base.train_quality = suffix.removeprefix("pubmedbert_")
    elif suffix.startswith("biomedbert_"):
        base.model_family = "token_classifier"
        base.encoder = BIOMEDBERT
        base.train_quality = suffix.removeprefix("biomedbert_")
    elif suffix.startswith("scibert_"):
        base.model_family = "token_classifier"
        base.encoder = SCIBERT
        base.train_quality = suffix.removeprefix("scibert_")
    elif suffix.startswith("biobert_"):
        base.model_family = "token_classifier"
        base.encoder = BIOBERT
        base.train_quality = suffix.removeprefix("biobert_")
    elif suffix.startswith("token_classifier_"):
        base.model_family = "token_classifier"
        base.encoder = BIOMEDBERT
        base.train_quality = suffix.removeprefix("token_classifier_")
        base.status = "legacy_generic"
        base.include_in_report = False
        base.notes = "Generic token-classifier filename from the default BiomedBERT config; excluded from final report to avoid counting it as a separate named experiment."
    else:
        base.status = "unparsed"
        base.include_in_report = False
        base.notes = "Unrecognized NER filename pattern."

    return with_prediction_note(base, prediction)


def parse_re_metadata(suffix: str, predictions_dir: Path) -> Metadata:
    prediction = prediction_for_re_suffix(suffix, predictions_dir)
    base = Metadata(task="T621_RE", model_role="relation_extractor")

    if suffix == "self":
        base.status = "sanity_check"
        base.include_in_report = False
        base.model_family = "gold_as_prediction"
        base.encoder = "none"
        base.entity_source = "gold"
        base.notes = "Gold relations evaluated against themselves; useful only as an evaluator sanity check."
        return with_prediction_note(base, prediction)

    if suffix.startswith("rule_"):
        base.model_family = "relation_prior_rule"
        base.encoder = "none"
        base.re_train_quality = "gold"
        base.train_quality = "gold"
        if suffix == "rule_gold_entities":
            base.evaluation_type = "internal_dev_exact_gold_entities"
            base.entity_source = "gold"
            base.entity_source_detail = "dev gold entities"
        elif suffix == "rule_ner_pubmedbert_gold_silver_silver_2025_entities":
            base.evaluation_type = "internal_dev_exact_end_to_end"
            base.entity_source = "predicted"
            base.entity_source_detail = "PubMedBERT NER predictions"
            base.ner_encoder = PUBMEDBERT
            base.ner_train_quality = "gold_silver_silver_2025"
        elif suffix == "rule_gold_silver_silver_2025_pubmedbert_entities":
            base.status = "legacy_duplicate"
            base.include_in_report = False
            base.evaluation_type = "internal_dev_exact_end_to_end"
            base.entity_source = "predicted"
            base.entity_source_detail = "Legacy filename for PubMedBERT NER gold_silver_silver_2025."
            base.ner_encoder = PUBMEDBERT
            base.ner_train_quality = "gold_silver_silver_2025"
            base.notes = "Duplicate of the canonical rule_ner_pubmedbert_gold_silver_silver_2025_entities result."
        elif suffix == "rule_predicted_entities":
            base.status = "legacy_ambiguous"
            base.include_in_report = False
            base.evaluation_type = "internal_dev_exact_end_to_end"
            base.entity_source = "predicted"
            base.entity_source_detail = "Unknown earlier predicted-entity file."
            base.notes = "Legacy ambiguous filename; keep for audit but exclude from report comparisons."
        else:
            base.status = "unparsed"
            base.include_in_report = False
            base.notes = "Unrecognized rule-baseline filename pattern."
        return with_prediction_note(base, prediction)

    if suffix.startswith("pair_classifier_"):
        return with_prediction_note(parse_pair_classifier_metadata(suffix.removeprefix("pair_classifier_")), prediction)

    if suffix == "gpt_pubmedbert_entities_50":
        base.model_family = "llm_verifier_gpt"
        base.encoder = "gpt-4o-mini"
        base.entity_source = "predicted"
        base.entity_source_detail = "PubMedBERT NER predictions"
        base.ner_encoder = PUBMEDBERT
        base.ner_train_quality = "gold_silver_silver_2025"
        base.max_candidates = "50"
        base.evaluation_type = "internal_dev_exact_end_to_end_limited_candidates"
        base.notes = "OpenAI GPT verifier run on a limited candidate subset; not directly comparable to full candidate sweeps."
        return with_prediction_note(base, prediction)

    if suffix in {"ollama_llama31", "ollama_llama31_1000"}:
        base.model_family = "llm_verifier_ollama"
        base.encoder = "llama3.1:8b"
        base.entity_source = "predicted"
        base.entity_source_detail = "PubMedBERT NER predictions"
        base.ner_encoder = PUBMEDBERT
        base.ner_train_quality = "gold_silver_silver_2025"
        base.max_candidates = "1000" if suffix.endswith("_1000") else "200"
        base.evaluation_type = "internal_dev_exact_end_to_end_limited_candidates"
        base.notes = "Local Ollama verifier; max-candidate setting limits direct comparability with full candidate sweeps."
        return with_prediction_note(base, prediction)

    base.status = "unparsed"
    base.include_in_report = False
    base.notes = "Unrecognized RE filename pattern."
    return with_prediction_note(base, prediction)


def parse_pair_classifier_metadata(suffix: str) -> Metadata:
    base = Metadata(
        task="T621_RE",
        model_family="pair_classifier",
        model_role="relation_extractor",
        encoder=PUBMEDBERT,
        re_encoder=PUBMEDBERT,
    )

    cases: dict[str, dict[str, str | bool]] = {
        "gold_gold_entities": {
            "re_train_quality": "gold",
            "entity_source": "gold",
            "evaluation_type": "internal_dev_exact_gold_entities",
            "entity_source_detail": "dev gold entities",
        },
        "gold_silver_gold_entities": {
            "re_train_quality": "gold_silver",
            "entity_source": "gold",
            "evaluation_type": "internal_dev_exact_gold_entities",
            "entity_source_detail": "dev gold entities",
        },
        "pubmedbert_re_gold_ner_pubmedbert_gold_silver_silver_2025_entities": {
            "re_train_quality": "gold",
            "entity_source": "predicted",
            "evaluation_type": "internal_dev_exact_end_to_end",
            "entity_source_detail": "PubMedBERT NER predictions",
            "ner_encoder": PUBMEDBERT,
            "ner_train_quality": "gold_silver_silver_2025",
        },
        "pubmedbert_re_gold_silver_ner_pubmedbert_gold_silver_entities": {
            "re_train_quality": "gold_silver",
            "entity_source": "predicted",
            "evaluation_type": "internal_dev_exact_end_to_end",
            "entity_source_detail": "PubMedBERT NER predictions",
            "ner_encoder": PUBMEDBERT,
            "ner_train_quality": "gold_silver",
        },
        "gold_silver_pubmedbert_entities": {
            "re_train_quality": "gold_silver",
            "entity_source": "predicted",
            "evaluation_type": "internal_dev_exact_end_to_end",
            "entity_source_detail": "Legacy ambiguous PubMedBERT NER predictions.",
            "ner_encoder": PUBMEDBERT,
            "ner_train_quality": "unknown",
            "status": "legacy_ambiguous",
            "include_in_report": False,
            "notes": "Ambiguous old filename; use canonical pubmedbert_re_*_ner_* rows for report comparisons.",
        },
        "biolinkbert_gold_gold_entities": {
            "encoder": BIOLINKBERT,
            "re_encoder": BIOLINKBERT,
            "re_train_quality": "gold",
            "entity_source": "gold",
            "evaluation_type": "internal_dev_exact_gold_entities",
            "entity_source_detail": "dev gold entities",
        },
        "biolinkbert_gold_pubmedbert_entities": {
            "encoder": BIOLINKBERT,
            "re_encoder": BIOLINKBERT,
            "re_train_quality": "gold",
            "entity_source": "predicted",
            "evaluation_type": "internal_dev_exact_end_to_end",
            "entity_source_detail": "Legacy ambiguous PubMedBERT NER predictions.",
            "ner_encoder": PUBMEDBERT,
            "ner_train_quality": "unknown",
            "status": "legacy_ambiguous",
            "include_in_report": False,
            "notes": "Ambiguous old filename; rerun or rename if this should be a canonical report row.",
        },
        "biolinkbert_gold_silver_gold_entities": {
            "encoder": BIOLINKBERT,
            "re_encoder": BIOLINKBERT,
            "re_train_quality": "gold_silver",
            "entity_source": "gold",
            "evaluation_type": "internal_dev_exact_gold_entities",
            "entity_source_detail": "dev gold entities",
        },
        "biolinkbert_gold_silver_pubmedbert_gold_silver_entities": {
            "encoder": BIOLINKBERT,
            "re_encoder": BIOLINKBERT,
            "re_train_quality": "gold_silver",
            "entity_source": "predicted",
            "evaluation_type": "internal_dev_exact_end_to_end",
            "entity_source_detail": "PubMedBERT NER predictions",
            "ner_encoder": PUBMEDBERT,
            "ner_train_quality": "gold_silver",
        },
        "biolinkbert_gold_silver_pubmedbert_gold_silver_silver_2025_entities": {
            "encoder": BIOLINKBERT,
            "re_encoder": BIOLINKBERT,
            "re_train_quality": "gold_silver",
            "entity_source": "predicted",
            "evaluation_type": "internal_dev_exact_end_to_end",
            "entity_source_detail": "PubMedBERT NER predictions",
            "ner_encoder": PUBMEDBERT,
            "ner_train_quality": "gold_silver_silver_2025",
        },
    }

    case = cases.get(suffix)
    if not case:
        base.status = "unparsed"
        base.include_in_report = False
        base.notes = "Unrecognized pair-classifier filename pattern."
        return base

    for key, val in case.items():
        setattr(base, key, val)
    base.train_quality = base.re_train_quality
    return base


def prediction_for_re_suffix(suffix: str, predictions_dir: Path) -> Path | None:
    candidates = [
        predictions_dir / f"dev_t621_{suffix}.json",
        predictions_dir / f"dev_t621_rule_{suffix.removeprefix('rule_')}.json",
        predictions_dir / f"dev_t621_pair_classifier_{suffix.removeprefix('pair_classifier_')}.json",
    ]
    if suffix == "rule_predicted_entities":
        candidates.append(predictions_dir / "dev_t621_rule_ner_unknown_predicted_entities.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def existing_prediction(path: Path) -> Path | None:
    return path if path.exists() else None


def with_prediction_note(metadata: Metadata, prediction: Path | None) -> Metadata:
    if prediction:
        metadata.prediction_path = rel(prediction)
    return metadata


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def value(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return ""


def experiment_id(path: Path) -> str:
    return path.stem.removeprefix("metrics_dev_")


def add_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank_groups = [
        ("rank_by_task_micro_f1", lambda row: row["task"]),
        ("rank_by_status_micro_f1", lambda row: (row["task"], row["status"])),
    ]
    for rank_field, group_key in rank_groups:
        grouped: dict[Any, list[dict[str, Any]]] = {}
        for row in rows:
            if row["status"] == "sanity_check":
                continue
            grouped.setdefault(group_key(row), []).append(row)
        for group_rows in grouped.values():
            sorted_rows = sorted(group_rows, key=lambda row: float_or_negative(row["micro_f1"]), reverse=True)
            for rank, row in enumerate(sorted_rows, start=1):
                row[rank_field] = rank
    return rows


def float_or_negative(value_: Any) -> float:
    try:
        return float(value_)
    except (TypeError, ValueError):
        return -1.0


def maybe_backup_legacy_output(output: Path, backup: Path) -> None:
    if not output.exists() or backup.exists():
        return
    first_line = output.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if "mean_iou" in first_line and "architecture" in first_line:
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, backup)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
