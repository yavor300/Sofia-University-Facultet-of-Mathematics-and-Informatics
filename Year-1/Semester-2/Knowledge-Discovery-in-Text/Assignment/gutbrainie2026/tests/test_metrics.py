import json

import pandas as pd
import pytest

from gutbrainie.cli import main
from gutbrainie.evaluation.ner_metrics import evaluate_ner
from gutbrainie.evaluation.re_metrics import evaluate_mention_relations


def test_evaluate_ner_exact_match_metrics():
    gold = pd.DataFrame(
        [
            {"pmid": "1", "location": "title", "start_idx": 0, "end_idx": 3, "text_span": "Gut", "label": "DDF"},
            {"pmid": "1", "location": "title", "start_idx": 4, "end_idx": 9, "text_span": "brain", "label": "DDF"},
            {"pmid": "2", "location": "abstract", "start_idx": 0, "end_idx": 8, "text_span": "Butyrate", "label": "chemical"},
        ]
    )
    pred = pd.DataFrame(
        [
            {"pmid": "1", "location": "title", "start_idx": 0, "end_idx": 3, "text_span": "Gut", "label": "DDF"},
            {"pmid": "1", "location": "title", "start_idx": 4, "end_idx": 9, "text_span": "brain", "label": "human"},
            {"pmid": "2", "location": "abstract", "start_idx": 0, "end_idx": 8, "text_span": "Wrong text", "label": "chemical"},
            {"pmid": "2", "location": "abstract", "start_idx": 0, "end_idx": 8, "text_span": "Wrong text", "label": "chemical"},
        ]
    )

    metrics = evaluate_ner(gold, pred)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 2
    assert metrics["fn"] == 1
    assert metrics["micro_precision"] == pytest.approx(0.5)
    assert metrics["micro_recall"] == pytest.approx(2 / 3)
    assert metrics["micro_f1"] == pytest.approx(4 / 7)
    assert metrics["per_label_f1"]["chemical"] == pytest.approx(2 / 3)
    assert metrics["per_label_f1"]["DDF"] == pytest.approx(2 / 3)
    assert metrics["per_label_f1"]["human"] == pytest.approx(0.0)


def test_evaluate_mention_relations_uses_triple_macro_labels():
    gold = pd.DataFrame(
        [
            {
                "pmid": "1",
                "subject_text_span": "Microbes",
                "subject_label": "bacteria",
                "predicate": "affect",
                "object_text_span": "neurons",
                "object_label": "anatomical location",
            },
            {
                "pmid": "1",
                "subject_text_span": "Butyrate",
                "subject_label": "chemical",
                "predicate": "influence",
                "object_text_span": "inflammation",
                "object_label": "DDF",
            },
        ]
    )
    pred = pd.DataFrame(
        [
            {
                "pmid": "1",
                "subject_text_span": "Microbes",
                "subject_label": "bacteria",
                "predicate": "affect",
                "object_text_span": "neurons",
                "object_label": "anatomical location",
            },
            {
                "pmid": "1",
                "subject_text_span": "Butyrate",
                "subject_label": "chemical",
                "predicate": "located in",
                "object_text_span": "inflammation",
                "object_label": "DDF",
            },
        ]
    )

    metrics = evaluate_mention_relations(gold, pred)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["micro_f1"] == pytest.approx(0.5)
    assert "bacteria|affect|anatomical location" in metrics["per_label_f1"]
    assert "chemical|influence|DDF" in metrics["per_label_f1"]
    assert "chemical|located in|DDF" in metrics["per_label_f1"]


def test_evaluate_cli_writes_metrics_json(tmp_path):
    gold = tmp_path / "gold_entities.csv"
    pred = tmp_path / "pred_entities.csv"
    output = tmp_path / "metrics.json"

    gold.write_text(
        "pmid|annotator|start_idx|end_idx|location|text_span|label\n"
        "1|ann|0|3|title|Gut|DDF\n",
        encoding="utf-8",
    )
    pred.write_text(
        "pmid|annotator|start_idx|end_idx|location|text_span|label\n"
        "1|model|0|3|title|Different span text ignored|DDF\n",
        encoding="utf-8",
    )

    assert main(["evaluate", "--task", "ner", "--gold", str(gold), "--prediction", str(pred), "--output", str(output)]) == 0

    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["micro_f1"] == pytest.approx(1.0)
