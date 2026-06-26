import pandas as pd

from app.components.entity_highlighter import compare_entities, highlight_entities
from app.components.relation_viewer import compare_relations, relation_cards_html


def test_compare_entities_marks_tp_fp_and_fn():
    gold = pd.DataFrame(
        [
            {"pmid": "1", "start_idx": 0, "end_idx": 9, "location": "title", "text_span": "microbiome", "label": "microbiome"},
            {"pmid": "1", "start_idx": 20, "end_idx": 30, "location": "title", "text_span": "depression", "label": "DDF"},
        ]
    )
    pred = pd.DataFrame(
        [
            {"pmid": "1", "start_idx": 0, "end_idx": 9, "location": "title", "text_span": "microbiome", "label": "microbiome"},
            {"pmid": "1", "start_idx": 20, "end_idx": 30, "location": "title", "text_span": "depression", "label": "chemical"},
        ]
    )

    compared = compare_entities(gold, pred)

    assert sorted(compared["status"].tolist()) == ["False Negative", "False Positive", "True Positive"]


def test_highlight_entities_escapes_raw_text():
    html = highlight_entities(
        "A < B microbiome",
        pd.DataFrame(
            [
                {
                    "pmid": "1",
                    "start_idx": 6,
                    "end_idx": 15,
                    "location": "title",
                    "text_span": "microbiome",
                    "label": "microbiome",
                }
            ]
        ),
    )

    assert "A &lt; B" in html
    assert "entity-span" in html


def test_compare_relations_and_cards():
    gold = pd.DataFrame(
        [
            {
                "pmid": "1",
                "subject_text_span": "microbiome",
                "subject_label": "microbiome",
                "predicate": "is linked to",
                "object_text_span": "depression",
                "object_label": "DDF",
            }
        ]
    )
    pred = pd.DataFrame(
        [
            {
                "pmid": "1",
                "subject_text_span": "microbiome",
                "subject_label": "microbiome",
                "predicate": "impact",
                "object_text_span": "depression",
                "object_label": "DDF",
            }
        ]
    )

    compared = compare_relations(gold, pred)
    cards = relation_cards_html(compared)

    assert sorted(compared["status"].tolist()) == ["False Negative", "False Positive"]
    assert "relation-card" in cards
