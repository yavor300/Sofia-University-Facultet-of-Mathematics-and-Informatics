import pandas as pd

from gutbrainie.re.candidates import generate_relation_candidates
from gutbrainie.re.relation_schema import is_valid_label_pair, valid_predicates
from gutbrainie.re.rule_baseline import RelationPriorBaseline
from gutbrainie.submission.export_t621 import load_t621_json, mention_relations_to_t621_json


def test_relation_schema_exposes_valid_predicates():
    assert is_valid_label_pair("bacteria", "DDF")
    assert valid_predicates("bacteria", "DDF") == ["influence"]
    assert not is_valid_label_pair("human", "DDF")


def test_generate_relation_candidates_keeps_offsets_and_text_between():
    articles = pd.DataFrame(
        [
            {
                "pmid": "1",
                "title": "Gut microbes influence disease",
                "abstract": "",
            }
        ]
    )
    entities = pd.DataFrame(
        [
            {
                "pmid": "1",
                "start_idx": 4,
                "end_idx": 11,
                "location": "title",
                "text_span": "microbes",
                "label": "bacteria",
            },
            {
                "pmid": "1",
                "start_idx": 23,
                "end_idx": 29,
                "location": "title",
                "text_span": "disease",
                "label": "DDF",
            },
        ]
    )

    candidates = generate_relation_candidates(articles, entities)

    assert len(candidates) == 2
    row = candidates.loc[candidates["subject_label"].eq("bacteria") & candidates["object_label"].eq("DDF")].iloc[0]
    assert row["subject_text_span"] == "microbes"
    assert row["object_text_span"] == "disease"
    assert row["predicate"] == "no_relation"
    assert row["subject_end_idx"] == 11
    assert row["text_between"] == "influence"
    assert row["sentence_distance"] == 0
    assert row["candidate_key"] == "microbes|bacteria|disease|DDF"


def test_generate_relation_candidates_can_assign_gold_predicate():
    articles = pd.DataFrame([{"pmid": "1", "title": "Microbes disease", "abstract": ""}])
    entities = pd.DataFrame(
        [
            {"pmid": "1", "start_idx": 0, "end_idx": 7, "location": "title", "text_span": "Microbes", "label": "bacteria"},
            {"pmid": "1", "start_idx": 9, "end_idx": 15, "location": "title", "text_span": "disease", "label": "DDF"},
        ]
    )
    gold = pd.DataFrame(
        [
            {
                "pmid": "1",
                "annotator": "ann",
                "subject_text_span": "Microbes",
                "subject_label": "bacteria",
                "predicate": "influence",
                "object_text_span": "disease",
                "object_label": "DDF",
            }
        ]
    )

    candidates = generate_relation_candidates(articles, entities, gold_relations=gold)

    assert candidates.iloc[0]["predicate"] == "influence"


def test_relation_prior_baseline_predicts_above_threshold_and_deduplicates():
    train = pd.DataFrame(
        [
            {"pmid": "1", "subject_text_span": "a", "subject_label": "bacteria", "predicate": "influence", "object_text_span": "b", "object_label": "DDF"},
            {"pmid": "2", "subject_text_span": "c", "subject_label": "bacteria", "predicate": "influence", "object_text_span": "d", "object_label": "DDF"},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"pmid": "3", "subject_text_span": "x", "subject_label": "bacteria", "object_text_span": "y", "object_label": "DDF"},
            {"pmid": "3", "subject_text_span": "x", "subject_label": "bacteria", "object_text_span": "y", "object_label": "DDF"},
        ]
    )

    predictions = RelationPriorBaseline(threshold=0.9).fit(train).predict(candidates)

    assert predictions.to_dict("records") == [
        {
            "pmid": "3",
            "subject_text_span": "x",
            "subject_label": "bacteria",
            "predicate": "influence",
            "object_text_span": "y",
            "object_label": "DDF",
        }
    ]


def test_t621_json_roundtrip(tmp_path):
    relations = pd.DataFrame(
        [
            {
                "pmid": "1",
                "subject_text_span": "microbes",
                "subject_label": "bacteria",
                "predicate": "influence",
                "object_text_span": "disease",
                "object_label": "DDF",
            }
        ]
    )
    path = tmp_path / "pred.json"
    path.write_text(__import__("json").dumps(mention_relations_to_t621_json(relations)), encoding="utf-8")

    loaded = load_t621_json(path)

    assert loaded.to_dict("records") == relations.to_dict("records")
