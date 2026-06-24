import json

import pandas as pd

from gutbrainie.ner.dictionary_baseline import (
    build_entity_dictionary,
    normalize_entity_key,
    predict_dictionary_entities,
    predict_dictionary_to_json,
)
from gutbrainie.submission.export_t611 import load_t611_json
from gutbrainie.submission.validate import validate_t611_payload


def test_normalize_entity_key_lowercases_and_strips_trailing_punctuation():
    assert normalize_entity_key(" Patients, ") == "patients"


def test_build_entity_dictionary_keeps_labels_for_normalized_spans():
    train = pd.DataFrame(
        [
            {"text_span": "Patients", "label": "human"},
            {"text_span": "patients.", "label": "human"},
            {"text_span": "patients", "label": "animal"},
        ]
    )

    dictionary = build_entity_dictionary(train)

    assert dictionary == {"patients": {"human", "animal"}}


def test_predict_dictionary_entities_matches_title_and_abstract_case_insensitively():
    train = pd.DataFrame(
        [
            {"text_span": "Gut microbiota", "label": "microbiome"},
            {"text_span": "microbiota", "label": "microbiome"},
            {"text_span": "Alzheimer disease", "label": "DDF"},
        ]
    )
    articles = pd.DataFrame(
        [
            {
                "pmid": "1",
                "title": "Gut Microbiota in Alzheimer disease",
                "abstract": "The microbiota changed.",
            }
        ]
    )

    predictions = predict_dictionary_entities(articles, train)

    assert list(predictions["text_span"]) == ["Gut Microbiota", "Alzheimer disease", "microbiota"]
    assert list(predictions["location"]) == ["title", "title", "abstract"]
    title_predictions = predictions[predictions["location"] == "title"]
    assert "Microbiota" not in set(title_predictions["text_span"])


def test_predict_dictionary_to_json_writes_valid_t611_payload(tmp_path):
    train_path = tmp_path / "entities.csv"
    articles_path = tmp_path / "articles.csv"
    output_path = tmp_path / "predictions.json"

    train_path.write_text(
        "pmid|annotator|start_idx|end_idx|location|text_span|label\n"
        "1|ann|0|8|title|Butyrate|chemical\n",
        encoding="utf-8",
    )
    articles_path.write_text(
        "pmid|title|authors|journal|year|abstract\n"
        "2|Butyrate response|A Author|Journal|2026|No abstract hit.\n",
        encoding="utf-8",
    )

    predictions = predict_dictionary_to_json(train_path, articles_path, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(predictions) == 1
    assert validate_t611_payload(payload)
    assert payload["2"]["entities"][0]["text_span"] == "Butyrate"
    assert payload["2"]["entities"][0]["end_idx"] == 7

    loaded = load_t611_json(output_path)
    assert loaded.loc[0, "pmid"] == "2"
    assert loaded.loc[0, "start_idx"] == 0
    assert loaded.loc[0, "end_idx"] == 7
    assert loaded.loc[0, "label"] == "chemical"
