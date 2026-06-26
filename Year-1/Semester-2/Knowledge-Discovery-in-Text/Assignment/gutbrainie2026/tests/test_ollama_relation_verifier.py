import json

from gutbrainie.llm.ollama_relation_verifier import (
    article_context,
    normalize_predicate,
    parse_relation_decision,
    predict_re_ollama_to_json,
)
from gutbrainie.submission.export_t621 import load_t621_json


class FakeOllamaClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_parse_relation_decision_accepts_json_only_response():
    decision = parse_relation_decision('{"predicate": "influence", "confidence": 0.8}', ["influence"])

    assert decision == {"predicate": "influence", "confidence": 0.8}


def test_parse_relation_decision_extracts_json_from_extra_text():
    decision = parse_relation_decision('Answer: {"predicate": "no_relation", "confidence": 0.9}', ["influence"])

    assert decision == {"predicate": "no_relation", "confidence": 0.9}


def test_normalize_predicate_rejects_schema_invalid_label():
    assert normalize_predicate("target", ["influence"]) == "no_relation"
    assert normalize_predicate("INFLUENCE", ["influence"]) == "influence"


def test_article_context_uses_single_location_when_possible():
    article = {"title": "Title text", "abstract": "Abstract text"}
    candidate = {"subject_location": "abstract", "object_location": "abstract"}

    assert article_context(article, candidate) == "Abstract: Abstract text"


def test_predict_re_ollama_to_json_writes_predictions_and_decisions(tmp_path):
    articles = tmp_path / "articles.csv"
    entities = tmp_path / "entities.csv"
    output = tmp_path / "predictions.json"
    decisions = tmp_path / "decisions.jsonl"
    articles.write_text(
        "pmid|title|authors|journal|year|abstract\n"
        "1|Title|A|J|2026|Bacteria influence depression.\n",
        encoding="utf-8",
    )
    entities.write_text(
        "pmid|annotator|start_idx|end_idx|location|text_span|label\n"
        "1|ann|0|7|abstract|Bacteria|bacteria\n"
        "1|ann|19|29|abstract|depression|DDF\n",
        encoding="utf-8",
    )
    client = FakeOllamaClient(['{"predicate": "influence", "confidence": 0.75}'])

    predictions = predict_re_ollama_to_json(
        articles_path=articles,
        entities_path=entities,
        output_path=output,
        threshold=0.5,
        max_candidates=1,
        max_distance=0,
        decisions_output=decisions,
        client=client,
    )

    assert len(predictions) == 1
    loaded = load_t621_json(output)
    assert loaded.iloc[0]["predicate"] == "influence"
    decision_rows = [json.loads(line) for line in decisions.read_text(encoding="utf-8").splitlines()]
    assert decision_rows[0]["predicate"] == "influence"
    assert "Allowed predicates: influence, no_relation." in client.prompts[0]
