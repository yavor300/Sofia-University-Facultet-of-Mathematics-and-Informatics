import json
import os

from gutbrainie.llm.gpt_relation_verifier import load_dotenv, predict_re_gpt_to_json
from gutbrainie.submission.export_t621 import load_t621_json


class FakeGPTClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_load_dotenv_strips_quotes_and_preserves_existing_values(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "OPENAI_API_KEY='from-file'\n"
        "OPENAI_JUDGE_MODEL=\"gpt-test\"\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")

    loaded = load_dotenv(env)

    assert loaded == {"OPENAI_API_KEY": "from-file", "OPENAI_JUDGE_MODEL": "gpt-test"}
    assert loaded["OPENAI_JUDGE_MODEL"] == "gpt-test"
    assert loaded["OPENAI_API_KEY"] == "from-file"
    assert os.environ["OPENAI_API_KEY"] == "already-set"


def test_predict_re_gpt_to_json_writes_predictions_and_decisions(tmp_path):
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
    client = FakeGPTClient(['{"predicate": "influence", "confidence": 0.82}'])

    predictions = predict_re_gpt_to_json(
        articles_path=articles,
        entities_path=entities,
        output_path=output,
        model="gpt-test",
        env_path=None,
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
    assert decision_rows[0]["provider"] == "openai"
    assert decision_rows[0]["model"] == "gpt-test"
    assert decision_rows[0]["predicate"] == "influence"
    assert "Allowed predicates: influence, no_relation." in client.prompts[0]
