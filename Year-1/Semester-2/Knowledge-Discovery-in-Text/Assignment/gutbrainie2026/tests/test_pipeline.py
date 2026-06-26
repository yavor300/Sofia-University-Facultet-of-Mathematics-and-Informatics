import json

import pandas as pd

from gutbrainie import pipeline


def test_resolve_pipeline_articles_path_handles_dev_and_test(tmp_path):
    data_root = tmp_path / "data"

    assert pipeline.resolve_pipeline_articles_path(data_root, "dev") == (
        data_root / "Articles" / "csv_format" / "articles_dev.csv"
    )
    assert pipeline.resolve_pipeline_articles_path(data_root, "test") == data_root / "Test_Data" / "articles_test.csv"


def test_run_prediction_pipeline_writes_config_and_metrics_without_gold(tmp_path, monkeypatch):
    articles = tmp_path / "articles_test.csv"
    articles.write_text(
        "pmid|title|authors|journal|year|abstract\n"
        "1|Title|A|J|2026|Abstract.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "pipeline_test"
    metrics_output = tmp_path / "reports" / "pipeline_test_metrics.json"

    def fake_ner(**kwargs):
        kwargs["output_path"].write_text('{"entities": []}\n', encoding="utf-8")
        return pd.DataFrame([{"pmid": "1", "text_span": "Title"}])

    def fake_re(**kwargs):
        kwargs["output_path"].write_text('{"mention_level_relations": []}\n', encoding="utf-8")
        return pd.DataFrame([{"pmid": "1", "predicate": "affect"}])

    monkeypatch.setattr(pipeline, "_run_ner", fake_ner)
    monkeypatch.setattr(pipeline, "_run_re", fake_re)

    result = pipeline.run_prediction_pipeline(
        data_root=tmp_path / "data",
        split="test",
        articles_path=articles,
        ner_model="ner-model",
        re_model="re-model",
        output_dir=output_dir,
        metrics_output=metrics_output,
    )

    assert result["counts"] == {"entities": 1, "mention_relations": 1}
    assert result["metrics"]["evaluated"] is False
    assert (output_dir / "pipeline_config.json").exists()
    payload = json.loads(metrics_output.read_text(encoding="utf-8"))
    assert payload["config"]["split"] == "test"
    assert payload["outputs"]["entities"].endswith("test_t611_entities.json")
