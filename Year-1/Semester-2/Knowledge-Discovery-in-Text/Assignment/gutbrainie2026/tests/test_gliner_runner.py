import json

import pandas as pd

import gutbrainie.ner.gliner_runner as gliner_runner
from gutbrainie.ner.gliner_runner import (
    _gliner_predictions_to_rows,
    convert_to_gliner_examples,
    gliner_examples_to_training_dataset,
    prepare_gliner_experiment_data,
    read_jsonl,
    split_articles_by_pmid,
    train_gliner_model,
    write_jsonl,
)


def test_convert_to_gliner_examples_keeps_title_and_abstract_offsets_separate():
    articles = pd.DataFrame(
        [
            {
                "pmid": "1",
                "title": "Gut brain",
                "abstract": "Microbes affect neurons.",
            }
        ]
    )
    entities = pd.DataFrame(
        [
            {"pmid": "1", "start_idx": 0, "end_idx": 3, "location": "title", "text_span": "Gut", "label": "DDF"},
            {
                "pmid": "1",
                "start_idx": 0,
                "end_idx": 8,
                "location": "abstract",
                "text_span": "Microbes",
                "label": "bacteria",
            },
        ]
    )

    examples = convert_to_gliner_examples(articles, entities)

    assert examples == [
        {"pmid": "1", "location": "title", "text": "Gut brain", "label": [[0, 3, "DDF"]]},
        {
            "pmid": "1",
            "location": "abstract",
            "text": "Microbes affect neurons.",
            "label": [[0, 8, "bacteria"]],
        },
    ]


def test_split_articles_by_pmid_is_deterministic():
    articles = pd.DataFrame(
        [{"pmid": str(index), "title": "", "abstract": ""} for index in range(10)]
    )

    first_train, first_val = split_articles_by_pmid(articles, validation_fraction=0.2, seed=7)
    second_train, second_val = split_articles_by_pmid(articles, validation_fraction=0.2, seed=7)

    assert list(first_train["pmid"]) == list(second_train["pmid"])
    assert list(first_val["pmid"]) == list(second_val["pmid"])
    assert len(first_val) == 2


def test_gliner_prediction_rows_remove_overlaps_by_score():
    rows = _gliner_predictions_to_rows(
        pmid="1",
        location="title",
        text="Gut microbiota",
        predictions=[
            {"start": 0, "end": 14, "label": "microbiome", "score": 0.8},
            {"start": 4, "end": 14, "label": "microbiome", "score": 0.9},
        ],
    )

    assert rows == [
        {
            "pmid": "1",
            "start_idx": 4,
            "end_idx": 13,
            "location": "title",
            "text_span": "microbiota",
            "label": "microbiome",
        }
    ]


def test_gliner_examples_to_training_dataset_converts_to_word_spans():
    dataset = gliner_examples_to_training_dataset(
        [
            {
                "text": "Gut microbiota affects mood",
                "label": [[0, 14, "microbiome"], [23, 27, "DDF"], [1, 5, "bad"]],
            }
        ]
    )

    assert dataset == [
        {
            "tokenized_text": ["Gut", "microbiota", "affects", "mood"],
            "ner": [(0, 1, "microbiome"), (3, 3, "DDF")],
        }
    ]


def test_gliner_examples_to_training_dataset_drops_empty_examples_by_default():
    examples = [
        {"text": "No annotations here", "label": []},
        {"text": "Misaligned span", "label": [[1, 5, "DDF"]]},
        {"text": "Gut brain", "label": [[0, 3, "DDF"]]},
    ]

    dataset = gliner_examples_to_training_dataset(examples)
    dataset_with_empty = gliner_examples_to_training_dataset(examples, keep_empty=True)

    assert dataset == [{"tokenized_text": ["Gut", "brain"], "ner": [(0, 0, "DDF")]}]
    assert len(dataset_with_empty) == 3


def test_prepare_gliner_experiment_data_writes_jsonl_and_metadata(tmp_path):
    data_root = tmp_path / "gutbrainie2026"
    output_dir = tmp_path / "gliner"
    _write_gold_fixture(data_root)

    metadata = prepare_gliner_experiment_data(data_root, "gold", output_dir, validation_fraction=0.5, seed=3)

    train_examples = read_jsonl(metadata["train_path"])
    validation_examples = read_jsonl(metadata["validation_path"])
    metadata_payload = json.loads((output_dir / "gliner_gold_metadata.json").read_text(encoding="utf-8"))

    assert len(train_examples) == 2
    assert len(validation_examples) == 2
    assert metadata_payload["experiment"] == "gold"
    assert metadata_payload["gold_train_articles"] == 1
    assert metadata_payload["gold_validation_articles"] == 1


def test_train_gliner_model_uses_positional_train_and_eval_datasets(tmp_path, monkeypatch):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    output_dir = tmp_path / "model"
    config_path = tmp_path / "config.yaml"
    write_jsonl([{"text": "Gut brain", "label": [[0, 3, "DDF"]]}], train_path)
    write_jsonl([{"text": "Butyrate response", "label": [[0, 8, "chemical"]]}], validation_path)
    config_path.write_text("learning_rate: 0.001\nbatch_size: 2\nepochs: 1\nmax_steps: 1\n", encoding="utf-8")

    captured = {}

    class FakeModel:
        def train_model(self, train_dataset, eval_dataset, **kwargs):
            captured["train_dataset"] = train_dataset
            captured["eval_dataset"] = eval_dataset
            captured["kwargs"] = kwargs

        def save_pretrained(self, path):
            captured["save_path"] = path

    class FakeGLiNER:
        @staticmethod
        def from_pretrained(model_name):
            captured["model_name"] = model_name
            return FakeModel()

    monkeypatch.setattr(gliner_runner, "_import_gliner_model", lambda: FakeGLiNER)

    result = train_gliner_model("fake-model", train_path, validation_path, output_dir, config_path)

    assert result == output_dir
    assert captured["model_name"] == "fake-model"
    assert captured["train_dataset"] == [{"tokenized_text": ["Gut", "brain"], "ner": [(0, 0, "DDF")]}]
    assert captured["eval_dataset"] == [{"tokenized_text": ["Butyrate", "response"], "ner": [(0, 0, "chemical")]}]
    assert captured["kwargs"]["output_dir"] == output_dir
    assert captured["kwargs"]["per_device_train_batch_size"] == 2
    assert captured["kwargs"]["max_steps"] == 1
    assert captured["save_path"] == str(output_dir)


def _write_gold_fixture(data_root):
    article_dir = data_root / "Articles" / "csv_format"
    annotation_dir = data_root / "Annotations" / "Train" / "gold_quality" / "csv_format"
    article_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)

    (article_dir / "articles_train_gold.csv").write_text(
        "pmid|title|authors|journal|year|abstract\n"
        "1|Gut brain|A Author|Journal|2026|Microbes affect neurons.\n"
        "2|Butyrate response|B Author|Journal|2026|Butyrate changes inflammation.\n",
        encoding="utf-8",
    )
    (annotation_dir / "train_gold_entities.csv").write_text(
        "pmid|annotator|start_idx|end_idx|location|text_span|label\n"
        "1|ann|0|3|title|Gut|DDF\n"
        "1|ann|0|8|abstract|Microbes|bacteria\n"
        "2|ann|0|8|title|Butyrate|chemical\n",
        encoding="utf-8",
    )
    (annotation_dir / "train_gold_mention_level_relations.csv").write_text(
        "pmid|annotator|subject_text_span|subject_label|predicate|object_text_span|object_label\n",
        encoding="utf-8",
    )
    (annotation_dir / "train_gold_relations.csv").write_text(
        "pmid|annotator|subject_start_idx|subject_end_idx|subject_location|subject_text_span|"
        "subject_label|predicate|object_start_idx|object_end_idx|object_location|object_text_span|object_label\n",
        encoding="utf-8",
    )
