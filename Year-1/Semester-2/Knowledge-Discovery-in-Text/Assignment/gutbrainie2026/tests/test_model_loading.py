from gutbrainie.ner.model_loading import load_token_classification_model


def test_load_token_classification_model_falls_back_to_bert_for_old_biobert_config():
    calls = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls.append(("auto", model_name, kwargs))
            raise ValueError("Unrecognized model in dmis-lab/biobert-base-cased-v1.1. Should have a `model_type` key in its config.json.")

    class FakeBertConfig:
        @staticmethod
        def from_pretrained(model_name):
            calls.append(("config", model_name))
            return FakeBertConfig()

    class FakeBertForTokenClassification:
        @staticmethod
        def from_pretrained(model_name, config, ignore_mismatched_sizes=False):
            calls.append(("bert", model_name, config.num_labels, config.id2label, config.label2id, ignore_mismatched_sizes))
            return "bert-model"

    class FakeTransformers:
        AutoModelForTokenClassification = FakeAutoModel
        BertConfig = FakeBertConfig
        BertForTokenClassification = FakeBertForTokenClassification

    model = load_token_classification_model(
        FakeTransformers,
        "dmis-lab/biobert-base-cased-v1.1",
        num_labels=2,
        id2label={0: "O", 1: "B-DDF"},
        label2id={"O": 0, "B-DDF": 1},
        ignore_mismatched_sizes=True,
    )

    assert model == "bert-model"
    assert calls[0][0] == "auto"
    assert calls[1] == ("config", "dmis-lab/biobert-base-cased-v1.1")
    assert calls[2] == (
        "bert",
        "dmis-lab/biobert-base-cased-v1.1",
        2,
        {0: "O", 1: "B-DDF"},
        {"O": 0, "B-DDF": 1},
        True,
    )
