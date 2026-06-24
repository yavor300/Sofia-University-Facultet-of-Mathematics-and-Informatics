import pandas as pd

from gutbrainie.ner.bio_tags import (
    IGNORE_INDEX,
    align_entities_to_bio_ids,
    articles_entities_to_token_features,
    build_bio_label_list,
    build_label_maps,
    decode_bio_spans,
)
from gutbrainie.ner.train_token_classifier import TokenClassificationDataset


def test_align_entities_to_bio_ids_accepts_inclusive_end_offsets():
    label_to_id, _ = build_label_maps(build_bio_label_list(["DDF", "microbiome"]))
    text = "Gut microbiota affects mood"
    offsets = [(0, 0), (0, 3), (4, 14), (15, 22), (23, 27), (0, 0)]
    entities = pd.DataFrame(
        [
            {
                "pmid": "1",
                "start_idx": 0,
                "end_idx": 13,
                "location": "title",
                "text_span": "Gut microbiota",
                "label": "microbiome",
            },
            {
                "pmid": "1",
                "start_idx": 23,
                "end_idx": 26,
                "location": "title",
                "text_span": "mood",
                "label": "DDF",
            },
        ]
    )

    labels = align_entities_to_bio_ids(text, offsets, entities, label_to_id)

    assert labels == [
        IGNORE_INDEX,
        label_to_id["B-microbiome"],
        label_to_id["I-microbiome"],
        label_to_id["O"],
        label_to_id["B-DDF"],
        IGNORE_INDEX,
    ]


def test_decode_bio_spans_returns_inclusive_challenge_offsets():
    label_to_id, id_to_label = build_label_maps(build_bio_label_list(["DDF", "microbiome"]))
    text = "Gut microbiota affects mood"
    offsets = [(0, 0), (0, 3), (4, 14), (15, 22), (23, 27), (0, 0)]
    label_ids = [
        IGNORE_INDEX,
        label_to_id["B-microbiome"],
        label_to_id["I-microbiome"],
        label_to_id["O"],
        label_to_id["B-DDF"],
        IGNORE_INDEX,
    ]

    spans = decode_bio_spans(text, offsets, label_ids, id_to_label)

    assert spans == [
        {"start_idx": 0, "end_idx": 13, "text_span": "Gut microbiota", "label": "microbiome"},
        {"start_idx": 23, "end_idx": 26, "text_span": "mood", "label": "DDF"},
    ]


def test_articles_entities_to_token_features_keeps_title_and_abstract_separate():
    label_to_id, _ = build_label_maps(build_bio_label_list(["bacteria", "microbiome"]))
    articles = pd.DataFrame(
        [
            {
                "pmid": "1",
                "title": "Gut microbiota",
                "abstract": "Microbes affect neurons",
            }
        ]
    )
    entities = pd.DataFrame(
        [
            {
                "pmid": "1",
                "start_idx": 0,
                "end_idx": 13,
                "location": "title",
                "text_span": "Gut microbiota",
                "label": "microbiome",
            },
            {
                "pmid": "1",
                "start_idx": 0,
                "end_idx": 7,
                "location": "abstract",
                "text_span": "Microbes",
                "label": "bacteria",
            },
        ]
    )

    features = articles_entities_to_token_features(articles, entities, FakeTokenizer(), label_to_id, max_length=32)

    assert [feature["location"] for feature in features] == ["title", "abstract"]
    assert features[0]["labels"][1:3] == [label_to_id["B-microbiome"], label_to_id["I-microbiome"]]
    assert features[1]["labels"][1] == label_to_id["B-bacteria"]


def test_token_classification_dataset_drops_metadata_fields():
    dataset = TokenClassificationDataset(
        [
            {
                "pmid": "1",
                "location": "title",
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 0, -100],
            }
        ]
    )

    assert dataset[0] == {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "labels": [-100, 0, -100],
    }


class FakeTokenizer:
    def __call__(self, text, truncation=True, max_length=512, return_offsets_mapping=True):
        del truncation, max_length, return_offsets_mapping
        offsets = [(0, 0)]
        cursor = 0
        for token in text.split():
            start = text.index(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        offsets.append((0, 0))
        return {
            "input_ids": list(range(len(offsets))),
            "attention_mask": [1] * len(offsets),
            "offset_mapping": offsets,
        }
