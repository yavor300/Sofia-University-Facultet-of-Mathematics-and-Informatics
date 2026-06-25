import pandas as pd

from gutbrainie.re.atlop_converter import split_to_atlop_examples


def test_split_to_atlop_examples_preserves_offsets_as_token_positions():
    articles = pd.DataFrame(
        [
            {
                "pmid": "1",
                "title": "AAA disease",
                "authors": "",
                "journal": "",
                "year": "",
                "abstract": "BBB affects CCC.",
            }
        ]
    )
    entities = pd.DataFrame(
        [
            {
                "pmid": "1",
                "annotator": "ann",
                "start_idx": 0,
                "end_idx": 3,
                "location": "title",
                "text_span": "AAA",
                "label": "DDF",
            },
            {
                "pmid": "1",
                "annotator": "ann",
                "start_idx": 0,
                "end_idx": 3,
                "location": "abstract",
                "text_span": "BBB",
                "label": "bacteria",
            },
        ]
    )
    relations = pd.DataFrame(
        [
            {
                "pmid": "1",
                "annotator": "ann",
                "subject_start_idx": 0,
                "subject_end_idx": 3,
                "subject_location": "abstract",
                "subject_text_span": "BBB",
                "subject_label": "bacteria",
                "predicate": "influence",
                "object_start_idx": 0,
                "object_end_idx": 3,
                "object_location": "title",
                "object_text_span": "AAA",
                "object_label": "DDF",
            }
        ]
    )

    examples = split_to_atlop_examples(articles, entities, relations)

    assert len(examples) == 1
    example = examples[0]
    assert example["title"] == "1"
    assert example["sents"] == [["AAA", "disease"], ["BBB", "affects", "CCC."]]
    assert example["vertexSet"][0][0]["pos"] == [0, 1]
    assert example["vertexSet"][1][0]["pos"] == [0, 1]
    assert example["labels"] == [{"h": 1, "t": 0, "r": "INFLUENCE", "evidence": [0, 1]}]


def test_split_to_atlop_examples_can_omit_labels_for_prediction():
    articles = pd.DataFrame(
        [{"pmid": "1", "title": "AAA", "authors": "", "journal": "", "year": "", "abstract": ""}]
    )
    entities = pd.DataFrame(
        [
            {
                "pmid": "1",
                "start_idx": 0,
                "end_idx": 3,
                "location": "title",
                "text_span": "AAA",
                "label": "DDF",
            }
        ]
    )

    examples = split_to_atlop_examples(articles, entities, pd.DataFrame(), include_labels=False)

    assert "labels" not in examples[0]
    assert examples[0]["vertexSet"][0][0]["name"] == "AAA"
