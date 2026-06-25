import pandas as pd

from gutbrainie.re.train_pair_classifier import (
    NO_RELATION,
    build_relation_label_list,
    candidate_to_marked_text,
    marker_tokens,
    sample_negative_examples,
)


def test_candidate_to_marked_text_same_location():
    article = {"title": "Unused", "abstract": "AAA causes BBB"}
    candidate = {
        "subject_location": "abstract",
        "object_location": "abstract",
        "subject_start_idx": 0,
        "subject_end_idx": 2,
        "subject_label": "bacteria",
        "subject_text_span": "AAA",
        "object_start_idx": 11,
        "object_end_idx": 13,
        "object_label": "DDF",
        "object_text_span": "BBB",
    }

    marked = candidate_to_marked_text(article, candidate)

    assert marked == "[SUBJ_BACTERIA] AAA [/SUBJ_BACTERIA] causes [OBJ_DDF] BBB [/OBJ_DDF]"


def test_candidate_to_marked_text_cross_location():
    article = {"title": "AAA title", "abstract": "BBB abstract"}
    candidate = {
        "subject_location": "title",
        "object_location": "abstract",
        "subject_start_idx": 0,
        "subject_end_idx": 2,
        "subject_label": "bacteria",
        "subject_text_span": "AAA",
        "object_start_idx": 0,
        "object_end_idx": 2,
        "object_label": "DDF",
        "object_text_span": "BBB",
    }

    marked = candidate_to_marked_text(article, candidate)

    assert marked == "[SUBJ_BACTERIA] AAA [/SUBJ_BACTERIA] title [SEP] [OBJ_DDF] BBB [/OBJ_DDF] abstract"


def test_sample_negative_examples_keeps_all_positives_and_caps_negatives():
    examples = [
        {"label": "influence", "candidate": {"pmid": "1", "subject_text_span": "A"}},
        {"label": "target", "candidate": {"pmid": "1", "subject_text_span": "B"}},
        {"label": NO_RELATION, "candidate": {"pmid": "1", "subject_text_span": "C"}},
        {"label": NO_RELATION, "candidate": {"pmid": "1", "subject_text_span": "D"}},
        {"label": NO_RELATION, "candidate": {"pmid": "1", "subject_text_span": "E"}},
    ]

    sampled = sample_negative_examples(examples, negative_sampling_ratio=1, seed=7)

    assert sum(example["label"] != NO_RELATION for example in sampled) == 2
    assert sum(example["label"] == NO_RELATION for example in sampled) == 2


def test_relation_label_list_has_no_relation_first_and_schema_predicates():
    relations = pd.DataFrame({"predicate": ["influence", "target"]})

    labels = build_relation_label_list(relations)

    assert labels[0] == NO_RELATION
    assert "influence" in labels
    assert "target" in labels


def test_marker_tokens_cover_subject_and_object_entity_labels():
    tokens = marker_tokens()

    assert "[SUBJ_BACTERIA]" in tokens
    assert "[/SUBJ_BACTERIA]" in tokens
    assert "[OBJ_STATISTICAL_TECHNIQUE]" in tokens
    assert "[/OBJ_STATISTICAL_TECHNIQUE]" in tokens
