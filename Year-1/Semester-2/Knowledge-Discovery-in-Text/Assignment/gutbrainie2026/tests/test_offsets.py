from gutbrainie.data.offsets import validate_entity_offsets


def test_validate_entity_offsets_uses_location_specific_text():
    article = {"title": "Gut brain axis", "abstract": "Microbes affect neurons."}

    assert validate_entity_offsets(
        article,
        {"location": "title", "start_idx": 0, "end_idx": 3, "text_span": "Gut"},
    )
    assert not validate_entity_offsets(
        article,
        {"location": "abstract", "start_idx": 0, "end_idx": 3, "text_span": "Gut"},
    )


def test_validate_entity_offsets_accepts_inclusive_end_offsets():
    article = {"title": "Gut brain axis", "abstract": ""}

    assert validate_entity_offsets(
        article,
        {"location": "title", "start_idx": 0, "end_idx": 2, "text_span": "Gut"},
    )
