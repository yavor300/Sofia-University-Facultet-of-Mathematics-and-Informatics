from openqa_textual.training_data import (
    SYSTEM_PROMPT,
    build_clean_question_training_records,
    build_ocr_training_records,
    build_training_record,
    gold_answers_from_dataset_split,
)


def test_build_training_record_uses_chat_message_format() -> None:
    record = build_training_record("What is 2 + 2?", "4", language="English")

    assert record == {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Language: English\nQuestion: What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ]
    }


def test_build_ocr_training_records_uses_ocr_text_and_dataset_gold() -> None:
    records = build_ocr_training_records(
        [
            {
                "question_id": "q-1",
                "language": "Bulgarian",
                "clean_question": "Кои отпадъци се разпадат най-\nтрудно?",
            }
        ],
        gold_by_id={"q-1": "пластмасови"},
    )

    assert records[0]["messages"][1]["content"] == (
        "Language: Bulgarian\nQuestion: Кои отпадъци се разпадат най-трудно?"
    )
    assert records[0]["messages"][2]["content"] == "пластмасови"


def test_build_ocr_training_records_skips_missing_answer_by_default() -> None:
    assert build_ocr_training_records([{"question_id": "q-1", "clean_question": "Q?"}]) == []


def test_build_clean_question_training_records_is_upper_bound_variant() -> None:
    records = build_clean_question_training_records(
        [{"question_id": "q-1", "language": "English", "question": "Clean Q?", "answer": "A"}]
    )

    assert records[0]["messages"][1]["content"] == "Language: English\nQuestion: Clean Q?"
    assert records[0]["messages"][2]["content"] == "A"


def test_gold_answers_from_dataset_split() -> None:
    assert gold_answers_from_dataset_split(
        [{"question_id": "q-1", "answers": ["A", "B"]}]
    ) == {"q-1": "A | B"}
