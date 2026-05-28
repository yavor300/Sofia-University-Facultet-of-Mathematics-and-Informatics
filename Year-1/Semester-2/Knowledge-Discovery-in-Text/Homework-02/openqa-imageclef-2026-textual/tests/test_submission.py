import json
import zipfile

from openqa_textual.submission import (
    create_submission_records,
    create_submission_zip,
    make_submission,
    validate_submission_file,
    validate_submission_records,
)


def test_create_submission_records_strips_debug_and_normalizes_answers() -> None:
    records = create_submission_records(
        [
            {
                "question_id": "q-1",
                "answers": [1, None],
                "language": "English",
                "debug": {"ocr_text": "secret"},
            }
        ]
    )

    assert records == [{"question_id": "q-1", "answers": ["1", ""], "language": "English"}]


def test_make_submission_writes_final_shape(tmp_path) -> None:
    prediction_path = tmp_path / "predictions.json"
    output_path = tmp_path / "submission.json"
    prediction_path.write_text(
        json.dumps([{"question_id": "q-1", "answer": "A", "language": "English", "debug": {}}]),
        encoding="utf-8",
    )

    submission = make_submission(prediction_path, output_path)

    assert submission == [{"question_id": "q-1", "answers": ["A"], "language": "English"}]
    assert json.loads(output_path.read_text(encoding="utf-8")) == submission


def test_validate_submission_records_accepts_valid_submission() -> None:
    result = validate_submission_records(
        [{"question_id": "q-1", "answers": ["A"], "language": "English"}],
        expected_size=1,
        expected_ids={"q-1"},
    )

    assert result.valid
    assert result.errors == []


def test_validate_submission_records_rejects_debug_duplicates_and_nulls() -> None:
    result = validate_submission_records(
        [
            {"question_id": "q-1", "answers": ["A"], "language": "English", "debug": {}},
            {"question_id": "q-1", "answers": [None], "language": None},
        ],
        expected_size=2,
    )

    assert not result.valid
    assert any("debug" in error for error in result.errors)
    assert any("Duplicate question_id" in error for error in result.errors)
    assert any("must not be null" in error for error in result.errors)


def test_validate_submission_file_rejects_zip_with_multiple_json_files(tmp_path) -> None:
    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("predictions.json", "[]")
        archive.writestr("extra.json", "[]")

    result = validate_submission_file(zip_path)

    assert not result.valid
    assert "exactly one JSON file" in result.errors[0]


def test_create_submission_zip_contains_one_json_file(tmp_path) -> None:
    submission_path = tmp_path / "predictions.json"
    zip_path = tmp_path / "openqa_textual_submission.zip"
    submission_path.write_text(
        json.dumps([{"question_id": "q-1", "answers": ["A"], "language": "English"}]),
        encoding="utf-8",
    )

    create_submission_zip(submission_path, zip_path, arcname="predictions.json")

    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["predictions.json"]
    assert validate_submission_file(zip_path, expected_size=1).valid
