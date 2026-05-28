"""Submission generation and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import zipfile

from openqa_textual.data import get_sample_id
from openqa_textual.evaluation import load_prediction_file
from openqa_textual.prediction import write_json


REQUIRED_SUBMISSION_FIELDS = ("question_id", "answers", "language")


@dataclass(frozen=True)
class SubmissionValidationResult:
    """Validation result for final submission files."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    total: int


def create_submission_records(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip internal/debug fields and normalize prediction rows for submission."""

    records = []
    for prediction in predictions:
        question_id = str(prediction.get("question_id") or prediction.get("id") or "")
        language = str(prediction.get("language") or "English")
        answers = prediction.get("answers")
        if not isinstance(answers, list):
            answer = prediction.get("answer")
            answers = ["" if answer is None else str(answer)]
        else:
            answers = ["" if answer is None else str(answer) for answer in answers]
        records.append(
            {
                "question_id": question_id,
                "answers": answers,
                "language": language,
            }
        )
    return records


def make_submission(prediction_path: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    """Create and write a submission JSON file from internal predictions."""

    predictions = load_prediction_file(prediction_path)
    submission = create_submission_records(predictions)
    write_json(output_path, submission)
    return submission


def create_submission_zip(
    submission_path: str | Path,
    output_path: str | Path,
    arcname: str | None = None,
) -> Path:
    """Create a ZIP archive containing exactly one submission JSON file."""

    source = Path(submission_path)
    if source.suffix.lower() != ".json":
        raise ValueError(f"Submission file must be a JSON file: {source}")
    destination = Path(output_path)
    archive_name = arcname or source.name
    if not archive_name.lower().endswith(".json"):
        raise ValueError(f"ZIP member name must end with .json: {archive_name}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(source, arcname=archive_name)
    return destination


def validate_submission_records(
    records: Any,
    expected_size: int | None = None,
    expected_ids: set[str] | None = None,
) -> SubmissionValidationResult:
    """Validate loaded submission records."""

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(records, list):
        return SubmissionValidationResult(
            valid=False,
            errors=["Submission top-level value must be a list."],
            warnings=[],
            total=0,
        )

    if expected_size is not None and len(records) != expected_size:
        errors.append(f"Expected {expected_size} records, found {len(records)}.")

    seen_ids: set[str] = set()
    for index, item in enumerate(records):
        prefix = f"record[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object.")
            continue
        null_paths = _null_paths(item)
        for path in null_paths:
            errors.append(f"{prefix}.{path} must not be null.")
        if "debug" in item:
            errors.append(f"{prefix} contains forbidden debug field.")
        for field in REQUIRED_SUBMISSION_FIELDS:
            if field not in item:
                errors.append(f"{prefix} missing required field '{field}'.")

        question_id = item.get("question_id")
        if isinstance(question_id, str) and question_id.strip():
            if question_id in seen_ids:
                errors.append(f"Duplicate question_id: {question_id}")
            seen_ids.add(question_id)
        elif "question_id" in item:
            errors.append(f"{prefix}.question_id must be a non-empty string.")

        answers = item.get("answers")
        if not isinstance(answers, list):
            if "answers" in item:
                errors.append(f"{prefix}.answers must be a list.")
        else:
            for answer_index, answer in enumerate(answers):
                if not isinstance(answer, str):
                    errors.append(f"{prefix}.answers[{answer_index}] must be a string.")

        language = item.get("language")
        if "language" in item and (not isinstance(language, str) or not language.strip()):
            errors.append(f"{prefix}.language must be a non-empty string.")

    if expected_ids is not None:
        missing = sorted(expected_ids - seen_ids)
        extra = sorted(seen_ids - expected_ids)
        if missing:
            errors.append(f"Missing {len(missing)} expected question_id values.")
            warnings.append(f"First missing question_id values: {missing[:5]}")
        if extra:
            errors.append(f"Found {len(extra)} unexpected question_id values.")
            warnings.append(f"First unexpected question_id values: {extra[:5]}")

    return SubmissionValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        total=len(records),
    )


def validate_submission_file(
    submission_path: str | Path,
    expected_size: int | None = None,
    expected_ids: set[str] | None = None,
) -> SubmissionValidationResult:
    """Load and validate a JSON submission or a ZIP containing one JSON file."""

    path = Path(submission_path)
    try:
        records = load_submission_records(path)
    except Exception as exc:
        return SubmissionValidationResult(
            valid=False,
            errors=[str(exc)],
            warnings=[],
            total=0,
        )
    return validate_submission_records(records, expected_size=expected_size, expected_ids=expected_ids)


def load_submission_records(path: str | Path) -> Any:
    """Load submission records from JSON or ZIP with exactly one JSON file."""

    submission_path = Path(path)
    if submission_path.suffix.lower() == ".zip":
        return _load_submission_zip(submission_path)
    with submission_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expected_ids_from_split(dataset_split: Any) -> set[str]:
    """Return all expected question IDs from a dataset split."""

    ids: set[str] = set()
    for index in range(len(dataset_split)):
        sample = dataset_split[index]
        try:
            ids.add(get_sample_id(sample))
        except KeyError:
            ids.add(f"sample-{index:05d}")
    return ids


def _load_submission_zip(path: Path) -> Any:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        json_names = [name for name in names if name.lower().endswith(".json")]
        if len(json_names) != 1:
            raise ValueError(f"ZIP must contain exactly one JSON file, found {len(json_names)}.")
        with archive.open(json_names[0]) as handle:
            return json.loads(handle.read().decode("utf-8"))


def _null_paths(value: Any, prefix: str = "") -> list[str]:
    if value is None:
        return [prefix or "<root>"]
    if isinstance(value, dict):
        paths: list[str] = []
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_null_paths(item, child_prefix))
        return paths
    if isinstance(value, list):
        paths = []
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            paths.extend(_null_paths(item, child_prefix))
        return paths
    return []
