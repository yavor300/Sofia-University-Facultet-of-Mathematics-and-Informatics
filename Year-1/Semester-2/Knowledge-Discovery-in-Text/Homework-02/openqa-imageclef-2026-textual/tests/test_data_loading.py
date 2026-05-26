from io import BytesIO

import pytest
from PIL import Image

from openqa_textual.data import (
    get_sample_id,
    get_sample_image,
    get_sample_language,
    save_debug_image_samples,
)


def _png_bytes() -> bytes:
    image = Image.new("RGB", (12, 8), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_get_sample_id_prefers_question_id() -> None:
    assert get_sample_id({"question_id": "q-1", "id": "fallback"}) == "q-1"


def test_get_sample_id_uses_id_fallback() -> None:
    assert get_sample_id({"id": 42}) == "42"


def test_get_sample_id_raises_for_missing_id() -> None:
    with pytest.raises(KeyError):
        get_sample_id({"question": "What is shown?"})


def test_get_sample_language_reads_language() -> None:
    assert get_sample_language({"language": "Bulgarian"}) == "Bulgarian"


def test_get_sample_language_defaults_to_english() -> None:
    assert get_sample_language({"question_id": "q-1"}) == "English"


def test_get_sample_image_from_pil() -> None:
    source = Image.new("L", (4, 3), color=255)
    decoded = get_sample_image({"image": source})
    assert decoded.mode == "RGB"
    assert decoded.size == (4, 3)


def test_get_sample_image_from_bytes() -> None:
    decoded = get_sample_image({"image": {"bytes": _png_bytes(), "path": None}})
    assert decoded.mode == "RGB"
    assert decoded.size == (12, 8)


def test_get_sample_image_from_path(tmp_path) -> None:
    image_path = tmp_path / "question.png"
    Image.new("RGB", (5, 6), color=(0, 0, 0)).save(image_path)
    decoded = get_sample_image({"image_path": str(image_path)})
    assert decoded.size == (5, 6)


def test_get_sample_image_scans_unknown_image_field() -> None:
    decoded = get_sample_image({"question_id": "q-1", "custom_payload": _png_bytes()})
    assert decoded.size == (12, 8)


def test_get_sample_image_raises_when_no_image_can_be_decoded() -> None:
    with pytest.raises(ValueError):
        get_sample_image({"question_id": "q-1", "question": "plain text only"})


def test_save_debug_image_samples_writes_pngs_and_manifest_data(tmp_path) -> None:
    samples = [
        {"question_id": "q/1", "language": "English", "image": _png_bytes()},
        {"id": "q-2", "lang": "French", "image": Image.new("RGB", (3, 4))},
    ]

    manifest = save_debug_image_samples(samples, tmp_path, n=5, split_name="train")

    assert len(manifest) == 2
    assert all(row["saved"] for row in manifest)
    assert manifest[0]["question_id"] == "q/1"
    assert manifest[0]["split"] == "train"
    assert manifest[0]["width"] == 12
    assert (tmp_path / "00000_q_1.png").exists()
    assert (tmp_path / "00001_q-2.png").exists()


def test_save_debug_image_samples_records_decode_errors(tmp_path) -> None:
    manifest = save_debug_image_samples([{"question_id": "q-1", "image": "missing.png"}], tmp_path)

    assert manifest == [
        {
            "index": 0,
            "question_id": "q-1",
            "split": "split",
            "saved": False,
            "error": "Could not decode an image from this sample. Tried common image fields first, then all fields. Available fields: question_id, image",
        }
    ]
