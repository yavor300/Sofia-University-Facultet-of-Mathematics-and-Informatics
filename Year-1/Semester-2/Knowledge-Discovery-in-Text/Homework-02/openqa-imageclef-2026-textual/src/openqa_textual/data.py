"""Dataset loading and sample decoding utilities."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
import re
from typing import Any

from PIL import Image, UnidentifiedImageError


ID_FIELD_CANDIDATES = (
    "question_id",
    "id",
    "sample_id",
    "uid",
    "uuid",
    "image_id",
)

LANGUAGE_FIELD_CANDIDATES = (
    "language",
    "lang",
    "question_language",
    "locale",
)

IMAGE_FIELD_CANDIDATES = (
    "image",
    "question_image",
    "question_img",
    "img",
    "image_file",
    "image_path",
    "path",
    "file",
    "bytes",
)

ANSWER_FIELD_CANDIDATES = (
    "gold_answer",
    "gold_answers",
    "answer",
    "answers",
    "label",
    "labels",
    "target",
    "targets",
    "reference",
    "references",
)

QUESTION_FIELD_CANDIDATES = (
    "question",
    "clean_question",
    "question_text",
    "text",
    "prompt",
)


def load_dataset_splits(dataset_name: str, cache_dir: str | None = None):
    """Load train/dev/test splits from Hugging Face.

    The ImageCLEF dataset schema may evolve, so this function leaves split names
    exactly as Hugging Face returns them instead of forcing a local convention.
    Callers can map `validation` to `dev` at the script/config layer.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required to load Hugging Face datasets. "
            "Install the local requirements first."
        ) from exc

    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return load_dataset(dataset_name, **kwargs)


def get_sample_id(sample: Mapping[str, Any]) -> str:
    """Return question_id/id from the sample."""

    for field_name in ID_FIELD_CANDIDATES:
        value = sample.get(field_name)
        if value is not None and str(value).strip():
            return str(value)

    available = ", ".join(sample.keys())
    raise KeyError(
        "Could not find a sample ID field. Tried "
        f"{', '.join(ID_FIELD_CANDIDATES)}. Available fields: {available}"
    )


def get_sample_language(sample: Mapping[str, Any]) -> str:
    """Return language value or a safe default."""

    for field_name in LANGUAGE_FIELD_CANDIDATES:
        value = sample.get(field_name)
        if value is not None and str(value).strip():
            return str(value)
    return "English"


def get_sample_gold_answer(sample: Mapping[str, Any]) -> str:
    """Return a gold answer string from common train/dev answer fields."""

    for field_name in ANSWER_FIELD_CANDIDATES:
        if field_name not in sample:
            continue
        answer = _normalize_answer_value(sample[field_name])
        if answer:
            return answer
    return ""


def get_sample_question_text(sample: Mapping[str, Any]) -> str:
    """Return clean question text if the dataset exposes it."""

    for field_name in QUESTION_FIELD_CANDIDATES:
        value = sample.get(field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def get_sample_image(sample: Mapping[str, Any]) -> Image.Image:
    """Return a PIL image decoded from the dataset sample."""

    for field_name in IMAGE_FIELD_CANDIDATES:
        if field_name in sample:
            image = _try_decode_image(sample[field_name])
            if image is not None:
                return image

    for value in sample.values():
        image = _try_decode_image(value)
        if image is not None:
            return image

    available = ", ".join(sample.keys())
    raise ValueError(
        "Could not decode an image from this sample. Tried common image fields "
        f"first, then all fields. Available fields: {available}"
    )


def save_debug_image_samples(
    dataset_split: Any,
    output_dir: str | Path,
    n: int = 30,
    split_name: str | None = None,
) -> list[dict[str, Any]]:
    """Save decoded sample images for manual inspection.

    Returns a manifest with one entry per saved image. Samples that cannot be
    decoded are skipped and recorded with an error entry.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    limit = min(max(n, 0), len(dataset_split))
    prefix = split_name or "split"

    for index in range(limit):
        sample = dataset_split[index]
        try:
            sample_id = get_sample_id(sample)
        except KeyError:
            sample_id = f"sample-{index:05d}"

        try:
            language = get_sample_language(sample)
            image = get_sample_image(sample)
            safe_id = _safe_filename_part(sample_id)
            filename = f"{index:05d}_{safe_id}.png"
            output_path = destination / filename
            image.save(output_path, format="PNG")
            manifest.append(
                {
                    "index": index,
                    "question_id": sample_id,
                    "language": language,
                    "split": prefix,
                    "path": str(output_path),
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "saved": True,
                }
            )
        except Exception as exc:
            manifest.append(
                {
                    "index": index,
                    "question_id": sample_id,
                    "split": prefix,
                    "saved": False,
                    "error": str(exc),
                }
            )

    return manifest


def _try_decode_image(value: Any) -> Image.Image | None:
    """Best-effort image decoder for common Hugging Face dataset representations."""

    if value is None:
        return None

    if isinstance(value, Image.Image):
        return _normalize_image(value)

    if isinstance(value, (bytes, bytearray, memoryview)):
        return _decode_image_bytes(bytes(value))

    if isinstance(value, (str, Path)):
        return _decode_image_path(Path(value))

    if isinstance(value, Mapping):
        return _decode_image_mapping(value)

    return _decode_array_like(value)


def _decode_image_mapping(value: Mapping[str, Any]) -> Image.Image | None:
    for key in ("image", "question_image", "img"):
        if key in value:
            image = _try_decode_image(value[key])
            if image is not None:
                return image

    raw_bytes = value.get("bytes")
    if raw_bytes:
        image = _try_decode_image(raw_bytes)
        if image is not None:
            return image

    path = value.get("path") or value.get("file") or value.get("filename")
    if path:
        image = _try_decode_image(path)
        if image is not None:
            return image

    return None


def _decode_image_bytes(raw_bytes: bytes) -> Image.Image | None:
    try:
        with Image.open(BytesIO(raw_bytes)) as image:
            return _normalize_image(image)
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _decode_image_path(path: Path) -> Image.Image | None:
    if not path.exists() or not path.is_file():
        return None

    try:
        with Image.open(path) as image:
            return _normalize_image(image)
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _decode_array_like(value: Any) -> Image.Image | None:
    try:
        import numpy as np
    except ImportError:
        return None

    if not isinstance(value, np.ndarray):
        return None

    try:
        return _normalize_image(Image.fromarray(value))
    except (TypeError, ValueError):
        return None


def _normalize_image(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    return image.convert("RGB")


def _normalize_answer_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ANSWER_FIELD_CANDIDATES:
            if key in value:
                answer = _normalize_answer_value(value[key])
                if answer:
                    return answer
        if "text" in value:
            return str(value["text"]).strip()
        return ""
    if isinstance(value, (list, tuple, set)):
        answers = [_normalize_answer_value(item) for item in value]
        answers = [answer for answer in answers if answer]
        return " | ".join(answers)
    return str(value).strip()


def _safe_filename_part(value: str, max_length: int = 80) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (safe or "sample")[:max_length]
