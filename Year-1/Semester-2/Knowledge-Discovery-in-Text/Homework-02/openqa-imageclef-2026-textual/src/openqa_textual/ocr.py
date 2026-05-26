"""OCR engine interfaces with optional, lazily imported implementations."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
from typing import Any

from PIL import Image

from openqa_textual.image_utils import to_rgb


LANGUAGE_TO_EASYOCR = {
    "English": ["en"],
    "Bulgarian": ["bg", "en"],
    "Chinese": ["ch_sim", "en"],
    "Croatian": ["hr", "en"],
    "German": ["de", "en"],
    "French": ["fr", "en"],
    "Italian": ["it", "en"],
    "Serbian": ["rs_latin", "en"],
    "Spanish": ["es", "en"],
}

LANGUAGE_TO_TESSERACT = {
    "English": "eng",
    "Bulgarian": "bul+eng",
    "Chinese": "chi_sim+eng",
    "Croatian": "hrv+eng",
    "German": "deu+eng",
    "French": "fra+eng",
    "Italian": "ita+eng",
    "Serbian": "srp+srp_latn+eng",
    "Spanish": "spa+eng",
}

LANGUAGE_TO_PADDLEOCR = {
    "English": "en",
    "Bulgarian": "en",
    "Chinese": "ch",
    "Croatian": "en",
    "German": "german",
    "French": "fr",
    "Italian": "it",
    "Serbian": "en",
    "Spanish": "es",
}

DEFAULT_EASYOCR_LANGUAGES = ["en", "bg", "hr", "it", "rs_latin"]
DEFAULT_TESSERACT_LANGUAGE = "eng+bul+hrv+ita+srp+srp_latn"
DEFAULT_PADDLEOCR_LANGUAGE = "en"

LANGUAGE_ALIASES = {
    "en": "English",
    "eng": "English",
    "english": "English",
    "bg": "Bulgarian",
    "bul": "Bulgarian",
    "bulgarian": "Bulgarian",
    "български": "Bulgarian",
    "zh": "Chinese",
    "zho": "Chinese",
    "chi": "Chinese",
    "chinese": "Chinese",
    "中文": "Chinese",
    "hr": "Croatian",
    "hrv": "Croatian",
    "croatian": "Croatian",
    "hrvatski": "Croatian",
    "de": "German",
    "deu": "German",
    "ger": "German",
    "german": "German",
    "deutsch": "German",
    "fr": "French",
    "fra": "French",
    "fre": "French",
    "french": "French",
    "français": "French",
    "it": "Italian",
    "ita": "Italian",
    "italian": "Italian",
    "italiano": "Italian",
    "sr": "Serbian",
    "srp": "Serbian",
    "serbian": "Serbian",
    "српски": "Serbian",
    "es": "Spanish",
    "spa": "Spanish",
    "spanish": "Spanish",
    "español": "Spanish",
}


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence: float | None
    engine: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OCRCacheRecord:
    question_id: str
    language: str
    ocr_engine: str
    preprocess_variant: str
    ocr_text: str
    confidence: float | None
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_result(self) -> OCRResult:
        metadata = dict(self.metadata)
        metadata["cache_hit"] = True
        return OCRResult(
            text=self.ocr_text,
            confidence=self.confidence,
            engine=self.ocr_engine,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "language": self.language,
            "ocr_engine": self.ocr_engine,
            "preprocess_variant": self.preprocess_variant,
            "ocr_text": self.ocr_text,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OCRCacheRecord":
        return cls(
            question_id=str(data["question_id"]),
            language=str(data.get("language") or "English"),
            ocr_engine=str(data["ocr_engine"]),
            preprocess_variant=str(data["preprocess_variant"]),
            ocr_text=str(data.get("ocr_text") or ""),
            confidence=data.get("confidence"),
            created_at=str(data["created_at"]),
            metadata=dict(data.get("metadata") or {}),
        )


class OCREngine:
    """Common OCR engine interface."""

    name = "base"

    def extract(self, image: Any) -> OCRResult:
        """Extract question text from an image-like object."""
        raise NotImplementedError


class MissingOCREngineDependency(RuntimeError):
    """Raised when a configured OCR engine is used without its optional package."""


class TesseractOCREngine(OCREngine):
    name = "tesseract"

    def __init__(self, lang: str = "eng", config: str = "--psm 6", cmd: str | None = None) -> None:
        self.lang = lang
        self.config = config
        self.cmd = cmd or os.getenv("TESSERACT_CMD") or None

    def extract(self, image: Any) -> OCRResult:
        try:
            import pytesseract
        except ImportError as exc:
            raise MissingOCREngineDependency(
                "pytesseract is not installed. Install requirements-ocr.txt or disable tesseract."
            ) from exc

        if self.cmd:
            pytesseract.pytesseract.tesseract_cmd = self.cmd

        text = pytesseract.image_to_string(_image_for_tesseract(image), lang=self.lang, config=self.config)
        return OCRResult(
            text=text.strip(),
            confidence=None,
            engine=self.name,
            metadata={"lang": self.lang, "config": self.config, "cmd": self.cmd or "tesseract"},
        )


class EasyOCREngine(OCREngine):
    name = "easyocr"

    def __init__(self, languages: list[str] | None = None, gpu: bool = True) -> None:
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = None

    def _get_reader(self) -> Any:
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise MissingOCREngineDependency(
                    "easyocr is not installed. Install requirements-ocr.txt or disable easyocr."
                ) from exc
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract(self, image: Any) -> OCRResult:
        reader = self._get_reader()
        rows = reader.readtext(_image_for_array_ocr(image), detail=1, paragraph=False)
        texts: list[str] = []
        confidences: list[float] = []
        for row in rows or []:
            if len(row) >= 2:
                texts.append(str(row[1]))
            if len(row) >= 3:
                confidences.append(float(row[2]))
        return OCRResult(
            text=" ".join(texts).strip(),
            confidence=_mean_confidence(confidences),
            engine=self.name,
            metadata={"languages": self.languages, "gpu": self.gpu, "boxes": len(rows or [])},
        )


class PaddleOCREngine(OCREngine):
    name = "paddleocr"

    def __init__(self, lang: str = "en", use_gpu: bool = True) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self._ocr = None

    def _get_ocr(self) -> Any:
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise MissingOCREngineDependency(
                    "paddleocr is not installed. Install requirements-ocr.txt or disable paddleocr."
                ) from exc
            self._ocr = PaddleOCR(lang=self.lang, use_gpu=self.use_gpu, show_log=False)
        return self._ocr

    def extract(self, image: Any) -> OCRResult:
        ocr = self._get_ocr()
        pages = ocr.ocr(_image_for_array_ocr(image), cls=True)
        texts: list[str] = []
        confidences: list[float] = []
        rows = _flatten_paddle_rows(pages)
        for row in rows:
            if len(row) >= 2 and isinstance(row[1], (list, tuple)) and row[1]:
                texts.append(str(row[1][0]))
                if len(row[1]) >= 2:
                    confidences.append(float(row[1][1]))
        return OCRResult(
            text=" ".join(texts).strip(),
            confidence=_mean_confidence(confidences),
            engine=self.name,
            metadata={"lang": self.lang, "use_gpu": self.use_gpu, "boxes": len(rows)},
        )


def build_ocr_engine(name: str, options: dict[str, Any] | None = None) -> OCREngine:
    """Construct an OCR engine from config without importing optional packages early."""

    opts = options or {}
    normalized = name.lower()
    if normalized == "tesseract":
        cmd = opts.get("cmd")
        return TesseractOCREngine(
            lang=str(opts.get("lang", "eng")),
            config=str(opts.get("config", "--psm 6")),
            cmd=str(cmd) if cmd else None,
        )
    if normalized == "easyocr":
        languages = list(opts.get("languages") or ["en"])
        return EasyOCREngine(languages=languages, gpu=bool(opts.get("gpu", False)))
    if normalized == "paddleocr":
        return PaddleOCREngine(
            lang=str(opts.get("lang", "en")),
            use_gpu=bool(opts.get("use_gpu", False)),
        )
    raise ValueError(f"Unknown OCR engine: {name}")


def ocr_options_for_language(
    engine_name: str,
    base_options: dict[str, Any] | None,
    language: str | None,
    ocr_config: dict[str, Any],
) -> dict[str, Any]:
    """Apply configs/ocr.yaml language_mapping to an engine options dictionary."""

    options = dict(base_options or {})
    normalized_engine = engine_name.lower()
    mapped_value = ocr_language_for_engine(normalized_engine, language, ocr_config)
    if not mapped_value:
        return options

    if normalized_engine == "easyocr":
        options["languages"] = list(mapped_value)
    elif normalized_engine == "tesseract":
        options["lang"] = str(mapped_value)
    elif normalized_engine == "paddleocr":
        options["lang"] = str(mapped_value)
    return options


def normalize_language(language: str | None) -> str | None:
    """Normalize dataset language values to canonical labels used by OCR mappings."""

    if language is None:
        return None
    normalized = str(language).strip()
    if not normalized:
        return None
    return LANGUAGE_ALIASES.get(normalized.casefold(), normalized)


def ocr_language_for_engine(
    engine_name: str,
    language: str | None,
    ocr_config: dict[str, Any] | None = None,
) -> str | list[str]:
    """Return OCR language code(s) for a canonical, aliased, or missing language."""

    normalized_engine = engine_name.lower()
    normalized_language = normalize_language(language)

    config_value = _configured_ocr_language(normalized_engine, normalized_language, ocr_config or {})
    if config_value:
        return config_value

    if normalized_engine == "easyocr":
        if normalized_language and normalized_language in LANGUAGE_TO_EASYOCR:
            return LANGUAGE_TO_EASYOCR[normalized_language]
        return DEFAULT_EASYOCR_LANGUAGES
    if normalized_engine == "tesseract":
        if normalized_language and normalized_language in LANGUAGE_TO_TESSERACT:
            return LANGUAGE_TO_TESSERACT[normalized_language]
        return DEFAULT_TESSERACT_LANGUAGE
    if normalized_engine == "paddleocr":
        if normalized_language and normalized_language in LANGUAGE_TO_PADDLEOCR:
            return LANGUAGE_TO_PADDLEOCR[normalized_language]
        return DEFAULT_PADDLEOCR_LANGUAGE

    raise ValueError(f"Unknown OCR engine for language mapping: {engine_name}")


def _configured_ocr_language(
    engine_name: str,
    language: str | None,
    ocr_config: dict[str, Any],
) -> str | list[str] | None:
    mapping = ocr_config.get("language_mapping", {})
    if not isinstance(mapping, dict):
        return None

    if language and isinstance(mapping.get(language), dict):
        mapped_value = mapping[language].get(engine_name)
        if mapped_value:
            return mapped_value

    default_mapping = mapping.get("default") or mapping.get("__default__")
    if isinstance(default_mapping, dict):
        return default_mapping.get(engine_name)
    return None


def safe_extract_ocr(engine: OCREngine, image: Any) -> OCRResult:
    """Run OCR and return an empty fallback result if the engine fails."""

    engine_name = getattr(engine, "name", engine.__class__.__name__)
    try:
        result = engine.extract(image)
    except Exception as exc:
        return OCRResult(
            text="",
            confidence=None,
            engine=str(engine_name),
            metadata={"error": str(exc), "failed": True},
        )

    if result.text is None:
        result.text = ""
    result.metadata.setdefault("failed", False)
    return result


def select_best_ocr_result(results: list[OCRResult]) -> OCRResult:
    """Choose the best OCR output from multiple engines/preprocessing variants."""

    if not results:
        return OCRResult(
            text="",
            confidence=None,
            engine="ensemble",
            metadata={"failed": True, "reason": "no OCR candidates"},
        )

    scored = [(result, _ocr_selection_score(result, results)) for result in results]
    best_result, best_score = max(scored, key=lambda item: item[1])
    candidates = [
        {
            "engine": result.engine,
            "preprocess_variant": result.metadata.get("preprocess_variant"),
            "text": result.text,
            "confidence": result.confidence,
            "score": score,
            "failed": result.metadata.get("failed", False),
            "cached": result.metadata.get("cache_hit", False),
        }
        for result, score in sorted(scored, key=lambda item: item[1], reverse=True)
    ]

    metadata = dict(best_result.metadata)
    metadata.update(
        {
            "ensemble_selected": True,
            "ensemble_score": best_score,
            "ensemble_candidates": candidates,
        }
    )
    return OCRResult(
        text=best_result.text,
        confidence=best_result.confidence,
        engine=best_result.engine,
        metadata=metadata,
    )


def ocr_cache_path(
    cache_dir: str | Path,
    split: str,
    engine: str,
    preprocess_variant: str,
    question_id: str,
) -> Path:
    """Return cache path: data/ocr_cache/{split}/{engine}/{variant}/{question_id}.json."""

    return (
        Path(cache_dir)
        / _safe_cache_part(split)
        / _safe_cache_part(engine)
        / _safe_cache_part(preprocess_variant)
        / f"{_safe_cache_part(question_id)}.json"
    )


def load_ocr_cache_record(
    cache_dir: str | Path,
    split: str,
    engine: str,
    preprocess_variant: str,
    question_id: str,
) -> OCRCacheRecord | None:
    path = ocr_cache_path(cache_dir, split, engine, preprocess_variant, question_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return OCRCacheRecord.from_dict(json.load(handle))


def write_ocr_cache_record(
    cache_dir: str | Path,
    split: str,
    question_id: str,
    language: str,
    preprocess_variant: str,
    result: OCRResult,
) -> OCRCacheRecord:
    record = OCRCacheRecord(
        question_id=question_id,
        language=language,
        ocr_engine=result.engine,
        preprocess_variant=preprocess_variant,
        ocr_text=result.text,
        confidence=result.confidence,
        created_at=datetime.now(timezone.utc).isoformat(),
        metadata=dict(result.metadata),
    )
    path = ocr_cache_path(cache_dir, split, result.engine, preprocess_variant, question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return record


def _ocr_selection_score(result: OCRResult, all_results: list[OCRResult]) -> float:
    text = result.text.strip()
    normalized = _normalize_for_similarity(text)
    token_count = len(normalized.split())
    score = 0.0

    if text:
        score += 5.0
    else:
        score -= 10.0

    if result.confidence is not None:
        score += max(0.0, min(1.0, result.confidence)) * 2.0

    if token_count == 0:
        score -= 6.0
    elif token_count == 1:
        score -= 2.0
    elif 3 <= token_count <= 40:
        score += 2.0
    elif token_count > 80:
        score -= 2.0

    score -= _unknown_character_ratio(text) * 6.0

    if text.endswith("?"):
        score += 1.0

    if result.metadata.get("failed"):
        score -= 8.0

    score += _agreement_score(normalized, all_results)
    return round(score, 6)


def _unknown_character_ratio(text: str) -> float:
    if not text:
        return 1.0
    suspicious = sum(1 for char in text if char in {"�", "□", "■"})
    return suspicious / max(len(text), 1)


def _agreement_score(normalized_text: str, all_results: list[OCRResult]) -> float:
    if not normalized_text:
        return 0.0

    similarities: list[float] = []
    for other in all_results:
        other_text = _normalize_for_similarity(other.text)
        if not other_text or other_text == normalized_text:
            continue
        similarities.append(SequenceMatcher(None, normalized_text, other_text).ratio())

    if not similarities:
        return 0.0
    return (sum(similarities) / len(similarities)) * 2.0


def _normalize_for_similarity(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    return normalized


def _image_for_tesseract(image: Any) -> Any:
    if isinstance(image, Image.Image):
        return to_rgb(image)
    return image


def _image_for_array_ocr(image: Any) -> Any:
    if not isinstance(image, Image.Image):
        return image

    try:
        import numpy as np
    except ImportError as exc:
        raise MissingOCREngineDependency(
            "numpy is required to pass PIL images to EasyOCR/PaddleOCR."
        ) from exc

    return np.array(to_rgb(image))


def _mean_confidence(confidences: list[float]) -> float | None:
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def _flatten_paddle_rows(pages: Any) -> list[Any]:
    if not pages:
        return []

    # PaddleOCR commonly returns [[box, (text, confidence)], ...] for one image,
    # or a list of pages/images around that structure depending on version.
    first = pages[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2 and _looks_like_paddle_text(first[1]):
        return list(pages)

    rows: list[Any] = []
    for page in pages:
        if page:
            rows.extend(page)
    return rows


def _looks_like_paddle_text(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 2 and isinstance(value[0], str)


def _safe_cache_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "unknown"


def enabled_ocr_engines(ocr_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return enabled OCR engine configs from configs/ocr.yaml."""

    engines = ocr_config.get("engines", {})
    if not isinstance(engines, dict):
        raise ValueError("ocr_config['engines'] must be a mapping.")
    enabled: list[tuple[str, dict[str, Any]]] = []
    for name, options in engines.items():
        engine_options = dict(options or {})
        if engine_options.pop("enabled", False):
            enabled.append((name, engine_options))
    return enabled
