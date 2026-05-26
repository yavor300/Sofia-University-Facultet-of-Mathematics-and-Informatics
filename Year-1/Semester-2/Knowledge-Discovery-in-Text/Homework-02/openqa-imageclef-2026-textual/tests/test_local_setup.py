from pathlib import Path

import pytest

from openqa_textual import __version__
from openqa_textual.config import expand_env_vars, project_root
from openqa_textual.ocr import (
    OCREngine,
    OCRResult,
    build_ocr_engine,
    enabled_ocr_engines,
    normalize_language,
    ocr_language_for_engine,
    ocr_options_for_language,
    safe_extract_ocr,
)


def test_package_imports() -> None:
    assert __version__


def test_project_root_contains_pyproject() -> None:
    assert (project_root() / "pyproject.toml").exists()


def test_expand_env_vars_default() -> None:
    assert expand_env_vars("${MISSING_OPENQA_VAR:-fallback}") == "fallback"


def test_enabled_ocr_engines_respects_config_flags() -> None:
    config = {
        "engines": {
            "tesseract": {"enabled": True, "lang": "eng"},
            "easyocr": {"enabled": False, "languages": ["en"]},
        }
    }
    assert enabled_ocr_engines(config) == [("tesseract", {"lang": "eng"})]


def test_build_ocr_engine_does_not_import_optional_dependency() -> None:
    engine = build_ocr_engine("tesseract", {"lang": "eng"})
    assert engine.name == "tesseract"


def test_ocr_options_for_language_applies_config_mapping() -> None:
    options = ocr_options_for_language(
        "tesseract",
        {"lang": "eng", "config": "--psm 6"},
        "Bulgarian",
        {"language_mapping": {"Bulgarian": {"tesseract": "bul+eng"}}},
    )
    assert options == {"lang": "bul+eng", "config": "--psm 6"}


def test_normalize_language_handles_aliases_and_missing_values() -> None:
    assert normalize_language("bg") == "Bulgarian"
    assert normalize_language(" БЪЛГАРСКИ ") == "Bulgarian"
    assert normalize_language("") is None
    assert normalize_language(None) is None


def test_ocr_language_for_engine_uses_builtin_fallbacks() -> None:
    assert ocr_language_for_engine("easyocr", "Bulgarian") == ["bg", "en"]
    assert ocr_language_for_engine("tesseract", "bg") == "bul+eng"
    assert ocr_language_for_engine("tesseract", "Chinese") == "chi_sim+eng"
    assert ocr_language_for_engine("tesseract", "Croatian") == "hrv+eng"
    assert ocr_language_for_engine("tesseract", "Italian") == "ita+eng"
    assert ocr_language_for_engine("tesseract", "Serbian") == "srp+srp_latn+eng"
    assert ocr_language_for_engine("easyocr", None) == ["en", "bg", "hr", "it", "rs_latin"]
    assert ocr_language_for_engine("paddleocr", "unknown-language") == "en"


def test_ocr_language_for_engine_allows_config_default_override() -> None:
    config = {
        "language_mapping": {
            "default": {"easyocr": ["en"], "tesseract": "eng", "paddleocr": "en"}
        }
    }
    assert ocr_language_for_engine("easyocr", None, config) == ["en"]
    assert ocr_language_for_engine("tesseract", "Klingon", config) == "eng"
    assert ocr_language_for_engine("tesseract", "Chinese", config) == "chi_sim+eng"


def test_ocr_engine_base_interface_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        OCREngine().extract(object())


def test_safe_extract_ocr_returns_empty_result_on_failure() -> None:
    class FailingEngine:
        name = "failing"

        def extract(self, image):
            raise RuntimeError("boom")

    result = safe_extract_ocr(FailingEngine(), object())
    assert result == OCRResult(
        text="",
        confidence=None,
        engine="failing",
        metadata={"error": "boom", "failed": True},
    )


def test_expected_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in ["configs", "data/raw", "data/processed", "data/ocr_cache", "experiments/runs"]:
        assert (root / path).exists()
