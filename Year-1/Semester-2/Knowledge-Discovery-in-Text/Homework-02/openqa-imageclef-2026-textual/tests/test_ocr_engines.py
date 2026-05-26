from PIL import Image

from openqa_textual.ocr import (
    EasyOCREngine,
    OCRResult,
    PaddleOCREngine,
    TesseractOCREngine,
    load_ocr_cache_record,
    ocr_cache_path,
    select_best_ocr_result,
    write_ocr_cache_record,
)


def test_tesseract_engine_parses_text_with_fake_module(monkeypatch) -> None:
    class FakePytesseract:
        @staticmethod
        def image_to_string(image, lang, config):
            assert image.mode == "RGB"
            assert lang == "eng"
            assert config == "--psm 6"
            return "  Hello OCR  \n"

    monkeypatch.setitem(__import__("sys").modules, "pytesseract", FakePytesseract)

    result = TesseractOCREngine(lang="eng", config="--psm 6").extract(Image.new("L", (5, 5)))

    assert result == OCRResult(
        text="Hello OCR",
        confidence=None,
        engine="tesseract",
        metadata={"lang": "eng", "config": "--psm 6", "cmd": "tesseract"},
    )


def test_easyocr_engine_joins_text_and_averages_confidence() -> None:
    class FakeReader:
        def readtext(self, image, detail, paragraph):
            assert image.shape == (5, 5, 3)
            assert detail == 1
            assert paragraph is False
            return [
                ([(0, 0)], "What", 0.8),
                ([(1, 1)], "now?", 1.0),
            ]

    engine = EasyOCREngine(["en"], gpu=False)
    engine._reader = FakeReader()

    result = engine.extract(Image.new("RGB", (5, 5), "white"))

    assert result.text == "What now?"
    assert result.confidence == 0.9
    assert result.engine == "easyocr"
    assert result.metadata == {"languages": ["en"], "gpu": False, "boxes": 2}


def test_paddleocr_engine_joins_text_and_averages_confidence() -> None:
    class FakePaddle:
        def ocr(self, image, cls):
            assert image.shape == (5, 5, 3)
            assert cls is True
            return [
                [
                    [[0, 0], [1, 0], [1, 1], [0, 1]],
                    ("Hello", 0.7),
                ],
                [
                    [[0, 1], [1, 1], [1, 2], [0, 2]],
                    ("world", 0.9),
                ],
            ]

    engine = PaddleOCREngine(lang="en", use_gpu=False)
    engine._ocr = FakePaddle()

    result = engine.extract(Image.new("RGB", (5, 5), "white"))

    assert result.text == "Hello world"
    assert result.confidence == 0.8
    assert result.engine == "paddleocr"
    assert result.metadata == {"lang": "en", "use_gpu": False, "boxes": 2}


def test_ocr_cache_path_sanitizes_question_id() -> None:
    path = ocr_cache_path("cache", "train/dev", "easyocr", "contrast", "q/1")
    assert str(path).replace("\\", "/") == "cache/train_dev/easyocr/contrast/q_1.json"


def test_write_and_load_ocr_cache_record(tmp_path) -> None:
    result = OCRResult(
        text="Какъв е отговорът?",
        confidence=0.91,
        engine="easyocr",
        metadata={"failed": False},
    )

    written = write_ocr_cache_record(
        cache_dir=tmp_path,
        split="train",
        question_id="q-1",
        language="Bulgarian",
        preprocess_variant="contrast",
        result=result,
    )
    loaded = load_ocr_cache_record(tmp_path, "train", "easyocr", "contrast", "q-1")

    assert loaded is not None
    assert loaded.question_id == "q-1"
    assert loaded.language == "Bulgarian"
    assert loaded.ocr_text == "Какъв е отговорът?"
    assert loaded.confidence == 0.91
    assert loaded.created_at == written.created_at
    assert loaded.to_result().metadata["cache_hit"] is True


def test_select_best_ocr_result_prefers_non_empty_confident_question() -> None:
    selected = select_best_ocr_result(
        [
            OCRResult(text="", confidence=0.99, engine="easyocr", metadata={}),
            OCRResult(text="Какво е фотосинтеза?", confidence=0.80, engine="tesseract", metadata={}),
            OCRResult(text="Какво е фотосинтеза", confidence=0.78, engine="paddleocr", metadata={}),
        ]
    )

    assert selected.text == "Какво е фотосинтеза?"
    assert selected.engine == "tesseract"
    assert selected.metadata["ensemble_selected"] is True
    assert len(selected.metadata["ensemble_candidates"]) == 3


def test_select_best_ocr_result_penalizes_unknown_characters() -> None:
    selected = select_best_ocr_result(
        [
            OCRResult(text="����", confidence=0.99, engine="bad", metadata={}),
            OCRResult(text="Дайте пример за организъм.", confidence=0.50, engine="good", metadata={}),
        ]
    )

    assert selected.engine == "good"


def test_select_best_ocr_result_handles_empty_candidate_list() -> None:
    selected = select_best_ocr_result([])

    assert selected.text == ""
    assert selected.engine == "ensemble"
    assert selected.metadata["failed"] is True
