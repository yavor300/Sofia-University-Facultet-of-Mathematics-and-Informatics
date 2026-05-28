from argparse import Namespace
from pathlib import Path

from PIL import Image

from openqa_textual.generation import GenerationResult
from openqa_textual.ocr import OCRResult
from scripts import predict as predict_script


class FakeGenerator:
    name = "fake_generator"

    def generate(self, question, language=None):
        assert question == "What is 2 + 2?"
        assert language == "English"
        return GenerationResult(["4"], {"baseline": self.name})


def test_predict_sample_builds_internal_prediction_with_debug(monkeypatch, tmp_path) -> None:
    def fake_run_single_ocr(**kwargs):
        return (
            OCRResult(
                text="What is 2 + 2?",
                confidence=0.9,
                engine="fake_ocr",
                metadata={"preprocess_variant": "raw"},
            ),
            True,
            [],
        )

    monkeypatch.setattr(predict_script, "run_single_ocr", fake_run_single_ocr)

    prediction = predict_script.predict_sample(
        sample={
            "question_id": "q-1",
            "language": "English",
            "image": Image.new("RGB", (10, 10), "white"),
        },
        index=0,
        split_name="validation",
        engine_name="fake_ocr",
        generator=FakeGenerator(),
        retriever=None,
        args=Namespace(
            preprocess_variant="raw",
            no_cache=False,
            rag_k=0,
        ),
        ocr_config={},
        ocr_cache_dir=Path(tmp_path),
        engine_cache={},
    )

    assert prediction["question_id"] == "q-1"
    assert prediction["answers"] == ["4"]
    assert prediction["language"] == "English"
    assert prediction["debug"]["ocr_text"] == "What is 2 + 2?"
    assert prediction["debug"]["clean_question"] == "What is 2 + 2?"
    assert prediction["debug"]["ocr_engine"] == "fake_ocr"
    assert prediction["debug"]["model"] == "fake_generator"


def test_postprocess_answer_cleans_answer_text() -> None:
    assert predict_script.postprocess_answer("  Answer: answer\nExplanation: no") == "answer"
