from openqa_textual.generation import (
    LocalLLMGenerator,
    build_openqa_prompt,
    build_rag_openqa_prompt,
    clean_generated_answer,
    format_rag_examples,
    format_prompt_for_model,
    sanitize_generation_kwargs,
)
from openqa_textual.prediction import build_llm_predictions_from_ocr_rows


class FakeInputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeTokenSlice:
    pass


class FakeInputIds:
    shape = (1, 3)


class FakeTokenRow:
    def __getitem__(self, item):
        return FakeTokenSlice()


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return "CHAT:" + messages[-1]["content"]

    def __call__(self, text, return_tensors):
        assert text.startswith("CHAT:")
        assert return_tensors == "pt"
        return FakeInputs({"input_ids": FakeInputIds()})

    def decode(self, tokens, skip_special_tokens=True):
        assert isinstance(tokens, FakeTokenSlice)
        assert skip_special_tokens is True
        return "Final answer: Photosynthesis\nExplanation: nope"


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        assert kwargs["pad_token_id"] == 99
        assert kwargs["max_new_tokens"] == 8
        return [FakeTokenRow()]


def test_build_openqa_prompt_contains_ocr_warning_and_question() -> None:
    messages = build_openqa_prompt("What is photosynthesis?", language="English")
    assert "OCR" in messages[0]["content"]
    assert "Language: English" in messages[1]["content"]
    assert "Question: What is photosynthesis?" in messages[1]["content"]


def test_build_rag_openqa_prompt_contains_examples_and_current_question() -> None:
    messages = build_rag_openqa_prompt(
        "What is 2 + 2?",
        language="English",
        retrieved_examples=[
            {
                "ocr_question": "What is 3 + 3?",
                "gold_answer": "6",
            }
        ],
    )

    assert "Use the examples only as guidance" in messages[0]["content"]
    assert "Q: What is 3 + 3?\nA: 6" in messages[1]["content"]
    assert "Current question language: English" in messages[1]["content"]
    assert "Current OCR question: What is 2 + 2?" in messages[1]["content"]


def test_format_rag_examples_skips_empty_records() -> None:
    assert format_rag_examples([{}, {"ocr_question": "Q?", "gold_answer": "A"}]) == "Q: Q?\nA: A"


def test_clean_generated_answer_removes_prefix_and_explanation() -> None:
    assert clean_generated_answer("Answer: 42\nBecause...") == "42"


def test_sanitize_generation_kwargs_removes_sampling_only_flags_when_deterministic() -> None:
    assert sanitize_generation_kwargs(
        {"do_sample": False, "temperature": 0.0, "top_p": 0.9, "max_new_tokens": 8}
    ) == {"do_sample": False, "max_new_tokens": 8}


def test_format_prompt_for_model_falls_back_without_chat_template() -> None:
    class PlainTokenizer:
        pass

    prompt = format_prompt_for_model(PlainTokenizer(), build_openqa_prompt("Q?", "English"))
    assert "Question: Q?" in prompt


def test_local_llm_generator_with_fake_model() -> None:
    generator = LocalLLMGenerator(
        model_name="fake/model",
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        generation_kwargs={"max_new_tokens": 8},
    )

    result = generator.generate(
        "What is photosynthesis?",
        language="English",
        retrieved_examples=[{"ocr_question": "What is a plant process?", "gold_answer": "photosynthesis"}],
    )

    assert result.answers == ["Photosynthesis"]
    assert result.metadata["model"] == "fake/model"
    assert result.metadata["baseline"] == "local_llm_prompted"
    assert result.metadata["prompt_type"] == "rag"
    assert result.metadata["retrieved_example_count"] == 1


def test_build_llm_predictions_from_ocr_rows() -> None:
    generator = LocalLLMGenerator(
        model_name="fake/model",
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        generation_kwargs={"max_new_tokens": 8},
    )
    predictions = build_llm_predictions_from_ocr_rows(
        [{"question_id": "q-1", "language": "English", "clean_question": "What is 2+2?"}],
        generator,
    )

    assert predictions[0]["question_id"] == "q-1"
    assert predictions[0]["answers"] == ["Photosynthesis"]
    assert predictions[0]["debug"]["model"] == "fake/model"
