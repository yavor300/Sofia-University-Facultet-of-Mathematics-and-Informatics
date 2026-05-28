"""Answer generation baselines."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from fractions import Fraction
import operator
import re
from typing import Any

from openqa_textual.answer_postprocess import clean_answer


SUPPORTED_LOCAL_LLM_MODELS = (
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
)

DEFAULT_GENERATION_KWARGS = {
    "temperature": 0.0,
    "do_sample": False,
    "max_new_tokens": 64,
    "num_beams": 1,
}


@dataclass(slots=True)
class GenerationResult:
    answers: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class OCRDiagnosticGenerator:
    """Baseline 0: record OCR output only and intentionally produce no answer."""

    name = "ocr_only_diagnostic"

    def generate(self, question: str, language: str | None = None) -> GenerationResult:
        return GenerationResult(
            answers=[],
            metadata={
                "baseline": self.name,
                "language": language or "English",
                "question_length": len(question or ""),
            },
        )


class HeuristicQAGenerator:
    """Baseline 1: tiny arithmetic and yes/no smoke-test generator."""

    name = "heuristic_qa"

    def generate(self, question: str, language: str | None = None) -> GenerationResult:
        text = str(question or "").strip()
        normalized_language = str(language or "English")

        comparison = answer_arithmetic_comparison(text, normalized_language)
        if comparison is not None:
            return GenerationResult(
                answers=[comparison],
                metadata={"baseline": self.name, "rule": "arithmetic_comparison"},
            )

        arithmetic = answer_arithmetic_expression(text)
        if arithmetic is not None:
            return GenerationResult(
                answers=[arithmetic],
                metadata={"baseline": self.name, "rule": "arithmetic_expression"},
            )

        if is_yes_no_question(text, normalized_language):
            return GenerationResult(
                answers=[""],
                metadata={"baseline": self.name, "rule": "yes_no_detected_unanswered"},
            )

        return GenerationResult(
            answers=[""],
            metadata={"baseline": self.name, "rule": "fallback_empty"},
        )


class LocalLLMGenerator:
    """Baseline 2: OCR question text answered by a local instruction model."""

    name = "local_llm_prompted"

    def __init__(
        self,
        model_name: str,
        tokenizer: Any,
        model: Any,
        adapter_path: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.adapter_path = adapter_path
        self.generation_kwargs = dict(DEFAULT_GENERATION_KWARGS)
        self.generation_kwargs.update(generation_kwargs or {})
        self.generation_kwargs = sanitize_generation_kwargs(self.generation_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        cache_dir: str | None = None,
        load_in_4bit: bool = False,
        device_map: str | None = "auto",
        torch_dtype: str | None = "auto",
        adapter_path: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> "LocalLLMGenerator":
        """Load a local Hugging Face causal LM for prompted answering."""

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for local LLM generation.") from exc

        model_kwargs: dict[str, Any] = {}
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
        if device_map:
            model_kwargs["device_map"] = device_map
        if torch_dtype:
            model_kwargs["torch_dtype"] = torch_dtype

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("transformers BitsAndBytesConfig is required for 4-bit loading.") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("peft is required to load a LoRA adapter.") from exc
            model = PeftModel.from_pretrained(model, adapter_path)
        return cls(
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            adapter_path=adapter_path,
            generation_kwargs=generation_kwargs,
        )

    def generate(
        self,
        question: str,
        language: str | None = None,
        retrieved_examples: list[dict[str, Any]] | None = None,
    ) -> GenerationResult:
        prompt = (
            build_rag_openqa_prompt(
                question=question,
                language=language,
                retrieved_examples=retrieved_examples or [],
            )
            if retrieved_examples
            else build_openqa_prompt(question=question, language=language)
        )
        model_input = format_prompt_for_model(self.tokenizer, prompt)
        inputs = self.tokenizer(model_input, return_tensors="pt")

        model_device = getattr(self.model, "device", None)
        if model_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)

        generate_kwargs = dict(self.generation_kwargs)
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            generate_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        elif getattr(self.tokenizer, "eos_token_id", None) is not None:
            generate_kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)

        outputs = self.model.generate(**inputs, **generate_kwargs)
        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_length:]
        raw_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        answer = clean_generated_answer(raw_answer)
        return GenerationResult(
            answers=[answer],
            metadata={
                "baseline": self.name,
                "model": self.model_name,
                "adapter_path": self.adapter_path,
                "prompt_type": "rag" if retrieved_examples else "zero_shot",
                "retrieved_example_count": len(retrieved_examples or []),
                "raw_answer": raw_answer,
                "prompt": model_input,
            },
        )


def build_openqa_prompt(question: str, language: str | None = None) -> list[dict[str, str]]:
    """Build chat messages for OCR + prompted LLM answering."""

    return [
        {
            "role": "system",
            "content": (
                "You are answering an exam-style open question. "
                "The question was extracted from an image using OCR, so it may contain minor OCR mistakes. "
                "Answer with only the final answer. Do not explain."
            ),
        },
        {
            "role": "user",
            "content": f"Language: {language or 'English'}\nQuestion: {question}\n\nFinal answer:",
        },
    ]


def build_rag_openqa_prompt(
    question: str,
    language: str | None = None,
    retrieved_examples: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build a few-shot RAG prompt from retrieved train examples."""

    examples = format_rag_examples(retrieved_examples or [])
    user_parts = [
        "Examples:",
        examples or "(No relevant examples retrieved.)",
        f"Current question language: {language or 'English'}",
        f"Current OCR question: {question}",
        "",
        "Final answer:",
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are answering exam-style open questions extracted from images by OCR. "
                "The OCR text may contain minor errors. "
                "Use the examples only as guidance. Do not copy an answer unless the question is equivalent. "
                "Return only the final answer."
            ),
        },
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def format_rag_examples(examples: list[dict[str, Any]]) -> str:
    """Format retrieved records as Q/A few-shot examples."""

    blocks = []
    for example in examples:
        question = str(example.get("ocr_question") or example.get("question") or "").strip()
        answer = str(example.get("gold_answer") or example.get("answer") or "").strip()
        if not question and not answer:
            continue
        blocks.append(f"Q: {question}\nA: {answer}")
    return "\n\n".join(blocks)


def format_prompt_for_model(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Use a tokenizer chat template when available, otherwise use a plain prompt."""

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    system = messages[0]["content"]
    user = messages[1]["content"]
    return f"{system}\n\n{user}"


def clean_generated_answer(answer: str) -> str:
    """Clean a short final answer emitted by a prompted model."""

    return clean_answer(answer)


def sanitize_generation_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove generation flags that Transformers ignores for deterministic decoding."""

    cleaned = dict(kwargs)
    if cleaned.get("do_sample") is False:
        cleaned.pop("temperature", None)
        cleaned.pop("top_p", None)
        cleaned.pop("top_k", None)
    return cleaned


def answer_arithmetic_expression(question: str) -> str | None:
    expression = _extract_arithmetic_expression(question)
    if expression is None:
        return None

    value = _safe_eval_arithmetic(expression)
    if value is None:
        return None
    return _format_number(value)


def answer_arithmetic_comparison(question: str, language: str | None = None) -> str | None:
    comparison = _extract_arithmetic_comparison(question)
    if comparison is None:
        return None

    left, right = comparison
    left_value = _safe_eval_arithmetic(left)
    right_value = _safe_eval_arithmetic(right)
    if left_value is None or right_value is None:
        return None

    is_true = left_value == right_value
    return _yes_no_answer(is_true, language)


def is_yes_no_question(question: str, language: str | None = None) -> bool:
    lowered = str(question or "").strip().casefold()
    if not lowered:
        return False

    english_prefixes = (
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "should ",
        "would ",
        "will ",
        "has ",
        "have ",
        "had ",
    )
    bulgarian_prefixes = ("дали ", "вярно ли", "може ли", "има ли", "ли ")

    if language and str(language).casefold() == "bulgarian":
        return lowered.startswith(bulgarian_prefixes) or " ли " in f" {lowered} "
    return lowered.startswith(english_prefixes) or lowered.startswith(bulgarian_prefixes)


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _extract_arithmetic_comparison(question: str) -> tuple[str, str] | None:
    text = _normalize_math_text(question)
    pattern = re.compile(
        r"(?P<left>[+-]?\d+(?:\.\d+)?(?:\s*[-+*/^]\s*[+-]?\d+(?:\.\d+)?)+)\s*"
        r"(?:=|==|equals?|is)\s*"
        r"(?P<right>[+-]?\d+(?:\.\d+)?(?:\s*[-+*/^]\s*[+-]?\d+(?:\.\d+)?)*)",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None
    return match.group("left"), match.group("right")


def _extract_arithmetic_expression(question: str) -> str | None:
    text = _normalize_math_text(question)
    pattern = re.compile(
        r"(?<![\w.])([+-]?\d+(?:\.\d+)?(?:\s*[-+*/^]\s*[+-]?\d+(?:\.\d+)?)+)(?![\w.])"
    )
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1)


def _normalize_math_text(text: str) -> str:
    normalized = str(text or "")
    normalized = normalized.replace(",", ".")
    normalized = normalized.replace("×", "*").replace("÷", "/").replace("−", "-")
    normalized = normalized.replace("^", "**")
    return normalized


def _safe_eval_arithmetic(expression: str) -> Fraction | None:
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_ast_node(tree.body)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None
    return value


def _eval_ast_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("exponent too large")
        result = _ALLOWED_BINOPS[type(node.op)](left, right)
        return Fraction(result)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return Fraction(_ALLOWED_UNARYOPS[type(node.op)](_eval_ast_node(node.operand)))
    raise ValueError(f"Unsupported arithmetic expression: {ast.dump(node)}")


def _format_number(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    decimal = float(value)
    formatted = f"{decimal:.10g}"
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def _yes_no_answer(is_true: bool, language: str | None = None) -> str:
    if language and str(language).casefold() == "bulgarian":
        return "да" if is_true else "не"
    return "yes" if is_true else "no"
