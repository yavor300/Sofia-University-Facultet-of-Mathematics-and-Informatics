from openqa_textual.answer_postprocess import clean_answer


def test_clean_answer_removes_common_prefixes() -> None:
    assert clean_answer("Final answer: 42") == "42"
    assert clean_answer("Answer: 42") == "42"
    assert clean_answer("A: 42") == "42"
    assert clean_answer("Отговор: София", language="Bulgarian") == "София"


def test_clean_answer_removes_explanation_after_answer() -> None:
    assert clean_answer("42\nExplanation: because math") == "42"
    assert clean_answer("42 Обяснение: защото", language="Bulgarian") == "42"


def test_clean_answer_keeps_first_meaningful_line() -> None:
    assert clean_answer("42\nBecause this follows from the formula.") == "42"


def test_clean_answer_preserves_math_notation() -> None:
    assert clean_answer(r"\(\frac{1}{2}\)") == r"\(\frac{1}{2}\)"
    assert clean_answer("x = -1, y = 2") == "x = -1, y = 2"


def test_clean_answer_normalizes_empty_to_string() -> None:
    assert clean_answer("") == ""
    assert clean_answer(None) == ""


def test_clean_answer_converts_safe_english_number_words() -> None:
    assert clean_answer("three", language="English") == "3"
    assert clean_answer("three apples", language="English") == "three apples"


def test_clean_answer_normalizes_decimal_separator_for_supported_languages() -> None:
    assert clean_answer("3.14", language="Bulgarian") == "3,14"
    assert clean_answer("x = 3.14", language="Bulgarian") == "x = 3.14"


def test_clean_answer_deduplicates_repeated_phrases() -> None:
    assert clean_answer("Paris Paris") == "Paris"
    assert clean_answer("Paris; Paris; Paris") == "Paris"
