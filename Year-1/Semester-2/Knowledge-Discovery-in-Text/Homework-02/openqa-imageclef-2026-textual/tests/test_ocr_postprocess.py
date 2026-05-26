from openqa_textual.ocr_postprocess import (
    clean_ocr_question,
    fix_common_ocr_errors,
    normalize_quotes_and_symbols,
    normalize_whitespace,
    restore_question_mark,
)


def test_normalize_whitespace_collapses_lines_and_spaces() -> None:
    assert normalize_whitespace("  Какво   е\n\nтова?  ") == "Какво е това?"


def test_normalize_quotes_and_symbols() -> None:
    assert normalize_quotes_and_symbols("“A”—B ≤ 3") == '"A"-B <= 3'


def test_fix_common_ocr_errors_is_conservative() -> None:
    assert fix_common_ocr_errors("he110 2O24") == "he110 2024"


def test_restore_question_mark_for_bulgarian_question() -> None:
    assert restore_question_mark("Кои са добри проводници на топлина.") == (
        "Кои са добри проводници на топлина?"
    )


def test_restore_question_mark_leaves_statement() -> None:
    assert restore_question_mark("Дайте примери за живи организми.") == (
        "Дайте примери за живи организми."
    )


def test_clean_ocr_question_fixes_bulgarian_hyphenated_line_break() -> None:
    raw = "Кои отпадъци се разпадат най-\nтрудно в почвата и я замърсяват?"
    assert clean_ocr_question(raw, language="Bulgarian") == (
        "Кои отпадъци се разпадат най-трудно в почвата и я замърсяват?"
    )


def test_clean_ocr_question_fixes_english_broken_hyphenation() -> None:
    raw = "What is photo-\nsynthesis"
    assert clean_ocr_question(raw, language="English") == "What is photosynthesis?"


def test_clean_ocr_question_removes_edge_artifacts() -> None:
    assert clean_ocr_question("||| What is energy. ___", language="English") == "What is energy?"

