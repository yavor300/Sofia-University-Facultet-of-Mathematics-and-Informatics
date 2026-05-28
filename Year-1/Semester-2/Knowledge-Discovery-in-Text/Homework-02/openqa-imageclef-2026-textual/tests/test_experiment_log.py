from openqa_textual.experiment_log import append_experiment, ensure_experiment_log, markdown_table_row


def test_markdown_table_row_escapes_pipes_and_newlines() -> None:
    assert markdown_table_row(["E08", "a|b", "line\nbreak"]) == "| E08 | a\\|b | line<br>break |"


def test_append_experiment_creates_log_and_appends_row(tmp_path) -> None:
    log_path = tmp_path / "experiments" / "experiment_log.md"

    ensure_experiment_log(log_path)
    row = append_experiment(
        path=log_path,
        experiment_id="E08",
        ocr="tesseract",
        preprocess="resize_only",
        ocr_correction="no",
        retrieval="none",
        llm="Qwen",
        fine_tuned="no",
        dev_score="TBD",
        notes="smoke",
    )

    text = log_path.read_text(encoding="utf-8")
    assert "# Experiment Log" in text
    assert row in text
    assert "| E08 | tesseract | resize_only | no | none | Qwen | no | TBD | smoke |" in text
