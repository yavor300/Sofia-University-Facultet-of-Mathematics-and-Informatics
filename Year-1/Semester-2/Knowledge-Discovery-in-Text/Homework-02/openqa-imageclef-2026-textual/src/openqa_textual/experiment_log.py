"""Markdown experiment log helpers."""

from __future__ import annotations

from pathlib import Path


EXPERIMENT_COLUMNS = (
    "ID",
    "OCR",
    "Preprocess",
    "OCR correction",
    "Retrieval",
    "LLM",
    "Fine-tuned",
    "Dev score",
    "Notes",
)


def default_experiment_log() -> str:
    """Return a fresh experiment log document."""

    header = "| " + " | ".join(EXPERIMENT_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in EXPERIMENT_COLUMNS) + " |"
    return (
        "# Experiment Log\n\n"
        "Dev labels are currently hidden, so local scores are diagnostics unless explicitly marked "
        "as official leaderboard scores.\n\n"
        f"{header}\n"
        f"{separator}\n"
    )


def ensure_experiment_log(path: str | Path) -> Path:
    """Create an experiment log if it does not exist."""

    log_path = Path(path)
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(default_experiment_log(), encoding="utf-8")
    return log_path


def append_experiment(
    path: str | Path,
    experiment_id: str,
    ocr: str,
    preprocess: str,
    ocr_correction: str,
    retrieval: str,
    llm: str,
    fine_tuned: str,
    dev_score: str,
    notes: str,
) -> str:
    """Append one experiment row to the Markdown log."""

    log_path = ensure_experiment_log(path)
    row = markdown_table_row(
        [
            experiment_id,
            ocr,
            preprocess,
            ocr_correction,
            retrieval,
            llm,
            fine_tuned,
            dev_score,
            notes,
        ]
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")
    return row


def markdown_table_row(values: list[str]) -> str:
    """Format a Markdown table row with escaped cell values."""

    return "| " + " | ".join(_escape_cell(value) for value in values) + " |"


def _escape_cell(value: str) -> str:
    text = str(value or "-").replace("\n", "<br>")
    return text.replace("|", "\\|")
