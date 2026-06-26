from gutbrainie.cli import main


def test_cli_help(capsys):
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "GutBrainIE 2026" in captured.out
    assert "prepare-data" in captured.out
    assert "run-ner-transformer" in captured.out
    assert "train-token-classifier" in captured.out
    assert "predict-token-classifier" in captured.out
    assert "run-re-transformer" in captured.out
    assert "predict-re-rule" in captured.out
    assert "train-re-pair-classifier" in captured.out
    assert "predict-re-pair-classifier" in captured.out
    assert "predict-re-ollama" in captured.out
    assert "atlop-notes" in captured.out
    assert "run-atlop" in captured.out
    assert "evaluate-official" in captured.out


def test_prepare_data_command_writes_report(tmp_path, capsys):
    data_root = tmp_path / "data" / "gutbrainie2026"
    output = tmp_path / "outputs" / "reports" / "validation.json"
    _write_gold_fixture(data_root)

    assert main(["prepare-data", "--data-root", str(data_root), "--quality", "gold", "--output", str(output)]) == 0

    captured = capsys.readouterr()
    assert "Validation report written" in captured.out
    assert output.exists()


def _write_gold_fixture(data_root):
    article_dir = data_root / "Articles" / "csv_format"
    annotation_dir = data_root / "Annotations" / "Train" / "gold_quality" / "csv_format"
    article_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)

    (article_dir / "articles_train_gold.csv").write_text(
        "pmid|title|authors|journal|year|abstract\n"
        "1|Gut brain axis|A Author|Journal|2026|Microbes affect neurons.\n",
        encoding="utf-8",
    )
    (annotation_dir / "train_gold_entities.csv").write_text(
        "pmid|annotator|start_idx|end_idx|location|text_span|label\n"
        "1|ann|0|3|title|Gut|anatomical location\n"
        "1|ann2|0|3|title|Gut|anatomical location\n",
        encoding="utf-8",
    )
    (annotation_dir / "train_gold_mention_level_relations.csv").write_text(
        "pmid|annotator|subject_text_span|subject_label|predicate|object_text_span|object_label\n"
        "1|ann|Microbes|bacteria|affect|neurons|anatomical location\n",
        encoding="utf-8",
    )
    (annotation_dir / "train_gold_relations.csv").write_text(
        "pmid|annotator|subject_start_idx|subject_end_idx|subject_location|subject_text_span|"
        "subject_label|predicate|object_start_idx|object_end_idx|object_location|object_text_span|object_label\n"
        "1|ann|0|8|abstract|Microbes|bacteria|affect|15|22|abstract|neurons|anatomical location\n",
        encoding="utf-8",
    )
