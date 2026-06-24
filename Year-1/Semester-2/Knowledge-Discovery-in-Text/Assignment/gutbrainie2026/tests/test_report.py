from gutbrainie.evaluation.report import generate_data_statistics


def test_generate_data_statistics_writes_requested_outputs(tmp_path):
    data_root = tmp_path / "gutbrainie2026"
    output_dir = tmp_path / "reports"
    _write_gold_fixture(data_root)

    result = generate_data_statistics(data_root, output_dir, qualities=("gold",))

    expected_files = {
        output_dir / "data_stats_gold.csv",
        output_dir / "entity_label_distribution.csv",
        output_dir / "relation_label_distribution.csv",
        output_dir / "relation_triple_distribution.csv",
        output_dir / "entity_distribution.png",
        output_dir / "relation_distribution.png",
        output_dir / "imbalance_summary.md",
    }

    for file_path in expected_files:
        assert file_path.exists()
        assert file_path.stat().st_size > 0

    stats = result["split_stats"][0]
    assert stats["documents"] == 2
    assert stats["entities"] == 3
    assert stats["mention_level_relations"] == 2
    assert stats["entity_majority_label"] == "DDF"
    assert stats["relation_majority_label"] == "affect"


def _write_gold_fixture(data_root):
    article_dir = data_root / "Articles" / "csv_format"
    annotation_dir = data_root / "Annotations" / "Train" / "gold_quality" / "csv_format"
    article_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)

    (article_dir / "articles_train_gold.csv").write_text(
        "pmid|title|authors|journal|year|abstract\n"
        "1|Gut brain axis|A Author|Journal|2026|Microbes affect neurons.\n"
        "2|Diet and mood|B Author|Journal|2026|Butyrate changes inflammation.\n",
        encoding="utf-8",
    )
    (annotation_dir / "train_gold_entities.csv").write_text(
        "pmid|annotator|start_idx|end_idx|location|text_span|label\n"
        "1|ann|0|3|title|Gut|DDF\n"
        "1|ann2|0|3|title|Gut|DDF\n"
        "1|ann|4|9|title|brain|DDF\n"
        "2|ann|0|8|abstract|Butyrate|chemical\n",
        encoding="utf-8",
    )
    (annotation_dir / "train_gold_mention_level_relations.csv").write_text(
        "pmid|annotator|subject_text_span|subject_label|predicate|object_text_span|object_label\n"
        "1|ann|Microbes|bacteria|affect|neurons|anatomical location\n"
        "2|ann|Butyrate|chemical|affect|inflammation|DDF\n",
        encoding="utf-8",
    )
    (annotation_dir / "train_gold_relations.csv").write_text(
        "pmid|annotator|subject_start_idx|subject_end_idx|subject_location|subject_text_span|"
        "subject_label|predicate|object_start_idx|object_end_idx|object_location|object_text_span|object_label\n"
        "1|ann|0|8|abstract|Microbes|bacteria|affect|15|22|abstract|neurons|anatomical location\n"
        "2|ann|0|8|abstract|Butyrate|chemical|affect|17|29|abstract|inflammation|DDF\n",
        encoding="utf-8",
    )
