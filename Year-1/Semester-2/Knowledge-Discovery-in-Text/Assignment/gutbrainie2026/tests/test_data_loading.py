import pytest

from gutbrainie.data.annotations import (
    deduplicate_entities,
    load_entities_csv,
    load_full_relations_csv,
    load_mention_relations_csv,
)
from gutbrainie.data.articles import load_articles_csv
from gutbrainie.data.dataset import build_validation_report, load_split
from gutbrainie.data.splits import SUPPORTED_QUALITIES


def test_loaders_normalize_pmid_and_integer_offsets(tmp_path):
    articles_path = tmp_path / "articles.csv"
    entities_path = tmp_path / "entities.csv"
    relations_path = tmp_path / "relations.csv"

    articles_path.write_text(
        "pmid|title|authors|journal|year|abstract\n"
        "123|Title|Author|Journal|2026|Abstract\n",
        encoding="utf-8",
    )
    entities_path.write_text(
        "pmid|annotator|start_idx|end_idx|location|text_span|label\n"
        "123|ann|0|5|title|Title|DDF\n",
        encoding="utf-8",
    )
    relations_path.write_text(
        "pmid|annotator|subject_start_idx|subject_end_idx|subject_location|subject_text_span|"
        "subject_label|predicate|object_start_idx|object_end_idx|object_location|object_text_span|object_label\n"
        "123|ann|0|5|title|Title|DDF|is a|0|8|abstract|Abstract|DDF\n",
        encoding="utf-8",
    )

    articles = load_articles_csv(articles_path)
    entities = load_entities_csv(entities_path)
    relations = load_full_relations_csv(relations_path)

    assert articles.loc[0, "pmid"] == "123"
    assert entities.loc[0, "start_idx"] == 0
    assert relations.loc[0, "object_end_idx"] == 8


def test_deduplicate_entities_ignores_annotator_for_exact_span_duplicates(tmp_path):
    path = tmp_path / "entities.csv"
    path.write_text(
        "pmid|annotator|start_idx|end_idx|location|text_span|label\n"
        "1|ann1|0|3|title|Gut|anatomical location\n"
        "1|ann2|0|3|title|Gut|anatomical location\n"
        "1|ann2|4|9|title|brain|anatomical location\n",
        encoding="utf-8",
    )

    entities = load_entities_csv(path)
    deduplicated = deduplicate_entities(entities)

    assert len(entities) == 3
    assert len(deduplicated) == 2


def test_load_entities_allows_pipe_inside_text_span(tmp_path):
    path = tmp_path / "entities.csv"
    path.write_text(
        "pmid|annotator|start_idx|end_idx|location|text_span|label\n"
        "1|ann|778|788|abstract|EE|MH group|human\n",
        encoding="utf-8",
    )

    entities = load_entities_csv(path)

    assert entities.loc[0, "text_span"] == "EE|MH group"
    assert entities.loc[0, "label"] == "human"


def test_load_mention_relations_allows_pipe_inside_text_span(tmp_path):
    path = tmp_path / "mention_relations.csv"
    path.write_text(
        "pmid|annotator|subject_text_span|subject_label|predicate|object_text_span|object_label\n"
        "1|ann|EE|NOMH group|human|used by|16S rRNA gene sequencing|biomedical technique\n",
        encoding="utf-8",
    )

    relations = load_mention_relations_csv(path)

    assert relations.loc[0, "subject_text_span"] == "EE|NOMH group"
    assert relations.loc[0, "subject_label"] == "human"
    assert relations.loc[0, "predicate"] == "used by"


def test_load_full_relations_allows_pipe_inside_text_span(tmp_path):
    path = tmp_path / "full_relations.csv"
    path.write_text(
        "pmid|annotator|subject_start_idx|subject_end_idx|subject_location|subject_text_span|"
        "subject_label|predicate|object_start_idx|object_end_idx|object_location|object_text_span|object_label\n"
        "1|ann|913|925|abstract|EE|NOMH group|human|used by|952|975|abstract|"
        "16S rRNA gene sequencing|biomedical technique\n",
        encoding="utf-8",
    )

    relations = load_full_relations_csv(path)

    assert relations.loc[0, "subject_text_span"] == "EE|NOMH group"
    assert relations.loc[0, "subject_label"] == "human"
    assert relations.loc[0, "object_text_span"] == "16S rRNA gene sequencing"


def test_validation_report_counts_offsets_and_duplicates(tmp_path):
    data_root = tmp_path / "gutbrainie2026"
    _write_gold_fixture(data_root)

    report = build_validation_report(data_root, "gold")

    assert report["articles"] == 1
    assert report["raw_entities"] == 3
    assert report["entities"] == 2
    assert report["duplicate_entities_removed"] == 1
    assert report["relations"] == 1
    assert report["offset_checks_passed"] == 1
    assert report["offset_checks_failed"] == 1
    assert report["missing_articles"] == 0


def test_official_csv_files_load_when_present():
    data_root = pytest.importorskip("pathlib").Path("data/gutbrainie2026")
    if not data_root.exists():
        pytest.skip("Official GutBrainIE data is not present.")

    for quality in SUPPORTED_QUALITIES:
        loaded = load_split(data_root, quality)
        assert len(loaded["articles"]) > 0
        assert len(loaded["entities"]) > 0
        assert len(loaded["mention_relations"]) > 0
        assert len(loaded["full_relations"]) > 0


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
        "1|ann2|0|3|title|Gut|anatomical location\n"
        "1|ann|9|13|title|axis|DDF\n",
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
