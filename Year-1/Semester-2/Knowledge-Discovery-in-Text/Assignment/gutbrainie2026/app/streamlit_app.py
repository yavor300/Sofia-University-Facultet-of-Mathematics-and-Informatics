"""Interactive Streamlit demo for GutBrainIE T611/T621 experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
import plotly.express as px
import streamlit as st

from components.article_selector import article_summary_table, filter_articles
from components.entity_highlighter import compare_entities, highlight_entities, legend_html, style_block
from components.metrics_dashboard import discover_metric_files, load_metric_report, metric_sections, per_label_dataframe, summary_dataframe
from components.relation_viewer import compare_relations, relation_cards_html
from gutbrainie.data.annotations import load_entities_csv, load_mention_relations_csv
from gutbrainie.data.articles import load_articles_csv
from gutbrainie.data.splits import resolve_split_paths
from gutbrainie.evaluation.ner_metrics import evaluate_ner
from gutbrainie.evaluation.re_metrics import evaluate_mention_relations
from gutbrainie.ner.dictionary_baseline import predict_dictionary_entities
from gutbrainie.submission.export_t611 import load_t611_json
from gutbrainie.submission.export_t621 import load_t621_json

SPLITS = {
    "Train Gold": "gold",
    "Train Silver": "silver",
    "Train Silver 2025": "silver_2025",
    "Train Bronze": "bronze",
    "Dev": "dev",
    "Test": "test",
}


def main() -> None:
    st.set_page_config(page_title="GutBrainIE T611/T621 Demo", layout="wide")
    st.markdown(style_block(), unsafe_allow_html=True)
    st.title("GutBrainIE 2026 Information Extraction Demo")

    settings = sidebar_settings()
    page = st.sidebar.radio(
        "Navigation",
        [
            "Project Overview",
            "Dataset Explorer",
            "T611 NER Visualization",
            "T621 Relation Visualization",
            "Metrics Dashboard",
            "Error Analysis",
            "Single Text Demo",
        ],
    )

    if page == "Project Overview":
        project_overview(settings)
    elif page == "Dataset Explorer":
        dataset_explorer(settings)
    elif page == "T611 NER Visualization":
        ner_visualization(settings)
    elif page == "T621 Relation Visualization":
        relation_visualization(settings)
    elif page == "Metrics Dashboard":
        metrics_dashboard(settings)
    elif page == "Error Analysis":
        error_analysis(settings)
    else:
        single_text_demo(settings)


def sidebar_settings() -> dict[str, Any]:
    st.sidebar.header("Data")
    data_root = Path(st.sidebar.text_input("Data root", "data/gutbrainie2026"))
    outputs_root = Path(st.sidebar.text_input("Outputs root", "outputs"))

    t611_files = discover_files(outputs_root / "predictions", "*t611*.json")
    t621_files = discover_files(outputs_root / "predictions", "*t621*.json")
    metric_files = discover_metric_files(outputs_root / "reports")

    default_ner = choose_default(
        t611_files,
        [
            "dev_t611_pubmedbert_gold_silver_silver_2025.json",
            "dev_t611_token_classifier_gold_silver_silver_2025.json",
            "dev_t611_gliner_gold_silver_silver_2025.json",
            "dev_t611_dictionary.json",
        ],
    )
    default_re = choose_default(
        t621_files,
        [
            "dev_t621_pair_classifier_gold_silver_silver_2025_pubmedbert_entities.json",
            "dev_t621_pair_classifier_pubmedbert_entities.json",
            "dev_t621_rule_predicted_entities.json",
            "dev_t621_rule_gold_entities.json",
        ],
    )

    ner_prediction = file_selectbox("NER prediction JSON", t611_files, default_ner)
    re_prediction = file_selectbox("RE prediction JSON", t621_files, default_re)

    return {
        "data_root": data_root,
        "outputs_root": outputs_root,
        "ner_prediction": ner_prediction,
        "re_prediction": re_prediction,
        "metric_files": metric_files,
    }


def project_overview(settings: dict[str, Any]) -> None:
    st.subheader("Project Overview")
    st.write(
        "This application visualizes a biomedical information extraction pipeline for the GutBrainIE 2026 challenge. "
        "The system detects biomedical entity mentions in PubMed titles and abstracts and extracts mention-level "
        "relations between these mentions."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Task 6.1.1 / T611**")
        st.write("Named Entity Recognition over PubMed title and abstract text.")
    with col2:
        st.markdown("**Task 6.2.1 / T621**")
        st.write("Mention-level relation extraction over detected or gold entity mentions.")

    st.subheader("Implemented Model Families")
    st.table(
        pd.DataFrame(
            [
                {"Area": "NER", "Model": "Dictionary baseline", "Purpose": "Transparent lower-bound baseline"},
                {"Area": "NER", "Model": "GLiNER", "Purpose": "Zero/few-shot and fine-tuned entity extraction"},
                {"Area": "NER", "Model": "PubMedBERT/BioBERT/SciBERT token classifier", "Purpose": "Main classical NER model"},
                {"Area": "RE", "Model": "Relation prior rule baseline", "Purpose": "Transparent lower-bound baseline"},
                {"Area": "RE", "Model": "PubMedBERT pair classifier", "Purpose": "Main trainable relation model"},
                {"Area": "RE", "Model": "Ollama/GPT verifier", "Purpose": "Optional LLM-assisted comparison"},
            ]
        )
    )

    st.subheader("Current Inputs")
    st.json(
        {
            "data_root": str(settings["data_root"]),
            "ner_prediction": str(settings["ner_prediction"]) if settings["ner_prediction"] else None,
            "re_prediction": str(settings["re_prediction"]) if settings["re_prediction"] else None,
        }
    )


def dataset_explorer(settings: dict[str, Any]) -> None:
    st.subheader("Dataset Explorer")
    split_label = st.selectbox("Dataset split", list(SPLITS), index=4)
    articles, gold_entities, gold_relations = load_split_for_app(settings["data_root"], SPLITS[split_label])
    query = st.text_input("Search PMID, title, or abstract")
    filtered = filter_articles(articles, query)

    c1, c2, c3 = st.columns(3)
    c1.metric("Articles", len(articles))
    c2.metric("Entities", len(gold_entities) if gold_entities is not None else "n/a")
    c3.metric("Mention relations", len(gold_relations) if gold_relations is not None else "n/a")

    left, right = st.columns(2)
    with left:
        if gold_entities is not None and not gold_entities.empty:
            counts = gold_entities["label"].value_counts().reset_index()
            counts.columns = ["label", "count"]
            st.plotly_chart(px.bar(counts, x="label", y="count", title="Entity Label Distribution"), use_container_width=True)
        else:
            st.info("No entity annotations for this split.")
    with right:
        if gold_relations is not None and not gold_relations.empty:
            counts = gold_relations["predicate"].value_counts().reset_index()
            counts.columns = ["predicate", "count"]
            st.plotly_chart(px.bar(counts, x="predicate", y="count", title="Relation Predicate Distribution"), use_container_width=True)
        else:
            st.info("No mention-level relation annotations for this split.")

    st.subheader("Articles")
    st.dataframe(article_summary_table(filtered, gold_entities, gold_relations), use_container_width=True, hide_index=True)


def ner_visualization(settings: dict[str, Any]) -> None:
    st.subheader("T611 NER Visualization")
    articles, gold_entities, _ = load_split_for_app(settings["data_root"], "dev")
    prediction_path = settings["ner_prediction"]
    predicted = load_entities_any(prediction_path) if prediction_path else pd.DataFrame()

    article = select_article(articles)
    pmid = str(article["pmid"])
    article_gold = subset_by_pmid(gold_entities, pmid)
    article_pred = subset_by_pmid(predicted, pmid)
    comparison = compare_entities(article_gold, article_pred)

    render_article_metadata(article)
    status_mode = st.toggle("Gold comparison colors", value=gold_entities is not None and not gold_entities.empty)
    render_highlighted_article(article, comparison if status_mode else article_pred, status_mode)

    st.subheader("Entity Tables")
    tab1, tab2, tab3 = st.tabs(["Predicted", "Gold vs Prediction", "Gold"])
    with tab1:
        st.dataframe(article_pred, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(comparison.sort_values(["location", "start_idx", "end_idx", "label"]), use_container_width=True, hide_index=True)
        if gold_entities is not None and not gold_entities.empty and not predicted.empty:
            metrics = evaluate_ner(gold_entities, predicted)
            metric_row(metrics)
    with tab3:
        st.dataframe(article_gold if article_gold is not None else pd.DataFrame(), use_container_width=True, hide_index=True)


def relation_visualization(settings: dict[str, Any]) -> None:
    st.subheader("T621 Mention-Level Relation Visualization")
    articles, gold_entities, gold_relations = load_split_for_app(settings["data_root"], "dev")
    predicted_entities = load_entities_any(settings["ner_prediction"]) if settings["ner_prediction"] else pd.DataFrame()
    predicted_relations = load_relations_any(settings["re_prediction"]) if settings["re_prediction"] else pd.DataFrame()

    article = select_article(articles)
    pmid = str(article["pmid"])
    render_article_metadata(article)

    context_entities = subset_by_pmid(predicted_entities, pmid)
    if context_entities.empty and gold_entities is not None:
        context_entities = subset_by_pmid(gold_entities, pmid)
    render_highlighted_article(article, context_entities, status_mode=False)

    article_gold_rel = subset_by_pmid(gold_relations, pmid)
    article_pred_rel = subset_by_pmid(predicted_relations, pmid)
    comparison = compare_relations(article_gold_rel, article_pred_rel)

    st.subheader("Relation Cards")
    st.markdown(relation_cards_html(comparison), unsafe_allow_html=True)
    st.subheader("Relation Table")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    if gold_relations is not None and not gold_relations.empty and not predicted_relations.empty:
        st.subheader("Dev RE Metrics")
        metric_row(evaluate_mention_relations(gold_relations, predicted_relations))


def metrics_dashboard(settings: dict[str, Any]) -> None:
    st.subheader("Metrics Dashboard")
    files = settings["metric_files"]
    if not files:
        st.info("No metric JSON reports found under outputs/reports.")
        return

    summary = summary_dataframe(files)
    st.dataframe(summary.sort_values("micro_f1", ascending=False, na_position="last"), use_container_width=True, hide_index=True)

    selected = st.selectbox("Metric report", files, format_func=lambda path: path.name)
    report = load_metric_report(selected)
    sections = metric_sections(report)
    for name, metrics in sections.items():
        st.markdown(f"### {name.upper()}")
        metric_row(metrics)
        per_label = per_label_dataframe(metrics)
        if not per_label.empty:
            st.plotly_chart(
                px.bar(per_label.sort_values("f1", ascending=False), x="label", y="f1", title=f"{name.upper()} Per-label F1"),
                use_container_width=True,
            )
            st.dataframe(per_label.sort_values("f1", ascending=False), use_container_width=True, hide_index=True)


def error_analysis(settings: dict[str, Any]) -> None:
    st.subheader("Error Analysis")
    articles, gold_entities, gold_relations = load_split_for_app(settings["data_root"], "dev")
    predicted_entities = load_entities_any(settings["ner_prediction"]) if settings["ner_prediction"] else pd.DataFrame()
    predicted_relations = load_relations_any(settings["re_prediction"]) if settings["re_prediction"] else pd.DataFrame()

    if gold_entities is None or gold_entities.empty:
        st.warning("Error analysis requires dev gold annotations.")
        return

    ner_cmp = compare_entities(gold_entities, predicted_entities)
    c1, c2, c3 = st.columns(3)
    c1.metric("NER false positives", int((ner_cmp["status"] == "False Positive").sum()))
    c2.metric("NER false negatives", int((ner_cmp["status"] == "False Negative").sum()))
    c3.metric("NER true positives", int((ner_cmp["status"] == "True Positive").sum()))

    st.markdown("### Frequent NER False Positives")
    st.dataframe(error_frequency(ner_cmp, "False Positive", ["text_span", "label"]), use_container_width=True, hide_index=True)
    st.markdown("### Frequent NER False Negatives")
    st.dataframe(error_frequency(ner_cmp, "False Negative", ["text_span", "label"]), use_container_width=True, hide_index=True)
    st.markdown("### Possible Boundary Errors")
    st.dataframe(find_boundary_errors(ner_cmp), use_container_width=True, hide_index=True)

    if gold_relations is not None and not gold_relations.empty:
        re_cmp = compare_relations(gold_relations, predicted_relations)
        st.markdown("### Relation Errors")
        r1, r2, r3 = st.columns(3)
        r1.metric("RE false positives", int((re_cmp["status"] == "False Positive").sum()))
        r2.metric("RE false negatives", int((re_cmp["status"] == "False Negative").sum()))
        r3.metric("RE true positives", int((re_cmp["status"] == "True Positive").sum()))
        st.dataframe(re_cmp[re_cmp["status"] != "True Positive"].head(500), use_container_width=True, hide_index=True)
        st.markdown("### Relation FNs Caused by Missing NER Mentions")
        st.dataframe(re_missing_ner_table(re_cmp, predicted_entities), use_container_width=True, hide_index=True)


def single_text_demo(settings: dict[str, Any]) -> None:
    st.subheader("Single Text Demo")
    st.write("This lightweight demo uses the dictionary NER baseline trained from a selected entity CSV.")
    title = st.text_input("Title", "Gut microbiota is linked to depression")
    abstract = st.text_area("Abstract", "Lactobacillus altered gut microbiota and reduced inflammatory symptoms in mice.", height=160)
    train_entities_path = st.text_input(
        "Training entities CSV",
        str(settings["data_root"] / "Annotations" / "Train" / "gold_quality" / "csv_format" / "train_gold_entities.csv"),
    )
    if st.button("Run dictionary NER"):
        train_entities = load_entities_csv(train_entities_path)
        article = pd.DataFrame(
            [{"pmid": "custom", "title": title, "authors": "", "journal": "", "year": "", "abstract": abstract}]
        )
        predictions = predict_dictionary_entities(article, train_entities)
        render_highlighted_article(article.iloc[0], predictions, status_mode=False)
        st.dataframe(predictions, use_container_width=True, hide_index=True)


def render_article_metadata(article: pd.Series) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("PMID", str(article["pmid"]))
    c2.metric("Year", str(article.get("year", "")))
    c3.metric("Journal", str(article.get("journal", ""))[:48] or "n/a")


def render_highlighted_article(article: pd.Series, entities: pd.DataFrame, status_mode: bool) -> None:
    labels = sorted(entities["label"].dropna().astype(str).unique()) if entities is not None and not entities.empty else []
    st.markdown(legend_html(labels, status_mode=status_mode), unsafe_allow_html=True)
    st.markdown("#### Title")
    title_entities = entities[entities["location"] == "title"] if entities is not None and "location" in entities else pd.DataFrame()
    st.markdown(highlight_entities(str(article["title"]), title_entities, status_mode), unsafe_allow_html=True)
    st.markdown("#### Abstract")
    abstract_entities = entities[entities["location"] == "abstract"] if entities is not None and "location" in entities else pd.DataFrame()
    st.markdown(highlight_entities(str(article["abstract"]), abstract_entities, status_mode), unsafe_allow_html=True)


def select_article(articles: pd.DataFrame) -> pd.Series:
    query = st.text_input("Filter article", "")
    filtered = filter_articles(articles, query)
    if filtered.empty:
        st.warning("No articles match the filter.")
        st.stop()
    pmids = filtered["pmid"].astype(str).tolist()
    pmid = st.selectbox("PMID", pmids, format_func=lambda value: article_label(filtered, value))
    return filtered[filtered["pmid"].astype(str) == str(pmid)].iloc[0]


def metric_row(metrics: dict[str, Any]) -> None:
    cols = st.columns(6)
    for col, key in zip(cols, ["micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1"], strict=True):
        col.metric(key.replace("_", " ").title(), f"{float(metrics.get(key, 0.0)):.3f}")
    cols = st.columns(3)
    cols[0].metric("TP", int(metrics.get("tp", 0)))
    cols[1].metric("FP", int(metrics.get("fp", 0)))
    cols[2].metric("FN", int(metrics.get("fn", 0)))


def load_split_for_app(data_root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    if split == "test":
        return load_articles(data_root / "Test_Data" / "articles_test.csv"), None, None
    paths = resolve_split_paths(data_root, split)
    articles = load_articles(paths.articles)
    entities = load_entities_any(paths.entities) if paths.entities.exists() else None
    relations = load_relations_any(paths.mention_relations) if paths.mention_relations.exists() else None
    return articles, entities, relations


@st.cache_data(show_spinner=False)
def load_articles(path: str | Path) -> pd.DataFrame:
    return load_articles_csv(path)


@st.cache_data(show_spinner=False)
def load_entities_any(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return load_t611_json(path) if path.suffix == ".json" else load_entities_csv(path)


@st.cache_data(show_spinner=False)
def load_relations_any(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return load_t621_json(path) if path.suffix == ".json" else load_mention_relations_csv(path)


def subset_by_pmid(df: pd.DataFrame | None, pmid: str) -> pd.DataFrame:
    if df is None or df.empty or "pmid" not in df.columns:
        return pd.DataFrame()
    return df[df["pmid"].astype(str) == str(pmid)].copy()


def discover_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.glob(pattern) if path.is_file())


def choose_default(paths: list[Path], preferred_names: list[str]) -> Path | None:
    for name in preferred_names:
        for path in paths:
            if path.name == name:
                return path
    return paths[0] if paths else None


def file_selectbox(label: str, paths: list[Path], default: Path | None) -> Path | None:
    options: list[Path | None] = [None, *paths]
    index = options.index(default) if default in options else 0
    return st.sidebar.selectbox(label, options, index=index, format_func=lambda path: "None" if path is None else path.name)


def article_label(articles: pd.DataFrame, pmid: str) -> str:
    row = articles[articles["pmid"].astype(str) == str(pmid)].iloc[0]
    title = str(row["title"])
    return f"{pmid} | {title[:90]}"


def error_frequency(df: pd.DataFrame, status: str, columns: list[str]) -> pd.DataFrame:
    subset = df[df["status"] == status]
    if subset.empty:
        return pd.DataFrame(columns=[*columns, "count"])
    return subset.groupby(columns).size().reset_index(name="count").sort_values("count", ascending=False).head(100)


def find_boundary_errors(ner_cmp: pd.DataFrame) -> pd.DataFrame:
    fp = ner_cmp[ner_cmp["status"] == "False Positive"]
    fn = ner_cmp[ner_cmp["status"] == "False Negative"]
    rows = []
    for _, pred in fp.iterrows():
        candidates = fn[
            (fn["pmid"].astype(str) == str(pred["pmid"]))
            & (fn["location"].astype(str) == str(pred["location"]))
            & (fn["label"].astype(str) == str(pred["label"]))
        ]
        for _, gold in candidates.iterrows():
            if spans_overlap(int(pred["start_idx"]), int(pred["end_idx"]), int(gold["start_idx"]), int(gold["end_idx"])):
                rows.append(
                    {
                        "pmid": pred["pmid"],
                        "label": pred["label"],
                        "prediction": pred["text_span"],
                        "gold": gold["text_span"],
                        "location": pred["location"],
                    }
                )
    return pd.DataFrame(rows).drop_duplicates().head(100)


def spans_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start <= right_end and right_start <= left_end


def re_missing_ner_table(re_cmp: pd.DataFrame, predicted_entities: pd.DataFrame) -> pd.DataFrame:
    fns = re_cmp[re_cmp["status"] == "False Negative"]
    rows = []
    for _, rel in fns.iterrows():
        ents = subset_by_pmid(predicted_entities, str(rel["pmid"]))
        spans = set(ents["text_span"].astype(str)) if not ents.empty and "text_span" in ents else set()
        subject_missing = str(rel["subject_text_span"]) not in spans
        object_missing = str(rel["object_text_span"]) not in spans
        if subject_missing or object_missing:
            rows.append(
                {
                    "pmid": rel["pmid"],
                    "subject_text_span": rel["subject_text_span"],
                    "predicate": rel["predicate"],
                    "object_text_span": rel["object_text_span"],
                    "subject_missing_from_ner": subject_missing,
                    "object_missing_from_ner": object_missing,
                }
            )
    return pd.DataFrame(rows).head(200)


if __name__ == "__main__":
    main()
