"""Streamlit UI for NutriEvidence Agent."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.agents.answer_generator import AnswerGenerator
from src.agents.evidence_extractor import EvidenceExtractionAgent
from src.agents.llm_recommender import LLMRecommendationReranker
from src.agents.query_planner import QueryPlannerAgent
from src.agents.recommendation_explainer import RecommendationExplainer
from src.agents.safety_checker import REQUIRED_DISCLAIMER
from src.graph.graph_builder import ArticleMeshGraphBuilder
from src.graph.node2vec_trainer import Node2VecTrainer
from src.llm.ollama_client import OllamaClient
from src.recommenders.graph_recommender import GraphRecommender
from src.recommenders.hybrid_recommender import HybridRecommender
from src.recommenders.mesh_overlap_recommender import MeshOverlapRecommender
from src.recommenders.semantic_recommender import SemanticRecommender
from src.retrieval.cache import load_articles, merge_articles
from src.retrieval.pubmed_client import PubMedClient
from src.utils.config import load_settings


PROJECT_ROOT = Path(__file__).resolve().parent
ARTICLES_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"
SEMANTIC_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embeddings.npy"
SEMANTIC_INDEX_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_embedding_index.json"
GRAPH_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_mesh_graph.gpickle"
NODE2VEC_PATH = PROJECT_ROOT / "data" / "artifacts" / "node2vec_embeddings.kv"
OPENAI_ANNOTATION_GLOB = "evaluation_annotations_openai*.csv"
GRAPH_VISUALIZATION_PATH = PROJECT_ROOT / "docs" / "article_mesh_graph.html"
NODE2VEC_OVERVIEW_PATH = PROJECT_ROOT / "docs" / "node2vec_embedding_overview.html"


def main() -> None:
    st.set_page_config(page_title="NutriEvidence Agent", layout="wide")
    settings = load_settings()

    st.title("NutriEvidence Agent")
    st.caption(
        "Educational biomedical literature recommendations from cached PubMed metadata, "
        "semantic embeddings, Article-MeSH graph signals, and local Ollama agents."
    )

    articles = load_cached_articles()
    sidebar = render_sidebar(settings, articles)

    if not articles:
        st.error("No cached articles found. Build the dataset before using the app.")
        st.code("python scripts/build_pubmed_dataset.py --max-results 50")
        return

    mode = st.tabs(["Search by research question", "Recommend by selected article"])
    with mode[0]:
        render_question_mode(articles, settings, sidebar)
    with mode[1]:
        render_article_mode(articles, settings, sidebar)

    st.divider()
    st.subheader("Safety Disclaimer")
    st.info(REQUIRED_DISCLAIMER)


def render_sidebar(settings, articles: list[dict]) -> dict[str, Any]:
    with st.sidebar:
        st.header("Configuration")
        top_k = st.slider("Results", min_value=3, max_value=10, value=5)
        use_llm = st.toggle("Use local Ollama agents", value=settings.use_llm)
        use_llm_reranker = st.toggle("Use Ollama LLM reranking", value=True)
        generate_explanations = st.toggle("Generate per-paper explanations", value=False)
        timeout = st.number_input("Ollama timeout", min_value=3, max_value=1000, value=10, step=5)
        allow_live_pubmed = st.toggle("Use live PubMed fallback", value=False)

        st.divider()
        st.metric("Cached articles", len(articles))
        st.write(f"LLM provider: `{settings.llm_provider}`")
        st.write(f"Ollama model: `{settings.ollama_model}`")
        st.write(f"Semantic artifacts: `{SEMANTIC_EMBEDDINGS_PATH.exists()}`")
        st.write(f"Graph artifacts: `{GRAPH_PATH.exists() and NODE2VEC_PATH.exists()}`")

    return {
        "top_k": top_k,
        "use_llm": use_llm,
        "use_llm_reranker": use_llm_reranker,
        "generate_explanations": generate_explanations,
        "timeout": int(timeout),
        "allow_live_pubmed": allow_live_pubmed,
    }


def render_question_mode(articles: list[dict], settings, options: dict[str, Any]) -> None:
    st.subheader("Project Description")
    st.write(
        "Ask a research-oriented biomedical question. The app plans a PubMed query, searches the cached dataset, "
        "ranks articles semantically, and generates cautious evidence-oriented text from supplied metadata."
    )

    default_question = "What is the evidence linking gut microbiome and Parkinson's disease?"
    question = st.text_area("Question input", value=default_question, height=90)

    if not st.button("Search evidence", type="primary"):
        return

    client = make_ollama_client(settings, options)
    with st.spinner("Planning PubMed query..."):
        planner = QueryPlannerAgent(client, use_llm=options["use_llm"])
        plan = planner.plan(question)
        pubmed_query = plan.get("pubmed_query") or question

    st.subheader("Generated PubMed Query")
    st.code(pubmed_query)
    with st.expander("Planner details"):
        st.json(plan)

    with st.spinner("Checking cached articles and optional live PubMed fallback..."):
        working_articles = maybe_fetch_live_articles(
            articles=articles,
            query=pubmed_query,
            settings=settings,
            options=options,
        )

    cached_matches = find_cached_matches(pubmed_query, working_articles, limit=options["top_k"])
    st.caption(f"Cached lexical matches found: {len(cached_matches)}")

    with st.spinner("Loading semantic recommender..."):
        semantic = load_semantic_recommender(working_articles)
        candidate_count = max(options["top_k"], 10) if options["use_llm_reranker"] else options["top_k"]
        candidate_recommendations = semantic.recommend_by_query(pubmed_query, top_k=candidate_count)

    rerank_output = maybe_rerank_recommendations(
        enabled=options["use_llm_reranker"],
        use_llm=options["use_llm"],
        client=client,
        user_question=question,
        candidates=candidate_recommendations,
        top_k=options["top_k"],
        caption="semantic query candidates",
    )
    recommendations = rerank_output["results"]

    st.subheader("Recommended Papers")
    render_initial_candidates(
        "Initial semantic candidates before LLM reranking",
        rerank_output,
    )
    if options["use_llm_reranker"]:
        st.markdown("#### Final LLM-reranked recommendations")
    render_recommendation_table(recommendations)
    if options["use_llm_reranker"]:
        render_openai_judgment_table(recommendations)
    if options["generate_explanations"]:
        render_explanations(None, recommendations, client, options["use_llm"])

    st.subheader("Evidence Summary")
    with st.spinner("Generating evidence summary..."):
        evidence_items = extract_evidence_items(recommendations, client, options["use_llm"])
        answer = AnswerGenerator(client, use_llm=options["use_llm"]).generate(
            user_question=question,
            recommendations=recommendations,
            evidence_items=evidence_items,
        )
    st.markdown(answer)

    st.subheader("Evaluation Note")
    st.write("Use `data/evaluation_annotations.csv` and `python scripts/evaluate_recommenders.py` to compare methods after manual relevance annotation.")


def render_article_mode(articles: list[dict], settings, options: dict[str, Any]) -> None:
    st.subheader("Question input")
    selected_label = st.selectbox("Seed article", article_options(articles))
    method_options = ["Semantic", "MeSH overlap baseline", "Graph node2vec", "Hybrid"]
    selected_methods = st.multiselect(
        "Recommendation methods",
        method_options,
        default=method_options,
    )
    seed_pmid = selected_label.split(" | ", 1)[0]
    seed_article = article_by_pmid(articles, seed_pmid)

    if seed_article:
        st.write(f"Selected PMID `{seed_pmid}`")
        st.write(seed_article.get("title", "Untitled"))

    if {"Graph node2vec", "Hybrid"} & set(selected_methods):
        render_node2vec_method_details(seed_pmid)

    if not st.button("Recommend similar papers", type="primary"):
        return
    if not selected_methods:
        st.warning("Select at least one recommendation method.")
        return

    client = make_ollama_client(settings, options)
    top_k = options["top_k"]
    candidate_count = max(top_k, 10) if options["use_llm_reranker"] else top_k
    selected_method_set = set(selected_methods)
    semantic_results: list[dict] = []
    semantic_rerank_output: dict[str, Any] | None = None
    graph_results: list[dict] = []
    graph_rerank_output: dict[str, Any] | None = None
    hybrid_results: list[dict] = []
    hybrid_rerank_output: dict[str, Any] | None = None
    mesh_results: list[dict] = []

    with st.spinner("Loading recommenders..."):
        semantic = None
        if {"Semantic", "Hybrid"} & selected_method_set:
            semantic = load_semantic_recommender(articles)
            if "Semantic" in selected_method_set:
                semantic_candidates = semantic.recommend_by_article(seed_pmid, top_k=candidate_count)
                semantic_rerank_output = maybe_rerank_recommendations(
                    enabled=options["use_llm_reranker"],
                    use_llm=options["use_llm"],
                    client=client,
                    user_question=f"Find papers similar to PMID {seed_pmid}: {seed_article.get('title', '') if seed_article else ''}",
                    candidates=semantic_candidates,
                    top_k=top_k,
                    caption="semantic article candidates",
                )
                semantic_results = semantic_rerank_output["results"]

        graph = None
        if {"Graph node2vec", "Hybrid"} & selected_method_set:
            graph = load_graph_recommender_if_available(articles)
            if graph and "Graph node2vec" in selected_method_set:
                graph_candidates = graph.recommend_by_article(seed_pmid, top_k=candidate_count)
                graph_rerank_output = maybe_rerank_recommendations(
                    enabled=options["use_llm_reranker"],
                    use_llm=options["use_llm"],
                    client=client,
                    user_question=f"Find graph-related papers similar to PMID {seed_pmid}: {seed_article.get('title', '') if seed_article else ''}",
                    candidates=graph_candidates,
                    top_k=top_k,
                    caption="graph article candidates",
                )
                graph_results = graph_rerank_output["results"]

        if "Hybrid" in selected_method_set and semantic is not None:
            hybrid_candidates = HybridRecommender(semantic, graph).recommend_by_article(seed_pmid, top_k=candidate_count)
            hybrid_rerank_output = maybe_rerank_recommendations(
                enabled=options["use_llm_reranker"],
                use_llm=options["use_llm"],
                client=client,
                user_question=f"Find papers most similar to PMID {seed_pmid}: {seed_article.get('title', '') if seed_article else ''}",
                candidates=hybrid_candidates,
                top_k=top_k,
                caption="hybrid article candidates",
            )
            hybrid_results = hybrid_rerank_output["results"]

        if "MeSH overlap baseline" in selected_method_set:
            mesh = MeshOverlapRecommender()
            mesh.fit(articles)
            mesh_results = mesh.recommend_by_article(seed_pmid, top_k=top_k)

    st.subheader("Recommended Papers Table")
    if "Semantic" in selected_method_set:
        render_method_section(
            "Semantic recommendations",
            semantic_results,
            seed_article,
            client,
            options["use_llm"],
            options["generate_explanations"],
            semantic_rerank_output,
        )

    if "Graph node2vec" in selected_method_set:
        st.subheader("Graph-based Recommendations")
        if graph_results:
            render_method_section(
                "Graph node2vec recommendations",
                graph_results,
                seed_article,
                client,
                options["use_llm"],
                options["generate_explanations"],
                graph_rerank_output,
            )
            render_node2vec_neighbor_analysis(
                seed_article=seed_article,
                graph_results=graph_results,
                rerank_output=graph_rerank_output,
            )
        else:
            st.warning("Graph artifacts are missing or no graph embeddings were available for this article.")

    if "Hybrid" in selected_method_set:
        render_method_section(
            "Hybrid recommendations",
            hybrid_results,
            seed_article,
            client,
            options["use_llm"],
            options["generate_explanations"],
            hybrid_rerank_output,
        )

    if "MeSH overlap baseline" in selected_method_set:
        render_method_section(
            "MeSH overlap baseline",
            mesh_results,
            seed_article,
            client,
            options["use_llm"],
            options["generate_explanations"],
            None,
        )

    st.subheader("Similarity Evidence Summary")
    summary_recommendations = hybrid_results or graph_results or mesh_results or semantic_results
    if not summary_recommendations:
        st.warning("No recommendations were available to summarize. Try selecting Semantic, MeSH overlap baseline, or Hybrid.")
    else:
        with st.spinner("Generating selected-article evidence summary..."):
            evidence_items = extract_evidence_items(summary_recommendations[:top_k], client, options["use_llm"])
            answer = AnswerGenerator(client, use_llm=options["use_llm"]).generate(
                user_question=f"Why are these papers similar to PMID {seed_pmid}?",
                recommendations=summary_recommendations[:top_k],
                evidence_items=evidence_items,
            )
        st.markdown(answer)

    st.subheader("Evaluation Note")
    st.write("Hybrid, semantic, graph, and MeSH overlap outputs can be annotated in `data/evaluation_annotations.csv`.")


def render_node2vec_method_details(seed_pmid: str) -> None:
    st.markdown("#### Graph node2vec method details")
    st.info(
        "Graph node2vec uses the Article-MeSH knowledge graph, not title/abstract text. "
        "It learns article vectors from random walks over article-MeSH connections and then "
        "ranks articles by cosine similarity in that learned graph embedding space."
    )

    stats = load_graph_artifact_stats()
    if stats["available"]:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Article nodes", stats["article_nodes"])
        col2.metric("MeSH nodes", stats["mesh_nodes"])
        col3.metric("Graph edges", stats["edges"])
        col4.metric("Embedded articles", stats["embedded_article_nodes"])
        st.caption(
            f"node2vec artifact: `{NODE2VEC_PATH.relative_to(PROJECT_ROOT)}` "
            f"({stats['embedded_nodes']} embedded graph nodes, {stats['vector_size']} dimensions)."
        )
    else:
        st.warning(
            "Graph artifacts are missing. Build them before using the Graph node2vec method."
        )
        st.code(
            ".venv/bin/python scripts/build_graph.py\n"
            ".venv/bin/python scripts/train_node2vec.py",
            language="bash",
        )

    with st.expander("How to interpret Graph node2vec recommendations"):
        st.markdown(
            """
- **Graph score** is cosine similarity between normalized node2vec article embeddings.
- **Shared MeSH Terms** shows exact MeSH overlap with the selected seed article when available.
- A graph recommendation can still have a good node2vec score even with few exact shared MeSH terms, because node2vec also captures indirect graph-neighborhood similarity.
- **Graph node2vec** uses only graph structure. **Hybrid** combines semantic text similarity with graph similarity using `0.6 * semantic_score + 0.4 * graph_score`.
"""
        )

    seed_visualization = PROJECT_ROOT / "docs" / f"node2vec_seed_{seed_pmid}.html"
    visualization_rows = [
        {
            "Visualization": "Raw Article-MeSH topology",
            "File": str(GRAPH_VISUALIZATION_PATH.relative_to(PROJECT_ROOT)),
            "Status": "available" if GRAPH_VISUALIZATION_PATH.exists() else "missing",
        },
        {
            "Visualization": "node2vec embedding overview",
            "File": str(NODE2VEC_OVERVIEW_PATH.relative_to(PROJECT_ROOT)),
            "Status": "available" if NODE2VEC_OVERVIEW_PATH.exists() else "missing",
        },
        {
            "Visualization": "node2vec seed-focused view",
            "File": str(seed_visualization.relative_to(PROJECT_ROOT)),
            "Status": "available" if seed_visualization.exists() else "generate if needed",
        },
    ]
    st.markdown("##### Presentation visualizations")
    st.dataframe(pd.DataFrame(visualization_rows), hide_index=True, width="stretch")
    if not seed_visualization.exists():
        st.caption("Generate a seed-specific node2vec visualization with:")
        st.code(
            f".venv/bin/python scripts/visualize_node2vec_embeddings.py "
            f"--seed-pmid {seed_pmid} "
            f"--output docs/node2vec_seed_{seed_pmid}.html "
            f"--max-articles 38 "
            f"--max-similarity-edges 70",
            language="bash",
        )


def render_node2vec_neighbor_analysis(
    seed_article: dict | None,
    graph_results: list[dict],
    rerank_output: dict[str, Any] | None,
) -> None:
    if not seed_article or not graph_results:
        return

    seed_mesh_terms = _clean_terms(seed_article.get("mesh_terms", []))
    algorithmic_candidates = (
        rerank_output.get("candidates", [])
        if rerank_output and rerank_output.get("candidates")
        else graph_results
    )

    st.markdown("##### node2vec neighbor analysis")
    st.caption(
        "This section explains the graph search result before any LLM interpretation. "
        "The neighbors below are articles that are close to the seed article in node2vec "
        "embedding space learned from Article-MeSH random walks."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Seed MeSH terms", len(seed_mesh_terms))
    col2.metric("Graph candidates inspected", len(algorithmic_candidates))
    col3.metric(
        "Candidates with exact MeSH overlap",
        sum(1 for result in algorithmic_candidates if result.get("shared_mesh_terms")),
    )

    with st.expander("Selected seed article metadata"):
        st.write(f"**PMID:** {seed_article.get('pmid', '')}")
        st.write(f"**Title:** {seed_article.get('title', 'Untitled')}")
        st.write(f"**Journal:** {seed_article.get('journal', '')}")
        st.write(f"**Year:** {seed_article.get('year', '')}")
        st.write("**Seed MeSH terms:**")
        st.write(", ".join(seed_mesh_terms) if seed_mesh_terms else "No MeSH terms available.")

    neighbor_rows = []
    for rank, result in enumerate(algorithmic_candidates, start=1):
        shared_terms = result.get("shared_mesh_terms", []) or []
        candidate_mesh_terms = _clean_terms(result.get("mesh_terms", []))
        neighbor_rows.append(
            {
                "Graph Rank": rank,
                "PMID": result.get("pmid", ""),
                "Title": result.get("title", ""),
                "Graph Cosine Score": round(float(result.get("score", result.get("graph_score", 0)) or 0), 4),
                "Shared MeSH Count": len(shared_terms),
                "Shared MeSH Terms": ", ".join(shared_terms),
                "Candidate MeSH Count": len(candidate_mesh_terms),
                "Why node2vec may link it": _node2vec_neighbor_reason(shared_terms),
            }
        )

    st.dataframe(pd.DataFrame(neighbor_rows), hide_index=True, width="stretch")
    st.info(
        "A high node2vec score means the articles are close in the learned graph embedding space. "
        "Exact shared MeSH terms are easy to inspect, but node2vec can also surface indirect neighbors "
        "that appear in similar MeSH neighborhoods even when the exact overlap is smaller."
    )


def render_method_section(
    title: str,
    results: list[dict],
    seed_article: dict | None,
    client,
    use_llm: bool,
    generate_explanations: bool,
    rerank_output: dict[str, Any] | None = None,
) -> None:
    st.markdown(f"#### {title}")
    if not results:
        st.write("No recommendations available.")
        return

    slug = slugify(title)
    render_initial_candidates(
        f"Initial {title.lower()} before LLM reranking",
        rerank_output,
    )
    if rerank_output and rerank_output.get("reranked"):
        st.markdown("##### Final LLM-reranked recommendations")
    render_recommendation_table(results)
    if rerank_output and rerank_output.get("reranked"):
        render_openai_judgment_table(results)
    if generate_explanations:
        render_explanations(seed_article, results, client, use_llm)


def maybe_rerank_recommendations(
    enabled: bool,
    use_llm: bool,
    client,
    user_question: str,
    candidates: list[dict],
    top_k: int,
    caption: str,
) -> dict[str, Any]:
    if not enabled:
        return {"results": candidates[:top_k], "candidates": candidates, "reranked": False}

    if not use_llm:
        st.warning("Ollama reranking failed. Showing algorithmic recommendations instead.")
        st.info(f"Initial algorithmic candidates: {len(candidates)}")
        st.info(f"Final LLM-reranked recommendations: {min(len(candidates), top_k)}")
        return {"results": candidates[:top_k], "candidates": candidates, "reranked": False}

    with st.spinner("Reranking candidate articles with local Ollama..."):
        reranked = LLMRecommendationReranker(client).rerank(
            user_question=user_question,
            candidate_articles=candidates,
            top_k=top_k,
        )

    if reranked and all(result.get("llm_rerank_status") == "fallback" for result in reranked):
        st.warning("Ollama reranking failed. Showing algorithmic recommendations instead.")
        st.info(f"Initial algorithmic candidates: {len(candidates)}")
        st.info(f"Final LLM-reranked recommendations: {min(len(candidates), top_k)}")
        return {"results": candidates[:top_k], "candidates": candidates, "reranked": False}

    final_results = reranked or candidates[:top_k]
    st.info(f"Initial algorithmic candidates: {len(candidates)}")
    st.info(f"Final LLM-reranked recommendations: {len(final_results)}")
    st.caption(f"LLM reranker received {len(candidates)} {caption}.")
    return {"results": final_results, "candidates": candidates, "reranked": True}


def render_initial_candidates(
    title: str,
    rerank_output: dict[str, Any] | None,
) -> pd.DataFrame | None:
    if not rerank_output or not rerank_output.get("reranked"):
        return None

    candidates = rerank_output.get("candidates", [])
    if not candidates:
        return None

    st.markdown(f"##### {title}")
    df = render_recommendation_table(candidates, show_llm_columns=False)
    return df


def render_recommendation_table(results: list[dict], show_llm_columns: bool = True) -> pd.DataFrame:
    if not results:
        st.write("No recommendations available.")
        return pd.DataFrame()

    df = build_recommendation_dataframe(results, show_llm_columns=show_llm_columns)
    st.dataframe(df, hide_index=True, width="stretch")
    return df


def build_recommendation_dataframe(results: list[dict], show_llm_columns: bool = True) -> pd.DataFrame:
    rows = []
    has_shared_mesh_terms = any(result.get("shared_mesh_terms") for result in results)
    has_llm_metadata = show_llm_columns and any(result.get("llm_rank") or result.get("llm_reason") for result in results)
    for rank, result in enumerate(results, start=1):
        row = {
            "Rank": result.get("llm_rank") or rank,
            "Title": result.get("title"),
            "PMID": result.get("pmid"),
            "Year": result.get("year"),
            "Journal": result.get("journal"),
            "Algorithmic Score": round(
                float(result.get("algorithmic_score", result.get("score", result.get("final_score", 0))) or 0),
                4,
            ),
            "Method": result.get("method"),
            "Evidence Type": result.get("evidence_type", ""),
            "Matched Concepts": ", ".join(result.get("matched_concepts", []) or []),
            "LLM Reason": result.get("llm_reason", ""),
        }
        if has_shared_mesh_terms:
            row["Shared MeSH Terms"] = ", ".join(result.get("shared_mesh_terms", []) or [])
        if has_llm_metadata and result.get("llm_rerank_status"):
            row["LLM Rerank Status"] = result.get("llm_rerank_status")
        if not has_llm_metadata:
            row["MeSH Terms"] = ", ".join(result.get("mesh_terms", []) or [])
        rows.append(row)

    return pd.DataFrame(rows)


def slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in slug.split("_") if part)


def render_openai_judgment_table(results: list[dict]) -> None:
    judgments = match_openai_judgments(results)
    if not judgments:
        st.caption("OpenAI judge labels: no matching saved annotations found.")
        return

    rows = []
    for rank, result in enumerate(results, start=1):
        judgment = judgments.get(str(result.get("pmid", "")).strip())
        if not judgment:
            continue

        human_relevance = judgment.get("human_relevance", "")
        judge_relevance = judgment.get("judge_relevance", "")
        rows.append(
            {
                "Rank": result.get("llm_rank") or rank,
                "PMID": result.get("pmid", ""),
                "Title": result.get("title", ""),
                "Effective Relevance": human_relevance or judge_relevance,
                "Judge Relevance": judge_relevance,
                "Human Relevance": human_relevance,
                "Judge Reason": judgment.get("judge_reason", ""),
                "Human Notes": judgment.get("human_notes", ""),
                "Judge Model": judgment.get("judge_model", ""),
                "Annotation File": judgment.get("source_file", ""),
            }
        )

    if not rows:
        st.caption("OpenAI judge labels: no matching saved annotations found for the final reranked PMIDs.")
        return

    st.markdown("##### OpenAI judge labels")
    st.caption(
        "Saved evaluation labels only. Human relevance overrides judge relevance when both are present."
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def match_openai_judgments(results: list[dict]) -> dict[str, dict]:
    annotations = load_openai_judgments()
    if not annotations:
        return {}

    matched: dict[str, dict] = {}
    for result in results:
        pmid = str(result.get("pmid", "")).strip()
        method = str(result.get("method", "")).strip()
        if not pmid:
            continue

        judgment = annotations.get((pmid, method)) or annotations.get((pmid, ""))
        if judgment:
            matched[pmid] = judgment

    return matched


@st.cache_data(show_spinner=False)
def load_openai_judgments() -> dict[tuple[str, str], dict]:
    judgments: dict[tuple[str, str], dict] = {}
    for path in sorted((PROJECT_ROOT / "data").glob(OPENAI_ANNOTATION_GLOB)):
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                pmid = str(row.get("pmid", "")).strip()
                if not pmid:
                    continue

                judgment = {
                    "method": str(row.get("method", "")).strip(),
                    "judge_relevance": str(row.get("judge_relevance", "")).strip(),
                    "judge_reason": str(row.get("judge_reason", "")).strip(),
                    "judge_model": str(row.get("judge_model", "")).strip(),
                    "human_relevance": str(row.get("human_relevance", "")).strip(),
                    "human_notes": str(row.get("human_notes", "")).strip(),
                    "source_file": path.name,
                }
                if not judgment["judge_relevance"] and not judgment["human_relevance"]:
                    continue

                method = judgment["method"]
                judgments[(pmid, method)] = judgment
                judgments.setdefault((pmid, ""), judgment)

    return judgments


def render_explanations(seed_article: dict | None, results: list[dict], client, use_llm: bool) -> None:
    explainer = RecommendationExplainer(client, use_llm=use_llm)
    with st.expander("Recommendation explanations"):
        for result in results:
            st.markdown(f"**PMID {result.get('pmid')} - {result.get('method')}**")
            st.write(explainer.explain(seed_article, result))


def extract_evidence_items(recommendations: list[dict], client, use_llm: bool) -> list[dict]:
    extractor = EvidenceExtractionAgent(client, use_llm=use_llm)
    return [extractor.extract(article) for article in recommendations]


@st.cache_data(show_spinner=False)
def load_cached_articles() -> list[dict]:
    return load_articles(str(ARTICLES_PATH))


@st.cache_resource(show_spinner=False)
def load_semantic_recommender(articles: list[dict]) -> SemanticRecommender:
    recommender = SemanticRecommender()
    if SEMANTIC_EMBEDDINGS_PATH.exists() and SEMANTIC_INDEX_PATH.exists():
        recommender.load_artifacts(
            articles,
            str(SEMANTIC_EMBEDDINGS_PATH),
            str(SEMANTIC_INDEX_PATH),
        )
    else:
        recommender.fit(articles)
    return recommender


@st.cache_resource(show_spinner=False)
def load_graph_recommender_if_available(articles: list[dict]) -> GraphRecommender | None:
    if not GRAPH_PATH.exists() or not NODE2VEC_PATH.exists():
        return None

    graph = ArticleMeshGraphBuilder().load(str(GRAPH_PATH))
    node2vec_model = Node2VecTrainer().load(str(NODE2VEC_PATH))
    recommender = GraphRecommender()
    recommender.fit(articles, graph, node2vec_model)
    return recommender


@st.cache_data(show_spinner=False)
def load_graph_artifact_stats() -> dict[str, Any]:
    if not GRAPH_PATH.exists() or not NODE2VEC_PATH.exists():
        return {
            "available": False,
            "article_nodes": 0,
            "mesh_nodes": 0,
            "edges": 0,
            "embedded_nodes": 0,
            "embedded_article_nodes": 0,
            "vector_size": 0,
        }

    graph = ArticleMeshGraphBuilder().load(str(GRAPH_PATH))
    vectors = Node2VecTrainer().load(str(NODE2VEC_PATH))
    article_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == "article"
    ]
    mesh_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == "mesh_term"
    ]
    return {
        "available": True,
        "article_nodes": len(article_nodes),
        "mesh_nodes": len(mesh_nodes),
        "edges": graph.number_of_edges(),
        "embedded_nodes": len(vectors),
        "embedded_article_nodes": sum(1 for node in article_nodes if node in vectors),
        "vector_size": getattr(vectors, "vector_size", 0),
    }


def make_ollama_client(settings, options: dict[str, Any]) -> OllamaClient:
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=options["timeout"],
    )


def maybe_fetch_live_articles(articles: list[dict], query: str, settings, options: dict[str, Any]) -> list[dict]:
    cached_matches = find_cached_matches(query, articles, limit=options["top_k"])
    if len(cached_matches) >= options["top_k"]:
        return articles
    if not options["allow_live_pubmed"] or not settings.ncbi_email:
        return articles

    try:
        client = PubMedClient(email=settings.ncbi_email, api_key=settings.ncbi_api_key)
        fetched = client.search_and_fetch(query, max_results=options["top_k"] * 2)
    except Exception as exc:
        st.warning(f"Live PubMed fallback was unavailable: {exc}")
        return articles

    if fetched:
        st.success(f"Fetched {len(fetched)} live PubMed articles for this session.")
        return merge_articles(articles, fetched)

    return articles


def find_cached_matches(query: str, articles: list[dict], limit: int = 5) -> list[dict]:
    terms = [term.lower() for term in query.replace('"', " ").split() if len(term) > 2]
    if not terms:
        return []

    scored = []
    for article in articles:
        haystack = f"{article.get('title', '')} {article.get('abstract', '')} {' '.join(article.get('mesh_terms', []) or [])}".lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            scored.append((score, article))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [article for _, article in scored[:limit]]


def article_options(articles: list[dict]) -> list[str]:
    return [
        f"{article.get('pmid', '')} | {article.get('title', 'Untitled')}"
        for article in articles
        if article.get("pmid")
    ]


def article_by_pmid(articles: list[dict], pmid: str) -> dict | None:
    for article in articles:
        if str(article.get("pmid", "")).strip() == str(pmid).strip():
            return article

    return None


def _clean_terms(terms: Any) -> list[str]:
    if not terms:
        return []
    if isinstance(terms, str):
        values = [terms]
    else:
        try:
            values = list(terms)
        except TypeError:
            values = [terms]

    cleaned = []
    seen = set()
    for term in values:
        value = str(term or "").strip()
        key = value.lower()
        if value and key not in seen:
            cleaned.append(value)
            seen.add(key)
    return cleaned


def _node2vec_neighbor_reason(shared_terms: list[str]) -> str:
    if len(shared_terms) >= 4:
        return "Strong exact MeSH overlap plus graph-neighborhood proximity."
    if shared_terms:
        return "Some exact MeSH overlap plus indirect node2vec graph proximity."
    return "No exact shown overlap; likely indirect similarity through nearby MeSH neighborhoods."


if __name__ == "__main__":
    main()
