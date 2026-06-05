"""Build and persist Article-MeSH knowledge graphs."""

from __future__ import annotations

from pathlib import Path
import pickle
import re
from typing import Any

import networkx as nx

from src.preprocessing.article_preprocessor import filter_valid_articles


_WHITESPACE_RE = re.compile(r"\s+")


class ArticleMeshGraphBuilder:
    """Build a bipartite graph connecting PubMed articles to MeSH terms."""

    def build(self, articles: list[dict]) -> nx.Graph:
        graph = nx.Graph()

        for article in filter_valid_articles(articles):
            pmid = str(article.get("pmid", "")).strip()
            if not pmid:
                continue

            article_node = article_node_id(pmid)
            graph.add_node(
                article_node,
                node_type="article",
                pmid=pmid,
                title=article.get("title", ""),
                year=article.get("year"),
                journal=article.get("journal", ""),
            )

            for mesh_label in article.get("mesh_terms", []):
                label = _clean_label(mesh_label)
                if not label:
                    continue

                mesh_node = mesh_node_id(label)
                if mesh_node not in graph:
                    graph.add_node(
                        mesh_node,
                        node_type="mesh_term",
                        label=label,
                    )

                graph.add_edge(
                    article_node,
                    mesh_node,
                    edge_type="has_mesh_term",
                )

        return graph

    def save(self, graph: nx.Graph, path: str) -> None:
        graph_path = Path(path)
        graph_path.parent.mkdir(parents=True, exist_ok=True)

        with graph_path.open("wb") as file:
            pickle.dump(graph, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> nx.Graph:
        with Path(path).open("rb") as file:
            graph = pickle.load(file)

        if not isinstance(graph, nx.Graph):
            raise TypeError(f"Expected a NetworkX Graph in {path}")

        return graph


def article_node_id(pmid: str) -> str:
    return f"article:{str(pmid).strip()}"


def mesh_node_id(label: str) -> str:
    return f"mesh:{normalize_mesh_label(label)}"


def normalize_mesh_label(label: Any) -> str:
    cleaned = _clean_label(label).lower()
    return _WHITESPACE_RE.sub("_", cleaned)


def _clean_label(label: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(label or "")).strip()
