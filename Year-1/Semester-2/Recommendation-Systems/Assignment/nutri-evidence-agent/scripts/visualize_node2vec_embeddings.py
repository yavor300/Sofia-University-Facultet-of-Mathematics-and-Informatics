"""Export tutor-friendly node2vec embedding visualizations as standalone HTML."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.graph_builder import ArticleMeshGraphBuilder, article_node_id
from src.graph.node2vec_trainer import Node2VecTrainer


DEFAULT_ARTICLES_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_mesh_graph.gpickle"
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "artifacts" / "node2vec_embeddings.kv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "node2vec_embedding_overview.html"


def main() -> int:
    try:
        args = parse_args()
        graph = ArticleMeshGraphBuilder().load(str(args.graph))
        vectors = Node2VecTrainer().load(str(args.embeddings))
        articles = _load_articles(args.articles)

        article_nodes = _available_article_nodes(graph, vectors)
        if not article_nodes:
            raise ValueError("No article nodes with node2vec embeddings were found.")

        if args.seed_pmid:
            selected_nodes, neighbor_rows, title, subtitle = _seed_view(
                graph=graph,
                vectors=vectors,
                article_nodes=article_nodes,
                articles=articles,
                seed_pmid=args.seed_pmid,
                max_articles=args.max_articles,
            )
        else:
            selected_nodes, neighbor_rows, title, subtitle = _overview_view(
                graph=graph,
                vectors=vectors,
                article_nodes=article_nodes,
                articles=articles,
                max_articles=args.max_articles,
            )

        html = _render_html(
            graph=graph,
            vectors=vectors,
            articles=articles,
            nodes=selected_nodes,
            neighbor_rows=neighbor_rows,
            title=title,
            subtitle=subtitle,
            seed_pmid=args.seed_pmid,
            max_edges=args.max_similarity_edges,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html, encoding="utf-8")

        print(
            "node2vec visualization saved successfully: "
            f"{args.output} ({len(selected_nodes)} article embeddings)"
        )
        return 0
    except Exception as exc:
        print(f"Error exporting node2vec visualization: {exc}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed-pmid", help="Focus the visualization on one seed article.")
    parser.add_argument("--max-articles", type=int, default=80)
    parser.add_argument("--max-similarity-edges", type=int, default=90)
    return parser.parse_args()


def _load_articles(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Article cache must be a JSON list: {path}")

    articles: dict[str, dict[str, Any]] = {}
    for article in data:
        if not isinstance(article, dict):
            continue
        pmid = str(article.get("pmid", "")).strip()
        if pmid:
            articles[pmid] = article
    return articles


def _available_article_nodes(graph, vectors) -> list[str]:
    nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == "article" and node in vectors
    ]
    nodes.sort(key=lambda node: graph.degree(node), reverse=True)
    return nodes


def _overview_view(graph, vectors, article_nodes: list[str], articles: dict[str, dict], max_articles: int):
    selected_nodes = article_nodes[: max(2, max_articles)]
    neighbor_rows = _nearest_neighbor_rows(
        graph=graph,
        vectors=vectors,
        articles=articles,
        seed_node=None,
        candidate_nodes=selected_nodes,
        limit=12,
    )
    title = "node2vec Article Embedding Map"
    subtitle = (
        "This view projects article-node embeddings learned from random walks over the "
        "Article-MeSH graph. Nearby blue points are articles that node2vec considers "
        "structurally similar because they appear in similar MeSH neighborhoods."
    )
    return selected_nodes, neighbor_rows, title, subtitle


def _seed_view(
    graph,
    vectors,
    article_nodes: list[str],
    articles: dict[str, dict],
    seed_pmid: str,
    max_articles: int,
):
    seed_node = article_node_id(seed_pmid)
    if seed_node not in vectors:
        raise ValueError(f"Seed article has no node2vec embedding: {seed_pmid}")

    seed_vector = _normalized(vectors[seed_node])
    scored: list[tuple[float, str]] = []
    for node in article_nodes:
        if node == seed_node:
            continue
        scored.append((float(np.dot(seed_vector, _normalized(vectors[node]))), node))

    scored.sort(reverse=True)
    selected_nodes = [seed_node] + [node for _, node in scored[: max(1, max_articles - 1)]]
    neighbor_rows = _nearest_neighbor_rows(
        graph=graph,
        vectors=vectors,
        articles=articles,
        seed_node=seed_node,
        candidate_nodes=selected_nodes,
        limit=min(15, max_articles - 1),
    )
    title = f"node2vec Nearest Articles for PMID {seed_pmid}"
    subtitle = (
        "This seed-focused view shows the articles closest to the selected PMID in "
        "node2vec embedding space. The similarity is learned from the Article-MeSH "
        "graph, not from title or abstract text."
    )
    return selected_nodes, neighbor_rows, title, subtitle


def _nearest_neighbor_rows(
    graph,
    vectors,
    articles: dict[str, dict],
    seed_node: str | None,
    candidate_nodes: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if seed_node:
        source_vector = _normalized(vectors[seed_node])
        scored = [
            (float(np.dot(source_vector, _normalized(vectors[node]))), node)
            for node in candidate_nodes
            if node != seed_node
        ]
        seed_terms = _mesh_terms(graph, seed_node)
    else:
        scored = []
        for node in candidate_nodes[:limit]:
            best_score = -1.0
            for other in candidate_nodes:
                if other == node:
                    continue
                score = float(np.dot(_normalized(vectors[node]), _normalized(vectors[other])))
                best_score = max(best_score, score)
            scored.append((best_score, node))
        seed_terms = set()

    scored.sort(reverse=True)
    rows: list[dict[str, Any]] = []
    for rank, (score, node) in enumerate(scored[:limit], start=1):
        pmid = _pmid_from_node(node)
        article = articles.get(pmid, {})
        shared_terms = sorted(seed_terms & _mesh_terms(graph, node)) if seed_terms else []
        rows.append(
            {
                "rank": rank,
                "pmid": pmid,
                "title": article.get("title") or graph.nodes[node].get("title", ""),
                "year": article.get("year") or graph.nodes[node].get("year") or "",
                "journal": article.get("journal") or graph.nodes[node].get("journal", ""),
                "score": score,
                "source_query": article.get("source_query", ""),
                "shared_mesh_terms": shared_terms,
            }
        )
    return rows


def _render_html(
    graph,
    vectors,
    articles: dict[str, dict],
    nodes: list[str],
    neighbor_rows: list[dict[str, Any]],
    title: str,
    subtitle: str,
    seed_pmid: str | None,
    max_edges: int,
) -> str:
    points = _project_nodes(vectors, nodes)
    scaled = _scale_points(points, width=1160, height=720, margin=70)
    colors = _source_query_colors(articles, nodes)
    seed_node = article_node_id(seed_pmid) if seed_pmid else ""
    edges = _similarity_edges(vectors, nodes, max_edges=max_edges)

    edges_svg = []
    for source, target, score in edges:
        x1, y1 = scaled[source]
        x2, y2 = scaled[target]
        width = 0.7 + max(0.0, score) * 1.6
        edges_svg.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'class="similarity-edge" style="stroke-width:{width:.2f}">'
            f"<title>node2vec cosine similarity: {score:.3f}</title></line>"
        )

    article_codes, article_rows = _article_codes_and_rows(graph, articles, nodes, seed_node)
    nodes_svg = []
    labels_svg = []
    for node in nodes:
        x, y = scaled[node]
        pmid = _pmid_from_node(node)
        article = articles.get(pmid, {})
        source_query = str(article.get("source_query", "") or "unknown")
        fill = "#f97316" if node == seed_node else colors.get(source_query, "#2563eb")
        radius = 15 if node == seed_node else 10
        tooltip = "\n".join(
            [
                f"PMID: {pmid}",
                f"Title: {article.get('title') or graph.nodes[node].get('title', '')}",
                f"Source query: {source_query}",
                f"MeSH degree: {graph.degree(node)}",
            ]
        )
        nodes_svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" '
            f'class="article-node"><title>{escape(tooltip)}</title></circle>'
        )
        labels_svg.append(
            f'<text x="{x:.1f}" y="{y + 3.2:.1f}" class="dot-label">'
            f"{escape(article_codes[node])}</text>"
        )

    legend_html = _legend_html(colors)
    neighbors_html = _neighbor_table_html(neighbor_rows)
    article_table_html = _article_table_html(article_rows)
    seed_note = (
        f"<p>The orange node is the selected seed article PMID <code>{escape(seed_pmid)}</code>.</p>"
        if seed_pmid
        else ""
    )

    total_article_embeddings = sum(
        1
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == "article" and node in vectors
    )
    mesh_nodes = sum(1 for _, data in graph.nodes(data=True) if data.get("node_type") == "mesh_term")
    vector_size = getattr(vectors, "vector_size", "?")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #172033;
      background: #f7f7f4;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.2;
    }}
    h2 {{
      margin: 24px 0 10px;
      font-size: 20px;
    }}
    p {{
      margin: 8px 0;
      color: #4b5563;
      line-height: 1.45;
    }}
    code {{
      background: #eef1ee;
      border-radius: 4px;
      padding: 2px 4px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .stat {{
      border: 1px solid #d9ded7;
      border-radius: 8px;
      background: white;
      padding: 12px;
    }}
    .stat strong {{
      display: block;
      font-size: 20px;
      color: #111827;
    }}
    .panel {{
      margin-top: 18px;
      background: #ffffff;
      border: 1px solid #d9ded7;
      border-radius: 8px;
      overflow: auto;
    }}
    svg {{
      display: block;
      min-width: 1160px;
    }}
    .similarity-edge {{
      stroke: #94a3b8;
      opacity: 0.36;
    }}
    .article-node {{
      stroke: #ffffff;
      stroke-width: 1.6;
    }}
    .dot-label {{
      fill: #ffffff;
      font-size: 8px;
      font-weight: 700;
      text-anchor: middle;
      pointer-events: none;
    }}
    .axis-label {{
      fill: #64748b;
      font-size: 12px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin-top: 14px;
      color: #374151;
      font-size: 13px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      display: inline-block;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 880px;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f1f5f9;
      color: #243044;
      font-weight: 700;
    }}
    .small {{
      font-size: 12px;
      color: #64748b;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    <p>{escape(subtitle)}</p>
    <p>This is a visualization of <strong>node2vec embeddings</strong>, not the raw Article-MeSH edges. node2vec learns vectors from random walks over the graph, so proximity in this map means structural similarity in biomedical concept space.</p>
    {seed_note}
    <div class="stats">
      <div class="stat"><strong>{total_article_embeddings}</strong><span>article embeddings available</span></div>
      <div class="stat"><strong>{mesh_nodes}</strong><span>MeSH nodes in graph</span></div>
      <div class="stat"><strong>{graph.number_of_edges()}</strong><span>Article-MeSH edges</span></div>
      <div class="stat"><strong>{escape(str(vector_size))}</strong><span>node2vec dimensions</span></div>
    </div>
    {legend_html}
    <div class="panel">
      <svg width="1160" height="720" role="img" aria-label="{escape(title)}">
        <text x="32" y="700" class="axis-label">PCA component 1</text>
        <text x="32" y="24" class="axis-label">PCA projection of node2vec vectors</text>
        {"".join(edges_svg)}
        {"".join(nodes_svg)}
        {"".join(labels_svg)}
      </svg>
    </div>
    {neighbors_html}
    {article_table_html}
  </main>
</body>
</html>
"""


def _project_nodes(vectors, nodes: list[str]) -> dict[str, tuple[float, float]]:
    from sklearn.decomposition import PCA

    matrix = np.vstack([_normalized(vectors[node]) for node in nodes])
    if len(nodes) == 1:
        return {nodes[0]: (0.0, 0.0)}
    if len(nodes) == 2:
        return {nodes[0]: (-1.0, 0.0), nodes[1]: (1.0, 0.0)}

    projected = PCA(n_components=2, random_state=7).fit_transform(matrix)
    return {
        node: (float(projected[index, 0]), float(projected[index, 1]))
        for index, node in enumerate(nodes)
    }


def _scale_points(points: dict[str, tuple[float, float]], width: int, height: int, margin: int):
    xs = [point[0] for point in points.values()]
    ys = [point[1] for point in points.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 0.001)
    span_y = max(max_y - min_y, 0.001)

    scaled = {}
    for node, (x_value, y_value) in points.items():
        x = margin + ((x_value - min_x) / span_x) * (width - 2 * margin)
        y = height - margin - ((y_value - min_y) / span_y) * (height - 2 * margin)
        scaled[node] = (x, y)
    return scaled


def _similarity_edges(vectors, nodes: list[str], max_edges: int) -> list[tuple[str, str, float]]:
    edges: list[tuple[str, str, float]] = []
    node_set = set(nodes)
    for source in nodes:
        scored: list[tuple[float, str]] = []
        source_vector = _normalized(vectors[source])
        for target in nodes:
            if target == source:
                continue
            scored.append((float(np.dot(source_vector, _normalized(vectors[target]))), target))
        scored.sort(reverse=True)
        for score, target in scored[:2]:
            if target in node_set:
                edge = tuple(sorted([source, target]))
                edges.append((edge[0], edge[1], score))

    dedup: dict[tuple[str, str], float] = {}
    for source, target, score in edges:
        dedup[(source, target)] = max(score, dedup.get((source, target), -1.0))

    output = [(source, target, score) for (source, target), score in dedup.items()]
    output.sort(key=lambda item: item[2], reverse=True)
    return output[: max(0, max_edges)]


def _source_query_colors(articles: dict[str, dict], nodes: list[str]) -> dict[str, str]:
    palette = [
        "#2563eb",
        "#16a34a",
        "#9333ea",
        "#dc2626",
        "#0891b2",
        "#ca8a04",
        "#475569",
        "#db2777",
        "#059669",
        "#7c3aed",
    ]
    queries = []
    for node in nodes:
        query = str(articles.get(_pmid_from_node(node), {}).get("source_query", "") or "unknown")
        if query not in queries:
            queries.append(query)
    return {query: palette[index % len(palette)] for index, query in enumerate(queries)}


def _legend_html(colors: dict[str, str]) -> str:
    items = [
        f'<span><i class="dot" style="background:{escape(color)}"></i>{escape(_truncate(query, 46))}</span>'
        for query, color in colors.items()
    ]
    items.append('<span><i class="dot" style="background:#f97316"></i>Seed article</span>')
    return f'<div class="legend">{"".join(items)}</div>'


def _article_codes_and_rows(
    graph,
    articles: dict[str, dict],
    nodes: list[str],
    seed_node: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    codes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    article_number = 1
    for node in nodes:
        pmid = _pmid_from_node(node)
        article = articles.get(pmid, {})
        code = "S" if node == seed_node else f"A{article_number}"
        if node != seed_node:
            article_number += 1
        codes[node] = code
        rows.append(
            {
                "code": code,
                "pmid": pmid,
                "year": article.get("year") or graph.nodes[node].get("year") or "",
                "title": article.get("title") or graph.nodes[node].get("title", ""),
                "journal": article.get("journal") or graph.nodes[node].get("journal", ""),
                "source_query": article.get("source_query", ""),
                "mesh_degree": graph.degree(node),
            }
        )
    return codes, rows


def _neighbor_table_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""

    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row['rank']}</td>"
            f"<td>{escape(row['pmid'])}</td>"
            f"<td>{row['score']:.3f}</td>"
            f"<td>{escape(str(row['year']))}</td>"
            f"<td>{escape(_truncate(row['title'], 100))}</td>"
            f"<td>{escape(_truncate(row['source_query'], 50))}</td>"
            f"<td>{escape(', '.join(row['shared_mesh_terms']) or 'not shown')}</td>"
            "</tr>"
        )
    return (
        "<h2>Nearest articles in node2vec space</h2>"
        '<div class="panel"><table>'
        "<thead><tr><th>Rank</th><th>PMID</th><th>Cosine</th><th>Year</th>"
        "<th>Title</th><th>Source query</th><th>Shared MeSH with seed</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _article_table_html(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td><strong>{escape(row['code'])}</strong></td>"
            f"<td>{escape(row['pmid'])}</td>"
            f"<td>{escape(str(row['year']))}</td>"
            f"<td>{escape(_truncate(row['title'], 110))}</td>"
            f"<td>{escape(_truncate(row['journal'], 55))}</td>"
            f"<td>{escape(_truncate(row['source_query'], 52))}</td>"
            f"<td>{row['mesh_degree']}</td>"
            "</tr>"
        )
    return (
        "<h2>Article nodes shown in the map</h2>"
        '<div class="panel"><table>'
        "<thead><tr><th>Node</th><th>PMID</th><th>Year</th><th>Title</th>"
        "<th>Journal</th><th>Source query</th><th>MeSH edges</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _mesh_terms(graph, article_node: str) -> set[str]:
    if article_node not in graph:
        return set()
    terms = set()
    for neighbor in graph.neighbors(article_node):
        data = graph.nodes[neighbor]
        if data.get("node_type") == "mesh_term":
            terms.add(str(data.get("label", neighbor)))
    return terms


def _normalized(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0:
        return array
    return array / norm


def _pmid_from_node(node: str) -> str:
    return node.split("article:", 1)[-1]


def _truncate(text: Any, max_length: int) -> str:
    value = str(text or "")
    if len(value) <= max_length:
        return value
    return value[: max_length - 1] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
