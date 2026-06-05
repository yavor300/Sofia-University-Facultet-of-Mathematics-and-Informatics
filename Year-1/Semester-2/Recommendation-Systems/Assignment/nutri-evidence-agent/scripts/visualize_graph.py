"""Export a teacher-friendly HTML visualization of the Article-MeSH graph."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "artifacts" / "article_mesh_graph.gpickle"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "article_mesh_graph.html"


def main() -> int:
    try:
        args = parse_args()

        import networkx as nx

        from src.graph.graph_builder import ArticleMeshGraphBuilder, article_node_id

        graph = ArticleMeshGraphBuilder().load(str(args.graph))
        if graph.number_of_nodes() == 0:
            raise ValueError(f"Graph has no nodes: {args.graph}")

        if args.seed_pmid:
            subgraph, title = _seed_neighborhood(
                graph=graph,
                seed_node=article_node_id(args.seed_pmid),
                max_articles=args.max_articles,
                max_mesh_terms=args.max_mesh_terms,
            )
        else:
            subgraph, title = _overview_sample(
                graph=graph,
                max_articles=args.max_articles,
                max_mesh_terms=args.max_mesh_terms,
            )

        if subgraph.number_of_nodes() == 0:
            raise ValueError("No nodes selected for visualization.")

        html = _render_html(
            graph=graph,
            subgraph=subgraph,
            title=title,
            seed_pmid=args.seed_pmid,
            layout_seed=args.layout_seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html, encoding="utf-8")

        article_nodes = _count_nodes(subgraph, "article")
        mesh_nodes = _count_nodes(subgraph, "mesh_term")
        print(
            "Graph visualization saved successfully: "
            f"{args.output} ({article_nodes} articles, {mesh_nodes} MeSH terms, "
            f"{subgraph.number_of_edges()} edges)"
        )
        return 0
    except Exception as exc:
        print(f"Error exporting graph visualization: {exc}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed-pmid", help="Focus on one article and its MeSH neighborhood.")
    parser.add_argument("--max-articles", type=int, default=35)
    parser.add_argument("--max-mesh-terms", type=int, default=40)
    parser.add_argument("--layout-seed", type=int, default=7)
    return parser.parse_args()


def _seed_neighborhood(graph, seed_node: str, max_articles: int, max_mesh_terms: int):
    if seed_node not in graph:
        raise ValueError(f"Seed article is not in the graph: {seed_node}")

    mesh_neighbors = [
        node
        for node in graph.neighbors(seed_node)
        if graph.nodes[node].get("node_type") == "mesh_term"
    ]
    mesh_neighbors.sort(key=lambda node: graph.degree(node), reverse=True)
    selected_mesh = mesh_neighbors[: max(0, max_mesh_terms)]

    article_scores: list[tuple[int, int, str]] = []
    selected_mesh_set = set(selected_mesh)
    for node, data in graph.nodes(data=True):
        if node == seed_node or data.get("node_type") != "article":
            continue

        shared_count = len(selected_mesh_set & set(graph.neighbors(node)))
        if shared_count:
            article_scores.append((shared_count, graph.degree(node), node))

    article_scores.sort(reverse=True)
    selected_articles = [node for _, _, node in article_scores[: max(0, max_articles - 1)]]
    selected_nodes = {seed_node, *selected_mesh, *selected_articles}
    subgraph = graph.subgraph(selected_nodes).copy()
    title = f"Article-MeSH neighborhood for PMID {graph.nodes[seed_node].get('pmid', seed_node)}"
    return subgraph, title


def _overview_sample(graph, max_articles: int, max_mesh_terms: int):
    mesh_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == "mesh_term"
    ]
    mesh_nodes.sort(key=lambda node: graph.degree(node), reverse=True)
    selected_mesh = mesh_nodes[: max(0, max_mesh_terms)]

    article_scores: dict[str, int] = {}
    for mesh_node in selected_mesh:
        for neighbor in graph.neighbors(mesh_node):
            if graph.nodes[neighbor].get("node_type") == "article":
                article_scores[neighbor] = article_scores.get(neighbor, 0) + 1

    selected_articles = sorted(
        article_scores,
        key=lambda node: (article_scores[node], graph.degree(node)),
        reverse=True,
    )[: max(0, max_articles)]
    selected_nodes = {*selected_mesh, *selected_articles}
    subgraph = graph.subgraph(selected_nodes).copy()
    return subgraph, "Article-MeSH graph overview sample"


def _render_html(graph, subgraph, title: str, seed_pmid: str | None, layout_seed: int) -> str:
    import networkx as nx

    width = 1180
    height = 760
    margin = 70
    positions = nx.spring_layout(subgraph, seed=layout_seed, k=0.75, iterations=120)
    scaled = _scale_positions(positions, width=width, height=height, margin=margin)
    seed_node = f"article:{seed_pmid.strip()}" if seed_pmid else ""
    article_codes, article_rows = _article_codes_and_rows(subgraph, seed_node)

    edges_svg = []
    for source, target in subgraph.edges():
        x1, y1 = scaled[source]
        x2, y2 = scaled[target]
        edges_svg.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="edge" />'
        )

    nodes_svg = []
    dot_labels_svg = []
    labels_svg = []
    for node, data in subgraph.nodes(data=True):
        x, y = scaled[node]
        node_type = data.get("node_type", "")
        css_class = "article-node" if node_type == "article" else "mesh-node"
        radius = 12 if node_type == "article" else 6
        if node == seed_node:
            css_class = "seed-node"
            radius = 15

        tooltip = _tooltip(node, data, graph.degree(node))
        nodes_svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" class="{css_class}">'
            f"<title>{escape(tooltip)}</title></circle>"
        )
        if node_type == "article":
            dot_labels_svg.append(
                f'<text x="{x:.1f}" y="{y + 3.2:.1f}" class="article-dot-label">'
                f"{escape(article_codes.get(node, 'A'))}</text>"
            )

        label = _node_label(data)
        if node == seed_node or (node_type == "mesh_term" and graph.degree(node) >= 6):
            labels_svg.append(
                f'<text x="{x + radius + 4:.1f}" y="{y + 4:.1f}" class="node-label">'
                f"{escape(_truncate(label, 34))}</text>"
            )

    article_count = _count_nodes(subgraph, "article")
    mesh_count = _count_nodes(subgraph, "mesh_term")
    subtitle = (
        f"Showing {article_count} article nodes, {mesh_count} MeSH term nodes, "
        f"and {subgraph.number_of_edges()} Article-MeSH edges."
    )
    article_table_html = _article_table_html(article_rows)

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
    p {{
      margin: 8px 0;
      color: #4b5563;
      line-height: 1.45;
    }}
    .panel {{
      margin-top: 20px;
      background: #ffffff;
      border: 1px solid #d9ded7;
      border-radius: 8px;
      overflow: auto;
    }}
    svg {{
      display: block;
      min-width: {width}px;
    }}
    .edge {{
      stroke: #b9c0ba;
      stroke-width: 1.15;
      opacity: 0.62;
    }}
    .article-node {{
      fill: #2563eb;
      stroke: #ffffff;
      stroke-width: 1.5;
    }}
    .mesh-node {{
      fill: #16a34a;
      stroke: #ffffff;
      stroke-width: 1.25;
    }}
    .seed-node {{
      fill: #f97316;
      stroke: #7c2d12;
      stroke-width: 2;
    }}
    .node-label {{
      fill: #253044;
      font-size: 11px;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 3px;
      stroke-linejoin: round;
    }}
    .article-dot-label {{
      fill: #ffffff;
      font-size: 8px;
      font-weight: 700;
      text-anchor: middle;
      pointer-events: none;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-top: 14px;
      color: #374151;
      font-size: 14px;
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
    .article {{ background: #2563eb; }}
    .mesh {{ background: #16a34a; }}
    .seed {{ background: #f97316; }}
    .article-list {{
      margin-top: 18px;
      background: #ffffff;
      border: 1px solid #d9ded7;
      border-radius: 8px;
      overflow: auto;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 760px;
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
    code {{
      background: #eef1ee;
      border-radius: 4px;
      padding: 2px 4px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    <p>{escape(subtitle)}</p>
    <p>Blue nodes are PubMed articles, green nodes are MeSH terms, and edges mean that an article was indexed with that MeSH term. Article IDs are printed inside the blue nodes and listed below. Hover over a node to inspect metadata.</p>
    <div class="legend">
      <span><i class="dot article"></i> Article</span>
      <span><i class="dot mesh"></i> MeSH term</span>
      <span><i class="dot seed"></i> Selected seed article</span>
    </div>
    <div class="panel">
      <svg width="{width}" height="{height}" role="img" aria-label="{escape(title)}">
        {"".join(edges_svg)}
        {"".join(nodes_svg)}
        {"".join(dot_labels_svg)}
        {"".join(labels_svg)}
      </svg>
    </div>
    {article_table_html}
  </main>
</body>
</html>
"""


def _article_codes_and_rows(subgraph, seed_node: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    article_nodes = [
        node
        for node, data in subgraph.nodes(data=True)
        if data.get("node_type") == "article"
    ]
    article_nodes.sort(
        key=lambda node: (
            0 if node == seed_node else 1,
            str(subgraph.nodes[node].get("pmid", "")),
        )
    )

    codes: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    article_number = 1
    for node in article_nodes:
        data = subgraph.nodes[node]
        if node == seed_node:
            code = "S"
        else:
            code = f"A{article_number}"
            article_number += 1

        codes[node] = code
        rows.append(
            {
                "code": code,
                "pmid": str(data.get("pmid", "")),
                "year": str(data.get("year") or ""),
                "title": str(data.get("title", "")),
                "journal": str(data.get("journal", "")),
            }
        )

    return codes, rows


def _article_table_html(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""

    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>"
            f"<td><strong>{escape(row['code'])}</strong></td>"
            f"<td>{escape(row['pmid'])}</td>"
            f"<td>{escape(row['year'])}</td>"
            f"<td>{escape(_truncate(row['title'], 110))}</td>"
            f"<td>{escape(_truncate(row['journal'], 60))}</td>"
            "</tr>"
        )

    return (
        '<section class="article-list">'
        "<table>"
        "<thead><tr><th>Node</th><th>PMID</th><th>Year</th><th>Title</th><th>Journal</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</section>"
    )


def _scale_positions(positions: dict[str, Any], width: int, height: int, margin: int) -> dict[str, tuple[float, float]]:
    xs = [float(position[0]) for position in positions.values()]
    ys = [float(position[1]) for position in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 0.001)
    span_y = max(max_y - min_y, 0.001)

    scaled = {}
    for node, position in positions.items():
        x = margin + ((float(position[0]) - min_x) / span_x) * (width - 2 * margin)
        y = margin + ((float(position[1]) - min_y) / span_y) * (height - 2 * margin)
        scaled[node] = (x, y)
    return scaled


def _tooltip(node: str, data: dict[str, Any], degree: int) -> str:
    if data.get("node_type") == "article":
        return "\n".join(
            [
                f"Article PMID: {data.get('pmid', '')}",
                f"Title: {data.get('title', '')}",
                f"Journal: {data.get('journal', '')}",
                f"Year: {data.get('year', '')}",
                f"MeSH edges: {degree}",
            ]
        )

    return "\n".join(
        [
            f"MeSH term: {data.get('label', node)}",
            f"Connected articles: {degree}",
        ]
    )


def _node_label(data: dict[str, Any]) -> str:
    if data.get("node_type") == "article":
        return f"PMID {data.get('pmid', '')}"
    return str(data.get("label", "MeSH"))


def _truncate(text: str, max_length: int) -> str:
    text = str(text or "")
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "..."


def _count_nodes(graph, node_type: str) -> int:
    return sum(1 for _, data in graph.nodes(data=True) if data.get("node_type") == node_type)


if __name__ == "__main__":
    raise SystemExit(main())
