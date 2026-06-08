"""Convert project HTML inline SVG visualizations to pdfLaTeX-friendly TikZ."""

from __future__ import annotations

import argparse
from html import unescape
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML_FILES = [
    PROJECT_ROOT / "docs" / "article_mesh_graph.html",
    PROJECT_ROOT / "docs" / "node2vec_embedding_overview.html",
    PROJECT_ROOT / "docs" / "node2vec_seed_37656239.html",
    PROJECT_ROOT / "docs" / "node2vec_seed_35867728.html",
]
OUTPUT_NAMES = {
    "article_mesh_graph.html": "fig-article-mesh-graph.tex",
    "node2vec_embedding_overview.html": "fig-node2vec-overview.tex",
    "node2vec_seed_37656239.html": "fig-node2vec-seed-37656239.tex",
    "node2vec_seed_35867728.html": "fig-node2vec-seed-35867728.tex",
}

CLASS_COLORS = {
    "edge": "B9C0BA",
    "similarity-edge": "94A3B8",
    "article-node": "2563EB",
    "mesh-node": "16A34A",
    "seed-node": "F97316",
}


def main() -> int:
    args = parse_args()
    for html_path in args.html:
        output_path = html_path.with_name(OUTPUT_NAMES.get(html_path.name, f"{html_path.stem}_tikz.tex"))
        output_path.write_text(convert_html_svg_to_tikz(html_path), encoding="utf-8")
        print(f"Saved {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "html",
        nargs="*",
        type=Path,
        default=DEFAULT_HTML_FILES,
        help="HTML files containing one inline SVG.",
    )
    return parser.parse_args()


def convert_html_svg_to_tikz(path: Path) -> str:
    html = path.read_text(encoding="utf-8")
    svg = _extract_svg(html)
    width = _float_attr(svg, "width", 1180.0)
    height = _float_attr(svg, "height", 760.0)

    color_registry: dict[str, str] = {}
    commands: list[str] = []
    commands.extend(_line_commands(svg, color_registry))
    commands.extend(_circle_commands(svg, color_registry))
    commands.extend(_text_commands(svg))

    color_defs = "\n".join(
        f"\\definecolor{{{name}}}{{HTML}}{{{value}}}"
        for name, value in sorted(color_registry.items())
    )

    return (
        "% Auto-generated from "
        f"{path.name}. Regenerate with scripts/html_svg_to_tikz.py.\n"
        f"{color_defs}\n"
        "\\begin{tikzpicture}[x=1cm,y=-1cm,line cap=round,line join=round]\n"
        f"  \\path[use as bounding box] (0,0) rectangle ({width / 100:.3f},{height / 100:.3f});\n"
        + "\n".join(commands)
        + "\n\\end{tikzpicture}\n"
    )


def _extract_svg(html: str) -> str:
    match = re.search(r"<svg\b.*?</svg>", html, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("No inline SVG found.")
    return match.group(0)


def _line_commands(svg: str, colors: dict[str, str]) -> list[str]:
    commands = []
    for tag in re.findall(r"<line\b[^>]*>", svg, flags=re.IGNORECASE):
        attrs = _attrs(tag)
        x1 = _float(attrs.get("x1"))
        y1 = _float(attrs.get("y1"))
        x2 = _float(attrs.get("x2"))
        y2 = _float(attrs.get("y2"))
        css_class = attrs.get("class", "edge").split()[0]
        color = _color_name(colors, CLASS_COLORS.get(css_class, "94A3B8"))
        opacity = "0.34" if css_class == "similarity-edge" else "0.55"
        width = "0.25pt" if css_class == "similarity-edge" else "0.22pt"
        commands.append(
            f"  \\draw[{color}, opacity={opacity}, line width={width}] "
            f"({_cm(x1)},{_cm(y1)}) -- ({_cm(x2)},{_cm(y2)});"
        )
    return commands


def _circle_commands(svg: str, colors: dict[str, str]) -> list[str]:
    commands = []
    for tag in re.findall(r"<circle\b[^>]*>", svg, flags=re.IGNORECASE):
        attrs = _attrs(tag)
        cx = _float(attrs.get("cx"))
        cy = _float(attrs.get("cy"))
        radius = _float(attrs.get("r"))
        color_value = _circle_color(attrs)
        color = _color_name(colors, color_value)
        stroke = "white" if attrs.get("class", "") != "seed-node" else "black"
        commands.append(
            f"  \\filldraw[fill={color}, draw={stroke}, line width=0.18pt] "
            f"({_cm(cx)},{_cm(cy)}) circle ({radius / 100:.3f}cm);"
        )
    return commands


def _text_commands(svg: str) -> list[str]:
    commands = []
    pattern = re.compile(r"<text\b([^>]*)>(.*?)</text>", flags=re.DOTALL | re.IGNORECASE)
    for attrs_text, body in pattern.findall(svg):
        attrs = _attrs(attrs_text)
        css_class = attrs.get("class", "")
        if "dot-label" not in css_class and "article-dot-label" not in css_class:
            continue
        x = _float(attrs.get("x"))
        y = _float(attrs.get("y"))
        label = _latex_escape(unescape(re.sub(r"<[^>]+>", "", body).strip()))
        commands.append(
            f"  \\node[font=\\scriptsize\\bfseries, text=white, inner sep=0pt] "
            f"at ({_cm(x)},{_cm(y)}) {{{label}}};"
        )
    return commands


def _attrs(tag: str) -> dict[str, str]:
    return {
        key: value
        for key, value in re.findall(r"([A-Za-z_:][-A-Za-z0-9_:.]*)=\"([^\"]*)\"", tag)
    }


def _circle_color(attrs: dict[str, str]) -> str:
    fill = attrs.get("fill", "").strip()
    if fill.startswith("#"):
        return fill[1:].upper()
    css_class = attrs.get("class", "").split()[0]
    return CLASS_COLORS.get(css_class, "2563EB")


def _color_name(colors: dict[str, str], value: str) -> str:
    normalized = value.replace("#", "").upper()
    for name, existing in colors.items():
        if existing == normalized:
            return name
    name = f"htmlColor{len(colors) + 1}"
    colors[name] = normalized
    return name


def _float_attr(text: str, attr: str, default: float) -> float:
    match = re.search(rf"{attr}=\"([0-9.]+)\"", text)
    if not match:
        return default
    return _float(match.group(1), default)


def _float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except ValueError:
        return default


def _cm(value: float) -> str:
    return f"{value / 100:.3f}"


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


if __name__ == "__main__":
    raise SystemExit(main())
