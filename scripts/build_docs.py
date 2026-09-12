#!/usr/bin/env python3
"""Regenerate the GitHub Pages site (docs/) from the sample corpus in data/samples/.

    python scripts/build_docs.py            # writes docs/index.html, docs/samples/*.html, docs/vendor/

No LLM is needed: community names come from the *.meta.json sidecars written by generate-graph.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from jinja2 import Environment, FileSystemLoader  # noqa: E402

from knowledge_graph.main import load_graph_meta, load_triples_from_json  # noqa: E402
from knowledge_graph.visualization import TEMPLATE_DIR, build_graph_data, render_html  # noqa: E402

SAMPLES = [
    # (file stem, title, domain)
    ("industrial-revolution", "The Industrial Revolutions", "Technology history"),
    ("marie-curie", "Marie Curie", "Biography"),
    ("apollo-program", "The Apollo Program", "Program history"),
    ("coffee-supply-chain", "Coffee, from farm to cup", "Process description"),
    ("la-alhambra", "La Alhambra de Granada", "Spanish text"),
]


def build(samples_dir=None, docs_dir=None, model="deepseek-v4.1-flash"):
    samples_dir = samples_dir or os.path.join(ROOT, "data", "samples")
    docs_dir = docs_dir or os.path.join(ROOT, "docs")
    pages_dir = os.path.join(docs_dir, "samples")
    vendor_dir = os.path.join(docs_dir, "vendor")
    os.makedirs(pages_dir, exist_ok=True)

    cards = []
    for stem, title, domain in SAMPLES:
        json_path = os.path.join(samples_dir, f"{stem}.json")
        if not os.path.exists(json_path):
            print(f"skipping {stem}: {json_path} not found")
            continue
        triples = load_triples_from_json(json_path)
        meta = load_graph_meta(json_path)
        graph_data = build_graph_data(triples, community_names=meta.get("community_names"))
        page_path = os.path.join(pages_dir, f"{stem}.html")
        with open(page_path, "w", encoding="utf-8") as f:
            f.write(render_html(graph_data, library_dir=vendor_dir, page_dir=pages_dir))
        stats = graph_data["meta"]["stats"]
        names = [c["name"] for c in graph_data["meta"]["communities"] if c.get("name")]
        cards.append({"href": f"samples/{stem}.html", "title": title, "domain": domain, "nodes": stats["nodes"],
                      "edges": stats["edges"], "inferred": stats["inferred_edges"], "communities": stats["communities"],
                      "community_names": names[:6]})
        print(f"wrote {os.path.relpath(page_path, ROOT)}  ({stats['nodes']} nodes, {stats['edges']} relationships)")

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)
    index = env.get_template("landing.html.j2").render(graphs=cards, model=model)
    with open(os.path.join(docs_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index)
    print(f"wrote {os.path.relpath(os.path.join(docs_dir, 'index.html'), ROOT)}  ({len(cards)} examples)")
    return cards


if __name__ == "__main__":
    build()
