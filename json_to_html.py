#!/usr/bin/env python3
"""Convert a saved triples JSON file to an HTML visualization (no LLM calls).

Equivalent to: generate-graph --from-json input.json --output output.html
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.knowledge_graph.config import load_config
from src.knowledge_graph.main import InputError, load_triples_from_json
from src.knowledge_graph.visualization import visualize_knowledge_graph


def json_to_html(json_file, output_file, config_file="config.toml"):
    try:
        triples = load_triples_from_json(json_file)
    except InputError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Loaded {len(triples)} triples from {json_file}")
    config = load_config(config_file) if os.path.exists(config_file) else None
    stats = visualize_knowledge_graph(triples, output_file, config=config)
    print(f"Nodes: {stats['nodes']}  Edges: {stats['edges']}  Extracted: {stats['original_edges']}  "
          f"Inferred: {stats['inferred_edges']}  Communities: {stats['communities']}")
    print(f"\nTo view the visualization, open: file://{os.path.abspath(output_file)}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python json_to_html.py <input.json> <output.html> [config.toml]")
        sys.exit(1)
    json_to_html(*sys.argv[1:])
