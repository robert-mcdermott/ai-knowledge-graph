#!/usr/bin/env python3
"""
Utility script to convert existing JSON knowledge graph data to HTML visualization.
This allows testing the visualization features without running the full pipeline.
"""

import json
import sys
import os

# Add the src directory to Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from knowledge_graph.visualization import visualize_knowledge_graph

def json_to_html(json_file, output_file):
    """
    Convert JSON knowledge graph data to HTML visualization.
    
    Args:
        json_file: Path to JSON file containing triples data
        output_file: Path to save the HTML visualization
    """
    try:
        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            triples = json.load(f)
        
        print(f"Loaded {len(triples)} triples from {json_file}")
        
        # Generate HTML visualization
        stats = visualize_knowledge_graph(triples, output_file)
        
        print(f"Generated HTML visualization: {output_file}")
        print("Graph Statistics:")
        print(f"  Nodes: {stats['nodes']}")
        print(f"  Edges: {stats['edges']}")
        print(f"  Original Edges: {stats.get('original_edges', 'N/A')}")
        print(f"  Inferred Edges: {stats.get('inferred_edges', 'N/A')}")
        print(f"  Communities: {stats['communities']}")
        
        print(f"\nTo view the visualization, open: file://{os.path.abspath(output_file)}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_html.py <input.json> <output.html>")
        print("Example: python json_to_html.py docs/industrialRev.json test_inferred_filter.html")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_file = sys.argv[2]
    
    json_to_html(json_file, output_file)
