"""Visualization utilities for knowledge graphs.

Renders a list of subject/predicate/object triples as a self-contained interactive
HTML page (vis-network, inlined by PyVis, plus the project's own template).
"""
from __future__ import annotations

import html as html_lib
import json
import os
import re

import networkx as nx
from pyvis.network import Network

# 20 distinct colours that read on both white and black backgrounds (no pure yellow).
COMMUNITY_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
    "#17becf", "#bcbd22", "#393b79", "#637939",
]
INFERRED_EDGE_COLOR = "#8a8a8a"
_SCRIPT_MARKER = "<!-- KG_SCRIPT"


def _load_html_template():
    """Load the HTML template from the template file."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "graph_template.html")
    try:
        with open(template_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load template file: {e}")
        return '<div id="mynetwork" class="card-body"></div>'


def community_color(index):
    return COMMUNITY_PALETTE[index % len(COMMUNITY_PALETTE)]


def visualize_knowledge_graph(triples, output_file="knowledge_graph.html", edge_smooth=None, config=None):
    """
    Create and visualize a knowledge graph from subject-predicate-object triples.

    Args:
        triples: List of dictionaries with 'subject', 'predicate', and 'object' keys
                 (optionally 'inferred', 'method', 'via', 'chunk')
        output_file: HTML file to save the visualization
        edge_smooth: Edge smoothing setting (overrides config)
        config: Configuration dictionary (optional)

    Returns:
        Dictionary with graph statistics
    """
    if edge_smooth is None:
        edge_smooth = (config or {}).get("visualization", {}).get("edge_smooth", False)

    if not triples:
        print("Warning: No triples provided for visualization")
        return {"nodes": 0, "edges": 0, "original_edges": 0, "inferred_edges": 0, "communities": 0}

    print(f"Processing {len(triples)} triples for visualization")

    all_nodes = set()
    for triple in triples:
        all_nodes.add(triple["subject"])
        all_nodes.add(triple["object"])
    inferred_count = sum(1 for t in triples if t.get("inferred", False))
    print(f"Found {len(all_nodes)} unique nodes")
    print(f"Found {inferred_count} inferred relationships")

    # Undirected simple graph for metrics and community detection.
    G_undirected = nx.Graph()
    G_undirected.add_nodes_from(all_nodes)
    for triple in triples:
        G_undirected.add_edge(triple["subject"], triple["object"])

    centrality = _calculate_centrality_metrics(G_undirected, all_nodes)
    degree = centrality["degree"]
    node_communities, community_count = _detect_communities(G_undirected, all_nodes)
    node_sizes = _calculate_node_sizes(all_nodes, centrality["betweenness"], degree, centrality["eigenvector"])

    # Directed multigraph so parallel edges (same pair, different predicates) are all drawn.
    G = nx.MultiDiGraph()
    for node in all_nodes:
        community = node_communities[node]
        G.add_node(
            node,
            color=community_color(community),
            community=community,
            degree=degree.get(node, 0),
            label=node,
            title=f"{node}\nConnections: {degree.get(node, 0)}\nCommunity: {community + 1}",
            size=node_sizes[node],
        )
    for triple in triples:
        is_inferred = bool(triple.get("inferred", False))
        method = triple.get("method")
        title = f"{triple['subject']} → {triple['predicate']} → {triple['object']}"
        if is_inferred:
            title += f"\nInferred ({method or 'unknown'})"
            if triple.get("via"):
                title += f" via {triple['via']}"
        elif triple.get("chunk"):
            title += f"\nExtracted from chunk {triple['chunk']}"
        attrs = {"title": title, "label": triple["predicate"], "inferred": is_inferred}
        if method:
            attrs["method"] = method
        if triple.get("via"):
            attrs["via"] = triple["via"]
        if triple.get("chunk"):
            attrs["chunk"] = triple["chunk"]
        if is_inferred:
            attrs["dashes"] = True
            attrs["color"] = INFERRED_EDGE_COLOR
        G.add_edge(triple["subject"], triple["object"], **attrs)

    net = Network(
        height="100%",
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line",  # vis-network is embedded; the output needs no network access
        bgcolor="#ffffff",
        font_color="#000000",
        select_menu=False,
        filter_menu=False,
    )
    print(f"Nodes in NetworkX graph: {G.number_of_nodes()}")
    print(f"Edges in NetworkX graph: {G.number_of_edges()}")

    _add_nodes_and_edges_to_network(net, G)
    net.set_options(json.dumps(_get_visualization_options(edge_smooth)))
    _save_and_modify_html(net, output_file, community_count, all_nodes, triples)

    stats = {
        "nodes": len(all_nodes),
        "edges": len(triples),
        "original_edges": len(triples) - inferred_count,
        "inferred_edges": inferred_count,
        "communities": community_count,
    }
    print(f"Graph Statistics: {json.dumps(stats, indent=2)}")
    return stats


def _calculate_centrality_metrics(G_undirected, all_nodes):
    """Calculate centrality metrics for the graph nodes."""
    betweenness = nx.betweenness_centrality(G_undirected)
    degree = dict(G_undirected.degree())
    try:
        eigenvector = nx.eigenvector_centrality(G_undirected, max_iter=1000)
    except Exception:
        eigenvector = {node: 0.5 for node in all_nodes}
    return {"betweenness": betweenness, "degree": degree, "eigenvector": eigenvector}


def _detect_communities(G_undirected, all_nodes):
    """Detect communities (Louvain). Ids are assigned by community size, largest first."""
    try:
        communities = nx.community.louvain_communities(G_undirected, seed=42)
        ordered = sorted(communities, key=len, reverse=True)
        partition = {node: idx for idx, members in enumerate(ordered) for node in members}
        print(f"Detected {len(ordered)} communities using Louvain method")
        return partition, len(ordered)
    except Exception as e:
        print(f"Community detection failed ({e}); using degree-based grouping")
        partition = {node: min(G_undirected.degree(node) if node in G_undirected else 0, 7) for node in all_nodes}
        count = len(set(partition.values()))
        return partition, count


def _calculate_node_sizes(all_nodes, betweenness, degree, eigenvector):
    """Calculate node sizes (10-30) from a weighted mix of centrality metrics."""
    max_betweenness = max(betweenness.values()) if betweenness else 1
    max_degree = max(degree.values()) if degree else 1
    max_eigenvector = max(eigenvector.values()) if eigenvector else 1
    node_sizes = {}
    for node in all_nodes:
        degree_norm = degree.get(node, 1) / max_degree if max_degree else 0
        betweenness_norm = betweenness.get(node, 0) / max_betweenness if max_betweenness > 0 else 0
        eigenvector_norm = eigenvector.get(node, 0) / max_eigenvector if max_eigenvector > 0 else 0
        importance = 0.5 * degree_norm + 0.3 * betweenness_norm + 0.2 * eigenvector_norm
        node_sizes[node] = 10 + (20 * importance)
    return node_sizes


def _add_nodes_and_edges_to_network(net, G):
    """Add nodes and edges from the NetworkX multigraph to the PyVis network."""
    for node_id, data in G.nodes(data=True):
        net.add_node(
            node_id,
            color=data.get("color", "#4e79a7"),
            label=str(node_id),
            title=str(data.get("title", node_id)),
            shape="dot",
            size=data.get("size", 10),
            community=data.get("community", 0),
            degree=data.get("degree", 0),
        )
    for source, target, _key, data in G.edges(keys=True, data=True):
        options = {"arrows": "to"}
        options.update(data)
        net.add_edge(source, target, **options)


def _get_visualization_options(edge_smooth=False):
    """Options for the vis-network instance (only keys vis-network accepts)."""
    physics_options = {
        "enabled": True,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 100,
            "springConstant": 0.08,
        },
        "stabilization": {"iterations": 200, "enabled": True},
    }
    if isinstance(edge_smooth, str):
        edge_smoothing = False if edge_smooth.lower() == "false" else {"type": edge_smooth}
    elif edge_smooth:
        edge_smoothing = {"type": "continuous"}
    else:
        edge_smoothing = False
    return {
        "physics": physics_options,
        "edges": {"color": {"inherit": True}, "font": {"size": 11}, "smooth": edge_smoothing},
        "nodes": {"font": {"size": 14}, "scaling": {"min": 10, "max": 50}},
        "interaction": {"hover": True, "navigationButtons": True, "keyboard": True, "tooltipDelay": 200},
        "layout": {"improvedLayout": True},
    }


def _strip_external_resources(page):
    """Remove the Bootstrap CDN tags PyVis injects so the file works offline."""
    page = re.sub(r'<link[^>]*bootstrap[^>]*>\s*', "", page, flags=re.IGNORECASE)
    page = re.sub(r'<script[^>]*bootstrap[^>]*>\s*</script>\s*', "", page, flags=re.IGNORECASE)
    return page


def _save_and_modify_html(net, output_file, community_count, all_nodes, triples):
    """Render PyVis's HTML, merge in the project template, and write the file (UTF-8)."""
    net.generate_html()
    page = net.html

    title = f"Knowledge Graph – {len(all_nodes)} nodes, {len(triples)} relationships, {community_count} communities"
    template = _load_html_template().replace("{{KG_TITLE}}", html_lib.escape(title))

    # The template's <script> must run after PyVis has created `network`, so split it off
    # and append it at the end of <body>.
    if _SCRIPT_MARKER in template:
        markup, script = template.split(_SCRIPT_MARKER, 1)
        script = "<!-- KG_SCRIPT" + script
    else:
        markup, script = template, ""

    page = page.replace('<div id="mynetwork" class="card-body"></div>', markup)
    page = re.sub(r"<center>\s*<h1>.*?</h1>\s*</center>", "", page, flags=re.DOTALL)
    page = _strip_external_resources(page)
    if "<title>" not in page:
        page = page.replace("<head>", f"<head>\n        <title>{html_lib.escape(title)}</title>", 1)
    if script:
        if "</body>" in page:
            page = page.replace("</body>", script + "\n    </body>", 1)
        else:
            page += script

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Knowledge graph visualization saved to {output_file}")


SAMPLE_TRIPLES = [
    {"subject": "Industrial Revolution", "predicate": "began in", "object": "Great Britain"},
    {"subject": "Industrial Revolution", "predicate": "characterized by", "object": "machine manufacturing"},
    {"subject": "Industrial Revolution", "predicate": "led to", "object": "urbanization"},
    {"subject": "Industrial Revolution", "predicate": "led to", "object": "rise of capitalism"},
    {"subject": "Industrial Revolution", "predicate": "led to", "object": "new labor movements"},
    {"subject": "Industrial Revolution", "predicate": "fueled by", "object": "technological innovations"},
    {"subject": "James Watt", "predicate": "developed", "object": "steam engine"},
    {"subject": "James Watt", "predicate": "born in", "object": "Scotland"},
    {"subject": "Scotland", "predicate": "a country in", "object": "Europe"},
    {"subject": "steam engine", "predicate": "revolutionized", "object": "transportation"},
    {"subject": "steam engine", "predicate": "revolutionized", "object": "manufacturing processes"},
    {"subject": "steam engine", "predicate": "spread to", "object": "Europe"},
    {"subject": "steam engine", "predicate": "lead to", "object": "Industrial Revolution"},
    {"subject": "steam engine", "predicate": "spread to", "object": "North America"},
    {"subject": "technological innovations", "predicate": "led to", "object": "Digital Computers"},
    {"subject": "Digital Computers", "predicate": "enabled", "object": "Artificial Intelligence"},
    {"subject": "Artificial Intelligence", "predicate": "will replace", "object": "Humanity"},
    {"subject": "Artificial Intelligence", "predicate": "led to", "object": "LLMs"},
    {"subject": "Robert McDermott", "predicate": "likes", "object": "LLMs"},
    {"subject": "Robert McDermott", "predicate": "owns", "object": "Digital Computers"},
    {"subject": "Robert McDermott", "predicate": "lives in", "object": "North America"},
    {"subject": "steam engine", "predicate": "is a", "object": "engine", "inferred": True, "method": "taxonomy"},
    {"subject": "James Watt", "predicate": "influenced", "object": "Industrial Revolution",
     "inferred": True, "method": "transitive", "via": "steam engine"},
]


def sample_data_visualization(output_file="sample_knowledge_graph.html", edge_smooth=None, config=None):
    """Generate a visualization from built-in sample data to test the renderer."""
    if edge_smooth is None:
        edge_smooth = (config or {}).get("visualization", {}).get("edge_smooth", False)
    print(f"Generating sample visualization with {len(SAMPLE_TRIPLES)} triples")
    stats = visualize_knowledge_graph(SAMPLE_TRIPLES, output_file, edge_smooth=edge_smooth, config=config)
    print("\nSample Knowledge Graph Statistics:")
    print(f"Nodes: {stats['nodes']}")
    print(f"Edges: {stats['edges']}")
    print(f"Communities: {stats['communities']}")
    print(f"\nVisualization saved to {output_file}")
    print(f"To view, open: file://{os.path.abspath(output_file)}")
    return stats


if __name__ == "__main__":
    sample_data_visualization("sample_knowledge_graph.html")
