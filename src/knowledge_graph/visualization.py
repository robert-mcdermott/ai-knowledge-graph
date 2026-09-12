"""Visualization utilities for knowledge graphs.

Renders subject/predicate/object triples as a single self-contained interactive
HTML page: the vendored vis-network library plus the project's own explorer UI
(``templates/graph.html.j2``). No network access is needed to view the result.
"""
from __future__ import annotations

import datetime as _dt
import json
import os

import networkx as nx
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
VENDOR_DIR = os.path.join(TEMPLATE_DIR, "vendor")

# 20 distinct colours that read on both white and black backgrounds (no pure yellow).
COMMUNITY_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
    "#17becf", "#bcbd22", "#393b79", "#637939",
]
INFERRED_EDGE_COLOR = "#8a8a8a"
FREEZE_PHYSICS_ABOVE = 300  # nodes; larger graphs stop simulating once laid out


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
    graph_data = build_graph_data(triples, edge_smooth)
    stats = graph_data["meta"]["stats"]
    print(f"Found {stats['nodes']} unique nodes")
    print(f"Found {stats['inferred_edges']} inferred relationships")
    print(f"Detected {stats['communities']} communities using Louvain method")

    html = render_html(graph_data)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Knowledge graph visualization saved to {output_file}")
    print(f"Graph Statistics: {json.dumps(stats, indent=2)}")
    return stats


def build_graph_data(triples, edge_smooth=False):
    """Compute nodes, edges, options and metadata for the page (pure data, no I/O)."""
    all_nodes = set()
    for triple in triples:
        all_nodes.add(triple["subject"])
        all_nodes.add(triple["object"])
    inferred_count = sum(1 for t in triples if t.get("inferred", False))

    G_undirected = nx.Graph()
    G_undirected.add_nodes_from(all_nodes)
    for triple in triples:
        G_undirected.add_edge(triple["subject"], triple["object"])

    centrality = _calculate_centrality_metrics(G_undirected, all_nodes)
    degree = centrality["degree"]
    node_communities, community_count = _detect_communities(G_undirected, all_nodes)
    node_sizes = _calculate_node_sizes(all_nodes, centrality["betweenness"], degree, centrality["eigenvector"])

    nodes = []
    for node in sorted(all_nodes):
        community = node_communities[node]
        nodes.append({
            "id": node,
            "label": node,
            "title": f"{node}\nConnections: {degree.get(node, 0)}\nCommunity: {community + 1}",
            "color": community_color(community),
            "community": community,
            "degree": degree.get(node, 0),
            "size": round(node_sizes[node], 2),
            "shape": "dot",
        })

    edges = []
    for index, triple in enumerate(triples):
        is_inferred = bool(triple.get("inferred", False))
        method = triple.get("method")
        title = f"{triple['subject']} → {triple['predicate']} → {triple['object']}"
        if is_inferred:
            title += f"\nInferred ({method or 'unknown'})"
            if triple.get("via"):
                title += f" via {triple['via']}"
        elif triple.get("chunk"):
            title += f"\nExtracted from chunk {triple['chunk']}"
        edge = {"id": f"e{index}", "from": triple["subject"], "to": triple["object"], "label": triple["predicate"],
                "title": title, "inferred": is_inferred, "arrows": "to"}
        if method:
            edge["method"] = method
        if triple.get("via"):
            edge["via"] = triple["via"]
        if triple.get("chunk"):
            edge["chunk"] = triple["chunk"]
        if is_inferred:
            edge["dashes"] = True
            edge["color"] = {"color": INFERRED_EDGE_COLOR, "opacity": 0.8}
        else:
            edge["color"] = {"inherit": "from", "opacity": 0.9}
        edges.append(edge)

    members = {}
    for node, community in node_communities.items():
        members.setdefault(community, []).append(node)
    communities = []
    for community in range(community_count):
        names = sorted(members.get(community, []), key=lambda n: -degree.get(n, 0))
        communities.append({"id": community, "color": community_color(community), "size": len(names), "top": names[:3]})

    stats = {
        "nodes": len(all_nodes),
        "edges": len(triples),
        "original_edges": len(triples) - inferred_count,
        "inferred_edges": inferred_count,
        "communities": community_count,
    }
    title = f"Knowledge Graph – {stats['nodes']} nodes, {stats['edges']} relationships, {community_count} communities"
    return {
        "nodes": nodes,
        "edges": edges,
        "options": _get_visualization_options(edge_smooth),
        "meta": {
            "title": title,
            "stats": stats,
            "communities": communities,
            "freezePhysicsAbove": FREEZE_PHYSICS_ABOVE,
            "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    }


def render_html(graph_data):
    """Render the explorer page with the graph data and vendored library embedded."""
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=False)
    template = env.get_template("graph.html.j2")
    data_json = json.dumps(graph_data, ensure_ascii=False).replace("</", "<\\/")
    return template.render(
        title=graph_data["meta"]["title"],
        data_json=data_json,
        vis_js=_read_vendor("vis-network.min.js"),
        vis_css=_read_vendor("vis-network.min.css"),
    )


def _read_vendor(name):
    with open(os.path.join(VENDOR_DIR, name), encoding="utf-8") as f:
        return f.read()


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
        return partition, len(ordered)
    except Exception as e:
        print(f"Community detection failed ({e}); using degree-based grouping")
        partition = {node: min(G_undirected.degree(node) if node in G_undirected else 0, 7) for node in all_nodes}
        return partition, len(set(partition.values()))


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


def _get_visualization_options(edge_smooth=False):
    """Options for the vis-network instance (only keys vis-network accepts)."""
    if isinstance(edge_smooth, str):
        edge_smoothing = False if edge_smooth.lower() == "false" else {"type": edge_smooth}
    elif edge_smooth:
        edge_smoothing = {"type": "continuous"}
    else:
        edge_smoothing = False
    return {
        "physics": {
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08},
            "stabilization": {"iterations": 200, "enabled": True},
        },
        "edges": {"smooth": edge_smoothing, "width": 1, "arrows": {"to": {"scaleFactor": 0.6}},
                  "font": {"size": 0, "align": "middle"}, "selectionWidth": 2, "hoverWidth": 1.5},
        "nodes": {"shape": "dot", "borderWidth": 1, "font": {"size": 14}, "scaling": {"min": 10, "max": 50}},
        "interaction": {"hover": True, "navigationButtons": True, "keyboard": False, "tooltipDelay": 150},
        "layout": {"improvedLayout": True},
    }


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
