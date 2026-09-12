"""Visualization utilities for knowledge graphs.

Renders subject/predicate/object triples as a single self-contained interactive
HTML page: the vendored vis-network library plus the project's own explorer UI
(``templates/graph.html.j2``). No network access is needed to view the result.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import shutil

import networkx as nx
from jinja2 import Environment, FileSystemLoader

log = logging.getLogger("knowledge_graph.visualization")

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

ENTITY_TYPES = ("person", "organization", "place", "event", "technology", "product", "work", "date", "concept")
# vis-network shapes with the label drawn outside the shape, one per entity type.
TYPE_SHAPES = {
    "person": "diamond", "organization": "square", "place": "triangle", "event": "star",
    "technology": "hexagon", "product": "hexagon", "work": "triangleDown", "date": "square", "concept": "dot",
}
TYPE_GLYPHS = {"diamond": "◆", "square": "■", "triangle": "▲", "star": "★", "hexagon": "⬢", "triangleDown": "▼", "dot": "●"}


_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "via", "with"}


def display_name(name):
    """Title-case an all-lower-case entity name for display ("steam engine" -> "Steam Engine").

    Names that already contain capitals (proper nouns, acronyms the model kept) are returned
    unchanged; so are words with digits ("mid-20th century" keeps "20th").
    """
    if not name or name != name.lower():
        return name

    def cap(word):
        return "-".join(part if any(ch.isdigit() for ch in part) else part[:1].upper() + part[1:]
                        for part in word.split("-"))

    words = name.split()
    return " ".join(cap(w) if i == 0 or w not in _SMALL_WORDS else w for i, w in enumerate(words))


def entity_types(triples):
    """Majority-vote entity type per node from the ``subject_type``/``object_type`` fields."""
    votes = {}
    for t in triples:
        for name, key in ((t["subject"], "subject_type"), (t["object"], "object_type")):
            value = t.get(key)
            if isinstance(value, str) and value.strip().lower() in ENTITY_TYPES:
                votes.setdefault(name, {}).setdefault(value.strip().lower(), 0)
                votes[name][value.strip().lower()] += 1
    return {name: max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0] for name, counts in votes.items()}


def community_color(index):
    return COMMUNITY_PALETTE[index % len(COMMUNITY_PALETTE)]


def visualize_knowledge_graph(triples, output_file="knowledge_graph.html", edge_smooth=None, config=None,
                              community_namer=None):
    """Render the page and return the statistics dict (see :func:`render_knowledge_graph`)."""
    return render_knowledge_graph(triples, output_file, edge_smooth, config, community_namer)[0]


def render_knowledge_graph(triples, output_file="knowledge_graph.html", edge_smooth=None, config=None,
                           community_namer=None, community_names=None, library_dir=None):
    """
    Create and visualize a knowledge graph from subject-predicate-object triples.

    Args:
        triples: List of dictionaries with 'subject', 'predicate', and 'object' keys
                 (optionally 'inferred', 'method', 'via', 'chunk', 'source', 'subject_type', 'object_type')
        output_file: HTML file to save the visualization
        edge_smooth: Edge smoothing setting (overrides config)
        config: Configuration dictionary (optional)
        community_namer: optional callable(list_of_community_dicts) -> {community_id: name}
        community_names: optional {community_id: name} already known (e.g. from a .meta.json sidecar);
                         when given, the namer is not called
        library_dir: write the vis-network files there and reference them from the page instead of
                     embedding them (smaller pages that share one library copy, e.g. for GitHub Pages)

    Returns:
        ``(stats, graph_data)``: the statistics dict and the data embedded in the page
        (nodes, edges, options, meta including named communities)
    """
    if edge_smooth is None:
        edge_smooth = (config or {}).get("visualization", {}).get("edge_smooth", False)

    if not triples:
        log.warning("No triples provided for visualization")
        empty = {"nodes": 0, "edges": 0, "original_edges": 0, "inferred_edges": 0, "communities": 0}
        return empty, {"nodes": [], "edges": [], "options": {}, "meta": {"stats": empty, "communities": [], "types": []}}

    log.info(f"Processing {len(triples)} triples for visualization")
    vis_cfg = (config or {}).get("visualization", {})
    graph_data = build_graph_data(triples, edge_smooth, community_names=community_names,
                                  show_inferred=vis_cfg.get("show_inferred", True),
                                  theme=vis_cfg.get("theme", "light"), edge_labels=vis_cfg.get("edge_labels", "all"),
                                  title_case=vis_cfg.get("title_case", True),
                                  collapse_parallel_edges=vis_cfg.get("collapse_parallel_edges", True))
    stats = graph_data["meta"]["stats"]
    log.info(f"Found {stats['nodes']} unique nodes")
    log.info(f"Found {stats['inferred_edges']} inferred relationships")
    log.info(f"Detected {stats['communities']} communities using Louvain method")
    if community_names:
        log.info(f"Using {len(community_names)} stored community names")
    elif community_namer is not None and stats["communities"] > 1:
        try:
            names = community_namer(graph_data["meta"]["communities"]) or {}
        except Exception as e:  # naming is cosmetic; never fail the render
            log.warning(f"community naming failed: {e}")
            names = {}
        if names:
            for entry in graph_data["meta"]["communities"]:
                if entry["id"] in names:
                    entry["name"] = names[entry["id"]]
            log.info(f"Named {len(names)} communities")

    html = render_html(graph_data, library_dir=library_dir, page_dir=os.path.dirname(os.path.abspath(output_file)))
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"Knowledge graph visualization saved to {output_file}")
    log.info(f"Graph Statistics: {json.dumps(stats, indent=2)}")
    return stats, graph_data


def build_graph_data(triples, edge_smooth=False, community_names=None, show_inferred=True,
                     theme="light", edge_labels="all", title_case=True, collapse_parallel_edges=True):
    """Compute nodes, edges, options and metadata for the page (pure data, no I/O)."""
    all_nodes = set()
    for triple in triples:
        all_nodes.add(triple["subject"])
        all_nodes.add(triple["object"])
    inferred_count = sum(1 for t in triples if t.get("inferred", False))

    # Insert nodes and edges in sorted order: Louvain depends on iteration order, and Python's
    # set order changes between processes, which would make community ids (and the stored
    # community names) drift between runs of the same graph.
    G_undirected = nx.Graph()
    G_undirected.add_nodes_from(sorted(all_nodes))
    G_undirected.add_edges_from(sorted((t["subject"], t["object"]) for t in triples))

    centrality = _calculate_centrality_metrics(G_undirected, all_nodes)
    degree = centrality["degree"]
    node_communities, community_count = _detect_communities(G_undirected, all_nodes)
    node_sizes = _calculate_node_sizes(all_nodes, centrality["betweenness"], degree, centrality["eigenvector"])

    types = entity_types(triples)
    nodes = []
    for node in sorted(all_nodes):
        community = node_communities[node]
        node_type = types.get(node)
        title = f"{node}\nConnections: {degree.get(node, 0)}\nCommunity: {community + 1}"
        if node_type:
            title += f"\nType: {node_type}"
        entry = {
            "id": node,
            "label": display_name(node) if title_case else node,
            "title": title,
            "color": community_color(community),
            "community": community,
            "degree": degree.get(node, 0),
            "size": round(node_sizes[node], 2),
            "shape": TYPE_SHAPES.get(node_type, "dot"),
        }
        if node_type:
            entry["type"] = node_type
        nodes.append(entry)

    edges = []
    for index, triple in enumerate(triples):
        is_inferred = bool(triple.get("inferred", False))
        method = triple.get("method")
        title = f"{triple['subject']} → {triple['predicate']} → {triple['object']}"
        if is_inferred:
            title += f"\nInferred ({method or 'unknown'})"
            if triple.get("via"):
                title += f" via {triple['via']}"
        elif triple.get("source"):
            title += f"\n“{triple['source']}”"
        elif triple.get("chunk"):
            title += f"\nExtracted from chunk {triple['chunk']}"
        if triple.get("document"):
            title += f"\nDocument: {triple['document']}"
        edge = {"id": f"e{index}", "from": triple["subject"], "to": triple["object"], "label": triple["predicate"],
                "title": title, "inferred": is_inferred, "arrows": "to"}
        if method:
            edge["method"] = method
        if triple.get("via"):
            edge["via"] = triple["via"]
        if triple.get("chunk"):
            edge["chunk"] = triple["chunk"]
        if triple.get("source"):
            edge["source"] = triple["source"]
        if triple.get("document"):
            edge["document"] = triple["document"]
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
        names = sorted(members.get(community, []), key=lambda n: (-degree.get(n, 0), n))
        entry = {"id": community, "color": community_color(community), "size": len(names), "top": names[:8]}
        if community_names and community_names.get(community):
            entry["name"] = community_names[community]
        communities.append(entry)
    present_types = sorted({n["type"] for n in nodes if "type" in n})
    type_legend = [{"type": t, "shape": TYPE_SHAPES[t], "glyph": TYPE_GLYPHS[TYPE_SHAPES[t]],
                    "count": sum(1 for n in nodes if n.get("type") == t)} for t in present_types]

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
            "types": type_legend,
            "freezePhysicsAbove": FREEZE_PHYSICS_ABOVE,
            "showInferred": bool(show_inferred),
            "theme": theme if theme in ("light", "dark") else "light",
            "edgeLabels": edge_labels if edge_labels in ("all", "selection", "none") else "all",
            "collapseParallelEdges": bool(collapse_parallel_edges),
            "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    }


VENDOR_FILES = ("vis-network.min.js", "vis-network.min.css")


def render_html(graph_data, library_dir=None, page_dir=None):
    """Render the explorer page.

    By default the vis-network library is embedded so the file is self-contained. With
    ``library_dir`` the library files are copied there (once) and referenced by a path relative
    to ``page_dir`` (the directory the page will be saved in).
    """
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=False)
    template = env.get_template("graph.html.j2")
    data_json = json.dumps(graph_data, ensure_ascii=False).replace("</", "<\\/")
    context = {"title": graph_data["meta"]["title"], "data_json": data_json}
    if library_dir:
        rel = ensure_library(library_dir, page_dir or os.getcwd())
        context["vis_js_href"] = f"{rel}/vis-network.min.js"
        context["vis_css_href"] = f"{rel}/vis-network.min.css"
    else:
        context["vis_js"] = _read_vendor("vis-network.min.js")
        context["vis_css"] = _read_vendor("vis-network.min.css")
    return template.render(**context)


def ensure_library(library_dir, page_dir):
    """Copy the vendored library into ``library_dir`` if missing/outdated; return the URL path from ``page_dir``."""
    os.makedirs(library_dir, exist_ok=True)
    for name in VENDOR_FILES:
        src, dst = os.path.join(VENDOR_DIR, name), os.path.join(library_dir, name)
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
            shutil.copyfile(src, dst)
    rel = os.path.relpath(os.path.abspath(library_dir), os.path.abspath(page_dir))
    return rel.replace(os.sep, "/")


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
        log.info(f"Community detection failed ({e}); using degree-based grouping")
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
        node_sizes[node] = 8 + (24 * importance ** 0.5)  # sqrt scale: hubs stand out without dwarfing the rest
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
    {"subject": "James Watt", "predicate": "developed", "object": "steam engine", "subject_type": "person",
     "object_type": "technology", "source": "The steam engine was refined by James Watt."},
    {"subject": "James Watt", "predicate": "born in", "object": "Scotland", "subject_type": "person", "object_type": "place"},
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
    log.info(f"Generating sample visualization with {len(SAMPLE_TRIPLES)} triples")
    stats = visualize_knowledge_graph(SAMPLE_TRIPLES, output_file, edge_smooth=edge_smooth, config=config)
    log.info("\nSample Knowledge Graph Statistics:")
    log.info(f"Nodes: {stats['nodes']}")
    log.info(f"Edges: {stats['edges']}")
    log.info(f"Communities: {stats['communities']}")
    log.info(f"\nVisualization saved to {output_file}")
    log.info(f"To view, open: file://{os.path.abspath(output_file)}")
    return stats


if __name__ == "__main__":
    sample_data_visualization("sample_knowledge_graph.html")
