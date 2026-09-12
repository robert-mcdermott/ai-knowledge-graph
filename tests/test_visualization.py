import json
import re

import pytest

from knowledge_graph.visualization import (
    COMMUNITY_PALETTE,
    SAMPLE_TRIPLES,
    community_color,
    visualize_knowledge_graph,
)


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    out = tmp_path_factory.mktemp("viz") / "graph.html"
    stats = visualize_knowledge_graph(SAMPLE_TRIPLES, str(out))
    return out.read_text(encoding="utf-8"), stats


def test_stats_count_every_triple_including_parallel_edges(page):
    _, stats = page
    assert stats["edges"] == len(SAMPLE_TRIPLES)
    assert stats["inferred_edges"] == 2
    assert stats["original_edges"] == len(SAMPLE_TRIPLES) - 2
    assert stats["nodes"] == len({t["subject"] for t in SAMPLE_TRIPLES} | {t["object"] for t in SAMPLE_TRIPLES})


def test_output_is_self_contained(page):
    html, _ = page
    assert "bootstrap" not in html.lower()
    assert "cdn.jsdelivr.net" not in html and "cdnjs.cloudflare.com" not in html
    assert "vis-network" in html  # embedded library


def test_template_script_runs_after_network_is_created(page):
    html, _ = page
    assert html.index("drawGraph();") < html.index("<!-- KG_SCRIPT")
    assert html.index("<!-- KG_SCRIPT") < html.index("</body>")
    assert "MutationObserver" not in html


def test_no_invalid_vis_options(page):
    html, _ = page
    options = re.search(r"var options = (\{.*?\});\s*\n", html, re.DOTALL)
    assert options, "options block not found"
    parsed = json.loads(options.group(1))
    assert "tooltipDelay" not in parsed["nodes"]
    assert "background" not in parsed
    assert "network.setOptions({ background" not in html


def test_nodes_and_edges_carry_metadata(page):
    html, _ = page
    nodes = json.loads(re.search(r"nodes = new vis.DataSet\((\[.*?\])\);", html, re.DOTALL).group(1))
    edges = json.loads(re.search(r"edges = new vis.DataSet\((\[.*?\])\);", html, re.DOTALL).group(1))
    assert all("community" in n and "degree" in n for n in nodes)
    assert all(n["color"] in COMMUNITY_PALETTE for n in nodes)
    inferred = [e for e in edges if e.get("inferred")]
    assert len(inferred) == 2
    assert {e["method"] for e in inferred} == {"taxonomy", "transitive"}
    assert any(e.get("via") == "steam engine" for e in inferred)
    assert all(e.get("dashes") for e in inferred)
    # parallel edges kept: "steam engine" has two distinct "spread to" targets and one "lead to"
    assert sum(1 for e in edges if e["from"] == "steam engine") == 6
    assert any("Inferred (transitive) via steam engine" in e["title"] for e in inferred)


def test_title_and_legend_present(page):
    html, _ = page
    assert "<title>Knowledge Graph" in html
    assert 'id="inferred-toggle"' in html
    assert "{{KG_TITLE}}" not in html


def test_palette_wraps_and_has_no_pure_yellow():
    assert community_color(0) == COMMUNITY_PALETTE[0]
    assert community_color(len(COMMUNITY_PALETTE)) == COMMUNITY_PALETTE[0]
    assert "#ffff33" not in COMMUNITY_PALETTE and len(set(COMMUNITY_PALETTE)) == 20


def test_communities_numbered_largest_first(tmp_path):
    triples = [{"subject": f"a{i}", "predicate": "p", "object": "hub"} for i in range(6)]
    triples += [{"subject": "x", "predicate": "p", "object": "y"}]
    out = tmp_path / "g.html"
    visualize_knowledge_graph(triples, str(out))
    nodes = json.loads(re.search(r"nodes = new vis.DataSet\((\[.*?\])\);", out.read_text(), re.DOTALL).group(1))
    by_id = {n["id"]: n["community"] for n in nodes}
    assert by_id["hub"] == 0 and by_id["x"] == by_id["y"] != 0


def test_empty_input(tmp_path):
    assert visualize_knowledge_graph([], str(tmp_path / "e.html"))["nodes"] == 0
