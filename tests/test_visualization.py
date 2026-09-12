import json
import re

import pytest

from knowledge_graph.visualization import (
    COMMUNITY_PALETTE,
    SAMPLE_TRIPLES,
    build_graph_data,
    community_color,
    visualize_knowledge_graph,
)


def embedded_data(html):
    match = re.search(r"const KG = (\{.*?\});\n</script>", html, re.DOTALL)
    assert match, "embedded KG data not found"
    return json.loads(match.group(1).replace("<\\/", "</"))


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
    assert "<!DOCTYPE html>" in html
    assert re.search(r'<script[^>]+src=|<link[^>]+href=', html) is None  # nothing loaded from elsewhere
    assert "vis-network" in html and "@version" in html  # embedded library header


def test_embedded_data_matches_triples(page):
    html, _ = page
    data = embedded_data(html)
    assert len(data["edges"]) == len(SAMPLE_TRIPLES)
    assert all("community" in n and "degree" in n and n["color"] in COMMUNITY_PALETTE for n in data["nodes"])
    inferred = [e for e in data["edges"] if e["inferred"]]
    assert {e["method"] for e in inferred} == {"taxonomy", "transitive"}
    assert any(e.get("via") == "steam engine" for e in inferred)
    assert all(e.get("dashes") for e in inferred)
    assert sum(1 for e in data["edges"] if e["from"] == "steam engine") == 6  # parallel edges kept
    assert any("Inferred (transitive) via steam engine" in e["title"] for e in inferred)
    assert data["meta"]["stats"]["communities"] == len(data["meta"]["communities"])
    assert data["meta"]["communities"][0]["size"] >= data["meta"]["communities"][-1]["size"]


def test_title_and_ui_present(page):
    html, _ = page
    assert "<title>Knowledge Graph – " in html
    for element in ('id="search"', 'id="details"', 'id="legend"', 'id="btn-inferred"', 'id="export-png"'):
        assert element in html
    assert "{{" not in html  # every placeholder rendered


def test_script_breakout_is_escaped(tmp_path):
    triples = [{"subject": "</script><script>alert(1)</script>", "predicate": "p", "object": "b"}]
    out = tmp_path / "x.html"
    visualize_knowledge_graph(triples, str(out))
    html = out.read_text(encoding="utf-8")
    assert "</script><script>alert" not in html
    assert embedded_data(html)["nodes"][0]["id"].startswith("</script>")


def test_palette_wraps_and_has_no_pure_yellow():
    assert community_color(0) == COMMUNITY_PALETTE[0]
    assert community_color(len(COMMUNITY_PALETTE)) == COMMUNITY_PALETTE[0]
    assert "#ffff33" not in COMMUNITY_PALETTE and len(set(COMMUNITY_PALETTE)) == 20


def test_communities_numbered_largest_first():
    triples = [{"subject": f"a{i}", "predicate": "p", "object": "hub"} for i in range(6)]
    triples += [{"subject": "x", "predicate": "p", "object": "y"}]
    data = build_graph_data(triples)
    by_id = {n["id"]: n["community"] for n in data["nodes"]}
    assert by_id["hub"] == 0 and by_id["x"] == by_id["y"] != 0
    assert data["meta"]["communities"][0]["top"][0] == "hub"


def test_empty_input(tmp_path):
    assert visualize_knowledge_graph([], str(tmp_path / "e.html"))["nodes"] == 0
