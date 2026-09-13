"""The committed sample corpus must stay loadable, well-formed and renderable without an LLM."""
import glob
import json
import os

import pytest

from knowledge_graph.main import load_triples_from_json
from knowledge_graph.query import GraphIndex, match_entities, retrieve_subgraph
from knowledge_graph.visualization import ENTITY_TYPES, build_graph_data

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples")
SAMPLE_JSON = [os.path.join(SAMPLES_DIR, name + ".json") for name in
               ("apollo-program", "coffee-supply-chain", "industrial-revolution", "la-alhambra", "marie-curie")]
METHODS = {"llm_bridge", "llm_hub", "llm_within", "taxonomy", "transitive", "lexical"}


def test_every_sample_text_has_a_graph():
    texts = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(SAMPLES_DIR, "*.txt"))}
    graphs = {os.path.splitext(os.path.basename(p))[0] for p in SAMPLE_JSON}
    assert texts and texts == graphs


@pytest.mark.parametrize("path", SAMPLE_JSON, ids=[os.path.basename(p) for p in SAMPLE_JSON])
def test_sample_graph_is_well_formed(path):
    triples = load_triples_from_json(path)
    assert len(triples) >= 40
    extracted = [t for t in triples if not t.get("inferred")]
    inferred = [t for t in triples if t.get("inferred")]
    assert len(extracted) >= 30 and len(inferred) >= 5
    for t in extracted:
        assert t["subject"] != t["object"] and t["predicate"]
        assert t.get("subject_type") in ENTITY_TYPES and t.get("object_type") in ENTITY_TYPES
        assert t.get("chunk")
    # A few triples paraphrase entities that never appear verbatim; nearly all must carry their sentence.
    assert sum(1 for t in extracted if t.get("source")) >= 0.95 * len(extracted)
    for t in inferred:
        assert t["method"] in METHODS
    data = build_graph_data(triples)
    assert data["meta"]["stats"]["edges"] == len(triples)
    assert data["meta"]["stats"]["communities"] >= 3
    assert len(data["meta"]["types"]) >= 4


@pytest.mark.parametrize("path", SAMPLE_JSON, ids=[os.path.basename(p) for p in SAMPLE_JSON])
def test_sample_graph_is_mostly_connected(path):
    triples = load_triples_from_json(path)
    index = GraphIndex.build(triples)
    hub = index.entities[0]
    reached = retrieve_subgraph(index, [hub], hops=6, max_triples=10_000)
    assert len(reached) >= 0.8 * len(triples), f"{path}: only {len(reached)} of {len(triples)} triples reachable from {hub}"


def test_sample_questions_find_their_entities():
    graphs = {os.path.basename(p): load_triples_from_json(p) for p in SAMPLE_JSON}
    checks = {
        "marie-curie.json": ("Who discovered radium?", "radium"),
        "apollo-program.json": ("Who walked on the Moon during Apollo 11?", "apollo 11"),
        "coffee-supply-chain.json": ("What happens during roasting?", "roast"),
        "la-alhambra.json": ("¿Quién construyó el Patio de los Leones?", "leones"),
    }
    for name, (question, fragment) in checks.items():
        if name not in graphs:
            continue
        seeds = match_entities(GraphIndex.build(graphs[name]), question)
        assert any(fragment in s.lower() for s in seeds), f"{name}: {question!r} matched {seeds}"


def test_samples_are_valid_json_and_have_named_communities():
    from knowledge_graph.main import load_graph_meta
    for path in SAMPLE_JSON:
        raw = json.load(open(path, encoding="utf-8"))
        assert isinstance(raw, list)
        names = load_graph_meta(path).get("community_names") or {}
        assert len(names) >= 3, f"{path} has no community-name sidecar"
