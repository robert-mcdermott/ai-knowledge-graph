
from knowledge_graph import entity_standardization as es
from knowledge_graph.entity_standardization import (
    _apply_transitive_inference,
    _identify_communities,
    _infer_relationships_by_lexical_similarity,
    infer_relationships,
)


def T(s, p, o, **extra):
    return {"subject": s, "predicate": p, "object": o, **extra}


def no_llm(**inference):
    cfg = {"inference": {"use_llm_for_inference": False}}
    cfg["inference"].update(inference)
    return cfg


def inferred(result):
    return [t for t in result if t.get("inferred")]


CHAIN = [T("manchester", "located in", "england"), T("england", "located in", "europe"),
         T("watt", "invented", "steam engine")]


def test_no_rules_by_default_means_no_inferred_edges():
    result = infer_relationships(CHAIN, no_llm())
    assert inferred(result) == []
    assert len(result) == 3


def test_transitive_keeps_predicate_and_records_via():
    result = infer_relationships(CHAIN, no_llm(apply_transitive=True))
    new = inferred(result)
    assert new == [{"subject": "manchester", "predicate": "located in", "object": "europe",
                    "inferred": True, "method": "transitive", "via": "england"}]


def test_transitive_only_for_transitive_predicate_families():
    triples = [T("watt", "invented", "steam engine"), T("steam engine", "powered", "trains")]
    assert inferred(infer_relationships(triples, no_llm(apply_transitive=True))) == []


def test_transitive_requires_same_family_on_both_hops():
    triples = [T("a", "part of", "b"), T("b", "located in", "c")]
    assert inferred(infer_relationships(triples, no_llm(apply_transitive=True))) == []


def test_transitive_can_be_limited_to_chosen_groups():
    cfg = no_llm(apply_transitive=True, transitive_predicate_groups=["is a"])
    assert inferred(infer_relationships(CHAIN, cfg)) == []


def test_transitive_skips_hub_intermediates():
    hub_edges = [T("x", "led to", "technology")] + [T("technology", "led to", f"thing{i}") for i in range(12)]
    degree = es._degrees(hub_edges)
    graph = {"x": {"technology"}, "technology": {f"thing{i}" for i in range(12)}}
    assert _apply_transitive_inference(hub_edges, graph, degree, {"transitive_max_hub_degree": 10}) == []
    loose = _apply_transitive_inference(hub_edges, graph, degree,
                                        {"transitive_max_hub_degree": 100, "transitive_max_per_subject": 3})
    assert len(loose) == 3  # per-subject cap


def test_transitive_never_duplicates_existing_pair():
    triples = CHAIN + [T("europe", "contains", "manchester")]
    assert inferred(infer_relationships(triples, no_llm(apply_transitive=True))) == []


def test_lexical_is_opt_in_and_ignores_stopwords_and_short_words():
    entities = {"steam engine", "steam power", "the revolution", "industrial revolution", "iron", "iron ore"}
    triples = [T("iron", "used in", "steam engine")]
    assert inferred(infer_relationships(triples + [T("a", "b", "c")], no_llm())) == []
    new = _infer_relationships_by_lexical_similarity(entities, triples)
    pairs = {(t["subject"], t["object"], t["predicate"]) for t in new}
    assert ("steam engine", "steam power", "related to") in pairs
    assert all("revolution" not in s and "revolution" not in o for s, o, _ in pairs)  # domain stop-word
    assert all("iron" not in (s, o) or "ore" in s + o for s, o, _ in pairs)  # 'iron' too short
    assert all(t["method"] == "lexical" for t in new)


def test_budget_caps_inferred_edges_relative_to_extracted():
    triples = [T(f"a{i}", "located in", f"b{i}") for i in range(4)] + \
              [T(f"b{i}", "located in", f"c{i}") for i in range(4)]
    result = infer_relationships(triples, no_llm(apply_transitive=True, max_inferred_ratio=0.25))
    assert len(inferred(result)) == 2  # 8 extracted * 0.25
    unlimited = infer_relationships(triples, no_llm(apply_transitive=True, max_inferred_ratio=10))
    assert len(inferred(unlimited)) == 4


def test_llm_results_take_priority_over_rules(monkeypatch):
    monkeypatch.setattr(es, "_infer_bridges_with_llm",
                        lambda *a, **k: [es._make_inferred("manchester", "trades with", "europe", "llm_bridge")])
    monkeypatch.setattr(es, "_infer_hub_relationships_with_llm", lambda *a, **k: [])
    monkeypatch.setattr(es, "_infer_within_community_relationships", lambda *a, **k: [])
    cfg = {"inference": {"use_llm_for_inference": True, "apply_transitive": True, "max_inferred_ratio": 1.0}}
    new = inferred(infer_relationships(CHAIN, cfg))
    # The LLM edge for the same pair wins; the transitive duplicate is dropped.
    assert [t["method"] for t in new] == ["llm_bridge"]


def test_llm_results_drop_self_loops_and_already_connected_pairs(monkeypatch):
    monkeypatch.setattr(es, "_infer_bridges_with_llm",
                        lambda *a, **k: [es._make_inferred("england", "part of", "europe", "llm_bridge"),
                                         es._make_inferred("europe", "same", "europe", "llm_bridge")])
    monkeypatch.setattr(es, "_infer_hub_relationships_with_llm", lambda *a, **k: [])
    monkeypatch.setattr(es, "_infer_within_community_relationships", lambda *a, **k: [])
    cfg = {"inference": {"use_llm_for_inference": True}}
    assert inferred(infer_relationships(CHAIN, cfg)) == []


def test_identify_communities_is_iterative_and_undirected():
    graph = {f"n{i}": {f"n{i+1}"} for i in range(5000)}  # would overflow a recursive DFS
    graph["island"] = {"rock"}
    comms = _identify_communities(graph)
    assert len(comms) == 2
    assert {"island", "rock"} in comms


def test_llm_inference_parses_and_tags(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k): pass
        def complete(self, user, system=None):
            return '[{"subject": "a", "predicate": "influenced deeply and lastingly", "object": "b"}, {"subject": "c", "predicate": "p", "object": "c"}]'
    monkeypatch.setattr(es.LLMClient, "from_config", classmethod(lambda cls, cfg: FakeClient()))
    out = es._llm_infer({}, "s", "u", "llm_within", "test")
    assert out == [{"subject": "a", "predicate": "influenced deeply and", "object": "b",
                    "inferred": True, "method": "llm_within"}]


def test_llm_inference_maps_names_case_insensitively_and_drops_unknown(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k): pass
        def complete(self, user, system=None):
            return ('[{"subject": "Isabel De Castilla", "predicate": "reinó en", "object": "castilla"},'
                    ' {"subject": "isabel de castilla", "predicate": "visitó", "object": "atlantis"}]')
    monkeypatch.setattr(es.LLMClient, "from_config", classmethod(lambda cls, cfg: FakeClient()))
    known = {"isabel de castilla": "isabel de castilla", "castilla": "castilla"}
    out = es._llm_infer({}, "s", "u", "llm_bridge", "test", known)
    assert out == [{"subject": "isabel de castilla", "predicate": "reinó en", "object": "castilla",
                    "inferred": True, "method": "llm_bridge"}]


def test_llm_inference_error_is_swallowed(monkeypatch, caplog):
    class Boom:
        def __init__(self, *a, **k): pass
        def complete(self, *a, **k): raise RuntimeError("down")
    monkeypatch.setattr(es.LLMClient, "from_config", classmethod(lambda cls, cfg: Boom()))
    with caplog.at_level("INFO", logger="knowledge_graph"):
        assert es._llm_infer({}, "s", "u", "llm_within", "test") == []
    assert "down" in caplog.text
