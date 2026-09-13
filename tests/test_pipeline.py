"""Wave 7: typed triples, provenance, parallel extraction, community naming."""
import json

import pytest

from knowledge_graph import main as m
from knowledge_graph.llm import LLMTruncatedError
from knowledge_graph.visualization import build_graph_data, entity_types

CFG = {"llm": {"model": "x", "base_url": "http://x", "concurrency": 3}, "chunking": {"chunk_size": 12, "overlap": 3},
       "standardization": {"enabled": False}, "inference": {"enabled": False}}


class FakeClient:
    def __init__(self, replies):
        self.replies = replies
        self.calls = 0

    def complete(self, user, system=None):
        self.calls += 1
        for key, reply in self.replies.items():
            if key in user:
                if isinstance(reply, Exception):
                    raise reply
                return reply
        return "[]"


def patch(monkeypatch, client):
    monkeypatch.setattr(m.LLMClient, "from_config", classmethod(lambda cls, cfg: client))


def test_normalize_triple_validates_types_and_attaches_source():
    chunk = "James Watt refined the steam engine. It changed factories."
    t = m.normalize_triple({"subject": " James Watt ", "predicate": "refined", "object": "steam engine",
                            "subject_type": "Person", "object_type": "gadget"}, chunk)
    assert t == {"subject": "James Watt", "predicate": "refined", "object": "steam engine", "subject_type": "person",
                 "evidence_status": "unverified"}


def test_process_with_llm_keeps_types_and_drops_bad_items(monkeypatch):
    reply = json.dumps([{"subject": "a", "predicate": "p", "object": "b", "subject_type": "place", "object_type": "event"},
                        {"subject": "a", "predicate": "p"}, {"subject": 5, "predicate": "p", "object": "b"}])
    patch(monkeypatch, FakeClient({"```": reply}))
    with pytest.raises(m.LLMError, match="Triple 2"):
        m.process_with_llm(CFG, "a p b.")


def test_chunks_run_in_parallel_and_keep_order(monkeypatch):
    text = " ".join(f"Sentence {i} is here now." for i in range(1, 9))  # 40 words -> several chunks of <=12
    replies = {f"Sentence {i} is": json.dumps([{"subject": f"Sentence {i}", "predicate": "is", "object": "here"}]) for i in (1, 4, 7)}
    client = FakeClient(replies)
    patch(monkeypatch, client)
    out = m.process_text_in_chunks(CFG, text)
    assert client.calls >= 3
    chunks = [t["chunk"] for t in out]
    assert chunks == sorted(chunks)  # order preserved despite concurrency
    assert all(t["evidence_status"] == "unverified" for t in out)


def test_failed_chunk_aborts_unless_continue(monkeypatch):
    text = " ".join(f"Sentence {i} is here now." for i in range(1, 9))
    client = FakeClient({"Sentence 1 is": LLMTruncatedError("cut off"),
                         "Sentence 4 is": json.dumps([{"subject": "s", "predicate": "p", "object": "o"}])})
    patch(monkeypatch, client)
    try:
        m.process_text_in_chunks(CFG, text)
        raise AssertionError("expected LLMError")
    except m.LLMError as e:
        assert "Chunk 1/" in str(e)
    out = m.process_text_in_chunks(CFG, text, continue_on_error=True)
    assert out and all(t["chunk"] != 1 for t in out)


def test_entity_types_majority_vote():
    triples = [{"subject": "watt", "predicate": "p", "object": "engine", "subject_type": "person", "object_type": "technology"},
               {"subject": "engine", "predicate": "p", "object": "x", "subject_type": "concept"},
               {"subject": "engine", "predicate": "p", "object": "y", "subject_type": "technology"},
               {"subject": "z", "predicate": "p", "object": "engine", "object_type": "bogus"}]
    assert entity_types(triples) == {"watt": "person", "engine": "technology"}


def test_graph_data_has_shapes_types_and_type_legend():
    triples = [{"subject": "watt", "predicate": "p", "object": "engine", "subject_type": "person", "object_type": "technology"},
               {"subject": "engine", "predicate": "q", "object": "mill"}]
    data = build_graph_data(triples, community_names={0: "Steam power"})
    by_id = {n["id"]: n for n in data["nodes"]}
    assert by_id["watt"]["shape"] == "diamond" and by_id["watt"]["type"] == "person"
    assert by_id["engine"]["shape"] == "hexagon"
    assert by_id["mill"]["shape"] == "dot" and "type" not in by_id["mill"]
    assert [t["type"] for t in data["meta"]["types"]] == ["person", "technology"]
    assert data["meta"]["communities"][0]["name"] == "Steam power"


def test_community_namer_parses_mapping(monkeypatch):
    patch(monkeypatch, FakeClient({"clusters": '{"1": "Steam power", "2": "  Labour  ", "x": "bad", "9": "out of range"}'}))
    namer = m.make_community_namer({"visualization": {"name_communities": True}})
    communities = [{"id": 0, "top": ["a", "b"]}, {"id": 1, "top": ["c"]}]
    assert namer(communities) == {0: "Steam power", 1: "Labour"}
    assert m.make_community_namer({"visualization": {"name_communities": False}}) is None
    assert namer([{"id": 0, "top": ["a"]}]) == {}  # a single community needs no name
