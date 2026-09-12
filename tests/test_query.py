import json

from knowledge_graph.query import (
    GraphChat,
    GraphIndex,
    format_facts,
    main,
    match_entities,
    parse_answer,
    retrieve_subgraph,
    shortest_path,
)

TRIPLES = [
    {"subject": "james watt", "predicate": "refined", "object": "steam engine", "source": "Watt refined the steam engine."},
    {"subject": "steam engine", "predicate": "powered", "object": "railways", "source": "Steam engines powered railways."},
    {"subject": "railways", "predicate": "connected", "object": "cities"},
    {"subject": "cities", "predicate": "grew during", "object": "industrial revolution"},
    {"subject": "steam engine", "predicate": "is a", "object": "engine", "inferred": True, "method": "taxonomy"},
    {"subject": "cotton gin", "predicate": "invented by", "object": "eli whitney"},
    {"subject": "eli whitney", "predicate": "lived in", "object": "america"},
]
CFG = {"llm": {"model": "m", "base_url": "u"}, "query": {"hops": 2, "max_triples": 50}}


def test_index_and_entities_by_degree():
    index = GraphIndex.build(TRIPLES)
    assert index.entities[0] == "steam engine"
    assert index.adjacency["railways"] == {"steam engine", "cities"}


def test_match_entities_prefers_whole_names_and_drops_contained_ones():
    index = GraphIndex.build(TRIPLES)
    assert match_entities(index, "How did the Steam Engine change cities?") == ["steam engine", "cities"]
    assert "engine" not in match_entities(index, "what about the steam engine")
    # word overlap fallback: "railway" matches "railways", "city" matches "cities"
    assert match_entities(index, "tell me about railway travel") == ["railways"]
    assert match_entities(index, "which city grew?") == ["cities"]
    assert match_entities(index, "what is the meaning of life") == []
    # a shared 5-letter prefix is not a match ("transportation" must not pull in "transistor")
    idx2 = GraphIndex.build([{"subject": "transistor", "predicate": "p", "object": "engineer"}])
    assert match_entities(idx2, "what about transportation and the engine?") == []


def test_shortest_path_and_retrieval_includes_seed_paths_first():
    index = GraphIndex.build(TRIPLES)
    assert shortest_path(index, "james watt", "cities") == ["james watt", "steam engine", "railways", "cities"]
    assert shortest_path(index, "james watt", "america") is None
    indices = retrieve_subgraph(index, ["james watt", "cities"], hops=1, max_triples=10)
    assert indices[:3] == [0, 1, 2]  # the path triples come first
    assert all(i < 5 for i in indices)  # cotton gin cluster is unreachable


def test_retrieval_caps_and_ranks_extracted_before_inferred():
    index = GraphIndex.build(TRIPLES)
    indices = retrieve_subgraph(index, ["steam engine"], hops=1, max_triples=2)
    assert len(indices) == 2 and 4 not in indices


def test_format_and_parse():
    index = GraphIndex.build(TRIPLES)
    text, numbering = format_facts(index, [0, 4])
    assert text.splitlines()[0].startswith('[1] james watt → refined → steam engine  (extracted: "Watt refined')
    assert "(inferred: taxonomy)" in text.splitlines()[1] and numbering == {1: 0, 2: 4}
    answer, cited = parse_answer("Watt refined it.\n\nFacts used: [1, 2]")
    assert answer == "Watt refined it." and cited == [1, 2]
    assert parse_answer("No facts.") == ("No facts.", [])


class FakeClient:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, user, system=None):
        self.prompts.append((system, user))
        return self.reply


def test_chat_ask_returns_cited_facts_and_keeps_history():
    client = FakeClient("The steam engine powered railways that connected cities.\nFacts used: [1, 2]")
    chat = GraphChat(TRIPLES, CFG, client=client)
    result = chat.ask("How did the steam engine change cities?")
    assert result["seeds"] == ["steam engine", "cities"]
    assert [t["predicate"] for t in result["facts"]] == ["powered", "connected"]
    assert "Facts used" not in result["answer"]
    follow_up = chat.ask("And who built it?")
    assert follow_up["seeds"] == ["steam engine", "cities"]  # previous context reused, no LLM entity pick
    assert len(client.prompts) == 2 and "Earlier in this conversation" in client.prompts[1][1]
    assert "ONLY the knowledge-graph facts" in client.prompts[0][0]


def test_chat_falls_back_to_llm_entity_pick(monkeypatch):
    class PickThenAnswer(FakeClient):
        def complete(self, user, system=None):
            self.prompts.append((system, user))
            return '["cotton gin"]' if "JSON array of matching" in user else "Whitney invented it.\nFacts used: [1]"
    client = PickThenAnswer("")
    result = GraphChat(TRIPLES, CFG, client=client).ask("who made the fiber separating machine?")
    assert result["seeds"] == ["cotton gin"] and result["facts"][0]["object"] == "eli whitney"


def test_cli_single_question_json(tmp_path, monkeypatch, capsys):
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps(TRIPLES))
    cfg = tmp_path / "c.toml"
    cfg.write_text('[llm]\nmodel = "m"\nbase_url = "http://x"\n')
    from knowledge_graph import query as q
    monkeypatch.setattr(q.LLMClient, "from_config", classmethod(lambda cls, c: FakeClient("Yes.\nFacts used: [1]")))
    main([str(graph), "Did James Watt refine the steam engine?", "--config", str(cfg), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["answer"] == "Yes." and out["facts"][0]["subject"] == "james watt"
