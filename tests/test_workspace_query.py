from knowledge_graph.query import GraphChat, GraphIndex, match_entities, passage_search


class Client:
    def __init__(self): self.prompts = []
    def complete(self, user, system=None):
        self.prompts.append(user)
        return 'An answer [1].\nFacts used: [1]'


def test_unicode_and_alias_matching_across_large_graph():
    triples = [{"subject": f"Hub{i}", "predicate": "has", "object": f"Item{i}"} for i in range(350)]
    triples.append({"subject": "北京", "predicate": "位于", "object": "中国", "subject_aliases": ["Beijing"]})
    index = GraphIndex.build(triples)
    assert "北京" in match_entities(index, "北京位于哪里？")
    assert "北京" in match_entities(index, "Where is Beijing?")


def test_source_passages_retrieve_paraphrased_subject():
    index = GraphIndex.build([{"subject": "Ada", "predicate": "studied", "object": "engines",
                               "source": "The mathematician studied mechanical computation."}])
    assert passage_search(index, "Who studied mechanical computation?") == [0]


def test_zero_history_extracted_only_and_citation_mapping():
    client = Client()
    triples = [{"subject": "Alice", "predicate": "founded", "object": "Acme"},
               {"subject": "Alice", "predicate": "visited", "object": "Paris", "inferred": True}]
    chat = GraphChat(triples, {"query": {"history_turns": 0}}, client)
    result = chat.ask("What did Alice do?", extracted_only=True)
    chat.ask("And then?", extracted_only=True)
    assert chat.history == [] and 'Earlier' not in client.prompts[-1]
    assert 'Paris' not in client.prompts[0]
    assert result["citations"]["1"]["object"] == "Acme"
    assert result["citation_status"] == "linked"


def test_overview_balances_separate_communities():
    client = Client()
    triples = [{"subject": f"a{i}", "predicate": "p", "object": f"b{i}"} for i in range(8)]
    chat = GraphChat(triples, {"query": {"max_triples": 8}}, client)
    result = chat.ask("Summarize the main themes across my documents")
    assert result["mode"] == "overview" and result["facts_considered"] == 8
    assert all(f'a{i}' in client.prompts[0] for i in range(8))


def test_invalid_citation_and_token_budget():
    class Invalid(Client):
        def complete(self, user, system=None): return 'Made up [999].'
    chat = GraphChat([{"subject": "a", "predicate": "p", "object": "b", "source": 'x' * 10000}],
                     {"query": {"max_context_tokens": 50, "use_llm_for_entity_matching": False}}, Invalid())
    result = chat.ask('What about a?')
    assert result["facts_considered"] == 0 and result["citation_status"] == "invalid"
