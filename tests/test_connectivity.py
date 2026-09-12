"""Wave 3: taxonomy, singularization, LLM bridging and hub enrichment."""
from knowledge_graph import entity_standardization as es
from knowledge_graph.entity_standardization import (
    _infer_taxonomy,
    _singularize,
    infer_relationships,
    standardize_entities,
)


def T(s, p, o, **extra):
    return {"subject": s, "predicate": p, "object": o, **extra}


def inferred(result):
    return [t for t in result if t.get("inferred")]


# ---- taxonomy -------------------------------------------------------------- #
def test_taxonomy_links_specific_to_head_term():
    entities = {"quantum computing", "computing", "steam engine", "engine"}
    new = _infer_taxonomy(entities, [])
    pairs = {(t["subject"], t["predicate"], t["object"]) for t in new}
    assert ("quantum computing", "is a", "computing") in pairs
    assert ("steam engine", "is a", "engine") in pairs
    assert all(t["method"] == "taxonomy" for t in new)


def test_taxonomy_does_not_treat_prepositional_phrase_as_head():
    new = _infer_taxonomy({"developments in electronics", "electronics"}, [])
    assert [(t["subject"], t["predicate"], t["object"]) for t in new] == [
        ("developments in electronics", "involves", "electronics")]


def test_taxonomy_containment_uses_involves():
    new = _infer_taxonomy({"internet of things", "internet", "things"}, [])
    preds = {(t["predicate"], t["object"]) for t in new}
    # "of things" is a prepositional tail, so "things" is not treated as the head noun.
    assert preds == {("involves", "internet"), ("involves", "things")}


def test_taxonomy_never_invents_nodes_or_duplicates_pairs():
    triples = [T("quantum computing", "extends", "computing")]
    assert _infer_taxonomy({"quantum computing", "computing", "lonely term"}, triples) == []


def test_taxonomy_on_by_default_in_inference():
    triples = [T("quantum computing", "used in", "cryptography"), T("computing", "enabled", "internet")]
    cfg = {"inference": {"use_llm_for_inference": False}}
    new = inferred(infer_relationships(triples, cfg))
    assert [(t["subject"], t["object"], t["method"]) for t in new] == [("quantum computing", "computing", "taxonomy")]
    assert inferred(infer_relationships(triples, {"inference": {"use_llm_for_inference": False, "taxonomy": False}})) == []


def test_taxonomy_skips_people_places_and_organizations():
    triples = [T("nikola tesla", "founded", "tesla", subject_type="person", object_type="organization"),
               T("great britain", "led", "industrialization", subject_type="place"),
               T("quantum computing", "extends", "cryptography", subject_type="technology")]
    entities = {"nikola tesla", "tesla", "great britain", "britain", "quantum computing", "computing", "industrialization", "cryptography"}
    new = _infer_taxonomy(entities, triples)
    assert [(t["subject"], t["object"]) for t in new] == [("quantum computing", "computing")]


def test_taxonomy_respects_non_english_phrases_and_predicates():
    from knowledge_graph.entity_standardization import taxonomy_predicates
    # "entrega de granada" is not a kind of "granada": "de" ends the modifier phrase
    new = _infer_taxonomy({"entrega de granada", "granada", "palacio nazarí", "palacio"}, [], taxonomy_predicates("Spanish"),
                          head_first=True)
    assert [(t["subject"], t["predicate"], t["object"]) for t in new] == [
        ("entrega de granada", "incluye", "granada"), ("palacio nazarí", "es un", "palacio")]
    assert taxonomy_predicates("auto") == ("is a", "involves") and taxonomy_predicates("Deutsch") == ("ist ein", "umfasst")
    cfg = {"inference": {"use_llm_for_inference": False}, "extraction": {"language": "French"}}
    out = inferred(infer_relationships([T("moteur à vapeur", "alimente", "usines"), T("moteur", "est", "machine")], cfg))
    assert out and (out[0]["predicate"], out[0]["object"]) == ("est un", "moteur")  # head-first: a "moteur à vapeur" is a moteur


# ---- singularization ------------------------------------------------------- #
def test_singularize_rules():
    assert _singularize("factories") == "factory"
    assert _singularize("processes") == "process"
    assert _singularize("assistants") == "assistant"
    assert _singularize("physics") == "physics"
    assert _singularize("glass") == "glass"
    assert _singularize("bus") == "bus"
    assert _singularize("gas") == "gas"


def test_plural_variants_merge_only_when_both_exist():
    triples = [T("smart assistants", "use", "voice"), T("smart assistant", "made by", "amazon"),
               T("factories", "employed", "workers")]
    out = standardize_entities(triples, {"standardization": {}})
    names = {t["subject"] for t in out} | {t["object"] for t in out}
    assert len(names & {"smart assistants", "smart assistant"}) == 1
    assert "factories" in names  # no singular variant present, name untouched


# ---- LLM bridging & hub ---------------------------------------------------- #
class RecordingClient:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, user, system=None):
        self.prompts.append(user)
        return self.reply


def patch_client(monkeypatch, client):
    monkeypatch.setattr(es.LLMClient, "from_config", classmethod(lambda cls, cfg: client))


MAIN = [T("steam engine", "powered", "railways"), T("railways", "connected", "cities"),
        T("steam engine", "invented by", "watt"), T("cities", "grew", "fast")]
ISLAND1 = [T("internet", "developed by", "darpa")]
ISLAND2 = [T("assembly line", "pioneered by", "ford")]


def test_bridging_batches_components_and_tags_results(monkeypatch):
    client = RecordingClient('[{"subject": "internet", "predicate": "succeeded", "object": "railways"},'
                             ' {"subject": "assembly line", "predicate": "used in", "object": "cities"}]')
    patch_client(monkeypatch, client)
    cfg = {"inference": {"use_llm_for_inference": True, "llm_hub": False, "taxonomy": False,
                         "bridge_groups_per_call": 5}}
    new = inferred(infer_relationships(MAIN + ISLAND1 + ISLAND2, cfg))
    assert {t["method"] for t in new} == {"llm_bridge"}
    assert len(client.prompts) == 1  # both islands in one call
    prompt = client.prompts[0]
    assert "steam engine" in prompt and "Group 1" in prompt and "Group 2" in prompt
    assert "internet developed by darpa" in prompt  # island context included


def test_bridging_respects_per_call_and_component_limits(monkeypatch):
    client = RecordingClient("[]")
    patch_client(monkeypatch, client)
    cfg = {"inference": {"use_llm_for_inference": True, "llm_hub": False, "taxonomy": False,
                         "bridge_groups_per_call": 1, "max_bridge_components": 1}}
    infer_relationships(MAIN + ISLAND1 + ISLAND2, cfg)
    assert len(client.prompts) == 1  # only one component considered


def test_bridging_skipped_for_single_component(monkeypatch):
    client = RecordingClient("[]")
    patch_client(monkeypatch, client)
    infer_relationships(MAIN, {"inference": {"use_llm_for_inference": True, "llm_hub": False}})
    assert client.prompts == []


def test_hub_enrichment_filters_to_hub_entities_and_caps(monkeypatch):
    reply = ('[{"subject": "steam engine", "predicate": "preceded", "object": "cities"},'
             ' {"subject": "steam engine", "predicate": "invented", "object": "not an entity"},'
             ' {"subject": "railways", "predicate": "reached", "object": "watt"}]')
    client = RecordingClient(reply)
    patch_client(monkeypatch, client)
    cfg = {"inference": {"use_llm_for_inference": True, "llm_bridge": False, "taxonomy": False,
                         "hub_entities": 5, "hub_max_new": 1}}
    new = inferred(infer_relationships(MAIN, cfg))
    assert [(t["subject"], t["object"], t["method"]) for t in new] == [("steam engine", "cities", "llm_hub")]
    assert "Known relationships" in client.prompts[0] and "steam engine powered railways" in client.prompts[0]


def test_priority_order_llm_before_taxonomy(monkeypatch):
    # Hub proposes an edge for the same pair the taxonomy rule would create; the LLM edge wins.
    client = RecordingClient('[{"subject": "quantum computing", "predicate": "extends", "object": "computing"}]')
    patch_client(monkeypatch, client)
    triples = [T("quantum computing", "used in", "cryptography"), T("computing", "enabled", "internet"),
               T("cryptography", "protects", "internet")]
    cfg = {"inference": {"use_llm_for_inference": True, "llm_bridge": False, "hub_entities": 5}}
    new = inferred(infer_relationships(triples, cfg))
    assert [(t["predicate"], t["method"]) for t in new] == [("extends", "llm_hub")]
