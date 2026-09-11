from src.knowledge_graph.entity_standardization import standardize_entities


def T(s, p, o):
    return {"subject": s, "predicate": p, "object": o, "chunk": 1}


def names(result):
    return {t["subject"] for t in result} | {t["object"] for t in result}


def test_case_and_stopword_variants_are_merged():
    triples = [T("The Steam Engine", "powered", "mills"), T("steam engine", "invented by", "watt"),
               T("Steam engine", "spread to", "europe")]
    out = standardize_entities(triples, {"standardization": {}})
    assert names(out) & {"The Steam Engine", "steam engine", "Steam engine"} == {"steam engine"}


def test_word_subset_merging_is_off_by_default():
    triples = [T("steam engine", "powered", "steam engine factories"),
               T("industrial cities", "grew during", "industrial revolution")]
    out = standardize_entities(triples, {"standardization": {}})
    assert len(out) == 2
    assert {"steam engine factories", "industrial cities", "industrial revolution"} <= names(out)


def test_word_subset_merging_can_be_enabled():
    triples = [T("steam engine", "powered", "steam engine factories"), T("mills", "used", "steam engine")]
    out = standardize_entities(triples, {"standardization": {"merge_word_subsets": True}})
    assert "steam engine factories" not in names(out)
    assert all(t["subject"] != t["object"] for t in out)  # self-loop removed


def test_extra_fields_and_long_predicates_are_preserved_or_trimmed():
    triples = [dict(T("a", "one two three four", "b"), source="s1")]
    out = standardize_entities(triples, {"standardization": {}})
    assert out[0]["predicate"] == "one two three"
    assert out[0]["source"] == "s1" and out[0]["chunk"] == 1


def test_invalid_triples_are_dropped():
    out = standardize_entities([T("a", "p", "b"), {"subject": "x"}, "junk"], {"standardization": {}})
    assert len(out) == 1
