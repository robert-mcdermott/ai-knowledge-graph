"""Wave 8: LLM response cache, predicate normalization, node sizing, page options."""
import json
import re

from fakes import FakeSession, completion
from knowledge_graph.entity_standardization import normalize_predicates
from knowledge_graph.llm import LLMClient
from knowledge_graph.visualization import _calculate_node_sizes, build_graph_data


def make_client(tmp_path, script):
    session = FakeSession(script)
    client = LLMClient(model="m", base_url="http://x", api_key="k", cache_dir=str(tmp_path / "cache"),
                       _session=session, _sleep=lambda s: None)
    return client, session


def test_cache_serves_repeat_requests_without_calling_the_api(tmp_path):
    client, session = make_client(tmp_path, [completion("[1]"), completion("[2]")])
    assert client.complete("prompt", "sys") == "[1]"
    assert client.complete("prompt", "sys") == "[1]"  # cached
    assert client.complete("other prompt", "sys") == "[2]"
    assert len(session.calls) == 2 and client.cache_hits == 1
    files = list((tmp_path / "cache").glob("*.json"))
    assert len(files) == 2 and json.loads(files[0].read_text())["model"] == "m"


def test_cache_key_includes_model_and_parameters(tmp_path):
    a, _ = make_client(tmp_path, [completion("A")])
    b, sb = make_client(tmp_path, [completion("B")])
    b.model = "other-model"
    assert a.complete("p") == "A"
    assert b.complete("p") == "B" and len(sb.calls) == 1


def test_truncated_replies_are_not_cached(tmp_path):
    client, session = make_client(tmp_path, [completion("partial", finish_reason="length"), completion("full")])
    assert client.complete("p", allow_truncated=True) == "partial"
    assert client.complete("p") == "full"
    assert len(session.calls) == 2


def test_cache_disabled_when_dir_unset(tmp_path):
    session = FakeSession([completion("x"), completion("y")])
    client = LLMClient(model="m", base_url="http://x", _session=session, _sleep=lambda s: None)
    assert client.complete("p") == "x" and client.complete("p") == "y"


def test_from_config_reads_cache_dir():
    assert LLMClient.from_config({"llm": {"model": "m", "base_url": "u", "cache_dir": ".c"}}).cache_dir == ".c"
    assert LLMClient.from_config({"llm": {"model": "m", "base_url": "u", "cache_dir": ""}}).cache_dir is None


def test_normalize_predicates_merges_tense_variants_keeping_frequent_form():
    triples = [{"subject": "a", "predicate": "Involves ", "object": "b"},
               {"subject": "c", "predicate": "involves", "object": "d"},
               {"subject": "e", "predicate": "involve", "object": "f"},
               {"subject": "g", "predicate": "led to", "object": "h"},
               {"subject": "i", "predicate": "is", "object": "j"},
               {"subject": "k", "predicate": "process", "object": "l"}]
    normalize_predicates(triples)
    preds = [t["predicate"] for t in triples]
    assert preds == ["involves", "involves", "involves", "led to", "is", "process"]


def test_node_sizes_use_sqrt_scale():
    sizes = _calculate_node_sizes({"a", "b", "c"}, {"a": 0, "b": 0, "c": 0}, {"a": 1, "b": 4, "c": 16}, {"a": 0, "b": 0, "c": 0})
    assert sizes["a"] < sizes["b"] < sizes["c"]
    # doubling the degree ratio should grow the size by ~sqrt(2), not 2
    assert (sizes["c"] - 8) / (sizes["b"] - 8) < 2.2


def test_show_inferred_option_reaches_the_page(tmp_path):
    from knowledge_graph.visualization import SAMPLE_TRIPLES, visualize_knowledge_graph
    out = tmp_path / "g.html"
    visualize_knowledge_graph(SAMPLE_TRIPLES, str(out), config={"visualization": {"show_inferred": False}})
    html = out.read_text(encoding="utf-8")
    data = json.loads(re.search(r"const KG = (\{.*?\});\n</script>", html, re.DOTALL).group(1).replace("<\\/", "</"))
    assert data["meta"]["showInferred"] is False
    assert data["meta"]["theme"] == "light" and data["meta"]["edgeLabels"] == "all"  # defaults
    visualize_knowledge_graph(SAMPLE_TRIPLES, str(out), config={"visualization": {"theme": "dark", "edge_labels": "none"}})
    html2 = out.read_text(encoding="utf-8")
    data2 = json.loads(re.search(r"const KG = (\{.*?\});\n</script>", html2, re.DOTALL).group(1).replace("<\\/", "</"))
    assert data2["meta"]["theme"] == "dark" and data2["meta"]["edgeLabels"] == "none"
    assert "localStorage" not in html2
    assert 'id="path-target"' in html and "function findPath" in html
    assert build_graph_data(SAMPLE_TRIPLES)["meta"]["showInferred"] is True
