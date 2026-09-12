import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from knowledge_graph import server as srv  # noqa: E402

TRIPLES = [
    {"subject": "james watt", "predicate": "refined", "object": "steam engine", "source": "Watt refined the steam engine.",
     "subject_type": "person", "object_type": "technology"},
    {"subject": "steam engine", "predicate": "powered", "object": "railways"},
    {"subject": "railways", "predicate": "connected", "object": "cities"},
    {"subject": "steam engine", "predicate": "is a", "object": "engine", "inferred": True, "method": "taxonomy"},
]
CFG = {"llm": {"model": "test-model", "base_url": "http://x"}, "visualization": {"name_communities": False}}


class FakeClient:
    def complete(self, user, system=None):
        return "Watt refined the steam engine, which powered railways.\nFacts used: [1, 2]"


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "industrial.json").write_text(json.dumps(TRIPLES))
    (tmp_path / "notes.json").write_text(json.dumps({"not": "a graph"}))
    (tmp_path / "other.txt").write_text("ignored")
    from knowledge_graph.query import GraphChat
    monkeypatch.setattr(srv, "GraphChat", lambda triples, config: GraphChat(triples, config, client=FakeClient()))
    return TestClient(srv.create_app(CFG, str(tmp_path)))


def test_library_lists_only_triple_files(client):
    html = client.get("/").text
    assert "industrial" in html and "notes" not in html and "other" not in html
    data = client.get("/api/graphs").json()
    assert data["graphs"][0]["name"] == "industrial" and data["graphs"][0]["nodes"] == 5
    assert data["graphs"][0]["inferred"] == 1


def test_graph_page_is_the_explorer_with_chat_enabled(client):
    html = client.get("/graph/industrial").text
    assert '"chatEndpoint": "/api/chat/industrial"' in html and '"libraryUrl": "/"' in html
    assert 'id="chat-form"' in html and "function askGraph" in html
    assert client.get("/graph/industrial").text == html  # cached render


def test_static_render_has_no_chat_endpoint(tmp_path):
    from knowledge_graph.visualization import visualize_knowledge_graph
    out = tmp_path / "static.html"
    visualize_knowledge_graph(TRIPLES, str(out))
    import re
    html = out.read_text(encoding="utf-8")
    data = json.loads(re.search(r"const KG = (\{.*?\});\n</script>", html, re.DOTALL).group(1).replace("<\\/", "</"))
    assert "chatEndpoint" not in data["meta"] and "libraryUrl" not in data["meta"]
    assert 'id="btn-chat" title="Ask questions about this graph (a)">Ask</button>' in html  # present but hidden


def test_unknown_and_unsafe_names_are_404(client):
    assert client.get("/graph/missing").status_code == 404
    assert client.get("/graph/..%2Fsecret").status_code == 404
    assert client.post("/api/chat/missing", json={"question": "x"}).status_code == 404


def test_chat_endpoint_answers_with_cited_facts_and_resets(client):
    res = client.post("/api/chat/industrial", json={"question": "How did James Watt affect railways?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"].startswith("Watt refined")
    assert [f["predicate"] for f in body["facts"]] == ["refined", "powered"]
    assert client.post("/api/chat/industrial", json={"question": ""}).status_code == 400
    assert client.post("/api/chat/industrial/reset").json() == {"ok": True}


def test_store_rejects_traversal(tmp_path):
    store = srv.GraphStore(str(tmp_path), CFG)
    for bad in ("../x", "a/b", "", ".hidden"):
        with pytest.raises(KeyError):
            store.path_for(bad)
