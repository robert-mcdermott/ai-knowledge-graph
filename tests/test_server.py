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
    assert 'href="/graph/industrial"' in html and 'href="/graph/notes"' not in html and 'href="/graph/other"' not in html
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


# ---- wave 11b: ingest in the browser ---------------------------------------- #
def _fake_pipeline(monkeypatch):
    import logging
    logging.getLogger("knowledge_graph").setLevel(logging.INFO)
    log = logging.getLogger("knowledge_graph.main")

    def fake_process(config, documents, debug=False, continue_on_error=False, run=None):
        log.info("PHASE 1: INITIAL TRIPLE EXTRACTION")
        log.info(f"Processing text in {len(documents)} chunks (size: 500 words, overlap: 50 words)")
        for i, _ in enumerate(documents, start=1):
            log.info(f"Chunk {i}: 2 triples")
        log.info("PHASE 3: RELATIONSHIP INFERENCE")
        if run:
            run.total = run.completed = len(documents)
            if run.callback:
                run.callback(run.snapshot())
        return [dict(t, document=name) for name, _ in documents for t in TRIPLES[:2]]
    monkeypatch.setattr(srv, "process_documents", fake_process)


def _wait(client, job_id, tries=100):
    import time
    for _ in range(tries):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["state"] in ("done", "error"):
            return status
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_ingest_pasted_text_creates_graph_files_and_reports_progress(client, tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    res = client.post("/api/ingest", data={"name": "My Notes!", "text": "Watt refined the steam engine."})
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "My Notes" and body["page_url"] == f"/jobs/{body['job_id']}"
    status = _wait(client, body["job_id"])
    assert status["state"] == "done" and status["phase"] == "done"
    assert status["chunks_total"] == 1 and status["chunks_done"] == 1
    assert status["graph_url"] == "/graph/My%20Notes"
    assert status["stats"]["nodes"] > 0 and "PHASE 1" in "\n".join(status["log"])
    assert (tmp_path / "My Notes.json").exists() and (tmp_path / "My Notes.html").exists()
    assert client.get("/graph/My%20Notes").status_code == 200
    assert client.get(f"/jobs/{body['job_id']}").status_code == 200
    assert any(g["name"] == "My Notes" for g in client.get("/api/graphs").json()["graphs"])


def test_ingest_uploads_use_format_readers_and_tag_documents(client, tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    files = [("files", ("a.md", b"# A\nAlpha text.", "text/markdown")), ("files", ("b.txt", b"Beta text.", "text/plain"))]
    res = client.post("/api/ingest", data={"name": "docs"}, files=files)
    assert res.status_code == 200
    status = _wait(client, res.json()["job_id"])
    assert status["state"] == "done" and status["documents"] == ["a.md", "b.txt"]
    triples = json.loads((tmp_path / "docs.json").read_text())
    assert {e["document"] for t in triples for e in t.get("evidence", [])} == {"a.md", "b.txt"}


def test_ingest_validation_errors(client, monkeypatch):
    _fake_pipeline(monkeypatch)
    assert client.post("/api/ingest", data={"name": "x"}).status_code == 400
    res = client.post("/api/ingest", data={"name": "x"}, files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))])
    assert res.status_code == 400 and "Unsupported" in res.json()["detail"]
    assert client.get("/api/jobs/nope").status_code == 404


def test_ingest_names_are_unique_and_pipeline_errors_are_reported(client, tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    first = client.post("/api/ingest", data={"name": "industrial", "text": "x"}).json()
    _wait(client, first["job_id"])
    assert first["name"] == "industrial-2"  # industrial.json already existed

    def boom(config, documents, debug=False, continue_on_error=False, run=None):
        raise srv.LLMError("model unreachable")
    monkeypatch.setattr(srv, "process_documents", boom)
    failed = client.post("/api/ingest", data={"name": "bad", "text": "x"}).json()
    status = _wait(client, failed["job_id"])
    assert status["state"] == "error" and "model unreachable" in status["error"]
    assert not (tmp_path / "bad.json").exists()


def test_only_one_job_at_a_time(client, monkeypatch):
    import threading
    gate = threading.Event()

    def slow(config, documents, debug=False, continue_on_error=False, run=None):
        gate.wait(5)
        return TRIPLES[:2]
    monkeypatch.setattr(srv, "process_documents", slow)
    first = client.post("/api/ingest", data={"name": "slow", "text": "x"})
    assert first.status_code == 200
    assert client.post("/api/ingest", data={"name": "other", "text": "y"}).status_code == 409
    assert "being prepared" in client.get("/").text
    gate.set()
    _wait(client, first.json()["job_id"])
