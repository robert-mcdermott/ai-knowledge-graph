"""Behavioral regressions for document lifecycle, evidence and review persistence."""
import copy
import json

import pytest

from knowledge_graph.entity_standardization import standardize_entities
from knowledge_graph.llm import LLMClient, LLMError
from knowledge_graph.main import RunContext, process_documents, process_with_llm, update_collection
from knowledge_graph.workspace import (
    WorkspaceError,
    apply_document_update,
    claim_id,
    document_record,
    empty_workspace,
    load_workspace,
    merge_claims,
    passage_chunks,
    preview_documents,
    projected_claims,
    save_workspace,
    set_correction,
    validate_triples,
)

CFG = {"llm": {"model": "test", "base_url": "http://unused", "concurrency": 2},
       "standardization": {"enabled": False}, "inference": {"enabled": False}}


def sourced(name, text="Alice founded Acme.", **qualifiers):
    doc = document_record(name, text)
    return {"subject": "Alice", "predicate": "founded", "object": "Acme", "document": name,
            "document_id": doc["id"], "source": text, "evidence": doc["passages"], **qualifiers}


class Fake:
    def __init__(self, reply):
        self.reply, self.calls, self.invalidated = reply, 0, False

    def complete(self, user, system=None):
        self.calls += 1
        return self.reply(user) if callable(self.reply) else self.reply

    def invalidate(self, *args):
        self.invalidated = True


def install(monkeypatch, client):
    monkeypatch.setattr(LLMClient, "from_config", classmethod(lambda cls, cfg: client))


def test_exact_evidence_offsets_pages_and_overlap():
    text = "Alice founded Acme.\n\nShe worked in Paris.\f\f北京位于中国。北京有公园。"
    doc = document_record("research.pdf", text)
    assert doc["passages"][-1]["page"] == 3
    for p in doc["passages"]:
        assert text[p["start"]:p["end"]] == p["text"]
    chunks = passage_chunks(doc["passages"], 6, 2)
    for chunk in chunks:
        for p in chunk:
            assert text[p["start"]:p["end"]] == p["text"]
    assert any("北京" in p["text"] for chunk in chunks for p in chunk)


def test_pronoun_claim_uses_explicit_passage_not_entity_matching(monkeypatch):
    install(monkeypatch, Fake(json.dumps([{"subject": "Alice", "predicate": "founded", "object": "Acme", "source_ids": ["P2"]}])))
    result = process_documents(CFG, [("bio.txt", "Alice visited Paris. She founded Acme.")])
    assert result[0]["source"] == "She founded Acme."
    assert result[0]["evidence"][0]["start"] == 21
    assert result[0]["document"] == "bio.txt"


def test_no_guessed_source_and_strict_mode_rejects_missing_ids(monkeypatch):
    fake = Fake('[{"subject":"Alice","predicate":"founded","object":"Acme"}]')
    install(monkeypatch, fake)
    claim = process_with_llm(CFG, "Alice visited Acme.")[0]
    assert "source" not in claim and claim["evidence_status"] == "unverified"
    cfg = dict(CFG, extraction={"strict_evidence": True})
    with pytest.raises(LLMError, match="no supporting"):
        process_with_llm(cfg, "Alice visited Acme.")
    assert fake.invalidated


@pytest.mark.parametrize("reply", ['not JSON', '[{"subject":"a","predicate":"p","object":"b"},',
                                    '[{"subject":"a","object":"b"}]',
                                    '[{"subject":"a","predicate":"p","object":"b","source_ids":["P99"]}]'])
def test_bad_extraction_is_a_failure_and_does_not_poison_cache(monkeypatch, reply):
    fake = Fake(reply)
    install(monkeypatch, fake)
    with pytest.raises(LLMError):
        process_documents(CFG, [("x.txt", "Some text.")])
    assert fake.invalidated


def test_partial_status_and_valid_empty_result_are_distinct(monkeypatch):
    fake = Fake(lambda user: 'bad' if '[P1] Broken' in user else '[]')
    install(monkeypatch, fake)
    run = RunContext()
    assert process_documents(CFG, [("bad.txt", "Broken text."), ("empty.txt", "No claims.")],
                             continue_on_error=True, run=run) == []
    assert run.completed == 2 and run.snapshot()["status"] == "partial"
    assert [f["document"] for f in run.failures] == ["bad.txt"]


def test_duplicate_claims_keep_all_evidence_and_qualified_events_stay_distinct():
    a, b = sourced("a.txt"), sourced("b.txt")
    result = merge_claims([a, b, a])
    assert len(result) == 1
    assert {e["document"] for e in result[0]["evidence"]} == {"a.txt", "b.txt"}
    assert len(merge_claims([a, dict(a, time="2020"), dict(a, polarity="negative")])) == 3


def test_document_replace_remove_noop_and_corrections_persist(tmp_path):
    docs = [("a.txt", "Alice founded Acme."), ("b.txt", "Alice founded Acme.")]
    ws, _ = apply_document_update(empty_workspace(), docs, [sourced(n, t) for n, t in docs])
    cid = claim_id(ws["claims"][0])
    set_correction(ws, "reject", {"claim_id": cid})
    assert projected_claims(ws) == []
    ws["views"] = [{"title": "Founders", "state": {"selected": "Alice"}}]
    path = tmp_path / "collection.json"
    save_workspace(path, ws)
    loaded = load_workspace(path)
    assert loaded["views"] == ws["views"] and loaded["overrides"] == ws["overrides"]
    unchanged = preview_documents(loaded, docs)
    assert unchanged["unchanged"] == ["a.txt", "b.txt"]
    revised, diff = apply_document_update(loaded, [("a.txt", "Alice visited Paris.")], [], run={"status": "complete"})
    assert diff["updated"] == ["a.txt"]
    assert [e["document"] for e in revised["claims"][0]["evidence"]] == ["b.txt"]
    assert projected_claims(revised) == []
    set_correction(revised, "restore", {"claim_id": cid})
    assert len(projected_claims(revised)) == 1
    final, _ = apply_document_update(revised, [], [], remove=["b.txt"])
    assert final["claims"] == [] and final["views"] == ws["views"]


def test_unchanged_collection_makes_no_calls(monkeypatch):
    ws, _ = apply_document_update(empty_workspace(), [("a", "Alice founded Acme.")], [sourced("a")])
    fake = Fake('[]')
    install(monkeypatch, fake)
    updated, _ = update_collection(ws, [("a", "Alice founded Acme.")], CFG)
    assert updated == ws and fake.calls == 0


def test_only_changed_documents_are_extracted(monkeypatch):
    ws, _ = apply_document_update(empty_workspace(), [("a", "Alice founded Acme.")], [sourced("a")])
    fake = Fake('[{"subject":"Bob","predicate":"visited","object":"Paris","source_ids":["P1"]}]')
    install(monkeypatch, fake)
    updated, diff = update_collection(ws, [("a", "Alice founded Acme."), ("b", "Bob visited Paris.")], CFG)
    assert fake.calls == 1 and diff["added"] == ["b"]
    assert len(updated["claims"]) == 2


def test_failed_document_can_retry_with_identical_content():
    ws, _ = apply_document_update(empty_workspace(), [("a", "Hello.")], [],
                                  run={"status": "partial", "failures": [{"document": "a"}]})
    assert preview_documents(ws, [("a", "Hello.")])["updated"] == ["a"]


def test_merge_is_reversible_and_rejects_cycles():
    ws = empty_workspace()
    ws["claims"] = [sourced("a"), {"subject": "A. Smith", "predicate": "visited", "object": "Paris"}]
    original = copy.deepcopy(ws["claims"])
    set_correction(ws, "merge", {"source": "A. Smith", "target": "Alice"})
    assert projected_claims(ws)[1]["subject"] == "Alice"
    with pytest.raises(WorkspaceError, match="cycle"):
        set_correction(ws, "merge", {"source": "Alice", "target": "A. Smith"})
    set_correction(ws, "unmerge", {"source": "A. Smith"})
    assert ws["claims"] == original and projected_claims(ws)[1]["subject"] == "A. Smith"


def test_punctuation_sensitive_names_remain_distinct():
    triples = [{"subject": "C++", "predicate": "influenced", "object": "C#"}]
    out = standardize_entities(triples, {"standardization": {"use_llm_for_entities": False}})
    assert len(out) == 1 and out[0]["subject"] != out[0]["object"]


def test_endpoint_cache_namespaces_and_invalid_imports(tmp_path):
    a = LLMClient(model="m", base_url="http://a", cache_dir=str(tmp_path))
    b = LLMClient(model="m", base_url="http://b", cache_dir=str(tmp_path))
    assert a._cache_path("q", "s") != b._cache_path("q", "s")
    with pytest.raises(WorkspaceError):
        validate_triples([{"subject": "x", "object": "y"}])
    path = tmp_path / 'g.json'
    path.write_text('{"schema_version": 999, "claims": []}')
    with pytest.raises(WorkspaceError, match="schema"):
        load_workspace(path)


def test_cancelled_run_preserves_existing_workspace(monkeypatch):
    run = RunContext()
    run.cancel.set()
    fake = Fake('[]')
    install(monkeypatch, fake)
    ws = empty_workspace()
    with pytest.raises(LLMError, match="cancelled"):
        update_collection(ws, [("a", "Some text.")], CFG, run=run)
    assert ws["documents"] == [] and fake.calls == 0


def test_qualified_relationship_is_visible_in_graph_label(tmp_path):
    from knowledge_graph.visualization import render_knowledge_graph
    output = tmp_path / 'nested' / 'graph.html'
    _, data = render_knowledge_graph([{'subject': 'Alice', 'predicate': 'founded', 'object': 'Acme',
                                     'polarity': 'negative', 'time': '1900'}], str(output))
    edge = data['edges'][0]
    assert edge['label'] == 'founded'
    assert edge['displayLabel'] == '[negative] founded · 1900'
    assert 'polarity: negative' in edge['title']
    assert output.exists()


def test_unverified_claim_retains_membership_in_multiple_documents():
    from knowledge_graph.workspace import document_record
    triples = [{'subject': 'Alice', 'predicate': 'founded', 'object': 'Acme',
                'document': name, 'document_id': document_record(name, '')['id'], 'evidence_status': 'unverified'}
               for name in ('a', 'b')]
    ws, _ = apply_document_update(empty_workspace(), [('a', 'First source'), ('b', 'Second source')], triples)
    assert len(ws['claims']) == 1
    revised, _ = apply_document_update(ws, [], [], remove=['a'])
    assert len(revised['claims']) == 1 and revised['claims'][0]['document'] == 'b'
    assert revised['claims'][0]['evidence_status'] == 'unverified'
    final, _ = apply_document_update(revised, [], [], remove=['b'])
    assert final['claims'] == []


def test_cli_regeneration_preserves_rejections_and_views(tmp_path, monkeypatch):
    from knowledge_graph import main as cli
    from knowledge_graph.config import apply_defaults
    config = apply_defaults(copy.deepcopy(CFG))
    config['visualization']['name_communities'] = False
    monkeypatch.setattr(cli, 'load_config', lambda path: config)
    monkeypatch.setattr(cli, 'process_documents', lambda *a, **kw: [sourced('a.txt')])
    source = tmp_path / 'a.txt'
    source.write_text('Alice founded Acme.')
    output = tmp_path / 'collection.html'
    monkeypatch.setattr('sys.argv', ['generate-graph', '--input', str(source), '--output', str(output)])
    cli.main()
    path = tmp_path / 'collection.json'
    ws = load_workspace(path)
    ws['views'] = [{'title': 'Keep this view'}]
    set_correction(ws, 'reject', {'claim_id': claim_id(ws['claims'][0])})
    save_workspace(path, ws)
    revision = ws['revision']
    monkeypatch.setattr('sys.argv', ['generate-graph', '--input', str(source), '--output', str(output)])
    cli.main()
    updated = load_workspace(path)
    assert updated['revision'] == revision + 1
    assert updated['views'] == ws['views']
    assert projected_claims(updated) == [] and json.loads(path.read_text()) == []
    assert '0 relationships' in output.read_text()
