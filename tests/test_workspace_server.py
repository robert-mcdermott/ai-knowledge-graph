"""Workspace contracts across the HTTP, persistence and pipeline boundaries."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from knowledge_graph import main as pipeline
from knowledge_graph import server
from knowledge_graph.config import apply_defaults
from knowledge_graph.main import load_triples_from_json
from knowledge_graph.query import GraphChat
from knowledge_graph.workspace import (
    apply_document_update,
    claim_id,
    document_record,
    empty_workspace,
    load_workspace,
    save_workspace,
)

CFG = apply_defaults({'llm': {'model': 'fake', 'base_url': 'http://unused'},
                      'inference': {'enabled': False}, 'standardization': {'enabled': False},
                      'visualization': {'name_communities': False}})


def claim(name, text, subject='Alice', obj='Acme'):
    doc = document_record(name, text)
    return {'subject': subject, 'predicate': 'founded', 'object': obj,
            'document_id': doc['id'], 'document': name,
            'source': text, 'evidence': doc['passages']}


@pytest.fixture
def app(tmp_path, monkeypatch):
    ws, _ = apply_document_update(empty_workspace('Collection'), [('a.txt', 'Alice founded Acme.')],
                                  [claim('a.txt', 'Alice founded Acme.')])
    save_workspace(tmp_path / 'Collection.json', ws)

    def extract(config, documents, *args, run=None, **kwargs):
        return [claim(n, t, 'Bob', 'Beta') for n, t in documents]

    monkeypatch.setattr(pipeline, 'process_documents', extract)
    return server.create_app(copy.deepcopy(CFG), str(tmp_path))


def test_sidecars_not_listed_and_primary_reads_authoritative_corrections(app, tmp_path):
    client = TestClient(app)
    assert [g['name'] for g in client.get('/api/graphs').json()['graphs']] == ['Collection']
    ws = client.get('/api/workspaces/Collection').json()
    cid = claim_id(ws['claims'][0])
    assert client.post('/api/review/Collection', json={'revision': 1, 'action': 'reject', 'claim_id': cid}).status_code == 200
    assert load_triples_from_json(tmp_path / 'Collection.json') == []
    assert '0 relationships' in client.get('/graph/Collection').text
    assert client.post('/api/review/Collection', json={'revision': 1, 'action': 'restore', 'claim_id': cid}).status_code == 409
    assert client.post('/api/review/Collection', json={'revision': 2, 'action': 'restore', 'claim_id': cid}).status_code == 200
    assert len(load_triples_from_json(tmp_path / 'Collection.json')) == 1


def test_saved_view_revision_matches_export_and_rejects_invalid_state(app, tmp_path):
    client = TestClient(app)
    bad = client.post('/api/views/Collection', json={'revision': 1, 'view': {'title': 'Bad', 'state': {'hiddenTypes': 5}}})
    assert bad.status_code == 409
    response = client.post('/api/views/Collection', json={'revision': 1, 'view': {'title': 'My view', 'note': 'Useful', 'state': {'selected': 'Alice'}}})
    assert response.status_code == 200
    assert response.json()['revision'] == 2
    html = (tmp_path / 'Collection.html').read_text()
    assert '"revision": 2' in html and 'My view' in html
    assert client.post('/api/views/Collection', json={'revision': 1, 'view': {'title': 'stale'}}).status_code == 409
    assert len(client.get('/api/workspaces/Collection').json()['views']) == 1


def test_incremental_update_is_staged_and_only_applied_once(app, tmp_path):
    before = (tmp_path / 'Collection.workspace.json').read_bytes()
    job = app.state.jobs.start('', [('b.txt', 'Bob founded Beta.')], target='Collection', background=False)
    assert job.state == 'ready'
    assert (tmp_path / 'Collection.workspace.json').read_bytes() == before
    assert job.diff['added_examples'][0]['subject'] == 'Bob'
    client = TestClient(app)
    assert client.post(f'/api/jobs/{job.id}/apply').status_code == 200
    assert client.post(f'/api/jobs/{job.id}/apply').status_code == 409
    ws = load_workspace(tmp_path / 'Collection.json')
    assert len(ws['claims']) == 2 and len(ws['documents']) == 2
    assert ws['revision'] == 2


def test_revision_conflict_preserves_both_review_and_staged_update(app):
    job = app.state.jobs.start('', [('b.txt', 'Bob founded Beta.')], target='Collection', background=False)
    client = TestClient(app)
    assert client.post('/api/views/Collection', json={'revision': 1, 'view': {'title': 'Keep me'}}).status_code == 200
    assert client.post(f'/api/jobs/{job.id}/apply').status_code == 409
    ws = client.get('/api/workspaces/Collection').json()
    assert ws['views'][0]['title'] == 'Keep me' and len(ws['documents']) == 1
    assert job.state == 'ready'
    assert client.post(f'/api/jobs/{job.id}/cancel').status_code == 200
    assert job.state == 'cancelled' and job.pending is None


def test_preview_and_noop_never_extract(app, monkeypatch):
    monkeypatch.setattr(pipeline, 'process_documents', lambda *a, **k: pytest.fail('Unexpected extraction'))
    client = TestClient(app)
    res = client.post('/api/ingest', data={'target': 'Collection', 'dry_run': 'true', 'text': 'New text'})
    assert res.status_code == 200 and res.json()['chunks'] == 1
    res = client.post('/api/ingest', data={'target': 'Collection'}, files={'files': ('a.txt', b'Alice founded Acme.', 'text/plain')})
    assert res.json()['unchanged'] is True
    assert app.state.jobs.jobs == {}


def test_removal_keeps_other_evidence_and_saved_views(app, tmp_path):
    ws = load_workspace(tmp_path / 'Collection.json')
    ws, _ = apply_document_update(ws, [('b.txt', 'Alice founded Acme.')], [claim('b.txt', 'Alice founded Acme.')])
    ws['views'] = [{'title': 'Keep this view'}]
    save_workspace(tmp_path / 'Collection.json', ws)
    job = app.state.jobs.start('', [], target='Collection', remove=['a.txt'], background=False)
    assert job.state == 'ready' and job.diff['removed_claims'] == 0
    client = TestClient(app)
    assert client.post(f'/api/jobs/{job.id}/apply').status_code == 200
    ws = load_workspace(tmp_path / 'Collection.json')
    assert len(ws['claims']) == 1 and ws['claims'][0]['evidence'][0]['document'] == 'b.txt'
    assert ws['views'][0]['title'] == 'Keep this view'


def test_cross_origin_mutations_rejected(app):
    response = TestClient(app).post('/api/views/Collection', json={}, headers={'Origin': 'https://unrelated.example'})
    assert response.status_code == 403


def test_chat_sessions_isolated_and_invalidated_by_review(app, monkeypatch):
    class Fake:
        def complete(self, *args):
            return 'Alice founded Acme [1].'
    monkeypatch.setattr(server, 'GraphChat', lambda triples, config: GraphChat(triples, config, client=Fake()))
    store = app.state.store
    first, second = store.chat_for('Collection', 'one'), store.chat_for('Collection', 'two')
    first.ask('Who founded Acme?')
    assert len(first.history) == 1 and second.history == []
    ws = load_workspace(store.path_for('Collection'))
    store.save('Collection', ws)
    assert store.chat_for('Collection', 'one') is not first


def test_partial_run_is_marked_and_retry_retains_strict_setting(app, monkeypatch):
    def partial(config, documents, *args, run=None, **kwargs):
        run.failures.append({'document': 'b.txt', 'chunk': 2, 'error': 'Retry me'})
        run.total, run.completed = 2, 2
        return [claim('b.txt', 'Bob founded Beta.', 'Bob', 'Beta')]
    monkeypatch.setattr(pipeline, 'process_documents', partial)
    job = app.state.jobs.start('', [('b.txt', 'Bob founded Beta.')], target='Collection', strict=True, background=False)
    client = TestClient(app)
    assert client.post(f'/api/jobs/{job.id}/apply').json()['state'] == 'partial'
    ws = client.get('/api/workspaces/Collection').json()
    assert next(d for d in ws['documents'] if d['name'] == 'b.txt')['partial'] is True
    # Pause the background worker so the retry request can be inspected deterministically.
    original_start = app.state.jobs.start
    monkeypatch.setattr(app.state.jobs, 'start', lambda *args, **kwargs: original_start(*args, background=False, **kwargs))
    response = client.post(f'/api/jobs/{job.id}/retry')
    assert response.status_code == 200
    retry = list(app.state.jobs.jobs.values())[-1]
    assert retry.config['extraction']['strict_evidence'] is True
    assert retry.target == 'Collection'


def test_corrupt_workspace_is_not_a_library_card(app, tmp_path):
    (tmp_path / 'Broken.json').write_text(json.dumps([]))
    (tmp_path / 'Broken.workspace.json').write_text(json.dumps({'schema_version': 1, 'claims': [], 'documents': 42}))
    assert [g['name'] for g in TestClient(app).get('/api/graphs').json()['graphs']] == ['Collection']
