import json
import sys
import types

import pytest

from knowledge_graph.main import (
    InputError,
    collect_input_files,
    load_triples_from_json,
    process_documents,
    read_documents,
    read_input_text,
)


def test_utf8_and_bom(tmp_path, capsys):
    p = tmp_path / "a.txt"
    p.write_bytes("héllo".encode())
    assert read_input_text(str(p)) == "héllo"
    p.write_bytes(b"\xef\xbb\xbfhello")
    assert read_input_text(str(p)) == "hello"


def test_cp1252_fallback(tmp_path, capsys):
    p = tmp_path / "w.txt"
    p.write_bytes("smart “quotes” and – dash".encode("cp1252"))
    assert "“quotes”" in read_input_text(str(p))
    assert "decoded as cp1252" in capsys.readouterr().out


def test_pdf_rejected_with_hint(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 \x80 binary")
    with pytest.raises(InputError, match="pdftotext"):
        read_input_text(str(p))


def test_binary_content_rejected(tmp_path):
    p = tmp_path / "blob.txt"
    p.write_bytes(b"abc\x00def")
    with pytest.raises(InputError, match="binary"):
        read_input_text(str(p))


def test_empty_and_missing(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   \n")
    with pytest.raises(InputError, match="empty"):
        read_input_text(str(p))
    with pytest.raises(InputError, match="Could not read"):
        read_input_text(str(tmp_path / "missing.txt"))


def test_load_triples_from_json_accepts_list_and_wrapper(tmp_path):
    triples = [{"subject": "a", "predicate": "p", "object": "b"}]
    p = tmp_path / "g.json"
    p.write_text(json.dumps(triples))
    assert load_triples_from_json(str(p)) == triples
    p.write_text(json.dumps({"triples": triples}))
    assert load_triples_from_json(str(p)) == triples
    p.write_text(json.dumps({"nodes": []}))
    with pytest.raises(InputError):
        load_triples_from_json(str(p))
    p.write_text("not json")
    with pytest.raises(InputError):
        load_triples_from_json(str(p))


# ---- wave 9: formats, directories, multiple documents ----


def test_markdown_is_plain_text(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# Title\n\nSome *text*.")
    assert read_input_text(str(p)) == "# Title\n\nSome *text*."


def test_pdf_requires_optional_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)  # simulate "not installed"
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4")
    with pytest.raises(InputError, match=r"ai-knowledge-graph\[pdf\]"):
        read_input_text(str(p))


def test_pdf_text_is_extracted_with_fake_pypdf(tmp_path, monkeypatch):
    class Page:
        def __init__(self, text): self._t = text
        def extract_text(self): return self._t
    class PdfReader:
        def __init__(self, path): self.pages = [Page("Page one."), Page(""), Page("Page two.")]
    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=PdfReader))
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4")
    assert read_input_text(str(p)) == "Page one.\n\nPage two."
    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=lambda path: types.SimpleNamespace(pages=[Page("")])))
    with pytest.raises(InputError, match="scanned"):
        read_input_text(str(p))


def test_docx_requires_optional_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "docx", None)
    p = tmp_path / "doc.docx"
    p.write_bytes(b"PK")
    with pytest.raises(InputError, match=r"ai-knowledge-graph\[docx\]"):
        read_input_text(str(p))


def test_collect_input_files_expands_directories_and_validates(tmp_path):
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "skip.png").write_bytes(b"x")
    (tmp_path / ".hidden.txt").write_text("h")
    files = collect_input_files([str(tmp_path)])
    assert [p.rsplit("/", 1)[-1] for p in files] == ["a.md", "b.txt"]
    with pytest.raises(InputError, match="not found"):
        collect_input_files([str(tmp_path / "nope.txt")])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InputError, match="No supported"):
        collect_input_files([str(empty)])


def test_read_documents_names_by_basename(tmp_path):
    (tmp_path / "one.txt").write_text("First doc.")
    (tmp_path / "two.txt").write_text("Second doc.")
    docs = read_documents([str(tmp_path / "one.txt"), str(tmp_path / "two.txt")])
    assert docs == [("one.txt", "First doc."), ("two.txt", "Second doc.")]


def test_process_documents_tags_triples_with_document(monkeypatch):
    from knowledge_graph import main as m

    class Client:
        def complete(self, user, system=None):
            doc = "alpha" if "Alpha" in user else "beta"
            return json.dumps([{"subject": doc, "predicate": "mentions", "object": "thing"}])
    monkeypatch.setattr(m.LLMClient, "from_config", classmethod(lambda cls, cfg: Client()))
    cfg = {"llm": {"model": "x", "base_url": "u"}, "standardization": {"enabled": False}, "inference": {"enabled": False}}
    out = process_documents(cfg, [("a.txt", "Alpha text here."), ("b.txt", "Beta text here.")])
    assert {(t["subject"], t["document"]) for t in out} == {("alpha", "a.txt"), ("beta", "b.txt")}
    single = process_documents(cfg, [("a.txt", "Alpha text here.")])
    assert "document" not in single[0]  # a single document is not tagged
