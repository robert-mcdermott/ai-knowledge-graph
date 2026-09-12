import json

import pytest

from knowledge_graph.main import InputError, load_triples_from_json, read_input_text


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
