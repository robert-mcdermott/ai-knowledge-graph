import xml.etree.ElementTree as ET

import pytest

from knowledge_graph.exports import cypher_label, cypher_rel_type, export_graph, parse_formats

TRIPLES = [
    {"subject": "james watt", "predicate": "refined", "object": "steam engine", "subject_type": "person",
     "object_type": "technology", "source": "Watt refined the steam engine.", "chunk": 1, "document": "a.txt"},
    {"subject": "steam engine", "predicate": "used in", "object": "mills"},
    {"subject": "steam engine", "predicate": "is a", "object": "engine", "inferred": True, "method": "taxonomy"},
    {"subject": "james watt", "predicate": "influenced", "object": "mills", "inferred": True, "method": "transitive",
     "via": "steam engine"},
    {"subject": 'quote "name"', "predicate": "2nd-order relates to", "object": "mills"},
]


def test_parse_formats():
    assert parse_formats("graphml, Cypher,csv") == ["graphml", "cypher", "csv"]
    with pytest.raises(ValueError, match="bogus"):
        parse_formats("json,bogus")


def test_export_writes_all_formats(tmp_path):
    out = tmp_path / "graph.html"
    paths = export_graph(TRIPLES, str(out), ["json", "csv", "graphml", "cypher"])
    assert [p.rsplit(".", 1)[1] for p in paths] == ["json", "csv", "graphml", "cypher"]
    assert all((tmp_path / p.rsplit("/", 1)[-1]).exists() for p in paths)


def test_csv_has_header_and_provenance(tmp_path):
    export_graph(TRIPLES, str(tmp_path / "g.html"), ["csv"])
    lines = (tmp_path / "g.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("subject,predicate,object,subject_type,object_type,inferred,method,via,chunk,document,source")
    assert "Watt refined the steam engine." in lines[1] and lines[1].split(",")[5] == "false"
    assert any(",true,taxonomy," in line for line in lines)


def test_graphml_carries_node_and_edge_attributes(tmp_path):
    export_graph(TRIPLES, str(tmp_path / "g.html"), ["graphml"])
    root = ET.parse(tmp_path / "g.graphml").getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    keys = {k.get("attr.name") for k in root.findall("g:key", ns)}
    assert {"type", "community", "community_name", "degree", "predicate", "inferred", "method", "via", "source"} <= keys
    graph = root.find("g:graph", ns)
    assert len(graph.findall("g:node", ns)) == 5
    assert len(graph.findall("g:edge", ns)) == len(TRIPLES)  # parallel edges preserved


def test_cypher_is_idempotent_merge_with_labels_and_types(tmp_path):
    export_graph(TRIPLES, str(tmp_path / "g.html"), ["cypher"])
    text = (tmp_path / "g.cypher").read_text(encoding="utf-8")
    assert "CREATE CONSTRAINT" in text
    assert 'MERGE (e:Entity {name: "james watt"}) SET e:Person' in text
    assert 'MERGE (e:Entity {name: "mills"}) SET e:Entity' in text  # untyped node keeps the base label
    assert '-[r:REFINED {predicate: "refined", time: "", polarity: "", attribution: ""}]->' in text
    assert "-[r:R_2ND_ORDER_RELATES_TO " in text
    assert 'name: "quote \\"name\\""' in text  # quotes escaped
    assert 'via: "steam engine"' in text and "inferred: true" in text and "chunk: 1" in text
    assert "CREATE " not in text.replace("CREATE CONSTRAINT", "")  # everything else is MERGE


def test_label_and_rel_type_helpers():
    assert cypher_label("technology") == "Technology"
    assert cypher_label(None) == "Entity"
    assert cypher_rel_type("led to") == "LED_TO"
    assert cypher_rel_type("  ") == "RELATED_TO"
