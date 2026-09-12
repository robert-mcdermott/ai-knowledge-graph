import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_docs  # noqa: E402


def test_build_docs_writes_landing_page_and_sample_pages(tmp_path):
    cards = build_docs.build(docs_dir=str(tmp_path))
    assert cards, "no sample graphs found"
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    for card in cards:
        page = tmp_path / card["href"]
        assert page.exists()
        html = page.read_text(encoding="utf-8")
        assert '<script src="../vendor/vis-network.min.js"></script>' in html
        assert card["href"] in index and card["title"] in index
    assert (tmp_path / "vendor" / "vis-network.min.js").exists()
    assert sum(1 for c in cards if c["community_names"]) >= len(cards) - 1  # names come from the sidecars
