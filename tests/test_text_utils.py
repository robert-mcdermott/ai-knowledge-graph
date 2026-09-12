import pytest

from knowledge_graph.text_utils import chunk_text, chunk_words, find_source_sentence, split_sentences


def test_split_sentences_handles_abbreviations_and_paragraphs():
    text = "Dr. Watt refined the engine. It spread to Europe!\n\nRailways followed. Mr. Stephenson built them."
    assert split_sentences(text) == ["Dr. Watt refined the engine.", "It spread to Europe!", "Railways followed.",
                                     "Mr. Stephenson built them."]


def test_chunk_text_respects_sentence_boundaries_and_overlaps():
    sentences = [f"Sentence number {i} has exactly six words." for i in range(1, 21)]  # 7 words each
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=30, overlap=7)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.endswith(".")  # never cut mid-sentence
        assert len(chunk.split()) <= 30
    # every sentence appears in some chunk, and consecutive chunks share the overlap sentence
    assert all(any(s in c for c in chunks) for s in sentences)
    assert chunks[0].split(". ")[-1].rstrip(".") in chunks[1]


def test_chunk_text_short_text_and_empty():
    assert chunk_text("Short text.", 100, 10) == ["Short text."]
    assert chunk_text("   ", 100, 10) == []


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", 5, 5)


def test_very_long_sentence_is_split_by_words():
    text = " ".join(f"w{i}" for i in range(120))  # no punctuation at all
    chunks = chunk_text(text, chunk_size=50, overlap=5)
    assert len(chunks) >= 3 and all(len(c.split()) <= 50 for c in chunks)


def test_chunk_words_overlap():
    chunks = chunk_words(" ".join(f"w{i}" for i in range(1, 26)), 10, 3)
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert chunks[-1].split()[-1] == "w25"


def test_find_source_sentence_prefers_both_entities():
    text = "James Watt was born in Scotland. Watt refined the steam engine in 1776. The engine spread widely."
    assert find_source_sentence(text, "james watt", "steam engine") == "Watt refined the steam engine in 1776." or \
        find_source_sentence(text, "watt", "steam engine") == "Watt refined the steam engine in 1776."
    assert find_source_sentence(text, "watt", "railways") == "James Watt was born in Scotland."  # subject-only fallback
    assert find_source_sentence(text, "nobody", "nothing") is None


def test_find_source_sentence_trims_long_sentences():
    text = "start " + "word " * 200 + "end."
    result = find_source_sentence(text, "start", "end", max_length=50)
    assert result.endswith("…") and len(result) <= 50
