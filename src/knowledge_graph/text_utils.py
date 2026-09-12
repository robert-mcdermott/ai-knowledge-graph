"""Text processing utilities: sentence splitting, sentence-aware chunking, provenance lookup."""
from __future__ import annotations

import re

# End of sentence: terminal punctuation (optionally followed by a closing quote/bracket) then whitespace.
_SENTENCE_END = re.compile(r'(?<=[.!?])["\'”’)\]]*\s+')
_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "no."}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (paragraph breaks always split; common abbreviations do not)."""
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        pieces = _SENTENCE_END.split(paragraph)
        current = ""
        for piece in pieces:
            current = f"{current} {piece}".strip() if current else piece
            last_word = current.split()[-1].lower() if current.split() else ""
            if last_word in _ABBREVIATIONS or (len(last_word) <= 2 and last_word.endswith(".")):
                continue  # "Dr." / "J." — keep accumulating
            sentences.append(current)
            current = ""
        if current:
            sentences.append(current)
    return sentences


def chunk_words(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks of ``chunk_size`` words with ``overlap`` words repeated between chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if words else []
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks of roughly ``chunk_size`` words without cutting sentences.

    Chunks end on sentence boundaries; the next chunk starts with the trailing sentences of
    the previous one so that about ``overlap`` words are shared. A single sentence longer
    than ``chunk_size`` is split by words.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words_total = len(text.split())
    if words_total <= chunk_size:
        return [text.strip()] if text.strip() else []

    units: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence.split()) > chunk_size:
            units.extend(chunk_words(sentence, chunk_size, 0))
        else:
            units.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        n = len(unit.split())
        if current and current_words + n > chunk_size:
            chunks.append(" ".join(current))
            # Carry trailing sentences forward as overlap (at most half a chunk).
            carried, carried_words = [], 0
            for prev in reversed(current):
                pw = len(prev.split())
                if carried_words + pw > max(overlap, 0) or carried_words + pw > chunk_size // 2:
                    break
                carried.insert(0, prev)
                carried_words += pw
            current, current_words = carried, carried_words
        current.append(unit)
        current_words += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def find_source_sentence(text: str, subject: str, obj: str, max_length: int = 300) -> str | None:
    """Return the sentence of ``text`` that mentions both entities (or at least the subject)."""
    subject_l, obj_l = subject.lower(), obj.lower()
    fallback = None
    for sentence in split_sentences(text):
        s = sentence.lower()
        if subject_l in s and obj_l in s:
            return _trim(sentence, max_length)
        if fallback is None and (subject_l in s or obj_l in s):
            fallback = sentence
    return _trim(fallback, max_length) if fallback else None


def _trim(sentence: str, max_length: int) -> str:
    sentence = sentence.strip()
    return sentence if len(sentence) <= max_length else sentence[:max_length - 1].rstrip() + "…"
