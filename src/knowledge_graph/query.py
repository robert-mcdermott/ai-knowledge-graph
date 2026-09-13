"""Ask questions about a generated knowledge graph ("chat with the graph").

This is an optional companion to ``generate-graph``: it reads the ``.json`` triples file a
normal run writes, retrieves the part of the graph relevant to a question, and asks the
configured LLM to answer from those facts only. The static HTML workflow is unaffected.

Usage::

    graph-chat graph.json "How did the steam engine change cities?"
    graph-chat graph.json            # interactive session

The retrieval functions (:func:`match_entities`, :func:`retrieve_subgraph`) are pure and
reusable, e.g. for a future in-page chat.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field

import networkx as nx

from knowledge_graph.config import load_config
from knowledge_graph.entity_standardization import _singularize
from knowledge_graph.llm import LLMClient, LLMError, extract_json_from_text
from knowledge_graph.logging_utils import configure_logging
from knowledge_graph.prompts import prompt_factory, system_prompt_for

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "from", "at", "as", "is", "are",
    "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "what", "which", "who", "whom",
    "how", "why", "when", "where", "did", "do", "does", "has", "have", "had", "can", "could", "would", "should",
    "about", "into", "than", "then", "there", "their", "they", "them", "between", "relate", "related", "relation",
    "relationship", "relationships", "connect", "connected", "connection", "affect", "affected", "impact", "role",
    "explain", "describe", "tell", "me", "us", "any", "some", "all", "most", "many", "much", "over", "under",
}


@dataclass
class GraphIndex:
    """In-memory index of a triples list for retrieval."""

    triples: list[dict]
    adjacency: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    by_pair: dict[tuple[str, str], list[int]] = field(default_factory=lambda: defaultdict(list))
    degree: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    terms: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    names: dict[str, str] = field(default_factory=dict)  # lower-case -> canonical

    @classmethod
    def build(cls, triples):
        index = cls(triples=triples)
        for i, t in enumerate(triples):
            s, o = t["subject"], t["object"]
            index.adjacency[s].add(o)
            index.adjacency[o].add(s)
            index.by_pair[(s, o)].append(i)
            index.by_pair[(o, s)].append(i)
            index.degree[s] += 1
            index.degree[o] += 1
            index.names[s.lower()] = s
            index.names[o.casefold()] = o
            for side in ("subject", "object"):
                for alias in t.get(side + "_aliases", []):
                    index.names[alias.casefold()] = t[side]
            searchable = " ".join([s, o, t["predicate"], t.get("source", "")] +
                                  [e.get("text", "") for e in t.get("evidence", [])])
            for term in _content_words(searchable):
                index.terms[term].add(i)
        return index

    @property
    def entities(self):
        return sorted(set(self.names.values()), key=lambda n: (-self.degree[n], n))


def _content_words(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    words = {w for w in re.findall(r"[\w+#'-]+", text) if len(w) >= 2 and w not in _STOPWORDS}
    # Bigrams provide a useful no-dependency baseline for unsegmented CJK text.
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        words.update(run[i:i + 2] for i in range(len(run) - 1))
    return {_singularize(w) for w in words}


def _words_match(a, b):
    """Same word after singularization, or a one-letter tense/plural difference ("railway"/"railways")."""
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= 5 and longer.startswith(shorter) and len(longer) - len(shorter) <= 1


def match_entities(index, question, limit=8):
    """Entities the question is about: whole-name matches first, then shared content words."""
    q = " " + re.sub(r"[^\w+#'\- ]+", " ", unicodedata.normalize("NFKC", question).casefold()) + " "
    q = re.sub(r"\s+", " ", q)
    matched = list(dict.fromkeys(name for lower, name in index.names.items()
                                if f" {lower} " in q or (re.search(r"[\u3400-\u9fff]", lower) and lower in q)))
    # Drop names contained in a longer matched name ("engine" when "steam engine" matched).
    matched = [n for n in matched if not any(n != m and f" {n.lower()} " in f" {m.lower()} " for m in matched)]
    matched.sort(key=lambda n: (-len(n), -index.degree[n]))
    if len(matched) < limit:
        words = _content_words(question)
        padded_matches = [f" {m.lower()} " for m in matched]
        seen = set(matched) | {n for n in index.names.values() if any(f" {n.lower()} " in pm for pm in padded_matches)}
        candidates = []
        for name in index.entities:
            if name in seen:
                continue
            name_words = _content_words(name) or {name.lower()}
            overlap = sum(1 for nw in name_words if any(_words_match(nw, qw) for qw in words))
            if overlap:
                candidates.append((-overlap / len(name_words), -index.degree[name], name))
        candidates.sort()
        matched.extend(name for _, _, name in candidates[: limit - len(matched)])
    return matched[:limit]


def shortest_path(index, start, goal, max_depth=6):
    """Node list of a shortest path between two entities (undirected), or None."""
    if start == goal:
        return [start]
    parent = {start: None}
    queue = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nxt in sorted(index.adjacency.get(node, ())):
            if nxt in parent:
                continue
            parent[nxt] = node
            if nxt == goal:
                path = [goal]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            queue.append((nxt, depth + 1))
    return None


def retrieve_subgraph(index, seeds, hops=2, max_triples=150):
    """Triple indices around the seeds: seed-to-seed paths first, then by hop distance."""
    seeds = [s for s in seeds if s in index.adjacency]
    if not seeds:
        return []
    distance = {}
    queue = deque()
    for s in seeds:
        distance[s] = 0
        queue.append(s)
    while queue:
        node = queue.popleft()
        if distance[node] >= hops:
            continue
        for nxt in sorted(index.adjacency.get(node, ())):
            if nxt not in distance:
                distance[nxt] = distance[node] + 1
                queue.append(nxt)

    chosen = []
    seen = set()

    def add(indices):
        for i in indices:
            if i not in seen:
                seen.add(i)
                chosen.append(i)

    # Paths between seeds explain "how does A relate to B" questions.
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            path = shortest_path(index, a, b)
            if path:
                for u, v in zip(path, path[1:], strict=False):
                    add(index.by_pair[(u, v)])

    ranked = []
    for i, t in enumerate(index.triples):
        if i in seen:
            continue
        ds, do = distance.get(t["subject"]), distance.get(t["object"])
        if ds is None and do is None:
            continue
        d = min(x for x in (ds, do) if x is not None)
        both = ds is not None and do is not None
        ranked.append((d, 0 if both else 1, 1 if t.get("inferred") else 0,
                       -(index.degree[t["subject"]] + index.degree[t["object"]]), i))
    ranked.sort()
    add(i for *_, i in ranked)
    return chosen[:max_triples]


def format_facts(index, indices):
    """Numbered fact lines for the prompt; returns (text, numbering) with 1-based numbers."""
    lines = []
    numbering = {}
    for n, i in enumerate(indices, start=1):
        t = index.triples[i]
        numbering[n] = i
        line = f"[{n}] {t['subject']} → {t['predicate']} → {t['object']}"
        if t.get("inferred"):
            line += f"  (inferred: {t.get('method', 'unknown')}" + (f" via {t['via']}" if t.get("via") else "") + ")"
        else:
            line += "  (extracted" + (f': "{t["source"]}"' if t.get("source") else "") + ")"
        if t.get("document"):
            line += f" [document: {t['document']}]"
        for qualifier in ("time", "polarity", "attribution"):
            if t.get(qualifier):
                line += f" [{qualifier}: {t[qualifier]}]"
        for evidence in t.get("evidence", [])[1:4]:
            line += f"\n    Additional passage ({evidence.get('document', 'source')}, page {evidence.get('page', '?')}): {evidence['text']}"
        lines.append(line)
    return "\n".join(lines), numbering


_FACTS_USED = re.compile(r"facts?\s+used\s*:\s*\[?([0-9,\s]*)\]?", re.IGNORECASE)


def parse_answer(text):
    """Split the model reply into (answer, cited fact numbers)."""
    cited = []
    match = _FACTS_USED.search(text)
    if match:
        cited = [int(x) for x in re.findall(r"\d+", match.group(1))]
        text = text[: match.start()].rstrip()
    cited = list(dict.fromkeys(cited + [int(n) for group in re.findall(r"\[([0-9, ]+)\]", text) for n in re.findall(r"\d+", group)]))
    return text.strip(), cited


class GraphChat:
    """Answer questions about a triples list with the configured LLM."""

    def __init__(self, triples, config, client=None):
        self.index = GraphIndex.build(triples)
        self.config = config
        self.client = client or LLMClient.from_config(config)
        self.history: list[tuple[str, str]] = []
        self.last_seeds: list[str] = []
        q = config.get("query", {})
        self.hops = int(q.get("hops", 2))
        self.max_triples = int(q.get("max_triples", 150))
        self.max_seeds = int(q.get("max_seed_entities", 8))
        self.llm_matching = bool(q.get("use_llm_for_entity_matching", True))
        self.history_turns = int(q.get("history_turns", 3))
        self.max_context_tokens = int(q.get("max_context_tokens", 6000))
        self._lock = __import__("threading").Lock()

    def seeds_for(self, question):
        seeds = match_entities(self.index, question, self.max_seeds)
        if not seeds and self.last_seeds and re.search(r"\b(it|they|them|that|those|he|she|and|also)\b|它|他们|她|他", question, re.I):
            seeds = self.last_seeds  # follow-up such as "and who built it?" keeps the previous context
        if not seeds:
            hits = passage_search(self.index, question)
            seeds = list(dict.fromkeys(self.index.triples[i][k] for i in hits[:4] for k in ("subject", "object")))[:self.max_seeds]
        if not seeds and self.llm_matching:
            seeds = self._pick_entities_with_llm(question)
        if not seeds:  # fall back to the hubs so the model can at least say what the graph covers
            seeds = self.index.entities[: self.max_seeds]
        return seeds

    def _pick_entities_with_llm(self, question):
        entities = self.index.entities[:300]
        user_prompt = prompt_factory.get_prompt("entity_pick_user", question, "\n".join(entities))
        try:
            reply = self.client.complete(user_prompt, system_prompt_for("entity_pick_system", self.config))
        except LLMError:
            return []
        picked = extract_json_from_text(reply, expect="array") or []
        valid = {e.lower(): e for e in entities}
        return [valid[str(p).lower()] for p in picked if str(p).lower() in valid][: self.max_seeds]

    def ask(self, question, extracted_only=False):
        with self._lock:
            return self._ask(question, extracted_only)

    def _ask(self, question, extracted_only=False):
        """Return a dict with the answer, the seed entities, and the cited facts."""
        broad = bool(re.search(r"main themes|overview|summari[sz]e|across (?:the |my )?documents|big picture|主要主题|总结", question, re.I))
        index = GraphIndex.build([t for t in self.index.triples if not t.get("inferred")]) if extracted_only else self.index
        seeds = self.seeds_for(question) if not broad else []
        if broad:
            indices = community_context(index, self.max_triples)
        else:
            neighborhood = retrieve_subgraph(index, seeds, self.hops, self.max_triples)
            source_hits = passage_search(index, question)[:min(20, self.max_triples)]
            extra = [i for i in source_hits if i not in neighborhood]
            indices = list(dict.fromkeys(neighborhood[:max(0, self.max_triples - len(extra))] + source_hits))[:self.max_triples]
        # Budget actual serialized context, not just the number of relationships.
        kept, used = [], 0
        for i in indices:
            line, _ = format_facts(index, [i])
            cost = max(len(line.split()) * 2, math.ceil(len(line) / 3))
            if used + cost <= self.max_context_tokens:
                kept.append(i)
                used += cost
        facts_text, numbering = format_facts(index, kept)
        history = self.history[-self.history_turns:] if self.history_turns else []
        history_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
        user_prompt = prompt_factory.get_prompt("query_user", question, facts_text or "(no facts found)", history_text)
        reply = self.client.complete(user_prompt, system_prompt_for("query_system", self.config))
        answer, cited = parse_answer(reply)
        facts = [index.triples[numbering[n]] for n in cited if n in numbering]
        citations = {str(n): index.triples[numbering[n]] for n in cited if n in numbering}
        if self.history_turns:
            self.history = (self.history + [(question, answer)])[-self.history_turns:]
        else:
            self.history.clear()
        self.last_seeds = seeds
        return {"question": question, "answer": answer, "seeds": seeds, "facts": facts,
                "citations": citations, "citation_status": "invalid" if any(n not in numbering for n in cited)
                else "linked" if facts else "uncited", "mode": "overview" if broad else "focused",
                "facts_considered": len(kept), "context_tokens_estimate": used}


def passage_search(index, question):
    scores = defaultdict(float)
    for word in _content_words(question):
        hits = index.terms.get(word, set())
        weight = math.log(1 + len(index.triples) / (1 + len(hits)))
        for i in hits:
            scores[i] += weight
    return sorted(scores, key=lambda i: (-scores[i], bool(index.triples[i].get("inferred")), i))


def community_context(index, limit):
    """Balanced community evidence for broad questions, without extra LLM calls."""
    graph = nx.Graph()
    graph.add_edges_from((t["subject"], t["object"]) for t in index.triples)
    if not graph:
        return []
    groups = nx.community.louvain_communities(graph, seed=42)
    buckets = []
    for group in sorted(groups, key=lambda g: (-len(g), min(g))):
        bucket = [i for i, t in enumerate(index.triples) if t["subject"] in group]
        bucket.sort(key=lambda i: (bool(index.triples[i].get("inferred")), -len(index.triples[i].get("evidence", [])), i))
        buckets.append(deque(bucket))
    result = []
    while any(buckets) and len(result) < limit:
        for bucket in buckets:
            if bucket and len(result) < limit:
                result.append(bucket.popleft())
    return result


def _print_result(result, show_facts=True):
    print(result["answer"])
    if show_facts and result["facts"]:
        print("\nFacts used:")
        for t in result["facts"]:
            tag = f"inferred ({t.get('method', 'unknown')})" if t.get("inferred") else "extracted"
            print(f"  - {t['subject']} → {t['predicate']} → {t['object']}  [{tag}]")
            if t.get("source"):
                print(f"      “{t['source']}”")
    print(f"\n({len(result['facts'])} facts cited of {result['facts_considered']} considered; "
          f"entities: {', '.join(result['seeds']) or 'none'})")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="graph-chat", description="Ask questions about a generated knowledge graph")
    parser.add_argument("graph", help="Triples JSON written by generate-graph (e.g. knowledge_graph.json)")
    parser.add_argument("question", nargs="?", help="Question to answer; omit for an interactive session")
    parser.add_argument("--config", default="config.toml", help="Path to configuration file (uses its [llm] section)")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON (single-question mode)")
    parser.add_argument("--extracted-only", action="store_true", help="Exclude inferred relationships from answers")
    parser.add_argument("--no-facts", action="store_true", help="Do not list the cited facts after the answer")
    args = parser.parse_args(argv)
    configure_logging(logging.WARNING)

    config = load_config(args.config)
    if not config:
        sys.exit(1)
    from knowledge_graph.main import InputError, load_triples_from_json  # local import avoids a cycle
    try:
        triples = load_triples_from_json(args.graph)
    except InputError as e:
        print(f"Error: {e}")
        sys.exit(1)
    chat = GraphChat(triples, config)

    if args.question:
        try:
            result = chat.ask(args.question, extracted_only=args.extracted_only)
        except LLMError as e:
            print(f"Error: {e}")
            sys.exit(1)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            _print_result(result, show_facts=not args.no_facts)
        return

    print(f"Loaded {len(triples)} relationships from {os.path.basename(args.graph)}. "
          f"Ask a question (empty line or Ctrl-D to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        try:
            _print_result(chat.ask(question, extracted_only=args.extracted_only), show_facts=not args.no_facts)
        except LLMError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
