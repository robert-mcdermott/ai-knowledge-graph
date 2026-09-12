"""Entity standardization and relationship inference for knowledge graphs.

Two public entry points:

* :func:`standardize_entities` merges different surface forms of the same entity.
* :func:`infer_relationships` adds *inferred* triples. Every inferred triple carries
  ``inferred: True`` and a ``method``: ``llm_bridge`` (links an isolated component to
  the main graph), ``llm_hub`` (general-knowledge links between central entities),
  ``llm_within`` (lexically related pairs inside a component), ``taxonomy``
  (``quantum computing`` is a ``computing``), ``transitive`` (with the intermediate
  node in ``via``) or ``lexical``.

Inference is deliberately conservative. Rule-based methods are opt-in, the total
number of inferred edges is capped relative to the extracted ones, and a node pair
is never connected twice.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict, deque

from knowledge_graph.llm import LLMClient, extract_json_from_text
from knowledge_graph.prompts import prompt_factory, system_prompt_for

logger = logging.getLogger(__name__)

REQUIRED_KEYS = ("subject", "predicate", "object")

# Predicate families for which A -p-> B -p-> C reasonably implies A -p-> C.
# Keys are the canonical predicate written on the inferred edge.
TRANSITIVE_PREDICATE_GROUPS: dict[str, set[str]] = {
    "is a": {"is a", "is an", "is", "is type of", "type of", "subclass of", "kind of", "instance of"},
    "part of": {"part of", "component of", "belongs to", "member of", "included in", "contained in",
                "within"},
    "located in": {"located in", "in", "situated in", "based in", "city in", "country in", "region of"},
    "led to": {"led to", "leads to", "lead to", "resulted in", "results in", "caused", "causes",
               "enabled", "enables", "gave rise to", "paved way for", "contributed to", "drove",
               "fueled", "spurred", "triggered"},
}

INFERENCE_PRIORITY = ("llm_bridge", "llm_hub", "llm_within", "taxonomy", "transitive", "lexical")

# Words that, when they immediately precede a candidate head noun, mean the phrase is
# not "<modifier> <head>" ("developments in electronics" is not a kind of electronics).
_PHRASE_BREAKERS = {"in", "of", "for", "on", "at", "to", "with", "by", "from", "and", "or", "the", "a", "an"}

_PLURAL_EXCEPTIONS = {"physics", "economics", "politics", "mathematics", "ethics", "news", "series",
                      "species", "analysis", "crisis", "basis", "thesis", "bus", "gas", "plus", "status"}

_LEXICAL_STOPWORDS = {
    "about", "above", "after", "again", "against", "along", "among", "around", "because", "before",
    "being", "below", "between", "during", "early", "first", "great", "having", "later", "major",
    "modern", "other", "second", "several", "since", "their", "there", "these", "third", "those",
    "through", "under", "until", "where", "which", "while", "would", "years", "system", "systems",
    "process", "processes", "industry", "industries", "industrial", "revolution", "revolutions",
    "technology", "technologies", "development", "developments", "production",
}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def limit_predicate_length(predicate, max_words=3):
    """Enforce a maximum word limit on predicates (drops a trailing stop-word)."""
    words = predicate.split()
    if len(words) <= max_words:
        return predicate
    shortened = " ".join(words[:max_words])
    stop_words = {"a", "an", "the", "of", "with", "by", "to", "from", "in", "on", "for", "via"}
    if shortened.split()[-1].lower() in stop_words and len(words) > 1:
        shortened = " ".join(shortened.split()[:-1])
    return shortened


def _valid_triples(triples, context):
    valid, invalid = [], 0
    for triple in triples:
        if isinstance(triple, dict) and all(k in triple for k in REQUIRED_KEYS):
            valid.append(triple)
        else:
            invalid += 1
    if invalid:
        print(f"Warning: filtered out {invalid} invalid triples missing required fields ({context})", flush=True)
    return valid


def _pair_key(subject, obj):
    """Order-independent key so A→B and B→A count as the same connection."""
    return (subject, obj) if subject <= obj else (obj, subject)


def _norm_pred(predicate):
    return re.sub(r"\s+", " ", predicate.strip().lower())


def _degrees(triples):
    degree = Counter()
    for t in triples:
        degree[t["subject"]] += 1
        degree[t["object"]] += 1
    return degree


def _make_inferred(subject, predicate, obj, method, **extra):
    triple = {"subject": subject, "predicate": limit_predicate_length(predicate), "object": obj,
              "inferred": True, "method": method}
    triple.update(extra)
    return triple


# --------------------------------------------------------------------------- #
# Standardization
# --------------------------------------------------------------------------- #
def standardize_entities(triples, config):
    """Standardize entity names across all triples.

    Always merges forms that differ only by case, whitespace or stop-words
    ("The Steam Engine" / "steam engine"). Merging by shared words or word stems
    ("steam engine factories" → "steam engine") is aggressive and only runs when
    ``standardization.merge_word_subsets`` is true. LLM-based resolution runs when
    ``standardization.use_llm_for_entities`` is true.
    """
    if not triples:
        return triples

    print("Standardizing entity names across all triples...", flush=True)
    std_cfg = config.get("standardization", {})
    valid_triples = _valid_triples(triples, "standardization")
    if not valid_triples:
        print("Error: No valid triples found for entity standardization", flush=True)
        return []

    all_entities = set()
    for triple in valid_triples:
        all_entities.add(triple["subject"].lower())
        all_entities.add(triple["object"].lower())

    counts = Counter()
    for triple in valid_triples:
        counts[triple["subject"].lower()] += 1
        counts[triple["object"].lower()] += 1

    # Pass 1: exact match after lower-casing and stop-word removal.
    stopwords = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "with", "by", "as"}

    def normalize_text(text):
        words = [_singularize(w) for w in re.findall(r"\b\w+\b", text.lower()) if w not in stopwords]
        return " ".join(words)

    groups = defaultdict(list)
    for entity in sorted(all_entities, key=lambda x: (-len(x), x)):
        normalized = normalize_text(entity)
        if normalized:
            groups[normalized].append(entity)

    mapping = {}
    for variants in groups.values():
        standard = sorted(variants, key=lambda v: (-counts[v], len(v)))[0]
        for variant in variants:
            mapping[variant] = standard
            if variant != standard:
                logger.debug("standardize (normalize): %r -> %r", variant, standard)

    # Pass 2 (opt-in): shared-word / shared-stem merging.
    if std_cfg.get("merge_word_subsets", False):
        for entity, standard in _word_subset_merges(set(mapping.values())).items():
            logger.debug("standardize (word subset): %r -> %r", entity, standard)
            mapping[entity] = standard
        # Re-point anything that mapped to a now-merged standard.
        for entity, standard in list(mapping.items()):
            while mapping.get(standard, standard) != standard:
                standard = mapping[standard]
            mapping[entity] = standard

    standardized = []
    for triple in valid_triples:
        new_triple = dict(triple)
        new_triple["subject"] = mapping.get(triple["subject"].lower(), triple["subject"])
        new_triple["object"] = mapping.get(triple["object"].lower(), triple["object"])
        new_triple["predicate"] = limit_predicate_length(triple["predicate"])
        new_triple.setdefault("chunk", 0)
        standardized.append(new_triple)

    if std_cfg.get("use_llm_for_entities", False):
        standardized = _resolve_entities_with_llm(standardized, config)

    filtered = [t for t in standardized if t["subject"] != t["object"]]
    if len(filtered) < len(standardized):
        print(f"Removed {len(standardized) - len(filtered)} self-referencing triples", flush=True)
    normalize_predicates(filtered)

    print(f"Standardized {len(all_entities)} entities into {len(set(mapping.values()))} standard forms", flush=True)
    return filtered


def normalize_predicates(triples):
    """Canonicalize predicates in place: case/whitespace, and merge tense variants
    that differ only by a trailing "s" on the verb ("involve" / "involves") when both occur,
    keeping the more frequent form."""
    counts = Counter()
    for t in triples:
        t["predicate"] = _norm_pred(t["predicate"])
        counts[t["predicate"]] += 1

    def key(pred):
        words = pred.split()
        if not words:
            return pred
        first = words[0]
        if first.endswith("ies") and len(first) > 4:
            first = first[:-3] + "y"
        elif first.endswith(("ses", "xes", "ches", "shes")):
            first = first[:-2]
        elif first.endswith("s") and not first.endswith("ss") and len(first) > 3:
            first = first[:-1]
        return " ".join([first] + words[1:])

    groups = defaultdict(list)
    for pred in counts:
        groups[key(pred)].append(pred)
    canonical = {}
    merged = 0
    for variants in groups.values():
        best = max(variants, key=lambda v: (counts[v], -len(v)))
        for v in variants:
            canonical[v] = best
            if v != best:
                merged += 1
    if merged:
        for t in triples:
            t["predicate"] = canonical[t["predicate"]]
        print(f"Merged {merged} predicate variants (e.g. tense forms)", flush=True)
    return triples


def _singularize(word):
    """Cheap English singularization used only to *group* variants ("factories"/"factory")."""
    if len(word) <= 3 or word in _PLURAL_EXCEPTIONS or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "ches", "shes", "zes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def _word_subset_merges(standard_forms):
    """Aggressive merges: one name's words are a subset of another's, or they share word stems."""
    merges = {}
    sorted_standards = sorted(standard_forms, key=len)
    for i, entity1 in enumerate(sorted_standards):
        e1_words = set(entity1.split())
        for entity2 in sorted_standards[i + 1:]:
            e2_words = set(entity2.split())
            if e1_words and e1_words.issubset(e2_words):
                merges[entity2] = entity1
            elif e2_words and e2_words.issubset(e1_words):
                merges[entity1] = entity2
            else:
                stems1 = {w[:4] for w in e1_words if len(w) > 4}
                stems2 = {w[:4] for w in e2_words if len(w) > 4}
                shared = stems1 & stems2
                if shared and len(shared) / max(len(stems1), len(stems2)) > 0.5:
                    if len(entity1) <= len(entity2):
                        merges[entity2] = entity1
                    else:
                        merges[entity1] = entity2
    return merges


def _resolve_entities_with_llm(triples, config):
    """Ask the LLM to group variants of the same entity and apply the mapping."""
    counts = _degrees(triples)
    entities = [e for e, _ in counts.most_common(100)]  # cap prompt size

    system_prompt = system_prompt_for("entity_resolution_system", config)
    user_prompt = prompt_factory.get_prompt("entity_resolution_user", "\n".join(sorted(entities)))

    try:
        response = LLMClient.from_config(config).complete(user_prompt, system_prompt)
        entity_mapping = extract_json_from_text(response, expect="object")
        if not entity_mapping:
            print("Could not extract valid entity mapping from LLM response", flush=True)
            return triples

        entity_to_standard = {}
        for standard, variants in entity_mapping.items():
            if not isinstance(variants, list):
                continue
            for variant in variants:
                if isinstance(variant, str) and variant != standard:
                    entity_to_standard[variant] = standard
                    logger.debug("standardize (llm): %r -> %r", variant, standard)
            entity_to_standard[standard] = standard

        for triple in triples:
            triple["subject"] = entity_to_standard.get(triple["subject"], triple["subject"])
            triple["object"] = entity_to_standard.get(triple["object"], triple["object"])
        print(f"Applied LLM-based entity standardization for {len(entity_mapping)} entity groups", flush=True)
    except Exception as e:  # optional phase: never abort the run
        print(f"Error in LLM-based entity resolution: {e}", flush=True)
    return triples


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def infer_relationships(triples, config):
    """Add inferred triples according to the ``[inference]`` config.

    Methods run in priority order (LLM between communities, LLM within
    communities, transitive rules, lexical similarity). Candidates that would
    connect an already-connected node pair are dropped, and the total is capped
    at ``max_inferred_ratio`` × the number of extracted triples.
    """
    if not triples or len(triples) < 2:
        return triples

    print("Inferring additional relationships between entities...", flush=True)
    inf_cfg = config.get("inference", {})
    valid_triples = _valid_triples(triples, "inference")
    if not valid_triples:
        print("Error: No valid triples found for relationship inference", flush=True)
        return []

    graph = defaultdict(set)
    all_entities = set()
    for t in valid_triples:
        graph[t["subject"]].add(t["object"])
        all_entities.update((t["subject"], t["object"]))
    degree = _degrees(valid_triples)

    communities = _identify_communities(graph)
    print(f"Identified {len(communities)} disconnected components in the graph", flush=True)

    candidates: dict[str, list] = {m: [] for m in INFERENCE_PRIORITY}
    if inf_cfg.get("use_llm_for_inference", True):
        if inf_cfg.get("llm_bridge", True):
            candidates["llm_bridge"] = _infer_bridges_with_llm(valid_triples, communities, degree, config)
        if inf_cfg.get("llm_hub", True):
            candidates["llm_hub"] = _infer_hub_relationships_with_llm(valid_triples, degree, config)
        candidates["llm_within"] = _infer_within_community_relationships(valid_triples, communities, config)
    if inf_cfg.get("taxonomy", True):
        candidates["taxonomy"] = _infer_taxonomy(all_entities, valid_triples)
    if inf_cfg.get("apply_transitive", False):
        candidates["transitive"] = _apply_transitive_inference(valid_triples, graph, degree, inf_cfg)
    if inf_cfg.get("lexical", False):
        candidates["lexical"] = _infer_relationships_by_lexical_similarity(
            all_entities, valid_triples, min_word_length=int(inf_cfg.get("lexical_min_word_length", 5)))

    extracted_count = sum(1 for t in valid_triples if not t.get("inferred"))
    ratio = float(inf_cfg.get("max_inferred_ratio", 0.5))
    budget = int(extracted_count * ratio) if ratio >= 0 else None

    connected = {_pair_key(t["subject"], t["object"]) for t in valid_triples}
    accepted, dropped_dup, dropped_budget = [], 0, 0
    per_method = Counter()
    for method in INFERENCE_PRIORITY:
        for t in candidates[method]:
            if t["subject"] == t["object"]:
                continue
            key = _pair_key(t["subject"], t["object"])
            if key in connected:
                dropped_dup += 1
                continue
            if budget is not None and len(accepted) >= budget:
                dropped_budget += 1
                continue
            connected.add(key)
            accepted.append(t)
            per_method[method] += 1

    breakdown = ", ".join(f"{m}: {per_method[m]}" for m in INFERENCE_PRIORITY if candidates[m])
    print(f"Accepted {len(accepted)} inferred relationships ({breakdown or 'none'}); "
          f"dropped {dropped_dup} already-connected pairs"
          + (f", {dropped_budget} over the budget of {budget} ({ratio:g} x {extracted_count} extracted)"
             if dropped_budget else ""), flush=True)

    result = _deduplicate_triples(valid_triples + accepted)
    for t in result:
        t["predicate"] = limit_predicate_length(t["predicate"])
    result = [t for t in result if t["subject"] != t["object"]]
    normalize_predicates(result)
    return _deduplicate_triples(result)


def _identify_communities(graph):
    """Connected components of the (undirected view of the) graph, as a list of sets."""
    adjacency = defaultdict(set)
    for source, targets in graph.items():
        for target in targets:
            adjacency[source].add(target)
            adjacency[target].add(source)
    visited, communities = set(), []
    for node in sorted(adjacency):  # sorted for deterministic output
        if node in visited:
            continue
        component, queue = set(), deque([node])
        visited.add(node)
        while queue:
            current = queue.popleft()
            component.add(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        communities.append(component)
    return communities


def _transitive_group(predicate):
    normalized = _norm_pred(predicate)
    for canonical, members in TRANSITIVE_PREDICATE_GROUPS.items():
        if normalized in members:
            return canonical
    return None


def _apply_transitive_inference(triples, graph, degree, inf_cfg):
    """A -p-> B -p-> C  ⇒  A -p-> C (via B), for transitive predicate families only.

    Skips hub intermediates (degree above ``transitive_max_hub_degree``) because
    they connect everything to everything, and caps new edges per subject.
    """
    allowed = set(inf_cfg.get("transitive_predicate_groups", list(TRANSITIVE_PREDICATE_GROUPS)))
    max_hub = int(inf_cfg.get("transitive_max_hub_degree", 10))
    max_per_subject = int(inf_cfg.get("transitive_max_per_subject", 5))

    predicates = defaultdict(set)
    for t in triples:
        predicates[(t["subject"], t["object"])].add(t["predicate"])
    connected = {_pair_key(t["subject"], t["object"]) for t in triples}

    new_triples, per_subject, skipped_hubs = [], Counter(), 0
    for subj in sorted(graph):
        for mid in sorted(graph[subj]):
            groups1 = {_transitive_group(p) for p in predicates[(subj, mid)]} & allowed
            if not groups1:
                continue
            if degree.get(mid, 0) > max_hub:
                skipped_hubs += 1
                continue
            for obj in sorted(graph.get(mid, ())):
                if obj == subj or _pair_key(subj, obj) in connected:
                    continue
                groups2 = {_transitive_group(p) for p in predicates[(mid, obj)]}
                shared = groups1 & groups2
                if not shared or per_subject[subj] >= max_per_subject:
                    continue
                canonical = sorted(shared)[0]
                new_triples.append(_make_inferred(subj, canonical, obj, "transitive", via=mid))
                connected.add(_pair_key(subj, obj))
                per_subject[subj] += 1
    print(f"Transitive inference proposed {len(new_triples)} relationships"
          + (f" (skipped {skipped_hubs} paths through hub nodes)" if skipped_hubs else ""), flush=True)
    return new_triples


def _deduplicate_triples(triples):
    """Remove exact duplicate triples, preferring extracted over inferred."""
    unique = {}
    for triple in triples:
        key = (triple["subject"], _norm_pred(triple["predicate"]), triple["object"])
        if key not in unique or not triple.get("inferred", False):
            unique[key] = triple
    return list(unique.values())


def _llm_infer(config, system_prompt, user_prompt, method, context):
    """Shared LLM call + parsing for the two LLM inference methods."""
    try:
        response = LLMClient.from_config(config).complete(user_prompt, system_prompt)
        parsed = extract_json_from_text(response, expect="array")
    except Exception as e:  # optional phase: never abort the run
        print(f"Error in LLM-based relationship inference ({context}): {e}", flush=True)
        return []
    if not parsed:
        print(f"Could not extract valid inferred relationships from LLM response ({context})", flush=True)
        return []
    results = []
    for t in parsed:
        if isinstance(t, dict) and all(k in t for k in REQUIRED_KEYS) and t["subject"] != t["object"]:
            results.append(_make_inferred(str(t["subject"]), str(t["predicate"]), str(t["object"]), method))
    return results


def _format_triples(triples, limit):
    return "\n".join(f"{t['subject']} {t['predicate']} {t['object']}" for t in triples[:limit])


def _infer_bridges_with_llm(triples, communities, degree, config):
    """Ask the LLM to connect every isolated component to the largest one.

    Small components are batched several per call. Representatives are the
    highest-degree entities of each component.
    """
    if len(communities) <= 1:
        print("Only one connected component found, skipping LLM bridging", flush=True)
        return []
    inf_cfg = config.get("inference", {})
    max_components = int(inf_cfg.get("max_bridge_components", 20))
    per_call = max(1, int(inf_cfg.get("bridge_groups_per_call", 5)))

    ordered = sorted(communities, key=len, reverse=True)
    main, others = ordered[0], ordered[1:1 + max_components]
    main_reps = sorted(main, key=lambda n: (-degree.get(n, 0), n))[:20]
    main_text = ", ".join(main_reps)
    system_prompt = system_prompt_for("bridge_inference_system", config)

    new_triples = []
    for batch_start in range(0, len(others), per_call):
        batch = others[batch_start:batch_start + per_call]
        groups = []
        for i, comp in enumerate(batch, start=batch_start + 1):
            reps = sorted(comp, key=lambda n: (-degree.get(n, 0), n))[:5]
            comp_triples = [t for t in triples if t["subject"] in comp and t["object"] in comp]
            groups.append(f"Group {i}: {', '.join(reps)}\n  known: " + _format_triples(comp_triples, 6).replace("\n", "; "))
        user_prompt = prompt_factory.get_prompt("bridge_inference_user", main_text, "\n".join(groups))
        found = _llm_infer(config, system_prompt, user_prompt, "llm_bridge", f"bridging groups {batch_start + 1}-{batch_start + len(batch)}")
        new_triples.extend(found)
    print(f"LLM bridging proposed {len(new_triples)} relationships for {len(others)} isolated components", flush=True)
    return new_triples


def _infer_hub_relationships_with_llm(triples, degree, config):
    """Ask the LLM for well-known relationships between the most central entities."""
    inf_cfg = config.get("inference", {})
    top_n = int(inf_cfg.get("hub_entities", 25))
    max_new = int(inf_cfg.get("hub_max_new", 25))
    hubs = [n for n, _ in sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]
    if len(hubs) < 3:
        return []
    hub_set = set(hubs)
    existing = [t for t in triples if t["subject"] in hub_set and t["object"] in hub_set]
    user_prompt = prompt_factory.get_prompt(
        "hub_inference_user", "\n".join(hubs), _format_triples(existing, 60) or "(none)", max_new)
    found = _llm_infer(config, system_prompt_for("hub_inference_system", config), user_prompt, "llm_hub", "hub enrichment")
    found = [t for t in found if t["subject"] in hub_set and t["object"] in hub_set][:max_new]
    print(f"LLM hub enrichment proposed {len(found)} relationships among the {len(hubs)} most central entities", flush=True)
    return found


def _infer_taxonomy(entities, triples):
    """Deterministic ``<modifier> <head>`` → ``is a`` ``<head>`` links, plus containment.

    ``quantum computing`` is a ``computing``; ``internet of things`` involves ``internet``.
    Only fires when the head/contained term already exists as an entity, so it never
    invents nodes. Nearly always true, hence enabled by default.
    """
    by_lower = {e.lower(): e for e in entities}
    connected = {_pair_key(t["subject"], t["object"]) for t in triples}
    new_triples = []
    for entity in sorted(entities):
        words = entity.lower().split()
        if len(words) < 2:
            continue
        linked = None
        # 1. Longest suffix that is itself an entity and is a true head noun.
        for k in range(1, len(words)):
            head = " ".join(words[k:])
            if head in by_lower and by_lower[head] != entity and words[k - 1] not in _PHRASE_BREAKERS:
                linked = by_lower[head]
                if _pair_key(entity, linked) not in connected:
                    new_triples.append(_make_inferred(entity, "is a", linked, "taxonomy"))
                    connected.add(_pair_key(entity, linked))
                break
        # 2. Another entity appears as a whole phrase inside this one (not as its head).
        padded = f" {' '.join(words)} "
        for other_lower, other in by_lower.items():
            if other == entity or other == linked or len(other_lower) < 5:
                continue
            if f" {other_lower} " in padded and _pair_key(entity, other) not in connected:
                new_triples.append(_make_inferred(entity, "involves", other, "taxonomy"))
                connected.add(_pair_key(entity, other))
    print(f"Taxonomy rule proposed {len(new_triples)} relationships", flush=True)
    return new_triples


def _infer_within_community_relationships(triples, communities, config):
    """Ask the LLM about lexically related but unconnected pairs inside large components."""
    new_triples = []
    connected = {_pair_key(t["subject"], t["object"]) for t in triples}
    system_prompt = system_prompt_for("within_community_system", config)

    for community in sorted(communities, key=len, reverse=True)[:3]:
        if len(community) < 5:
            continue
        members = sorted(community)
        pairs = []
        for i, a in enumerate(members):
            a_words = set(a.lower().split())
            for b in members[i + 1:]:
                if _pair_key(a, b) in connected:
                    continue
                b_words = set(b.lower().split())
                if (a_words & b_words) or a.lower() in b.lower() or b.lower() in a.lower():
                    pairs.append((a, b))
        pairs = pairs[:10]
        if not pairs:
            continue

        of_interest = {n for pair in pairs for n in pair}
        context = [t for t in triples if t["subject"] in of_interest or t["object"] in of_interest][:20]
        triples_text = "\n".join(f"{t['subject']} {t['predicate']} {t['object']}" for t in context)
        pairs_text = "\n".join(f"{a} and {b}" for a, b in pairs)
        user_prompt = prompt_factory.get_prompt("within_community_user", pairs_text, triples_text)
        found = _llm_infer(config, system_prompt, user_prompt, "llm_within", "within community")
        new_triples.extend(found)
    print(f"Within-community LLM inference proposed {len(new_triples)} relationships", flush=True)
    return new_triples


def _infer_relationships_by_lexical_similarity(entities, triples, min_word_length=5):
    """Connect entities that share a significant content word or contain one another.

    Opt-in: produces generic ``related to`` / ``is type of`` edges, which are weak
    evidence. Stop-words and very common domain words are ignored.
    """
    connected = {_pair_key(t["subject"], t["object"]) for t in triples}
    new_triples = []
    entities_list = sorted(entities)
    for i, e1 in enumerate(entities_list):
        e1_lower = e1.lower()
        e1_words = {w for w in e1_lower.split() if len(w) >= min_word_length and w not in _LEXICAL_STOPWORDS}
        for e2 in entities_list[i + 1:]:
            if _pair_key(e1, e2) in connected:
                continue
            e2_lower = e2.lower()
            e2_words = {w for w in e2_lower.split() if len(w) >= min_word_length and w not in _LEXICAL_STOPWORDS}
            shared = e1_words & e2_words
            if shared:
                main = max(shared, key=len)
                if e1_lower.startswith(main) and not e2_lower.startswith(main):
                    new_triples.append(_make_inferred(e2, "relates to", e1, "lexical"))
                elif e2_lower.startswith(main) and not e1_lower.startswith(main):
                    new_triples.append(_make_inferred(e1, "relates to", e2, "lexical"))
                else:
                    new_triples.append(_make_inferred(e1, "related to", e2, "lexical"))
            elif len(e1_lower) >= min_word_length and f" {e1_lower} " in f" {e2_lower} ":
                new_triples.append(_make_inferred(e2, "is type of", e1, "lexical"))
            elif len(e2_lower) >= min_word_length and f" {e2_lower} " in f" {e1_lower} ":
                new_triples.append(_make_inferred(e1, "is type of", e2, "lexical"))
            else:
                continue
            connected.add(_pair_key(e1, e2))
    print(f"Lexical similarity proposed {len(new_triples)} relationships", flush=True)
    return new_triples
