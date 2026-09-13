# Product and code review — September 2026

> Review snapshot before implementation. The current branch implements the core evidence, collection, discovery, story, retrieval, profile, and progress workflows described here; see [CHANGELOG.md](CHANGELOG.md) and [the workspace guide](docs/WORKSPACES.md). Source entailment scoring, embedding retrieval, generated/cited narrative suggestions, automatic merge suggestions, and collaboration remain future work. References to existing defects below describe the reviewed version.

The strongest next step is an evidence-backed knowledge workspace: a graph people can inspect, correct, extend, and return to. Preserve the project's lightweight Python core, choice of model provider, and portable offline HTML output. Those are valuable constraints for the next release.

This review covers all three existing project Markdown files (`README.md`, `IMPROVEMENT_PLAN.md`, and the vendored-library README), the extraction/standardization/inference pipeline, chat, server, exports, renderer, templates, and representative tests. I also opened the published Marie Curie demo and exercised search and the relationship panel. Recommendations below are product judgments based on those observations; they are not claims about measured user demand.

## Recommended priorities

| Order | Investment | What the user gains | Initial scope |
|---|---|---|---|
| 1 | Evidence and corrections | Understand why a claim exists and fix it | Exact source references, multiple supporting passages, review status |
| 2 | Incremental document collections | A graph that gets more useful over time | Add/update/remove documents, stable identities, change preview |
| 3 | Guided exploration and saved stories | An immediate starting point and a useful shareable result | Suggested paths, focused views, bookmarks, portable tours |
| 4 | Better graph questions | Answers to paraphrases and collection-wide questions | Unicode/alias retrieval, source passages, question routing |
| 5 | Extraction profiles | Useful graphs for a specific task | Configurable types/predicates, event qualifiers, negation |
| 6 | First-run and long-run experience | Easier setup and visible, recoverable progress | Connection check, trial extraction, partial-result status, retry |

### 1. Make every claim inspectable and correctable

**The experience:** select a relationship and see the exact passage, document, and page/section; inspect additional supporting passages; reject an inference; undo an incorrect entity merge. Corrections survive regeneration.

There is already a source field, but it is currently a heuristic attachment. [find_source_sentence](src/knowledge_graph/text_utils.py) chooses a sentence containing both entity names, or falls back to one containing either name. It never checks the predicate. [normalize_triple](src/knowledge_graph/main.py) uses this result as the source.

This is visible in the [published Marie Curie graph](https://robert-mcdermott.github.io/ai-knowledge-graph/samples/marie-curie.html#node=marie%20curie): the relationship to Flying University displays her birth sentence as evidence. A local reproduction also attached “Alice visited Acme.” to the claim “Alice founded Acme.”

Implement sentence or passage IDs before extraction, ask the model to return supporting IDs, and retain original text with offsets. Preserve PDF page boundaries during input reading. Validate that cited passages exist; separately assess whether they support the claim. An exact quotation alone is not proof of entailment. Represent unverified evidence explicitly instead of silently choosing an unrelated sentence.

Store multiple evidence records per claim. [_deduplicate_triples](src/knowledge_graph/entity_standardization.py) currently keeps a single dictionary per SPO key, so the same fact in two documents loses one document's provenance. Show “supported by 3 documents,” while avoiding treating copied passages as independent corroboration.

Add an optional strict extraction profile: only source-supported claims enter the extracted layer; general knowledge stays in a separate inference layer. The [extraction prompt](src/knowledge_graph/prompts/main_prompts.py) currently invites some information “if known,” which weakens that boundary.

For corrections, start with reject/restore relationship and approve/reject proposed alias merge. Store overrides separately from generated output so reruns do not erase human work. Add a “keep these entities separate” rule before attempting a full graph editor.

**Acceptance example:** a claim supported in two documents retains both passages after deduplication; rejecting it persists after regeneration; an unrelated sentence is never presented as verified support.

### 2. Turn multi-file generation into a persistent collection

**The experience:** “Add this paper to my research graph” shows new entities, new claims, additional evidence, and potentially conflicting claims before applying the update.

[process_documents](src/knowledge_graph/main.py) already combines documents within one run. The missing capability is updating an existing collection with explicit lifecycle semantics.

Start with a versioned graph bundle containing documents, entities, claims, evidence, aliases, and run metadata. Keep a reader for existing flat triples JSON and continue offering portable HTML exports. A local file bundle is sufficient for a first implementation; a mandatory graph database would add deployment work before it adds user value.

Use stable document and entity IDs distinct from display names. [read_documents](src/knowledge_graph/main.py) currently uses basenames, and single-document extraction omits the document field. IDs and document fingerprints would also distinguish same-named files and enable changed-document extraction.

Important update behavior:

- Re-adding an unchanged document is a no-op.
- A revised document replaces its previous evidence rather than accumulating stale claims.
- Removing one document keeps claims still supported by another.
- Entity merge decisions are preserved and can be reversed.
- Inferred relationships are revalidated or marked stale when their supporting graph changes.

**Acceptance example:** add document A, then B, then revise A; show a meaningful diff and preserve evidence from B and existing user corrections.

### 3. Give people an immediate discovery and a story to share

**The experience:** open a graph and choose “Explore the discoveries,” “Trace this connection,” or “Show the main themes.” Save an interesting path with a short note and share it as an offline guided tour.

The live demo opens successfully, but the full Marie Curie graph is dense at its initial zoom. Searching helps, although selecting the central person produces a long relationship list. The [explorer template](src/knowledge_graph/templates/graph.html.j2) exposes many controls before explaining what someone might discover.

Build on the existing communities and path finder:

- Show 3–5 curated starting points on sample graphs, then support generated, cited suggestions for new graphs.
- Add “focus on this neighborhood” with progressive expansion; dimming alone still leaves the whole graph occupying the canvas.
- Group a person's relationships into understandable facets such as education, collaborators, discoveries, and dates.
- Offer a community overview that expands into its entities and relationships.
- Save selection, filters, path, camera position, and optional notes together. Current URL state preserves only the selected node.
- Export a small sequence of annotated views inside the existing self-contained HTML format.

Prefer deterministic path discovery first, with an optional model explanation grounded in the path's evidence. A graph path demonstrates a connection; it does not automatically establish a causal explanation.

**Acceptance example:** a first-time visitor can discover and share a three-step, sourced connection without configuring physics or learning the full toolbar.

### 4. Make graph chat work across language, phrasing, and scope

**The experience:** “Which discoveries connected these researchers?” and “What are the main themes across my documents?” both retrieve appropriate context, with clickable evidence attached to specific answer claims.

The existing [retriever](src/knowledge_graph/query.py) is a useful small-graph baseline. It uses entity-name/content-word matching, seed-to-seed paths, neighborhoods, and an LLM fallback. However:

- The deterministic matcher strips non-ASCII characters. A direct test with `北京位于哪里？` returned no entities for a graph containing `北京`.
- The LLM entity picker only sees the top 300 entities. It is not a complete fallback for larger collections.
- An unmatched follow-up reuses earlier seeds before trying the LLM, which can keep an unrelated new question stuck in the old topic.
- Retrieval caps facts by count rather than a token budget.
- Citations are parsed from a final “Facts used” line; citation membership is checked, but claim support is not validated.

Start with Unicode-aware matching, preserved aliases, indexed source-passage search, and explicit follow-up/topic-change handling. Combine graph paths with source passages; add optional embeddings only after evaluating the simpler baseline. Route broad questions through community summaries instead of arbitrarily choosing high-degree entities.

This direction is consistent with Microsoft's documented [GraphRAG local search](https://github.com/microsoft/graphrag/blob/main/docs/query/local_search.md), which combines graph data with source text, and its distinct [local/global query modes](https://github.com/microsoft/graphrag/blob/main/docs/query/overview.md). Those are useful design references; adopting the entire framework is unnecessary.

Use answer records with explicit claim-to-evidence references and clearly mark unsupported output. Keep an extracted-only question mode. Include the source document in the model context and returned citations.

**Acceptance example:** an entity outside the top 300 is found through an alias or non-English query; a broad question retrieves across communities; every displayed citation resolves to preserved evidence.

### 5. Add task-specific extraction profiles and qualified claims

**The experience:** select a profile such as general reading, research literature, or organizations and products, and get a graph shaped around that purpose.

Today the [prompt](src/knowledge_graph/prompts/main_prompts.py) uses a fixed type set and aggressively short predicates. The [normalizer](src/knowledge_graph/entity_standardization.py) also truncates predicates in stored data. Display constraints should not remove meaning from the underlying claim.

Let profiles define types, relationship vocabulary, examples, and inference policy. Keep the current general-purpose defaults. Preserve the full predicate with a separate short display label. Introduce optional polarity, time, and attribution before attempting automatic contradiction detection.

For example, “Acme acquired Beta in 2020” is an event with participants and a date. Separate unrelated edges from Acme to Beta and Acme to 2020 cannot reliably express which event the date qualifies. Likewise, “did not acquire” must remain distinct from “acquired.”

Entity normalization also needs context: a direct test showed `c++` and `c#` collapse together through punctuation removal, causing their relationship to be removed as a self-loop. Even within prose, names and acronyms need identity separate from normalized search keys.

Start potential-conflict detection with a profile that defines meaningful constraints. Different job titles or locations are not necessarily contradictions when dates differ. Show candidates with evidence for human review rather than declaring a fact false.

### 6. Make setup and generation reassuring

Add a first-run connection check and tiny extraction preview before processing a large document. Show the selected endpoint/model, estimated chunk count, actual token usage when returned, cache hits, and elapsed time. Price estimates should require a known or user-entered rate.

The [LLM client](src/knowledge_graph/llm.py) already receives usage data, but normal completion returns only text. A shared run context could collect metrics and structured progress across phases.

Distinguish complete, partial, and failed runs in saved metadata and the UI. The [web job](src/knowledge_graph/server.py) always enables continuation on chunk errors and may report `done` for an incomplete graph. Add retry-failed-chunks and cancellation. The response cache already provides part of the groundwork for resuming extraction.

For a shorter first session, retain the no-model sample workflow and add a visible “Try a sample” action. Put new personal graphs in a separate default output directory: the current quick start points generation at the sample-fixture directory, which can cause fixture tests to fail after normal use.

## Correctness work to pair with these features

These are observed implementation gaps, separate from the proposed product features:

| Finding | Evidence | Recommended repair |
|---|---|---|
| Malformed extraction can produce an incomplete graph even in strict mode | Reproduced with one non-JSON chunk and one valid chunk; [main.py](src/knowledge_graph/main.py) only treats raised LLM errors as failed chunks | Distinguish valid empty extraction, parse failure, and partially salvaged output; persist completeness and make strict mode enforce it |
| Repeated claims discard source documents | Reproduced with identical SPO records from `a.txt` and `b.txt`; only `b.txt` survived deduplication | Aggregate evidence independently of claim identity |
| Cache entries collide across endpoints using the same model name | Reproduced by comparing cache paths for different base URLs | Include endpoint and relevant nonsecret request identity in the cache key; use atomic cache writes |
| Browser exports lose provenance fields | `visibleTriples()` omits `document`; browser CSV only includes six fields | Define one export contract and verify JSON/CSV round trips |
| Imported JSON can lack a predicate | Reproduced acceptance by [load_triples_from_json](src/knowledge_graph/main.py) | Validate nonempty string SPO values, field types, and supported schema versions at the boundary |
| Chat history can become stale or shared across tabs | [GraphStore](src/knowledge_graph/server.py) caches one chat per graph without file-change invalidation | Key sessions separately and invalidate their graph indexes on document updates |
| A zero history limit includes all history | `history[-0:]` in [GraphChat.ask](src/knowledge_graph/query.py) is the entire list | Handle zero explicitly and bound retained history |
| Job errors are interpolated into HTML | [job.html.j2](src/knowledge_graph/templates/job.html.j2) inserts `s.error` via `innerHTML` | Render error messages with `textContent`; include an HTML-containing error regression case |

For collection scale, profile before replacing the renderer. [Visualization metrics](src/knowledge_graph/visualization.py) currently calculate exact betweenness for every graph, and within-community candidate discovery scans pairs. Cache metrics by graph revision, use indexed candidate discovery, and introduce sampled betweenness above a measured threshold. [NetworkX documents the `k` sampling parameter](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html). Browser physics freezing alone does not reduce Python preprocessing cost. No large-graph benchmark was run in this review.

## Documentation refresh

The README is substantially more current than the improvement plan. Preserve its working quick start and move the long configuration and reference sections into focused docs as they grow.

- Mark the old plan's opening audit as historical. It still describes no tests, no CI, PyVis-era bugs, and old dependencies as if current, even though later completion notes describe their replacement.
- Reconcile completed checkboxes with actual behavior. The plan claims confidence fields, token-based chunk sizing, predicate synonym merging, optional Leiden, browser GraphML export, predicate filters, persistent theme preferences, and richer URL state; those are absent or only partially implemented in the reviewed code.
- The completed docs/CI items mention CHANGELOG, CONTRIBUTING, issue templates, and a nightly docs build. These are absent from this checkout. The existing workflow runs lint, tests, and a renderer smoke test on pushes/PRs.
- Resolve release-version ambiguity: the plan describes completed 0.8 waves, while package metadata still says 0.7.0.
- Update the hero screenshot and add a short discover → inspect evidence → ask → share walkthrough. The live examples already provide a strong no-install introduction.
- Document the limitations of source attachment and answer grounding until the proposed evidence work lands. Current wording is stronger than the implementation guarantees.
- Pin an explicit vendored-library version and source/checksum in its README instead of referring readers only to the minified file header.

## Delivery sequence and validation

1. **Trust foundation:** schema validation, correct source references, evidence aggregation, explicit partial runs, and export parity. Introduce a versioned bundle with backward-compatible import.
2. **Useful returning-user workflow:** incremental document updates, persistent corrections, document filters, and a change preview.
3. **Visible release feature:** guided exploration, saved paths/tours, and a refreshed sample walkthrough. A small curated tour can ship earlier alongside the foundation work.
4. **Deeper questions:** hybrid graph/passage retrieval, multilingual cases, global summaries, and profile-based extraction.

Extend the tests with a small, manually annotated corpus covering supporting passages, pronouns, aliases, punctuation-sensitive names, negative claims, dates, and overlapping/revised documents. Evaluate extraction correctness and citation support separately from graph connectivity. Add browser checks for the actual search → path → filter → export flow; existing renderer tests mostly inspect generated data and HTML text. Keep live-model evaluations opt-in so ordinary tests remain offline and inexpensive.

Review validation: `ruff check .` passed. `pytest -q` produced **171 passed, 1 failed**. The failure was `test_every_sample_text_has_a_graph`, caused by the existing untracked `data/samples/graph.json` having no corresponding sample text. Those user files were not changed or removed. No live model requests were made. The declared development dependencies were installed from the local uv cache to run the suite. No application code or existing documentation was edited by this review.
