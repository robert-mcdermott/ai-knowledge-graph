# Changelog

## Unreleased — evidence and knowledge workspaces

### Added

- Versioned workspace sidecars with retained source documents, qualified claims, review decisions, views, and recent extraction run summaries.
- Incremental collections: preview additions/revisions/removals, extract changed documents, inspect proposed claim changes, and apply with revision conflict protection.
- Exact source passage IDs, multiple evidence records, PDF page offsets, and strict reference checking.
- Reversible claim rejection and entity merges in the local app.
- Community discovery cards, neighborhood focus/expansion, source and relationship filters, saved annotated views, URL view state, and offline story exports.
- General, research, and organizations extraction profiles, with optional custom types and preferred predicates.
- Unicode/alias and source-term retrieval, community-balanced overview context, inline chat citations, extracted-only retrieval, bounded history, and session isolation.
- Model connection checks, short extraction trials, progress/usage summaries, partial status, cancellation, and retries.

### Changed and fixed

- Preserve qualifiers in inference context, skip qualified premises in transitive rules, and label negative/dated relationships explicitly.
- Preserve full predicates and relationship qualifiers instead of shortening stored facts.
- Replace guessed source sentences with explicit source references; old citations are visibly unverified.
- Union evidence during deduplication, and keep differently qualified claims distinct.
- Preserve punctuation distinctions such as C++ versus C# during deterministic name normalization.
- Include endpoint identity in reply-cache keys and invalidate malformed extraction replies.
- Validate triples and workspace imports; render empty graphs and clear stale community metadata.
- Sample betweenness on large graphs and bound within-community inference candidate searches.
- Default the local library to `out/`; keep internal sidecars out of the library and invalidate cached pages/chat when workspace metadata changes.
- Keep source evidence and qualifiers in CSV, GraphML, and Cypher exports.

### Compatibility

Existing flat triples files continue to work. When present, `.workspace.json` is authoritative. Collection creation/update preserves original entity names and uses manual review merges; automatic standardization remains available for ordinary single-run CLI generation. Jobs are local and in memory, and the server remains a personal, unauthenticated localhost application.

Source linkage and citation-number validation do not establish semantic entailment. Embedding retrieval, automatic evidence verification, generated narrative suggestions, and multi-user collaboration remain follow-on work.
