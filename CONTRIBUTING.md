# Contributing

Keep the core local, provider-independent, and easy to share. Existing triples JSON and offline HTML are public interfaces; preserve them when adding features.

## Set up and validate

```bash
uv sync --extra dev --extra all
uv run ruff check src tests
uv run pytest -q
```

Tests use fake model responses and temporary directories. They must not require credentials, spend API tokens, or modify the committed sample corpus. Run browser checks with a temporary graphs directory and an explicit test model endpoint.

For UI changes, exercise the library, extraction preview, staged update/apply, source evidence, reject/restore, saved views, chat citations, and offline HTML export. Check a narrow viewport and inspect browser errors. Rebuild the sample gallery after template changes:

```bash
uv run python scripts/build_docs.py
```

## Where to work

| Area | Module |
|---|---|
| Extraction, source reading, CLI, chunk progress | `src/knowledge_graph/main.py` |
| Workspace schema, exact passages, evidence merging, corrections | `src/knowledge_graph/workspace.py` |
| Profiles and prompts | `src/knowledge_graph/profiles.py`, `prompts/` |
| HTTP client, retries, reply cache | `src/knowledge_graph/llm.py` |
| Entity standardization and inference | `src/knowledge_graph/entity_standardization.py` |
| Retrieval, overview context, citations, chat history | `src/knowledge_graph/query.py` |
| Local API, jobs, revision checks | `src/knowledge_graph/server.py` |
| Graph metrics and rendering | `src/knowledge_graph/visualization.py` |
| Explorer and workspace UI | `src/knowledge_graph/templates/` |
| Export contracts | `src/knowledge_graph/exports.py` |

## Behavioral contracts

- Cite exact supplied passage IDs. Never guess supporting text from entity overlap.
- Keep negation, time, attribution, and full predicates through every export.
- Deduplicate claims without losing evidence from other documents.
- Apply corrections as reversible overlays; keep original claims.
- Skip unchanged documents. A stale browser preview must not overwrite a newer revision.
- Surface skipped chunks as partial results; malformed model output is not a successful empty result.
- Keep extracted and inferred claims distinguishable in retrieval and presentation.
- Test Unicode names, aliases, hostile HTML text, legacy imports, empty graphs, and zero-history chat.
- Keep HTML usable offline, including the story download from shared-library pages.

Do not edit the vendored vis-network assets casually. Preserve their license and follow the bundled vendor README when updating them.
