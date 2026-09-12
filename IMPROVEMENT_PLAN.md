# AI Knowledge Graph – Improvement Plan (v0.7 "nextgen")

This plan comes from a full review of the code, a real end-to-end run against Ollama
(`config-working.toml`, `deepseek-v4.1-flash:cloud`, `data/industrial-revolution.txt`),
and a hands-on audit of the published demo (`docs/index.html`) in a browser, in light
and dark mode, on desktop and mobile widths.

**Project goal (unchanged):** take unstructured text, discover the direct *and inferred*
relationships between the concepts, objects and entities in it, and present them as a
beautiful, explorable knowledge graph.

Legend: `P0` fix first · `P1` important · `P2` nice to have · effort `S` (<1h) `M` (½–1 day) `L` (multi-day)

---

## 0. What the review found (summary)

**The pipeline can silently produce a near-empty graph.** In the test run, 2 of 3 chunks
came back with *empty* content: the reasoning model spent its whole 8192-token budget on
hidden reasoning (`finish_reason: "length"`, `content: ""`, ~40K chars in `reasoning`).
The app printed a one-line warning and continued, building a "knowledge graph" of the
final 66-word chunk (12 nodes). Nothing tells the user the result is garbage.

**Inference is the main source of visual noise, not insight.** In the README example 355
of 564 edges are inferred. Transitive edges get predicates like `"essential for via"`
and `"influenced via human"` because the 3-word truncation chops off the `via <entity>`
part. Lexical-similarity edges (`related to`, `is type of`) fire for *any* pair sharing a
4-letter word. Together they turn a readable graph into a hairball and drown the real,
extracted relationships. The `inference.apply_transitive` config key is not read anywhere:
transitive inference always runs.

*Confirmed on a full run (`testnew.json`, `max_tokens = 32768`, all 3 chunks succeeded):*

| | count |
|---|---|
| extracted triples | 414 |
| inferred triples | 1058 |
| of which transitive ("X via Y") | 962 |
| of which lexical similarity | 78 |
| of which LLM-inferred | 18 |
| node pairs with 2–6 parallel edges | 144 |

72 % of all edges are rule-generated transitive edges; the top predicates are
`develops via computing` (78), `pioneered via computing` (77), `developed via technology`
(64). Hiding inferred edges with the page's own filter turns the hairball into a readable
graph with clear clusters. **Fixing inference is worth more than any rendering change.**

**The visualization has real bugs.** The browser console shows
`ReferenceError: network is not defined` on load: the template's `<script>` runs before
PyVis defines `network`, so the stabilization handlers and the resize listener never
register. vis.js rejects two options we pass (`nodes.tooltipDelay`, top-level
`background`). Only 8 community colors exist, so a 9-community graph shows as 7 in the
Stats panel (communities are counted by color). Clicking a node does nothing. Hover shows
only `"name - Connections: N"`. Edge labels are drawn for all ~500 edges at once. The
output file depends on a Bootstrap 5 *beta* CDN link, so it is not actually offline-safe.
The mobile layout wraps the toolbar over the graph.

**Engineering hygiene is behind a 3k-star project.** No tests, no CI, no type hints,
`sys.path` hacks instead of a proper package, `requirements.txt` carries IPython/pandas,
`pyvis-network` is an unrelated PyPI package, `python-louvain` duplicates
`networkx.community.louvain_communities`, prints instead of logging (and no flush, so
redirected output is blank until the end), API key lives in a committed `config.toml`,
`chunk_text` loops forever when `overlap >= chunk_size`, and the README's project layout
is stale (`prompts.py` is now a package).

---

## Phase 1 – Stop the bleeding (P0, mostly S)

> **Wave 1 (done):** LLM client + JSON extraction + config validation, with a pytest suite
> (`tests/`).
> **Wave 2 (done):** inference defaults, constraints, budget, method tags and standardization guard.
> **Wave 3 (done):** connectivity. Deterministic `taxonomy` method ("quantum computing" is a
> "computing", on by default), plural singularization in standardization, LLM *bridging* pass that
> links every isolated component to the main graph, LLM *hub enrichment* pass for general-knowledge
> edges between central entities, default temperature 0.2 and a tighter atomic-entity instruction.
> Measured on the wave-2 run: taxonomy alone took the graph from 22 components to 7 (228/286 nodes connected).
> **Wave 4 (done):** template fixes (script runs after `network` exists, no invalid vis options, no init
> hacks), 20-colour community palette with real `community`/`inferred`/`method`/`via` attributes, a
> *Hide Inferred* toggle and per-method stats, Bootstrap CDN removed (fully offline HTML), MultiDiGraph so
> parallel edges are no longer dropped, encoding fallback for input files, `--from-json` re-rendering.
> Phase 1 is complete except the `logging` migration (moved to Phase 4).
> **Wave 7 (done):** entity types (closed set, majority vote per node, shapes + type filter in the page),
> provenance (source sentence per extracted edge, shown in tooltips and the details panel),
> sentence-aware chunking, parallel chunk extraction (`llm.concurrency`), LLM-named communities.
> Live run: 143/143 extracted triples typed and sourced; 11 communities named; extraction 2x faster.
> **Wave 8 (done):** shortest-path finder in the details panel, sqrt node sizing, `visualization.show_inferred`,
> predicate tense normalization, on-disk LLM response cache (`llm.cache_dir`, `--no-cache`; a cached re-run
> takes ~3 s instead of ~20 s), deterministic representative ordering so prompts and cache keys are stable.
> **Wave 9 (done):** inputs and outputs. `.md/.rst/.pdf/.docx` inputs (pdf/docx via optional extras),
> multiple files or directories with per-triple `document` tags, `extraction.language`, and `--export`
> csv/graphml/cypher (Neo4j MERGE script with a label per entity type). README refreshed and Pages demo
> regenerated (only the screenshot is still old).
> **Wave 10 (done, 0.8):** optional `graph-chat` command (grounded, cited question answering over the
> JSON; `[query]` config); `generate-graph` unchanged. Taxonomy rule no longer applies to people/places/orgs.
> **Wave 11a (done, 0.8):** optional `graph-serve` (`[web]` extra): graph library, explorer pages, Ask panel
> backed by the same retrieval code; localhost, no accounts. Static output unchanged.
> **Wave 11b (done, 0.8):** ingest in the browser: paste/upload → background job with progress page → explorer.
> **Wave 12 (done, 0.8):** page polish: parallel edges collapsed with a "+N" badge (raw relationships kept for
> the details panel, paths, stats and exports), title-cased display names; both configurable.
> **Wave 13 (done, 0.8):** `logging` migration: pipeline modules log via `knowledge_graph.*` loggers, the CLIs
> configure a console handler (`--verbose`/`--debug`, `--quiet`), default output unchanged; graph-serve captures
> job progress through a log handler instead of a stdout tee.
> **Wave 6 (done):** PyVis removed; the page is rendered from `templates/graph.html.j2` with vis-network
> 9.1.9 vendored. Search, click-to-highlight with a relationships panel, edge labels on selection,
> communities panel with toggles + min-degree slider + inferred switch, stats, physics settings,
> PNG/JSON/CSV export, keyboard shortcuts, URL hash state, layout progress bar, auto physics freeze,
> light/dark themes, responsive layout. Still open in Phase 2: path finder, node-size rescale,
> title-case display names, inferred hidden by default.
> **Wave 5 (done):** `src/` layout installed as `knowledge_graph` (no `sys.path` hacks), version 0.7.0,
> deps pruned to networkx/pyvis/requests (IPython remains only because PyVis requires it), ruff config
> and a clean lint, GitHub Actions CI on 3.11-3.13 with a renderer smoke test.

Correctness fixes that change results today. Do these first and cut a 0.6.2 patch.

### LLM client (`src/knowledge_graph/llm.py`)
- [x] **Detect truncated / empty completions.** If `finish_reason == "length"` or content is
      empty, raise a clear error that names the chunk and suggests raising `max_tokens`
      (reasoning models need 16–32K) or using a non-reasoning model. Never silently skip a
      chunk; a `--continue-on-error` flag can opt into the old behaviour. `P0 S`
- [x] **Raise the default `max_tokens` to 32768** in `config.toml` and the README
      (confirmed: 32K makes every chunk succeed with `deepseek-v4.1-flash`; 8K fails). `P0 S`
- [x] **Pass through reasoning controls from config**: optional `reasoning_effort`, `think`,
      and a generic `extra_body` table so users can tune any OpenAI-compatible server
      (Ollama, vLLM, LiteLLM, OpenRouter) without code changes. `P0 S`
- [x] **Support `max_completion_tokens`** (GPT-5 and newer OpenAI models reject
      `max_tokens`; issue #15). Try `max_tokens`, fall back on the specific 400 error, or add
      a `token_param` config key. Also allow omitting `temperature` (some models require
      the default). `P0 S`
- [x] **Timeouts and retries.** `requests.post` has no timeout; add `timeout` (config,
      default 300 s) and exponential-backoff retries on 429/5xx/connection errors. `P0 S`
- [x] **Read the API key from the environment** (`api_key = "env:OPENAI_API_KEY"` or
      `${VAR}` syntax) and stop shipping a real-looking key in `config.toml`. Add
      `config.local.toml` / `config-*.toml` to `.gitignore`. `P0 S`
- [x] **Optional JSON mode**: send `response_format = {"type": "json_object"}` when
      `llm.json_mode = true` (supported by OpenAI, Ollama, vLLM, LiteLLM). Wrap the array in
      `{"triples": [...]}` in the prompt so JSON-object mode works. `P1 S`

### JSON extraction (`extract_json_from_text`)
- [x] **Strip `<think>…</think>` blocks before parsing** (currently returns `[1]` from
      inside a think block in the test). `P0 S`
- [x] **Return the right container type.** For a dict response with prose around it, the
      fallback finds the first `[` and returns the inner array (entity-resolution mapping is
      then silently discarded). Add an `expect="array"|"object"` argument and scan for the
      matching opener. `P0 S`
- [x] **Fix the "unquoted key" regex**: `(\w+)\s*:` also rewrites colons inside string values
      (`"time: 10am"` → parse failure). Use a JSON-repair approach (e.g. `json_repair`
      package) or only quote keys that follow `{` or `,`. `P1 S`

### Inference & standardization (`entity_standardization.py`)
- [x] **Honor `inference.apply_transitive`** (currently never read) and **default it to
      `false`**. In the confirmed run it produced 962 of 1472 edges. `P0 S`
- [x] **Constrain transitive inference when it is on**: only through low-degree
      intermediates (skip hubs like `technology`/`computing`, which fan out to everything),
      only 1 hop, cap at N per subject and at a fraction of extracted edges, and only for
      predicate classes where transitivity is meaningful (`is a`, `part of`, `located in`,
      `caused`/`led to`). `P0 M`
- [x] **Stop truncating `"<pred> via <entity>"` predicates into nonsense.** Keep the
      predicate as `pred1` and store the intermediate node in a `via` field; show
      "via X" in the tooltip instead of the label. `P0 S`
- [x] **Make lexical-similarity inference opt-in** (`inference.lexical = false` by default)
      and require a shared *content* word (not a stop-word) plus a length ≥ 5. `P0 S`
- [x] **Fix the misleading counters**: "Inferred N new relationships between communities"
      prints the cumulative total on every pair; "Added -22 inferred relationships" can go
      negative because dedup runs after the count. `P1 S`
- [x] **Over-merging guard in standardization**: the 4-char stem rule merges
      `steam engine factories` into `steam engine` (verified). Require the subset rule to be
      a *prefix* match or the LLM to confirm; log every merge at debug level so users can
      audit. `P1 M`
- [x] **Pick community representatives by degree, not `list(set)[:5]`** (arbitrary order
      today), and cap total inference LLM calls (5 communities → 10 pair calls). `P1 S`

### Text handling (`text_utils.py`, `main.py`)
- [x] **Guard `chunk_text` against `overlap >= chunk_size`** (infinite loop, verified) and
      validate config values at load time. `P0 S`
- [x] **Encoding fallback and file-type check** for input: try UTF-8, then UTF-8-sig,
      then `latin-1`, and give a clear message for binary files like PDF (issue #9). `P0 S`
- [x] **Unbuffered / logged output** (prints now flush; full `logging` migration pending). Replace `print` with `logging` (`--verbose`,
      `--quiet`), flush progress lines, and print a final summary table. `P1 S`

### Visualization template (`templates/graph_template.html`, `visualization.py`)
- [x] **Fix `network is not defined`**: run template code after PyVis's `drawGraph()` (inject
      the template script *after* the network is created, or guard with a ready hook).
      Remove the duplicated `stabilizationIterationsDone` handler. `P0 S`
- [x] **Remove invalid vis options** (`nodes.tooltipDelay`, `background`); drive the
      background via CSS only. `P0 S`
- [x] **Store `community` and `inferred` as real node/edge attributes** instead of deriving
      them from color / `dashes`; generate a palette with ≥ 20 distinct, contrast-checked
      colors (Tableau 20 / Okabe-Ito extended) and drop `#ffff33` (unreadable on white). `P0 S`
- [x] **Make the HTML truly self-contained**: no Bootstrap CDN (the beta-3 link), inline all
      CSS/JS, and offer `--cdn` to produce a small file instead. `P1 S`

---

## Phase 2 – A visualization people want to explore (P1, M–L)

This is the biggest visible win. Recommendation: **drop PyVis and own the HTML.** PyVis
adds a stale Bootstrap CDN link, an unused `neighbourhoodHighlight()` blob, and fights
every customization. Keep **vis-network 9.x** (already inlined, physics controls work,
users know it) but render it from our own Jinja2 template with the graph embedded as JSON.
Consider Sigma.js/Graphology (WebGL) later only if users hit >2–3k nodes.

### Layout & chrome
- [x] **Redesign the chrome**: slim top bar (title, search, theme, export) + collapsible
      right-hand *details panel* + a floating legend. No Bootstrap; a small hand-written CSS
      with design tokens for light/dark. Respect `prefers-color-scheme` and remember the
      choice in `localStorage`. `P1 M`
- [x] **Responsive**: toolbar collapses into a menu below ~800 px; panels become bottom
      sheets on mobile; `network.fit()` after layout. `P1 S`
- [x] **Loading state**: progress bar during stabilization (vis `stabilizationProgress`),
      then freeze physics automatically for graphs > 300 nodes (big perf win). `P1 S`

### Exploration features (the "explore the data" goal)
- [x] **Search with autocomplete** over node names; Enter focuses and selects. `P1 S`
- [x] **Click a node → highlight its neighborhood** (1–2 hops), dim everything else, and
      open the details panel listing every incoming/outgoing relationship as clickable rows
      (predicate, direction, extracted vs inferred, source chunk/sentence). `P1 M`
- [x] **Edge labels only on hover/selection** (or above a zoom threshold); hide node labels
      for low-degree nodes when zoomed out. This alone removes most of the clutter. `P1 S`
- [x] **Collapse parallel edges**: draw one edge per node pair (144 pairs had 2–6 edges in
      the confirmed run) with a count badge, and list all predicates in the tooltip /
      details panel. Use `smooth: curvedCW/CCW` only when two directions exist. `P1 S`
- [x] **Inferred edges visibility** is configurable (`visualization.show_inferred`, default true because inference is now conservative and traceable); the page has a one-click toggle and a count
      in the legend, so the first impression is the extracted graph. `P1 S`
- [x] **Community legend with toggles**: click a color to isolate/hide a community; show
      counts. `P1 S`
- [x] **Quick filters**: "Show inferred edges" toggle, minimum-degree slider, predicate
      multi-select, "hide leaf nodes". Replace the current three-dropdown filter form. `P1 M`
- [x] **Path finder**: pick two nodes and highlight the shortest path (reuse BFS in JS);
      option to exclude inferred edges. Directly serves "find relationships between
      concepts". `P1 M`
- [x] **Rich tooltips**: name, type, community, degree, top relationships, and the source
      sentence for edges. `P1 S`
- [x] **Export**: PNG of the current view, and JSON / CSV / GraphML downloads of the
      (filtered) graph from the page. `P2 S`
- [x] **Keyboard shortcuts** (`/` search, `Esc` clear, `F` fit, `L` labels, `P` physics) and
      a `?` help overlay. `P2 S`
- [x] **URL state** (`#node=steam%20engine&hide=inferred`) so views can be shared. `P2 S`

### Visual quality
- [x] **Node sizing**: use a log/sqrt scale of degree so hubs don't dwarf everything;
      current mix of degree/betweenness/eigenvector is fine but scale it 8–40 px. `P1 S`
- [x] **Edge styling**: extracted edges solid and slightly thicker; inferred edges thin,
      dashed, lower opacity; arrows scaled to node size; `smooth: continuous` for parallel
      edges only. `P1 S`
- [x] **Typography**: system font stack instead of Tahoma; label halo/stroke that matches
      the theme instead of the `!important` CSS hacks. `P1 S`
- [x] **Title-case display names** while keeping lowercase for matching (the prompt asks
      for lowercase; store `display_name` separately). `P2 S`

---

## Phase 3 – Better knowledge, not just more edges (P1, M)

- [x] **Entity types.** Extend the extraction prompt to return
      `subject_type` / `object_type` from a small closed set (person, organization, place,
      event, concept, technology, work, date). Use type for node shape/icon and community
      for color; filter by type in the UI. `P1 M`
- [x] **Provenance.** Keep the source sentence (or chunk id + sentence index) on every
      extracted triple and mark inferred triples with their method
      (`transitive`, `llm_community`, `llm_within`, `lexical`) and a confidence. Show it in
      the details panel. This is what makes users trust inferred edges. `P1 M`
- [x] **Sentence/paragraph-aware chunking** (split on sentence boundaries, size in tokens
      via a cheap estimator) so relationships aren't cut mid-sentence. `P1 S`
- [x] **Parallel chunk extraction** with a configurable worker count (`llm.concurrency`,
      default 4); order-preserving results. Cuts wall-clock time 3–5×. `P1 S`
- [x] **Response cache** keyed by hash(model, prompt) in `.kg-cache/`, so re-running
      visualization changes or tweaking inference doesn't re-pay for extraction.
      Promote `json_to_html.py` to `generate-graph --from-json graph.json`. `P1 S`
- [x] **Smarter inference budget**: cap inferred edges to a configurable fraction of extracted
      edges (default 50 %), prefer LLM-inferred over rule-inferred when over budget, and
      never infer between nodes already connected in either direction. `P1 S`
- [x] **LLM-named communities**: after Louvain, ask the LLM for a 2–4 word label per
      community from its top nodes; show labels in the legend and as optional cluster
      captions. Cheap (one call) and a strong "AI-powered" differentiator. `P1 S`
- [x] **Predicate normalization**: lower-case, lemmatize simple tense variants
      (`involve` / `involves`), and merge synonyms via an LLM pass on the predicate list;
      the run above produced both `involve` and `involves`. `P2 S`
- [x] **Use `networkx.community.louvain_communities`** (built in) and drop `python-louvain`;
      offer Leiden via optional `igraph`/`leidenalg` extra (issue #24). `P2 S`
- [x] **Language option** (`extraction.language = "auto"|"zh"|…`) instructing the model to
      keep entities in the source language (issue #7). `P2 S`
- [x] **More input formats**: `.md`, `.pdf` (pypdf), `.docx` (python-docx) as optional
      extras; accept multiple `--input` files / a directory and tag triples by document. `P2 M`

---

## Phase 4 – Engineering hygiene (P1, S–M)

- [x] **Proper package layout**: `src/knowledge_graph` installed as `knowledge_graph`
      (`package-dir = {"" = "src"}`), drop all `sys.path.insert` hacks, keep
      `generate-graph.py` as a thin shim. Bump to 0.7.0. `P1 M`
- [x] **Dependencies**: remove `pyvis-network` (unrelated package), `tomli` (use stdlib
      `tomllib`, Python ≥ 3.11), IPython/pandas/numpy from `requirements.txt`; add `jinja2`
      explicitly if PyVis is dropped. Regenerate `uv.lock`. `P1 S`
- [ ] **Typed data model**: `Triple`, `Node`, `Edge`, `Config` dataclasses (or pydantic)
      with validation of config values and helpful error messages. `P1 M`
- [x] **Tests (pytest)** with a fake LLM: chunking edge cases, JSON extraction corpus
      (think tags, prose, truncated arrays, dict vs list), standardization (no over-merge),
      inference (no self-loops, respects config, predicate integrity), HTML generation smoke
      test that asserts no `CDN` links and valid embedded JSON. `P1 M`
- [x] **CI**: GitHub Actions running ruff + pytest on 3.11/3.12/3.13; a nightly job that
      regenerates `docs/index.html` from the checked-in sample JSON. `P1 S`
- [x] **Docs refresh**: fix the project layout section, add config examples for OpenAI,
      Anthropic (via LiteLLM), Gemini (merge PR #16), OpenRouter, vLLM; a "reasoning
      models" note (`max_tokens ≥ 16K`); a troubleshooting section; screenshots of the new
      UI; CHANGELOG; CONTRIBUTING; issue templates. `P1 M`
- [ ] **Sample corpus for testing**: keep `industrial-revolution.txt`, add 2–3 more
      (a short bio, a technical doc, a non-English text) plus their cached JSON so the
      visualization can be developed without an LLM. `P2 S`
- [ ] **`--test` and `sample_data_visualization`**: move sample data to `data/sample.json`
      and make `--test` exercise the real renderer. `P2 S`

---

## Phase 5 – Stretch capabilities (P2, L)

- [x] **"Chat with the graph"** (issue #6): `generate-graph query graph.json "How did the
      steam engine affect cities?"` which retrieves the relevant subgraph (search + k-hop)
      and asks the LLM with the triples as context; later expose the same in the HTML via a
      user-supplied endpoint. `P2 L`
- [x] **Lightweight local web UI** (waves 11a/11b: ingest, library, explorer and chat; localhost, no accounts) (`generate-graph serve`): upload text, watch progress,
      open the result. Evaluate PR #22 (2.3k-line generated FastAPI app) as a starting point
      but keep it optional and out of the core package. `P2 L`
- [x] **Neo4j / Cypher export** (issue #18) and GraphML for Gephi. `P2 S`
- [ ] **Incremental / multi-document graphs**: merge new documents into an existing JSON
      graph with entity standardization across documents. `P2 L`
- [ ] **Code-project graphs** (issue #12): out of scope for prompts aimed at prose; answer
      the issue with guidance and close. `P2 S`

---

## Open issues & PRs triage

| Item | Recommendation |
|------|----------------|
| #15 gpt-5 needs `max_completion_tokens` | Fix in Phase 1 (LLM client). |
| #9 `utf-8` decode error on PDF | Phase 1 encoding fallback + Phase 3 PDF input. |
| #11 graph readability | Phase 2 (label thresholds, filters, freeze physics). |
| #6 chat with KG | Phase 5. |
| #7 Chinese output | Phase 3 language option. |
| #17 / PR #16 Gemini config | Merge as documentation in Phase 4. |
| #23 xllamacpp / grammar-constrained JSON | Decline as a dependency; JSON mode + repair covers it. |
| #24 Leiden via icebug | Optional extra in Phase 3; default to networkx Louvain. |
| #12 code projects | Answer and close. |
| #8 "api can not be request" | Ask for details; timeouts/retries + clearer errors will help. |
| PR #22 web interface | Do not merge as-is; revisit in Phase 5. |
| #20 collaboration | Personal decision; no code action. |

---

## Suggested order of work

1. **Phase 1** in one PR each for `llm.py`, inference, template fixes → release **0.6.2**.
2. **Phase 4 package layout + tests + CI** early, so the visualization rewrite lands on a
   tested base.
3. **Phase 2** visualization rewrite (own template, details panel, search, filters, path
   finder) → release **0.7.0** with a regenerated demo page and new screenshots.
4. **Phase 3** entity types, provenance, parallel extraction, community names → **0.8.0**.
5. Phase 5 as interest dictates.

## Notes from the test run (for reproducing)

```bash
.venv/bin/python generate-graph.py --config config-working.toml \
  --input data/industrial-revolution.txt --output out/ir.html --debug
```

- 3 chunks of 500 words; chunks 1–2 returned empty content with
  `finish_reason: "length"` (8192 completion tokens, all reasoning). Chunk 3 (66 words)
  succeeded → 20 extracted + 18 inferred triples, 12 nodes.
- `reasoning_effort: "low"` produced partial JSON (still truncated); `think: false` had
  no effect on this model through the OpenAI-compatible endpoint. Raising `max_tokens`
  to 32768 or using a non-reasoning model is the practical fix; the app must detect the
  truncation either way.
- **Follow-up run with `max_tokens = 32768`** (`testnew.html` / `testnew.json`): all
  chunks succeeded → 414 extracted, 1058 inferred, 232 nodes. Visually crowded; the
  page's own "edge → inferred → false" filter makes it readable, which is the evidence
  behind the inference defaults above.
- Browser console on `docs/index.html`: `ReferenceError: network is not defined`,
  plus vis.js "Unknown option" errors for `tooltipDelay` and `background`.
