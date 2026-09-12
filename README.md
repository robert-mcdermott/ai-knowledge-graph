![ai-knowledge-graph-example](https://github.com/robert-mcdermott/ai-knowledge-graph/blob/main/data/ai-knowledge-graph-example.png)

# AI Powered Knowledge Graph Generator

This system takes an unstructured text document, and uses an LLM of your choice to extract knowledge in the form of Subject-Predicate-Object (SPO) triplets, and visualizes the relationships as an interactive knowledge graph.
A demo of a knowlege graph created with this project can be found here: [Industrial-Revolution Knowledge Graph](https://robert-mcdermott.github.io/ai-knowledge-graph/)


## Features

- **Any text input**: `.txt`, `.md`, `.rst`, `.pdf` and `.docx` files or whole directories, in any language (`extraction.language`); large documents are split on sentence boundaries with overlap and extracted in parallel
- **Typed knowledge extraction**: the LLM returns Subject-Predicate-Object triples with entity types (person, organization, place, event, technology, product, work, date, concept) and every extracted relationship keeps the sentence it came from
- **Entity standardization**: case, stop-word and plural variants are merged, with an optional LLM pass for the rest
- **Conservative, traceable inference**: LLM passes bridge isolated parts of the graph and add well-known relationships between central entities; a deterministic taxonomy rule links specific terms to general ones; every inferred edge carries its method and is capped relative to the extracted edges
- **Exports**: JSON, CSV, GraphML for Gephi/yEd/Cytoscape and a Cypher script for Neo4j
- **Chat with the graph** (optional `graph-chat` command): grounded, cited answers from the generated graph
- **Local web interface** (optional `graph-serve` command): ingest, browse, explore and ask questions in the browser
- **Interactive explorer**: a single self-contained HTML file with search, click-to-highlight, a relationships panel with sources, named communities, entity-type filters, a shortest-path finder, exports and light/dark themes
- **Robust LLM client**: truncation detection for reasoning models, retries with back-off, automatic `max_completion_tokens` fallback, environment-variable API keys and an on-disk response cache
- **Works with any OpenAI-compatible endpoint**: Ollama, LM Studio, vLLM, OpenAI, Gemini, OpenRouter, LiteLLM (which fronts AWS Bedrock, Azure OpenAI, Anthropic and many others)

## Requirements

- Python 3.11+
- Dependencies: `networkx`, `jinja2`, `requests` (installed by `pip install -e .` or `uv sync`)

## Quick Start

1. Clone this repository
2. Install: `pip install -e .` (or `uv sync`)
3. Configure your settings in `config.toml`
4. Run the system:

```bash
generate-graph --input your_text_file.txt --output knowledge_graph.html
```

Or with UV:

```bash
uv run generate-graph --input your_text_file.txt --output knowledge_graph.html
```

Or straight from the checkout without installing:

```bash
python generate-graph.py --input your_text_file.txt --output knowledge_graph.html
```

### Development

```bash
pip install -e ".[dev]"   # adds pytest and ruff
pip install -e ".[all]"   # adds pypdf, python-docx and the web server
pytest -q                 # 149 tests, no LLM needed
ruff check .
```

## Configuration

The system is configured with a TOML file (`config.toml` by default, or `--config other.toml`).
Every key except `model` and `base_url` is optional; the values shown are the defaults.

```toml
[llm]
model = "gemma3"                 # any model name your endpoint accepts
api_key = "sk-1234"              # or "env:OPENAI_API_KEY" to read it from the environment
base_url = "http://localhost:11434/v1/chat/completions"  # any OpenAI-compatible chat completions URL
max_tokens = 32768               # reasoning models need a large budget, see note below
temperature = 0.2                # omit for models that only accept the default (e.g. gpt-5)
#reasoning_effort = "low"        # passed through to servers/models that support it
#timeout = 300                   # seconds per request
#max_retries = 3                 # retries on 429/5xx/connection errors
#token_param = "auto"            # auto-switches to max_completion_tokens for newer OpenAI models
#json_mode = false               # request response_format = json_object
#concurrency = 4                 # chunks extracted in parallel
#cache_dir = ".kg-cache"         # cache LLM replies; re-running the same input is free ("" disables, or --no-cache)
#[llm.extra_body]                # arbitrary extra request fields, e.g. Ollama's think switch (keep last in [llm])
#think = false
#[llm.extra_headers]             # extra HTTP headers, e.g. OpenRouter attribution

[extraction]
language = "auto"                # e.g. "Chinese" to get entity names and predicates in Chinese

[chunking]
chunk_size = 500                 # target words per chunk; chunks end on sentence boundaries
overlap = 50                     # words of trailing sentences repeated in the next chunk

[standardization]
enabled = true                   # merge case / stop-word / plural variants of the same entity
use_llm_for_entities = true      # extra LLM pass that groups remaining variants (one call)
#merge_word_subsets = false      # aggressive merging by shared words ("steam engine factories" -> "steam engine")

[inference]
enabled = true                   # master switch
use_llm_for_inference = true     # master switch for the LLM methods below
#llm_bridge = true               # link every isolated component to the main graph
#max_bridge_components = 20      # ...for up to this many components, batched
#bridge_groups_per_call = 5
#llm_hub = true                  # well-known relationships among the most central entities
#hub_entities = 25
#hub_max_new = 25
#taxonomy = true                 # rule: "quantum computing" is a "computing" (deterministic)
apply_transitive = false         # rule: A->B->C => A->C for transitive predicate families only
#transitive_predicate_groups = ["is a", "part of", "located in", "led to"]
#transitive_max_hub_degree = 10  # never chain through nodes with more connections than this
#transitive_max_per_subject = 5
#lexical = false                 # rule: "related to" edges for names sharing a word (noisy)
#lexical_min_word_length = 5
#max_inferred_ratio = 0.5        # cap inferred edges at this fraction of extracted edges

[query]                          # used by the optional graph-chat command
hops = 2                         # neighbourhood radius around the entities a question mentions
max_triples = 150                # facts sent to the model per question
max_seed_entities = 8
history_turns = 3                # previous Q&A pairs kept for follow-up questions

[visualization]
edge_smooth = false              # or "dynamic", "continuous", "curvedCW", ... (true = "continuous")
name_communities = true          # ask the LLM for a short name per community (one call)
show_inferred = true             # whether inferred (dashed) relationships are visible when the page opens
theme = "light"                  # initial theme: "light" or "dark"
edge_labels = "all"              # initial edge label mode: "all", "selection" or "none"
title_case = true                # show "Steam Engine" for the entity "steam engine" (ids are unchanged)
collapse_parallel_edges = true   # draw one edge per node pair with a "+N" badge; all predicates stay in the tooltip
```

Rule-based inference (`apply_transitive`, `lexical`) is off by default because in testing it generated
roughly 70 % of all edges and hid the relationships actually found in the text. Local overrides such as
`config-*.toml` and `config.local.toml` are ignored by git.

### Provider examples

```toml
# OpenAI
[llm]
model = "gpt-4.1-mini"
api_key = "env:OPENAI_API_KEY"
base_url = "https://api.openai.com/v1/chat/completions"

# Google Gemini (OpenAI-compatible endpoint)
[llm]
model = "gemini-2.5-flash"
api_key = "env:GEMINI_API_KEY"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# LM Studio / vLLM / LiteLLM proxy (any OpenAI-compatible server)
[llm]
model = "your-model-name"
api_key = "not-needed"
base_url = "http://localhost:1234/v1/chat/completions"
```

### A note on reasoning models

Models such as DeepSeek, Qwen3, gpt-5 and the o-series "think" before they answer, and that
hidden reasoning is charged against `max_tokens`. With a small budget the model can spend
everything on reasoning and return **no answer at all**. The generator detects this
(`finish_reason = "length"`) and aborts with an explanation instead of silently producing an
empty graph. Set `max_tokens` to 16k–32k for such models, lower `reasoning_effort`, use
smaller chunks, or pass `--continue-on-error` to skip failed chunks and accept an
incomplete graph.

## Command Line Options

- `--input PATH [PATH ...]`: Input file(s) or directories. Plain text (`.txt`, `.md`, `.rst`), `.pdf` (install the `[pdf]` extra) and `.docx` (`[docx]` extra). With several inputs every triple is tagged with its `document`
- `--output FILE`: Output HTML file for visualization (default: knowledge_graph.html)
- `--config FILE`: Path to config file (default: config.toml)
- `--debug` / `--verbose`: Show debug output, including raw LLM responses and standardization merges
- `--quiet`: Only warnings and errors on the console (the final summary is still printed)
- `--no-standardize`: Disable entity standardization
- `--no-inference`: Disable relationship inference
- `--continue-on-error`: Skip chunks whose LLM call fails or is truncated instead of aborting
- `--from-json FILE`: Re-render the visualization from a previously saved `.json` triples file (no LLM calls)
- `--no-cache`: Bypass the LLM response cache for this run
- `--export FORMATS`: Extra outputs next to the HTML, comma-separated: `json` (always), `csv`, `graphml` (Gephi, yEd, Cytoscape), `cypher` (Neo4j `MERGE` script)
- `--test`: Generate sample visualization using test data

### Usage message (--help)

```bash
generate-graph --help
usage: generate-graph [-h] [--test] [--config CONFIG] [--output OUTPUT]
                      [--input PATH [PATH ...]] [--from-json FILE]
                      [--debug] [--quiet] [--no-standardize] [--no-inference]
                      [--continue-on-error] [--no-cache] [--export FORMATS]

Knowledge Graph Generator and Visualizer

options:
  -h, --help           show this help message and exit
  --test               Generate a test visualization with sample data
  --config CONFIG      Path to configuration file
  --output OUTPUT      Output HTML file path
  --input INPUT        Path to input text file (required unless --test or
                       --from-json is used)
  --from-json FILE     Render a visualization from a previously saved triples
                       JSON file (no LLM calls)
  --debug, --verbose   Enable debug output (raw LLM responses and extracted
                       JSON)
  --quiet              Only show warnings and errors
  --no-standardize     Disable entity standardization
  --no-inference       Disable relationship inference
  --continue-on-error  Skip chunks whose LLM call fails or is truncated
                       instead of aborting
  --no-cache           Do not read or write the LLM response cache
                       (llm.cache_dir)
```

### Example Run

**Command:**

```bash
generate-graph --input data/industrial-revolution.txt --output industrial-revolution-kg.html
```
**Console Output** (gemma4 via Ollama, about 20 seconds):

```text
Using input text from file: data/industrial-revolution.txt
==================================================
PHASE 1: INITIAL TRIPLE EXTRACTION
==================================================
Processing text in 3 chunks (size: 500 words, overlap: 50 words)
Extracting with 3 parallel requests
Processing chunk 1/3 (490 words)
Processing chunk 2/3 (495 words)
Processing chunk 3/3 (76 words)
Chunk 1: 58 triples
Chunk 2: 79 triples
Chunk 3: 15 triples

Extracted a total of 152 triples from all chunks

==================================================
PHASE 2: ENTITY STANDARDIZATION
==================================================
Starting with 152 triples and 154 unique entities
Standardizing entity names across all triples...
Applied LLM-based entity standardization for 12 entity groups
Removed 7 self-referencing triples
Standardized 154 entities into 152 standard forms
After standardization: 145 triples and 133 unique entities

==================================================
PHASE 3: RELATIONSHIP INFERENCE
==================================================
Starting with 145 triples
Top 5 relationship types before inference:
  - enabled: 11 occurrences
  - used in: 8 occurrences
  - altered: 8 occurrences
  - invented: 7 occurrences
  - involved: 7 occurrences
Inferring additional relationships between entities...
Identified 11 disconnected components in the graph
LLM bridging proposed 11 relationships for 10 isolated components
LLM hub enrichment proposed 25 relationships among the 25 most central entities
Within-community LLM inference proposed 10 relationships
Taxonomy rule proposed 3 relationships
Transitive inference proposed 2 relationships (skipped 3 paths through hub nodes)
Accepted 48 inferred relationships (llm_bridge: 11, llm_hub: 23, llm_within: 10, taxonomy: 2, transitive: 2); dropped 3 already-connected pairs

Top 5 relationship types after inference:
  - enabled: 18 occurrences
  - is a: 15 occurrences
  - used in: 8 occurrences
  - altered: 8 occurrences
  - invented: 7 occurrences

Added 48 inferred relationships
Final knowledge graph: 189 triples
Saved raw knowledge graph data to industrial-revolution-kg.json
Processing 189 triples for visualization
Found 133 unique nodes
Found 48 inferred relationships
Detected 13 communities using Louvain method
Named 13 communities
Knowledge graph visualization saved to industrial-revolution-kg.html

Knowledge Graph Statistics:
Nodes: 133
Edges: 189
Communities: 13

To view the visualization, open the following file in your browser:
file:///path/to/industrial-revolution-kg.html
```

## Chat with your graph (optional)

`generate-graph` is unchanged: text in, static HTML (and JSON) out. The optional `graph-chat` command
reads that JSON and answers questions from it, using the same `[llm]` settings:

```bash
graph-chat knowledge_graph.json "How did the steam engine change cities?"
graph-chat knowledge_graph.json            # interactive session; Ctrl-D or an empty line quits
```

For each question it finds the entities the question mentions, retrieves the surrounding subgraph (and the
shortest paths between the mentioned entities), and asks the model to answer **only from those facts**. The
reply lists the facts it relied on, each marked *extracted* (with the source sentence) or *inferred*
(with the method), so answers stay traceable to the document. If the graph does not contain an answer it
says so. Tuning lives under `[query]` in the config (`hops`, `max_triples`, `max_seed_entities`,
`history_turns`); `--json` prints the result as JSON for scripting.

## Local web interface (optional)

`graph-serve` is a small local server over the same code: it lists the graphs in a directory, opens each one in
the explorer, and adds an **Ask** panel that answers questions from the graph with cited facts (click a fact to
jump to it in the graph). Install the extra and point it at the directory where you write your graphs:

```bash
pip install "ai-knowledge-graph[web]"      # FastAPI + uvicorn
graph-serve --config config.toml --graphs ./out --open
```

The library page also has a **New graph** form: paste text or upload files (`.txt`, `.md`, `.rst`, `.pdf`,
`.docx`, several at once), watch the phases run, and land in the explorer when it finishes. It runs the same
pipeline as `generate-graph` and writes the same `.json` and `.html` into the graphs directory, so the result is
usable from the command line and `graph-chat` too. One generation runs at a time.

It binds to `127.0.0.1:8008` by default, has no accounts or authentication, and only reads and writes the graphs
directory, so it is meant for your own machine. Static HTML output is unchanged: the chat panel only exists in
served pages. `--host 0.0.0.0` exposes it on your network if you put your own access control in front.

## Output files

Next to the HTML page the generator writes a JSON file with the same base name containing every triple:

```json
{
  "subject": "james watt",
  "predicate": "refined",
  "object": "steam engine",
  "subject_type": "person",
  "object_type": "technology",
  "source": "A key catalyst of the First Industrial Revolution was the refinement of the steam engine by Scottish engineer James Watt, ...",
  "chunk": 1
}
```

With `--export graphml,cypher,csv` the same graph is also written as GraphML (typed nodes with community
and degree, edges with provenance), an idempotent Neo4j Cypher script (`:Entity` plus a label per entity
type, one relationship type per predicate; run it with `cypher-shell -f graph.cypher`), and a flat CSV.

Inferred triples carry `"inferred": true` and a `"method"` (`llm_bridge`, `llm_hub`, `llm_within`,
`taxonomy`, `transitive` or `lexical`); transitive triples also record the intermediate node in `"via"`.
The JSON can be re-rendered at any time with `generate-graph --from-json graph.json --output graph.html`
(or `python json_to_html.py graph.json graph.html`), and the page itself can export the visible triples as
JSON or CSV.

## How It Works

1. **Chunking**: The document is split into overlapping chunks on sentence boundaries to fit within the LLM's context window
2. **First Pass - SPO Extraction**: 
   - Chunks are processed by the LLM in parallel (`llm.concurrency`) to extract typed Subject-Predicate-Object triplets, each tagged with the sentence it came from
   - Implemented in the `process_with_llm` function
   - The LLM identifies entities and their relationships within each text segment
   - Results are collected across all chunks to form the initial knowledge graph
3. **Second Pass - Entity Standardization**:
   - Variants that differ only by case, stop-words or plural form are merged ("The Steam Engines" / "steam engine")
   - Optional LLM-assisted entity alignment (`standardization.use_llm_for_entities`, one call): the LLM reviews the most frequent entities and groups the ones that refer to the same concept (e.g., "AI", "artificial intelligence", "AI system")
   - Predicate tense variants are normalized ("involve" / "involves"), and self-referencing triples are dropped
4. **Third Pass - Relationship Inference** (`inference.enabled`):
   - LLM *bridging* links every isolated component of the graph to the main component, in batched calls
   - LLM *hub enrichment* adds well-known relationships between the most central entities (e.g. "internet enabled e-commerce")
   - LLM *within-component* inference connects lexically related but unconnected pairs
   - A deterministic *taxonomy* rule links specific terms to their general term ("quantum computing" is a "computing")
   - Optional rules (off by default): transitive chains through transitive predicate families, and lexical similarity
   - Candidates are accepted in that priority order; a node pair is never connected twice, and the total is capped at `max_inferred_ratio` × the number of extracted triples. Every inferred triple records its method.
5. **Visualization**: centrality metrics and Louvain communities are computed with NetworkX, the LLM names the communities (one call), and the interactive page is rendered from the project's own template with the vis-network library embedded

The second and third passes and community naming are optional and can be disabled in the configuration to minimize LLM usage. LLM replies are cached on disk, so re-running the same document with the same settings makes no API calls.

## Visualization Features

The generated HTML is a single self-contained file (vis-network is embedded) that works offline.

- **Explore by clicking**: click a node to highlight its neighbourhood and open a details panel listing every
  incoming and outgoing relationship, tagged *extracted* or with its inference method; click a row to jump to
  that node. Double-click to zoom in.
- **Entity types**: the extraction pass labels every entity (person, organization, place, event, technology,
  product, work, date, concept); node shapes reflect the type and the Communities panel can filter by it.
- **Provenance**: each extracted relationship carries the sentence it came from, shown in tooltips and in
  the details panel, so inferred edges are easy to tell apart from what the text actually says.
- **Named communities**: after community detection the LLM gives each community a short name (one call;
  disable with `visualization.name_communities = false`).
- **Path finder**: from any selected node, type another node's name to highlight the shortest chain of
  relationships between them, optionally through extracted relationships only.
- **Search** with autocomplete (`/`), and shareable links: the selected node is kept in the URL (`#node=...`).
- **Communities panel**: colour-coded Louvain communities with their top entities, toggle any of them on/off,
  a minimum-connections slider, and a switch for inferred relationships.
- **Parallel edges collapsed**: when two entities share several relationships one edge is drawn with a "+N" badge; the tooltip and the details panel still list every predicate (`visualization.collapse_parallel_edges`).
- **Display names**: lower-case entity names are shown title-cased ("Steam Engine"); the underlying names, search and exports are unchanged (`visualization.title_case`).
- **Edge labels**: shown for all edges by default (`visualization.edge_labels`); cycle to *none* or *selection only*, which keeps dense graphs readable. Around a selected node only its own edges are labelled.
- **Node size** reflects importance (degree, betweenness and eigenvector centrality).
- **Extracted vs inferred**: solid lines are extracted from the text, dashed lines are inferred; tooltips show
  the inference method and, for transitive edges, the intermediate node.
- **Export** the current view as PNG, or the visible triples as JSON or CSV.
- **Physics controls**, a layout progress bar, automatic physics freeze for graphs over 300 nodes, light and
  dark themes (`visualization.theme`), keyboard shortcuts (`?` for the list), responsive layout.

## Project Layout

```
.
├── config.toml                     # Main configuration file for the system
├── generate-graph.py               # Run from a checkout without installing
├── json_to_html.py                 # Re-render a saved .json graph (same as --from-json)
├── pyproject.toml                  # Project metadata, dependencies, tool config
├── requirements.txt                # Pinned dependencies for 'pip' users
├── uv.lock                         # Lock file for 'uv' users
├── IMPROVEMENT_PLAN.md             # Roadmap / task list
├── .github/workflows/ci.yml        # Lint + tests on Python 3.11-3.13
├── data/                           # Sample input text and screenshot
├── docs/                           # GitHub Pages demo
├── tests/                          # pytest suite (runs without an LLM)
└── src/knowledge_graph/            # Core package (installed as `knowledge_graph`)
    ├── __init__.py                 # Package initialization and version
    ├── config.py                   # Configuration loading, defaults and validation
    ├── entity_standardization.py   # Entity standardization and relationship inference
    ├── exports.py                  # CSV, GraphML and Cypher exports
    ├── query.py                    # Optional graph-chat command (question answering over the JSON)
    ├── server.py                   # Optional graph-serve command (local web interface)
    ├── llm.py                      # LLM client (retries, truncation detection, cache) and JSON extraction
    ├── main.py                     # CLI, input handling and pipeline orchestration
    ├── text_utils.py               # Sentence splitting, chunking and provenance lookup
    ├── visualization.py            # Graph metrics, communities and page rendering
    ├── prompts/                    # LLM prompts (extraction, entity resolution, inference, community naming)
    └── templates/
        ├── graph.html.j2           # The interactive explorer page (Jinja2)
        ├── library.html.j2         # Graph library + new-graph form for graph-serve
        ├── job.html.j2             # Generation progress page for graph-serve
        └── vendor/                 # Embedded vis-network library
```

## Program Flow

This diagram illustrates the program flow.

```mermaid
flowchart TD
    A[main.py - Entry Point] --> B{Parse Arguments}

    B -->|--test| C[sample_data_visualization]
    B -->|--from-json| C2[load_triples_from_json]
    C --> O
    C2 --> O
    B -->|normal processing| E[load_config / validate_config]
    E --> E2[read_input_text]
    E2 --> F[process_text_in_chunks]

    F --> G[chunk_text - sentence aware]
    G --> H[process_with_llm - parallel]
    H --> I[LLMClient.complete - cache, retries, truncation check]
    I --> J[extract_json_from_text]
    J --> J2[normalize_triple - types + source sentence]

    F --> K{standardization enabled?}
    K -->|yes| L[standardize_entities]
    K -->|no| M{inference enabled?}
    L --> L1[_resolve_entities_with_llm]
    L --> L2[normalize_predicates]
    L --> M

    M -->|yes| N[infer_relationships]
    M -->|no| O[visualize_knowledge_graph]
    N --> N1[_identify_communities]
    N --> N2[_infer_bridges_with_llm]
    N --> N3[_infer_hub_relationships_with_llm]
    N --> N4[_infer_within_community_relationships]
    N --> N5[_infer_taxonomy]
    N --> N6[_apply_transitive_inference - optional]
    N --> N7[_infer_relationships_by_lexical_similarity - optional]
    N --> N8[budget, pair dedupe, _deduplicate_triples]
    N --> O

    O --> P[build_graph_data - centrality, Louvain, types, shapes]
    P --> P2[community namer - LLM]
    P2 --> Q[render_html - graph.html.j2 + vis-network]
    Q --> V[HTML explorer page]
    F --> W[JSON triples export]

    Y[prompts/] --> H
    Y --> L1
    Y --> N2
    Y --> N3
    Y --> N4
    Y --> P2
```

## Program Flow Description

1. **Entry Point**: `main.py` parses the command line and loads and validates the configuration (`config.py`), applying defaults and resolving `env:` API keys.

2. **Mode Selection**:
   - `--test` renders the built-in sample graph
   - `--from-json` re-renders a previously saved triples file without any LLM calls
   - Otherwise the input text file is read (UTF-8 with fallbacks; binary formats are rejected with a hint)

3. **Extraction**: `text_utils.py` splits the text into sentence-aligned chunks; chunks are sent to the LLM in parallel through `LLMClient` (`llm.py`), which caches replies, retries transient failures and aborts on truncated answers. Replies are parsed with `extract_json_from_text`, typed, and tagged with their source sentence.

4. **Entity Standardization** (optional): merges entity variants, optionally with an LLM pass, and normalizes predicates.

5. **Relationship Inference** (optional): LLM bridging, hub enrichment and within-component passes plus the taxonomy rule (and optional transitive/lexical rules), with priority ordering, pair de-duplication and a budget.

6. **Visualization**: `visualization.py` computes centrality and Louvain communities, asks the LLM to name the communities, and renders the explorer page from `templates/graph.html.j2` with vis-network embedded.

7. **Output**: the HTML page and the JSON triples file, plus a statistics summary on the console.

## Troubleshooting

- **"Response ... was cut off (finish_reason='length')"**: the model spent the token budget on hidden reasoning. Raise `max_tokens` to 32k, set `reasoning_effort = "low"`, use smaller chunks, or switch to a non-reasoning model. `--continue-on-error` skips the failed chunk instead of aborting.
- **gpt-5 / o-series reject `max_tokens` or `temperature`**: the client switches to `max_completion_tokens` automatically; remove the `temperature` line for models that only accept the default.
- **"Reading PDFs needs the optional 'pypdf' package"**: `pip install "ai-knowledge-graph[pdf]"` (or `[docx]`, `[all]`). Scanned PDFs have no text layer and need OCR first. Text encodings are detected automatically.
- **Identical re-runs still call the API**: the cache key includes the model and every request parameter, so any config change is a miss. Delete `.kg-cache/` to start fresh.
- **Graph looks fragmented**: check that `inference.enabled` and `use_llm_for_inference` are on; small models sometimes name the same concept differently across chunks, and a lower `temperature` helps.
