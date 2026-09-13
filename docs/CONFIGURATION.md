# Configuration reference

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
#cache_dir = ".kg-cache"         # cache successful replies ("" disables, or --no-cache)
#[llm.extra_body]                # arbitrary extra request fields, e.g. Ollama's think switch (keep last in [llm])
#think = false
#[llm.extra_headers]             # extra HTTP headers, e.g. OpenRouter attribution

[extraction]
profile = "general"               # general, research, organizations
strict_evidence = false           # reject chunks whose claims omit valid source IDs
#entity_types = ["person", "method", "dataset", "finding"]
#predicates = ["evaluated on", "reported", "limited by"]
language = "auto"                # e.g. "Chinese" to get entity names and predicates in Chinese

[chunking]
chunk_size = 500                 # target words (CJK characters count separately); preserve exact passage spans
overlap = 50                     # maximum tokens of trailing whole passages repeated in the next chunk

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
max_context_tokens = 6000        # estimated token budget for retrieved facts
use_llm_for_entity_matching = true
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


Collections preserve extracted entity names and use reversible manual merges; automatic standardization settings apply to ordinary single-run generation. Inference is optional in both workflows.

Use `generate-graph --check` or **Check connection** to make a small uncached request. The check has a 30-second request timeout and no retries; a reasoning model may need a larger budget for actual extraction. Request limits and model parameters depend on your configured endpoint.

Extraction profiles guide the model; they are not rigid ontologies. Custom entity types are accepted and shown with a default shape. Preferred predicates never truncate a more precise relationship.

The query context limit is a conservative estimate for retrieved facts, not a provider tokenizer count or a cap on total billed tokens. Conversation history and instructions add context. Run usage displays extraction tokens reported by the provider; inference and community naming may make additional calls.
