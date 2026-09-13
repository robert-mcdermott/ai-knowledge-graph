"""
Knowledge Graph Generator and Visualizer main module.
"""
import argparse
import copy
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from knowledge_graph.config import load_config
from knowledge_graph.entity_standardization import infer_relationships, standardize_entities
from knowledge_graph.exports import export_graph, parse_formats
from knowledge_graph.llm import LLMClient, LLMError, extract_json_from_text
from knowledge_graph.logging_utils import configure_logging, level_from_flags
from knowledge_graph.profiles import profile_prompt
from knowledge_graph.prompts import prompt_factory, system_prompt_for
from knowledge_graph.visualization import ENTITY_TYPES, render_knowledge_graph, sample_data_visualization
from knowledge_graph.workspace import (
    WorkspaceError,
    apply_document_update,
    document_record,
    empty_workspace,
    load_workspace,
    merge_claims,
    passage_chunks,
    passages,
    preview_documents,
    projected_claims,
    save_workspace,
    validate_triples,
    workspace_path,
)

log = logging.getLogger("knowledge_graph.main")

@dataclass
class RunContext:
    cancel: threading.Event = field(default_factory=threading.Event)
    failures: list = field(default_factory=list)
    completed: int = 0
    total: int = 0
    started: float = field(default_factory=time.monotonic)
    usage: dict = field(default_factory=dict)
    cache_hits: int = 0
    callback: object = None

    def snapshot(self):
        return {"status": "partial" if self.failures else "complete", "failures": self.failures,
                "chunks_total": self.total, "chunks_done": self.completed, "usage": self.usage,
                "cache_hits": self.cache_hits, "elapsed_seconds": round(time.monotonic() - self.started, 2)}

    def check(self):
        if self.cancel.is_set():
            raise LLMError("Generation cancelled; the existing graph was preserved.")


def process_with_llm(config, input_text, debug=False, client=None, source_passages=None):
    """Extract qualified claims with explicit, validated references into the source."""
    source_passages = source_passages if source_passages is not None else passages(input_text)
    numbered = {f"P{i + 1}": p for i, p in enumerate(source_passages)}
    system_prompt = system_prompt_for("main_system", config) + "\n" + profile_prompt(config)
    user_prompt = prompt_factory.get_prompt("main_user") + "```\n" + "\n".join(
        f"[{pid}] {p['text']}" for pid, p in numbered.items()) + "\n```"
    client = client or LLMClient.from_config(config)
    response = client.complete(user_prompt, system_prompt)
    if debug:
        log.debug("Raw extraction response: %s", response)
    result = extract_json_from_text(response, allow_salvage=False)
    if result is None:
        if hasattr(client, "invalidate"):
            client.invalidate(user_prompt, system_prompt)
        raise LLMError("Extraction returned malformed JSON; this chunk needs retrying.")
    if not result:
        return []  # Valid empty extraction is different from a parsing failure.
    try:
        validate_triples(result)
    except WorkspaceError as e:
        if hasattr(client, "invalidate"):
            client.invalidate(user_prompt, system_prompt)
        raise LLMError(str(e)) from e
    try:
        return [normalize_triple(item, input_text, numbered, config) for item in result]
    except LLMError:
        if hasattr(client, "invalidate"):
            client.invalidate(user_prompt, system_prompt)
        raise


def normalize_triple(item, chunk_text_value, numbered=None, config=None):
    """Never manufacture a citation from entity co-occurrence."""
    triple = {key: item[key].strip() for key in ("subject", "predicate", "object")}
    allowed_types = (config or {}).get("extraction", {}).get("entity_types") or ENTITY_TYPES
    for key in ("subject_type", "object_type"):
        value = item.get(key)
        if isinstance(value, str) and value.strip().lower() in allowed_types:
            triple[key] = value.strip().lower()
    for key in ("time", "attribution", "polarity"):
        if isinstance(item.get(key), str) and item[key].strip():
            triple[key] = item[key].strip()
    if triple.get("polarity") not in (None, "positive", "negative", "uncertain"):
        triple["polarity"] = "uncertain"
    ids = item.get("source_ids", [])
    if not isinstance(ids, list) or any(not isinstance(pid, str) for pid in ids):
        raise LLMError("source_ids must be an array of passage IDs")
    if any(pid not in (numbered or {}) for pid in ids):
        raise LLMError("The model cited a passage ID that is not in this chunk")
    evidence = [dict(numbered[pid]) for pid in dict.fromkeys(ids)]
    if evidence:
        triple["evidence"] = evidence
        triple["source"] = evidence[0]["text"]
        triple["evidence_status"] = "source-linked"
    else:
        triple["evidence_status"] = "unverified"
        if (config or {}).get("extraction", {}).get("strict_evidence", False):
            raise LLMError("A claim has no supporting passage IDs; retry or disable strict_evidence")
    return triple


def process_text_in_chunks(config, full_text, debug=False, continue_on_error=False):
    """Process a single text (see :func:`process_documents`)."""
    return process_documents(config, [(None, full_text)], debug, continue_on_error)


def process_documents(config, documents, debug=False, continue_on_error=False, run=None):
    """
    Chunk every document, extract triples from all chunks in parallel, then run the
    standardization and inference phases over the combined result.

    Args:
        config: Configuration dictionary
        documents: list of ``(name, text)`` pairs; ``name`` may be None for a single text
        debug: If True, print detailed debug information
        continue_on_error: If True, skip chunks whose LLM call fails instead of aborting

    Returns:
        List of all extracted (and inferred) triples; extracted triples carry ``chunk``,
        ``document_id``, and a ``document`` name when one was supplied

    Raises:
        LLMError: when a chunk fails and continue_on_error is False
    """
    run = run or RunContext()
    run.check()
    chunk_size = config.get("chunking", {}).get("chunk_size", 500)
    overlap = config.get("chunking", {}).get("overlap", 50)

    text_chunks, chunk_docs, chunk_passages = [], [], []
    for name, text in documents:
        doc = document_record(name, text)
        batches = passage_chunks(doc["passages"], chunk_size, overlap)
        for batch in batches:
            text_chunks.append(" ".join(p["text"] for p in batch))
            chunk_docs.append(name)
            chunk_passages.append(batch)
    run.total = len(text_chunks)
    multi_doc = len(documents) > 1

    log.info("=" * 50)
    log.info("PHASE 1: INITIAL TRIPLE EXTRACTION")
    log.info("=" * 50)
    if multi_doc:
        log.info(f"Processing {len(documents)} documents in {len(text_chunks)} chunks (size: {chunk_size} words, overlap: {overlap} words)")
    else:
        log.info(f"Processing text in {len(text_chunks)} chunks (size: {chunk_size} words, overlap: {overlap} words)")

    # Process chunks concurrently with a single shared client; results keep chunk order.
    client = LLMClient.from_config(config)
    concurrency = max(1, min(int(config.get("llm", {}).get("concurrency", 4)), len(text_chunks)))
    if concurrency > 1:
        log.info(f"Extracting with {concurrency} parallel requests")

    def run_chunk(index_chunk):
        i, chunk = index_chunk
        log.info(f"Processing chunk {i+1}/{len(text_chunks)} ({len(chunk.split())} words)")
        try:
            run.check()
            results = process_with_llm(config, chunk, debug, client=client, source_passages=chunk_passages[i])
        except LLMError as e:
            return i, None, e
        return i, results, None

    outcomes = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = [pool.submit(run_chunk, pair) for pair in enumerate(text_chunks)]
        for future in as_completed(pending):
            if run.cancel.is_set():
                for task in pending:
                    task.cancel()
                break
            outcome = future.result()
            outcomes.append(outcome)
            run.completed += 1
            run.cache_hits = getattr(client, "cache_hits", 0)
            run.usage = dict(getattr(client, "usage", {}))
            if run.callback:
                run.callback(run.snapshot())
    run.check()
    outcomes.sort(key=lambda outcome: outcome[0])
    cache_hits = getattr(client, "cache_hits", 0)
    if cache_hits:
        log.info(f"Served {cache_hits} of {len(text_chunks)} chunks from the LLM cache ({client.cache_dir})")

    all_results = []
    failed_chunks = []
    for i, chunk_results, error in outcomes:
        if error is not None:
            if not continue_on_error:
                raise LLMError(f"Chunk {i+1}/{len(text_chunks)} failed: {error}\n"
                               f"(Use --continue-on-error to skip failed chunks instead of aborting.)") from error
            log.warning(f"skipping chunk {i+1}: {error}")
            failed_chunks.append(i + 1)
            run.failures.append({"chunk": i + 1, "document": chunk_docs[i], "error": str(error)})
            continue
        if chunk_results:
            for item in chunk_results:
                item["chunk"] = i + 1
                item["document_id"] = document_record(chunk_docs[i], "")["id"]
                if chunk_docs[i]:
                    item["document"] = chunk_docs[i]
            all_results.extend(chunk_results)
            log.info(f"Chunk {i+1}: {len(chunk_results)} triples")
        else:
            log.info(f"Chunk {i+1}: 0 triples (no claims found)")

    log.info(f"\nExtracted a total of {len(all_results)} triples from all chunks")
    if failed_chunks:
        log.warning(f"{len(failed_chunks)} of {len(text_chunks)} chunks failed and were skipped: "
              f"{failed_chunks}. The graph is incomplete.")
    if not all_results:
        return []

    run.check()
    all_results = merge_claims(all_results)
    # Apply entity standardization if enabled
    if config.get("standardization", {}).get("enabled", False):
        log.info("\n" + "="*50)
        log.info("PHASE 2: ENTITY STANDARDIZATION")
        log.info("="*50)
        log.info(f"Starting with {len(all_results)} triples and {len(get_unique_entities(all_results))} unique entities")

        all_results = standardize_entities(all_results, config)

        log.info(f"After standardization: {len(all_results)} triples and {len(get_unique_entities(all_results))} unique entities")

    run.check()
    # Apply relationship inference if enabled
    if config.get("inference", {}).get("enabled", False):
        log.info("\n" + "="*50)
        log.info("PHASE 3: RELATIONSHIP INFERENCE")
        log.info("="*50)
        log.info(f"Starting with {len(all_results)} triples")

        # Count existing relationships
        relationship_counts = {}
        for triple in all_results:
            relationship_counts[triple["predicate"]] = relationship_counts.get(triple["predicate"], 0) + 1

        log.info("Top 5 relationship types before inference:")
        for pred, count in sorted(relationship_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            log.info(f"  - {pred}: {count} occurrences")

        all_results = infer_relationships(all_results, config)

        # Count relationships after inference
        relationship_counts_after = {}
        for triple in all_results:
            relationship_counts_after[triple["predicate"]] = relationship_counts_after.get(triple["predicate"], 0) + 1

        log.info("\nTop 5 relationship types after inference:")
        for pred, count in sorted(relationship_counts_after.items(), key=lambda x: x[1], reverse=True)[:5]:
            log.info(f"  - {pred}: {count} occurrences")

        # Count inferred relationships
        inferred_count = sum(1 for triple in all_results if triple.get("inferred", False))
        log.info(f"\nAdded {inferred_count} inferred relationships")
        log.info(f"Final knowledge graph: {len(all_results)} triples")

    return all_results

TEXT_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")  # utf-8-sig also reads plain UTF-8 and strips a BOM
TEXT_EXTENSIONS = {".txt", ".text", ".md", ".markdown", ".rst", ".log", ""}
BINARY_EXTENSIONS = {".doc", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".gz", ".png", ".jpg", ".jpeg", ".gif"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx"}


class InputError(Exception):
    """Raised when the input file cannot be read as text."""


def read_input_text(path):
    """Read a text file trying several encodings; refuse binary formats with a clear message.

    Raises:
        InputError: with a message that explains what to do.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    if ext in BINARY_EXTENSIONS:
        raise InputError(f"{path} looks like a {ext} file. Supported inputs are plain text (.txt, .md, .rst), "
                         f".pdf and .docx; convert it first.")
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise InputError(f"Could not read {path}: {e}") from e
    if not raw.strip():
        raise InputError(f"{path} is empty.")
    if b"\x00" in raw[:4096]:
        raise InputError(f"{path} contains binary data (NUL bytes); only plain text files are supported.")
    for encoding in TEXT_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding != "utf-8-sig":
            log.info(f"Note: {path} decoded as {encoding}")
        return text
    raise InputError(f"Could not decode {path} as text (tried {', '.join(TEXT_ENCODINGS)}).")


def _read_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise InputError("Reading PDFs needs the optional 'pypdf' package: pip install 'ai-knowledge-graph[pdf]' "
                         "(or convert the file with `pdftotext file.pdf file.txt`).") from e
    try:
        reader = PdfReader(path)
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as e:
        raise InputError(f"Could not read PDF {path}: {e}") from e
    text = "\f".join(pages)
    if not text.strip():
        raise InputError(f"{path} contains no extractable text (scanned PDF?). Run OCR first.")
    return text


def _read_docx(path):
    try:
        import docx
    except ImportError as e:
        raise InputError("Reading .docx files needs the optional 'python-docx' package: "
                         "pip install 'ai-knowledge-graph[docx]'.") from e
    try:
        document = docx.Document(path)
        paragraphs = [p.text.strip() for p in document.paragraphs]
    except Exception as e:
        raise InputError(f"Could not read {path}: {e}") from e
    text = "\n\n".join(p for p in paragraphs if p)
    if not text.strip():
        raise InputError(f"{path} contains no text.")
    return text


def collect_input_files(paths):
    """Expand directories into their supported files; validate that every path exists."""
    files = []
    for path in paths:
        if os.path.isdir(path):
            found = sorted(
                os.path.join(path, name) for name in os.listdir(path)
                if os.path.isfile(os.path.join(path, name)) and os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS
                and not name.startswith(".")
            )
            if not found:
                raise InputError(f"No supported input files found in directory {path}.")
            files.extend(found)
        elif os.path.exists(path):
            files.append(path)
        else:
            raise InputError(f"Input file not found: {path}")
    return files


def read_documents(paths):
    """Return ``(name, text)`` pairs for every input path (directories expanded)."""
    files = collect_input_files(paths)
    names = [os.path.basename(p) for p in files]
    return [(os.path.abspath(p) if names.count(os.path.basename(p)) > 1 else os.path.basename(p), read_input_text(p))
            for p in files]


def load_triples_from_json(path):
    """Load a previously generated *.json triples file (skips all LLM phases)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise InputError(f"Could not load triples from {path}: {e}") from e
    try:
        if os.path.exists(workspace_path(path)) or isinstance(data, dict) and "schema_version" in data:
            return projected_claims(load_workspace(path))
        if isinstance(data, dict):
            data = data.get("triples")
        return validate_triples(data)
    except (WorkspaceError, KeyError, TypeError) as e:
        raise InputError(f"{path}: {e}") from e


def meta_path_for(json_path):
    return os.path.splitext(json_path)[0] + ".meta.json"


def save_graph_meta(json_path, graph_data, config=None):
    """Write the sidecar with community names (and provenance) next to a triples JSON file."""
    communities = [{"id": c["id"], "name": c.get("name"), "top": c.get("top", [])[:3]}
                   for c in graph_data["meta"]["communities"] if c.get("name")]
    meta = {"community_names": {str(c["id"]): c["name"] for c in communities}, "communities": communities,
            "generated": graph_data["meta"].get("generated"),
            "model": (config or {}).get("llm", {}).get("model")}
    path = meta_path_for(json_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path


def load_graph_meta(json_path):
    """Read the sidecar written by :func:`save_graph_meta`; returns ``{}`` when absent or unreadable."""
    path = meta_path_for(json_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    names = meta.get("community_names") or {}
    try:
        meta["community_names"] = {int(k): v for k, v in names.items() if isinstance(v, str) and v.strip()}
    except (TypeError, ValueError):
        meta["community_names"] = {}
    return meta


def make_community_namer(config):
    """Return a callable that asks the LLM for short community names, or None if disabled."""
    if not config.get("visualization", {}).get("name_communities", True):
        return None

    def namer(communities):
        lines = [f"{c['id'] + 1}. {', '.join(c['top'])}" for c in communities if c.get("top")]
        if len(lines) < 2:
            return {}
        system_prompt = system_prompt_for("community_naming_system", config)
        user_prompt = prompt_factory.get_prompt("community_naming_user", "\n".join(lines))
        response = LLMClient.from_config(config).complete(user_prompt, system_prompt)
        mapping = extract_json_from_text(response, expect="object") or {}
        names = {}
        for key, value in mapping.items():
            try:
                index = int(str(key).strip()) - 1
            except ValueError:
                continue
            if isinstance(value, str) and value.strip() and 0 <= index < len(communities):
                names[index] = value.strip()[:60]
        return names

    return namer


def get_unique_entities(triples):
    """
    Get the set of unique entities from the triples.

    Args:
        triples: List of triple dictionaries

    Returns:
        Set of unique entity names
    """
    entities = set()
    for triple in triples:
        if not isinstance(triple, dict):
            continue
        if "subject" in triple:
            entities.add(triple["subject"])
        if "object" in triple:
            entities.add(triple["object"])
    return entities

def update_collection(workspace, documents, config, remove=(), run=None, continue_on_error=False):
    """Extract only changed documents; apply a complete revision in memory before saving."""
    run = run or RunContext()
    diff = preview_documents(workspace, documents, remove)
    changed = set(diff["added"] + diff["updated"])
    if not changed and not diff["removed"]:
        return copy.deepcopy(workspace), diff
    scoped = copy.deepcopy(config)
    scoped.setdefault("inference", {})["enabled"] = False
    # Keep raw identities in collections. Explicit, reversible merge overrides own alignment.
    scoped.setdefault("standardization", {})["enabled"] = False
    selected = [(name, text) for name, text in documents if name in changed]
    triples = process_documents(scoped, selected, continue_on_error=continue_on_error, run=run) if selected else []
    run.check()
    updated, diff = apply_document_update(workspace, documents, triples, remove, run.snapshot())
    if config.get("inference", {}).get("enabled", False) and updated["claims"]:
        active = projected_claims(updated)
        enriched = infer_relationships(active, config)
        updated["inferred"] = [t for t in enriched if t.get("inferred")]
    run.check()
    return updated, diff


def main():
    """Main entry point for the knowledge graph generator."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Knowledge Graph Generator and Visualizer')
    parser.add_argument('--test', action='store_true', help='Generate a test visualization with sample data')
    parser.add_argument('--config', type=str, default='config.toml', help='Path to configuration file')
    parser.add_argument('--output', type=str, default='knowledge_graph.html', help='Output HTML file path')
    parser.add_argument('--input', type=str, nargs='+', metavar='PATH',
                        help='Input file(s) or directories: .txt/.md/.rst, .pdf (needs [pdf] extra), .docx (needs [docx] extra). '
                             'Required unless --test or --from-json is used')
    parser.add_argument('--from-json', type=str, metavar='FILE',
                        help='Render a visualization from a previously saved triples JSON file (no LLM calls)')
    parser.add_argument('--debug', '--verbose', action='store_true', help='Enable debug output (raw LLM responses and extracted JSON)')
    parser.add_argument('--quiet', action='store_true', help='Only show warnings and errors')
    parser.add_argument('--no-standardize', action='store_true', help='Disable entity standardization')
    parser.add_argument('--no-inference', action='store_true', help='Disable relationship inference')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Skip chunks whose LLM call fails or is truncated instead of aborting')
    parser.add_argument('--no-cache', action='store_true',
                        help='Do not read or write the LLM response cache (llm.cache_dir)')
    parser.add_argument('--export', type=str, default='json', metavar='FORMATS',
                        help='Comma-separated extra outputs written next to the HTML: json (always), csv, graphml, cypher')
    parser.add_argument('--library-path', type=str, metavar='DIR',
                        help='Reference the vis-network library from DIR (copied there if missing) instead of embedding it; '
                             'useful when publishing many pages, e.g. on GitHub Pages')

    parser.add_argument("--collection", metavar="JSON", help="Create or update a persistent collection; extract only changed documents")
    parser.add_argument("--remove-document", action="append", default=[], metavar="NAME", help="Remove a document from --collection")
    parser.add_argument("--preview-update", action="store_true", help="Show document changes without LLM calls or writes")
    parser.add_argument("--profile", choices=("general", "research", "organizations"), help="Extraction profile")
    parser.add_argument("--strict-evidence", action="store_true", help="Fail a chunk if a claim omits source passage references")
    parser.add_argument("--check", action="store_true", help="Check the configured model with a tiny request")
    parser.add_argument("--preview", action="store_true", help="Extract only a short sample of the first document")
    args = parser.parse_args()
    configure_logging(level_from_flags(verbose=args.debug, quiet=args.quiet))

    # Load configuration
    config = load_config(args.config)
    if not config:
        print(f"Failed to load configuration from {args.config}. Exiting.")
        sys.exit(1)

    if args.profile:
        config["extraction"]["profile"] = args.profile
    if args.strict_evidence:
        config["extraction"]["strict_evidence"] = True
    if args.check:
        try:
            client = LLMClient.from_config(config)
            client.cache_dir = None
            client.timeout = min(client.timeout, 30)
            client.max_retries = 0
            client.max_tokens = min(client.max_tokens, 2048)
            client.complete('Return {"ok": true}.', 'Connection check. Return JSON.')
            print(f"Connected to {client.model} at {client.base_url}")
        except LLMError as e:
            parser.exit(1, f"Connection failed: {e}\n")
        return
    if (args.remove_document or args.preview_update) and not args.collection:
        parser.error("--remove-document and --preview-update require --collection")
    # If test flag is provided, generate a sample visualization
    if args.test:
        print("Generating sample data visualization...")
        sample_data_visualization(args.output, config=config)
        print(f"\nSample visualization saved to {args.output}")
        print("To view the visualization, open the following file in your browser:")
        print(f"file://{os.path.abspath(args.output)}")
        return

    try:
        export_formats = parse_formats(args.export)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)

    # Re-render an existing graph without touching the LLM
    if args.from_json:
        try:
            triples = load_triples_from_json(args.from_json)
        except InputError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"Loaded {len(triples)} triples from {args.from_json}")
        meta = load_graph_meta(args.from_json)
        stats, graph_data = render_knowledge_graph(triples, args.output, config=config,
                                                   community_names=meta.get("community_names") or None,
                                                   library_dir=args.library_path,
                                                   workspace=load_workspace(args.from_json))
        for path in export_graph(triples, args.output, [f for f in export_formats if f != "json"], graph_data):
            print(f"Exported {path}")
        print(f"\nNodes: {stats['nodes']}  Edges: {stats['edges']}  Communities: {stats['communities']}")
        print(f"file://{os.path.abspath(args.output)}")
        return

    # For normal processing, input file is required
    if not args.input and not (args.collection and args.remove_document):
        print("Error: --input is required unless --test or --from-json is used")
        parser.print_help()
        sys.exit(2)

    # Override configuration settings with command line arguments
    if args.no_cache:
        config.setdefault("llm", {})["cache_dir"] = None
    if args.no_standardize:
        config.setdefault("standardization", {})["enabled"] = False
    if args.no_inference:
        config.setdefault("inference", {})["enabled"] = False

    # Load input text from file(s)
    try:
        documents = read_documents(args.input or [])
        if args.preview and documents:
            documents = [(documents[0][0], documents[0][1][:2000])]
        if len(documents) == 1:
            print(f"Using input text from file: {documents[0][0]}")
        else:
            print(f"Using input text from {len(documents)} files: {', '.join(name for name, _ in documents)}")
    except InputError as e:
        print(f"Error: {e}")
        sys.exit(1)

    run = RunContext()
    workspace = None
    previous_workspace = None
    previous_path = os.path.splitext(args.output)[0] + '.json'
    if not args.collection and os.path.exists(workspace_path(previous_path)):
        try:
            previous_workspace = load_workspace(previous_path)
        except (WorkspaceError, OSError, ValueError) as e:
            parser.exit(1, f"Cannot preserve existing workspace: {e}\n")
    if args.collection:
        args.output = os.path.splitext(args.collection)[0] + ".html"
        try:
            workspace = load_workspace(args.collection) if os.path.exists(args.collection) else empty_workspace(os.path.basename(args.collection).rsplit(".", 1)[0])
            diff = preview_documents(workspace, documents, args.remove_document)
            print(json.dumps(diff, ensure_ascii=False, indent=2))
            if args.preview_update:
                return
            if not any(diff[k] for k in ("added", "updated", "removed")):
                print("Collection is unchanged; no model calls or writes needed.")
                return
        except (WorkspaceError, OSError, ValueError) as e:
            parser.exit(1, f"Collection error: {e}\n")
    # Process text in chunks
    try:
        if workspace is not None:
            workspace, diff = update_collection(workspace, documents, config, args.remove_document, run, args.continue_on_error)
            result = projected_claims(workspace)
        else:
            result = process_documents(config, documents, args.debug, args.continue_on_error, run=run)
            workspace, _ = apply_document_update(empty_workspace(os.path.basename(args.output).rsplit(".", 1)[0]), documents,
                                                  result, run=run.snapshot())
            workspace["inferred"] = [t for t in result if t.get("inferred")]
            if previous_workspace:
                for key in ("overrides", "views", "revision", "title"):
                    workspace[key] = copy.deepcopy(previous_workspace.get(key, workspace[key]))
                result = projected_claims(workspace)

    except LLMError as e:
        print(f"\nERROR: {e}", flush=True)
        print("Knowledge graph generation aborted.")
        sys.exit(1)

    if result or args.collection or previous_workspace:
        # Save the raw data as JSON for potential reuse (before rendering, so it survives a render error)
        json_output = os.path.splitext(args.output)[0] + '.json'
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        save_workspace(json_output, workspace)
        print(f"Saved graph and workspace to {json_output}")

        # Visualize the knowledge graph, then write any extra export formats (with community names)
        stats, graph_data = render_knowledge_graph(result, args.output, config=config,
                                                   community_namer=make_community_namer(config),
                                                   library_dir=args.library_path, workspace=workspace)
        if save_graph_meta(json_output, graph_data, config):
            print(f"Saved community names to {meta_path_for(json_output)}")
        for path in export_graph(result, args.output, [f for f in export_formats if f != "json"], graph_data):
            print(f"Exported {path}")
        print("\nRun: " + json.dumps(run.snapshot(), ensure_ascii=False))
        print("\nKnowledge Graph Statistics:")
        print(f"Nodes: {stats['nodes']}")
        print(f"Edges: {stats['edges']}")
        print(f"Communities: {stats['communities']}")

        # Provide command to open the visualization in a browser
        print("\nTo view the visualization, open the following file in your browser:")
        print(f"file://{os.path.abspath(args.output)}")
    else:
        print("Knowledge graph generation failed: no triples were extracted.")
        sys.exit(1)

if __name__ == "__main__":
    main()
