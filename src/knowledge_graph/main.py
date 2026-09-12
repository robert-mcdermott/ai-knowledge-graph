"""
Knowledge Graph Generator and Visualizer main module.
"""
import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from knowledge_graph.config import load_config
from knowledge_graph.entity_standardization import infer_relationships, limit_predicate_length, standardize_entities
from knowledge_graph.exports import export_graph, parse_formats
from knowledge_graph.llm import LLMClient, LLMError, extract_json_from_text
from knowledge_graph.logging_utils import configure_logging, level_from_flags
from knowledge_graph.prompts import prompt_factory, system_prompt_for
from knowledge_graph.text_utils import chunk_text, find_source_sentence
from knowledge_graph.visualization import ENTITY_TYPES, render_knowledge_graph, sample_data_visualization

log = logging.getLogger("knowledge_graph.main")

def process_with_llm(config, input_text, debug=False, client=None):
    """
    Process input text with LLM to extract triples.

    Args:
        config: Configuration dictionary
        input_text: Text to analyze
        debug: If True, print detailed debug information
        client: Optional LLMClient to reuse (built from config when omitted)

    Returns:
        List of extracted triples or None if the response contained no valid triples

    Raises:
        LLMError: if the request failed or the response was truncated/empty
    """
    # Use prompts from the centralized prompt factory
    system_prompt = system_prompt_for("main_system", config)
    user_prompt = prompt_factory.get_prompt("main_user")
    user_prompt += f"```\n{input_text}```\n"

    if client is None:
        client = LLMClient.from_config(config)

    # Process with LLM (raises LLMError on truncated/empty/failed responses)
    response = client.complete(user_prompt, system_prompt)

    # Print raw response only if debug mode is on
    if debug:
        log.debug("Raw LLM response:")
        log.info(response)
        log.info("\n---\n")

    # Extract JSON from the response
    result = extract_json_from_text(response)

    if result:
        # Validate and filter triples to ensure they have all required fields
        valid_triples = []
        invalid_count = 0

        for item in result:
            if isinstance(item, dict) and "subject" in item and "predicate" in item and "object" in item \
                    and isinstance(item["subject"], str) and isinstance(item["object"], str):
                valid_triples.append(normalize_triple(item, input_text))
            else:
                invalid_count += 1

        if invalid_count > 0:
            log.warning(f"Filtered out {invalid_count} invalid triples missing required fields")

        if not valid_triples:
            log.error("No valid triples found in LLM response")
            return None

        # Apply predicate length limit to all valid triples
        for triple in valid_triples:
            triple["predicate"] = limit_predicate_length(triple["predicate"])

        # Print extracted JSON only if debug mode is on
        if debug:
            log.debug("Extracted JSON:")
            log.info(json.dumps(valid_triples, indent=2))  # Pretty print the JSON

        return valid_triples
    else:
        # Always print error messages even if debug is off
        log.error("Could not extract valid JSON from response: " + str(response) + "\n\n")
        return None

def normalize_triple(item, chunk_text_value):
    """Keep the known fields of an extracted triple, validate types and attach the source sentence."""
    triple = {"subject": item["subject"].strip(), "predicate": str(item["predicate"]).strip(),
              "object": item["object"].strip()}
    for key in ("subject_type", "object_type"):
        value = item.get(key)
        if isinstance(value, str) and value.strip().lower() in ENTITY_TYPES:
            triple[key] = value.strip().lower()
    source = find_source_sentence(chunk_text_value, triple["subject"], triple["object"])
    if source:
        triple["source"] = source
    return triple


def process_text_in_chunks(config, full_text, debug=False, continue_on_error=False):
    """Process a single text (see :func:`process_documents`)."""
    return process_documents(config, [(None, full_text)], debug, continue_on_error)


def process_documents(config, documents, debug=False, continue_on_error=False):
    """
    Chunk every document, extract triples from all chunks in parallel, then run the
    standardization and inference phases over the combined result.

    Args:
        config: Configuration dictionary
        documents: list of ``(name, text)`` pairs; ``name`` may be None for a single text
        debug: If True, print detailed debug information
        continue_on_error: If True, skip chunks whose LLM call fails instead of aborting

    Returns:
        List of all extracted (and inferred) triples; triples carry ``chunk`` and, when more
        than one document was given, ``document``

    Raises:
        LLMError: when a chunk fails and continue_on_error is False
    """
    chunk_size = config.get("chunking", {}).get("chunk_size", 500)
    overlap = config.get("chunking", {}).get("overlap", 50)

    text_chunks, chunk_docs = [], []
    for name, text in documents:
        pieces = chunk_text(text, chunk_size, overlap)
        text_chunks.extend(pieces)
        chunk_docs.extend([name] * len(pieces))
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
            results = process_with_llm(config, chunk, debug, client=client)
        except LLMError as e:
            return i, None, e
        return i, results, None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(run_chunk, enumerate(text_chunks)))
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
            continue
        if chunk_results:
            for item in chunk_results:
                item["chunk"] = i + 1
                if multi_doc and chunk_docs[i]:
                    item["document"] = chunk_docs[i]
            all_results.extend(chunk_results)
            log.info(f"Chunk {i+1}: {len(chunk_results)} triples")
        else:
            log.warning(f"Failed to extract triples from chunk {i+1}")

    log.info(f"\nExtracted a total of {len(all_results)} triples from all chunks")
    if failed_chunks:
        log.warning(f"{len(failed_chunks)} of {len(text_chunks)} chunks failed and were skipped: "
              f"{failed_chunks}. The graph is incomplete.")
    if not all_results:
        return []

    # Apply entity standardization if enabled
    if config.get("standardization", {}).get("enabled", False):
        log.info("\n" + "="*50)
        log.info("PHASE 2: ENTITY STANDARDIZATION")
        log.info("="*50)
        log.info(f"Starting with {len(all_results)} triples and {len(get_unique_entities(all_results))} unique entities")

        all_results = standardize_entities(all_results, config)

        log.info(f"After standardization: {len(all_results)} triples and {len(get_unique_entities(all_results))} unique entities")

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
    text = "\n\n".join(p for p in pages if p)
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
    return [(os.path.basename(p), read_input_text(p)) for p in collect_input_files(paths)]


def load_triples_from_json(path):
    """Load a previously generated *.json triples file (skips all LLM phases)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise InputError(f"Could not load triples from {path}: {e}") from e
    if isinstance(data, dict) and isinstance(data.get("triples"), list):
        data = data["triples"]
    if not isinstance(data, list) or not all(isinstance(t, dict) and "subject" in t and "object" in t for t in data):
        raise InputError(f"{path} does not contain a list of subject/predicate/object triples.")
    return data


def meta_path_for(json_path):
    return os.path.splitext(json_path)[0] + ".meta.json"


def save_graph_meta(json_path, graph_data, config=None):
    """Write the sidecar with community names (and provenance) next to a triples JSON file."""
    communities = [{"id": c["id"], "name": c.get("name"), "top": c.get("top", [])[:3]}
                   for c in graph_data["meta"]["communities"] if c.get("name")]
    if not communities:
        return None
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

    args = parser.parse_args()
    configure_logging(level_from_flags(verbose=args.debug, quiet=args.quiet))

    # Load configuration
    config = load_config(args.config)
    if not config:
        print(f"Failed to load configuration from {args.config}. Exiting.")
        sys.exit(1)

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
                                                   library_dir=args.library_path)
        for path in export_graph(triples, args.output, [f for f in export_formats if f != "json"], graph_data):
            print(f"Exported {path}")
        print(f"\nNodes: {stats['nodes']}  Edges: {stats['edges']}  Communities: {stats['communities']}")
        print(f"file://{os.path.abspath(args.output)}")
        return

    # For normal processing, input file is required
    if not args.input:
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
        documents = read_documents(args.input)
        if len(documents) == 1:
            print(f"Using input text from file: {documents[0][0]}")
        else:
            print(f"Using input text from {len(documents)} files: {', '.join(name for name, _ in documents)}")
    except InputError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Process text in chunks
    try:
        result = process_documents(config, documents, args.debug, args.continue_on_error)
    except LLMError as e:
        print(f"\nERROR: {e}", flush=True)
        print("Knowledge graph generation aborted.")
        sys.exit(1)

    if result:
        # Save the raw data as JSON for potential reuse (before rendering, so it survives a render error)
        json_output = os.path.splitext(args.output)[0] + '.json'
        try:
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved raw knowledge graph data to {json_output}")
        except Exception as e:
            print(f"Warning: Could not save raw data to {json_output}: {e}")

        # Visualize the knowledge graph, then write any extra export formats (with community names)
        stats, graph_data = render_knowledge_graph(result, args.output, config=config,
                                                   community_namer=make_community_namer(config),
                                                   library_dir=args.library_path)
        if save_graph_meta(json_output, graph_data, config):
            print(f"Saved community names to {meta_path_for(json_output)}")
        for path in export_graph(result, args.output, [f for f in export_formats if f != "json"], graph_data):
            print(f"Exported {path}")
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
