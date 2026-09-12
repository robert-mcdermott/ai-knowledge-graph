"""
Knowledge Graph Generator and Visualizer main module.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from knowledge_graph.config import load_config
from knowledge_graph.entity_standardization import infer_relationships, limit_predicate_length, standardize_entities
from knowledge_graph.llm import LLMClient, LLMError, extract_json_from_text
from knowledge_graph.prompts import prompt_factory
from knowledge_graph.text_utils import chunk_text, find_source_sentence
from knowledge_graph.visualization import ENTITY_TYPES, sample_data_visualization, visualize_knowledge_graph


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
    system_prompt = prompt_factory.get_prompt("main_system")
    user_prompt = prompt_factory.get_prompt("main_user")
    user_prompt += f"```\n{input_text}```\n"

    if client is None:
        client = LLMClient.from_config(config)

    # Process with LLM (raises LLMError on truncated/empty/failed responses)
    response = client.complete(user_prompt, system_prompt)

    # Print raw response only if debug mode is on
    if debug:
        print("Raw LLM response:")
        print(response)
        print("\n---\n")

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
            print(f"Warning: Filtered out {invalid_count} invalid triples missing required fields")

        if not valid_triples:
            print("Error: No valid triples found in LLM response")
            return None

        # Apply predicate length limit to all valid triples
        for triple in valid_triples:
            triple["predicate"] = limit_predicate_length(triple["predicate"])

        # Print extracted JSON only if debug mode is on
        if debug:
            print("Extracted JSON:")
            print(json.dumps(valid_triples, indent=2))  # Pretty print the JSON

        return valid_triples
    else:
        # Always print error messages even if debug is off
        print("\n\nERROR ### Could not extract valid JSON from response: ", response, "\n\n")
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
    """
    Process a large text by breaking it into chunks with overlap,
    and then processing each chunk separately.

    Args:
        config: Configuration dictionary
        full_text: The complete text to process
        debug: If True, print detailed debug information
        continue_on_error: If True, skip chunks whose LLM call fails instead of aborting

    Returns:
        List of all extracted triples from all chunks

    Raises:
        LLMError: when a chunk fails and continue_on_error is False
    """
    # Get chunking parameters from config
    chunk_size = config.get("chunking", {}).get("chunk_size", 500)
    overlap = config.get("chunking", {}).get("overlap", 50)

    # Split text into chunks
    text_chunks = chunk_text(full_text, chunk_size, overlap)

    print("=" * 50)
    print("PHASE 1: INITIAL TRIPLE EXTRACTION")
    print("=" * 50)
    print(f"Processing text in {len(text_chunks)} chunks (size: {chunk_size} words, overlap: {overlap} words)")

    # Process chunks concurrently with a single shared client; results keep chunk order.
    client = LLMClient.from_config(config)
    concurrency = max(1, min(int(config.get("llm", {}).get("concurrency", 4)), len(text_chunks)))
    if concurrency > 1:
        print(f"Extracting with {concurrency} parallel requests", flush=True)

    def run_chunk(index_chunk):
        i, chunk = index_chunk
        print(f"Processing chunk {i+1}/{len(text_chunks)} ({len(chunk.split())} words)", flush=True)
        try:
            results = process_with_llm(config, chunk, debug, client=client)
        except LLMError as e:
            return i, None, e
        return i, results, None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(run_chunk, enumerate(text_chunks)))
    cache_hits = getattr(client, "cache_hits", 0)
    if cache_hits:
        print(f"Served {cache_hits} of {len(text_chunks)} chunks from the LLM cache ({client.cache_dir})", flush=True)

    all_results = []
    failed_chunks = []
    for i, chunk_results, error in outcomes:
        if error is not None:
            if not continue_on_error:
                raise LLMError(f"Chunk {i+1}/{len(text_chunks)} failed: {error}\n"
                               f"(Use --continue-on-error to skip failed chunks instead of aborting.)") from error
            print(f"Warning: skipping chunk {i+1}: {error}", flush=True)
            failed_chunks.append(i + 1)
            continue
        if chunk_results:
            for item in chunk_results:
                item["chunk"] = i + 1
            all_results.extend(chunk_results)
            print(f"Chunk {i+1}: {len(chunk_results)} triples", flush=True)
        else:
            print(f"Warning: Failed to extract triples from chunk {i+1}")

    print(f"\nExtracted a total of {len(all_results)} triples from all chunks", flush=True)
    if failed_chunks:
        print(f"Warning: {len(failed_chunks)} of {len(text_chunks)} chunks failed and were skipped: "
              f"{failed_chunks}. The graph is incomplete.", flush=True)
    if not all_results:
        return []

    # Apply entity standardization if enabled
    if config.get("standardization", {}).get("enabled", False):
        print("\n" + "="*50)
        print("PHASE 2: ENTITY STANDARDIZATION")
        print("="*50)
        print(f"Starting with {len(all_results)} triples and {len(get_unique_entities(all_results))} unique entities")

        all_results = standardize_entities(all_results, config)

        print(f"After standardization: {len(all_results)} triples and {len(get_unique_entities(all_results))} unique entities")

    # Apply relationship inference if enabled
    if config.get("inference", {}).get("enabled", False):
        print("\n" + "="*50)
        print("PHASE 3: RELATIONSHIP INFERENCE")
        print("="*50)
        print(f"Starting with {len(all_results)} triples")

        # Count existing relationships
        relationship_counts = {}
        for triple in all_results:
            relationship_counts[triple["predicate"]] = relationship_counts.get(triple["predicate"], 0) + 1

        print("Top 5 relationship types before inference:")
        for pred, count in sorted(relationship_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {pred}: {count} occurrences")

        all_results = infer_relationships(all_results, config)

        # Count relationships after inference
        relationship_counts_after = {}
        for triple in all_results:
            relationship_counts_after[triple["predicate"]] = relationship_counts_after.get(triple["predicate"], 0) + 1

        print("\nTop 5 relationship types after inference:")
        for pred, count in sorted(relationship_counts_after.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {pred}: {count} occurrences")

        # Count inferred relationships
        inferred_count = sum(1 for triple in all_results if triple.get("inferred", False))
        print(f"\nAdded {inferred_count} inferred relationships")
        print(f"Final knowledge graph: {len(all_results)} triples")

    return all_results

TEXT_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")  # utf-8-sig also reads plain UTF-8 and strips a BOM
BINARY_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".gz", ".png", ".jpg", ".jpeg", ".gif"}


class InputError(Exception):
    """Raised when the input file cannot be read as text."""


def read_input_text(path):
    """Read a text file trying several encodings; refuse binary formats with a clear message.

    Raises:
        InputError: with a message that explains what to do.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in BINARY_EXTENSIONS:
        raise InputError(f"{path} looks like a {ext} file. Only plain text is supported; "
                         f"convert it to .txt or .md first (for PDFs: `pdftotext file.pdf file.txt`).")
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
            print(f"Note: {path} decoded as {encoding}", flush=True)
        return text
    raise InputError(f"Could not decode {path} as text (tried {', '.join(TEXT_ENCODINGS)}).")


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


def make_community_namer(config):
    """Return a callable that asks the LLM for short community names, or None if disabled."""
    if not config.get("visualization", {}).get("name_communities", True):
        return None

    def namer(communities):
        lines = [f"{c['id'] + 1}. {', '.join(c['top'])}" for c in communities if c.get("top")]
        if len(lines) < 2:
            return {}
        system_prompt = prompt_factory.get_prompt("community_naming_system")
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
    parser.add_argument('--input', type=str, required=False, help='Path to input text file (required unless --test or --from-json is used)')
    parser.add_argument('--from-json', type=str, metavar='FILE',
                        help='Render a visualization from a previously saved triples JSON file (no LLM calls)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output (raw LLM responses and extracted JSON)')
    parser.add_argument('--no-standardize', action='store_true', help='Disable entity standardization')
    parser.add_argument('--no-inference', action='store_true', help='Disable relationship inference')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Skip chunks whose LLM call fails or is truncated instead of aborting')
    parser.add_argument('--no-cache', action='store_true',
                        help='Do not read or write the LLM response cache (llm.cache_dir)')

    args = parser.parse_args()

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

    # Re-render an existing graph without touching the LLM
    if args.from_json:
        try:
            triples = load_triples_from_json(args.from_json)
        except InputError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"Loaded {len(triples)} triples from {args.from_json}")
        stats = visualize_knowledge_graph(triples, args.output, config=config)
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

    # Load input text from file
    try:
        input_text = read_input_text(args.input)
        print(f"Using input text from file: {args.input}")
    except InputError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Process text in chunks
    try:
        result = process_text_in_chunks(config, input_text, args.debug, args.continue_on_error)
    except LLMError as e:
        print(f"\nERROR: {e}", flush=True)
        print("Knowledge graph generation aborted.")
        sys.exit(1)

    if result:
        # Save the raw data as JSON for potential reuse
        json_output = args.output.replace('.html', '.json')
        try:
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Saved raw knowledge graph data to {json_output}")
        except Exception as e:
            print(f"Warning: Could not save raw data to {json_output}: {e}")

        # Visualize the knowledge graph
        stats = visualize_knowledge_graph(result, args.output, config=config,
                                          community_namer=make_community_namer(config))
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
