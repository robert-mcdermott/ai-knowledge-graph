# Working with knowledge collections

A collection combines source documents, extracted claims, optional inferences, review decisions, and saved views. It lives in a JSON sidecar alongside the existing flat triples export. No service besides the optional local web app is needed.

## Add, revise, and remove sources

Start `graph-serve --graphs out --open`. Choose a new collection or select an existing one in **Collection**. Add text or files and choose **Preview changes** to see which documents are added, changed, unchanged, or removed, plus the planned extraction chunk count. This step makes no model calls.

For an existing collection, **Prepare update for review** extracts only added or changed documents. The resulting page shows claim counts and up to 20 added/removed examples. **Apply update** publishes the result; **Discard update** leaves the collection alone. If another edit changes the collection revision first, applying the stale update is refused.

A filename identifies a source within the collection. Upload a revised file with the same name to replace its evidence. Use a different name to add a different source. Pasted text uses the name `pasted text`, so subsequent pasted text updates that source. Duplicate uploaded filenames are rejected. CLI directory input with duplicate basenames uses absolute paths as the names; keep paths stable, or give files unique names before import.

Removing a source deletes its evidence from claims. A claim remains if another source still supports it. Claims from legacy imports without identifiable source documents cannot be removed by removing a newly added document; reject those claims in Review instead. Existing inferences are discarded and, if enabled, recomputed after a document change.

Renaming a source is an add plus a remove. A content hash skips unchanged documents. Changing extraction settings does not force unchanged sources to be re-extracted: create a separate collection to compare profiles without replacing your original.

## Correct a claim or entity

Select an entity and open a source passage to inspect its claim. **Reject this claim** hides it from the active graph and future exports. **Review → Restore** brings it back. Rejection applies to the claim identity, including its polarity, time, and attribution; it also applies if that same claim appears again in a later document update.

**Review → Merge entities** redirects one entity to a chosen existing name. The source name becomes a searchable alias. **Undo merge** restores the original identities. Merge cycles are rejected. These are explicit review operations; the app does not infer that two similarly named people are identical.

Collections preserve original extracted names and use these reversible merges. The ordinary single-run CLI still supports automatic entity standardization; that preprocessing is separate from the reversible review layer. Re-rendering preserves workspace decisions. Re-running ordinary generation to the same output also carries forward decisions and saved views, though a changed extraction may produce new claim IDs requiring a new review.

## Evidence and qualifiers

Each new source has a stable document ID and a content hash. Passages have IDs derived from the document, exact text, and start offset. Extraction receives local labels such as `P1`; returned labels are resolved to the retained passage records. Offsets count Unicode characters in extracted text, not bytes in the original file. PDF form feeds preserve physical page numbering, including blank pages. DOCX and plain text have page 1; they do not pretend to recover a Word document's print pagination.

A claim can include `time`, `polarity`, and `attribution`. These fields qualify that relationship. Profiles ask the model to retain them where appropriate; their absence does not establish timelessness or certainty.

- **Source linked:** the referenced passage exists in the supplied text.
- **Unverified:** the model omitted a valid reference, or a legacy graph supplied only a source snippet.
- **Inferred:** a model or rule proposed a connection; its inference method is retained separately.

None of these labels independently proves the claim. The strict-reference option validates source linkage, not semantic entailment. A malformed triple or missing strict reference fails the whole affected chunk so a partially valid reply cannot silently look complete.

## Partial work, retries, and cancellation

Jobs show completed/total chunks, elapsed time, cache hits, extraction token usage when reported by the endpoint, and recent logs. Inference and optional community naming can add requests beyond the extraction token total. No dollar estimate is made without provider pricing.

A run that skips failed chunks is visibly partial, and affected documents retain a partial flag. Retry from the job page, or resubmit those documents to the collection. The document is processed again so successful and newly recovered evidence can be merged consistently; matching successful chunk requests reuse the cache when enabled. This is not a provider-side continuation of an interrupted request.

Cancellation stops further scheduled work and prevents publishing the result. An HTTP request already in flight may continue until its response or configured timeout; cancellation does not revoke a provider request or its usage. Jobs and pending previews live in server memory. After restarting the server, saved collections and run summaries remain; resubmit partial documents to retry. Unapplied previews must be prepared again.

## Saved views and stories

A saved view contains a title, note, selected entity, source/relationship/community/type filters, neighborhood focus, path destination, and camera position. Views are stored in the workspace when served, or in browser storage for standalone pages. A view referring to a removed entity restores the remaining available context.

**Play story** presents the views in saved order. **Export HTML** includes those views and their notes with the graph, sources linked to claims, and embedded visualization assets. Model endpoints are removed from the exported page. The HTML contains the whole active graph even when the initial view is filtered; to export only visible claims, use **Export → JSON/CSV**.

Saved notes are plain text. HTML exports contain source evidence and review metadata; inspect them before sharing. Browser-only views can disappear if browser storage is cleared, so export stories you want to keep.

## Storage and backup

Back up `collection.json`, `collection.workspace.json`, and `collection.meta.json` together. The workspace sidecar is authoritative. It retains extracted source text, original claims, review overrides, saved views, and the latest 20 run summaries. The flat JSON remains compatible with other tools but is a projection of the workspace.

Writes replace each JSON file atomically. Browser updates render to a temporary file before publication. This is not a multi-file database transaction: if interrupted between file replacements, re-render the authoritative workspace to refresh derived files. Do not run two server processes or simultaneous CLI writers against the same collection; revision checks coordinate edits within one server process.

Copying just a flat JSON file imports a legacy graph. Copying its sidecar preserves the complete workspace. Use **Review** to change active claims rather than editing the projection directly. Unsupported schema versions and malformed workspaces are rejected.
