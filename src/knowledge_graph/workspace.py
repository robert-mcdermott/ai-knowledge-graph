"""Versioned, local graph workspaces and lossless evidence records.

The .json triples export remains compatible. A .workspace.json sidecar owns source
documents, raw claims, corrections and saved views; no database service is needed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1


class WorkspaceError(ValueError):
    pass


def identity(kind, *parts):
    data = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return kind + "_" + hashlib.sha256(data.encode()).hexdigest()[:20]


def claim_key(t):
    return tuple(str(t.get(k, "positive" if k == "polarity" else "")).strip().casefold() for k in
                 ("subject", "predicate", "object", "polarity", "time", "attribution"))


def claim_id(t):
    return identity("claim", *claim_key(t))


def validate_triples(triples):
    if not isinstance(triples, list):
        raise WorkspaceError("Expected a list of subject/predicate/object triples")
    for i, t in enumerate(triples):
        if not isinstance(t, dict) or any(not isinstance(t.get(k), str) or not t[k].strip()
                                          for k in ("subject", "predicate", "object")):
            raise WorkspaceError(f"Triple {i + 1}: subject, predicate and object must be nonempty strings")
        if "inferred" in t and not isinstance(t["inferred"], bool):
            raise WorkspaceError(f"Triple {i + 1}: inferred must be a boolean")
        for field in ("source", "document", "method", "via", "polarity", "time", "attribution"):
            if field in t and not isinstance(t[field], str):
                raise WorkspaceError(f"Triple {i + 1}: {field} must be a string")
        for field in ("subject_aliases", "object_aliases"):
            if field in t and (not isinstance(t[field], list) or any(not isinstance(a, str) for a in t[field])):
                raise WorkspaceError(f"Triple {i + 1}: {field} must be a list of strings")
        if "origins" in t and (not isinstance(t["origins"], list) or any(
            not isinstance(o, dict) or not isinstance(o.get("document_id"), str) or not isinstance(o.get("document", ""), str)
            for o in t["origins"]
        )):
            raise WorkspaceError(f"Triple {i + 1}: origins must contain document identifiers")
        if "evidence" in t:
            if not isinstance(t["evidence"], list) or any(
                not isinstance(e, dict) or not isinstance(e.get("text"), str) for e in t["evidence"]
            ):
                raise WorkspaceError(f"Triple {i + 1}: evidence must contain passage records with text")
    return triples


def evidence_for(t):
    if t.get("evidence"):
        return copy.deepcopy(t["evidence"])
    if t.get("source"):
        return [{"id": identity("legacy", t.get("document"), t["source"]),
                 "text": t["source"], "document": t.get("document", "Legacy source"),
                 "status": "unverified"}]
    return []


def origins_for(t):
    """Document membership also survives when a model omitted passage references."""
    origins = copy.deepcopy(t.get("origins", []))
    if not origins and t.get("document_id"):
        origins.append({"document_id": t["document_id"], "document": t.get("document", "")})
    for e in evidence_for(t):
        if e.get("document_id") and not any(o["document_id"] == e["document_id"] for o in origins):
            origins.append({"document_id": e["document_id"], "document": e.get("document", "")})
    return origins


def merge_claims(triples):
    """Keep qualified claims distinct and union their evidence, including overlap."""
    result = {}
    for item in triples:
        key = claim_key(item)
        if key not in result:
            result[key] = copy.deepcopy(item)
            continue
        previous = result[key]
        preferred = item if previous.get("inferred") and not item.get("inferred") else previous
        merged = copy.deepcopy(preferred)
        origins = {o["document_id"]: o for o in origins_for(previous) + origins_for(item)}
        if origins:
            merged["origins"] = list(origins.values())
        evidence = {e.get("id") or identity("evidence", e): e for e in evidence_for(previous) + evidence_for(item)}
        if evidence:
            merged["evidence"] = list(evidence.values())
            merged["source"] = merged["evidence"][0]["text"]
        for field in ("subject_aliases", "object_aliases"):
            aliases = sorted(set(previous.get(field, [])) | set(item.get(field, [])))
            if aliases:
                merged[field] = aliases
        result[key] = merged
    return list(result.values())


def passages(text, document_id="", name=""):
    """Exact source spans. Form feeds retain physical PDF page numbers, including blanks."""
    result = []
    # A passage may span multiple sentences; preserve the original text and offsets.
    pattern = r"[^\f]+?(?:[.!?。！？](?=\s|$)|[。！？]|\n\s*\n|(?=\f)|$)"
    for match in re.finditer(pattern, text, re.DOTALL):
        raw = match.group()
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.end() - (len(raw) - len(raw.rstrip()))
        if end <= start:
            continue
        # Bound unusually long paragraphs and languages without spaces.
        cursor = start
        while cursor < end:
            stop = min(end, cursor + 1800)
            if stop < end:
                boundary = text.rfind(" ", cursor + 900, stop)
                if boundary > cursor:
                    stop = boundary
            value = text[cursor:stop]
            result.append({"id": identity("passage", document_id, cursor, value), "text": value,
                           "start": cursor, "end": stop, "page": text.count("\f", 0, cursor) + 1,
                           "document_id": document_id, "document": name, "status": "source-linked"})
            cursor = stop
            while cursor < end and text[cursor].isspace():
                cursor += 1
    return result


def passage_chunks(records, size=500, overlap=50):
    """Pack exact passages with bounded overlap and split oversized units losslessly."""
    units = []
    for record in records:
        tokens = list(re.finditer(r"[\u3400-\u9fff]|[^\s\u3400-\u9fff]+", record["text"]))
        for start in range(0, len(tokens), size):
            batch = tokens[start:start + size]
            a, b = batch[0].start(), batch[-1].end()
            part = dict(record, text=record["text"][a:b], start=record["start"] + a, end=record["start"] + b)
            part["id"] = identity("passage", record["document_id"], part["start"], part["text"])
            units.append((part, len(batch)))
    chunks, current, count = [], [], 0
    for unit, weight in units:
        if current and count + weight > size:
            chunks.append([p for p, _ in current])
            carry, carry_count = [], 0
            for p, w in reversed(current):
                if carry_count + w > min(overlap, size // 2) or carry_count + w + weight > size:
                    break
                carry.insert(0, (p, w))
                carry_count += w
            current, count = carry, carry_count
        current.append((unit, weight))
        count += weight
    if current:
        chunks.append([p for p, _ in current])
    return chunks


def document_record(name, text):
    name = name or "Pasted text"
    doc_id = identity("doc", name)
    return {"id": doc_id, "name": name, "text": text, "hash": identity("content", text),
            "passages": passages(text, doc_id, name)}


def empty_workspace(title="Knowledge collection"):
    return {"schema_version": SCHEMA_VERSION, "title": title, "revision": 0,
            "documents": [], "claims": [], "inferred": [], "overrides": {"rejected": [], "merges": {}},
            "views": [], "runs": []}


def workspace_path(path):
    return str(Path(path).with_suffix(".workspace.json"))


def atomic_json(path, value):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".kg-", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_view(view):
    if not isinstance(view, dict) or not isinstance(view.get("title"), str) or not view["title"].strip():
        raise WorkspaceError("A view needs a title")
    if not isinstance(view.get("note", ""), str):
        raise WorkspaceError("A view note must be text")
    state = view.get("state", {})
    if not isinstance(state, dict):
        raise WorkspaceError("A view state must be an object")
    for field in ("hiddenCommunities", "hiddenTypes"):
        if field in state and (not isinstance(state[field], list) or any(not isinstance(v, (str, int)) for v in state[field])):
            raise WorkspaceError(f"View {field} must be a list")
    for field in ("selected", "pathTarget", "focusRoot", "document", "predicate", "edgeLabels"):
        if state.get(field) is not None and not isinstance(state[field], str):
            raise WorkspaceError(f"View {field} must be text")
    for field in ("scale", "focusHops", "minDegree"):
        if field in state and (not isinstance(state[field], (int, float)) or not math.isfinite(state[field])):
            raise WorkspaceError(f"View {field} must be a finite number")
    if "position" in state and (not isinstance(state["position"], dict) or any(
        not isinstance(state["position"].get(k), (int, float)) or not math.isfinite(state["position"][k]) for k in ("x", "y")
    )):
        raise WorkspaceError("View position needs finite x and y coordinates")


def validate_workspace(data):
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise WorkspaceError("Unsupported workspace schema version")
    validate_triples(data.get("claims"))
    validate_triples(data.get("inferred", []))
    if not isinstance(data.get("revision", 0), int) or data.get("revision", 0) < 0:
        raise WorkspaceError("Workspace revision must be a nonnegative integer")
    for field in ("documents", "views", "runs"):
        if not isinstance(data.get(field, []), list):
            raise WorkspaceError(f"Workspace {field} must be a list")
    names = set()
    for doc in data.get("documents", []):
        if not isinstance(doc, dict) or any(not isinstance(doc.get(k), str) for k in ("id", "name", "hash", "text")):
            raise WorkspaceError("Invalid workspace document")
        if doc["name"] in names:
            raise WorkspaceError("Workspace document names must be unique")
        names.add(doc["name"])
    for view in data.get("views", []):
        validate_view(view)
    overrides = data.get("overrides", {})
    if not isinstance(overrides, dict) or not isinstance(overrides.get("merges", {}), dict) or not isinstance(overrides.get("rejected", []), list):
        raise WorkspaceError("Invalid workspace corrections")
    if any(not isinstance(v, str) for v in overrides.get("rejected", [])) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in overrides.get("merges", {}).items()
    ):
        raise WorkspaceError("Invalid workspace correction identifiers")
    data.setdefault("documents", [])
    data.setdefault("views", [])
    data.setdefault("runs", [])
    data.setdefault("overrides", {"rejected": [], "merges": {}})
    data["overrides"].setdefault("rejected", [])
    data["overrides"].setdefault("merges", {})
    return data


def load_workspace(path):
    sidecar = workspace_path(path)
    if os.path.exists(sidecar):
        try:
            with open(sidecar, encoding="utf-8") as f:
                data = json.load(f)
            return validate_workspace(data)
        except (OSError, json.JSONDecodeError) as e:
            raise WorkspaceError(f"Cannot read workspace: {e}") from e
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "schema_version" in raw:
        return validate_workspace(raw)
    triples = validate_triples(raw.get("triples") if isinstance(raw, dict) else raw)
    data = empty_workspace(Path(path).stem)
    data["claims"] = [t for t in triples if not t.get("inferred")]
    data["inferred"] = [t for t in triples if t.get("inferred")]
    return data


def projected_claims(workspace, include_rejected=False):
    overrides = workspace.get("overrides", {})
    rejected = set(overrides.get("rejected", []))
    merges = overrides.get("merges", {})
    result = []
    for original in workspace["claims"] + workspace.get("inferred", []):
        t = copy.deepcopy(original)
        t["id"] = claim_id(original)
        t["rejected"] = t["id"] in rejected
        if t["rejected"] and not include_rejected:
            continue
        for field in ("subject", "object"):
            seen = set()
            while t[field] in merges and t[field] not in seen:
                seen.add(t[field])
                t[field] = merges[t[field]]
            if seen:
                t[field + "_aliases"] = sorted(seen | set(t.get(field + "_aliases", [])))
        if t["subject"] != t["object"]:
            result.append(t)
    return result


def save_workspace(path, workspace):
    workspace["revision"] = workspace.get("revision", 0) + 1
    workspace["updated"] = datetime.now(UTC).isoformat()
    atomic_json(workspace_path(path), workspace)
    atomic_json(path, projected_claims(workspace))


def preview_documents(workspace, documents, remove=()):
    previous = {d["name"]: d for d in workspace["documents"]}
    incoming = [document_record(name, text) for name, text in documents]
    names = [d["name"] for d in incoming]
    if len(set(names)) != len(names):
        raise WorkspaceError("Document names must be unique; rename same-named files before adding them")
    unknown = set(remove) - set(previous)
    if unknown:
        raise WorkspaceError(f"Unknown documents: {', '.join(sorted(unknown))}")
    if set(remove) & set(names):
        raise WorkspaceError("Cannot update and remove the same document in one change")
    added, updated, unchanged = [], [], []
    for doc in incoming:
        old = previous.get(doc["name"])
        target = added if old is None else unchanged if old["hash"] == doc["hash"] and not old.get("partial") else updated
        target.append(doc["name"])
    return {"added": added, "updated": updated, "unchanged": unchanged, "removed": list(remove)}


def apply_document_update(workspace, documents, extracted, remove=(), run=None):
    """Replace evidence for changed documents; never remove evidence belonging to others."""
    result = copy.deepcopy(workspace)
    diff = preview_documents(workspace, documents, remove)
    changed = set(diff["added"] + diff["updated"])
    removed = set(diff["removed"])
    if not changed and not removed:
        return result, diff
    replace_ids = {d["id"] for d in result["documents"] if d["name"] in changed | removed}
    kept = []
    for claim in result["claims"]:
        evidence = evidence_for(claim)
        surviving = [e for e in evidence if e.get("document_id") not in replace_ids]
        origins = origins_for(claim)
        remaining_origins = [o for o in origins if o["document_id"] not in replace_ids]
        if (evidence or origins) and not surviving and not remaining_origins:
            continue
        if origins:
            claim["origins"] = remaining_origins
        if surviving:
            claim["evidence"] = surviving
            claim["source"] = surviving[0]["text"]
            claim["document"] = surviving[0].get("document", "")
        elif evidence:
            claim["evidence"] = []
            claim.pop("source", None)
            claim["evidence_status"] = "unverified"
        if remaining_origins:
            claim.update(remaining_origins[0])
        kept.append(claim)
    result["claims"] = merge_claims(kept + [t for t in extracted if not t.get("inferred")])
    # Old inferences cannot claim freshness after their source graph changes.
    result["inferred"] = []
    result["documents"] = [d for d in result["documents"] if d["name"] not in changed | removed]
    for name, text in documents:
        if name in changed:
            doc = document_record(name, text)
            doc["partial"] = any(f.get("document") == name for f in (run or {}).get("failures", []))
            result["documents"].append(doc)
    if run:
        result["runs"] = (result.get("runs", []) + [run])[-20:]
    before = {claim_id(t) for t in workspace["claims"]}
    after = {claim_id(t) for t in result["claims"]}
    diff.update({"new_claims": len(after - before), "removed_claims": len(before - after),
                 "retained_claims": len(before & after),
                 "added_examples": [t for t in result["claims"] if claim_id(t) in after - before][:20],
                 "removed_examples": [t for t in workspace["claims"] if claim_id(t) in before - after][:20]})
    return result, diff


def set_correction(workspace, action, body):
    overrides = workspace.setdefault("overrides", {"rejected": [], "merges": {}})
    if action in ("reject", "restore"):
        cid = body.get("claim_id")
        if cid not in {claim_id(t) for t in workspace["claims"] + workspace.get("inferred", [])}:
            raise WorkspaceError("Unknown claim")
        rejected = set(overrides["rejected"])
        rejected.add(cid) if action == "reject" else rejected.discard(cid)
        overrides["rejected"] = sorted(rejected)
    elif action in ("merge", "unmerge"):
        source, target = body.get("source"), body.get("target")
        if action == "unmerge":
            overrides["merges"].pop(source, None)
            return
        names = {t[k] for t in workspace["claims"] for k in ("subject", "object")}
        if source not in names or target not in names or source == target:
            raise WorkspaceError("Choose two different existing entities")
        current, seen = target, {source}
        while current in overrides["merges"]:
            if current in seen:
                raise WorkspaceError("This merge would create an alias cycle")
            seen.add(current)
            current = overrides["merges"][current]
        if current in seen:
            raise WorkspaceError("This merge would create an alias cycle")
        overrides["merges"][source] = target
    else:
        raise WorkspaceError("Unknown correction action")
