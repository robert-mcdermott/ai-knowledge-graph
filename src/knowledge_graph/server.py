"""Optional local web interface: browse generated graphs, explore them, and chat with them.

Run with ``graph-serve`` (needs the ``[web]`` extra: FastAPI and uvicorn). It is a thin
layer over the same code the CLI uses: graphs are the ``.json`` files ``generate-graph``
writes, pages are rendered by :mod:`knowledge_graph.visualization`, and questions are
answered by :class:`knowledge_graph.query.GraphChat`. Single user, no accounts; it binds
to localhost by default because it wraps your configured LLM and reads local files.
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader

from knowledge_graph.config import load_config
from knowledge_graph.llm import LLMClient, LLMError
from knowledge_graph.logging_utils import ROOT_LOGGER, ConsoleFormatter, configure_logging
from knowledge_graph.main import (
    SUPPORTED_EXTENSIONS,
    InputError,
    RunContext,
    load_graph_meta,
    process_documents,
    read_documents,
    save_graph_meta,
    update_collection,
)
from knowledge_graph.profiles import PROFILES
from knowledge_graph.query import GraphChat
from knowledge_graph.visualization import (
    TEMPLATE_DIR,
    attach_workspace,
    build_graph_data,
    render_html,
    render_knowledge_graph,
)
from knowledge_graph.workspace import (
    WorkspaceError,
    apply_document_update,
    empty_workspace,
    load_workspace,
    preview_documents,
    projected_claims,
    save_workspace,
    set_correction,
    validate_view,
    workspace_path,
)

log = logging.getLogger("knowledge_graph.server")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]*$")


def _require_web_deps():
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as e:
        raise SystemExit("graph-serve needs the optional web dependencies: pip install 'ai-knowledge-graph[web]'") from e


class GraphStore:
    """The ``*.json`` graphs in a directory, with cached pages and chat sessions."""

    def __init__(self, graphs_dir, config):
        self.graphs_dir = os.path.abspath(graphs_dir)
        self.config = config
        self._pages: dict[str, tuple[float, str]] = {}
        self._chats: dict[str, GraphChat] = {}
        self._lock = threading.Lock()
        self.mutation_lock = threading.RLock()

    def list_graphs(self):
        entries = []
        if not os.path.isdir(self.graphs_dir):
            return entries
        for name in sorted(os.listdir(self.graphs_dir)):
            if not name.lower().endswith(".json") or name.startswith(".") or name.endswith((".workspace.json", ".meta.json")):
                continue
            path = os.path.join(self.graphs_dir, name)
            try:
                self.path_for(name[:-5])
                workspace = load_workspace(path)
                triples = projected_claims(workspace)
            except (InputError, WorkspaceError, OSError, ValueError, KeyError):
                continue  # not a triples file
            nodes = {t["subject"] for t in triples} | {t["object"] for t in triples}
            entries.append({
                "name": name[:-5],
                "file": name,
                "documents": len(workspace.get("documents", [])),
                "partial": any(d.get("partial") for d in workspace.get("documents", [])),
                "views": len(workspace.get("views", [])),
                "triples": len(triples),
                "nodes": len(nodes),
                "inferred": sum(1 for t in triples if t.get("inferred")),
                "modified": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M"),
            })
        return entries

    def path_for(self, name):
        if not _SAFE_NAME.match(name or "") or "/" in name or "\\" in name or ".." in name:
            raise KeyError(name)
        path = os.path.join(self.graphs_dir, f"{name}.json")
        if not os.path.isfile(path) or os.path.commonpath([os.path.realpath(path), os.path.realpath(self.graphs_dir)]) != os.path.realpath(self.graphs_dir):
            raise KeyError(name)
        for sidecar in (workspace_path(path), os.path.splitext(path)[0] + ".meta.json"):
            if os.path.lexists(sidecar) and os.path.commonpath([os.path.realpath(sidecar), os.path.realpath(self.graphs_dir)]) != os.path.realpath(self.graphs_dir):
                raise KeyError(name)
        return path

    def triples_for(self, name):
        return projected_claims(load_workspace(self.path_for(name)))

    def page_for(self, name):
        """Rendered explorer page for a graph (cached until the file changes)."""
        path = self.path_for(name)
        mtime = self.fingerprint(path)
        with self._lock:
            cached = self._pages.get(name)
            if cached and cached[0] == mtime:
                return cached[1]
        workspace = load_workspace(path)
        triples = projected_claims(workspace)
        vis = self.config.get("visualization", {})
        stored_names = load_graph_meta(path).get("community_names") or None
        graph_data = build_graph_data(triples, vis.get("edge_smooth", False), community_names=stored_names,
                                      show_inferred=vis.get("show_inferred", True),
                                      theme=vis.get("theme", "light"), edge_labels=vis.get("edge_labels", "all"),
                                      title_case=vis.get("title_case", True),
                                      collapse_parallel_edges=vis.get("collapse_parallel_edges", True))
        attach_workspace(graph_data, workspace)
        graph_data["meta"]["chatEndpoint"] = f"/api/chat/{quote(name)}"
        graph_data["meta"]["workspaceEndpoint"] = f"/api/workspaces/{quote(name)}"
        graph_data["meta"]["reviewEndpoint"] = f"/api/review/{quote(name)}"
        graph_data["meta"]["viewsEndpoint"] = f"/api/views/{quote(name)}"
        graph_data["meta"]["manageUrl"] = "/#collection=" + quote(name)
        graph_data["meta"]["libraryUrl"] = "/"
        html = render_html(graph_data)
        with self._lock:
            self._pages[name] = (mtime, html)
        return html

    @staticmethod
    def fingerprint(path):
        from knowledge_graph.workspace import workspace_path
        return tuple((os.stat(p).st_mtime_ns, os.stat(p).st_size) if os.path.exists(p) else None
                     for p in (path, workspace_path(path), os.path.splitext(path)[0] + ".meta.json"))

    def chat_for(self, name, session="default"):
        path = self.path_for(name)
        key = (name, session)
        stamp = self.fingerprint(path)
        with self._lock:
            cached = self._chats.get(key)
            if cached is None or cached[0] != stamp:
                chat = GraphChat(projected_claims(load_workspace(path)), self.config)
                self._chats[key] = (stamp, chat)
            if len(self._chats) > 100:
                self._chats.pop(next(iter(self._chats)))
            return self._chats[key][1]

    def reset_chat(self, name, session="default"):
        self.path_for(name)
        with self._lock:
            self._chats.pop((name, session), None)

    def forget(self, name):
        with self._lock:
            self._pages.pop(name, None)
            for key in list(self._chats):
                if key[0] == name:
                    self._chats.pop(key, None)

    def save(self, name, workspace, expected_revision=None):
        with self.mutation_lock:
            path = os.path.join(self.graphs_dir, f"{name}.json")
            if expected_revision is not None and load_workspace(path).get("revision", 0) != expected_revision:
                raise WorkspaceError("This collection changed. Create a fresh preview before applying the update.")
            triples = projected_claims(workspace)
            # Render to a temporary file before publishing any change.
            with tempfile.TemporaryDirectory(dir=self.graphs_dir, prefix=".render-") as tmp:
                html_path = os.path.join(tmp, "graph.html")
                rendered_workspace = dict(workspace, revision=workspace.get("revision", 0) + 1)
                stats, graph_data = render_knowledge_graph(triples, html_path, config=self.config, workspace=rendered_workspace)
                save_workspace(path, workspace)
                os.replace(html_path, os.path.splitext(path)[0] + ".html")
                save_graph_meta(path, graph_data, self.config)
            self.forget(name)
            return stats

    def unique_name(self, wanted):
        base = re.sub(r"[^A-Za-z0-9._ -]+", "-", (wanted or "graph").strip()).strip(" .-") or "graph"
        base = re.sub(r"\.{2,}", "-", base)
        if base.endswith((".workspace", ".meta")):
            base += "-collection"
        name, n = base, 2
        while os.path.exists(os.path.join(self.graphs_dir, f"{name}.json")):
            name = f"{base}-{n}"
            n += 1
        return name


# --------------------------------------------------------------------------- #
# Ingest jobs (one at a time)
# --------------------------------------------------------------------------- #
_PHASES = (
    ("PHASE 1", "extracting"),
    ("PHASE 2", "standardizing"),
    ("PHASE 3", "inferring"),
    ("triples for visualization", "rendering"),
)


class _JobLogHandler(logging.Handler):
    """Records the pipeline's log lines for the job status page (console output is unaffected)."""

    def __init__(self, job):
        super().__init__(logging.INFO)
        self.job = job

    def emit(self, record):
        try:
            self.job.record(self.format(record))
        except Exception:  # never let status bookkeeping break the pipeline
            pass


class IngestJob:
    def __init__(self, name, documents, store, target=None, remove=(), profile=None, strict=False):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.documents = documents
        self.store = store
        self.target = target
        self.remove = list(remove)
        self.config = copy.deepcopy(store.config)
        if profile:
            self.config.setdefault("extraction", {})["profile"] = profile
        if strict:
            self.config.setdefault("extraction", {})["strict_evidence"] = True
        self.run_context = RunContext(callback=self.progress)
        self.pending = None
        self.diff = None
        self.base_revision = None
        self.state = "queued"  # queued | running | done | error
        self.phase = "starting"
        self.log: list[str] = []
        self.chunks_total = 0
        self.chunks_done = 0
        self.error = None
        self.stats = None
        self.started = datetime.now()
        self.finished = None
        self._lock = threading.Lock()

    def progress(self, snapshot):
        with self._lock:
            self.chunks_total = snapshot["chunks_total"]
            self.chunks_done = snapshot["chunks_done"]

    def record(self, line):
        with self._lock:
            self.log.append(line)
            for marker, phase in _PHASES:
                if marker in line:
                    self.phase = phase
            m = re.search(r"in (\d+) chunks", line)
            if m:
                self.chunks_total = int(m.group(1))
            if re.match(r"Chunk \d+: \d+ triples", line) or "skipping chunk" in line:
                self.chunks_done = max(self.chunks_done, self.run_context.completed)

    def status(self):
        with self._lock:
            return {
                "id": self.id, "name": self.name, "state": self.state, "phase": self.phase,
                "chunks_total": self.chunks_total, "chunks_done": self.chunks_done,
                "log": self.log[-40:], "error": self.error, "stats": self.stats,
                "documents": [d for d, _ in self.documents],
                "graph_url": f"/graph/{quote(self.name)}" if self.state in ("done", "partial") else None,
                "run": self.run_context.snapshot(), "diff": self.diff, "target": self.target,
                "elapsed_seconds": round(time.monotonic() - self.run_context.started, 1),
            }

    def run(self):
        self.state = "running"
        handler = _JobLogHandler(self)
        handler.setFormatter(ConsoleFormatter())
        pipeline_logger = logging.getLogger(ROOT_LOGGER)
        pipeline_logger.addHandler(handler)
        try:
            if self.target:
                workspace = load_workspace(self.store.path_for(self.target))
                self.base_revision = workspace.get("revision", 0)
                self.pending, self.diff = update_collection(workspace, self.documents, self.config, self.remove,
                                                            self.run_context, continue_on_error=True)
                self.stats = build_graph_data(projected_claims(self.pending))["meta"]["stats"]
                self.state, self.phase = "ready", "ready"
            else:
                extraction_config = copy.deepcopy(self.config)
                extraction_config.setdefault("standardization", {})["enabled"] = False
                triples = process_documents(extraction_config, self.documents, continue_on_error=True, run=self.run_context)
                self.run_context.check()
                if not triples:
                    raise InputError("No relationships could be extracted from the input.")
                workspace, self.diff = apply_document_update(empty_workspace(self.name), self.documents, triples,
                                                               run=self.run_context.snapshot())
                workspace["inferred"] = [t for t in triples if t.get("inferred")]
                with self.store.mutation_lock:
                    self.run_context.check()
                    self.stats = self.store.save(self.name, workspace)
                self.state = "partial" if self.run_context.failures else "done"
                self.phase = self.state
        except Exception as e:  # surfaced to the user in the status page
            self.error = str(e)
            self.state = "cancelled" if self.run_context.cancel.is_set() else "error"
            self.phase = self.state
        finally:
            pipeline_logger.removeHandler(handler)
            self.finished = datetime.now()


class JobManager:
    def __init__(self, store):
        self.store = store
        self.jobs: dict[str, IngestJob] = {}
        self._lock = threading.Lock()

    def active(self):
        return next((j for j in list(self.jobs.values()) if j.state in ("queued", "running", "ready")), None)

    def start(self, name, documents, background=True, target=None, remove=(), profile=None, strict=False):
        with self._lock:
            if self.active() is not None:
                raise RuntimeError("Another graph is being generated; wait for it to finish.")
            if target:
                workspace = load_workspace(self.store.path_for(target))
                preview_documents(workspace, documents, remove)
            job = IngestJob(target or self.store.unique_name(name), documents, self.store, target, remove, profile, strict)
            if len(self.jobs) >= 100:
                self.jobs.pop(next(iter(self.jobs)))
            self.jobs[job.id] = job
        if background:
            threading.Thread(target=job.run, name=f"ingest-{job.id}", daemon=True).start()
        else:
            job.run()
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)


def documents_from_upload(text, uploads):
    """Build (name, text) pairs from pasted text and uploaded (filename, bytes) pairs.

    Uploads are written to a temporary directory so the normal format readers handle them.
    """
    documents = []
    if text and text.strip():
        documents.append(("pasted text", text))
    if uploads:
        with tempfile.TemporaryDirectory(prefix="kg-upload-") as tmp:
            paths = []
            seen_names = set()
            for filename, data in uploads:
                safe = os.path.basename(filename or "upload.txt") or "upload.txt"
                ext = os.path.splitext(safe)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    raise InputError(f"Unsupported file type {ext or '(none)'} for {safe}. "
                                     f"Use {', '.join(sorted(e for e in SUPPORTED_EXTENSIONS if e))}, .pdf or .docx.")
                if safe in seen_names:
                    raise InputError("Uploaded filenames must be unique; rename duplicates first.")
                seen_names.add(safe)
                path = os.path.join(tmp, safe)
                with open(path, "wb") as f:
                    f.write(data)
                paths.append(path)
            documents.extend(read_documents(paths))
    if not documents:
        raise InputError("Paste some text or upload at least one file.")
    return documents


def create_app(config, graphs_dir):
    """Build the FastAPI application."""
    _require_web_deps()
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse
    from starlette.requests import Request

    os.makedirs(graphs_dir, exist_ok=True)
    store = GraphStore(graphs_dir, config)
    jobs = JobManager(store)
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)
    app = FastAPI(title="AI Knowledge Graph", docs_url=None, redoc_url=None)
    app.state.store = store
    app.state.jobs = jobs

    @app.exception_handler(WorkspaceError)
    async def workspace_error(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(InputError)
    async def input_error(request, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            return JSONResponse(status_code=403, content={"detail": "Use the local application to make changes"})
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def library():
        template = env.get_template("library.html.j2")
        return template.render(graphs=store.list_graphs(), graphs_dir=store.graphs_dir,
                               model=config.get("llm", {}).get("model", ""), active_job=jobs.active(), profiles=PROFILES)

    @app.post("/api/ingest")
    async def api_ingest(name: str = Form(""), text: str = Form(""), files: list[UploadFile] = File(default=[]),  # noqa: B008
                         target: str = Form(""), remove: str = Form("[]"), profile: str = Form("general"),
                         dry_run: bool = Form(False), strict: bool = Form(False)):  # noqa: B008
        uploads = []
        for upload in files:
            if upload.filename:
                uploads.append((upload.filename, await upload.read()))
        try:
            if profile not in PROFILES:
                raise InputError("Unknown extraction profile")
            try:
                removed = json.loads(remove)
            except ValueError:
                raise InputError("remove must be a JSON list of document names") from None
            if not isinstance(removed, list) or any(not isinstance(n, str) for n in removed):
                raise InputError("remove must be a JSON list of document names")
            documents = documents_from_upload(text, uploads) if text.strip() or uploads else []
            if not documents and not (target and removed):
                raise InputError("Add text, upload documents, or choose documents to remove.")
            workspace = load_workspace(store.path_for(target)) if target else empty_workspace(name)
            diff = preview_documents(workspace, documents, removed)
            if dry_run:
                from knowledge_graph.workspace import document_record, passage_chunks
                changed = set(diff["added"] + diff["updated"])
                chunks = sum(len(passage_chunks(document_record(n, t)["passages"], config.get("chunking", {}).get("chunk_size", 500),
                                                config.get("chunking", {}).get("overlap", 50))) for n, t in documents if n in changed)
                return {"diff": diff, "chunks": chunks, "model": config["llm"]["model"]}
            if target and not any(diff[k] for k in ("added", "updated", "removed")):
                return {"unchanged": True, "page_url": f"/graph/{quote(target)}"}
            job = jobs.start(name, documents, target=target or None, remove=removed, profile=profile, strict=strict)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown collection") from None
        except InputError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from None
        return {"job_id": job.id, "name": job.name, "status_url": f"/api/jobs/{job.id}", "page_url": f"/jobs/{job.id}"}

    @app.post("/api/connection")
    def check_connection():
        try:
            client = LLMClient.from_config(config)
            client.cache_dir = None
            client.timeout = min(client.timeout, 30)
            client.max_retries = 0
            client.max_tokens = min(client.max_tokens, 2048)
            client.complete('Return {"ok": true}.', "Connection check; return JSON.")
            return {"ok": True, "model": client.model}
        except LLMError as e:
            return JSONResponse(status_code=502, content={"detail": str(e)})

    @app.post("/api/preview")
    def extraction_preview(body: dict):
        text = str(body.get("text", ""))[:2000].strip()
        if not text:
            raise HTTPException(status_code=400, detail="Paste some text to preview extraction")
        scoped = copy.deepcopy(config)
        profile = body.get("profile", "general")
        if profile not in PROFILES:
            raise HTTPException(status_code=400, detail="Unknown profile")
        scoped.setdefault("extraction", {})["profile"] = profile
        scoped.setdefault("inference", {})["enabled"] = False
        scoped.setdefault("standardization", {})["enabled"] = False
        run = RunContext()
        try:
            triples = process_documents(scoped, [("Preview", text)], run=run)
            return {"triples": triples, "run": run.snapshot()}
        except LLMError as e:
            return JSONResponse(status_code=502, content={"detail": str(e)})

    def workspace_for(name):
        try:
            return load_workspace(store.path_for(name))
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown collection") from None

    @app.get("/api/workspaces/{name}")
    def get_workspace(name: str):
        return workspace_for(name)

    @app.post("/api/review/{name}")
    def review(name: str, body: dict):
        with store.mutation_lock:
            workspace = workspace_for(name)
            if body.get("revision") != workspace.get("revision", 0):
                raise WorkspaceError("The collection changed; reload before reviewing")
            set_correction(workspace, body.get("action"), body)
            store.save(name, workspace)
            return {"ok": True, "revision": workspace["revision"]}

    @app.post("/api/views/{name}")
    def save_view(name: str, body: dict):
        with store.mutation_lock:
            workspace = workspace_for(name)
            if body.get("revision") != workspace.get("revision", 0):
                raise WorkspaceError("The collection changed; reload before saving a view")
            views = workspace.setdefault("views", [])
            if body.get("delete"):
                workspace["views"] = [v for v in views if v.get("id") != body["delete"]]
            else:
                view = body.get("view")
                validate_view(view)
                if len(json.dumps(view)) > 50000 or len(views) >= 100:
                    raise WorkspaceError("View storage limit reached")
                view["title"] = view["title"][:120]
                view["id"] = uuid.uuid4().hex[:12]
                views.append(view)
            store.save(name, workspace)
            return {"views": workspace["views"], "revision": workspace["revision"]}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        with store.mutation_lock:
            if job.state in ("queued", "running", "ready"):
                job.run_context.cancel.set()
                if job.state == "ready":
                    job.pending = None
                    job.state = job.phase = "cancelled"
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/apply")
    def apply_job(job_id: str):
        job = jobs.get(job_id)
        with store.mutation_lock:
            if job is None or job.state != "ready" or job.pending is None:
                raise HTTPException(status_code=409, detail="No pending change to apply")
            job.stats = store.save(job.name, job.pending, job.base_revision)
            job.pending = None
            job.state = job.phase = "partial" if job.run_context.failures else "done"
        return job.status()

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str):
        old = jobs.get(job_id)
        if old is None or old.state not in ("error", "partial", "cancelled"):
            raise HTTPException(status_code=409, detail="This job does not need a retry")
        target = old.name if old.state == "partial" else old.target
        try:
            job = jobs.start(old.name, old.documents, target=target, remove=old.remove,
                             profile=old.config.get("extraction", {}).get("profile", "general"),
                             strict=old.config.get("extraction", {}).get("strict_evidence", False))
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from None
        return {"page_url": f"/jobs/{job.id}"}

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return job.status()

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return env.get_template("job.html.j2").render(job=job.status())

    @app.get("/api/graphs")
    def api_graphs():
        return {"graphs_dir": store.graphs_dir, "graphs": store.list_graphs()}

    @app.get("/graph/{name}", response_class=HTMLResponse)
    def graph_page(name: str):
        try:
            return store.page_for(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"No graph named {name!r} in {store.graphs_dir}") from None

    @app.post("/api/chat/{name}")
    def api_chat(name: str, body: dict):
        question = str(body.get("question", "")).strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        try:
            session = str(body.get("session", "default"))[:80]
            chat = store.chat_for(name, session)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"No graph named {name!r}") from None
        try:
            result = chat.ask(question, extracted_only=bool(body.get("extracted_only", False)))
        except LLMError as e:
            return JSONResponse(status_code=502, content={"error": str(e)})
        return result

    @app.post("/api/chat/{name}/reset")
    def api_chat_reset(name: str, body: dict = None):
        try:
            store.reset_chat(name, str((body or {}).get("session", "default"))[:80])
        except KeyError:
            raise HTTPException(status_code=404, detail=f"No graph named {name!r}") from None
        return {"ok": True}

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(prog="graph-serve",
                                     description="Local web interface to browse, explore and chat with generated graphs")
    parser.add_argument("--config", default="config.toml", help="Path to configuration file (uses [llm], [query], [visualization])")
    parser.add_argument("--graphs", default="out", metavar="DIR", help="Directory containing the .json graphs written by generate-graph")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1; the server has no authentication)")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--open", action="store_true", help="Open the library in your browser")
    args = parser.parse_args(argv)

    configure_logging(logging.INFO)
    config = load_config(args.config)
    if not config:
        sys.exit(1)
    os.makedirs(args.graphs, exist_ok=True)
    _require_web_deps()
    import uvicorn

    app = create_app(config, args.graphs)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving graphs from {os.path.abspath(args.graphs)} at {url}  (Ctrl-C to stop)")
    if args.open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
