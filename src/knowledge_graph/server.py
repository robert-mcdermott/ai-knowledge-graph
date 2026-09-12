"""Optional local web interface: browse generated graphs, explore them, and chat with them.

Run with ``graph-serve`` (needs the ``[web]`` extra: FastAPI and uvicorn). It is a thin
layer over the same code the CLI uses: graphs are the ``.json`` files ``generate-graph``
writes, pages are rendered by :mod:`knowledge_graph.visualization`, and questions are
answered by :class:`knowledge_graph.query.GraphChat`. Single user, no accounts; it binds
to localhost by default because it wraps your configured LLM and reads local files.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import webbrowser
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

from knowledge_graph.config import load_config
from knowledge_graph.llm import LLMError
from knowledge_graph.main import InputError, load_triples_from_json, make_community_namer
from knowledge_graph.query import GraphChat
from knowledge_graph.visualization import TEMPLATE_DIR, build_graph_data, render_html

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

    def list_graphs(self):
        entries = []
        if not os.path.isdir(self.graphs_dir):
            return entries
        for name in sorted(os.listdir(self.graphs_dir)):
            if not name.lower().endswith(".json") or name.startswith("."):
                continue
            path = os.path.join(self.graphs_dir, name)
            try:
                triples = load_triples_from_json(path)
            except InputError:
                continue  # not a triples file
            nodes = {t["subject"] for t in triples} | {t["object"] for t in triples}
            entries.append({
                "name": name[:-5],
                "file": name,
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
        if not os.path.isfile(path):
            raise KeyError(name)
        return path

    def triples_for(self, name):
        return load_triples_from_json(self.path_for(name))

    def page_for(self, name):
        """Rendered explorer page for a graph (cached until the file changes)."""
        path = self.path_for(name)
        mtime = os.path.getmtime(path)
        with self._lock:
            cached = self._pages.get(name)
            if cached and cached[0] == mtime:
                return cached[1]
        triples = load_triples_from_json(path)
        vis = self.config.get("visualization", {})
        graph_data = build_graph_data(triples, vis.get("edge_smooth", False), show_inferred=vis.get("show_inferred", True),
                                      theme=vis.get("theme", "light"), edge_labels=vis.get("edge_labels", "all"))
        namer = make_community_namer(self.config)
        if namer is not None and graph_data["meta"]["stats"]["communities"] > 1:
            try:
                names = namer(graph_data["meta"]["communities"]) or {}
                for entry in graph_data["meta"]["communities"]:
                    if entry["id"] in names:
                        entry["name"] = names[entry["id"]]
            except Exception as e:  # cosmetic; never block the page
                print(f"Warning: community naming failed: {e}", flush=True)
        graph_data["meta"]["chatEndpoint"] = f"/api/chat/{name}"
        graph_data["meta"]["libraryUrl"] = "/"
        html = render_html(graph_data)
        with self._lock:
            self._pages[name] = (mtime, html)
        return html

    def chat_for(self, name):
        path = self.path_for(name)
        with self._lock:
            chat = self._chats.get(name)
            if chat is None:
                chat = GraphChat(load_triples_from_json(path), self.config)
                self._chats[name] = chat
            return chat

    def reset_chat(self, name):
        self.path_for(name)
        with self._lock:
            self._chats.pop(name, None)


def create_app(config, graphs_dir):
    """Build the FastAPI application."""
    _require_web_deps()
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse

    store = GraphStore(graphs_dir, config)
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)
    app = FastAPI(title="AI Knowledge Graph", docs_url=None, redoc_url=None)
    app.state.store = store

    @app.get("/", response_class=HTMLResponse)
    def library():
        template = env.get_template("library.html.j2")
        return template.render(graphs=store.list_graphs(), graphs_dir=store.graphs_dir,
                               model=config.get("llm", {}).get("model", ""))

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
            chat = store.chat_for(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"No graph named {name!r}") from None
        try:
            result = chat.ask(question)
        except LLMError as e:
            return JSONResponse(status_code=502, content={"error": str(e)})
        return result

    @app.post("/api/chat/{name}/reset")
    def api_chat_reset(name: str):
        try:
            store.reset_chat(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"No graph named {name!r}") from None
        return {"ok": True}

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(prog="graph-serve",
                                     description="Local web interface to browse, explore and chat with generated graphs")
    parser.add_argument("--config", default="config.toml", help="Path to configuration file (uses [llm], [query], [visualization])")
    parser.add_argument("--graphs", default=".", metavar="DIR", help="Directory containing the .json graphs written by generate-graph")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1; the server has no authentication)")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--open", action="store_true", help="Open the library in your browser")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if not config:
        sys.exit(1)
    if not os.path.isdir(args.graphs):
        print(f"Error: {args.graphs} is not a directory")
        sys.exit(1)
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
