"""FastAPI application for Knowledge Graph Generator."""

import asyncio
import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from knowledge_graph.config import load_config
from knowledge_graph.main import process_text_in_chunks
from knowledge_graph.text_utils import chunk_text
from knowledge_graph.visualization import visualize_knowledge_graph

app = FastAPI(title="AI Knowledge Graph Generator", version="0.6.1")

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
web_dir = Path(__file__).parent.parent.parent.parent / "web"
app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")

# Store active processing tasks
active_tasks: Dict[str, Dict] = {}


class ProcessingConfig(BaseModel):
    """Configuration for text processing."""
    chunk_size: int = 200
    overlap: int = 20
    enable_standardization: bool = True
    enable_inference: bool = True


class ProgressUpdate:
    """Progress update tracker for processing."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.websocket: Optional[WebSocket] = None

    async def send_update(self, phase: str, message: str, progress: float):
        """Send progress update via WebSocket."""
        if self.websocket:
            try:
                await self.websocket.send_json({
                    "phase": phase,
                    "message": message,
                    "progress": progress,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception:
                pass


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    html_file = web_dir / "templates" / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return HTMLResponse(content="<h1>Knowledge Graph Generator</h1><p>Upload interface not found.</p>")


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    chunk_size: int = 200,
    overlap: int = 20,
    enable_standardization: bool = True,
    enable_inference: bool = True
):
    """
    Upload a text file for processing.

    Returns a task_id for tracking progress.
    """
    # Validate file type
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(status_code=400, detail="Only .txt and .md files are supported")

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Save uploaded file temporarily
    temp_dir = Path(tempfile.gettempdir()) / "kg_uploads"
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / f"{task_id}_{file.filename}"

    try:
        content = await file.read()
        with open(temp_file, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Store task info
    active_tasks[task_id] = {
        "id": task_id,
        "filename": file.filename,
        "filepath": str(temp_file),
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "config": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "enable_standardization": enable_standardization,
            "enable_inference": enable_inference
        }
    }

    return {
        "task_id": task_id,
        "filename": file.filename,
        "status": "queued",
        "message": "File uploaded successfully. Connect to WebSocket to start processing."
    }


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time progress updates."""
    await websocket.accept()

    if task_id not in active_tasks:
        await websocket.send_json({"error": "Task not found"})
        await websocket.close()
        return

    task = active_tasks[task_id]
    progress = ProgressUpdate(task_id)
    progress.websocket = websocket

    try:
        # Update status
        task["status"] = "processing"
        await progress.send_update("init", "Starting processing...", 0)

        # Load config
        config = load_config()

        # Override config with user settings
        user_config = task["config"]
        config["chunking"]["chunk_size"] = user_config["chunk_size"]
        config["chunking"]["overlap"] = user_config["overlap"]
        config["standardization"]["enabled"] = user_config["enable_standardization"]
        config["inference"]["enabled"] = user_config["enable_inference"]

        # Read text file
        await progress.send_update("reading", f"Reading file: {task['filename']}", 10)
        with open(task["filepath"], "r", encoding="utf-8") as f:
            text = f.read()

        # Create output directory
        output_dir = Path(tempfile.gettempdir()) / "kg_outputs" / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process text
        await progress.send_update("chunking", "Splitting text into chunks...", 20)
        chunks = chunk_text(text, config["chunking"]["chunk_size"], config["chunking"]["overlap"])

        await progress.send_update("extraction", f"Extracting knowledge from {len(chunks)} chunks...", 30)

        # Process chunks (this will take time)
        triples = await asyncio.to_thread(
            process_text_in_chunks,
            text,
            config,
            debug=False
        )

        await progress.send_update("processing", f"Extracted {len(triples)} relationships", 70)

        # Generate visualization
        await progress.send_update("visualizing", "Creating interactive visualization...", 80)

        output_html = output_dir / "graph.html"
        output_json = output_dir / "graph.json"

        await asyncio.to_thread(
            visualize_knowledge_graph,
            triples,
            str(output_html),
            config
        )

        # Save JSON data
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(triples, f, indent=2, ensure_ascii=False)

        # Calculate statistics
        nodes = set()
        for triple in triples:
            nodes.add(triple.get("subject"))
            nodes.add(triple.get("object"))

        stats = {
            "nodes": len(nodes),
            "edges": len(triples),
            "original_edges": len([t for t in triples if not t.get("inferred", False)]),
            "inferred_edges": len([t for t in triples if t.get("inferred", False)])
        }

        # Update task
        task["status"] = "completed"
        task["completed_at"] = datetime.now().isoformat()
        task["output_html"] = str(output_html)
        task["output_json"] = str(output_json)
        task["stats"] = stats

        await progress.send_update("completed", "Knowledge graph generated successfully!", 100)
        await websocket.send_json({
            "status": "completed",
            "stats": stats,
            "download_url": f"/api/download/{task_id}"
        })

    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        await websocket.send_json({
            "status": "error",
            "message": f"Processing failed: {str(e)}"
        })

    finally:
        await websocket.close()


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a processing task."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = active_tasks[task_id]
    return {
        "task_id": task_id,
        "filename": task["filename"],
        "status": task["status"],
        "created_at": task["created_at"],
        "stats": task.get("stats")
    }


@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    """Download the generated knowledge graph HTML."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = active_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")

    html_path = Path(task["output_html"])
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        html_path,
        media_type="text/html",
        filename=f"knowledge_graph_{task['filename']}.html"
    )


@app.get("/api/view/{task_id}")
async def view_result(task_id: str):
    """View the generated knowledge graph in browser."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = active_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")

    html_path = Path(task["output_html"])
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(html_path, media_type="text/html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.6.1"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
