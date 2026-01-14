#!/usr/bin/env python3
"""Main entry point for the web application."""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "src.knowledge_graph.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
