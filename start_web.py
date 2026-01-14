#!/usr/bin/env python3
"""Start the Knowledge Graph Web Interface."""

import sys
import subprocess
from pathlib import Path


def main():
    """Start the web server."""
    print("=" * 60)
    print("🚀 Starting AI Knowledge Graph Web Interface")
    print("=" * 60)
    print()
    print("📝 The web interface will be available at:")
    print("   http://localhost:8000")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    # Get the project root directory
    project_root = Path(__file__).parent

    # Start uvicorn server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "src.knowledge_graph.api.app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], cwd=project_root, check=True)
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped successfully")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
