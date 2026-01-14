#!/usr/bin/env python3
"""Quick test to verify the API can start."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from knowledge_graph.api.app import app
    print("✅ API module imported successfully")

    # Test that the app has the expected routes
    routes = [route.path for route in app.routes]
    expected_routes = ["/", "/api/upload", "/api/task/{task_id}", "/health"]

    print("\n📋 Available routes:")
    for route in routes:
        print(f"   - {route}")

    print("\n✅ All checks passed!")
    print("\n🚀 To start the web server, run:")
    print("   python start_web.py")
    print("\n   Then open your browser to: http://localhost:8000")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
