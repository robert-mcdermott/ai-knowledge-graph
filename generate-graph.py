#!/usr/bin/env python3
"""Knowledge Graph Generator and Visualizer.

Convenience entry point for running from a source checkout without installing:

    python generate-graph.py --input file.txt --output graph.html

When the package is installed (``pip install -e .``) use the ``generate-graph`` command instead.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from knowledge_graph.main import main  # noqa: E402

if __name__ == "__main__":
    main()
