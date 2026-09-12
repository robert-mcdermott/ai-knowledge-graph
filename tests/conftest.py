"""Make ``src/`` (the package) and ``tests/`` (shared fakes) importable regardless of how pytest is launched."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for path in (SRC, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)
