"""Make ``src/`` (the package) and ``tests/`` (shared fakes) importable regardless of how pytest is launched."""
import logging
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for path in (SRC, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(autouse=True)
def restore_console_handlers():
    logger = logging.getLogger("knowledge_graph")
    before = list(logger.handlers)
    yield
    for handler in list(logger.handlers):
        if handler not in before and getattr(handler, "_kg_console_handler", False):
            logger.removeHandler(handler)
