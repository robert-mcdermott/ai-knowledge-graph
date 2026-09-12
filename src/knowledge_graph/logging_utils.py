"""Console logging for the command-line tools.

The pipeline modules log through ``logging.getLogger("knowledge_graph.<module>")``.
Library users see nothing unless they configure logging; the CLI commands call
:func:`configure_logging` so the console shows the familiar progress output.
"""
from __future__ import annotations

import logging
import sys

ROOT_LOGGER = "knowledge_graph"
_HANDLER_FLAG = "_kg_console_handler"


class ConsoleFormatter(logging.Formatter):
    """Plain messages; warnings and errors get a prefix unless they already carry one."""

    def format(self, record):
        message = record.getMessage()
        if record.levelno >= logging.ERROR and not message.lower().startswith(("error", "\nerror")):
            return f"Error: {message}"
        if record.levelno == logging.WARNING and not message.lower().startswith("warning"):
            return f"Warning: {message}"
        return message


def configure_logging(level=logging.INFO, stream=None):
    """Send ``knowledge_graph`` log output to the console at ``level`` (idempotent)."""
    logger = logging.getLogger(ROOT_LOGGER)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_FLAG, False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(ConsoleFormatter())
    setattr(handler, _HANDLER_FLAG, True)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def level_from_flags(verbose=False, quiet=False, debug=False):
    if debug or verbose:
        return logging.DEBUG
    if quiet:
        return logging.WARNING
    return logging.INFO
