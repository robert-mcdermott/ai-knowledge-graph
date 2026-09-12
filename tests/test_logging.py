import io
import logging

from knowledge_graph.logging_utils import ConsoleFormatter, configure_logging, level_from_flags


def test_formatter_prefixes_warnings_and_errors_once():
    f = ConsoleFormatter()
    rec = lambda lvl, msg: logging.LogRecord("knowledge_graph.x", lvl, "", 0, msg, None, None)  # noqa: E731
    assert f.format(rec(logging.INFO, "Processing chunk 1/3")) == "Processing chunk 1/3"
    assert f.format(rec(logging.WARNING, "skipping chunk 2")) == "Warning: skipping chunk 2"
    assert f.format(rec(logging.WARNING, "Warning: already prefixed")) == "Warning: already prefixed"
    assert f.format(rec(logging.ERROR, "boom")) == "Error: boom"


def test_configure_logging_is_idempotent_and_levels():
    stream = io.StringIO()
    configure_logging(logging.INFO, stream)
    configure_logging(logging.INFO, stream)
    logger = logging.getLogger("knowledge_graph")
    assert sum(1 for h in logger.handlers if getattr(h, "_kg_console_handler", False)) == 1
    logging.getLogger("knowledge_graph.main").info("hello")
    logging.getLogger("knowledge_graph.main").debug("hidden")
    assert stream.getvalue() == "hello\n"
    configure_logging(logging.WARNING, stream)
    logging.getLogger("knowledge_graph.main").info("quiet")
    assert "quiet" not in stream.getvalue()


def test_level_from_flags():
    assert level_from_flags() == logging.INFO
    assert level_from_flags(quiet=True) == logging.WARNING
    assert level_from_flags(verbose=True, quiet=True) == logging.DEBUG
    assert level_from_flags(debug=True) == logging.DEBUG
