"""Centralized logging configuration for the sparq package, backed by loguru.

Configuring sinks happens once at import time (loguru's `logger` is a process-wide
singleton, so re-importing this module is a no-op after the first import).
"""

import sys
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

logger.remove()  # drop loguru's default handler so we control format/sinks explicitly

# Default value for the `node` extra field, so format strings referencing it never
# KeyError for log calls made outside a node's `contextualize(node=...)` block (e.g.
# before any graph node has run).
logger.configure(extra={"node": "-"})

logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <magenta>{extra[node]}</magenta> | <cyan>{name}</cyan> - <level>{message}</level>",
    enqueue=True,
)

_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[node]} | {name}:{function}:{line} - {message}"


@contextmanager
def run_log_context(run_dir: Path, run_id: str):
    """
    Route this run's log messages to their own file for the duration of the block.

    Binds `run_id` via loguru's `contextualize`, which is contextvar-based, so any
    `logger` call made anywhere in the call stack during this block is tagged with it
    and routed to this run's log file — including across `asyncio` tasks (contextvars
    are copied into each new Task) and thread-pool workers spawned via LangChain's
    `run_in_executor` (which explicitly copies the context before submitting).

    Does NOT cross real process boundaries: code running inside the REPL's
    `multiprocessing.spawn`'d execution subprocess is a fresh interpreter and won't
    see this binding or this sink.
    """
    log_path = Path(run_dir) / "log.txt"
    sink_id = logger.add(
        log_path,
        level="DEBUG",
        format=_FILE_FORMAT,
        filter=lambda record: record["extra"].get("run_id") == run_id,
        enqueue=True,
    )
    with logger.contextualize(run_id=run_id):
        try:
            yield
        finally:
            logger.remove(sink_id)


def get_logger(name: str):
    """
    Back-compat shim for existing `get_logger(__name__)` call sites.

    loguru has a single global `logger`; the calling module/file/line is captured
    automatically per call site, so `name` doesn't need to be bound to anything.
    """
    return logger
