"""Application logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_HANDLER_MARKER = "_phenotype_network_handler"


def configure_logging(
    level: int | str = logging.INFO,
    log_file: Path | None = None,
    logger_name: str | None = None,
) -> logging.Logger:
    """Configure stderr and optional file logging without duplicate handlers."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_FORMAT)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    setattr(stream_handler, _HANDLER_MARKER, True)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_MARKER, True)
        logger.addHandler(file_handler)
    return logger

