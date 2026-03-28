"""Logging configuration for the GAVE project."""
import logging
import sys

__all__ = ["setup_logger", "default_logger"]


def setup_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with standard format.

    Args:
        name: Logger name. If None, returns root logger.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Default logger for common use
default_logger = setup_logger(__name__)