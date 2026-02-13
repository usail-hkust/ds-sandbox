"""
ds-sandbox logging utilities

Standard logging configuration for the sandbox.
"""

import logging
from typing import Optional

# Default logging format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with standard configuration.

    Args:
        name: Logger name (usually __name__)
        level: Optional log level override

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(DEFAULT_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Set level if provided
    if level is not None:
        logger.setLevel(level)
    elif logger.level == logging.NOTSET:
        # Default to INFO if not set
        logger.setLevel(logging.INFO)

    return logger


def configure_logging(
    level: int = logging.INFO,
    format: str = DEFAULT_FORMAT,
    file_path: Optional[str] = None
) -> None:
    """
    Configure root logging for the application.

    Args:
        level: Log level
        format: Log format string
        file_path: Optional file path for file logging
    """
    # Create formatter
    formatter = logging.Formatter(format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
