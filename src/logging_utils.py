"""
Centralized Logging Configuration
Provides unified logging to console (stdout) across the entire application.
Mengikuti standar 12-Factor App & Docker Logging (tanpa file handler).
"""

import logging
import sys


def setup_logging(name: str, log_level=logging.INFO) -> logging.Logger:
    """
    Setup centralized logging to console (stdout) for any module.

    Args:
        name: Logger name (usually __name__ or module name)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers if setup_logging called multiple times
    if logger.hasHandlers():
        return logger

    # Console handler (stdout for Docker & real-time terminal output)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)

    # Formatter - consistent format across all loggers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(ch)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerContextManager:
    """Context manager for temporary logging level changes."""

    def __init__(self, logger_name: str, temp_level=logging.DEBUG):
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.temp_level = temp_level

    def __enter__(self):
        self.logger.setLevel(self.temp_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
