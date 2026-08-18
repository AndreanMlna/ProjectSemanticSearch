"""
Centralized Logging Configuration
Provides unified logging across entire application
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logging(name: str, log_level=logging.INFO):
    """
    Setup centralized logging for any module
    
    Args:
        name: Logger name (usually __name__ or module name)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    
    Example:
        logger = setup_logging("main_api")
        logger.info("Server started")
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers if setup_logging called multiple times
    if logger.hasHandlers():
        return logger
    
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    
    # File handler with rotation (10MB per file, keep 5 backups)
    log_filename = f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log"
    fh = RotatingFileHandler(
        log_filename,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    fh.setLevel(log_level)
    
    # Console handler (for real-time output)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Formatter - consistent format across all loggers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def get_logger(name: str):
    """
    Get existing logger or create new one
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerContextManager:
    """Context manager for temporary logging level changes"""
    
    def __init__(self, logger_name: str, temp_level=logging.DEBUG):
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.temp_level = temp_level
    
    def __enter__(self):
        self.logger.setLevel(self.temp_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)

