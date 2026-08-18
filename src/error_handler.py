"""
Centralized Error Handling & Custom Exceptions
Standardized error responses and recovery mechanisms
"""

import asyncio
import logging
import traceback
import time
from functools import wraps
from typing import Callable, Any, Dict, Optional

logger = logging.getLogger("error_handler")


# ═══════════════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════

class ProjectException(Exception):
    """Base exception for all project-specific errors"""

    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ModelNotLoadedError(ProjectException):
    """Model AI belum di-load saat startup"""

    def __init__(self, message: str = "AI Model not loaded"):
        super().__init__(message, "MODEL_NOT_LOADED")


class DatabaseNotConnectedError(ProjectException):
    """Database/ChromaDB belum terhubung"""

    def __init__(self, message: str = "Database not connected"):
        super().__init__(message, "DATABASE_NOT_CONNECTED")


class FileExtractionError(ProjectException):
    """Gagal extract text dari file"""

    def __init__(self, message: str = "Failed to extract file"):
        super().__init__(message, "FILE_EXTRACTION_FAILED")


class InvalidQueryError(ProjectException):
    """Query tidak valid atau format salah"""

    def __init__(self, message: str = "Invalid query"):
        super().__init__(message, "INVALID_QUERY")


class InvalidConfigError(ProjectException):
    """Configuration tidak valid"""

    def __init__(self, message: str = "Invalid configuration"):
        super().__init__(message, "INVALID_CONFIG")


class DataValidationError(ProjectException):
    """Data validation failed"""

    def __init__(self, message: str = "Data validation failed"):
        super().__init__(message, "DATA_VALIDATION_FAILED")


# ═══════════════════════════════════════════════════════════════════
# ERROR RESPONSE FORMATTER
# ═══════════════════════════════════════════════════════════════════

def format_error_response(
    error_type: str,
    message: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format standardized error response

    Args:
        error_type: Type of error (e.g., "MODEL_ERROR")
        message: Human-readable error message
        status_code: HTTP status code
        details: Additional details dict

    Returns:
        Standardized error response dict

    Example:
        response = format_error_response(
            "MODEL_ERROR",
            "Model failed to load",
            503,
            {"model_path": "/path/to/model"}
        )
    """
    # Annotasi eksplisit Dict[str, Any] agar IDE tidak salah infer tipe value
    response: Dict[str, Any] = {
        "status": "error",
        "error_type": error_type,
        "message": message,
        "status_code": status_code
    }

    if details:
        response["details"] = details

    return response


# ═══════════════════════════════════════════════════════════════════
# DECORATORS
# ═══════════════════════════════════════════════════════════════════

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator untuk retry logic dengan exponential backoff

    Args:
        max_retries: Maximum retry attempts
        delay: Initial delay in seconds
        backoff: Delay multiplier for each retry

    Returns:
        Decorator function

    Example:
        @retry_on_failure(max_retries=3, delay=1)
        def risky_function():
            # This will retry 3 times if it fails
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed after {max_retries} attempts in {func.__name__}: {str(e)}",
                            exc_info=True
                        )
                        raise

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed in {func.__name__}, "
                        f"retrying in {current_delay}s: {str(e)}"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

            return None  # Explicit fallback — dicapai hanya jika max_retries=0

        return wrapper
    return decorator


def safe_execute(default_value: Any = None):
    """
    Decorator untuk safe execution dengan default fallback

    Args:
        default_value: Value to return if function fails

    Returns:
        Decorator function

    Example:
        @safe_execute(default_value={})
        def risky_function():
            # If fails, returns {}
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Safe execution of {func.__name__} failed: {str(e)}",
                    exc_info=True
                )
                return default_value

        return wrapper
    return decorator


def handle_errors(default_response: Optional[Dict[str, Any]] = None):
    """
    Decorator untuk standardized error handling

    Args:
        default_response: Response to return on error

    Returns:
        Decorator function

    Example:
        @handle_errors(default_response={"error": "Failed"})
        async def api_endpoint():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except ProjectException as e:
                logger.error(f"Project exception in {func.__name__}: {e.message}")
                return format_error_response(e.error_code, e.message)
            except Exception as e:
                logger.error(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                if default_response:
                    return default_response
                return format_error_response("INTERNAL_ERROR", "Internal server error", 500)

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except ProjectException as e:
                logger.error(f"Project exception in {func.__name__}: {e.message}")
                return format_error_response(e.error_code, e.message)
            except Exception as e:
                logger.error(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                if default_response:
                    return default_response
                return format_error_response("INTERNAL_ERROR", "Internal server error", 500)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def log_exception(func: Callable) -> Callable:
    """
    Decorator untuk log all exceptions dengan full traceback

    Args:
        func: Function to wrap

    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Exception in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
            )
            raise

    return wrapper


def validate_not_none(param_name: str, param_value: Any) -> None:
    """
    Validate that parameter is not None

    Args:
        param_name: Parameter name for error message
        param_value: Parameter value to check

    Raises:
        ValueError: If parameter is None
    """
    if param_value is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")


def validate_type(param_name: str, param_value: Any, expected_type: type) -> None:
    """
    Validate parameter type

    Args:
        param_name: Parameter name for error message
        param_value: Parameter value to check
        expected_type: Expected type

    Raises:
        TypeError: If type mismatch
    """
    if not isinstance(param_value, expected_type):
        raise TypeError(
            f"Parameter '{param_name}' must be {expected_type.__name__}, "
            f"got {type(param_value).__name__}"
        )