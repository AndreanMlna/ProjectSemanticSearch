"""
Request Logger Middleware
Logs all incoming requests and outgoing responses
"""

from datetime import datetime
from typing import Dict, Any


class RequestLogger:
    """Log HTTP requests and responses"""

    def __init__(self, logger=None):
        """
        Initialize request logger

        Args:
            logger: Logger instance to use
        """
        self.logger = logger
        self.request_count = 0
        self.request_history: Dict[str, Any] = {}

    def log_request(self, method: str, path: str, query: str = None) -> None:
        """
        Log incoming request

        Args:
            method: HTTP method (GET, POST, etc)
            path: Request path
            query: Query string if applicable
        """
        self.request_count += 1

        log_msg = f"[Request #{self.request_count}] {method} {path}"
        if query:
            log_msg += f" | Query: {query[:100]}"  # Limit log length

        if self.logger:
            self.logger.debug(log_msg)

    def log_response(
        self,
        method: str,
        path: str,
        status_code: int,
        response_time: float
    ) -> None:
        """
        Log response

        Args:
            method: HTTP method
            path: Request path
            status_code: HTTP status code
            response_time: Request duration in seconds
        """
        status_label = "OK" if status_code < 400 else "ERROR"
        log_msg = f"[Response] {method} {path} -> {status_code} {status_label} ({response_time:.4f}s)"

        if self.logger:
            if status_code >= 400:
                self.logger.warning(log_msg)
            else:
                self.logger.info(log_msg)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get request statistics

        Returns:
            Dictionary with stats
        """
        return {
            "total_requests": self.request_count,
            "request_history_size": len(self.request_history)
        }


class PerformanceMonitor:
    """Monitor request performance"""

    def __init__(self):
        """Initialize performance monitor"""
        self.request_times = []
        self.slow_requests = []
        self.slow_threshold = 1.0  # 1 second

    def record_request_time(self, path: str, duration: float) -> None:
        """
        Record request time

        Args:
            path: Request path
            duration: Request duration in seconds
        """
        self.request_times.append({
            "path": path,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })

        # Track slow requests
        if duration > self.slow_threshold:
            self.slow_requests.append({
                "path": path,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            })

    def get_average_time(self) -> float:
        """
        Get average request time

        Returns:
            Average duration in seconds
        """
        if not self.request_times:
            return 0

        total = sum(r["duration"] for r in self.request_times)
        return total / len(self.request_times)

    def get_slow_requests(self, limit: int = 10) -> list:
        """
        Get slowest requests

        Args:
            limit: Number of results to return

        Returns:
            List of slow requests
        """
        return sorted(
            self.slow_requests,
            key=lambda x: x["duration"],
            reverse=True
        )[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics

        Returns:
            Dictionary with stats
        """
        return {
            "total_requests": len(self.request_times),
            "avg_response_time": self.get_average_time(),
            "slow_requests_count": len(self.slow_requests),
            "slow_threshold": self.slow_threshold
        }


# Global instances
request_logger = RequestLogger()
performance_monitor = PerformanceMonitor()


def get_request_logger() -> RequestLogger:
    """Get global request logger instance"""
    return request_logger


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    return performance_monitor