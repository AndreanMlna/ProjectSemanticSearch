"""
Rate Limiting Utilities
Prevents API abuse and ensures fair usage
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple, Any
from collections import defaultdict


class RateLimiter:
    """Simple rate limiter implementation"""

    def __init__(self, requests_per_minute: int = 100):
        """
        Initialize rate limiter

        Args:
            requests_per_minute: Max requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.request_history: Dict[str, list] = defaultdict(list)

    def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed for client

        Args:
            client_id: Client identifier (usually IP address)

        Returns:
            Tuple of (allowed: bool, info: dict)
        """
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=1)

        # Remove old requests
        self.request_history[client_id] = [
            req_time for req_time in self.request_history[client_id]
            if req_time > cutoff_time
        ]

        current_count = len(self.request_history[client_id])

        info = {
            "requests_used": current_count,
            "requests_limit": self.requests_per_minute,
            "remaining": max(0, self.requests_per_minute - current_count),
            "reset_time": (cutoff_time + timedelta(minutes=1)).isoformat()
        }

        if current_count >= self.requests_per_minute:
            return False, info

        # Record this request
        self.request_history[client_id].append(now)

        return True, info

    def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """
        Get stats for specific client

        Args:
            client_id: Client identifier

        Returns:
            Dictionary with stats
        """
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=1)

        requests = [
            req for req in self.request_history[client_id]
            if req > cutoff_time
        ]

        return {
            "client_id": client_id,
            "requests_this_minute": len(requests),
            "limit": self.requests_per_minute
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get stats for all clients

        Returns:
            Dictionary with all stats
        """
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=1)

        stats = {}
        for client_id, requests in self.request_history.items():
            recent = [r for r in requests if r > cutoff_time]
            if recent:
                stats[client_id] = {
                    "requests": len(recent),
                    "limit": self.requests_per_minute
                }

        return stats

    def reset_client(self, client_id: str) -> None:
        """
        Reset rate limit for client

        Args:
            client_id: Client identifier
        """
        self.request_history[client_id] = []


class AdaptiveRateLimiter:
    """Rate limiter that adapts based on system load"""

    def __init__(self, base_limit: int = 100):
        """
        Initialize adaptive rate limiter

        Args:
            base_limit: Base requests per minute limit
        """
        self.base_limit = base_limit
        self.current_limit = base_limit
        self.limiter = RateLimiter(base_limit)
        self.last_adjustment = datetime.now()

    def adjust_limit(self, system_load: float) -> None:
        """
        Adjust rate limit based on system load

        Args:
            system_load: System load percentage (0-1)
        """
        if system_load > 0.8:
            # High load - reduce limit by 20%
            self.current_limit = int(self.base_limit * 0.8)
        elif system_load > 0.6:
            # Medium load - reduce limit by 10%
            self.current_limit = int(self.base_limit * 0.9)
        else:
            # Low load - use base limit
            self.current_limit = self.base_limit

        self.limiter = RateLimiter(self.current_limit)
        self.last_adjustment = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get limiter statistics

        Returns:
            Dictionary with stats
        """
        return {
            "base_limit": self.base_limit,
            "current_limit": self.current_limit,
            "active_clients": len(self.limiter.request_history),
            "last_adjustment": self.last_adjustment.isoformat()
        }


class EndpointRateLimiter:
    """Rate limiter for specific endpoints"""

    def __init__(self):
        """Initialize endpoint rate limiter"""
        self.limits: Dict[str, int] = {
            "search": 100,      # 100 per minute
            "upload": 20,       # 20 per minute
            "delete": 30        # 30 per minute
        }
        self.limiters: Dict[str, RateLimiter] = {
            endpoint: RateLimiter(limit)
            for endpoint, limit in self.limits.items()
        }

    def check_limit(self, endpoint: str, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed for endpoint

        Args:
            endpoint: Endpoint name
            client_id: Client identifier

        Returns:
            Tuple of (allowed: bool, info: dict)
        """
        if endpoint not in self.limiters:
            return True, {"status": "Unknown endpoint"}

        return self.limiters[endpoint].is_allowed(client_id)

    def set_limit(self, endpoint: str, requests_per_minute: int) -> None:
        """
        Set rate limit for endpoint

        Args:
            endpoint: Endpoint name
            requests_per_minute: New limit
        """
        self.limits[endpoint] = requests_per_minute
        self.limiters[endpoint] = RateLimiter(requests_per_minute)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get stats for all endpoints

        Returns:
            Dictionary with all stats
        """
        return {
            endpoint: {
                "limit": self.limits[endpoint],
                "active_clients": len(self.limiters[endpoint].request_history)
            }
            for endpoint in self.limits
        }


# Global rate limiter instances
rate_limiter = RateLimiter()
endpoint_rate_limiter = EndpointRateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    return rate_limiter


def get_endpoint_rate_limiter() -> EndpointRateLimiter:
    """Get global endpoint rate limiter instance"""
    return endpoint_rate_limiter