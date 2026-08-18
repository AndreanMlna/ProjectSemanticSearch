"""
Metrics Collection System
Collects and analyzes API performance metrics
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List
from collections import defaultdict
import json


class MetricsCollector:
    """Collect API metrics"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.endpoint_stats: Dict[str, Dict[str, Any]] = {}
    
    def record_search(self, query: str, results_count: int, response_time: float) -> None:
        """
        Record search metric
        
        Args:
            query: Search query
            results_count: Number of results returned
            response_time: Response time in seconds
        """
        self.metrics["search"].append({
            "timestamp": datetime.now().isoformat(),
            "query_length": len(query),
            "results_count": results_count,
            "response_time": response_time
        })
        
        self._update_endpoint_stats("search", response_time)
    
    def record_upload(self, file_size: int, response_time: float) -> None:
        """
        Record upload metric
        
        Args:
            file_size: File size in bytes
            response_time: Response time in seconds
        """
        self.metrics["upload"].append({
            "timestamp": datetime.now().isoformat(),
            "file_size": file_size,
            "response_time": response_time
        })
        
        self._update_endpoint_stats("upload", response_time)
    
    def record_delete(self, response_time: float) -> None:
        """
        Record delete metric
        
        Args:
            response_time: Response time in seconds
        """
        self.metrics["delete"].append({
            "timestamp": datetime.now().isoformat(),
            "response_time": response_time
        })
        
        self._update_endpoint_stats("delete", response_time)
    
    def _update_endpoint_stats(self, endpoint: str, response_time: float) -> None:
        """Update endpoint statistics"""
        if endpoint not in self.endpoint_stats:
            self.endpoint_stats[endpoint] = {
                "request_count": 0,
                "total_time": 0,
                "min_time": float('inf'),
                "max_time": 0,
                "error_count": 0
            }
        
        stats = self.endpoint_stats[endpoint]
        stats["request_count"] += 1
        stats["total_time"] += response_time
        stats["min_time"] = min(stats["min_time"], response_time)
        stats["max_time"] = max(stats["max_time"], response_time)
    
    def get_endpoint_stats(self, endpoint: str) -> Dict[str, Any]:
        """
        Get statistics for endpoint
        
        Args:
            endpoint: Endpoint name
        
        Returns:
            Dictionary with stats
        """
        if endpoint not in self.endpoint_stats:
            return {}
        
        stats = self.endpoint_stats[endpoint]
        avg_time = (stats["total_time"] / stats["request_count"] 
                   if stats["request_count"] > 0 else 0)
        
        return {
            "endpoint": endpoint,
            "request_count": stats["request_count"],
            "avg_response_time": avg_time,
            "min_response_time": stats["min_time"],
            "max_response_time": stats["max_time"],
            "total_time": stats["total_time"]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all endpoint statistics
        
        Returns:
            Dictionary with all stats
        """
        return {
            endpoint: self.get_endpoint_stats(endpoint)
            for endpoint in self.endpoint_stats
        }
    
    def get_search_metrics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent search metrics
        
        Args:
            limit: Number of results to return
        
        Returns:
            List of search metrics
        """
        return self.metrics["search"][-limit:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get overall performance summary
        
        Returns:
            Dictionary with summary
        """
        total_requests = sum(
            s["request_count"] for s in self.endpoint_stats.values()
        )
        
        if total_requests == 0:
            return {"status": "No metrics yet"}
        
        avg_time = sum(
            s["total_time"] for s in self.endpoint_stats.values()
        ) / total_requests
        
        return {
            "total_requests": total_requests,
            "avg_response_time": avg_time,
            "endpoints": list(self.endpoint_stats.keys()),
            "endpoint_stats": self.get_all_stats()
        }
    
    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to JSON file
        
        Args:
            filepath: Path to export to
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_performance_summary(),
            "endpoint_stats": self.get_all_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear_old_metrics(self, hours: int = 24) -> None:
        """
        Clear metrics older than specified hours
        
        Args:
            hours: Hours to keep metrics for
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for endpoint in self.metrics:
            self.metrics[endpoint] = [
                m for m in self.metrics[endpoint]
                if datetime.fromisoformat(m["timestamp"]) > cutoff_time
            ]


# Global metrics instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    return _metrics_collector

