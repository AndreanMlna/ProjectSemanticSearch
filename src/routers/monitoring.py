import time

from fastapi import APIRouter, HTTPException

from src.cache_manager import get_cache_manager
from src.chroma_client import get_collection
from src.lifespan import get_ml_models
from src.logging_utils import setup_logging
from src.metrics_collector import get_metrics_collector
from src.rate_limiter import get_endpoint_rate_limiter
from src.request_logger import get_performance_monitor

router = APIRouter()
logger = setup_logging("monitoring_router")


@router.get("/metrics", summary="Metrik Performa API")
async def get_metrics():
    logger.debug("Metrics endpoint accessed")
    try:
        metrics = get_metrics_collector()
        performance = get_performance_monitor()
        rate_limiter = get_endpoint_rate_limiter()

        return {
            "status": "success",
            "timestamp": time.time(),
            "metrics": metrics.get_performance_summary(),
            "performance": performance.get_stats(),
            "rate_limiting": rate_limiter.get_all_stats(),
        }
    except (ValueError, RuntimeError, AttributeError):
        logger.exception("Error getting metrics")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


@router.get("/status", summary="Status Kesehatan Komprehensif")
async def get_system_status():
    logger.debug("System status endpoint accessed")
    try:
        model_ok = get_ml_models().get("minilm") is not None
        col = get_collection()
        db_ok = col is not None
        db_count = col.count() if db_ok else 0

        metrics = get_metrics_collector()
        performance = get_performance_monitor()

        try:
            cache = get_cache_manager()
            cache_stats = cache.get_stats()
        except (ImportError, AttributeError, ValueError):
            cache_stats = {"enabled": False}

        try:
            rl = get_endpoint_rate_limiter()
            rate_limit_stats = rl.get_all_stats()
        except (ImportError, AttributeError, ValueError):
            rate_limit_stats = {}

        return {
            "status": "operational" if (model_ok and db_ok) else "degraded",
            "timestamp": time.time(),
            "components": {
                "model": {
                    "status": "loaded" if model_ok else "not_loaded",
                    "name": "minilm",
                },
                "database": {
                    "status": "connected" if db_ok else "disconnected",
                    "documents": db_count,
                },
                "cache": cache_stats,
                "rate_limiter": rate_limit_stats,
            },
            "performance": {
                "avg_response_time": performance.get_average_time(),
                "total_requests": len(performance.request_times),
            },
            "metrics": metrics.get_performance_summary(),
        }
    except (ValueError, RuntimeError, AttributeError):
        logger.exception("Error getting system status")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@router.get("/cache/stats", summary="Statistik Cache")
async def get_cache_stats():
    logger.debug("Cache stats endpoint accessed")
    try:
        cache = get_cache_manager()
        stats = cache.get_stats()
        return {
            "status": "success",
            "timestamp": time.time(),
            "cache": stats,
        }
    except (ValueError, RuntimeError, AttributeError):
        logger.exception("Error getting cache stats")
        raise HTTPException(status_code=500, detail="Failed to get cache stats")


@router.post("/cache/clear", summary="Bersihkan Cache")
async def clear_cache():
    logger.debug("Cache clear endpoint accessed")
    try:
        cache = get_cache_manager()
        cleared_count = cache.clear()
        logger.info(f"Cache manually cleared: {cleared_count} entries removed")

        return {
            "status": "success",
            "message": "Cache berhasil dibersihkan",
            "entries_cleared": cleared_count,
            "timestamp": time.time(),
        }
    except (ValueError, RuntimeError, AttributeError):
        logger.exception("Error clearing cache")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@router.get("/", summary="Root API Information")
def home():
    logger.debug("Root endpoint accessed")
    return {
        "message": "API Semantic Search & Reranker Aktif (No LLM)!",
        "docs": "/docs",
        "version": "1.0.0",
        "status": "operational",
    }
