import time

from fastapi import APIRouter, HTTPException, Request

from src.cache_manager import get_cache_manager
from src.chroma_client import get_collection
from src.error_handler import DatabaseNotConnectedError, ModelNotLoadedError
from src.helpers import check_rate_limit
from src.lifespan import get_ml_models
from src.logging_utils import setup_logging
from src.reranker import get_reranker
from src.schemas import SearchRequest

router = APIRouter()
logger = setup_logging("search_router")


@router.get("/health", summary="Health Check")
async def health_check():
    try:
        model_ok = get_ml_models().get("minilm") is not None
        active_col = get_collection()
        db_ok = active_col is not None
        db_count = active_col.count() if db_ok else 0

        status = "healthy" if (model_ok and db_ok) else "degraded"

        health_data = {
            "status": status,
            "model_loaded": model_ok,
            "db_connected": db_ok,
            "db_documents": db_count,
            "timestamp": time.time(),
        }

        logger.debug(f"Health check: {status}")
        return health_data

    except (ValueError, RuntimeError, AttributeError) as e:
        logger.exception("Health check failed")
        return {
            "status": "unhealthy",
            "error": f"{e!s}",
            "timestamp": time.time(),
        }


@router.post("/search", summary="Pencarian Semantik & Reranking")
async def search_endpoint(request: SearchRequest, http_request: Request):
    check_rate_limit(http_request, "search")
    logger.debug(f"Search request: query='{request.query}', top_k={request.top_k}")

    try:
        model = get_ml_models().get("minilm")
        if not model:
            logger.error("Model not loaded")
            raise ModelNotLoadedError("AI model not loaded during startup")

        active_col = get_collection()
        if active_col is None:
            logger.error("Database not connected")
            raise DatabaseNotConnectedError("ChromaDB connection failed")

        cache = get_cache_manager()
        cached_result = cache.get(query=request.query, top_k=request.top_k)
        if cached_result:
            logger.info(
                f"Cache HIT: '{request.query}' (top_k={request.top_k}) — skipping encode & DB query"
            )
            return cached_result

        start_time = time.perf_counter()

        try:
            query_vector = model.encode(request.query).tolist()
            logger.debug(f"Query encoded successfully ({len(query_vector)} dimensions)")
        except Exception as e:
            logger.error(f"Failed to encode query: {e!s}")
            raise

        candidate_count = max(20, request.top_k * 4)
        try:
            results = active_col.query(
                query_embeddings=[query_vector],
                n_results=candidate_count,
                include=["metadatas", "distances", "documents"],
            )

            full_docs_map = {}
            if "ids" in results and "documents" in results and results["documents"]:
                for idx, d_id in enumerate(results["ids"][0]):
                    full_docs_map[d_id] = results["documents"][0][idx]

        except Exception as e:
            logger.error(f"Database query failed: {e!s}")
            raise

        try:
            reranker = get_reranker()
            final_results = reranker.rerank(
                query=request.query, chroma_results=results, top_k=request.top_k
            )

            for doc in final_results:
                doc_id = doc.get("uuid") or doc.get("id", "unknown")
                doc["uuid"] = doc_id
                doc.pop("id", None)
                doc.pop("download_url", None)

                if len(doc.get("snippet", "")) > 200:
                    doc["snippet"] = doc["snippet"][:200] + "..."
                elif not doc.get("snippet"):
                    doc["snippet"] = "..."

                real_full_text = full_docs_map.get(doc_id, doc.get("content_only", ""))
                doc["content_only"] = real_full_text
                doc["document_asli"] = real_full_text

        except (ValueError, RuntimeError, AttributeError) as rerank_err:
            logger.warning(
                f"Reranking failed or skipped, fallback to semantic search: {rerank_err!s}"
            )
            final_results = []
            if results.get("metadatas") and results["metadatas"][0]:
                for i, meta in enumerate(results["metadatas"][0]):
                    score = 1 - results["distances"][0][i]
                    raw_id = results["ids"][0][i] if "ids" in results else "unknown"
                    doc_uuid = meta.get("uuid") or raw_id
                    fname = meta.get("file_name", "")

                    real_full_text = full_docs_map.get(
                        raw_id, meta.get("content_only", "")
                    )

                    doc_item = dict(meta) if meta else {}
                    doc_item.update(
                        {
                            "uuid": doc_uuid,
                            "score": round(score, 4),
                            "title": meta.get("title", "Tanpa Judul"),
                            "snippet": meta.get("snippet", ""),
                            "content_only": real_full_text,
                            "document_asli": real_full_text,
                            "file_name": fname,
                        }
                    )
                    doc_item.pop("id", None)
                    doc_item.pop("download_url", None)
                    final_results.append(doc_item)
                final_results = final_results[: request.top_k]

        duration = time.perf_counter() - start_time
        logger.debug(f"Search and rerank completed in {duration:.4f}s")

        response = {
            "status": "success",
            "time": f"{duration:.4f}s",
            "total_results": len(final_results),
            "data": final_results,
        }

        cache.set(query=request.query, top_k=request.top_k, results=response)
        return response

    except ModelNotLoadedError as e:
        logger.error(f"Model error: {e.message}")
        raise HTTPException(status_code=503, detail=e.message)
    except DatabaseNotConnectedError as e:
        logger.error(f"Database error: {e.message}")
        raise HTTPException(status_code=503, detail=e.message)
    except (ValueError, RuntimeError, AttributeError):
        logger.exception("Unexpected error in search")
        raise HTTPException(status_code=500, detail="Internal server error")
