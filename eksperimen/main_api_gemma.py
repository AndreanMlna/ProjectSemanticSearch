import os
import sys
import time
import chromadb
from typing import Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, Form, HTTPException, Path, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Setup path agar modul lokal dapat diimpor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.text_extractor import extract_text_from_file
from src.logging_utils import setup_logging
from src.error_handler import (
    ModelNotLoadedError,
    DatabaseNotConnectedError,
    FileExtractionError
)
from src.metrics_collector import get_metrics_collector
from src.request_logger import get_request_logger, get_performance_monitor
from src.rate_limiter import get_endpoint_rate_limiter
from src.cache_manager import get_cache_manager
from src.batch_processor import create_batch_uploader, DocumentItem
from src.reranker import get_reranker

from eksperimen.rag_agent_gemma import get_rag_agent

try:
    from src.config_manager import get_config

    config: Any = get_config()
except Exception:
    config = None

logger = setup_logging("main_api_gemma")

MODEL_PATH = os.path.join(ROOT, "output", "minilm-dokumen-arsip-boosted-new-seed-42")
if config:
    try:
        cfg_model_path = config["embedding"]["model_path"]
        MODEL_PATH = cfg_model_path if os.path.isabs(cfg_model_path) else os.path.join(ROOT, cfg_model_path)
    except (KeyError, TypeError):
        pass

LOCAL_DB_PATH = os.path.join(ROOT, "chroma_db_storage")
COLLECTION_NAME = "arsip_kampus_v2"
if config:
    try:
        chroma_cfg = config["chroma"]
        cfg_db_path = chroma_cfg.get("db_path", "chroma_db_storage")
        LOCAL_DB_PATH = cfg_db_path if os.path.isabs(cfg_db_path) else os.path.join(ROOT, cfg_db_path)
        COLLECTION_NAME = chroma_cfg.get("collection_name", COLLECTION_NAME)
    except (KeyError, TypeError):
        pass

UPLOAD_FOLDER = os.path.join(ROOT, "uploads")
if config:
    try:
        cfg_upload_dir = config["storage"]["upload_dir"]
        UPLOAD_FOLDER = cfg_upload_dir if os.path.isabs(cfg_upload_dir) else os.path.join(ROOT, cfg_upload_dir)
    except (KeyError, TypeError):
        pass

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info(f"Config Synced -> Model Path: {MODEL_PATH}, DB Path: {LOCAL_DB_PATH}, Collection: {COLLECTION_NAME}")

# --- GLOBAL VARIABLES ---
ml_models: dict = {}
chroma_client: Optional[Any] = None
collection: Optional[Any] = None


def _check_rate_limit(request: Request, endpoint: str) -> None:

    try:
        client_ip = request.client.host if request.client else "unknown"
        rate_limiter = get_endpoint_rate_limiter()
        allowed, info = rate_limiter.check_limit(endpoint=endpoint, client_id=client_ip)

        if not allowed:
            logger.warning(
                f"Rate limit exceeded: endpoint=/{endpoint}, "
                f"ip={client_ip}, used={info.get('requests_used')}/{info.get('requests_limit')}"
            )
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Too Many Requests",
                    "message": f"Batas request tercapai untuk endpoint /{endpoint}. Coba lagi dalam 1 menit.",
                    "limit": info.get("requests_limit"),
                    "reset_time": info.get("reset_time")
                }
            )

        logger.debug(f"Rate limit OK: endpoint=/{endpoint}, ip={client_ip}, remaining={info.get('remaining')}")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Rate limiter check failed (non-fatal): {e}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("=" * 50)
    logger.info("SERVER STARTING - Loading resources")

    # 1. Load Model AI (Bi-Encoder / Embedding)
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading Model from: {MODEL_PATH}")
            ml_models["minilm"] = SentenceTransformer(MODEL_PATH)
            logger.info("✅ Model loaded successfully")
        else:
            logger.error(f"Model not found at: {MODEL_PATH}")
            raise ModelNotLoadedError(f"Model path: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise

    # 2. Load Model Cross-Encoder Reranker
    try:
        logger.info("Loading Cross-Encoder Reranker Model...")
        _ = get_reranker()
        logger.info("✅ Cross-Encoder Reranker loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Reranker model: {str(e)}", exc_info=True)
        logger.warning("Proceeding without Reranker capabilities (Fallback mode enabled)")

    # 3. Konek ChromaDB
    try:
        logger.info(f"Connecting to ChromaDB at: {LOCAL_DB_PATH}")
        global chroma_client, collection

        if not os.path.exists(LOCAL_DB_PATH):
            os.makedirs(LOCAL_DB_PATH, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=LOCAL_DB_PATH)
            collection = chroma_client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
            logger.info("✅ Created new ChromaDB collection")
        else:
            chroma_client = chromadb.PersistentClient(path=LOCAL_DB_PATH)
            try:
                collection = chroma_client.get_collection(name=COLLECTION_NAME)
                doc_count = collection.count()
                logger.info(f"✅ Connected to ChromaDB. Total Documents: {doc_count}")
            except ValueError:
                logger.warning("Collection not found, creating new one")
                collection = chroma_client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    except Exception as e:
        logger.error(f"Failed to connect ChromaDB: {str(e)}", exc_info=True)
        raise

    # 4. Inisialisasi Cache Manager
    try:
        cache = get_cache_manager()
        logger.info(f"✅ Cache initialized: max_size={cache.max_size}, ttl={cache.ttl_seconds // 60}min")
    except Exception as e:
        logger.warning(f"Cache initialization warning: {str(e)}")

    # 5. Inisialisasi Rate Limiter
    try:
        rl = get_endpoint_rate_limiter()
        logger.info(
            f"✅ Rate limiter initialized: "
            f"search={rl.limits.get('search')}/min, "
            f"upload={rl.limits.get('upload')}/min, "
            f"delete={rl.limits.get('delete')}/min"
        )
    except Exception as e:
        logger.warning(f"Rate limiter initialization warning: {str(e)}")

    yield

    logger.info("Cleaning up resources")
    ml_models.clear()
    logger.info("SERVER STOPPED - Resources cleaned up")


app = FastAPI(title="Sistem Pencarian Arsip Cerdas (Gemma Edition)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=UPLOAD_FOLDER), name="files")


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        model_ok = ml_models.get("minilm") is not None
        db_ok = collection is not None
        db_count = collection.count() if db_ok else 0

        status = "healthy" if (model_ok and db_ok) else "degraded"

        health_data = {
            "status": status,
            "model_loaded": model_ok,
            "db_connected": db_ok,
            "db_documents": db_count,
            "timestamp": time.time()
        }

        logger.debug(f"Health check: {status}")
        return health_data

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/search")
async def search_endpoint(request: SearchRequest, http_request: Request):
    _check_rate_limit(http_request, "search")
    logger.debug(f"Search request: query='{request.query}', top_k={request.top_k}")

    try:
        model = ml_models.get("minilm")

        if not model:
            logger.error("Model not loaded")
            raise ModelNotLoadedError("AI model not loaded during startup")

        if collection is None:
            logger.error("Database not connected")
            raise DatabaseNotConnectedError("ChromaDB connection failed")

        cache = get_cache_manager()
        cached_result = cache.get(query=request.query, top_k=request.top_k)
        if cached_result:
            logger.info(f"Cache HIT: '{request.query}' (top_k={request.top_k}) — skipping encode & DB query")
            return cached_result

        start_time = time.perf_counter()

        try:
            query_vector = model.encode(request.query).tolist()
            logger.debug(f"Query encoded successfully ({len(query_vector)} dimensions)")
        except Exception as e:
            logger.error(f"Failed to encode query: {str(e)}")
            raise

        candidate_count = max(20, request.top_k * 4)
        try:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=candidate_count,
                include=["metadatas", "distances", "documents"]
            )

            # [PERUBAHAN] Petakan teks utuh dari db (documents) agar tidak terpotong
            full_docs_map = {}
            if 'ids' in results and 'documents' in results and results['documents']:
                for idx, d_id in enumerate(results['ids'][0]):
                    full_docs_map[d_id] = results['documents'][0][idx]

        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise

        try:
            reranker = get_reranker()
            final_results = reranker.rerank(query=request.query, chroma_results=results, top_k=request.top_k)

            for doc in final_results:
                doc_id = doc.get("id", "unknown")
                doc["download_url"] = f"http://localhost:8000/files/{doc.get('file_name', '')}"

                if len(doc.get("snippet", "")) > 200:
                    doc["snippet"] = doc["snippet"][:200] + "..."
                elif not doc.get("snippet"):
                    doc["snippet"] = "..."

                # [PERUBAHAN] Timpa metadata yang terpotong dengan teks penuh asli
                real_full_text = full_docs_map.get(doc_id, doc.get("content_only", ""))
                doc["content_only"] = real_full_text
                doc["document_asli"] = real_full_text

        except Exception as rerank_err:
            logger.warning(f"Reranking failed or skipped, fallback to semantic search: {rerank_err}")
            final_results = []
            if results.get('metadatas') and results['metadatas'][0]:
                for i, meta in enumerate(results['metadatas'][0]):
                    score = 1 - results['distances'][0][i]
                    doc_id = results['ids'][0][i] if 'ids' in results else "unknown"
                    fname = meta.get('file_name', '')

                    # [PERUBAHAN] Ambil teks utuh di proses fallback
                    real_full_text = full_docs_map.get(doc_id, meta.get('content_only', ''))

                    final_results.append({
                        "id": doc_id,
                        "score": round(score, 4),
                        "title": meta.get('title', 'Tanpa Judul'),
                        "snippet": meta.get('snippet', ''),  # Jangan dipotong di sini
                        "content_only": real_full_text,
                        "document_asli": real_full_text,
                        "file_name": fname,
                        "download_url": f"http://localhost:8000/files/{fname}"
                    })
                final_results = final_results[:request.top_k]

        duration = time.perf_counter() - start_time
        logger.debug(f"Search and rerank completed in {duration:.4f}s")

        response = {
            "status": "success",
            "time": f"{duration:.4f}s",
            "total_results": len(final_results),
            "data": final_results
        }

        cache.set(query=request.query, top_k=request.top_k, results=response)
        return response

    except ModelNotLoadedError as e:
        logger.error(f"Model error: {e.message}")
        raise HTTPException(status_code=503, detail=e.message)
    except DatabaseNotConnectedError as e:
        logger.error(f"Database error: {e.message}")
        raise HTTPException(status_code=503, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/upload")
async def upload_endpoint(
        http_request: Request,
        title: str = Form(...),
        content: str = Form(...),
        file: UploadFile = None
):
    _check_rate_limit(http_request, "upload")
    logger.debug(f"Upload request: title='{title}', file={file.filename if file else 'None'}")

    try:
        model = ml_models.get("minilm")

        if not model:
            logger.error("Model not loaded for upload")
            raise ModelNotLoadedError("AI model not loaded")
        if collection is None:
            logger.error("Database not connected for upload")
            raise DatabaseNotConnectedError("Database not connected")

        safe_filename = file.filename.replace(" ", "_")
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        logger.debug(f"Saving file to: {file_path}")

        file_content_bytes = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_content_bytes)

        logger.info(f"File saved: {safe_filename}")

        logger.debug(f"Extracting text from file: {safe_filename}")
        extracted_text = extract_text_from_file(file_path)

        full_text_to_embed = f"{title}. {content}. {extracted_text}"
        logger.debug(f"Encoding text ({len(full_text_to_embed)} chars)")
        vector = model.encode(full_text_to_embed).tolist()

        doc_id = f"upload_{int(time.time())}"

        logger.debug(f"Adding document to ChromaDB: {doc_id}")
        collection.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[full_text_to_embed],
            metadatas=[{
                "title": title,
                "file_name": safe_filename,
                "file_path": file_path,
                "snippet": content if content else extracted_text[:200],
                "content_only": full_text_to_embed
            }]
        )

        logger.info(f"Document uploaded successfully: {title} (ID: {doc_id})")

        try:
            cache = get_cache_manager()
            cleared = cache.clear()
            logger.info(f"Cache cleared after upload: {cleared} entries removed")
        except Exception as cache_err:
            logger.warning(f"Cache clear warning (non-fatal): {cache_err}")

        return {
            "status": "success",
            "message": f"Dokumen '{title}' berhasil disimpan & di-index!",
            "doc_id": doc_id,
            "file_url": f"http://localhost:8000/files/{safe_filename}"
        }

    except FileExtractionError as e:
        logger.error(f"File extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"File extraction failed: {str(e)}")
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


class BulkDocumentItem(BaseModel):
    doc_id: str
    title: str
    content: str = ""
    file_name: str = ""
    file_path: str = ""
    extracted_text: str = ""


class BulkUploadRequest(BaseModel):
    documents: list[BulkDocumentItem]


@app.post("/upload/bulk")
async def bulk_upload_endpoint(request: BulkUploadRequest, http_request: Request):
    """Endpoint untuk mengunggah dan meng-indeks dokumen dalam jumlah besar (bulk)."""
    _check_rate_limit(http_request, "upload")
    logger.info(f"Bulk upload request received: {len(request.documents)} documents")

    model = ml_models.get("minilm")
    if not model:
        logger.error("Model not loaded for bulk upload")
        raise ModelNotLoadedError("AI model not loaded")
    if collection is None:
        logger.error("Database not connected for bulk upload")
        raise DatabaseNotConnectedError("Database not connected")

    try:
        items_to_upload = []
        for doc in request.documents:
            full_text_to_embed = f"{doc.title}. {doc.content}. {doc.extracted_text}".strip()
            snippet_text = doc.content if doc.content else doc.extracted_text[:300]

            items_to_upload.append(
                DocumentItem(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    file_name=doc.file_name,
                    file_path=doc.file_path,
                    full_text=full_text_to_embed,
                    snippet=snippet_text
                )
            )

        uploader = create_batch_uploader(model, collection)
        result = uploader.upload_batch(items_to_upload, skip_existing=True)

        if result.success > 0:
            try:
                cache = get_cache_manager()
                cleared = cache.clear()
                logger.info(f"Cache cleared after bulk upload: {cleared} entries removed")
            except Exception as cache_err:
                logger.warning(f"Cache clear warning (non-fatal): {cache_err}")

        return {
            "status": "success",
            "message": f"Proses batch selesai. {result.success} sukses, {result.failed} gagal.",
            "summary": result.to_dict()
        }

    except Exception as e:
        logger.error(f"Bulk upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bulk upload failed: {str(e)}")


@app.delete("/documents/{doc_id}")
async def delete_document_endpoint(
        http_request: Request,
        doc_id: str = Path(..., description="ID Dokumen yang akan dihapus")
):
    _check_rate_limit(http_request, "delete")
    logger.debug(f"Delete request for document: {doc_id}")

    if collection is None:
        logger.error("Database not connected for delete")
        raise HTTPException(status_code=503, detail="Database belum terhubung")

    try:
        existing = collection.get(ids=[doc_id])
        if not existing['ids']:
            raise HTTPException(status_code=404, detail=f"Dokumen dengan ID '{doc_id}' tidak ditemukan.")

        collection.delete(ids=[doc_id])
        logger.info(f"Document deleted successfully: {doc_id}")

        try:
            cache = get_cache_manager()
            cleared = cache.clear()
            logger.info(f"Cache cleared after delete: {cleared} entries removed")
        except Exception as cache_err:
            logger.warning(f"Cache clear warning (non-fatal): {cache_err}")

        return {
            "status": "success",
            "message": f"Dokumen dengan ID '{doc_id}' berhasil dihapus permanen."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gagal menghapus: {str(e)}")


class RAGRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/rag/ask")
async def rag_ask_endpoint(request: RAGRequest):
    """RAG endpoint dengan pengurutan ulang dokumen (Rerank) ke model Gemma."""
    logger.info(f"RAG request: '{request.question}'")

    try:
        embedding_model = ml_models.get("minilm")
        if embedding_model is None:
            raise HTTPException(status_code=503, detail="Model AI belum dimuat ke memori API.")

        if collection is None:
            raise HTTPException(status_code=503, detail="Koneksi database ChromaDB kosong.")

        agent = get_rag_agent()

        response = agent.answer(
            question=request.question,
            embedding_model=embedding_model,
            chroma_collection=collection
        )

        return {
            "status": "success",
            "question": response.question,
            "answer": response.answer,
            "sources": response.sources,
            "search_results_count": response.search_results_count,
            "context_chars_total": response.context_chars_total,
            "error": response.error
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG failed: {str(e)}")


@app.get("/rag/status")
async def rag_status_endpoint():
    """Cek apakah Search API dan Ollama aktif."""
    try:
        agent = get_rag_agent()
        status = agent.is_ready()

        return {
            "status": "success",
            "rag_ready": status["ready"],
            "components": {
                "search_api": status["search_api"],
                "ollama": status["ollama"],
                "model": status["model"]
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"RAG status error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get RAG status")


@app.get("/metrics")
async def get_metrics():
    """Get API performance metrics"""
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
            "rate_limiting": rate_limiter.get_all_stats()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


@app.get("/performance")
async def get_performance_stats():
    """Get detailed performance statistics"""
    logger.debug("Performance endpoint accessed")
    try:
        monitor = get_performance_monitor()
        return {
            "status": "success",
            "timestamp": time.time(),
            "avg_response_time": monitor.get_average_time(),
            "slow_requests": monitor.get_slow_requests(limit=5),
            "total_requests": len(monitor.request_times),
            "slow_threshold": monitor.slow_threshold
        }
    except Exception as e:
        logger.error(f"Error getting performance stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get performance stats")


@app.get("/logs/requests")
async def get_request_logs(limit: int = 20):
    """Get recent request logs"""
    logger.debug(f"Request logs endpoint accessed (limit={limit})")
    try:
        req_logger = get_request_logger()
        stats = req_logger.get_stats()
        return {
            "status": "success",
            "timestamp": time.time(),
            "limit": min(limit, 100),
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting request logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get request logs")


@app.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    logger.debug("System status endpoint accessed")
    try:
        model_ok = ml_models.get("minilm") is not None
        db_ok = collection is not None
        db_count = collection.count() if db_ok else 0

        metrics = get_metrics_collector()
        performance = get_performance_monitor()

        try:
            cache = get_cache_manager()
            cache_stats = cache.get_stats()
        except Exception:
            cache_stats = {"enabled": False}

        try:
            rl = get_endpoint_rate_limiter()
            rate_limit_stats = rl.get_all_stats()
        except Exception:
            rate_limit_stats = {}

        return {
            "status": "operational" if (model_ok and db_ok) else "degraded",
            "timestamp": time.time(),
            "components": {
                "model": {
                    "status": "loaded" if model_ok else "not_loaded",
                    "name": "minilm-dokumen-arsip-boosted"
                },
                "database": {
                    "status": "connected" if db_ok else "disconnected",
                    "documents": db_count
                },
                "cache": cache_stats,
                "rate_limiter": rate_limit_stats
            },
            "performance": {
                "avg_response_time": performance.get_average_time(),
                "total_requests": len(performance.request_times)
            },
            "metrics": metrics.get_performance_summary()
        }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@app.post("/metrics/export")
async def export_metrics(export_format: str = "json"):
    """Export metrics to file"""
    logger.debug(f"Export metrics endpoint accessed (format={export_format})")
    try:
        if export_format != "json":
            raise ValueError("Only JSON format supported")

        metrics = get_metrics_collector()
        filepath = f"logs/metrics_export_{int(time.time())}.json"

        metrics.export_metrics(filepath)
        logger.info(f"Metrics exported to {filepath}")

        return {
            "status": "success",
            "message": f"Metrics exported to {filepath}",
            "filepath": filepath,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error exporting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {str(e)}")


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    logger.debug("Cache stats endpoint accessed")
    try:
        cache = get_cache_manager()
        stats = cache.get_stats()
        return {
            "status": "success",
            "timestamp": time.time(),
            "cache": stats
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get cache stats")


@app.post("/cache/clear")
async def clear_cache():
    """Clear all cache entries manually"""
    logger.debug("Cache clear endpoint accessed")
    try:
        cache = get_cache_manager()
        cleared_count = cache.clear()
        logger.info(f"Cache manually cleared: {cleared_count} entries removed")

        return {
            "status": "success",
            "message": "Cache berhasil dibersihkan",
            "entries_cleared": cleared_count,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@app.get("/")
def home():
    """Root endpoint - returns API information"""
    logger.debug("Root endpoint accessed")
    return {
        "message": "API Semantic Search (ChromaDB Local) + Reranker Aktif (Gemma Edition)!",
        "docs": "/docs",
        "version": "1.0",
        "status": "operational"
    }