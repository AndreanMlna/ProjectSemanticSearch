"""
Backend API untuk Sistem Pencarian Arsip Cerdas (Semantic Search & Reranker)
Menggunakan FastAPI, SentenceTransformer (MiniLM), Cross-Encoder Reranker, dan ChromaDB Docker.
"""

import json
import os
import re
import sys
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import chromadb
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    Security,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# =====================================================================
# 1. SETUP ENVIRONMENT & ROOT PATH
# =====================================================================
load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.batch_processor import DocumentItem, create_batch_uploader
from src.cache_manager import get_cache_manager
from src.error_handler import (
    DatabaseNotConnectedError,
    FileExtractionError,
    ModelNotLoadedError,
)
from src.logging_utils import setup_logging
from src.metrics_collector import get_metrics_collector
from src.rate_limiter import get_endpoint_rate_limiter
from src.reranker import get_reranker
from src.request_logger import get_performance_monitor
from src.text_extractor import extract_text_from_file

logger = setup_logging("main_api")

# =====================================================================
# 2. CONFIGURATION & GLOBAL VARIABLES
# =====================================================================
MODEL_PATH: str = os.getenv("HF_MODEL_NAME", "andrerean/minilm-arsip-kampus-seranah")
CE_MODEL_PATH: str = os.getenv("CE_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION", "arsip_kampus_v2")
UPLOAD_FOLDER: str = os.getenv("UPLOAD_DIR", os.path.join(ROOT, "uploads"))

# Konfigurasi Keamanan & API Authentication
API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "seranah_secret_key_2026")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="API Key Header")
http_bearer = HTTPBearer(auto_error=False, description="Bearer Token")
PUBLIC_ENDPOINTS = {"/health", "/", "/docs", "/redoc", "/openapi.json"}

# Konfigurasi CORS murni dari Environment Variable
ALLOWED_ORIGINS_RAW: str = os.getenv("ALLOWED_ORIGINS", "*")
if ALLOWED_ORIGINS_RAW.strip() == "*":
    ALLOWED_ORIGINS: List[str] = ["*"]
else:
    ALLOWED_ORIGINS: List[str] = [
        origin.strip() for origin in ALLOWED_ORIGINS_RAW.split(",") if origin.strip()
    ]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logger.info(
    f"Configuration Loaded -> Embedding Model: {MODEL_PATH}, Cross-Encoder: {CE_MODEL_PATH}, "
    f"Collection: {COLLECTION_NAME}, Upload Dir: {UPLOAD_FOLDER}, Allowed Origins: {ALLOWED_ORIGINS}, "
    f"Auth Protected: YES (Public: /health, /)"
)

# Global application state
ml_models: Dict[str, Any] = {}
chroma_client: Optional[Any] = None
collection: Optional[Any] = None


# =====================================================================
# 3. HELPER FUNCTIONS & AUTHENTICATION
# =====================================================================
async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(http_bearer),
) -> bool:
    """
    Middleware dependency untuk memverifikasi autentikasi API Key / Bearer Token.
    Mengecualikan endpoint /health, /, dan dokumentasi OpenAPI (/docs, /redoc, /openapi.json).
    Mendukung header 'X-API-Key' atau 'Authorization: Bearer <API_KEY>'.
    """
    path = request.url.path

    # 1. Pengecualian endpoint publik (/health, /, /docs, /redoc, /openapi.json)
    if path in PUBLIC_ENDPOINTS or path.startswith(("/docs", "/redoc", "/openapi.json")):
        return True

    # 2. Ambil token dari X-API-Key atau Authorization Bearer
    token = api_key or (bearer.credentials if bearer else None)

    # 3. Verifikasi token menggunakan constant-time comparison (mencegah timing attack)
    if not token or not secrets.compare_digest(token, API_SECRET_KEY):
        client_ip = request.client.host if request.client else "unknown"
        logger.warning(
            f"Akses tidak sah ditolak: endpoint={path}, ip={client_ip}, token_provided={'yes' if token else 'no'}"
        )
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "error": "Unauthorized",
                "message": "Autentikasi gagal. Sediakan API Key yang valid melalui header 'X-API-Key' atau 'Authorization: Bearer <API_KEY>'.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


def extract_keywords(text: str) -> str:
    """Ekstraksi kata kunci dari teks menggunakan pola regex 'kata kunci: ...'."""
    if not text:
        return "-"
    parts = re.split(r"kata\s+kunci\s*:?", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].strip()
    return "-"


def _build_file_url(request: Request, filename: str) -> str:
    """Membangun URL download file secara dinamis mengikuti host client (bebas hardcode)."""
    if not filename or filename == "-":
        return ""
    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/files/{filename}"


def _check_rate_limit(request: Request, endpoint: str) -> None:
    """Cek rate limit untuk endpoint tertentu berdasarkan IP client."""
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
                    "reset_time": info.get("reset_time"),
                },
            )

        logger.debug(f"Rate limit OK: endpoint=/{endpoint}, ip={client_ip}, remaining={info.get('remaining')}")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Rate limiter check failed (non-fatal): {e}")


def get_active_collection() -> Optional[Any]:
    """Mengambil koleksi ChromaDB yang aktif, atau melakukan auto-reconnect jika koleksi di-reset di server."""
    global chroma_client, collection
    if chroma_client is None:
        try:
            chroma_host = os.getenv("CHROMA_HOST", "localhost")
            chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
            chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        except Exception as e:
            logger.error(f"Gagal menghubungkan ke ChromaDB: {e}")
            return None

    try:
        if collection is not None:
            # Uji apakah koleksi masih valid di server
            collection.count()
            return collection
    except Exception:
        logger.warning(f"Koleksi '{COLLECTION_NAME}' terdeteksi telah di-reset/diperbarui. Melakukan re-fetch...")

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception:
        try:
            collection = chroma_client.create_collection(
                name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
            return collection
        except Exception as e:
            logger.error(f"Gagal mengambil/membuat koleksi: {e}")
            return None


# =====================================================================
# 4. LIFESPAN HANDLER (STARTUP & SHUTDOWN)
# =====================================================================
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Mengatur inisialisasi resource saat startup dan pembersihan saat shutdown."""
    logger.info("=" * 50)
    logger.info("SERVER STARTING - Loading resources")

    # 1. Load Model Embedding MiniLM
    try:
        logger.info(f"Loading Model from: {MODEL_PATH}")
        ml_models["minilm"] = SentenceTransformer(MODEL_PATH)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise

    # 2. Load Model Cross-Encoder Reranker
    try:
        logger.info(f"Loading Cross-Encoder Reranker Model from: {CE_MODEL_PATH} ...")
        _ = get_reranker(CE_MODEL_PATH)
        logger.info("✅ Cross-Encoder Reranker loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Reranker model: {str(e)}", exc_info=True)
        logger.warning("Proceeding without Reranker capabilities (Fallback mode enabled)")

    # 3. Konek ChromaDB (Menggunakan HttpClient untuk arsitektur Docker Server-Client)
    try:
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        logger.info(f"Connecting to ChromaDB Server at {chroma_host}:{chroma_port} ...")

        global chroma_client, collection
        chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

        try:
            collection = chroma_client.get_collection(name=COLLECTION_NAME)
            doc_count = collection.count()
            logger.info(f"✅ Connected to ChromaDB. Total Documents: {doc_count}")
        except Exception:
            logger.warning("Collection not found, creating new one")
            collection = chroma_client.create_collection(
                name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
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


# =====================================================================
# 5. FASTAPI APPLICATION SETUP
# =====================================================================
app = FastAPI(
    title="Sistem Pencarian Arsip Cerdas (Semantic Search Only)",
    description="REST API untuk pencarian arsip semantik, reranking, dan manajemen dokumen ChromaDB.",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=UPLOAD_FOLDER), name="files")


# =====================================================================
# 6. PYDANTIC SCHEMAS
# =====================================================================
class SearchRequest(BaseModel):
    query: str = Field(..., description="Kueri pencarian teks", min_length=1)
    top_k: int = Field(10, description="Jumlah dokumen teratas yang ingin dikembalikan", ge=1, le=50)


class UpdateDocumentRequest(BaseModel):
    title: Optional[str] = Field(None, description="Judul baru dokumen")
    content: Optional[str] = Field(None, description="Isi/deskripsi baru dokumen")
    keywords: Optional[str] = Field(None, description="Kata kunci baru dokumen")


# =====================================================================
# 7. CORE SEARCH & RETRIEVAL ENDPOINTS
# =====================================================================
@app.get("/health", summary="Health Check")
async def health_check():
    """Endpoint untuk monitoring kesehatan service, model AI, dan database."""
    try:
        model_ok = ml_models.get("minilm") is not None
        active_col = get_active_collection()
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

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time(),
        }


@app.post("/search", summary="Pencarian Semantik & Reranking")
async def search_endpoint(request: SearchRequest, http_request: Request):
    """Melakukan pencarian arsip berbasis semantik menggunakan embedding MiniLM dan Cross-Encoder Reranker."""
    _check_rate_limit(http_request, "search")
    logger.debug(f"Search request: query='{request.query}', top_k={request.top_k}")

    try:
        model = ml_models.get("minilm")
        if not model:
            logger.error("Model not loaded")
            raise ModelNotLoadedError("AI model not loaded during startup")

        active_col = get_active_collection()
        if active_col is None:
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
            logger.error(f"Database query failed: {str(e)}")
            raise

        try:
            reranker = get_reranker()
            final_results = reranker.rerank(query=request.query, chroma_results=results, top_k=request.top_k)

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

        except Exception as rerank_err:
            logger.warning(f"Reranking failed or skipped, fallback to semantic search: {rerank_err}")
            final_results = []
            if results.get("metadatas") and results["metadatas"][0]:
                for i, meta in enumerate(results["metadatas"][0]):
                    score = 1 - results["distances"][0][i]
                    raw_id = results["ids"][0][i] if "ids" in results else "unknown"
                    doc_uuid = meta.get("uuid") or raw_id
                    fname = meta.get("file_name", "")

                    real_full_text = full_docs_map.get(raw_id, meta.get("content_only", ""))

                    doc_item = dict(meta) if meta else {}
                    doc_item.update({
                        "uuid": doc_uuid,
                        "score": round(score, 4),
                        "title": meta.get("title", "Tanpa Judul"),
                        "snippet": meta.get("snippet", ""),
                        "content_only": real_full_text,
                        "document_asli": real_full_text,
                        "file_name": fname,
                    })
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
    except Exception as e:
        logger.error(f"Unexpected error in search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# =====================================================================
# 8. DOCUMENT MANAGEMENT & CRUD ENDPOINTS
# =====================================================================
@app.get("/documents", summary="[READ] Daftar Seluruh Dokumen")
async def list_documents_endpoint(
    limit: int = Query(20, ge=1, le=100, description="Jumlah dokumen per halaman"),
    offset: int = Query(0, ge=0, description="Offset dokumen"),
):
    """Mendapatkan daftar seluruh dokumen yang tersimpan di ChromaDB dengan pagination."""
    active_col = get_active_collection()
    if active_col is None:
        raise HTTPException(status_code=503, detail="Database belum terhubung")

    try:
        total_count = active_col.count()
        data = active_col.get(
            limit=limit,
            offset=offset,
            include=["metadatas", "documents"],
        )

        items = []
        if data.get("ids"):
            for i, doc_id in enumerate(data["ids"]):
                meta = data["metadatas"][i] if data.get("metadatas") else {}
                doc_text = data["documents"][i] if data.get("documents") else ""
                items.append({
                    "id": doc_id,
                    "title": meta.get("title", "Tanpa Judul"),
                    "content": meta.get("content_only", doc_text),
                    "snippet": meta.get("snippet", ""),
                    "keywords": meta.get("keywords", "-"),
                    "file_name": meta.get("file_name", "-"),
                })

        return {
            "status": "success",
            "total_documents": total_count,
            "limit": limit,
            "offset": offset,
            "documents": items,
        }
    except Exception as e:
        logger.error(f"Failed to fetch documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gagal mengambil data dokumen: {str(e)}")


@app.get("/documents/{doc_id}", summary="[READ] Detail Satu Dokumen")
async def get_document_endpoint(
    http_request: Request,
    doc_id: str = Path(..., description="ID Dokumen yang dicari"),
):
    """Mendapatkan detail satu dokumen berdasarkan ID."""
    active_col = get_active_collection()
    if active_col is None:
        raise HTTPException(status_code=503, detail="Database belum terhubung")

    try:
        data = active_col.get(ids=[doc_id], include=["metadatas", "documents"])
        if not data.get("ids") or len(data["ids"]) == 0:
            raise HTTPException(status_code=404, detail=f"Dokumen dengan ID '{doc_id}' tidak ditemukan")

        meta = data["metadatas"][0] if data.get("metadatas") else {}
        doc_text = data["documents"][0] if data.get("documents") else ""
        fname = meta.get("file_name", "-")

        return {
            "status": "success",
            "data": {
                "id": doc_id,
                "title": meta.get("title", ""),
                "content": meta.get("content_only", doc_text),
                "snippet": meta.get("snippet", ""),
                "keywords": meta.get("keywords", "-"),
                "file_name": fname,
                "download_url": _build_file_url(http_request, fname),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gagal membaca dokumen: {str(e)}")


@app.post("/upload", summary="[CREATE] Upload & Index Dokumen Tunggal")
async def upload_endpoint(
    http_request: Request,
    title: str = Form(..., description="Judul dokumen"),
    content: str = Form(..., description="Isi/deskripsi dokumen"),
    keywords: Optional[str] = Form(None, description="Kata kunci dokumen (opsional)"),
    file: Optional[UploadFile] = None,
):
    """Mengunggah dan meng-indeks satu dokumen beserta judul, konten/deskripsi, kata kunci, dan file lampiran."""
    _check_rate_limit(http_request, "upload")
    logger.debug(f"Upload request: title='{title}', file={file.filename if file else 'None'}")

    try:
        model = ml_models.get("minilm")
        if not model:
            logger.error("Model not loaded for upload")
            raise ModelNotLoadedError("AI model not loaded")
        active_col = get_active_collection()
        if active_col is None:
            logger.error("Database not connected for upload")
            raise DatabaseNotConnectedError("Database not connected")

        safe_filename = "-"
        file_path = "-"
        extracted_text = ""

        if file and file.filename:
            safe_filename = file.filename.replace(" ", "_")
            file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
            logger.debug(f"Saving file to: {file_path}")

            file_content_bytes = await file.read()
            with open(file_path, "wb") as f:
                f.write(file_content_bytes)

            logger.info(f"File saved: {safe_filename}")
            logger.debug(f"Extracting text from file: {safe_filename}")
            extracted_text = extract_text_from_file(file_path)

        # Penentuan kata kunci (jika tidak diisi, ekstrak otomatis dari konten atau default '-')
        final_keywords = keywords.strip() if keywords and keywords.strip() else extract_keywords(content)

        full_text_to_embed = f"{title}. {content}. kata kunci: {final_keywords}. {extracted_text}".strip()
        logger.debug(f"Encoding text ({len(full_text_to_embed)} chars)")
        vector = model.encode(full_text_to_embed).tolist()

        doc_id = f"upload_{int(time.time())}"

        logger.debug(f"Adding document to ChromaDB: {doc_id}")
        active_col.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[full_text_to_embed],
            metadatas=[{
                "title": title,
                "file_name": safe_filename,
                "file_path": file_path,
                "keywords": final_keywords,
                "snippet": content if content else (extracted_text[:200] if extracted_text else "-"),
                "content_only": full_text_to_embed,
            }],
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
            "keywords": final_keywords,
            "file_url": _build_file_url(http_request, safe_filename) if safe_filename != "-" else "",
        }

    except FileExtractionError as e:
        logger.error(f"File extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"File extraction failed: {str(e)}")
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/upload/jsonl", summary="[CREATE] Upload & Index File JSONL Massal")
async def upload_jsonl_endpoint(
    http_request: Request,
    file: UploadFile = File(..., description="File dataset berformat .jsonl (1 baris = 1 objek JSON)"),
    meta_file: Optional[UploadFile] = File(None, description="File metadata berformat .jsonl (opsional)"),
):
    """Mengunggah file .jsonl (seperti train.jsonl & metadata.jsonl) dan meng-indeks seluruh dokumen ke ChromaDB secara batch."""
    _check_rate_limit(http_request, "upload")
    logger.info(f"JSONL upload request received: {file.filename}")

    if not file.filename.endswith((".jsonl", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Format file tidak didukung. Harap unggah file dengan ekstensi .jsonl",
        )

    model = ml_models.get("minilm")
    if not model:
        logger.error("Model not loaded for jsonl upload")
        raise ModelNotLoadedError("AI model not loaded")
    active_col = get_active_collection()
    if active_col is None:
        logger.error("Database not connected for jsonl upload")
        raise DatabaseNotConnectedError("Database not connected")

    try:
        content_bytes = await file.read()
        lines = content_bytes.decode("utf-8", errors="replace").splitlines()

        # Baca file metadata jika disertakan
        metadata_map = {}
        if meta_file and meta_file.filename:
            meta_bytes = await meta_file.read()
            meta_lines = meta_bytes.decode("utf-8", errors="replace").splitlines()
            for idx, m_line in enumerate(meta_lines):
                if m_line.strip():
                    try:
                        m_data = json.loads(m_line)
                        metadata_map[idx] = m_data
                    except json.JSONDecodeError:
                        continue

        items_to_upload = []
        for idx, line in enumerate(lines):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as json_err:
                logger.warning(f"Skipping invalid JSON line {idx}: {json_err}")
                continue

            title = data.get("title", f"Dokumen {idx + 1}")
            content = data.get("content", "")
            kw = data.get("keywords") or extract_keywords(content)
            doc_id = data.get("id") or data.get("doc_id") or f"doc_{int(time.time())}_{idx}"
            file_name = data.get("file_name", "-")
            file_path = data.get("file_path", "-")
            extracted_text = data.get("extracted_text", "")

            # Gabungkan dengan metadata tambahan jika ada
            if idx in metadata_map:
                meta_item = metadata_map[idx]
                title = meta_item.get("title", title)
                file_name = meta_item.get("file_name", file_name)
                file_path = meta_item.get("file_path", file_path)
                kw = meta_item.get("keywords", kw)

            full_text_to_embed = f"{title}. {content}. kata kunci: {kw}. {extracted_text}".strip()
            snippet_text = content if content else (extracted_text[:300] if extracted_text else "-")

            items_to_upload.append(
                DocumentItem(
                    doc_id=str(doc_id),
                    title=title,
                    content=content,
                    file_name=file_name,
                    file_path=file_path,
                    full_text=full_text_to_embed,
                    snippet=snippet_text,
                )
            )

        if not items_to_upload:
            raise HTTPException(status_code=400, detail="Tidak ada data dokumen valid yang ditemukan dalam file .jsonl.")

        logger.info(f"Memulai proses batch upload untuk {len(items_to_upload)} dokumen JSONL...")
        uploader = create_batch_uploader(model, active_col)
        result = uploader.upload_batch(items_to_upload, skip_existing=False)

        if result.success > 0:
            try:
                cache = get_cache_manager()
                cleared = cache.clear()
                logger.info(f"Cache cleared after JSONL upload: {cleared} entries removed")
            except Exception as cache_err:
                logger.warning(f"Cache clear warning (non-fatal): {cache_err}")

        return {
            "status": "success",
            "message": f"Upload JSONL selesai. {result.success} dokumen berhasil di-index, {result.failed} gagal.",
            "total_documents": active_col.count(),
            "summary": result.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"JSONL upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gagal memproses file JSONL: {str(e)}")


@app.put("/documents/{doc_id}", summary="[UPDATE] Perbarui Data Dokumen")
async def update_document_endpoint(
    http_request: Request,
    doc_id: str = Path(..., description="ID Dokumen yang akan diperbarui"),
    update_data: UpdateDocumentRequest = ...,
):
    """Memperbarui judul, konten, atau kata kunci dokumen dan otomatis meng-update vektor embedding-nya."""
    _check_rate_limit(http_request, "upload")
    active_col = get_active_collection()
    if active_col is None:
        raise HTTPException(status_code=503, detail="Database belum terhubung")
    model = ml_models.get("minilm")
    if not model:
        raise HTTPException(status_code=503, detail="AI model belum dimuat")

    try:
        existing = active_col.get(ids=[doc_id], include=["metadatas", "documents"])
        if not existing.get("ids") or len(existing["ids"]) == 0:
            raise HTTPException(status_code=404, detail=f"Dokumen dengan ID '{doc_id}' tidak ditemukan")

        old_meta = existing["metadatas"][0] if existing.get("metadatas") else {}
        old_doc = existing["documents"][0] if existing.get("documents") else ""

        new_title = update_data.title if update_data.title is not None else old_meta.get("title", "")
        new_content = update_data.content if update_data.content is not None else old_meta.get("content", old_doc)
        new_keywords = (
            update_data.keywords
            if update_data.keywords is not None
            else old_meta.get("keywords", extract_keywords(new_content))
        )

        file_name = old_meta.get("file_name", "-")
        file_path = old_meta.get("file_path", "-")

        new_text_to_embed = f"{new_title}. {new_content}. kata kunci: {new_keywords}".strip()
        new_vector = model.encode(new_text_to_embed).tolist()

        updated_meta = {
            "title": new_title,
            "content": new_content,
            "snippet": new_content[:200] if new_content else "",
            "content_only": new_text_to_embed,
            "file_name": file_name,
            "file_path": file_path,
            "keywords": new_keywords,
        }

        active_col.update(
            ids=[doc_id],
            embeddings=[new_vector],
            documents=[new_text_to_embed],
            metadatas=[updated_meta],
        )

        try:
            get_cache_manager().clear()
        except Exception:
            pass

        return {
            "status": "success",
            "message": f"Dokumen '{doc_id}' berhasil diperbarui.",
            "data": {
                "id": doc_id,
                "title": new_title,
                "keywords": new_keywords,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gagal memperbarui dokumen: {str(e)}")


@app.delete("/documents/{doc_id}", summary="[DELETE] Hapus Satu Dokumen")
async def delete_document_endpoint(
    http_request: Request,
    doc_id: str = Path(..., description="ID Dokumen yang akan dihapus"),
):
    """Menghapus dokumen tunggal berdasarkan ID."""
    _check_rate_limit(http_request, "delete")
    logger.debug(f"Delete request for document: {doc_id}")

    active_col = get_active_collection()
    if active_col is None:
        logger.error("Database not connected for delete")
        raise HTTPException(status_code=503, detail="Database belum terhubung")

    try:
        existing = active_col.get(ids=[doc_id])
        if not existing["ids"]:
            raise HTTPException(status_code=404, detail=f"Dokumen dengan ID '{doc_id}' tidak ditemukan.")

        active_col.delete(ids=[doc_id])
        logger.info(f"Document deleted successfully: {doc_id}")

        try:
            cache = get_cache_manager()
            cleared = cache.clear()
            logger.info(f"Cache cleared after delete: {cleared} entries removed")
        except Exception as cache_err:
            logger.warning(f"Cache clear warning (non-fatal): {cache_err}")

        return {
            "status": "success",
            "message": f"Dokumen dengan ID '{doc_id}' berhasil dihapus permanen.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gagal menghapus: {str(e)}")


@app.delete("/documents", summary="[DELETE ALL] Reset / Kosongkan Database")
async def reset_all_documents_endpoint(
    http_request: Request,
    confirm: bool = Query(False, description="Set True untuk mengonfirmasi penghapusan seluruh data"),
):
    """Mengosongkan/menghapus seluruh dokumen dalam database vektor."""
    global collection
    _check_rate_limit(http_request, "delete")

    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Operasi berbahaya: Tambahkan parameter '?confirm=true' untuk mereset seluruh database.",
        )

    if chroma_client is None:
        raise HTTPException(status_code=503, detail="Database belum terhubung")

    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

        try:
            get_cache_manager().clear()
        except Exception:
            pass

        return {
            "status": "success",
            "message": "Seluruh dokumen dalam koleksi berhasil direset/dikosongkan.",
            "total_documents": 0,
        }
    except Exception as e:
        logger.error(f"Failed to reset collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gagal mereset koleksi: {str(e)}")


# =====================================================================
# 9. MONITORING, METRICS, CACHE & SYSTEM STATUS ENDPOINTS
# =====================================================================
@app.get("/metrics", summary="Metrik Performa API")
async def get_metrics():
    """Mengambil ringkasan metrik performa API, waktu respons, dan rate limiting."""
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
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


@app.get("/status", summary="Status Kesehatan Komprehensif")
async def get_system_status():
    """Mengambil status operasional seluruh komponen sistem (Model, DB, Cache, Rate Limiter)."""
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
                    "name": MODEL_PATH,
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
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@app.get("/cache/stats", summary="Statistik Cache")
async def get_cache_stats():
    """Melihat status ukuran dan efektivitas cache pencarian."""
    logger.debug("Cache stats endpoint accessed")
    try:
        cache = get_cache_manager()
        stats = cache.get_stats()
        return {
            "status": "success",
            "timestamp": time.time(),
            "cache": stats,
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get cache stats")


@app.post("/cache/clear", summary="Bersihkan Cache")
async def clear_cache():
    """Membersihkan seluruh entri cache secara manual."""
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
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


# =====================================================================
# 10. ROOT ENDPOINT
# =====================================================================
@app.get("/", summary="Root API Information")
def home():
    """Endpoint informasi status dan dokumentasi API."""
    logger.debug("Root endpoint accessed")
    return {
        "message": "API Semantic Search & Reranker Aktif (No LLM)!",
        "docs": "/docs",
        "version": "1.0.0",
        "status": "operational",
    }