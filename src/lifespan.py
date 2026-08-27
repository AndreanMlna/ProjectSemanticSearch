import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from src.cache_manager import get_cache_manager
from src.chroma_client import get_collection, init_chroma_client
from src.config import CE_MODEL_PATH, MODEL_PATH
from src.logging_utils import setup_logging
from src.rate_limiter import get_endpoint_rate_limiter
from src.reranker import get_reranker
from src.sync_seranah_archives import check_and_sync

logger = setup_logging("lifespan")

ml_models: dict[str, Any] = {}


async def _auto_sync_background_worker():
    """
    Background worker untuk memeriksa kesesuaian dataset dengan API live SERANAH secara berkala (setiap 2 jam).
    Berjalan secara non-blocking di background thread.
    """
    # Tunggu 30 detik setelah startup agar inisialisasi awal server dan database selesai
    await asyncio.sleep(30)
    interval_hours = float(os.getenv("SYNC_INTERVAL_HOURS", "2"))
    interval_seconds = max(300, int(interval_hours * 3600))  # Minimal 5 menit

    logger.info(f"🔄 [AutoSync Daemon] Background sync aktif. Interval: {interval_hours} jam.")

    while True:
        try:
            logger.info("⏰ [AutoSync] Menjalankan pengecekan dataset berkala dengan Live API SERANAH...")
            loop = asyncio.get_running_loop()
            # Jalankan di thread terpisah agar request pencarian pengguna tidak terblokir
            result = await loop.run_in_executor(None, check_and_sync, True)
            if result.get("updated"):
                logger.info(f"⚡ [AutoSync] Dataset dan ChromaDB berhasil diperbarui ({result.get('local_count')} dokumen).")
            else:
                logger.info(f"✅ [AutoSync] Dataset sudah sinkron ({result.get('local_count')} dokumen). Tidak ada perubahan.")
        except asyncio.CancelledError:
            logger.info("[AutoSync] Background worker dimatikan.")
            break
        except Exception as e:
            logger.error(f"[!] [AutoSync Error] Kesalahan saat sinkronisasi dataset: {e}")

        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("=" * 50)
    logger.info("SERVER STARTING - Loading resources")

    try:
        logger.info(f"Loading Model from: {MODEL_PATH}")
        ml_models["minilm"] = SentenceTransformer(MODEL_PATH)
        logger.info("✅ Model loaded successfully")
    except (ValueError, RuntimeError, AttributeError):
        logger.exception("Failed to load model")
        raise

    try:
        logger.info(f"Loading Cross-Encoder Reranker Model from: {CE_MODEL_PATH} ...")
        _x = get_reranker(CE_MODEL_PATH)
        logger.info("✅ Cross-Encoder Reranker loaded successfully")
    except (ValueError, RuntimeError, AttributeError):
        logger.exception("Failed to load Reranker model")
        logger.warning(
            "Proceeding without Reranker capabilities (Fallback mode enabled)"
        )

    try:
        init_chroma_client()
        col = get_collection()
        doc_count = col.count() if col else 0
        logger.info(f"✅ Connected to ChromaDB. Total Documents: {doc_count}")
    except (ValueError, RuntimeError, AttributeError):
        logger.exception("Failed to connect ChromaDB")
        raise

    try:
        cache = get_cache_manager()
        logger.info(
            f"✅ Cache initialized: max_size={cache.max_size}, ttl={cache.ttl_seconds // 60}min"
        )
    except (ImportError, AttributeError, ValueError) as e:
        logger.warning(f"Cache initialization warning: {e!s}")

    try:
        rl = get_endpoint_rate_limiter()
        logger.info(
            f"✅ Rate limiter initialized: search={rl.limits.get('search')}/min, upload={rl.limits.get('upload')}/min, delete={rl.limits.get('delete')}/min"
        )
    except (ImportError, AttributeError, ValueError) as e:
        logger.warning(f"Rate limiter initialization warning: {e!s}")

    # Luncurkan AutoSync background worker
    sync_task = asyncio.create_task(_auto_sync_background_worker())

    yield

    logger.info("Cleaning up resources")
    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass

    ml_models.clear()
    logger.info("SERVER STOPPED - Resources cleaned up")


def get_ml_models() -> dict[str, Any]:
    return ml_models

