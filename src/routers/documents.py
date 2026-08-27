import json
import os
import time
try:
    import aiofiles
except ImportError:
    aiofiles = None

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)

from src.batch_processor import DocumentItem, create_batch_uploader
from src.cache_manager import get_cache_manager
from src.chroma_client import get_collection, reset_collection
from src.config import UPLOAD_FOLDER
from src.error_handler import (
    DatabaseNotConnectedError,
    FileExtractionError,
    ModelNotLoadedError,
)
from src.helpers import build_file_url, check_rate_limit, extract_keywords
from src.lifespan import get_ml_models
from src.logging_utils import setup_logging
from src.schemas import UpdateDocumentRequest
from src.text_extractor import extract_text_from_file

router = APIRouter()
logger = setup_logging("documents_router")


@router.get("/documents", summary="[READ] Daftar Seluruh Dokumen")
async def list_documents_endpoint(
    limit: int = Query(20, ge=1, le=100, description="Jumlah dokumen per halaman"),
    offset: int = Query(0, ge=0, description="Offset dokumen"),
):
    active_col = get_collection()
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
                items.append(
                    {
                        "id": doc_id,
                        "title": meta.get("title", "Tanpa Judul"),
                        "content": meta.get("content_only", doc_text),
                        "snippet": meta.get("snippet", ""),
                        "keywords": meta.get("keywords", "-"),
                        "file_name": meta.get("file_name", "-"),
                    }
                )

        return {
            "status": "success",
            "total_documents": total_count,
            "limit": limit,
            "offset": offset,
            "documents": items,
        }
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.exception("Failed to fetch documents")
        raise HTTPException(
            status_code=500, detail=f"Gagal mengambil data dokumen: {e!s}"
        )


@router.get("/documents/{doc_id}", summary="[READ] Detail Satu Dokumen")
async def get_document_endpoint(
    http_request: Request,
    doc_id: str = Path(..., description="ID Dokumen yang dicari"),
):
    active_col = get_collection()
    if active_col is None:
        raise HTTPException(status_code=503, detail="Database belum terhubung")

    try:
        data = active_col.get(ids=[doc_id], include=["metadatas", "documents"])
        if not data.get("ids") or len(data["ids"]) == 0:
            raise HTTPException(
                status_code=404, detail=f"Dokumen dengan ID '{doc_id}' tidak ditemukan"
            )

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
                "download_url": build_file_url(http_request, fname),
            },
        }
    except HTTPException:
        raise
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.exception("Error reading document")
        raise HTTPException(status_code=500, detail=f"Gagal membaca dokumen: {e!s}")


@router.post("/upload", summary="[CREATE] Upload & Index Dokumen Tunggal")
async def upload_endpoint(
    http_request: Request,
    title: str = Form(..., description="Judul dokumen"),
    content: str = Form(..., description="Isi/deskripsi dokumen"),
    keywords: str | None = Form(None, description="Kata kunci dokumen (opsional)"),
    file: UploadFile | None = None,
):
    check_rate_limit(http_request, "upload")
    logger.debug(
        f"Upload request: title='{title}', file={file.filename if file else 'None'}"
    )

    try:
        model = get_ml_models().get("minilm")
        if not model:
            logger.error("Model not loaded for upload")
            raise ModelNotLoadedError("AI model not loaded")
        active_col = get_collection()
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

            if aiofiles is not None:
                file_content_bytes = await file.read()
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(file_content_bytes)
            else:
                file_content_bytes = await file.read()
                with open(file_path, "wb") as f:
                    f.write(file_content_bytes)

            logger.info(f"File saved: {safe_filename}")
            logger.debug(f"Extracting text from file: {safe_filename}")
            extracted_text = extract_text_from_file(file_path)

        final_keywords = (
            keywords.strip()
            if keywords and keywords.strip()
            else extract_keywords(content)
        )

        full_text_to_embed = f"{title}. {content}. kata kunci: {final_keywords}. {extracted_text}".strip()
        logger.debug(f"Encoding text ({len(full_text_to_embed)} chars)")
        vector = model.encode(full_text_to_embed).tolist()

        doc_id = f"upload_{int(time.time())}"

        logger.debug(f"Adding document to ChromaDB: {doc_id}")
        active_col.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[full_text_to_embed],
            metadatas=[
                {
                    "uuid": doc_id,
                    "title": title,
                    "file_name": safe_filename,
                    "file_path": file_path,
                    "keywords": final_keywords,
                    "snippet": (
                        content
                        if content
                        else (extracted_text[:200] if extracted_text else "-")
                    ),
                    "content": content,
                    "content_only": full_text_to_embed,
                    "document_number": "-",
                    "year": "-",
                    "category": "-",
                    "access_level": "PUBLIC",
                    "mime_type": "application/pdf" if safe_filename.endswith(".pdf") else "text/plain",
                    "uploader": "-",
                    "unit_kerja": "-",
                }
            ],
        )

        logger.info(f"Document uploaded successfully: {title} (ID: {doc_id})")

        try:
            cache = get_cache_manager()
            cleared = cache.clear()
            logger.info(f"Cache cleared after upload: {cleared} entries removed")
        except (ImportError, AttributeError, ValueError) as cache_err:
            logger.warning(f"Cache clear warning (non-fatal): {cache_err!s}")

        return {
            "status": "success",
            "message": f"Dokumen '{title}' berhasil disimpan & di-index!",
            "doc_id": doc_id,
            "keywords": final_keywords,
            "file_url": (
                build_file_url(http_request, safe_filename)
                if safe_filename != "-"
                else ""
            ),
        }

    except FileExtractionError as e:
        logger.error(f"File extraction error: {e!s}")
        raise HTTPException(status_code=400, detail=f"File extraction failed: {e!s}")
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e!s}")


@router.post("/upload/jsonl", summary="[CREATE] Upload & Index File JSONL Massal")
async def upload_jsonl_endpoint(
    http_request: Request,
    file: UploadFile = File(..., description="File dataset berformat .jsonl (1 baris = 1 objek JSON)"),
    meta_file: UploadFile | None = File(None, description="File metadata berformat .jsonl (opsional)"),
):
    check_rate_limit(http_request, "upload")
    logger.info(f"JSONL upload request received: {file.filename}")

    if not file.filename or not file.filename.endswith((".jsonl", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Format file tidak didukung. Harap unggah file dengan ekstensi .jsonl",
        )

    model = get_ml_models().get("minilm")
    if not model:
        logger.error("Model not loaded for jsonl upload")
        raise ModelNotLoadedError("AI model not loaded")
    active_col = get_collection()
    if active_col is None:
        logger.error("Database not connected for jsonl upload")
        raise DatabaseNotConnectedError("Database not connected")

    try:
        content_bytes = await file.read()
        lines = content_bytes.decode("utf-8", errors="replace").splitlines()

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
            doc_id = (
                data.get("id") or data.get("doc_id") or data.get("uuid") or f"doc_{int(time.time())}_{idx}"
            )
            file_name = data.get("file_name", "-")
            file_path = data.get("file_path", "-")
            extracted_text = data.get("extracted_text", "")

            if idx in metadata_map:
                meta_item = metadata_map[idx]
                title = meta_item.get("title", title)
                file_name = meta_item.get("file_name", file_name)
                file_path = meta_item.get("file_path", file_path)
                kw = meta_item.get("keywords", kw)

            full_text_to_embed = (
                f"{title}. {content}. kata kunci: {kw}. {extracted_text}".strip()
            )
            snippet_text = (
                content
                if content
                else (extracted_text[:300] if extracted_text else "-")
            )

            items_to_upload.append(
                DocumentItem(
                    doc_id=str(doc_id),
                    title=title,
                    content=content,
                    file_name=file_name,
                    file_path=file_path,
                    full_text=full_text_to_embed,
                    snippet=snippet_text,
                    keywords=kw,
                    document_number=str(data.get("document_number", "-")),
                    year=str(data.get("year", "-")),
                    category=str(data.get("category", "-")),
                )
            )

        if not items_to_upload:
            raise HTTPException(
                status_code=400,
                detail="Tidak ada data dokumen valid yang ditemukan dalam file .jsonl.",
            )

        logger.info(
            f"Memulai proses batch upload untuk {len(items_to_upload)} dokumen JSONL..."
        )
        uploader = create_batch_uploader(model, active_col)
        result = uploader.upload_batch(items_to_upload, skip_existing=False)

        if result.success > 0:
            try:
                cache = get_cache_manager()
                cleared = cache.clear()
                logger.info(
                    f"Cache cleared after JSONL upload: {cleared} entries removed"
                )
            except (ImportError, AttributeError, ValueError) as cache_err:
                logger.warning(f"Cache clear warning (non-fatal): {cache_err!s}")

        return {
            "status": "success",
            "message": f"Upload JSONL selesai. {result.success} dokumen berhasil di-index, {result.failed} gagal.",
            "total_documents": active_col.count(),
            "summary": result.to_dict(),
        }

    except HTTPException:
        raise
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.exception("JSONL upload failed")
        raise HTTPException(
            status_code=500, detail=f"Gagal memproses file JSONL: {e!s}"
        )


@router.put("/documents/{doc_id}", summary="[UPDATE] Perbarui Data Dokumen")
async def update_document_endpoint(
    http_request: Request,
    update_data: UpdateDocumentRequest,
    doc_id: str = Path(..., description="ID Dokumen yang akan diperbarui"),
):
    check_rate_limit(http_request, "upload")
    active_col = get_collection()
    if active_col is None:
        raise HTTPException(status_code=503, detail="Database belum terhubung")
    model = get_ml_models().get("minilm")
    if not model:
        raise HTTPException(status_code=503, detail="AI model belum dimuat")

    try:
        existing = active_col.get(ids=[doc_id], include=["metadatas", "documents"])
        if not existing.get("ids") or len(existing["ids"]) == 0:
            raise HTTPException(
                status_code=404, detail=f"Dokumen dengan ID '{doc_id}' tidak ditemukan"
            )

        old_meta = existing["metadatas"][0] if existing.get("metadatas") else {}
        old_doc = existing["documents"][0] if existing.get("documents") else ""

        new_title = (
            update_data.title
            if update_data.title is not None
            else old_meta.get("title", "")
        )
        new_content = (
            update_data.content
            if update_data.content is not None
            else old_meta.get("content", old_doc)
        )
        new_keywords = (
            update_data.keywords
            if update_data.keywords is not None
            else old_meta.get("keywords", extract_keywords(new_content))
        )

        file_name = old_meta.get("file_name", "-")
        file_path = old_meta.get("file_path", "-")

        new_text_to_embed = (
            f"{new_title}. {new_content}. kata kunci: {new_keywords}".strip()
        )
        new_vector = model.encode(new_text_to_embed).tolist()

        updated_meta = {
            **old_meta,
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
        except (ImportError, AttributeError, ValueError):
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
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.exception("Update error")
        raise HTTPException(status_code=500, detail=f"Gagal memperbarui dokumen: {e!s}")


@router.delete("/documents/{doc_id}", summary="[DELETE] Hapus Satu Dokumen")
async def delete_document_endpoint(
    http_request: Request,
    doc_id: str = Path(..., description="ID Dokumen yang akan dihapus"),
):
    check_rate_limit(http_request, "delete")
    logger.debug(f"Delete request for document: {doc_id}")

    active_col = get_collection()
    if active_col is None:
        logger.error("Database not connected for delete")
        raise HTTPException(status_code=503, detail="Database belum terhubung")

    try:
        existing = active_col.get(ids=[doc_id])
        if not existing["ids"]:
            raise HTTPException(
                status_code=404, detail=f"Dokumen dengan ID '{doc_id}' tidak ditemukan."
            )

        active_col.delete(ids=[doc_id])
        logger.info(f"Document deleted successfully: {doc_id}")

        try:
            cache = get_cache_manager()
            cleared = cache.clear()
            logger.info(f"Cache cleared after delete: {cleared} entries removed")
        except (ImportError, AttributeError, ValueError) as cache_err:
            logger.warning(f"Cache clear warning (non-fatal): {cache_err!s}")

        return {
            "status": "success",
            "message": f"Dokumen dengan ID '{doc_id}' berhasil dihapus permanen.",
        }

    except HTTPException:
        raise
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.exception("Delete error")
        raise HTTPException(status_code=500, detail=f"Gagal menghapus: {e!s}")


@router.delete("/documents", summary="[DELETE ALL] Reset / Kosongkan Database")
async def reset_all_documents_endpoint(
    http_request: Request,
    confirm: bool = Query(
        False, description="Set True untuk mengonfirmasi penghapusan seluruh data"
    ),
):
    check_rate_limit(http_request, "delete")

    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Operasi berbahaya: Tambahkan parameter '?confirm=true' untuk mereset seluruh database.",
        )

    try:
        reset_collection()

        try:
            get_cache_manager().clear()
        except (ImportError, AttributeError, ValueError):
            pass

        return {
            "status": "success",
            "message": "Seluruh dokumen dalam koleksi berhasil direset/dikosongkan.",
            "total_documents": 0,
        }
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.exception("Failed to reset collection")
        raise HTTPException(status_code=500, detail=f"Gagal mereset koleksi: {e!s}")


@router.post("/documents/sync", summary="[SYNC] Sinkronisasi Manual dengan Live API SERANAH")
async def trigger_manual_sync_endpoint(
    auto_reindex: bool = Query(True, description="Jalankan prepare dan re-indexing jika ada data baru")
):
    """
    Memicu pengecekan dan sinkronisasi dataset secara instan dengan API live SERANAH
    tanpa harus menunggu siklus otomatis 2 jam.
    """
    try:
        import asyncio
        from src.sync_seranah_archives import check_and_sync
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, check_and_sync, auto_reindex)
        return result
    except Exception as e:
        logger.exception("Manual sync failed")
        raise HTTPException(status_code=500, detail=f"Gagal melakukan sinkronisasi: {e!s}")
