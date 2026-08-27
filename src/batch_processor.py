"""
src/batch_processor.py
======================
Memproses banyak dokumen arsip sekaligus secara efisien (Batch Processing).

Dua kegunaan utama sesuai arsitektur project ini:
1. BatchEmbedder         — Generate embedding banyak teks sekaligus
                           (lebih cepat dan optimal dibanding encode satu-per-satu)
2. BatchDocumentUploader — Upload/index banyak dokumen ke ChromaDB sekaligus
                           (berguna untuk bulk import arsip kampus dan migrasi)

Hubungan dan Keselarasan dengan file lain:
- Metadata skema 100% selaras dengan sync_seranah_archives.py dan routers/documents.py
- Menggunakan SentenceTransformer model (ml_models["minilm"])
- Menggunakan ChromaDB collection yang sama (arsip_kampus)
- Cache invalidation via cache_manager.py setelah upload/delete
"""

import time
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("batch_processor")


# ═══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DocumentItem:
    """
    Satu dokumen yang akan di-batch upload.

    Struktur ini 100% selaras dengan skema metadata ChromaDB di:
    - src/sync_seranah_archives.py
    - src/routers/documents.py
    - src/routers/search.py
    """
    doc_id: str                      # ID unik / UUID dokumen
    title: str                       # Judul dokumen arsip
    content: str                     # Deskripsi / isi singkat
    file_name: str = "-"             # Nama file fisik
    file_path: str = "-"             # Path lengkap file di server
    full_text: str = ""              # Gabungan title + content + keywords untuk embedding
    snippet: str = ""                # Cuplikan untuk ditampilkan di hasil search
    keywords: str = "-"              # Kata kunci dokumen
    document_number: str = "-"       # Nomor dokumen resmi / SK
    year: str = "-"                  # Tahun dokumen
    category: str = "-"              # Kategori dokumen
    mime_type: str = "application/pdf"
    access_level: str = "PUBLIC"     # Tingkat akses arsip
    uploader: str = "-"              # Pengunggah dokumen
    unit_kerja: str = "-"            # Unit kerja penerbit dokumen
    extra_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Auto-generate full_text jika belum diisi
        if not self.full_text:
            if self.keywords and self.keywords != "-":
                self.full_text = f"{self.title}. {self.content}. kata kunci: {self.keywords}".strip()
            else:
                self.full_text = f"{self.title}. {self.content}".strip()

        # Auto-generate snippet jika belum diisi
        if not self.snippet:
            self.snippet = self.content[:200] if self.content else self.title[:200]

    def to_metadata(self) -> Dict[str, Any]:
        """Konversi ke format metadata standar ChromaDB."""
        meta = {
            "uuid": str(self.doc_id),
            "title": str(self.title).strip(),
            "content": str(self.content),
            "snippet": str(self.snippet),
            "content_only": str(self.content or self.full_text),
            "document_number": str(self.document_number),
            "year": str(self.year),
            "file_name": str(self.file_name),
            "file_path": str(self.file_path),
            "mime_type": str(self.mime_type),
            "access_level": str(self.access_level),
            "uploader": str(self.uploader),
            "unit_kerja": str(self.unit_kerja),
            "category": str(self.category),
            "keywords": str(self.keywords),
        }
        if self.extra_metadata and isinstance(self.extra_metadata, dict):
            for k, v in self.extra_metadata.items():
                if k not in meta and v is not None:
                    meta[k] = str(v)
        return meta


@dataclass
class BatchResult:
    """Hasil dari satu operasi batch"""
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return round(self.success / self.total * 100, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate_percent": self.success_rate,
            "duration_seconds": round(self.duration_seconds, 4),
            "errors": self.errors
        }


# ═══════════════════════════════════════════════════════════════════
# BATCH EMBEDDER
# ═══════════════════════════════════════════════════════════════════

class BatchEmbedder:
    """
    Generate embedding untuk banyak teks sekaligus menggunakan
    SentenceTransformer yang sama dengan main_api.py dan sync_seranah_archives.py.
    """

    def __init__(self, model, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        logger.info(f"BatchEmbedder initialized: batch_size={batch_size}")

    def encode_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[List[float]]:
        """
        Encode daftar teks menjadi daftar vektor embedding.
        """
        if not texts:
            raise ValueError("texts tidak boleh kosong")

        if self.model is None:
            raise RuntimeError("Model belum di-load. Pastikan server sudah startup.")

        total = len(texts)
        all_embeddings: List[List[float]] = []
        start_time = time.perf_counter()

        logger.info(f"Starting batch encode: {total} texts, batch_size={self.batch_size}")

        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch_texts = texts[batch_start:batch_end]

            try:
                batch_vectors = self.model.encode(
                    batch_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                all_embeddings.extend(batch_vectors.tolist())

                if progress_callback:
                    progress_callback(batch_end, total)

                logger.debug(f"Batch encoded: {batch_start + 1}-{batch_end}/{total}")

            except Exception as e:
                logger.error(
                    f"Batch encode failed at index {batch_start}-{batch_end}: {e}",
                    exc_info=True
                )
                raise RuntimeError(
                    f"Gagal encode batch [{batch_start}:{batch_end}]: {str(e)}"
                ) from e

        duration = time.perf_counter() - start_time
        avg_ms = (duration / total * 1000) if total > 0 else 0

        logger.info(
            f"Batch encode complete: {total} texts in {duration:.4f}s "
            f"(avg {avg_ms:.1f}ms/text)"
        )

        return all_embeddings

    def encode_single(self, text: str) -> List[float]:
        """Encode satu teks — wrapper convenience untuk konsistensi."""
        if self.model is None:
            raise RuntimeError("Model belum di-load.")
        return self.model.encode(text).tolist()


# ═══════════════════════════════════════════════════════════════════
# BATCH DOCUMENT UPLOADER
# ═══════════════════════════════════════════════════════════════════

class BatchDocumentUploader:
    """
    Upload dan index banyak dokumen arsip ke ChromaDB sekaligus.
    Selaras dengan sync_seranah_archives.py dan endpoint /upload/jsonl.
    """

    def __init__(self, model, collection, batch_size: int = 64):
        self.embedder = BatchEmbedder(model=model, batch_size=batch_size)
        self.collection = collection
        self.batch_size = batch_size
        logger.info(f"BatchDocumentUploader initialized: batch_size={batch_size}")

    def _get_existing_ids(self, doc_ids: List[str]) -> set:
        """Cek ID mana saja yang sudah ada di ChromaDB."""
        if self.collection is None:
            return set()
        try:
            existing = self.collection.get(ids=doc_ids)
            return set(existing.get("ids", []))
        except Exception as e:
            logger.debug(f"Could not fetch existing IDs from ChromaDB: {e}")
            return set()

    def upload_batch(
        self,
        documents: List[DocumentItem],
        skip_existing: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """
        Upload dan index daftar dokumen ke ChromaDB.
        """
        result = BatchResult(total=len(documents))
        start_time = time.perf_counter()

        if not documents:
            logger.warning("upload_batch dipanggil dengan list kosong")
            result.duration_seconds = 0.0
            return result

        if self.collection is None:
            raise RuntimeError("Koleksi ChromaDB belum diinisialisasi atau tidak tersedia.")

        logger.info(f"Starting batch upload: {len(documents)} documents")

        # ── Step 1: Filter duplicate ──────────────────────────────
        docs_to_process = documents
        if skip_existing:
            all_ids = [str(doc.doc_id) for doc in documents]
            existing_ids = self._get_existing_ids(all_ids)

            if existing_ids:
                docs_to_process = [
                    doc for doc in documents
                    if str(doc.doc_id) not in existing_ids
                ]
                result.skipped = len(existing_ids)
                logger.info(
                    f"Skipping {result.skipped} existing documents, "
                    f"processing {len(docs_to_process)} new"
                )

        if not docs_to_process:
            logger.info("Semua dokumen sudah ada di database, tidak ada yang diupload")
            result.duration_seconds = time.perf_counter() - start_time
            return result

        # ── Step 2: Encode semua teks sekaligus ───────────────────
        texts = [doc.full_text for doc in docs_to_process]
        try:
            logger.info(f"Encoding {len(texts)} texts...")
            all_embeddings = self.embedder.encode_batch(
                texts,
                progress_callback=lambda done, total: logger.debug(
                    f"Encoding progress: {done}/{total}"
                )
            )
        except Exception as e:
            logger.error(f"Batch encode failed: {e}")
            result.failed = len(docs_to_process)
            result.errors.append({
                "stage": "encoding",
                "error": str(e)
            })
            result.duration_seconds = time.perf_counter() - start_time
            return result

        # ── Step 3: Add ke ChromaDB per batch ─────────────────────
        for batch_start in range(0, len(docs_to_process), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(docs_to_process))
            batch_docs = docs_to_process[batch_start:batch_end]
            batch_embeddings = all_embeddings[batch_start:batch_end]

            ids = [str(doc.doc_id) for doc in batch_docs]
            embeddings = batch_embeddings
            documents_texts = [doc.full_text for doc in batch_docs]
            metadatas = [doc.to_metadata() for doc in batch_docs]

            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents_texts,
                    metadatas=metadatas
                )
                result.success += len(batch_docs)
                logger.info(
                    f"Batch uploaded: {batch_start + 1}-{batch_end}/"
                    f"{len(docs_to_process)} documents"
                )

                if progress_callback:
                    progress_callback(batch_end, len(docs_to_process))

            except Exception as e:
                failed_count = len(batch_docs)
                result.failed += failed_count
                error_info = {
                    "batch_range": f"{batch_start}-{batch_end}",
                    "doc_ids": ids,
                    "error": str(e)
                }
                result.errors.append(error_info)
                logger.error(
                    f"Batch ChromaDB add failed [{batch_start}:{batch_end}]: {e}",
                    exc_info=True
                )
                continue

        result.duration_seconds = time.perf_counter() - start_time

        # ── Step 4: Bersihkan cache pencarian ─────────────────────
        if result.success > 0:
            try:
                from src.cache_manager import get_cache_manager
                cache = get_cache_manager()
                cache.clear()
                logger.info("[+] Cache pencarian berhasil dibersihkan setelah batch upload.")
            except Exception as exc:
                logger.debug(f"Pembersihan cache dilewati / tidak aktif: {exc}")

        logger.info(
            f"Batch upload complete: {result.success} success, "
            f"{result.failed} failed, {result.skipped} skipped "
            f"in {result.duration_seconds:.4f}s"
        )

        return result

    def delete_batch(self, doc_ids: List[str]) -> BatchResult:
        """
        Hapus banyak dokumen dari ChromaDB sekaligus.
        """
        result = BatchResult(total=len(doc_ids))
        start_time = time.perf_counter()

        if not doc_ids:
            logger.warning("delete_batch dipanggil dengan list kosong")
            return result

        if self.collection is None:
            raise RuntimeError("Koleksi ChromaDB belum diinisialisasi.")

        logger.info(f"Starting batch delete: {len(doc_ids)} documents")

        existing_ids = self._get_existing_ids(doc_ids)
        ids_to_delete = [id_ for id_ in doc_ids if id_ in existing_ids]
        result.skipped = len(doc_ids) - len(ids_to_delete)

        if not ids_to_delete:
            logger.warning("Tidak ada dokumen yang ditemukan untuk dihapus")
            result.duration_seconds = time.perf_counter() - start_time
            return result

        try:
            self.collection.delete(ids=ids_to_delete)
            result.success = len(ids_to_delete)
            logger.info(f"Batch delete success: {result.success} documents deleted")

            # Bersihkan cache setelah penghapusan
            try:
                from src.cache_manager import get_cache_manager
                get_cache_manager().clear()
                logger.info("[+] Cache pencarian berhasil dibersihkan setelah batch delete.")
            except Exception as exc:
                logger.debug(f"Pembersihan cache dilewati: {exc}")

        except Exception as e:
            result.failed = len(ids_to_delete)
            result.errors.append({"error": str(e), "doc_ids": ids_to_delete})
            logger.error(f"Batch delete failed: {e}", exc_info=True)

        result.duration_seconds = time.perf_counter() - start_time
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Info konfigurasi dan jumlah dokumen pada koleksi."""
        doc_count = 0
        if self.collection is not None:
            try:
                doc_count = self.collection.count()
            except Exception as e:
                logger.debug(f"Could not get collection count: {e}")

        return {
            "batch_size": self.batch_size,
            "embedder_batch_size": self.embedder.batch_size,
            "collection_documents": doc_count,
        }


# ═══════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════

def create_batch_uploader(model, collection, batch_size: int = 64) -> BatchDocumentUploader:
    """
    Buat BatchDocumentUploader siap pakai.

    Args:
        model: SentenceTransformer instance (ml_models["minilm"])
        collection: ChromaDB collection instance
        batch_size: Ukuran batch (default 64, sama dengan sync_seranah_archives)

    Returns:
        BatchDocumentUploader siap pakai
    """
    return BatchDocumentUploader(
        model=model,
        collection=collection,
        batch_size=batch_size
    )
