"""
Batch Processor
Memproses banyak dokumen arsip sekaligus secara efisien

Dua kegunaan utama sesuai arsitektur project ini:
1. BatchEmbedder     — Generate embedding banyak teks sekaligus
                       (lebih cepat dari encode satu-per-satu)
2. BatchDocumentUploader — Upload/index banyak dokumen ke ChromaDB sekaligus
                           (berguna untuk bulk import arsip kampus)

Hubungan dengan file lain:
- Menggunakan SentenceTransformer yang sama dengan main_api.py (ml_models["minilm"])
- Menggunakan ChromaDB collection yang sama (arsip_kampus)
- Logging via logging_utils.py
- Error handling via error_handler.py
- Statistik via metrics_collector.py
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

    Struktur ini sama dengan metadata yang disimpan di ChromaDB
    pada endpoint /upload di main_api.py.
    """
    doc_id: str          # ID unik dokumen, e.g. "upload_1234567890"
    title: str           # Judul dokumen arsip
    content: str         # Deskripsi / isi singkat
    file_name: str       # Nama file fisik (sudah di-replace spasi → _)
    file_path: str       # Path lengkap file di server
    full_text: str       # Gabungan title + content + extracted_text untuk embedding
    snippet: str = ""    # Cuplikan untuk ditampilkan di hasil search


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
    SentenceTransformer yang sama dengan main_api.py.

    Keuntungan vs encode satu-per-satu:
    - SentenceTransformer.encode() sudah dioptimasi untuk batch
    - GPU (jika ada) digunakan lebih efisien
    - Overhead panggilan fungsi berkurang

    Contoh pemakaian:
        model = ml_models["minilm"]  # dari main_api.py
        embedder = BatchEmbedder(model=model, batch_size=32)

        texts = ["surat keputusan rektor", "SK wisuda 2024", ...]
        embeddings = embedder.encode_batch(texts)
        # embeddings: List[List[float]], satu vektor per teks
    """

    def __init__(self, model, batch_size: int = 32):
        """
        Args:
            model: Instance SentenceTransformer dari ml_models["minilm"]
            batch_size: Jumlah teks per batch. Default 32 aman untuk RAM biasa.
                        Turunkan ke 16 jika memory error.
        """
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

        Args:
            texts: List teks yang akan di-encode
            progress_callback: Optional callback dipanggil setiap batch selesai.
                               Signature: callback(processed_count, total_count)

        Returns:
            List of embedding vectors (List[List[float]])
            Urutan output = urutan input

        Raises:
            ValueError: Jika texts kosong
            RuntimeError: Jika model gagal encode
        """
        if not texts:
            raise ValueError("texts tidak boleh kosong")

        if self.model is None:
            raise RuntimeError("Model belum di-load. Pastikan server sudah startup.")

        total = len(texts)
        all_embeddings = []
        start_time = time.perf_counter()

        logger.info(f"Starting batch encode: {total} texts, batch_size={self.batch_size}")

        # Proses per chunk sesuai batch_size
        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch_texts = texts[batch_start:batch_end]

            try:
                # SentenceTransformer.encode() menerima list langsung
                batch_vectors = self.model.encode(
                    batch_texts,
                    show_progress_bar=False,  # kita handle progress sendiri
                    convert_to_numpy=True
                )
                # Convert numpy array → list of list (untuk ChromaDB)
                all_embeddings.extend(batch_vectors.tolist())

                if progress_callback:
                    progress_callback(batch_end, total)

                logger.debug(
                    f"Batch encoded: {batch_start+1}-{batch_end}/{total}"
                )

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
        """
        Encode satu teks — wrapper convenience untuk konsistensi.
        Sama dengan yang dipakai di endpoint /search main_api.py.

        Args:
            text: Teks yang akan di-encode

        Returns:
            Embedding vector sebagai List[float]
        """
        return self.model.encode(text).tolist()


# ═══════════════════════════════════════════════════════════════════
# BATCH DOCUMENT UPLOADER
# ═══════════════════════════════════════════════════════════════════

class BatchDocumentUploader:
    """
    Upload dan index banyak dokumen arsip ke ChromaDB sekaligus.

    Didesain untuk:
    - Bulk import awal dokumen arsip kampus (ratusan/ribuan dokumen)
    - Re-indexing setelah model update
    - Migrasi data dari sistem lama

    Menggunakan ChromaDB collection.add() dengan batch agar lebih
    efisien daripada add satu-per-satu seperti di endpoint /upload.

    Contoh pemakaian di script terpisah:
        from src.batch_processor import BatchDocumentUploader, DocumentItem
        from src.main_api import ml_models, collection  # import globals

        uploader = BatchDocumentUploader(
            model=ml_models["minilm"],
            collection=collection,
            batch_size=50
        )

        documents = [
            DocumentItem(
                doc_id="doc_001",
                title="SK Rektor No. 123/2024",
                content="Surat Keputusan tentang wisuda",
                file_name="SK_Rektor_123_2024.pdf",
                file_path="/uploads/SK_Rektor_123_2024.pdf",
                full_text="SK Rektor No. 123/2024. Surat Keputusan tentang wisuda ...",
                snippet="Surat Keputusan tentang wisuda mahasiswa semester ganjil"
            ),
            # ... dokumen lainnya
        ]

        result = uploader.upload_batch(documents)
        print(result.to_dict())
    """

    def __init__(self, model, collection, batch_size: int = 50):
        """
        Args:
            model: Instance SentenceTransformer dari ml_models["minilm"]
            collection: ChromaDB collection dari main_api.py (global collection)
            batch_size: Jumlah dokumen per batch ChromaDB add. Default 50.
        """
        self.embedder = BatchEmbedder(model=model, batch_size=batch_size)
        self.collection = collection
        self.batch_size = batch_size
        logger.info(
            f"BatchDocumentUploader initialized: batch_size={batch_size}"
        )

    def _get_existing_ids(self, doc_ids: List[str]) -> set:
        """
        Cek ID mana saja yang sudah ada di ChromaDB.

        Dipakai untuk skip duplicate saat upload batch.

        Args:
            doc_ids: List ID yang akan dicek

        Returns:
            Set ID yang sudah ada di collection
        """
        try:
            existing = self.collection.get(ids=doc_ids)
            return set(existing["ids"])
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

        Proses:
        1. Filter duplicate (jika skip_existing=True)
        2. Kumpulkan semua full_text → encode sekaligus (efisien)
        3. Add ke ChromaDB per batch

        Args:
            documents: List DocumentItem yang akan diupload
            skip_existing: Jika True, dokumen dengan ID yang sudah ada di-skip
            progress_callback: Optional callback(processed, total) per batch

        Returns:
            BatchResult dengan statistik sukses/gagal/skip
        """
        result = BatchResult(total=len(documents))
        start_time = time.perf_counter()

        if not documents:
            logger.warning("upload_batch dipanggil dengan list kosong")
            result.duration_seconds = 0.0
            return result

        logger.info(f"Starting batch upload: {len(documents)} documents")

        # ── Step 1: Filter duplicate ──────────────────────────────
        docs_to_process = documents
        if skip_existing:
            all_ids = [doc.doc_id for doc in documents]
            existing_ids = self._get_existing_ids(all_ids)

            if existing_ids:
                docs_to_process = [
                    doc for doc in documents
                    if doc.doc_id not in existing_ids
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

        # ── Step 2: Encode semua teks sekaligus ──────────────────
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

        # ── Step 3: Add ke ChromaDB per batch ────────────────────
        for batch_start in range(0, len(docs_to_process), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(docs_to_process))
            batch_docs = docs_to_process[batch_start:batch_end]
            batch_embeddings = all_embeddings[batch_start:batch_end]

            # Siapkan data untuk ChromaDB.add()
            # Format sama persis dengan endpoint /upload di main_api.py
            ids = [doc.doc_id for doc in batch_docs]
            embeddings = batch_embeddings
            documents_texts = [doc.full_text for doc in batch_docs]
            metadatas = [
                {
                    "title": doc.title,
                    "file_name": doc.file_name,
                    "file_path": doc.file_path,
                    "snippet": doc.snippet or doc.content[:300]
                }
                for doc in batch_docs
            ]

            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents_texts,
                    metadatas=metadatas
                )
                result.success += len(batch_docs)
                logger.info(
                    f"Batch uploaded: {batch_start+1}-{batch_end}/"
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
                # Lanjut batch berikutnya meski ada error
                continue

        result.duration_seconds = time.perf_counter() - start_time

        logger.info(
            f"Batch upload complete: {result.success} success, "
            f"{result.failed} failed, {result.skipped} skipped "
            f"in {result.duration_seconds:.4f}s"
        )

        return result

    def delete_batch(self, doc_ids: List[str]) -> BatchResult:
        """
        Hapus banyak dokumen dari ChromaDB sekaligus.

        Konsisten dengan endpoint DELETE /documents/{doc_id} di main_api.py
        tapi untuk operasi bulk.

        Args:
            doc_ids: List ID dokumen yang akan dihapus

        Returns:
            BatchResult dengan statistik
        """
        result = BatchResult(total=len(doc_ids))
        start_time = time.perf_counter()

        if not doc_ids:
            logger.warning("delete_batch dipanggil dengan list kosong")
            return result

        logger.info(f"Starting batch delete: {len(doc_ids)} documents")

        # Cek mana yang benar-benar ada
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
        except Exception as e:
            result.failed = len(ids_to_delete)
            result.errors.append({"error": str(e), "doc_ids": ids_to_delete})
            logger.error(f"Batch delete failed: {e}", exc_info=True)

        result.duration_seconds = time.perf_counter() - start_time
        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Info tentang batch processor ini.

        Returns:
            Dictionary berisi konfigurasi batch processor
        """
        doc_count = 0
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
# FACTORY FUNCTION — dipakai di main_api.py atau script terpisah
# ═══════════════════════════════════════════════════════════════════

def create_batch_uploader(model, collection) -> BatchDocumentUploader:
    """
    Buat BatchDocumentUploader dengan konfigurasi dari config.yaml.

    Membaca batch_size dari config jika tersedia, fallback ke 50.

    Args:
        model: SentenceTransformer instance (ml_models["minilm"])
        collection: ChromaDB collection instance

    Returns:
        BatchDocumentUploader siap pakai

    Contoh di main_api.py (endpoint bulk upload opsional):
        from src.batch_processor import create_batch_uploader
        uploader = create_batch_uploader(ml_models["minilm"], collection)
    """
    batch_size = 50

    try:
        from src.config_manager import get_config
        cfg = get_config()
        # Ambil dari cache.max_size sebagai proxy ukuran batch yang wajar
        # atau bisa tambah key batch.size di config.yaml nanti
        batch_size = cfg.get("batch.size", 50)
    except Exception as e:
        logger.debug(f"Using default batch_size=50: {e}")

    return BatchDocumentUploader(
        model=model,
        collection=collection,
        batch_size=batch_size
    )  