"""
src/sync_seranah_archives.py
============================
Modul sinkronisasi dataset live API SERANAH langsung ke ChromaDB (Stateless / In-Memory).

Prinsip:
1. Mengambil data arsip dari live API SERANAH UNIDA Gontor (.env: METADATA_SERANAH).
2. Memeriksa perbedaan data dengan ChromaDB menggunakan validasi sampel dokumen terbaru.
3. Melakukan pembersihan teks in-memory dan indexing vektor langsung ke ChromaDB tanpa perantara file fisik (100% Stateless).
4. Membersihkan cache pencarian setelah sinkronisasi selesai agar kueri pengguna selalu mutakhir.
"""

import os
import re
import time
import logging
import urllib3
import requests
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Muat variabel environment dari .env
load_dotenv()

# Nonaktifkan warning SSL jika sertifikat internal kampus self-signed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SyncSeranah")

SERANAH_API_URL = os.getenv("METADATA_SERANAH", "https://seranah-be.unida.gontor.ac.id/api/v1/archives")


def clean_text(text: str) -> str:
    """Membersihkan teks dari tag HTML, URL, email, simbol aneh, dan escape characters."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s.,?!-]", " ", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def get_chroma_archives_info() -> Tuple[int, set]:
    """
    Membaca jumlah dokumen dan himpunan UUID langsung dari database ChromaDB (bukan dari file).
    """
    from src.chroma_client import get_collection
    collection = get_collection()
    if collection is None:
        return 0, set()

    try:
        count = collection.count()
        data = collection.get()
        uuids = set(data.get("ids", []))
        return count, uuids
    except Exception as e:
        logger.warning(f"Gagal mengambil informasi koleksi ChromaDB: {e}")
        return 0, set()


def fetch_seranah_archives_from_api(
    url: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """
    Mengunduh seluruh metadata arsip dari endpoint live API SERANAH UNIDA Gontor.
    """
    api_url = url or SERANAH_API_URL
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
    }

    logger.info(f"[*] Menghubungi Live API SERANAH: {api_url}")

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.perf_counter()
            response = requests.get(api_url, headers=headers, timeout=timeout, verify=False)
            elapsed = time.perf_counter() - t0

            if response.status_code == 200:
                payload = response.json()
                data = payload.get("data", [])
                logger.info(f"[+] Berhasil mengunduh data dalam {elapsed:.2f} detik (HTTP 200).")
                logger.info(f"[+] Total arsip diterima dari API: {len(data)} dokumen.")
                return data
            else:
                logger.warning(f"[!] Percobaan {attempt}/{max_retries}: Server merespons HTTP {response.status_code}. Menunggu coba ulang...")
                time.sleep(3)
        except requests.exceptions.RequestException as req_err:
            logger.warning(f"[!] Percobaan {attempt}/{max_retries}: Terjadi kesalahan jaringan ({req_err}).")
            time.sleep(3)

    raise ConnectionError(f"Gagal mengambil data dari {api_url} setelah {max_retries} kali percobaan.")


def check_chroma_synced_with_sample(
    remote_data: List[Dict[str, Any]],
    sample_size: int = 10
) -> Tuple[bool, str]:
    """
    Memeriksa kesesuaian data live API dengan database ChromaDB:
    1. Mengecek jumlah total dokumen di ChromaDB vs API.
    2. Mengambil sampel dokumen terbaru dari API dan memverifikasi keberadaan serta kontennya di ChromaDB.
    """
    from src.chroma_client import get_collection

    collection = get_collection()
    if collection is None:
        return False, "ChromaDB belum terhubung atau koleksi belum diinisialisasi."

    chroma_count = collection.count()
    remote_count = len(remote_data)

    if chroma_count == 0 and remote_count > 0:
        return False, f"ChromaDB masih kosong (0 dokumen), API memiliki {remote_count} dokumen."

    if chroma_count != remote_count:
        return False, f"Jumlah dokumen berbeda (ChromaDB: {chroma_count}, Live API: {remote_count})."

    # Ambil sample dokumen teratas (dokumen paling baru)
    sample_docs = [d for d in remote_data[:sample_size] if isinstance(d, dict) and d.get("uuid")]
    if not sample_docs:
        return False, "Sample dokumen dari API tidak memiliki UUID valid."

    sample_uuids = [str(doc["uuid"]) for doc in sample_docs]

    try:
        existing_in_chroma = collection.get(ids=sample_uuids)
        existing_ids = set(existing_in_chroma.get("ids", []))

        # 1. Pastikan semua UUID sampel ada di ChromaDB
        for sample_doc in sample_docs:
            uid = str(sample_doc["uuid"])
            if uid not in existing_ids:
                return False, f"Dokumen sampel '{uid}' belum ada di ChromaDB."

        # 2. Periksa kesesuaian konten sampel (misal: title)
        metas = existing_in_chroma.get("metadatas", []) or []
        existing_metas: Dict[str, Dict[str, Any]] = {
            str(m["uuid"]): dict(m)
            for m in metas
            if isinstance(m, dict) and m.get("uuid") is not None
        }

        for sample_doc in sample_docs:
            uid = str(sample_doc["uuid"])
            chroma_meta = existing_metas.get(uid)
            if not chroma_meta:
                return False, f"Metadata untuk sampel '{uid}' tidak ditemukan di ChromaDB."

            remote_title = str(sample_doc.get("title") or "").strip()
            chroma_title = str(chroma_meta.get("title") or "").strip()

            if remote_title != chroma_title:
                return False, f"Dokumen sampel '{uid}' mengalami perubahan judul (Chroma: '{chroma_title}', Live API: '{remote_title}')."

        return True, f"Dataset lokal ChromaDB sudah sinkron ({chroma_count} dokumen, {len(sample_docs)} sampel terverifikasi)."

    except Exception as e:
        logger.warning(f"Gagal memvalidasi sampel dokumen dengan ChromaDB: {e}")
        return False, f"Pemeriksaan sampel dokumen gagal: {e}"


def index_archives_direct_to_chromadb(
    archives: List[Dict[str, Any]],
    batch_size: int = 64
) -> int:
    """
    Memproses dan meng-index dokumen langsung dari memori (RAM) ke ChromaDB
    tanpa membuat atau menyimpan file fisik ke harddisk (Stateless).
    """
    from sentence_transformers import SentenceTransformer
    from src.config import MODEL_PATH
    from src.chroma_client import reset_collection
    from src.cache_manager import get_cache_manager

    logger.info("=" * 65)
    logger.info("[*] MEMULAI DIRECT IN-MEMORY INDEXING KE CHROMADB (STATELESS)")
    logger.info(f"[*] Model Embedding : {MODEL_PATH}")
    logger.info(f"[*] Total Arsip     : {len(archives)} dokumen")
    logger.info("=" * 65)

    # 1. Inisialisasi Model Embedding
    logger.info(f"[1] Memuat model embedding: {MODEL_PATH} ...")
    model = SentenceTransformer(MODEL_PATH)

    # 2. Reset Koleksi ChromaDB untuk pembaruan data bersih
    logger.info("[2] Menyiapkan koleksi baru ChromaDB...")
    collection = reset_collection()
    if collection is None:
        raise RuntimeError("Gagal menginisialisasi atau mereset koleksi ChromaDB.")

    # 3. Proses ekstraksi teks & penyusunan metadata in-memory
    ids, documents, metadatas = [], [], []
    seen_uuids = set()

    for idx, source in enumerate(archives, start=1):
        if not isinstance(source, dict):
            continue

        doc_uuid = source.get("uuid")
        raw_title = source.get("title") or ""
        raw_description = source.get("description") or ""
        raw_keywords = source.get("keywords") or "-"

        if not doc_uuid or (not raw_title and not raw_description):
            continue

        str_uuid = str(doc_uuid)
        if str_uuid in seen_uuids:
            continue
        seen_uuids.add(str_uuid)

        clean_title = clean_text(str(raw_title))
        clean_content = clean_text(str(raw_description))
        clean_keywords = clean_text(str(raw_keywords))

        if clean_keywords and clean_keywords != "-":
            text_to_embed = f"{clean_title}. {clean_content}. kata kunci: {clean_keywords}"
        else:
            text_to_embed = f"{clean_title}. {clean_content}"

        category_raw = source.get("category") or {}
        category_name = (
            category_raw.get("categoryName", "-")
            if isinstance(category_raw, dict)
            else (category_raw or "-")
        )

        title_val = str(source.get("title") or clean_title).strip()
        keywords_val = str(source.get("keywords") or (clean_keywords if clean_keywords else "-"))

        meta = {
            "uuid": str_uuid,
            "title": title_val,
            "content": clean_content,
            "snippet": clean_content[:200],
            "content_only": clean_content,
            "document_number": str(source.get("documentNumber") or "-"),
            "year": str(source.get("year") or "-"),
            "file_name": str(source.get("fileName") or "-"),
            "file_path": str(source.get("fileName") or "-"),
            "mime_type": str(source.get("mimeType") or "application/pdf"),
            "access_level": str(source.get("accessLevel") or "PUBLIC"),
            "uploader": str(source.get("uploader") or "-"),
            "unit_kerja": str(source.get("unitKerja") or "-"),
            "category": str(category_name),
            "keywords": keywords_val,
        }

        ids.append(str_uuid)
        documents.append(text_to_embed)
        metadatas.append(meta)

    total_valid = len(documents)
    logger.info(f"[3] Menghitung embedding dan menyimpan {total_valid} dokumen ke ChromaDB...")

    # 4. Batch encode & insert ke ChromaDB
    for i in range(0, total_valid, batch_size):
        b_ids = ids[i:i + batch_size]
        b_docs = documents[i:i + batch_size]
        b_metas = metadatas[i:i + batch_size]

        b_embeddings = model.encode(b_docs, batch_size=batch_size, show_progress_bar=False).tolist()
        collection.add(
            ids=b_ids,
            documents=b_docs,
            embeddings=b_embeddings,
            metadatas=b_metas
        )

    # 5. Bersihkan cache pencarian agar hasil kueri terbaru langsung aktif
    try:
        cache = get_cache_manager()
        cache.clear()
        logger.info("[+] Cache pencarian berhasil dibersihkan.")
    except Exception as exc:
        logger.debug(f"Pembersihan cache dilewati / tidak aktif: {exc}")

    logger.info(f"[✅] Sinkronisasi selesai: {total_valid} dokumen tersimpan langsung di ChromaDB.")
    return total_valid


def check_and_sync(auto_reindex: bool = True, **_kwargs) -> Dict[str, Any]:
    """
    Fungsi utama sinkronisasi (Stateless):
    1. Mengunduh data dari live API SERANAH.
    2. Memeriksa perbedaan dengan membandingkan sampel dokumen terbaru ke ChromaDB.
    3. Jika terdeteksi perubahan: langsung meng-index in-memory ke ChromaDB (tanpa file).
    """
    logger.info("=" * 65)
    logger.info("   MEMERIKSA KESESUAIAN DATASET DENGAN LIVE API SERANAH UNIDA   ")
    logger.info("=" * 65)

    # 1. Unduh data dari Live API
    try:
        remote_data = fetch_seranah_archives_from_api()
    except Exception as e:
        logger.error(f"[!] Gagal menghubungi live API SERANAH: {e}")
        return {
            "status": "error",
            "updated": False,
            "message": str(e),
            "remote_count": 0
        }

    remote_count = len(remote_data)

    # 2. Cek apakah ChromaDB sudah sinkron via validasi sampel
    is_synced, reason = check_chroma_synced_with_sample(remote_data, sample_size=10)

    if is_synced:
        logger.info(f"[✅] STATUS: {reason}")
        return {
            "status": "success",
            "updated": False,
            "message": reason,
            "local_count": remote_count,
            "remote_count": remote_count
        }

    logger.info(f"[⚡] PERUBAHAN TERDETEKSI: {reason}")
    if not auto_reindex:
        return {
            "status": "needs_update",
            "updated": False,
            "message": reason,
            "remote_count": remote_count
        }

    # 3. Direct in-memory indexing ke ChromaDB (Stateless / Tanpa file)
    synced_count = index_archives_direct_to_chromadb(remote_data)

    return {
        "status": "success",
        "updated": True,
        "message": f"Data live berhasil disinkronkan langsung ke ChromaDB ({synced_count} dokumen).",
        "local_count": synced_count,
        "remote_count": remote_count
    }


def sync_seranah_to_chromadb(model=None, collection=None) -> Tuple[bool, str, int]:
    """
    Helper kompatibel untuk mengunduh dan menyinkronkan data live API SERANAH ke ChromaDB.
    Mengembalikan (success: bool, message: str, doc_count: int).
    """
    res = check_and_sync(auto_reindex=True)
    ok = res.get("status") == "success"
    msg = res.get("message", "")
    cnt = res.get("local_count") or res.get("remote_count", 0)
    return ok, msg, cnt


if __name__ == "__main__":
    check_and_sync(auto_reindex=True)

