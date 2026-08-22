import os
import time
import chromadb
from typing import Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from src.text_extractor import extract_text_from_file

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH: str = os.getenv("HF_MODEL_NAME", "andrerean/minilm-arsip-kampus-seranah")
LOCAL_DB_PATH: str = os.path.join(ROOT, "chroma_db_storage")
COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION", "arsip_kampus_v2")

CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
DEFAULT_PORT: str = "8001" if CHROMA_HOST == "localhost" else "8000"
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", DEFAULT_PORT))

# Lazy model loader
_model_instance: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        print(f"[*] Inisialisasi Model Embedding: {MODEL_PATH} ...")
        _model_instance = SentenceTransformer(MODEL_PATH)
    return _model_instance


def get_db_collection() -> Optional[Any]:
    """Menghubungkan ke ChromaDB melalui HttpClient (Docker) atau PersistentClient."""
    # 1. Coba koneksi Client-Server API
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    # 2. Fallback ke PersistentClient lokal
    try:
        client = chromadb.PersistentClient(path=LOCAL_DB_PATH)
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return None


def process_single_document(title: str, content: str, file_name: str, file_path: str) -> Tuple[bool, str]:
    """Mengekstraksi teks isi file, membuat vektor embedding, dan menyimpan dokumen ke ChromaDB."""
    try:
        model = get_embedding_model()
    except Exception as e:
        return False, f"Model Error: {str(e)}"

    collection = get_db_collection()
    if collection is None:
        return False, "Database ChromaDB belum aktif atau belum di-index awal."

    # Ekstraksi isi file (PDF, Docx, TXT)
    print(f"[*] Mengekstrak teks isi file: {file_name}...")
    file_content_text = extract_text_from_file(file_path)

    if not file_content_text:
        print(f"[!] Peringatan: Tidak ada teks terbaca dari {file_name} (Mungkin hasil scan?)")

    full_text_to_embed = f"{title}. {content}. {file_content_text}".strip()
    vector = model.encode(full_text_to_embed).tolist()
    doc_id = f"doc_manual_{int(time.time())}"

    try:
        collection.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[full_text_to_embed],
            metadatas=[{
                "uuid": doc_id,
                "title": title,
                "file_name": file_name,
                "file_path": file_path,
                "content": content,
                "snippet": content[:200] if content else file_content_text[:200],
                "access_level": "PUBLIC",
                "uploader": "Admin (Manual Upload)",
            }]
        )
        print(f"[+] Berhasil menambahkan & mengindeks dokumen: {title}")
        return True, "Dokumen berhasil diindeks ke database vektor."
    except Exception as e:
        print(f"[-] Gagal menambah data: {e}")
        return False, str(e)


def delete_document_by_id(doc_id: str) -> Tuple[bool, str]:
    """Menghapus dokumen dari ChromaDB berdasarkan ID-nya."""
    collection = get_db_collection()
    if collection is None:
        return False, "Database belum terhubung atau tidak ditemukan."

    try:
        existing_doc = collection.get(ids=[doc_id])
        if not existing_doc or not existing_doc.get("ids"):
            return False, f"ID Dokumen '{doc_id}' tidak ditemukan di database."

        collection.delete(ids=[doc_id])
        print(f"[-] Berhasil menghapus dokumen ID: {doc_id}")
        return True, f"Dokumen '{doc_id}' berhasil dihapus."
    except Exception as e:
        print(f"[!] Error saat menghapus dokumen: {e}")
        return False, str(e)