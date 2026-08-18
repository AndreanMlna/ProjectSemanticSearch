import os
import time
import chromadb
from sentence_transformers import SentenceTransformer
from src.text_extractor import extract_text_from_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "output", "minilm-dokumen-arsip-boosted")
LOCAL_DB_PATH = os.path.join(ROOT, "chroma_db_storage")
COLLECTION_NAME = "arsip_kampus_v2"

print("[*] Pre-loading AI Model...")
if os.path.exists(MODEL_PATH):
    model = SentenceTransformer(MODEL_PATH)
else:
    model = None
    print("[!] Model tidak ditemukan!")


def get_db_collection():
    client = chromadb.PersistentClient(path=LOCAL_DB_PATH)
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except ValueError:
        return None


def process_single_document(title, content, file_name, file_path):
    if model is None: return False, "Model Error"

    collection = get_db_collection()
    if collection is None: return False, "Database belum di-index awal"

    # --- LANGKAH BARU: EKSTRAKSI ISI FILE ---
    print(f"[*] Mengekstrak teks isi file: {file_name}...")
    file_content_text = extract_text_from_file(file_path)

    if not file_content_text:
        print(f"[!] Warning: Tidak ada teks terbaca dari {file_name} (Mungkin gambar scan?)")

    full_text_to_embed = f"{title}. {content}. {file_content_text}"

    vector = model.encode(full_text_to_embed).tolist()

    doc_id = f"doc_auto_{int(time.time())}"

    try:
        collection.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[full_text_to_embed],
            metadatas=[{
                "title": title,
                "file_name": file_name,
                "file_path": file_path,
                "snippet": content if content else file_content_text[:300]
            }]
        )
        print(f"[+] Berhasil menambahkan & index isi file: {title}")
        return True, "Sukses"
    except Exception as e:
        print(f"[-] Gagal menambah data: {e}")
        return False, str(e)


# --- [DITAMBAHKAN] FUNGSI HAPUS DOKUMEN ---
def delete_document_by_id(doc_id):
    """
    Menghapus dokumen dari ChromaDB berdasarkan ID-nya.
    """
    collection = get_db_collection()
    if collection is None:
        return False, "Database belum terhubung/tidak ditemukan"

    try:
        # Cek apakah dokumen ada (opsional, untuk memastikan)
        existing_doc = collection.get(ids=[doc_id])
        if not existing_doc['ids']:
            return False, "ID Dokumen tidak ditemukan dalam database."

        # Lakukan penghapusan
        collection.delete(ids=[doc_id])
        print(f"[-] Berhasil menghapus dokumen ID: {doc_id}")
        return True, "Dokumen berhasil dihapus."

    except Exception as e:
        print(f"[!] Error saat menghapus: {e}")
        return False, str(e)