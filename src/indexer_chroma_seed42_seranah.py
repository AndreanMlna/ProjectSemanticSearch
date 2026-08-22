import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- KONFIGURASI DARI .ENV & PATH ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model Embedding murni dikonfigurasi via file .env (HF_MODEL_NAME)
MODEL_PATH: str = os.getenv("HF_MODEL_NAME", "andrerean/minilm-arsip-kampus-seranah")

# Koleksi Target ChromaDB
COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION", "arsip_kampus_v2")

# File Dataset SERANAH
TRAIN_FILE = os.path.join(ROOT, "data", "indodoc", "train_seranah.jsonl")
ARCHIVES_FILE = os.path.join(ROOT, "data", "indodoc", "seranah_archives.jsonl")
if not os.path.exists(ARCHIVES_FILE):
    # Fallback ke folder data/ jika file diletakkan langsung di root data/
    fallback_archives = os.path.join(ROOT, "data", "seranah_archives.jsonl")
    if os.path.exists(fallback_archives):
        ARCHIVES_FILE = fallback_archives


def get_chroma_client() -> chromadb.HttpClient:
    """Menghubungkan ke server ChromaDB Docker."""
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8001" if host == "localhost" else "8000"))

    print(f"[*] Menghubungkan ke ChromaDB Server API di http://{host}:{port} ...")
    return chromadb.HttpClient(host=host, port=port)


def load_archives_metadata():
    """Memuat data metadata lengkap dari seranah_archives.jsonl."""
    archives_list = []
    if not os.path.exists(ARCHIVES_FILE):
        print(f"[!] Peringatan: File {ARCHIVES_FILE} tidak ditemukan. Metadata tambahan akan disesuaikan.")
        return archives_list

    with open(ARCHIVES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    archives_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return archives_list


def load_train_data():
    """Memuat data bersih untuk embedding dari train_seranah.jsonl."""
    train_data = []
    if not os.path.exists(TRAIN_FILE):
        print(f"[!] ERROR: File {TRAIN_FILE} tidak ditemukan. Jalankan prepare_dataset.py terlebih dahulu!")
        return train_data

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    train_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return train_data


def build_chroma_index():
    print("=" * 65)
    print("[*] MEMULAI PROSES INDEXING & EMBEDDING KE CHROMADB (SERANAH)")
    print(f"[*] Model AI Embedding : {MODEL_PATH}")
    print(f"[*] Dataset Training   : {TRAIN_FILE}")
    print(f"[*] Dataset Metadata   : {ARCHIVES_FILE}")
    print(f"[*] Target Collection  : {COLLECTION_NAME}")
    print("=" * 65)

    # 1. Inisialisasi Model Embedding dari .env
    print(f"\n[1] Memuat model embedding dari: {MODEL_PATH} ...")
    model = SentenceTransformer(MODEL_PATH)

    # 2. Inisialisasi Koneksi ChromaDB
    print(f"[2] Menyiapkan koneksi ChromaDB...")
    client = get_chroma_client()

    # 3. Load Data
    print(f"[3] Membaca data dataset...")
    train_dataset = load_train_data()
    archives_metadata = load_archives_metadata()

    if not train_dataset:
        print("[!] Pembatalan: Dataset training kosong.")
        return

    print(f"[+] Total data training : {len(train_dataset)} dokumen")
    print(f"[+] Total data metadata : {len(archives_metadata)} dokumen")

    # 4. Reset & Buat Collection Baru (Cosine Similarity)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"[*] Collection lama '{COLLECTION_NAME}' berhasil di-reset.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 5. Susun Dokumen Teks & Metadata
    ids, documents, metadatas = [], [], []

    for idx, data in enumerate(train_dataset):
        clean_title = data.get("title", "")
        clean_content = data.get("content", "")
        clean_keywords = data.get("keywords", "-")

        # Teks terpadu yang di-embed ke vektor (Title + Content + Keywords)
        if clean_keywords and clean_keywords != "-":
            text_to_embed = f"{clean_title}. {clean_content}. kata kunci: {clean_keywords}"
        else:
            text_to_embed = f"{clean_title}. {clean_content}"

        # Ambil info asli dari seranah_archives.jsonl jika tersedia
        if idx < len(archives_metadata):
            source = archives_metadata[idx]
            doc_uuid = source.get("uuid", f"doc_{idx}")
            original_title = source.get("title", clean_title)
            category_raw = source.get("category", {})
            category_name = category_raw.get("categoryName", "-") if isinstance(category_raw, dict) else str(category_raw)

            meta = {
                "uuid": doc_uuid,
                "title": original_title,
                "content": clean_content,
                "snippet": clean_content[:200],
                "content_only": clean_content,
                "document_number": str(source.get("documentNumber", "-")),
                "year": str(source.get("year", "-")),
                "file_name": str(source.get("fileName", "-")),
                "file_path": str(source.get("fileName", "-")),
                "mime_type": str(source.get("mimeType", "application/pdf")),
                "access_level": str(source.get("accessLevel", "PUBLIC")),
                "uploader": str(source.get("uploader", "-")),
                "unit_kerja": str(source.get("unitKerja", "-")),
                "category": category_name,
                "keywords": str(source.get("keywords", clean_keywords)),
            }
        else:
            doc_uuid = f"doc_{idx}"
            meta = {
                "uuid": doc_uuid,
                "title": clean_title,
                "content": clean_content,
                "snippet": clean_content[:200],
                "content_only": clean_content,
                "document_number": "-",
                "year": "-",
                "file_name": "-",
                "file_path": "-",
                "mime_type": "application/pdf",
                "access_level": "PUBLIC",
                "uploader": "-",
                "unit_kerja": "-",
                "category": "-",
                "keywords": clean_keywords,
            }

        ids.append(doc_uuid)
        documents.append(text_to_embed)
        metadatas.append(meta)

    # 6. Hitung Vektor Embedding
    print(f"\n[4] Memproses inferensi embedding {len(documents)} dokumen...")
    embeddings = model.encode(documents, batch_size=32, show_progress_bar=True).tolist()

    # 7. Kirim Batch ke ChromaDB
    print(f"\n[5] Mengirimkan data vektor ke server ChromaDB...")
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end_idx = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=metadatas[i:end_idx],
            documents=documents[i:end_idx],
        )

    print("\n" + "=" * 65)
    print(f"[✅] SUKSES! {collection.count()} dokumen berhasil di-vektorisasi dan disimpan ke ChromaDB.")
    print(f"[📊] Koleksi : {COLLECTION_NAME}")
    print("=" * 65)


if __name__ == "__main__":
    build_chroma_index()