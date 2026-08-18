import os
import json
import chromadb
import re
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.getenv("HF_MODEL_NAME", "andrerean/minilm-arsip-kampus-v1")

TRAIN_FILE = os.path.join(ROOT, "data", "indodoc", "train.jsonl")
META_FILE = os.path.join(ROOT, "data", "indodoc", "metadata.jsonl")
COLLECTION_NAME = "arsip_kampus_v2"


def get_chroma_client():
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8001" if host == "localhost" else "8000"))

    print(f"[*] Menghubungkan ke ChromaDB Server API di {host}:{port} ...")
    return chromadb.HttpClient(host=host, port=port)


def extract_keywords(text):
    parts = re.split(r'kata\s+kunci\s*:?', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].strip()
    return "-"


def extract_keywords_vector(text):
    return extract_keywords(text)


def load_metadata_list():
    metadata_list = []
    with open(META_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            content = data.get("content", "")
            data["keywords"] = extract_keywords(content)
            metadata_list.append(data)
    return metadata_list


def load_train_data():
    """Fungsi 2: Load train.jsonl"""
    train_data = []
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    return train_data


def build_chroma_index():
    print(f"[*] Memuat model AI dari: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)

    client = get_chroma_client()

    metadata_list = load_metadata_list()
    train_dataset = load_train_data()

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:  # Diberi Exception eksplisit agar sesuai standar Python
        pass

    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids, documents, metadatas = [], [], []

    for idx, data in enumerate(train_dataset):
        title = data.get("title", "")
        content = data.get("content", "")
        kw_embedding = extract_keywords_vector(content)
        text_to_embed = f"{title}. {content}. kata kunci: {kw_embedding}"

        if idx < len(metadata_list):
            source = metadata_list[idx]
            meta = {
                "title": source.get("title", title),
                "content": source.get("content", content),
                "snippet": source.get("content", "")[:200],
                "content_only": content,
                "file_path": source.get("file_path", "-"),
                "file_name": source.get("file_name", "-"),
                "keywords": source.get("keywords", "-")
            }
        else:
            meta = {
                "title": title,
                "content": content,
                "snippet": content[:200],
                "content_only": content,
                "file_path": "-",
                "file_name": "-",
                "keywords": "-"
            }

        ids.append(f"doc_{idx}")
        documents.append(text_to_embed)
        metadatas.append(meta)

    print("[*] Memulai proses embedding dan pengiriman data ke server API ChromaDB...")
    collection.add(
        ids=ids,
        embeddings=model.encode(documents, show_progress_bar=True).tolist(),
        metadatas=metadatas,
        documents=documents
    )
    print(f"[DONE] Tersimpan {collection.count()} dokumen ke database API.")


if __name__ == "__main__":
    build_chroma_index()