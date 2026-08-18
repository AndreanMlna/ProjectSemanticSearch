import os
import time
import chromadb
from sentence_transformers import SentenceTransformer


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


MODEL_PATH = os.path.join(ROOT, "output", "minilm-dokumen-arsip-boosted-new-seed-42")


COLLECTION_NAME = "arsip_kampus_v2"
LOCAL_DB_PATH = os.path.join(ROOT, "chroma_db_storage")


def semantic_search():

    print(f"[*] Loading Model AI dari: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("[!] Model tidak ditemukan. Harap jalankan training dulu!")
        return

    model = SentenceTransformer(MODEL_PATH)


    print(f"[*] Menghubungkan ke Database Lokal: {LOCAL_DB_PATH}")
    if not os.path.exists(LOCAL_DB_PATH):
        print("[!] Database belum ditemukan. Jalankan 'src/indexer_chroma.py' dulu!")
        return

    client = chromadb.PersistentClient(path=LOCAL_DB_PATH)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"[*] Berhasil terhubung! Total Dokumen: {count}")
    except ValueError:
        print(f"[!] Koleksi '{COLLECTION_NAME}' tidak ditemukan di database.")
        return

    print("\n" + "=" * 50)
    print("   SISTEM PENCARIAN ARSIP CERDAS (CHROMA DB)")
    print("=" * 50)
    print("[*] Ketik 'exit' untuk keluar.")

    while True:
        query_text = input("\nMasukkan kata kunci pencarian: ")
        if query_text.lower() in ['exit', 'keluar']:
            break

        start_time = time.perf_counter()

        # Encode Query (Teks User -> Angka Vektor)
        query_vector = model.encode(query_text).tolist()

        # Query ke Database (Minta Chroma Carikan yang Mirip)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=10,
            include=["metadatas", "distances"]
        )

        duration = time.perf_counter() - start_time
        print(f"\n[+] Ditemukan dalam {duration:.4f} detik:\n")

        # Cek apakah ada hasil
        if not results['metadatas'] or not results['metadatas'][0]:
            print("   Tidak ada hasil yang cocok.")
            continue

        # Tampilkan Hasil
        for i, meta in enumerate(results['metadatas'][0]):

            dist = results['distances'][0][i]
            score = 1 - dist

            title = meta.get('title', 'Tanpa Judul')
            file_name = meta.get('file_name', '-')

            snippet = meta.get('snippet', '')

            print(f"   [Relevansi: {score:.4f}]")
            print(f"   Judul : {title}")
            print(f"   Info  : {snippet[:150]}...")
            print(f"   File  : {file_name}")
            print("   " + "-" * 40)


if __name__ == "__main__":
    semantic_search()