import os
import chromadb

# Sesuaikan dengan path absolut database Anda
DB_PATH = r"D:\3. ML\projectSkripsiSemantic\chroma_db_storage"


def inspect_db_vector():
    if not os.path.exists(DB_PATH):
        print(f"[!] Folder database tidak ditemukan di: {DB_PATH}")
        return

    print(f"[*] Menghubungkan ke: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        # Mengambil koleksi
        collection = client.get_collection("arsip_kampus_v2")
        total_docs = collection.count()
        print(f"[*] Koleksi ditemukan. Total dokumen: {total_docs}")

        # Mengambil 3 sampel data untuk memastikan variasi data
        # 'include'=["metadatas", "documents"] memungkinkan kita melihat teks asli dan metadatanya
        samples = collection.get(limit=10, include=["metadatas", "documents"])

        if samples['metadatas']:
            print(f"\n--- MENAMPILKAN {len(samples['metadatas'])} SAMPEL DATA ---")

            for i in range(len(samples['metadatas'])):
                print(f"\n{'=' * 20} DOKUMEN {i + 1} {'=' * 20}")
                meta = samples['metadatas'][i]

                # Menampilkan Metadata
                print("[Metadata]")
                for key, value in meta.items():
                    # Menampilkan detail content_only jika ada
                    if key == "content_only":
                        print(f"  {key}: {str(value)[:150]}... (Panjang: {len(str(value))} karakter)")
                    else:
                        print(f"  {key}: {str(value)[:100]}")

                # Menampilkan Dokumen Asli (Teks untuk Embedding)
                doc_text = samples['documents'][i]
                print(f"\n[Dokumen Asli (Embed text)]:")
                print(f"  {str(doc_text)[:200]}...")
        else:
            print("[!] Tidak ada data dalam koleksi.")

    except Exception as e:
        print(f"[!] Terjadi kesalahan saat mengakses database: {e}")


if __name__ == "__main__":
    inspect_db_vector()