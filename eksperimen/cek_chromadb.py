import chromadb
import os
import json

# Sesuaikan dengan path database Anda
DB_PATH = r"D:\3. ML\projectSkripsiSemantic\chroma_db_storage"


def list_all_collections_and_contents():
    if not os.path.exists(DB_PATH):
        print(f"[!] Database tidak ditemukan di: {DB_PATH}")
        return

    client = chromadb.PersistentClient(path=DB_PATH)
    collections = client.list_collections()

    if not collections:
        print("[*] Tidak ada koleksi yang ditemukan dalam database.")
    else:
        print(f"[*] Ditemukan {len(collections)} koleksi dalam database:\n")

        for col_info in collections:
            # Mengatasi perbedaan versi ChromaDB (terkadang mengembalikan string, terkadang objek)
            col_name = col_info.name if hasattr(col_info, 'name') else col_info

            collection = client.get_collection(col_name)
            count = collection.count()

            print(f"{'=' * 70}")
            print(f"📁 NAMA KOLEKSI : {col_name}")
            print(f"📊 JUMLAH DATA  : {count} dokumen")
            print(f"{'=' * 70}")

            if count > 0:
                print(f"[*] Mengambil 2 sampel data untuk pengecekan (Inspeksi Bug)...\n")
                # get() digunakan untuk melihat isi. Dibatasi 2 agar terminal tidak penuh (banjir teks).
                results = collection.get(limit=2)

                ids = results.get("ids", [])
                documents = results.get("documents", [])
                metadatas = results.get("metadatas", [])

                for i in range(len(ids)):
                    print(f"🔹 Sampel {i + 1} (ID: {ids[i]})")

                    # Menampilkan dokumen vektor (Dibatasi panjangnya agar enak dibaca)
                    doc_text = documents[i] if documents[i] else "-"
                    if len(doc_text) > 300:
                        doc_text = doc_text[:300] + " ... [DIPOTONG]"

                    print(f"   [Document / Text to Embed]:\n   {doc_text}\n")

                    # Menampilkan metadata dengan format JSON yang rapi (pretty print)
                    meta_text = metadatas[i] if metadatas[i] else {}
                    meta_formatted = json.dumps(meta_text, indent=4)

                    # Menambahkan spasi di setiap baris JSON agar indentasinya sejajar di terminal
                    meta_indented = "\n".join([f"   {line}" for line in meta_formatted.split("\n")])
                    print(f"   [Metadata]:\n{meta_indented}")
                    print(f"{'-' * 70}")
            else:
                print("   [!] Koleksi ini kosong. Tidak ada data untuk ditampilkan.\n")


if __name__ == "__main__":
    list_all_collections_and_contents()