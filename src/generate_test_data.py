import os
import json
import random

# Konfigurasi Seed agar pengambilan data konsisten
random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(ROOT, "data", "indodoc", "train.jsonl")

# Output File (Hanya Test saja)
TEST_FILE = os.path.join(ROOT, "data", "indodoc", "test_new.jsonl")


def paraphrase_title_to_query(title):
    """
    Fungsi cerdas untuk mengubah judul dokumen resmi
    menjadi gaya pertanyaan natural yang sering diketik pengguna.
    """
    title_lower = title.lower()

    if "sop" in title_lower or "standar operasional" in title_lower:
        prefixes = [
            "Bagaimana prosedur untuk ",
            "Tolong carikan SOP tentang ",
            "Apa langkah-langkah dalam "
        ]
        # Hapus kata SOP di awal agar tidak dobel
        clean_title = title_lower.replace("sop ", "").strip()
        return random.choice(prefixes) + clean_title + "?"

    elif "sk " in title_lower or "pengangkatan" in title_lower or "penetapan" in title_lower:
        prefixes = [
            "Dokumen surat keputusan untuk ",
            "Siapa saja yang masuk dalam ",
            "Tolong carikan penetapan terkait "
        ]
        return random.choice(prefixes) + title_lower + "?"

    elif "foto" in title_lower or "dokumentasi" in title_lower:
        prefixes = [
            "Ada foto untuk acara ",
            "Tolong tampilkan dokumentasi saat ",
            "Saya ingin melihat arsip foto "
        ]
        clean_title = title_lower.replace("foto ", "").replace("dokumentasi ", "").strip()
        return random.choice(prefixes) + clean_title + "?"

    elif "pedoman" in title_lower or "panduan" in title_lower or "buku" in title_lower:
        prefixes = [
            "Di mana saya bisa membaca ",
            "Saya butuh buku panduan mengenai ",
            "Apakah ada pedoman untuk "
        ]
        return random.choice(prefixes) + title_lower + "?"

    elif "sertifikat" in title_lower or "akreditasi" in title_lower:
        prefixes = [
            "Berapa nilai akreditasi untuk ",
            "Tolong carikan sertifikat ",
            "Saya butuh bukti dokumen "
        ]
        clean_title = title_lower.replace("sertifikat ", "").strip()
        return random.choice(prefixes) + clean_title + "?"

    elif "laporan" in title_lower:
        return "Saya ingin membaca " + title_lower + "."

    elif "kontrak" in title_lower or "hibah" in title_lower:
        return "Tolong carikan dokumen " + title_lower + "."

    else:
        prefixes = [
            "Tolong carikan informasi tentang ",
            "Apa yang dimaksud dengan ",
            "Berikan saya dokumen mengenai "
        ]
        return random.choice(prefixes) + title_lower + "?"


def process_and_generate_test():
    if not os.path.exists(INPUT_FILE):
        print(f"[!] File {INPUT_FILE} tidak ditemukan. Pastikan file input tersedia.")
        return

    all_data = []
    print(f"[*] Membaca data dari {INPUT_FILE}...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "title" in data and "content" in data:
                    all_data.append(data)
            except json.JSONDecodeError:
                continue

    total_data = len(all_data)
    print(f"[*] Total dataset ditemukan: {total_data} baris.")

    # Acak data agar pengambilan 20% tidak bias berdasarkan urutan
    random.shuffle(all_data)

    # Hitung batas 80% - 20%
    split_index = int(total_data * 0.8)

    # Kita HANYA mengambil bagian 20% akhir untuk test
    test_data_raw = all_data[split_index:]

    print(f"[*] Mengambil {len(test_data_raw)} data (20%) untuk dijadikan Test Set.")

    # Simpan Test Data (Ubah 'title' menjadi pertanyaan natural)
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        for item in test_data_raw:
            original_title = item["title"]

            # Jadikan title sebagai pertanyaan natural
            natural_query = paraphrase_title_to_query(original_title)

            new_item = {
                "title": natural_query,
                "content": item["content"]  # Konten/dokumen jawaban biarkan persis seperti asli
            }
            json.dump(new_item, f, ensure_ascii=False)
            f.write("\n")

    print(f"[+] File Test baru berhasil dibuat dan tersimpan di: {TEST_FILE}")
    print("\n🎉 Proses Selesai! Anda siap untuk menjalankan skrip evaluasi MiniLM.")


if __name__ == "__main__":
    process_and_generate_test()