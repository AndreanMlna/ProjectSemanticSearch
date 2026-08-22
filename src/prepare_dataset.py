# src/prepare_dataset.py
import json
import os
import re
import random

# --- KONFIGURASI PATH ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "data", "indodoc")

# Sumber dataset dari SERANAH UNIDA Gontor
SOURCE_FILE = os.path.join(OUTPUT_DIR, "seranah_archives.jsonl")
if not os.path.exists(SOURCE_FILE):
    # Fallback ke folder data/ jika file diletakkan langsung di root data/
    fallback_path = os.path.join(ROOT, "data", "seranah_archives.jsonl")
    if os.path.exists(fallback_path):
        SOURCE_FILE = fallback_path

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_seranah.jsonl")
TEST_FILE = os.path.join(OUTPUT_DIR, "test_seranah.jsonl")

# Pastikan folder output tersedia
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text: str) -> str:

    if not text or not isinstance(text, str):
        return ""

    # 1. Lowercase (jadikan huruf kecil semua)
    text = text.lower()

    # 2. Hapus Tag HTML jika ada
    text = re.sub(r"<.*?>", " ", text)

    # 3. Hapus URL / Tautan Web & Alamat Email
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", " ", text)

    # 4. Hapus Karakter Anomali & Simbol Aneh
    # Hanya menyisakan huruf alfabet a-z, angka 0-9, spasi, dan tanda baca dasar
    text = re.sub(r"[^a-z0-9\s.,?!-]", " ", text)

    # 5. Hapus escape character (\n, \r, \t)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # 6. Hapus spasi ganda/berlebih dan rapikan ujung teks
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare():
    """
    Membaca data seranah_archives.jsonl, mengekstraksi HANYA:
    - title
    - content (dari field description)
    - keywords
    Lalu membersihkan teks dan menyimpannya ke train.jsonl dan test.jsonl.
    """
    print("=" * 65)
    print("[*] MEMULAI DATA PREPROCESSING DATASET SERANAH")
    print(f"[*] File Sumber: {SOURCE_FILE}")
    print("=" * 65)

    if not os.path.exists(SOURCE_FILE):
        print(f"[!] ERROR: File sumber {SOURCE_FILE} tidak ditemukan.")
        print(f"    Pastikan file seranah_archives.jsonl berada di folder data/ atau data/indodoc/")
        return

    raw_data = []
    try:
        with open(SOURCE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))
    except Exception as e:
        print(f"[!] Gagal membaca file JSONL: {e}")
        return

    print(f"[*] Total baris data mentah yang dibaca: {len(raw_data)}")

    processed_data = []
    print("[*] Memulai ekstraksi & pembersihan (preprocessing) teks...")

    for row in raw_data:
        # 1. Ekstraksi Field Sesuai Kebutuhan:
        raw_title = row.get("title", "")
        raw_description = row.get("description", "")  # Diubah namanya menjadi 'content'
        raw_keywords = row.get("keywords", "-")

        # 2. Pembersihan Teks (Text Preprocessing)
        title = clean_text(raw_title)
        content = clean_text(raw_description)
        keywords = clean_text(raw_keywords)

        # 3. Validasi: Abaikan data jika title dan content kosong
        if not title and not content:
            continue

        # 4. Bentuk Struktur Bersih: HANYA title, content, dan keywords
        entry = {
            "title": title,
            "content": content,
            "keywords": keywords if keywords else "-",
        }

        processed_data.append(entry)

    print(f"[*] Total data bersih yang berhasil diproses: {len(processed_data)}")

    # --- PEMBAGIAN TRAIN & TEST ---
    # 1. Simpan Train (Semua Data Bersih - 100%)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for entry in processed_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    print(f"[+] Berhasil menyimpan {len(processed_data)} data ke file TRAIN: {TRAIN_FILE}")

    # 2. Simpan Test (Sampling 20% untuk Evaluasi)
    sample_size = int(len(processed_data) * 0.2)
    sample_size = max(1, sample_size) if len(processed_data) > 0 else 0

    if sample_size > 0:
        test_data = random.sample(processed_data, k=sample_size)
        with open(TEST_FILE, "w", encoding="utf-8") as f:
            for entry in test_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        print(f"[+] Berhasil menyimpan {len(test_data)} data sampel ke file TEST: {TEST_FILE}")
    else:
        print("[!] Data tidak cukup untuk membuat test.jsonl")

    print("\n" + "=" * 65)
    print(" DATA PREPROCESSING SELESAI!")
    print(f" Output Train (Untuk Embedding & ChromaDB): {TRAIN_FILE}")
    print(f" Output Test  (Untuk Evaluasi): {TEST_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    prepare()