# src/prepare_dataset.py
import json
import os
import re
import random

# --- KONFIGURASI ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(ROOT, "data", "indodoc")
# Menyesuaikan target posisi file ke data/indodoc/metadata.jsonl
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.jsonl")

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.jsonl")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.jsonl")

# Buat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text):
    """
    Fungsi preprocessing untuk membersihkan teks sebelum fine-tuning.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Lowercase (jadikan huruf kecil semua)
    text = text.lower()

    # 2. Hapus URL / Link (jika ada terselip di metadata)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. Hapus Karakter Anomali & Simbol Aneh
    # (Hanya menyisakan huruf a-z, angka 0-9, spasi, dan tanda baca dasar: .,?-)
    text = re.sub(r'[^a-z0-9\s.,?!-]', ' ', text)

    # 4. Hapus newline (\n) dan tab (\t)
    text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')

    # 5. Hapus spasi ganda/berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def prepare():
    print(f"[*] Membaca data dari {METADATA_FILE}...")

    if not os.path.exists(METADATA_FILE):
        print(f"[!] ERROR: File {METADATA_FILE} tidak ditemukan. Pastikan path-nya benar.")
        return

    raw_data = []
    try:
        # Membaca file JSONL (baris per baris)
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Abaikan baris kosong
                    raw_data.append(json.loads(line))
    except Exception as e:
        print(f"[!] Gagal membaca JSONL: {e}")
        return

    print(f"[*] Total baris data mentah: {len(raw_data)}")

    processed_data = []
    print("[*] Memulai pembersihan (preprocessing) data...")

    for row in raw_data:
        # PERBAIKAN: Ambil langsung dari key 'title' dan 'content' sesuai isi metadata.jsonl
        raw_title = row.get('title', '')
        raw_content = row.get('content', '')

        # Terapkan fungsi pembersihan data
        title = clean_text(raw_title)
        content = clean_text(raw_content)

        # Validasi: Abaikan data jika kosong setelah dibersihkan
        if not title and not content:
            continue

        # Simpan struktur yang dibutuhkan untuk fine-tuning
        entry = {
            "title": title,
            "content": content
        }

        processed_data.append(entry)

    print(f"[*] Data bersih siap diproses: {len(processed_data)}")

    # --- PEMBAGIAN TRAIN & TEST ---
    # Sesuai instruksi: TRAIN diisi semua data, TEST hanya sebagian.

    # 1. Simpan Train (Semua Data)
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    print(f"[*] Berhasil menyimpan {len(processed_data)} data ke file TRAIN.")

    # 2. Simpan Test (Sampling 20% untuk Evaluasi)
    sample_size = int(len(processed_data) * 0.2)
    sample_size = max(1, sample_size) if len(processed_data) > 0 else 0

    if sample_size > 0:
        test_data = random.sample(processed_data, k=sample_size)
        with open(TEST_FILE, 'w', encoding='utf-8') as f:
            for entry in test_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f"[*] Berhasil menyimpan {len(test_data)} data sampel ke file TEST.")
    else:
        print("[!] Tidak ada cukup data untuk membuat file test.jsonl")

    print(f"\n[DONE] Dataset siap digunakan untuk training!")
    print(f"       Target: {TRAIN_FILE}")


if __name__ == "__main__":
    prepare()