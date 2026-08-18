# src/preprocess.py
import json
import os
import re
import pandas as pd
# ═══════════════════════════════════════════════════════════════════
# 1. IMPORT TEXT EXTRACTOR ANDA
# ═══════════════════════════════════════════════════════════════════
from src.text_extractor import extract_text_from_file

# --- KONFIGURASI ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_FILE = os.path.join(ROOT, "dataset_arsip.xlsx")
OUTPUT_DIR = os.path.join(ROOT, "data", "indodoc")

# Asumsikan folder tempat menyimpan file PDF/Docx asli di server Anda
UPLOAD_DIR = os.path.join(ROOT, "uploads")

# File Output (Metadata + Rich Content)
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.jsonl")

# Buat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text):
    if pd.isna(text) or text == "":
        return ""
    text = str(text).replace('\n', ' ').replace('\r', '').strip()
    return re.sub(' +', ' ', text)


def generate_metadata():
    print(f"[*] Membaca data dari {EXCEL_FILE}...")

    try:
        df = pd.read_excel(EXCEL_FILE)
    except Exception as e:
        print(f"[!] Gagal membaca Excel: {e}")
        return

    print(f"[*] Kolom ditemukan: {list(df.columns)}")

    metadata_data = []
    print("[*] Memproses baris data dan mengekstrak teks fisik...")

    for index, row in df.iterrows():
        title = clean_text(row.get('title'))
        description = clean_text(row.get('description'))
        keywords = clean_text(row.get('keywords'))
        file_path_rel = clean_text(row.get('file_path'))  # Path dari excel
        file_name = clean_text(row.get('file_name'))

        if not title and not description:
            continue

        # ═══════════════════════════════════════════════════════════════════
        # 2. PROSES EKSTRAKSI TEKS DOKUMEN FISIK
        # ═══════════════════════════════════════════════════════════════════
        # Gabungkan folder UPLOAD_DIR dengan path file dari Excel
        absolute_file_path = os.path.join(UPLOAD_DIR, file_path_rel)

        print(f"    [-] Mencoba mengekstrak teks: {file_name}")
        extracted_text = extract_text_from_file(absolute_file_path)

        # ═══════════════════════════════════════════════════════════════════
        # 3. STRATEGI KONTEN KAYA (HYBRID CONTENT)
        # ═══════════════════════════════════════════════════════════════════
        # Jika file fisik (.pdf/.docx) ada dan berhasil diekstrak teksnya:
        if extracted_text.strip():
            # Gabungkan Deskripsi Excel + Isu Dokumen Asli untuk akurasi maksimal
            full_content = f"{description}\n\n[Isi Dokumen Penuh]:\n{extracted_text}"
            logger_status = "Berhasil Ekstrak File"
        else:
            # Fallback ke deskripsi excel jika file fisik tidak ditemukan / gagal baca
            full_content = description
            logger_status = "Gagal/Tidak Ada File (Fallback ke Deskripsi)"

        if keywords:
            full_content += f" Kata kunci: {keywords}."

        # --- SUSUN DATA LENGKAP UNTUK CHROMADB / API ---
        entry_metadata = {
            "title": title,
            "content": full_content,  # Sekarang berisi teks kaya/full-text jika sukses
            "file_path": file_path_rel,
            "file_name": file_name,
            "status": logger_status
        }
        metadata_data.append(entry_metadata)

    print(f"[*] Menyimpan {len(metadata_data)} data ke {METADATA_FILE}...")

    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        for entry in metadata_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print("-" * 30)
    print(f"[DONE] File Metadata & Rich Content Berhasil Dibuat!")
    print(f"Lokasi: {METADATA_FILE}")
    print("-" * 30)


if __name__ == "__main__":
    generate_metadata()