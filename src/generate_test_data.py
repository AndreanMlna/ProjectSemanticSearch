import os
import json
import random
import re

# Konfigurasi Seed agar sampling data konsisten & deterministik
random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input Dataset Hasil Preprocessing SERANAH
INPUT_FILE = os.path.join(ROOT, "data", "indodoc", "train_seranah.jsonl")

# Output File Test Baru untuk Evaluasi Search & IR Benchmark
TEST_FILE = os.path.join(ROOT, "data", "indodoc", "test_new_seranah.jsonl")


def paraphrase_title_to_query(title: str) -> str:
    """
    Fungsi cerdas berbasis pola tematik arsip kampus UNIDA Gontor
    untuk mengubah judul dokumen resmi menjadi kueri pertanyaan natural
    yang realistis diketik oleh pengguna / civitas akademika.
    """
    title_lower = title.lower().strip()

    # 1. Dokumen SOP / Standar Operasional
    if "sop" in title_lower or "standar operasional" in title_lower:
        prefixes = [
            "Bagaimana prosedur untuk ",
            "Tolong carikan SOP tentang ",
            "Apa langkah-langkah dalam ",
            "Di mana panduan standar operasional untuk ",
        ]
        clean_title = re.sub(r"\bsop\b|\bstandar operasional\b", "", title_lower).strip()
        return random.choice(prefixes) + clean_title + "?"

    # 2. Dokumen SK / Keputusan / Pengangkatan / Penetapan / Panitia / Pemberhentian
    elif any(k in title_lower for k in ["sk ", "keputusan", "pengangkatan", "penetapan", "panitia", "pemberhentian"]):
        prefixes = [
            "Dokumen surat keputusan untuk ",
            "Siapa yang ditetapkan dalam ",
            "Tolong carikan SK penetapan terkait ",
            "Surat keputusan rektor mengenai ",
            "Informasi resmi tentang ",
        ]
        return random.choice(prefixes) + title_lower + "?"

    # 3. Dokumen Perencanaan (Renstra, Renbang, Renop)
    elif any(k in title_lower for k in ["renstra", "renbang", "renop", "rencana strategis", "rencana pengembangan"]):
        prefixes = [
            "Tolong carikan dokumen rencana strategis ",
            "Di mana dokumen Renstra atau Renbang untuk ",
            "Apa saja target rencana pengembangan ",
            "Buku rencana strategis mengenai ",
        ]
        return random.choice(prefixes) + title_lower + "?"

    # 4. Dokumen Panduan / Pedoman / Buku
    elif any(k in title_lower for k in ["pedoman", "panduan", "buku"]):
        prefixes = [
            "Di mana saya bisa membaca ",
            "Saya butuh buku panduan mengenai ",
            "Apakah ada pedoman resmi untuk ",
            "Tolong berikan petunjuk atau panduan ",
        ]
        return random.choice(prefixes) + title_lower + "?"

    # 5. Dokumen Surat Permohonan / Pengajuan
    elif any(k in title_lower for k in ["permohonan", "pengajuan", "surat permohonan"]):
        prefixes = [
            "Tolong carikan arsip surat permohonan ",
            "Format dan isi pengajuan untuk ",
            "Dokumen permohonan mengenai ",
        ]
        return random.choice(prefixes) + title_lower + "?"

    # 6. Dokumen Laporan Kegiatan / Workshop / Bimtek / Kerjasama
    elif any(k in title_lower for k in ["laporan", "workshop", "bimtek", "sosialisasi", "kerjasama", "magang", "internship"]):
        prefixes = [
            "Saya ingin melihat laporan pelaksanaan ",
            "Tolong tampilkan hasil kegiatan ",
            "Dokumen laporan kegiatan mengenai ",
            "Informasi pelaksanaan acara ",
        ]
        return random.choice(prefixes) + title_lower + "?"

    # 7. Dokumen Foto / Dokumentasi Visual
    elif any(k in title_lower for k in ["foto", "dokumentasi"]):
        prefixes = [
            "Ada dokumentasi foto untuk ",
            "Tolong tampilkan foto arsip kegiatan ",
            "Saya ingin melihat dokumentasi ",
        ]
        clean_title = re.sub(r"\bfoto\b|\bdokumentasi\b", "", title_lower).strip()
        return random.choice(prefixes) + clean_title + "?"

    # 8. Dokumen Profil Unit Kerja / Lembaga
    elif "profil" in title_lower:
        prefixes = [
            "Informasi profil mengenai ",
            "Tolong carikan struktur dan profil ",
            "Bagaimana profil dan visi dari ",
        ]
        return random.choice(prefixes) + title_lower + "?"

    # 9. Dokumen Seleksi / Kelulusan / Akreditasi
    elif any(k in title_lower for k in ["seleksi", "kelulusan", "akreditasi", "sertifikat"]):
        prefixes = [
            "Bagaimana hasil dan dokumen ",
            "Tolong carikan pengumuman resmi tentang ",
            "Bukti kelulusan atau sertifikat ",
        ]
        return random.choice(prefixes) + title_lower + "?"

    # 10. Default Pola Umum
    else:
        prefixes = [
            "Tolong carikan informasi tentang ",
            "Apa isi dokumen mengenai ",
            "Berikan saya arsip terkait ",
            "Saya mencari arsip ",
        ]
        return random.choice(prefixes) + title_lower + "?"


def process_and_generate_test():
    """Membaca train_seranah.jsonl, mengambil 20% data uji, memparafrase judul menjadi
    kueri pencarian alami, dan menyimpannya ke test_new_seranah.jsonl."""
    print("=" * 65)
    print("[*] MEMULAI GENERASI DATA UJI (TEST SET) DARI DATASET SERANAH")
    print(f"[*] File Input : {INPUT_FILE}")
    print(f"[*] File Output: {TEST_FILE}")
    print("=" * 65)

    if not os.path.exists(INPUT_FILE):
        print(f"[!] ERROR: File {INPUT_FILE} tidak ditemukan.")
        print(f"    Pastikan Anda sudah menjalankan prepare_dataset.py terlebih dahulu.")
        return

    all_data = []
    print(f"\n[*] Membaca data dari {INPUT_FILE}...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "title" in data and "content" in data:
                    all_data.append(data)
            except json.JSONDecodeError:
                continue

    total_data = len(all_data)
    print(f"[+] Total data ditemukan: {total_data} baris dokumen.")

    if total_data == 0:
        print("[!] ERROR: Data input kosong.")
        return

    # Acak urutan secara deterministik (Seed 42)
    random.shuffle(all_data)

    # Ambil 20% data untuk dijadikan Test Set Evaluasi
    split_index = int(total_data * 0.8)
    test_data_raw = all_data[split_index:]

    print(f"[+] Mengambil {len(test_data_raw)} dokumen (20%) sebagai Test Set evaluasi.")
    print("[*] Memparafrase judul resmi menjadi kueri pencarian natural...")

    # Simpan ke Test File dengan kueri natural yang telah diparafrase
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        for item in test_data_raw:
            original_title = item["title"]
            natural_query = paraphrase_title_to_query(original_title)

            new_item = {
                "title": natural_query,
                "content": item["content"],
                "keywords": item.get("keywords", "-"),
            }
            json.dump(new_item, f, ensure_ascii=False)
            f.write("\n")

    print("\n" + "=" * 65)
    print("[✅] SUKSES! Data uji evaluasi berhasil dibuat.")
    print(f"[📁] File Test Tersimpan di: {TEST_FILE}")
    print(f"[📊] Total Kueri Uji: {len(test_data_raw)} pasangan kueri-dokumen")
    print("=" * 65)


if __name__ == "__main__":
    process_and_generate_test()