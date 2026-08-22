"""
Script Stress Testing & Concurrency Benchmark (10.000 Kueri Unik)
Menguji performa, latensi, dan throughput endpoint /search (Semantic Search & Reranker)
di bawah beban akses tinggi secara konkuren (Cache Miss aktual).
"""

import concurrent.futures
import os
import time
from typing import List, Tuple
from dotenv import load_dotenv

# Muat konfigurasi environment
load_dotenv()

API_URL: str = os.getenv("API_URL", "http://localhost:8000/search")
API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "seranah_secret_key_2026")
HEADERS: dict = {"X-API-Key": API_SECRET_KEY}

# =====================================================================
# 1. GENERATOR 10.000 VARIASI KUERI ARSIP UNIK (REALISTIS & DOMAIN-SPECIFIC)
# =====================================================================
# 20 Prefiks / Jenis Dokumen
prefixes: List[str] = [
    "pedoman", "aturan", "SK rektor tentang", "panduan", "syarat",
    "prosedur", "surat keputusan", "standar operasional", "SOP", "petunjuk teknis",
    "alur pengajuan", "mekanisme", "kebijakan", "ketentuan", "kriteria",
    "dokumen", "laporan", "formulir", "jadwal", "evaluasi"
]  # 20 item

# 25 Topik Utama Arsip Kampus
topik: List[str] = [
    "magang", "skripsi", "wisuda", "UKT", "beasiswa",
    "cuti akademik", "KRS", "KHS", "MBKM", "organisasi mahasiswa",
    "penelitian dosen", "pengabdian masyarakat", "akreditasi prodi", "renstra fakultas", "yudisium",
    "ujian komprehensif", "sidang tugas akhir", "registrasi ulang", "pembimbing akademik", "etika akademik",
    "sarana prasarana", "laboratorium", "seminar proposal", "prestasi mahasiswa", "tata tertib kampus"
]  # 25 item

# 20 Unit Kerja / Satuan Organisasi Kampus
unit_kerja: List[str] = [
    "fakultas sains dan teknologi", "fakultas tarbiyah", "fakultas syariah", "fakultas ushuluddin", "pascasarjana",
    "program studi informatika", "program studi agroteknologi", "sekretariat universitas", "biro administrasi akademik", "lembaga penjaminan mutu",
    "LPPM", "UPT perpustakaan", "UPT bahasa", "dewan mahasiswa", "senat universitas",
    "laboratorium komputer", "badan pengelola keuangan", "kemahasiswaan", "alumni", "dosen dan tendik"
]  # 20 item

# Menghasilkan tepat 10.000 kombinasi kueri unik (20 * 25 * 20 = 10.000)
QUERIES: List[str] = [
    f"{p} {t} {u}"
    for p in prefixes
    for t in topik
    for u in unit_kerja
][:10000]

JUMLAH_REQUEST: int = len(QUERIES)
MAX_WORKERS: int = int(os.getenv("BENCHMARK_WORKERS", "10"))  # Jumlah thread konkuren

# =====================================================================
# 2. FUNGSI WORKER PENGIRIMAN REQUEST
# =====================================================================
import requests


def tembak_api(id_request: int) -> Tuple[float, int]:
    """Mengirim 1 request pencarian semantik ke API backend dan mengukur durasi respons."""
    start = time.perf_counter()
    kata_kunci = QUERIES[id_request]
    payload = {"query": kata_kunci, "top_k": 5}

    try:
        response = requests.post(API_URL, json=payload, headers=HEADERS, timeout=120)
        durasi = time.perf_counter() - start
        return durasi, response.status_code
    except Exception:
        durasi = time.perf_counter() - start
        return durasi, 500


# =====================================================================
# 3. EKSEKUSI STRESS TESTING & BENCHMARKING
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("MEMULAI STRESS TESTING & BENCHMARK (10.000 KUERI UNIK)")
    print(f"[*] Target Endpoint   : {API_URL}")
    print(f"[*] Total Kueri Unik  : {JUMLAH_REQUEST:,} request")
    print(f"[*] Konkurensi Worker : {MAX_WORKERS} concurrent threads")
    print(f"[*] Skenario Beban    : CACHE MISS (100% Komputasi AI Aktual)")
    print("=" * 65)

    waktu_mulai_total = time.perf_counter()
    hasil: List[Tuple[float, int]] = []
    selesai_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(tembak_api, i): i for i in range(JUMLAH_REQUEST)}

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            hasil.append(res)
            selesai_count += 1

            # Log kemajuan setiap 500 request agar monitoring mudah & bersih
            if selesai_count % 500 == 0 or selesai_count == JUMLAH_REQUEST:
                persentase = (selesai_count / JUMLAH_REQUEST) * 100
                elapsed = time.perf_counter() - waktu_mulai_total
                rps = selesai_count / elapsed if elapsed > 0 else 0
                print(
                    f"[{time.strftime('%H:%M:%S')}] Progress: {selesai_count:,}/{JUMLAH_REQUEST:,} "
                    f"({persentase:.1f}%) | Elapsed: {elapsed:.1f}s | Throughput: {rps:.2f} req/s"
                )

    waktu_selesai_total = time.perf_counter() - waktu_mulai_total

    # =====================================================================
    # 4. ANALISIS STATISTIK & METRIK HASIL PENGUJIAN
    # =====================================================================
    durasi_list = sorted([h[0] for h in hasil])
    sukses = sum(1 for h in hasil if h[1] == 200)
    gagal = JUMLAH_REQUEST - sukses
    rata_rata = sum(durasi_list) / len(durasi_list) if durasi_list else 0
    throughput = JUMLAH_REQUEST / waktu_selesai_total if waktu_selesai_total > 0 else 0

    # Persentil Latensi
    p50 = durasi_list[int(len(durasi_list) * 0.50)] if durasi_list else 0
    p95 = durasi_list[int(len(durasi_list) * 0.95)] if durasi_list else 0
    p99 = durasi_list[int(len(durasi_list) * 0.99)] if durasi_list else 0
    min_lat = durasi_list[0] if durasi_list else 0
    max_lat = durasi_list[-1] if durasi_list else 0

    print("\n" + "=" * 65)
    print("RINGKASAN LAPORAN STRESS TESTING (10.000 REQUESTS)")
    print("=" * 65)
    print(f"Total Permintaan (Requests)   : {JUMLAH_REQUEST:,}")
    print(f"Permintaan Sukses (200 OK)    : {sukses:,} ({sukses / JUMLAH_REQUEST * 100:.2f}%)")
    print(f"Permintaan Gagal (Error)      : {gagal:,}")
    print(f"Total Waktu Pengujian         : {waktu_selesai_total:.2f} detik ({waktu_selesai_total / 60:.2f} menit)")
    print(f"Throughput Rata-rata          : {throughput:.2f} request/detik (RPS)")
    print("-" * 65)
    print("METRIK LATENSI RESPONS (DETIK):")
    print(f"  • Rata-rata Latensi         : {rata_rata:.4f} detik")
    print(f"  • Minimum Latensi           : {min_lat:.4f} detik")
    print(f"  • Median (P50) Latensi      : {p50:.4f} detik")
    print(f"  • Persentil 95 (P95)        : {p95:.4f} detik")
    print(f"  • Persentil 99 (P99)        : {p99:.4f} detik")
    print(f"  • Maksimum Latensi          : {max_lat:.4f} detik")
    print("=" * 65)