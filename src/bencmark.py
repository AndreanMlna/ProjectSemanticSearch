import requests
import time
import concurrent.futures

API_URL = "http://localhost:8000/search"

# 1. Membuat 50 kata kunci unik secara otomatis
prefixes = ["pedoman", "aturan", "SK rektor tentang", "panduan", "syarat"]
topik = [
    "magang", "skripsi", "wisuda", "UKT", "beasiswa",
    "cuti akademik", "KRS", "KHS", "MBKM", "organisasi mahasiswa"
]

# Menghasilkan 50 kombinasi unik
QUERIES = [f"{p} {t}" for p in prefixes for t in topik]
JUMLAH_REQUEST = len(QUERIES)


def tembak_api(id_request):
    start = time.time()
    # Mengambil kata kunci yang berbeda berdasarkan urutan
    kata_kunci = QUERIES[id_request]
    payload = {"query": kata_kunci, "top_k": 5}

    try:
        response = requests.post(API_URL, json=payload)
        durasi = time.time() - start
        return durasi, response.status_code
    except Exception as e:
        return 0, str(e)


print(f"[*] Memulai Benchmark CACHE MISS: {JUMLAH_REQUEST} kata kunci BERBEDA...")
waktu_mulai_total = time.time()

# 2. Menjalankan 50 request secara paralel
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    hasil = list(executor.map(tembak_api, range(JUMLAH_REQUEST)))

waktu_selesai_total = time.time() - waktu_mulai_total

sukses = sum(1 for h in hasil if h[1] == 200)
rata_rata = sum(h[0] for h in hasil) / len(hasil)

print("-" * 30)
print(f"Skenario        : CACHE MISS (Kompilasi AI Aktual)")
print(f"Total Request   : {JUMLAH_REQUEST}")
print(f"Berhasil        : {sukses}")
print(f"Rata-rata Waktu : {rata_rata:.3f} detik per request")
print(f"Total Waktu Tes : {waktu_selesai_total:.3f} detik")
print("-" * 30)