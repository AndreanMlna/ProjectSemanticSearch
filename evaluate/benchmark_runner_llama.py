import os
import time
import logging
from sentence_transformers import SentenceTransformer, util

# Pilih salah satu agent yang mau dites
from eksperimen.rag_agent_llama import get_rag_agent
from evaluate.benchmark_logger_llama import log_evaluation

logging.basicConfig(level=logging.INFO)

# --- KONFIGURASI MODEL EVALUATOR MANDIRI ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALUATOR_MODEL_PATH = os.path.join(ROOT, "output", "minilm-dokumen-arsip-boosted")

print(f"[*] Loading Evaluator Model dari: {EVALUATOR_MODEL_PATH}")
try:
    evaluator_model = SentenceTransformer(EVALUATOR_MODEL_PATH)
    print("[*] Evaluator Model berhasil dimuat.")
except Exception as e:
    logging.error(f"[!] Gagal memuat model evaluator. Pastikan path benar. Error: {e}")
    evaluator_model = None

TEST_CASES = [
    {
        "q": "Untuk apa tim perumus visi dan misi UNIDA Gontor dibentuk?",
        "ideal": "Tim tersebut dibentuk berdasarkan amanat rektor untuk merumuskan visi dan misi Universitas Darussalam Gontor."
    },
    {
        "q": "Siapa yang mengesahkan visi dan misi UNIDA Gontor?",
        "ideal": "Pengesahan visi dan misi oleh Rektor Universitas Darussalam Gontor."
    },
    {
        "q": "Siapa yang menetapkan struktur fungsionaris CIES?",
        "ideal": "Struktur fungsionaris Centre for Islamic Economic Studies (CIES) ditetapkan oleh Rektor UNIDA Gontor."
    },
    {
        "q": "Buku pedoman penelitian dan PKM UNIDA berfungsi sebagai apa?",
        "ideal": "Buku ini menjadi pedoman pelaksanaan program hibah internal penelitian dan pengabdian kepada masyarakat di UNIDA Gontor."
    },
    {
        "q": "Program ujian bahasa tiap semester di UNIDA bertujuan untuk apa?",
        "ideal": "Program ini diberlakukan untuk menunjang peningkatan kemampuan bahasa Arab dan Inggris mahasiswa UNIDA Gontor."
    },
    {
        "q": "Panitia apa yang diangkat untuk ibadah qurban di Kampus A?",
        "ideal": "Rektor mengangkat panitia ibadah qurban guna kelancaran pelaksanaan ibadah qurban di Kampus A UNIDA Gontor."
    },
    {
        "q": "Berapa rentang waktu rencana induk pengembangan UNIDA Gontor?",
        "ideal": "Rencana induk pengembangan atau disingkat RENIP tahun 2014 sampai dengan tahun 2040 Universitas Darussalam Gontor."
    },
    {
        "q": "Apa tujuan kebijakan SPMI di UNIDA Gontor?",
        "ideal": "Kebijakan SPMI bertujuan meningkatkan relevansi atmosfer akademik serta daya saing dan efisiensi manajemen pendidikan di UNIDA Gontor."
    },
    {
        "q": "Siapa yang mengeluarkan SK pengangkatan Rektor UNIDA Gontor?",
        "ideal": "SK pengangkatan Rektor dikeluarkan oleh Ketua Yayasan Perguruan Tinggi Darussalam (YPTD) Pondok Modern Darussalam Gontor."
    },
    {
        "q": "Apa tugas utama Pusat Bahasa yang didirikan di UNIDA?",
        "ideal": "Pusat bahasa sebagai biro yang menangani standar isi, proses, dan evaluasi bahasa di Universitas Darussalam Gontor."
    },
    {
        "q": "Siapa yang menetapkan SOP penyusunan RKAT tahunan UNIDA?",
        "ideal": "Penyusunan RKAT ditetapkan oleh bagian keuangan Universitas Darussalam Gontor."
    },
    {
        "q": "Untuk tahun akademik berapa salah satu hasil seleksi PMB gelombang pertama yang tercatat?",
        "ideal": "Hasil seleksi penerimaan mahasiswa baru gelombang pertama Universitas Darussalam Gontor tahun akademik 2018 - 2019."
    },
    {
        "q": "Siapa saja yang paling berhak dapat menerima beasiswa tahfidz Al-Qur'an di UNIDA?",
        "ideal": "Penerima beasiswa tahfidz Al-Qur'an bagi mahasiswa atau mahasiswi Universitas Darussalam Gontor yang memiliki hafalan Al-Qur'an."
    },
    {
        "q": "Kapan dan di mana KH Imam Zarkasyi lahir?",
        "ideal": "Lahir di desa Gontor Ponorogo pada tanggal 21 Maret 1910."
    },
    {
        "q": "Tanggal berapa KH Ahmad Sahal lahir?",
        "ideal": "KH Ahmad Sahal lahir di Desa Gontor, Ponorogo, pada 22 Mei 1901."
    },
    {
        "q": "Apa peringkat akreditasi terbaru Prodi Manajemen UNIDA?",
        "ideal": "Sertifikat akreditasi program studi manajemen dengan peringkat unggul yang berlaku dari 24 Juli 2023 sampai 24 Juli 2028."
    },
    {
        "q": "Bagaimana status akreditasi Prodi Ilmu Gizi UNIDA saat ini?",
        "ideal": "Sertifikat akreditasi program studi ilmu gizi dengan peringkat unggul yang berlaku dari 24 Maret 2023 sampai 23 Maret 2028."
    },
    {
        "q": "Apa fungsi Statuta bagi UNIDA Gontor?",
        "ideal": "Statuta berfungsi sebagai pedoman dasar penyelenggaraan kegiatan, mulai dari penetapan hingga evaluasi program pendidikan di UNIDA Gontor."
    },
    {
        "q": "Bagaimana cara jika ingin mengajukan Serdos di unida ?",
        "ideal": "Dosen harus sudah memiliki jabatan fungsional (jabfung) dan inpassing sebelum mengajukan sertifikasi dosen (Serdos)."
    },
    {
        "q": "Apa tujuan pendirian U3 di UNIDA Gontor?",
        "ideal": "Pendirian unit usaha UNIDA Gontor atau U3 guna penyediaan layanan pemenuhan kebutuhan harian bagi civitas akademika di Universitas Darussalam Gontor."
    }
]

def calculate_answer_similarity(answer: str, ideal: str) -> float:
    if evaluator_model is None:
        return 0.0
    if not answer or not ideal:
        return 0.0
    emb_answer = evaluator_model.encode(answer)
    emb_ideal = evaluator_model.encode(ideal)
    return util.cos_sim(emb_answer, emb_ideal).item()

def run_benchmark():
    agent = get_rag_agent()
    print(f"🚀 Memulai benchmark RAG End-to-End untuk model: {agent.model}")

    for i, case in enumerate(TEST_CASES, 1):
        q = case["q"]
        ideal = case["ideal"]

        print(f"\n[{i}/{len(TEST_CASES)}] Kueri: {q}")

        # 1. Catat Waktu & Panggil Agent
        start_time = time.time()
        resp = agent.answer(q)
        latency = time.time() - start_time

        # 2. Perbaikan Ekstraksi Konteks yang lebih aman
        # Jika 'full_context' belum ada, ini akan menghasilkan string kosong ""
        # Pastikan di rag_agent.py Anda menambahkan "full_context" ke sources
        contexts = []
        if hasattr(resp, 'sources') and isinstance(resp.sources, list):
            for source in resp.sources:
                # Mengambil dari 'full_context', jika tidak ada cari key lain
                text = source.get('full_context') or source.get('snippet') or source.get('content') or ''
                contexts.append(text)

        sim_score = calculate_answer_similarity(resp.answer, ideal)

        # 3. Log Hasil ke CSV
        log_evaluation(
            model_name=agent.model,
            question=q,
            answer=resp.answer,
            ground_truth=ideal,
            latency=latency,
            sources_count=resp.search_results_count,
            similarity_score=sim_score,
            contexts=contexts
        )
        print(f"✅ Selesai ({latency:.2f}s) | Similarity: {sim_score:.4f} | Konteks: {len(contexts[0]) if contexts else 0} chars")

if __name__ == "__main__":
    run_benchmark()