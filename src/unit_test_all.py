import os
import sys
import time
import json
import unittest
import numpy as np
from typing import Dict, Any, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv()

# ANSI Color formatting untuk output terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class Test01_EnvironmentAndConfig(unittest.TestCase):
    """Pengujian 1: Validasi Environment & File Konfigurasi"""

    def test_env_variables_exist(self):
        """Memastikan semua variabel environment wajib terdefinisi dari .env."""
        from src.config import (
            MODEL_PATH,
            CE_MODEL_PATH,
            COLLECTION_NAME,
            API_SECRET_KEY,
            CHROMA_PORT,
        )
        self.assertTrue(bool(MODEL_PATH), "HF_MODEL_NAME tidak boleh kosong di .env / config")
        self.assertTrue(bool(CE_MODEL_PATH), "CE_MODEL tidak boleh kosong di .env / config")
        self.assertTrue(bool(COLLECTION_NAME), "CHROMA_COLLECTION tidak boleh kosong di .env / config")
        self.assertTrue(bool(API_SECRET_KEY), "API_SECRET_KEY harus terdefinisi di .env")
        self.assertTrue(bool(CHROMA_PORT), "CHROMA_PORT harus terdefinisi di .env")

    def test_metadata_seranah_url(self):
        """Memastikan variabel environment URL metadata SERANAH terdefinisi dan valid."""
        seranah_url = os.getenv("METADATA_SERANAH", "").strip()
        self.assertTrue(bool(seranah_url), "Variabel METADATA_SERANAH tidak boleh kosong di .env")
        self.assertTrue(
            seranah_url.startswith(("http://", "https://")),
            f"URL METADATA_SERANAH harus diawali dengan http:// atau https://, ditemukan: {seranah_url}"
        )

    def test_sync_interval_hours(self):
        """Memastikan interval sinkronisasi bernilai numerik valid."""
        interval = float(os.getenv("SYNC_INTERVAL_HOURS", "2"))
        self.assertGreaterEqual(interval, 0.1, "Interval sync minimal 0.1 jam")



class Test02_TextPreprocessingAndHelpers(unittest.TestCase):
    """Pengujian 2: Preprocessing Teks & Helper Utility"""

    def test_clean_text_basic(self):
        """Menguji pembersihan teks dari whitespace berlebih dan karakter noise."""
        from src.preprocess import clean_text
        raw_text = "  Pengesahan   Pedoman   Organisasi Kemahasiswaan \n\n 2024  "
        cleaned = clean_text(raw_text)
        self.assertNotIn("  ", cleaned)
        self.assertEqual(cleaned, "Pengesahan Pedoman Organisasi Kemahasiswaan 2024")

    def test_extract_keywords_helper(self):
        """Menguji ekstraksi kata kunci dari teks."""
        from src.helpers import extract_keywords
        sample_text = "Dokumen pengesahan SK Rektor. Kata Kunci: SK, Rektor, Pedoman, Mahasiswa"
        keywords = extract_keywords(sample_text)
        self.assertIn("SK", keywords)
        self.assertIn("Pedoman", keywords)

    def test_extract_keywords_fallback(self):
        """Menguji fallback ekstraksi kata kunci jika pola tidak ditemukan."""
        from src.helpers import extract_keywords
        self.assertEqual(extract_keywords(""), "-")
        self.assertEqual(extract_keywords("Dokumen tanpa pola kata kunci"), "-")


class Test03_EmbeddingModel(unittest.TestCase):
    """Pengujian 3: Bi-Encoder Embedding Model (MiniLM)"""

    @classmethod
    def setUpClass(cls):
        from sentence_transformers import SentenceTransformer
        from src.config import MODEL_PATH
        cls.model = SentenceTransformer(MODEL_PATH)

    def test_embedding_dimension(self):
        """Memastikan dimensi vektor embedding berukuran 384."""
        sample_query = "Surat Keputusan Rektor tentang wisuda sarjana"
        vec = self.model.encode(sample_query)
        self.assertEqual(len(vec), 384, f"Dimensi embedding harus 384, ditemukan {len(vec)}")
        self.assertFalse(np.isnan(vec).any(), "Vektor embedding mengandung nilai NaN")

    def test_semantic_similarity_ranking(self):
        """Memastikan kueri semantik memiliki kemiripan lebih tinggi pada teks relevan."""
        query = "Pedoman beasiswa mahasiswa berprestasi"
        doc_relevant = "Informasi petunjuk teknis dan syarat pendaftaran beasiswa prestasi mahasiswa"
        doc_irrelevant = "Jadwal pemeliharaan genset dan instalasi listrik gedung asrama"

        emb_q = self.model.encode(query, normalize_embeddings=True)
        emb_rel = self.model.encode(doc_relevant, normalize_embeddings=True)
        emb_irrel = self.model.encode(doc_irrelevant, normalize_embeddings=True)

        sim_rel = np.dot(emb_q, emb_rel)
        sim_irrel = np.dot(emb_q, emb_irrel)

        self.assertGreater(
            sim_rel, sim_irrel,
            f"Dokumen relevan ({sim_rel:.4f}) harus memiliki skor lebih tinggi dari dokumen tidak relevan ({sim_irrel:.4f})"
        )


class Test04_CrossEncoderReranker(unittest.TestCase):
    """Pengujian 4: Cross-Encoder Hybrid Reranker & Format Output Bersih"""

    @classmethod
    def setUpClass(cls):
        from src.reranker import get_reranker
        from src.config import CE_MODEL_PATH
        cls.reranker = get_reranker(CE_MODEL_PATH)

    def test_reranker_clean_output_format(self):
        """
        Memastikan hasil reranking memiliki format output bersih:
        - Memiliki field 'uuid'
        - TIDAK memiliki field ganda 'id'
        - TIDAK memiliki field 'download_url'
        - Memiliki field 'score'
        """
        mock_chroma_results = {
            "ids": [["uuid-test-001", "uuid-test-002"]],
            "documents": [[
                "dokumen pedoman organisasi mahasiswa universitas darussalam gontor",
                "dokumen jadwal kuliah dan kalender akademik semester genap"
            ]],
            "metadatas": [[
                {
                    "uuid": "uuid-test-001",
                    "title": "Pedoman Organisasi Kemahasiswaan",
                    "category": "Arsip SK",
                    "year": "2024",
                    "uploader": "Sekretariat Universitas",
                    "unit_kerja": "Biro Administrasi",
                    "mime_type": "application/pdf",
                    "access_level": "PUBLIC",
                    "document_number": "1270",
                    "file_name": "SK_Pedoman_Organisasi.pdf",
                    "file_path": "SK_Pedoman_Organisasi.pdf",
                    "keywords": "SK, Rektor, Pedoman, Mahasiswa",
                },
                {
                    "uuid": "uuid-test-002",
                    "title": "Kalender Akademik 2024",
                    "category": "Akademik",
                    "year": "2024",
                    "uploader": "BAAK",
                    "unit_kerja": "Akademik",
                    "mime_type": "application/pdf",
                    "access_level": "PUBLIC",
                    "document_number": "501",
                    "file_name": "Kalender_Akademik_2024.pdf",
                    "file_path": "Kalender_Akademik_2024.pdf",
                    "keywords": "Kalender, Akademik, Jadwal",
                }
            ]],
            "distances": [[0.15, 0.65]]
        }

        results = self.reranker.rerank(
            query="pedoman organisasi mahasiswa",
            chroma_results=mock_chroma_results,
            top_k=2
        )

        self.assertGreaterEqual(len(results), 1, "Reranker harus mengembalikan minimal 1 hasil")
        top_doc = results[0]

        # Validasi struktur data bersih
        self.assertIn("uuid", top_doc, "Field 'uuid' harus ada pada output")
        self.assertNotIn("id", top_doc, "Field 'id' harus dihapus (tidak boleh duplikat dengan uuid)")
        self.assertNotIn("download_url", top_doc, "Field 'download_url' harus dihilangkan")
        self.assertIn("score", top_doc, "Field 'score' harus ada pada output reranker")
        self.assertEqual(top_doc["uuid"], "uuid-test-001", "Dokumen relevan harus menjadi peringkat 1")


class Test05_ChromaDBConnection(unittest.TestCase):
    """Pengujian 5: Koneksi Database Vektor ChromaDB"""

    def test_chroma_client_and_collection(self):
        """Memastikan koneksi ke ChromaDB dan koleksi arsip_kampus_v2 aktif."""
        from src.chroma_client import get_collection
        collection = get_collection()
        self.assertIsNotNone(collection, "Gagal mendapatkan koleksi ChromaDB")
        total_docs = collection.count()
        self.assertGreaterEqual(total_docs, 0, "Jumlah dokumen dalam ChromaDB harus berupa integer >= 0")


class Test06_SeranahSyncEngine(unittest.TestCase):
    """Pengujian 6: Mesin Sinkronisasi Metadata Live SERANAH (Direct-to-Chroma & Sample Checking)"""

    def test_sample_validation_logic(self):
        """Memastikan fungsi validasi sampel mendeteksi kesesuaian dan perbedaan data."""
        from src.sync_seranah_archives import check_chroma_synced_with_sample
        mock_remote = [
            {"uuid": "sample-uuid-001", "title": "Pedoman Organisasi", "description": "Isi pedoman"},
            {"uuid": "sample-uuid-002", "title": "Kalender Akademik", "description": "Isi jadwal"}
        ]
        is_synced, reason = check_chroma_synced_with_sample(mock_remote, sample_size=2)
        self.assertIsInstance(is_synced, bool)
        self.assertIsInstance(reason, str)

    def test_text_cleaning_in_sync_engine(self):
        """Memastikan fungsi clean_text in-memory membersihkan teks dengan benar."""
        from src.sync_seranah_archives import clean_text
        raw = "<p>SK Rektor <b>2024</b> https://unida.gontor.ac.id/sk.pdf</p>"
        cleaned = clean_text(raw)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("sk rektor", cleaned)

    def test_get_chroma_archives_info(self):
        """Memastikan pembacaan arsip langsung dari ChromaDB berjalan dengan benar."""
        from src.sync_seranah_archives import get_chroma_archives_info
        count, uuids = get_chroma_archives_info()
        self.assertIsInstance(count, int)
        self.assertIsInstance(uuids, set)
        self.assertGreaterEqual(count, 0)



class Test07_CacheAndRateLimiter(unittest.TestCase):
    """Pengujian 7: Cache Manager & Rate Limiting System"""

    def test_cache_hit_and_miss(self):
        """Menguji mekanisme simpan, ambil, dan hit/miss pada CacheManager."""
        from src.cache_manager import get_cache_manager
        cache = get_cache_manager()
        cache.clear()

        test_query = "kueri_uji_coba_cache_123"
        test_results = [{"uuid": "test-1", "title": "Arsip Uji Coba"}]

        # Pastikan cache awal kosong
        cached_val = cache.get(test_query, top_k=5)
        self.assertIsNone(cached_val, "Cache harus bernilai None sebelum disimpan")

        # Simpan ke cache: (query, top_k, results)
        cache.set(test_query, 5, test_results)

        # Ambil kembali dari cache
        retrieved = cache.get(test_query, top_k=5)
        self.assertIsNotNone(retrieved, "Cache harus mengembalikan data yang disimpan")
        self.assertEqual(retrieved[0]["uuid"], "test-1")

    def test_rate_limiter_logic(self):
        """Menguji batasan request per menit pada RateLimiter."""
        from src.rate_limiter import EndpointRateLimiter
        limiter = EndpointRateLimiter()
        limiter.set_limit("test_endpoint", 3)
        client_id = "127.0.0.1"

        # 3 request pertama harus diperbolehkan
        self.assertTrue(limiter.check_limit("test_endpoint", client_id)[0])
        self.assertTrue(limiter.check_limit("test_endpoint", client_id)[0])
        self.assertTrue(limiter.check_limit("test_endpoint", client_id)[0])

        # Request ke-4 harus ditolak (rate limit exceeded)
        allowed_4, info_4 = limiter.check_limit("test_endpoint", client_id)
        self.assertFalse(allowed_4, "Request ke-4 harus ditolak oleh Rate Limiter")
        self.assertEqual(info_4.get("remaining"), 0)


class Test08_FastApiEndpointsAndSecurity(unittest.TestCase):
    """Pengujian 8: Endpoint FastAPI, Autentikasi API Key & Keamanan"""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from src.main_api import app
        cls.client = TestClient(app)
        cls.valid_key = os.getenv("API_SECRET_KEY", "seranah_secret_key_2026")

    def test_health_check_public_access(self):
        """Memastikan endpoint /health bersifat publik (bisa diakses tanpa API Key)."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200, "Endpoint /health harus mengembalikan HTTP 200")
        data = response.json()
        self.assertIn("status", data)
        self.assertIn(data["status"], ["healthy", "degraded"])

    def test_search_unauthorized_without_key(self):
        """Memastikan endpoint /search menolak akses jika tanpa API Key (HTTP 401)."""
        response = self.client.post("/search", json={"query": "pedoman beasiswa", "top_k": 5})
        self.assertEqual(response.status_code, 401, "Harus mengembalikan HTTP 401 Unauthorized tanpa API Key")

    def test_search_unauthorized_with_invalid_key(self):
        """Memastikan endpoint /search menolak akses jika API Key salah (HTTP 401)."""
        headers = {"X-API-Key": "kunci_salah_12345"}
        response = self.client.post("/search", json={"query": "pedoman beasiswa", "top_k": 5}, headers=headers)
        self.assertEqual(response.status_code, 401, "Harus mengembalikan HTTP 401 jika API Key salah")

    def test_search_authorized_with_valid_key(self):
        """Memastikan endpoint /search berhasil diakses dengan API Key valid."""
        headers = {"X-API-Key": self.valid_key}
        response = self.client.post("/search", json={"query": "pedoman organisasi", "top_k": 3}, headers=headers)
        self.assertIn(response.status_code, [200, 503], "Harus mengembalikan HTTP 200 (atau 503 jika server DB offline)")

        if response.status_code == 200:
            res_data = response.json()
            self.assertEqual(res_data.get("status"), "success")
            self.assertIn("data", res_data)
            if len(res_data["data"]) > 0:
                first_item = res_data["data"][0]
                self.assertIn("uuid", first_item, "Response JSON harus memuat field 'uuid'")
                self.assertNotIn("id", first_item, "Response JSON TIDAK boleh memuat field 'id' ganda")
                self.assertNotIn("download_url", first_item, "Response JSON TIDAK boleh memuat 'download_url'")

    def test_documents_list_authorized(self):
        """Memastikan endpoint /documents dapat diakses dengan API Key valid."""
        headers = {"X-API-Key": self.valid_key}
        response = self.client.get("/documents?limit=5&offset=0", headers=headers)
        self.assertIn(response.status_code, [200, 503])


# =====================================================================
# CUSTOM TEST RUNNER DENGAN LAPORAN VISUAL
# =====================================================================

def run_all_tests():
    print("\n" + "=" * 70)
    print(f"{CYAN}{BOLD}🔬 MEMULAI UNIT TESTING LENGKAP - SISTEM PENCARIAN SEMANTIK SERANAH{RESET}")
    print("=" * 70)
    print(f"🕒 Waktu Eksekusi: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Direktori Kerja: {ROOT_DIR}\n")

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    test_classes = [
        Test01_EnvironmentAndConfig,
        Test02_TextPreprocessingAndHelpers,
        Test03_EmbeddingModel,
        Test04_CrossEncoderReranker,
        Test05_ChromaDBConnection,
        Test06_SeranahSyncEngine,
        Test07_CacheAndRateLimiter,
        Test08_FastApiEndpointsAndSecurity,
    ]

    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time

    total = result.testsRun
    failed = len(result.failures)
    errors = len(result.errors)
    passed = total - failed - errors

    print("\n" + "=" * 70)
    print(f"{BOLD}📊 REKAPITULASI HASIL PENGUJIAN FITUR SISTEM:{RESET}")
    print("=" * 70)
    print(f"  • Total Test Case    : {total}")
    print(f"  • Berhasil ({GREEN}PASSED{RESET})   : {GREEN}{passed}{RESET}")
    print(f"  • Gagal ({RED}FAILED{RESET})      : {RED}{failed}{RESET}")
    print(f"  • Error ({YELLOW}ERRORS{RESET})      : {YELLOW}{errors}{RESET}")
    print(f"  • Waktu Pengujian    : {elapsed:.2f} detik")
    print("-" * 70)

    if result.wasSuccessful():
        print(f"{GREEN}{BOLD}🎉 SEMUA FITUR SISTEM BERJALAN DENGAN SEMPURNA & AMAN!{RESET}")
    else:
        print(f"{RED}{BOLD}⚠️ ADA PENGUJIAN YANG GAGAL / PERLU PENYESUAIAN. SILAKAN CEK DETAIL DI ATAS.{RESET}")
    print("=" * 70 + "\n")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
