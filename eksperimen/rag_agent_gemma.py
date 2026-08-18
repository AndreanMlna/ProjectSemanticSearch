import os
import time
import requests
import logging
import re
import yaml
from typing import Optional, Any
from dataclasses import dataclass, field

from src.document_reader import get_context_for_results
from src.reranker import get_reranker

# Inisialisasi logger khusus untuk Llama
logger = logging.getLogger("rag_agent")

LLM_BACKEND_MODE = "auto"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_MODEL = "gemma2:2b "

VLLM_BASE_URL = "http://localhost:8001/v1"
VLLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

TOP_K_FOR_CONTEXT = 10

# --- Cache Status Kesehatan vLLM ---
_VLLM_HEALTH_CACHE = {"alive": False, "checked_at": 0.0}
_VLLM_HEALTH_TTL_SECONDS = 30
_VLLM_HEALTH_TIMEOUT = 1.5

# --- Pengaturan Path ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(ROOT, "uploads")
CONFIG_PATH = os.path.join(ROOT, "config_gemma.yaml")

# --- Muat Konfigurasi dari YAML ---
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            config_data = yaml.safe_load(f)
            if config_data and "llm" in config_data:
                llm_cfg = config_data["llm"]
                LLM_BACKEND_MODE = llm_cfg.get("backend", LLM_BACKEND_MODE)

                if "ollama" in llm_cfg:
                    ollama_cfg = llm_cfg["ollama"]
                    OLLAMA_BASE_URL = ollama_cfg.get("base_url", OLLAMA_BASE_URL).rstrip('/')
                    OLLAMA_MODEL = ollama_cfg.get("model_name", OLLAMA_MODEL)
                    OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"

                if "vllm" in llm_cfg:
                    vllm_cfg = llm_cfg["vllm"]
                    VLLM_BASE_URL = vllm_cfg.get("base_url", VLLM_BASE_URL).rstrip('/')
                    VLLM_MODEL = vllm_cfg.get("model_name", VLLM_MODEL)

                if "ollama" not in llm_cfg and "vllm" not in llm_cfg and "base_url" in llm_cfg:
                    legacy_base_url = llm_cfg.get("base_url", OLLAMA_BASE_URL).rstrip('/')
                    OLLAMA_BASE_URL = legacy_base_url
                    OLLAMA_MODEL = llm_cfg.get("model_name", OLLAMA_MODEL)
                    OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"

                logger.info(
                    f"Config loaded -> Mode: {LLM_BACKEND_MODE} | "
                    f"Ollama: {OLLAMA_API_URL} ({OLLAMA_MODEL}) | "
                    f"vLLM: {VLLM_BASE_URL} ({VLLM_MODEL})"
                )
    except Exception as e:
        logger.error(f"Gagal memuat {CONFIG_PATH}: {e}")

# --- Override Melalui Environment Variable (Jika Diperlukan) ---
_env_backend = os.environ.get("RAG_LLM_BACKEND")
if _env_backend:
    LLM_BACKEND_MODE = _env_backend

_env_ollama_base = os.environ.get("OLLAMA_HOST") or os.environ.get("RAG_OLLAMA_BASE_URL")
if _env_ollama_base:
    OLLAMA_BASE_URL = _env_ollama_base.rstrip('/')
    OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"

_env_vllm_base = os.environ.get("RAG_VLLM_BASE_URL")
if _env_vllm_base:
    VLLM_BASE_URL = _env_vllm_base.rstrip('/')

if _env_backend or _env_ollama_base or _env_vllm_base:
    logger.info(f"LLM Config overridden by ENV -> Mode: {LLM_BACKEND_MODE}")

from dotenv import dotenv_values

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
env_path = os.path.join(project_root_dir, ".env")

env_vars = dotenv_values(env_path)

if "HF_TOKEN" in env_vars:
    os.environ["HF_TOKEN"] = env_vars["HF_TOKEN"]
    print("✅ HF_TOKEN berhasil dimuat dari file .env!")
else:
    print("❌ PERINGATAN: HF_TOKEN tidak ditemukan di dalam file .env!")


@dataclass
class SearchResult:
    doc_id: str
    title: str
    snippet: str
    score: float
    file_name: str
    download_url: str
    content_only: str = ""
    document_asli: str = ""  # [PERUBAHAN 1] Field ini menjamin teks utuh terbawa


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list = field(default_factory=list)
    search_results_count: int = 0
    context_chars_total: int = 0
    latency: float = 0.0
    error: Optional[str] = None

    # [PENAMBAHAN LATENCY] Menyimpan waktu tiap tahapan spesifik
    search_time: float = 0.0
    rerank_time: float = 0.0
    llm_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "search_results_count": self.search_results_count,
            "context_chars_total": self.context_chars_total,
            "latency": self.latency,
            "error": self.error,
            "search_time": self.search_time,
            "rerank_time": self.rerank_time,
            "llm_time": self.llm_time
        }


def clean_user_query(query: str) -> str:

    cleaned = re.sub(r'^(halo|hai|min|admin|permisi|maaf|selamat pagi|siang|sore|malam)\s*,?\s*', '', query, flags=re.IGNORECASE)

    pattern_prefix = r'\b(tolong carikan|bantu cari|tunjukkan|cari tentang|bisa carikan|tolong tampilkan|apakah ada|jelaskan|apa itu|bagaimana|carikan)\b\s*'
    cleaned = re.sub(pattern_prefix, '', cleaned, flags=re.IGNORECASE)

    pattern_location = r'\b(di\s+|pada\s+|lingkungan\s+|seputar\s+)(unida gontor|unida|universitas darussalam gontor|universitas darussalam|kampus)\b'
    cleaned = re.sub(pattern_location, '', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r'[!?,.]+$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # 5. Failsafe (Mencegah kueri kosong)
    return cleaned if cleaned else query.strip()


def search_tool(
        query: str,
        top_k: int = TOP_K_FOR_CONTEXT,
        embedding_model: Optional[Any] = None,
        chroma_collection: Optional[Any] = None,
        latency_metrics: Optional[dict] = None  # [PENAMBAHAN LATENCY]
) -> list[SearchResult]:
    if latency_metrics is None:
        latency_metrics = {}

    cleaned_query = clean_user_query(query)

    if embedding_model is not None and chroma_collection is not None:
        try:
            t_search_start = time.perf_counter()
            query_vector = embedding_model.encode(cleaned_query).tolist()
            candidate_count = max(20, top_k * 4)
            db_results = chroma_collection.query(
                query_embeddings=[query_vector],
                n_results=candidate_count,
                include=["metadatas", "distances", "documents"]
            )
            latency_metrics["search_time"] = time.perf_counter() - t_search_start

            full_docs_map = {}
            if 'ids' in db_results and 'documents' in db_results and db_results['documents']:
                for idx, d_id in enumerate(db_results['ids'][0]):
                    full_docs_map[d_id] = db_results['documents'][0][idx]

            results = []

            try:
                t_rerank_start = time.perf_counter()
                reranker = get_reranker()
                final_ranked = reranker.rerank(query=cleaned_query, chroma_results=db_results, top_k=top_k)
                latency_metrics["rerank_time"] = time.perf_counter() - t_rerank_start

                for doc in final_ranked:
                    is_dict = isinstance(doc, dict)
                    d_id = doc.get("id", "unknown") if is_dict else getattr(doc, "id", "unknown")
                    title = doc.get("title", "Tanpa Judul") if is_dict else getattr(doc, "title", "Tanpa Judul")
                    snippet = doc.get("snippet", "") if is_dict else getattr(doc, "snippet", "")
                    score = doc.get("score", 0.0) if is_dict else getattr(doc, "score", 0.0)
                    file_name = doc.get("file_name", "") if is_dict else getattr(doc, "file_name", "")

                    real_full_text = full_docs_map.get(d_id, "")
                    if not real_full_text:
                        real_full_text = doc.get("content_only", "") if is_dict else getattr(doc, "content_only", "")

                    results.append(SearchResult(
                        doc_id=d_id,
                        title=title,
                        snippet=snippet,
                        score=score,
                        file_name=file_name,
                        download_url=f"http://localhost:8000/files/{file_name}",
                        content_only=real_full_text,
                        document_asli=real_full_text
                    ))
                return results

            except Exception as rerank_err:
                logger.warning(f"Reranking gagal/fallback ke pencarian semantik standar: {rerank_err}")
                latency_metrics["rerank_time"] = 0.0  # Gagal rerank

                if db_results['metadatas'] and db_results['metadatas'][0]:
                    for i, meta in enumerate(db_results['metadatas'][0][:top_k]):
                        score = 1 - db_results['distances'][0][i]
                        doc_id = db_results['ids'][0][i] if 'ids' in db_results else f"doc_{i}"
                        fname = meta.get('file_name', '-')

                        real_full_text = full_docs_map.get(doc_id, meta.get('content_only', ''))

                        results.append(SearchResult(
                            doc_id=doc_id,
                            title=meta.get('title', 'Tanpa Judul'),
                            snippet=meta.get('snippet', ''),
                            score=round(score, 4),
                            file_name=fname,
                            download_url=f"http://localhost:8000/files/{fname}",
                            content_only=real_full_text,  # OVERRIDE yang terpotong
                            document_asli=real_full_text  # Kirim teks utuh aslinya
                        ))
                return results

        except Exception as injected_err:
            logger.error(f"Injected query error: {injected_err}. Mengalihkan ke HTTP API.")

    # --- Jalur Fallback HTTP API ---
    try:
        t_http_start = time.perf_counter()
        search_api_url = "http://localhost:8000/search"
        response = requests.post(
            search_api_url,
            json={"query": cleaned_query, "question": cleaned_query, "top_k": top_k},
            timeout=10
        )
        response.raise_for_status()
        latency_metrics["search_time"] = time.perf_counter() - t_http_start
        latency_metrics["rerank_time"] = 0.0  # HTTP API belum tentu pakai reranker eksternal

        data = response.json()

        results = []
        for item in data.get("data", []):
            results.append(SearchResult(
                doc_id=item.get("id", ""),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                content_only=item.get("content_only", ""),
                document_asli=item.get("content_only", ""),
                score=item.get("score", 0.0),
                file_name=item.get("file_name", ""),
                download_url=item.get("download_url", "")
            ))
        return results

    except Exception as http_err:
        logger.error(f"Search API error: {http_err}")
        return []


def build_prompt(question: str, enriched_docs: list[dict]) -> str:
    if not enriched_docs:
        return f"""Anda adalah asisten informasi akademik UNIDA Gontor yang sangat teliti.

Pertanyaan pengguna: {question}

Konteks arsip kosong.
Jawaban Langsung: Maaf, informasi spesifik mengenai hal tersebut belum ditemukan di dalam basis data arsip kami. Silakan hubungi unit terkait untuk informasi lebih lanjut.
"""

    context_parts = []
    # Metadata (Judul & Sumber File) disembunyikan dari LLM untuk mencegah prompt drift/kebocoran format
    for i, doc in enumerate(enriched_docs, 1):
        context_parts.append(
            f"--- KUTIPAN {i} ---\n"
            f"{doc.get('full_context', '').strip()}\n"
        )

    context_block = "\n".join(context_parts)

    return f"""Anda adalah asisten informasi akademik UNIDA Gontor yang sangat teliti.
Tugas Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan teks konteks yang disediakan.

[ATURAN KETAT]
1. Jawab HANYA menggunakan fakta yang secara eksplisit tertulis di dalam [KONTEKS ARSIP].
2. Jika teks konteks tidak mengandung informasi yang relevan untuk menjawab pertanyaan, Anda WAJIB menjawab persis dengan kalimat: "Maaf, informasi tersebut tidak ditemukan di dalam dokumen."
3. JANGAN mengarang informasi, angka, tanggal, atau nomor SK (Dilarang keras berhalusinasi).
4. JANGAN menggunakan frasa pengantar seperti "Berdasarkan dokumen referensi..." atau menyebutkan metadata dokumen. Langsung berikan inti jawabannya secara natural.
5. Jika konteks memuat prosedur, sebutkan langkah-langkahnya secara berurutan.
6. Jika konteks memuat nomor SK atau data spesifik, sebutkan secara eksplisit HANYA jika relevan dengan pertanyaan.

[KONTEKS ARSIP]
{context_block}

[PERTANYAAN PENGGUNA]
{question}

[JAWABAN LANGSUNG]:"""


def call_ollama(prompt: str, model: str = None) -> str:
    active_model = model or OLLAMA_MODEL
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": active_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 1024
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as req_err:
        logger.error(f"Ollama error: {req_err}")
        return f"❌ Error LLM: {str(req_err)}"


def call_vllm(prompt: str, model: str = None) -> str:
    active_model = model or VLLM_MODEL
    try:
        response = requests.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json={
                "model": active_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 1024,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as req_err:
        logger.error(f"vLLM error: {req_err}")
        return f"❌ Error LLM (vLLM): {str(req_err)}"


def _is_vllm_alive() -> bool:
    now = time.time()
    if (now - _VLLM_HEALTH_CACHE["checked_at"]) < _VLLM_HEALTH_TTL_SECONDS:
        return _VLLM_HEALTH_CACHE["alive"]

    try:
        r = requests.get(f"{VLLM_BASE_URL}/models", timeout=_VLLM_HEALTH_TIMEOUT)
        alive = r.status_code == 200
    except requests.exceptions.RequestException:
        alive = False

    _VLLM_HEALTH_CACHE["alive"] = alive
    _VLLM_HEALTH_CACHE["checked_at"] = now
    return alive


def call_llm(prompt: str) -> str:
    if LLM_BACKEND_MODE == "vllm":
        return call_vllm(prompt)
    if LLM_BACKEND_MODE == "ollama":
        return call_ollama(prompt)
    if _is_vllm_alive():
        return call_vllm(prompt)
    return call_ollama(prompt)


class RAGAgent:
    def __init__(self, model: str = None, top_k: int = TOP_K_FOR_CONTEXT, upload_dir: str = UPLOAD_DIR):
        self.model = model or OLLAMA_MODEL
        self.top_k = top_k
        self.upload_dir = upload_dir

    def answer(
            self,
            question: str,
            embedding_model: Optional[Any] = None,
            chroma_collection: Optional[Any] = None
    ) -> RAGResponse:
        start_time = time.perf_counter()  # Menggunakan perf_counter untuk akurasi tinggi
        cleaned_q = clean_user_query(question)

        # [PENAMBAHAN LATENCY] Dictionary penampung latensi komponen
        latency_metrics = {"search_time": 0.0, "rerank_time": 0.0}

        search_results = search_tool(
            query=question,
            top_k=self.top_k,
            embedding_model=embedding_model,
            chroma_collection=chroma_collection,
            latency_metrics=latency_metrics  # Meneruskan dictionary untuk diisi di dalam
        )

        if not search_results:
            prompt = build_prompt(question=question, enriched_docs=[])

            # --- TAHAP 3: MENGUKUR LATENSI INFERENSI LLM ---
            t_llm_start = time.perf_counter()
            answer_text = call_llm(prompt=prompt)
            llm_time = time.perf_counter() - t_llm_start

            latency = time.perf_counter() - start_time
            return RAGResponse(
                question=question,
                answer=answer_text,
                latency=latency,
                search_time=latency_metrics.get("search_time", 0.0),
                rerank_time=latency_metrics.get("rerank_time", 0.0),
                llm_time=llm_time
            )

        compat_results = []
        for r in search_results:
            compat_results.append({
                "id": r.doc_id,
                "doc_id": r.doc_id,
                "title": r.title,
                "snippet": r.snippet,
                "content": r.content_only,
                "content_only": r.content_only,
                "document_asli": r.document_asli,  # [PERUBAHAN 5] Loloskan ke document_reader.py
                "score": r.score,
                "file_name": r.file_name,
                "download_url": r.download_url
            })

        try:
            enriched_docs = get_context_for_results(
                search_results=compat_results,
                query=cleaned_q,
                upload_dir=self.upload_dir
            )

            if not enriched_docs or len(enriched_docs) == 0:
                enriched_docs = get_context_for_results(
                    search_results=search_results,
                    query=cleaned_q,
                    upload_dir=self.upload_dir
                )
        except Exception as reader_err:
            logger.error(f"Gagal memproses konteks: {str(reader_err)}")
            enriched_docs = []

        total_context_chars = sum(d.get("context_length", len(d.get("full_context", ""))) for d in enriched_docs)

        prompt = build_prompt(question=question, enriched_docs=enriched_docs)

        # --- TAHAP 3: MENGUKUR LATENSI INFERENSI LLM ---
        t_llm_start = time.perf_counter()
        answer_text = call_llm(prompt=prompt)
        llm_time = time.perf_counter() - t_llm_start

        sources = []
        for d in enriched_docs:
            sources.append({
                "title": d.get("title", "Tanpa Judul"),
                "score": d.get("score", 0.0),
                "file_name": d.get("file_name", ""),
                "download_url": d.get("download_url", ""),
                "context_chars": d.get("context_length", len(d.get("full_context", ""))),
                "full_context": d.get("full_context", "")
            })

        latency = time.perf_counter() - start_time
        logger.info(f"[BENCHMARK] Answer generated in {latency:.2f}s for query: '{cleaned_q}'")

        return RAGResponse(
            question=question,
            answer=answer_text,
            sources=sources,
            search_results_count=len(search_results),
            context_chars_total=total_context_chars,
            latency=latency,
            search_time=latency_metrics.get("search_time", 0.0),
            rerank_time=latency_metrics.get("rerank_time", 0.0),
            llm_time=llm_time
        )

    @staticmethod
    def is_ready() -> dict:
        status: dict[str, Any] = {
            "search_api": False, "ollama": False, "vllm": False,
            "active_backend": None, "model": None, "ready": False
        }

        try:
            r = requests.get("http://localhost:8000/health", timeout=3)
            status["search_api"] = r.status_code == 200
        except requests.exceptions.RequestException:
            pass

        try:
            r = requests.get(OLLAMA_API_URL.replace("/api/generate", "/api/tags"), timeout=3)
            status["ollama"] = r.status_code == 200
        except requests.exceptions.RequestException:
            pass

        try:
            r = requests.get(f"{VLLM_BASE_URL}/models", timeout=3)
            status["vllm"] = r.status_code == 200
        except requests.exceptions.RequestException:
            pass

        if LLM_BACKEND_MODE == "vllm":
            status["active_backend"], status["model"] = "vllm", VLLM_MODEL
            llm_ok = status["vllm"]
        elif LLM_BACKEND_MODE == "ollama":
            status["active_backend"], status["model"] = "ollama", OLLAMA_MODEL
            llm_ok = status["ollama"]
        else:
            if status["vllm"]:
                status["active_backend"], status["model"] = "vllm", VLLM_MODEL
            else:
                status["active_backend"], status["model"] = "ollama", OLLAMA_MODEL
            llm_ok = status["vllm"] or status["ollama"]

        status["ready"] = status["search_api"] and llm_ok
        return status


_agent_instance: Optional[RAGAgent] = None


def get_rag_agent() -> RAGAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = RAGAgent()
    return _agent_instance


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    agent = RAGAgent()
    current_status = agent.is_ready()

    print(f"\n📊 Status (Llama Edition):")
    print(f"  Search API      : {'✅' if current_status['search_api'] else '❌'}")
    print(f"  Ollama          : {'✅' if current_status['ollama'] else '❌'}")
    print(f"  vLLM (Docker)   : {'✅' if current_status['vllm'] else '❌'}")
    print(f"  Backend Aktif   : {current_status['active_backend']}")
    print(f"  Model           : {current_status['model']}")
    print(f"  Mode Konfigurasi: {LLM_BACKEND_MODE}")

    if not current_status["ready"]:
        print("\n⚠️ Pastikan API dan LLM Backend (Ollama/vLLM) berjalan.")
        sys.exit(1)

    while True:
        try:
            user_question = input("❓ Pertanyaan: ").strip()
            if user_question.lower() in ("exit", "quit", "keluar"):
                break
            if not user_question:
                continue

            resp = agent.answer(user_question)

            print(f"💬 Jawaban:\n{resp.answer}")
            print(f"\n📊 Statistik:")
            print(f"  Total Waktu (End-to-End) : {resp.latency:.2f} detik")
            print(f"    - Vector Search        : {resp.search_time:.2f} detik")
            print(f"    - Reranking            : {resp.rerank_time:.2f} detik")
            print(f"    - LLM Inference        : {resp.llm_time:.2f} detik")
            print(f"  Dokumen dibaca           : {resp.search_results_count}")
            print(f"  Total konteks            : {resp.context_chars_total} karakter")
            print(f"\n📚 Sumber:")
            for idx, source_item in enumerate(resp.sources, 1):
                print(f"  {idx}. {source_item['title']} (skor: {source_item['score']:.2f})")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            break