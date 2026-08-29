"""
eksperimen/rag_agent_langgraph_gemma.py
======================================
Implementasi Agentic Self-Corrective RAG (CRAG) berbasis LangGraph & LangChain
menggunakan Model Gemma 2 (gemma2:2b via Ollama / vLLM).

Alur Graph (Cyclic State Machine):
1. [clean_query] -> Menghapus noise & sapaan pengguna.
2. [retrieve]    -> Mengambil dokumen dari ChromaDB & Reranking Cross-Encoder.
3. [grade]       -> Menilai relevansi dokumen berbasis skor reranking.
4. [decision]    -> Jika relevan -> [generate]; jika tidak & retry < 1 -> [rewrite_query] -> [retrieve].
5. [generate]    -> Gemma 2 menghasilkan jawaban komprehensif berbasis konteks arsip.
"""

import os
import sys
import time
import re
import logging
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Setup Root Directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

load_dotenv()

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from sentence_transformers import SentenceTransformer
from src.chroma_client import get_collection
from src.reranker import get_reranker
from src.config import CE_MODEL_PATH, MODEL_PATH
from src.lifespan import get_ml_models
from src.logging_utils import setup_logging

logger = setup_logging("langgraph_gemma")

# Konfigurasi LLM Gemma 2
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("GEMMA_MODEL", "gemma2:2b")
TOP_K_DEFAULT = 10
SCORE_THRESHOLD = 0.25  # Ambang batas relevansi Cross-Encoder

_embedding_model_cache = None


def get_embedding_model():
    global _embedding_model_cache
    model = get_ml_models().get("minilm")
    if model is not None:
        return model
    if _embedding_model_cache is None:
        _embedding_model_cache = SentenceTransformer(MODEL_PATH)
    return _embedding_model_cache



@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list = field(default_factory=list)
    search_results_count: int = 0
    context_chars_total: int = 0
    latency: float = 0.0
    error: Optional[str] = None
    search_time: float = 0.0
    rerank_time: float = 0.0
    llm_time: float = 0.0
    retry_count: int = 0
    rewritten_query: Optional[str] = None

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
            "llm_time": self.llm_time,
            "retry_count": self.retry_count,
            "rewritten_query": self.rewritten_query,
        }


# ── 1. DEFINISI STATE GRAPH ─────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    cleaned_query: str
    documents: List[Dict[str, Any]]
    is_relevant: bool
    retry_count: int
    rewritten_query: Optional[str]
    answer: str
    search_time: float
    rerank_time: float
    llm_time: float


# ── 2. HELPER QUERY PREPROCESSING ──────────────────────────────────

def clean_query_text(query: str) -> str:
    """Membersihkan sapaan dan kata tanya pembuka dari kueri pengguna."""
    cleaned = re.sub(
        r'^(halo|hai|min|admin|permisi|maaf|selamat pagi|siang|sore|malam)\s*,?\s*',
        '', query, flags=re.IGNORECASE
    )
    pattern_prefix = (
        r'\b(tolong carikan|bantu cari|tunjukkan|cari tentang|bisa carikan|'
        r'tolong tampilkan|apakah ada|jelaskan|apa itu|bagaimana|carikan)\b\s*'
    )
    cleaned = re.sub(pattern_prefix, '', cleaned, flags=re.IGNORECASE)
    pattern_location = r'\b(di\s+|pada\s+|lingkungan\s+|seputar\s+)(unida gontor|unida|universitas darussalam gontor|universitas darussalam|kampus)\b'
    cleaned = re.sub(pattern_location, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'[!?,.]+$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else query.strip()


# ── 3. INISIALISASI LLM & RERANKER ─────────────────────────────────

def get_llm_instance():
    """
    Mengembalikan instance LLM (Hybrid Cloud / Local):
    1. Jika GROQ_API_KEY disetel di st.secrets / environment (.env),
       gunakan model Gemma 2 / Llama 3 via Groq Cloud LPU ultra cepat (~500 token/s).
    2. Jika tidak ada GROQ_API_KEY, gunakan backend Ollama lokal di laptop.
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                groq_key = str(st.secrets["GROQ_API_KEY"]).strip()
        except Exception:
            pass

    if groq_key:
        try:
            from langchain_groq import ChatGroq
            groq_model = os.getenv("GROQ_MODEL", "gemma2-9b-it")
            logger.info(f"[LLM] Menggunakan Groq Cloud LPU: model={groq_model}")
            return ChatGroq(
                model=groq_model,
                groq_api_key=groq_key,
                temperature=0.1,
            )
        except ImportError:
            logger.warning("[LLM] Package 'langchain-groq' belum terinstall. Menggunakan Ollama sebagai fallback.")

    logger.info(f"[LLM] Menggunakan Ollama Lokal: model={OLLAMA_MODEL}, url={OLLAMA_BASE_URL}")
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )




# ── 4. DEFINISI NODE-NODE LANGGRAPH ────────────────────────────────

def clean_query_node(state: AgentState) -> dict:
    """Node 1: Membersihkan kueri awal dari noise."""
    cleaned = clean_query_text(state["question"])
    return {
        "cleaned_query": cleaned,
        "retry_count": state.get("retry_count", 0),
        "search_time": 0.0,
        "rerank_time": 0.0,
        "llm_time": 0.0,
    }


def retrieve_node(state: AgentState) -> dict:
    """Node 2: Mengambil dokumen dari ChromaDB & melakukan reranking."""
    query = state.get("cleaned_query") or state["question"]
    col = get_collection()
    reranker = get_reranker(CE_MODEL_PATH)

    ranked_docs = []
    search_time = 0.0
    rerank_time = 0.0

    if col is not None:
        try:
            # 1. Vector Search ChromaDB menggunakan embedding lokal (GPU)
            t0 = time.perf_counter()
            embedder = get_embedding_model()
            query_vector = embedder.encode(query).tolist()
            candidate_count = max(20, TOP_K_DEFAULT * 4)
            db_results = col.query(
                query_embeddings=[query_vector],
                n_results=candidate_count,
                include=["metadatas", "documents", "distances"]
            )
            search_time = time.perf_counter() - t0

            # 2. Cross-Encoder Reranking
            t1 = time.perf_counter()
            ranked_docs = reranker.rerank(
                query=query,
                chroma_results=db_results,
                top_k=TOP_K_DEFAULT
            )
            rerank_time = time.perf_counter() - t1

        except Exception as e:
            logger.error(f"[LangGraph Retrieve Error] {e}")

    return {
        "documents": ranked_docs,
        "search_time": state.get("search_time", 0.0) + search_time,
        "rerank_time": state.get("rerank_time", 0.0) + rerank_time,
    }


def grade_relevance_node(state: AgentState) -> dict:
    """Node 3: Mengevaluasi apakah dokumen yang diambil memenuhi ambang relevansi."""
    docs = state.get("documents", [])
    if not docs:
        return {"is_relevant": False}

    top_doc = docs[0]
    top_score = top_doc.get("score", 0.0)
    is_rel = top_score >= SCORE_THRESHOLD

    logger.info(f"[LangGraph Grade] Top doc score: {top_score:.4f} -> Relevant: {is_rel}")
    return {"is_relevant": is_rel}


def rewrite_query_node(state: AgentState) -> dict:
    """Node 4 (Self-Correction): Gemma 2 menulis ulang kueri jika dokumen kurang relevan."""
    llm = get_llm_instance()
    old_query = state["question"]
    current_retry = state.get("retry_count", 0)

    prompt = (
        f"Kueri pencarian arsip kampus berikut belum menemukan dokumen relevan: '{old_query}'.\n"
        f"Tuliskan 1 kueri pencarian baru yang lebih formal, padat, dan spesifik untuk sistem arsip universitas.\n"
        f"Contoh: jika 'beasiswa unida', ubah jadi 'pedoman beasiswa mahasiswa berprestasi'.\n"
        f"HANYA berikan 1 kalimat kueri baru tanpa basa-basi atau tanda kutip:"
    )

    t0 = time.perf_counter()
    try:
        response = llm.invoke(prompt)
        new_q = clean_query_text(response.content.strip())
    except Exception as e:
        logger.warning(f"[LangGraph Rewrite Error] {e}")
        new_q = old_query

    llm_t = time.perf_counter() - t0
    logger.info(f"[LangGraph Rewrite] Iterasi #{current_retry + 1}: '{old_query}' -> '{new_q}'")

    return {
        "cleaned_query": new_q,
        "rewritten_query": new_q,
        "retry_count": current_retry + 1,
        "llm_time": state.get("llm_time", 0.0) + llm_t,
    }


def generate_answer_node(state: AgentState) -> dict:
    """Node 5: Gemma 2 menyusun jawaban berbasis dokumen arsip."""
    docs = state.get("documents", [])
    question = state["question"]
    llm = get_llm_instance()

    if not docs:
        return {
            "answer": "Maaf, tidak ditemukan dokumen arsip yang relevan dengan pertanyaan Anda di sistem SERANAH.",
            "llm_time": state.get("llm_time", 0.0),
        }

    # Susun konteks arsip
    context_blocks = []
    for i, d in enumerate(docs, 1):
        title = d.get("title", "Tanpa Judul")
        content = d.get("snippet") or d.get("content", "")
        doc_no = d.get("document_number", "-")
        year = d.get("year", "-")
        cat = d.get("category", "-")
        context_blocks.append(
            f"--- DOKUMEN [{i}] ---\n"
            f"Judul: {title}\n"
            f"Nomor/Tahun: {doc_no} / {year}\n"
            f"Kategori: {cat}\n"
            f"Isi: {content}"
        )

    context_text = "\n\n".join(context_blocks)

    prompt = (
        f"Anda adalah Asisten AI Cerdas Sistem Arsip Kampus Universitas Darussalam Gontor.\n"
        f"Tugas Anda: Jawab pertanyaan pengguna secara akurat, jelas, dan santun HANYA berdasarkan konteks dokumen arsip di bawah ini.\n"
        f"Sertakan nomor dokumen atau tahun jika relevan.\n\n"
        f"{context_text}\n\n"
        f"Pertanyaan: {question}\n"
        f"Jawaban:"
    )

    t0 = time.perf_counter()
    try:
        response = llm.invoke(prompt)
        ans_text = response.content.strip()
    except Exception as e:
        logger.error(f"[LangGraph Generate Error] {e}")
        ans_text = f"Terjadi kesalahan saat memproses jawaban AI: {e}"

    llm_t = time.perf_counter() - t0

    return {
        "answer": ans_text,
        "llm_time": state.get("llm_time", 0.0) + llm_t,
    }


# ── 5. MEMBANGUN STATEGRAPH ─────────────────────────────────────────

def build_langgraph_gemma():
    workflow = StateGraph(AgentState)

    workflow.add_node("clean_query", clean_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_relevance", grade_relevance_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.set_entry_point("clean_query")
    workflow.add_edge("clean_query", "retrieve")
    workflow.add_edge("retrieve", "grade_relevance")

    # Percabangan Logika (Conditional Edge): Ulangi jika belum relevan (max 1x retry)
    def route_decision(state: AgentState):
        if state.get("is_relevant", False):
            return "generate_answer"
        if state.get("retry_count", 0) < 1:
            return "rewrite_query"
        return "generate_answer"

    workflow.add_conditional_edges(
        "grade_relevance",
        route_decision,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query"
        }
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# ── 6. KELAS WRAPPER AGEN AGAR MUDAH DIPANGGIL DI FASTAPI ───────────

class LangGraphGemmaAgent:
    def __init__(self):
        self.app = build_langgraph_gemma()
        logger.info("✅ LangGraph Gemma Agent initialized successfully.")

    def answer(self, question: str, top_k: int = TOP_K_DEFAULT) -> RAGResponse:
        t_start = time.perf_counter()

        initial_state: AgentState = {
            "question": question,
            "cleaned_query": question,
            "documents": [],
            "is_relevant": False,
            "retry_count": 0,
            "rewritten_query": None,
            "answer": "",
            "search_time": 0.0,
            "rerank_time": 0.0,
            "llm_time": 0.0,
        }

        try:
            final_state = self.app.invoke(initial_state)
            latency = time.perf_counter() - t_start

            raw_docs = final_state.get("documents", [])
            sources = []
            total_chars = 0
            for d in raw_docs[:top_k]:
                cnt = d.get("snippet") or d.get("content", "")
                total_chars += len(cnt)
                sources.append({
                    "uuid": d.get("uuid", ""),
                    "title": d.get("title", "Tanpa Judul"),
                    "score": d.get("score", 0.0),
                    "document_number": d.get("document_number", "-"),
                    "year": d.get("year", "-"),
                    "category": d.get("category", "-"),
                    "file_name": d.get("file_name", "-"),
                    "snippet": cnt[:200],
                })

            return RAGResponse(
                question=question,
                answer=final_state.get("answer", ""),
                sources=sources,
                search_results_count=len(sources),
                context_chars_total=total_chars,
                latency=latency,
                search_time=final_state.get("search_time", 0.0),
                rerank_time=final_state.get("rerank_time", 0.0),
                llm_time=final_state.get("llm_time", 0.0),
                retry_count=final_state.get("retry_count", 0),
                rewritten_query=final_state.get("rewritten_query"),
            )

        except Exception as e:
            logger.exception("Error executing LangGraph Gemma agent")
            return RAGResponse(
                question=question,
                answer=f"Gagal memproses pertanyaan melalui AI Agent: {e}",
                latency=time.perf_counter() - t_start,
                error=str(e)
            )


_agent_singleton: Optional[LangGraphGemmaAgent] = None


def get_langgraph_gemma_agent() -> LangGraphGemmaAgent:
    global _agent_singleton
    if _agent_singleton is None:
        _agent_singleton = LangGraphGemmaAgent()
    return _agent_singleton


if __name__ == "__main__":
    agent = get_langgraph_gemma_agent()
    test_q = "Bagaimana pedoman organisasi kemahasiswaan di kampus?"
    print(f"\n[?] Pertanyaan: {test_q}\n" + "-" * 50)
    res = agent.answer(test_q)
    print(f"🤖 Jawaban Gemma 2:\n{res.answer}\n")
    print(f"⏱️ Total Latency : {res.latency:.2f}s (Search: {res.search_time:.2f}s, Rerank: {res.rerank_time:.2f}s, LLM: {res.llm_time:.2f}s)")
    print(f"🔄 Retry Count  : {res.retry_count}")
    if res.rewritten_query:
        print(f"✏️ Rewritten Q  : {res.rewritten_query}")
