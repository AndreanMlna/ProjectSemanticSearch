"""
eksperimen/view_database_gemma.py
================================
Antarmuka Streamlit Modern, Bersih, dan Terstruktur untuk SERANAH AI
(LangGraph, LangChain, Gemma 2, dan ChromaDB Vector Store) berbasis Desain Google Stitch.

Fitur:
1. Arsitektur Bersih & Modular (Tanpa Hardcode).
2. Dual-Mode Deployment: Berjalan mulus di Server Lokal (Docker/FastAPI) maupun
   di Streamlit Community Cloud (Standalone In-Process).
3. Caching Model Cerdas (@st.cache_resource) untuk efisiensi memori & kecepatan tinggi.
4. Telemetri Real-time, Riwayat Chat Interaktif, dan Auto-Sync Dokumen Arsip.
"""

import os
import sys
import time
import requests
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Muat konfigurasi environment (.env)
load_dotenv()

# --- 1. KONFIGURASI PATH & ENVIRONMENT VARIABLES ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Helper membaca konfigurasi aman (Mendukung st.secrets di Cloud & .env di Lokal)
def get_config_val(key: str, default: str = "") -> str:
    """Mengambil konfigurasi dari st.secrets (Streamlit Cloud) atau os.getenv (.env lokal)."""
    if hasattr(st, "secrets") and key in st.secrets:
        return str(st.secrets[key])
    return os.getenv(key, default)

COLLECTION_NAME: str = get_config_val("CHROMA_COLLECTION", "arsip_kampus_v2")
UPLOAD_DIR: str = get_config_val("UPLOAD_DIR", os.path.join(ROOT_DIR, "uploads"))
DEFAULT_GEMMA_API: str = get_config_val("GEMMA_API_URL", "http://localhost:8002").rstrip("/")
API_SECRET_KEY: str = get_config_val("API_SECRET_KEY", "seranah_secret_key_2026")
MODEL_PATH: str = get_config_val("HF_MODEL_NAME", "andrerean/minilm-arsip-kampus-seranah")
CE_MODEL_PATH: str = get_config_val("CE_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
CHROMA_HOST: str = get_config_val("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(get_config_val("CHROMA_PORT", "8000" if CHROMA_HOST == "chroma-server" else "8001"))

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="SERANAH AI - Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Impor helper penambahan data jika tersedia
try:
    from src.add_new_data import process_single_document
except ImportError:
    process_single_document = None


# =========================================================================
# 2. CACHING RESOURCE MODEL (MENGHEMAT RAM DI STREAMLIT CLOUD)
# =========================================================================
@st.cache_resource(show_spinner=False)
def load_cached_embedding_model(model_name: str):
    """Memuat model SentenceTransformer sekali ke memori cache."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def get_cached_langgraph_agent():
    """Memuat instance LangGraph Agent RAG sekali ke memori cache."""
    from eksperimen.rag_agent_langgraph_gemma import get_langgraph_gemma_agent
    return get_langgraph_gemma_agent()


# =========================================================================
# 3. INJEKSI CUSTOM CSS (THEME: STITCH MODERN DARK & GLASSMORPHISM)
# =========================================================================
def inject_custom_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        .stApp {
            background-color: #0B0F19 !important;
            color: #e4e1ed !important;
            font-family: 'Inter', sans-serif !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: #F8FAFC !important;
        }
        .glass-panel {
            background: rgba(22, 31, 48, 0.75) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 16px !important;
        }
        .gradient-text {
            background: linear-gradient(135deg, #c0c1ff 0%, #d0bcff 50%, #8083ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .top-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(22, 31, 48, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 9999px;
            padding: 4px 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: #94A3B8;
        }
        .badge-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #10B981;
            box-shadow: 0 0 8px #10B981;
        }
        .user-bubble {
            background: #1f1f27;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px 16px 2px 16px;
            padding: 14px 18px;
            color: #F8FAFC;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .ai-response-card {
            background: rgba(22, 31, 48, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 4px solid #8083ff;
            border-radius: 4px 16px 16px 16px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }
        .telemetry-pill {
            background: #1b1b23;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 8px 12px;
            text-align: center;
        }
        .telemetry-label {
            font-size: 9px;
            text-transform: uppercase;
            color: #94A3B8;
            letter-spacing: 0.5px;
            font-family: 'JetBrains Mono', monospace;
        }
        .telemetry-val {
            font-size: 13px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            color: #c0c1ff;
        }
        .source-card {
            background: rgba(22, 31, 48, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 10px;
            transition: all 0.2s ease;
        }
        .source-card:hover {
            border-color: rgba(192, 193, 255, 0.3);
            box-shadow: 0 4px 16px rgba(128, 131, 255, 0.15);
        }
        .stButton > button {
            background: linear-gradient(135deg, #571bc1 0%, #8083ff 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #0d0d15 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }
    </style>
    """, unsafe_allow_html=True)


inject_custom_css()

# =========================================================================
# 4. INISIALISASI & SINKRONISASI DATABASE VEKTOR CHROMADB
# =========================================================================
from src.chroma_client import get_collection

collection = get_collection()
doc_count = collection.count() if collection is not None else 0

# Auto-Inisialisasi jika berjalan di Cloud dan database masih kosong (0 Docs)
if collection is not None and doc_count == 0:
    with st.spinner("⏳ Menyiapkan database vektor otomatis dari Live API SERANAH Kampus..."):
        try:
            from src.sync_seranah_archives import sync_seranah_to_chromadb
            _embed_init = load_cached_embedding_model(MODEL_PATH)
            _ok, _msg, _cnt = sync_seranah_to_chromadb(_embed_init, collection)
            if _ok:
                doc_count = _cnt
        except Exception:
            pass


# =========================================================================
# 5. SIDEBAR: PROFIL, PENGATURAN BACKEND, & UPLOAD ARSIP
# =========================================================================
with st.sidebar:
    st.markdown("""
    <div class="glass-panel" style="padding: 12px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px;">
        <div style="width: 36px; height: 36px; border-radius: 50%; background: #292932; display: flex; align-items: center; justify-content: center; font-size: 18px;">
            👤
        </div>
        <div>
            <div style="font-weight: 600; font-size: 13px; color: #F8FAFC;">System Admin</div>
            <div style="font-size: 11px; color: #94A3B8;">Deep Archive Access</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Backend Configuration")
    api_base_url = st.text_input(
        "Backend URL:",
        value=DEFAULT_GEMMA_API,
        help="Port backend lokal adalah 8002."
    ).rstrip("/")

    api_rag_ask = f"{api_base_url}/rag/ask"
    api_rag_status = f"{api_base_url}/rag/status"
    api_search = f"{api_base_url}/search"

    st.markdown(f"""
    <div class="glass-panel" style="padding: 10px; margin: 10px 0; font-family: 'JetBrains Mono', monospace; font-size: 11px; line-height: 1.6;">
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #94A3B8;">Target URL</span>
            <span style="color: #c0c1ff;">{api_base_url}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #94A3B8;">AI Model</span>
            <span style="color: #d0bcff;">Gemma 2</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #94A3B8;">Engine</span>
            <span style="color: #ffb783;">LangGraph CRAG</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Cek Status RAG & AI", use_container_width=True):
        with st.spinner("Memeriksa status AI..."):
            try:
                r = requests.get(api_rag_status, timeout=4)
                if r.status_code == 200:
                    st_data = r.json().get("components", {})
                    st.success("✅ Backend RAG Online!")
                    st.caption(f"Backend: `{st_data.get('active_backend', '-')}` | Ollama: `{'Aktif' if st_data.get('ollama') else 'Nonaktif'}`")
                else:
                    st.warning(f"Status HTTP {r.status_code}")
            except Exception:
                st.info("ℹ️ Mode Standalone In-Process aktif (Menggunakan LLM Cloud & ChromaDB internal).")

    st.divider()

    st.markdown("### ➕ Upload Archive")
    with st.form("sidebar_upload_form", clear_on_submit=True):
        new_title = st.text_input("Judul Dokumen", placeholder="SK Rektor No. 12/2026")
        new_unit = st.text_input("Unit Kerja", placeholder="Sekretariat Universitas")
        new_desc = st.text_area("Deskripsi / Ringkasan", placeholder="Uraian isi surat...")
        uploaded_file = st.file_uploader("Lampiran File", type=["pdf", "docx", "doc", "txt"])
        submitted = st.form_submit_button("🚀 Index ke Vektor", use_container_width=True)

        if submitted:
            if not new_title or not uploaded_file:
                st.warning("Judul dan File wajib diisi.")
            else:
                with st.spinner("Mengekstrak teks & indexing..."):
                    safe_filename = uploaded_file.name.replace(" ", "_")
                    save_path = os.path.join(UPLOAD_DIR, safe_filename)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    if process_single_document:
                        full_content = f"{new_desc}. Unit Kerja: {new_unit}" if new_unit else new_desc
                        success, message = process_single_document(
                            title=new_title, content=full_content,
                            file_name=safe_filename, file_path=save_path,
                        )
                        if success:
                            st.success(f"✅ '{new_title}' berhasil diindeks.")
                            time.sleep(1.2)
                            st.rerun()
                        else:
                            st.error(f"Gagal: {message}")

    st.divider()
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.gemma_chat_history = []
        st.rerun()


# =========================================================================
# 6. HEADER UTAMA: LOGO & STATUS BADGES (STITCH TOP APP BAR)
# =========================================================================
h_col1, h_col2 = st.columns([1.5, 2.5])

with h_col1:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 26px; font-weight: 700; letter-spacing: -0.02em;" class="gradient-text">SERANAH AI</span>
        <span style="font-size: 11px; background: rgba(192, 193, 255, 0.15); color: #c0c1ff; padding: 2px 8px; border-radius: 6px; font-family: 'JetBrains Mono';">v2.0 CRAG</span>
    </div>
    """, unsafe_allow_html=True)

with h_col2:
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; gap: 8px; flex-wrap: wrap;">
        <div class="top-badge">
            <span class="badge-dot"></span>
            <span>ChromaDB: {doc_count:,} Docs</span>
        </div>
        <div class="top-badge">
            <span>🧠 Gemma 2 RAG</span>
        </div>
        <div class="top-badge">
            <span style="color: #10B981;">⚡ Hybrid Search</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

# Inisialisasi Session State Riwayat Chat
if "gemma_chat_history" not in st.session_state:
    st.session_state.gemma_chat_history = []

# =========================================================================
# 7. WORKSPACE TABS
# =========================================================================
tab_rag, tab_search, tab_explore = st.tabs([
    "🤖 Tanya Jawab Cerdas (LangGraph)",
    "🔍 Semantic Search",
    "📊 Vektor Database"
])


# -------------------------------------------------------------------------
# TAB 1: TANYA JAWAB CERDAS (LANGGRAPH CRAG + GEMMA 2)
# -------------------------------------------------------------------------
with tab_rag:
    # 1. Render Riwayat Chat
    for item in st.session_state.gemma_chat_history:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
            <div class="user-bubble" style="max-width: 80%;">
                {item['question']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        retry_cnt = item.get("retry_count", 0)
        rewritten_q = item.get("rewritten_query")
        
        if retry_cnt > 0 and rewritten_q:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; margin-left: 8px;">
                <span class="top-badge" style="color: #F59E0B; border-color: rgba(245, 158, 11, 0.3);">
                    🔄 Self-Correction (Retry #{retry_cnt}): "{rewritten_q}"
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; margin-left: 8px;">
                <span class="top-badge" style="color: #10B981; border-color: rgba(16, 185, 129, 0.3);">
                    ✅ Relevan Terverifikasi
                </span>
                <span style="font-size: 11px; color: #94A3B8; font-family: 'JetBrains Mono';">Query Analysis ➔ Vector Retrieval ➔ Reranking</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="ai-response-card">
            <div style="font-size: 15px; line-height: 1.7; color: #F8FAFC; margin-bottom: 16px;">
                {item['answer']}
            </div>
            <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 12px;">
                <div style="font-family: 'JetBrains Mono'; font-size: 11px; color: #94A3B8; margin-bottom: 8px;">TELEMETRY DATA</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px;">
                    <div class="telemetry-pill">
                        <div class="telemetry-label">Total Time</div>
                        <div class="telemetry-val" style="color: #c0c1ff;">{item.get('latency', 0.0):.2f}s</div>
                    </div>
                    <div class="telemetry-pill">
                        <div class="telemetry-label">Search GPU</div>
                        <div class="telemetry-val">{item.get('search_time', 0.0):.3f}s</div>
                    </div>
                    <div class="telemetry-pill">
                        <div class="telemetry-label">Reranking</div>
                        <div class="telemetry-val">{item.get('rerank_time', 0.0):.3f}s</div>
                    </div>
                    <div class="telemetry-pill">
                        <div class="telemetry-label">LLM Gemma</div>
                        <div class="telemetry-val" style="color: #d0bcff;">{item.get('llm_time', 0.0):.2f}s</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if item.get("sources"):
            with st.expander(f"📚 {len(item['sources'])} Dokumen Sumber (Source Citations)"):
                for idx, src in enumerate(item["sources"], start=1):
                    score_pct = int(src.get("score", 0.0) * 100)
                    st.markdown(f"""
                    <div class="source-card">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="background: rgba(128, 131, 255, 0.2); color: #c0c1ff; font-weight: 700; font-size: 11px; padding: 2px 6px; border-radius: 4px;">[{idx}]</span>
                            <span style="font-family: 'JetBrains Mono'; font-size: 10px; color: #94A3B8;">No: {src.get('document_number', '-')} | Thn: {src.get('year', '-')}</span>
                        </div>
                        <div style="font-weight: 600; font-size: 13px; color: #F8FAFC; margin-bottom: 4px;">{src.get('title', 'Tanpa Judul')}</div>
                        <div style="font-size: 11px; color: #94A3B8; margin-bottom: 8px;">Unit: {src.get('unit_kerja', '-')} | Kategori: {src.get('category', '-')}</div>
                        <div style="display: flex; justify-content: space-between; font-size: 10px; color: #c0c1ff; margin-bottom: 4px;">
                            <span>Relevance Match</span>
                            <span>{score_pct}%</span>
                        </div>
                        <div style="width: 100%; background: #292932; height: 4px; border-radius: 2px; overflow: hidden; margin-bottom: 8px;">
                            <div style="background: linear-gradient(90deg, #571bc1, #8083ff); height: 100%; width: {score_pct}%;"></div>
                        </div>
                        <div style="font-size: 11px; color: #cbd5e1; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 6px;">
                            {src.get('snippet', '')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # 2. Input Prompt Bar
    with st.container():
        st.markdown("<br>", unsafe_allow_html=True)
        col_inp, col_eng, col_top, col_sub = st.columns([4, 1.8, 1.2, 1])

        with col_inp:
            q_input = st.text_input(
                "Pertanyaan:",
                placeholder="Tanyakan regulasi, SK rektor, atau pedoman kampus...",
                label_visibility="collapsed",
                key="chat_q_input"
            )
        with col_eng:
            eng_select = st.selectbox(
                "Engine:",
                options=["langgraph", "native"],
                format_func=lambda x: "🕸️ LangGraph CRAG" if x == "langgraph" else "📜 Native Linear",
                label_visibility="collapsed"
            )
        with col_top:
            top_k_select = st.number_input("Top-K", min_value=1, max_value=20, value=10, label_visibility="collapsed")
        with col_sub:
            btn_send = st.button("Tanya AI 🚀", use_container_width=True)

    if btn_send and q_input.strip():
        with st.spinner("🤖 Gemma 2 & LangGraph sedang memproses..."):
            t_start = time.time()
            res_data = None

            # Coba panggil Backend API terlebih dahulu
            try:
                payload = {
                    "question": q_input.strip(),
                    "top_k": int(top_k_select),
                    "engine": eng_select
                }
                headers = {"X-API-Key": API_SECRET_KEY}
                res = requests.post(api_rag_ask, json=payload, headers=headers, timeout=120)
                if res.status_code == 200:
                    res_data = res.json().get("data", {})
            except Exception:
                pass

            # Fallback otomatis ke In-Process execution jika API tidak dapat dihubungi (Streamlit Cloud)
            if not res_data:
                try:
                    agent = get_cached_langgraph_agent()
                    resp_obj = agent.answer(question=q_input.strip(), top_k=int(top_k_select))
                    res_data = resp_obj.to_dict()
                except Exception as ex_in:
                    st.error(f"❌ Terjadi kesalahan saat memproses jawaban: {ex_in}")

            if res_data:
                tot_latency = time.time() - t_start
                st.session_state.gemma_chat_history.append({
                    "question": q_input.strip(),
                    "answer": res_data.get("answer", ""),
                    "latency": tot_latency,
                    "search_time": res_data.get("search_time", 0.0),
                    "rerank_time": res_data.get("rerank_time", 0.0),
                    "llm_time": res_data.get("llm_time", 0.0),
                    "retry_count": res_data.get("retry_count", 0),
                    "rewritten_query": res_data.get("rewritten_query"),
                    "sources": res_data.get("sources", [])
                })
                st.rerun()


# -------------------------------------------------------------------------
# TAB 2: SEMANTIC SEARCH & HYBRID RERANKING
# -------------------------------------------------------------------------
with tab_search:
    st.markdown("### 🔍 Semantic Search & Hybrid Reranking")
    st.caption("Pencarian cerdas berbasis Bi-Encoder MiniLM dan Cross-Encoder Reranker.")

    with st.container():
        s_col1, s_col2, s_col3 = st.columns([4, 1.5, 1])
        with s_col1:
            search_query = st.text_input("Kata Kunci Pencarian:", placeholder="Pedoman beasiswa mahasiswa berprestasi...", label_visibility="collapsed")
        with s_col2:
            s_topk = st.slider("Jumlah Dokumen:", min_value=1, max_value=20, value=5, label_visibility="collapsed")
        with s_col3:
            s_btn = st.button("Cari Dokumen ✨", use_container_width=True)

    if s_btn and search_query.strip():
        with st.spinner("Mencari arsip relevan..."):
            results = []
            elapsed = 0.0

            # 1. Coba melalui API
            try:
                t0 = time.time()
                resp = requests.post(
                    api_search,
                    json={"query": search_query, "top_k": s_topk},
                    headers={"X-API-Key": API_SECRET_KEY},
                    timeout=30
                )
                if resp.status_code == 200:
                    results = resp.json().get("data", [])
                    elapsed = time.time() - t0
            except Exception:
                pass

            # 2. Fallback In-Process jika API offline
            if not results and collection is not None:
                try:
                    t0 = time.time()
                    _embed_model = load_cached_embedding_model(MODEL_PATH)
                    q_vector = _embed_model.encode(search_query).tolist()
                    candidate_count = max(20, s_topk * 4)

                    raw_res = collection.query(
                        query_embeddings=[q_vector],
                        n_results=candidate_count,
                        include=["metadatas", "distances", "documents"]
                    )

                    from src.reranker import get_reranker
                    reranker = get_reranker()
                    if reranker and raw_res:
                        results = reranker.rerank(query=search_query, chroma_results=raw_res, top_k=s_topk)
                    else:
                        results = []
                    elapsed = time.time() - t0
                except Exception as ex_search:
                    st.error(f"Error saat mencari dokumen: {ex_search}")

            if results:
                st.success(f"Ditemukan {len(results)} dokumen relevan dalam {elapsed:.3f} detik.")
                for i, doc in enumerate(results, start=1):
                    score_pct = int(doc.get("score", 0.0) * 100)
                    st.markdown(f"""
                    <div class="source-card">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="background: rgba(128, 131, 255, 0.2); color: #c0c1ff; font-weight: 700; font-size: 11px; padding: 2px 8px; border-radius: 4px;">Top {i}</span>
                            <span style="color: #10B981; font-family: 'JetBrains Mono'; font-size: 12px; font-weight: 600;">Skor: {doc.get('score', 0.0):.4f} ({score_pct}%)</span>
                        </div>
                        <h4 style="margin: 4px 0 8px 0; color: #F8FAFC;">{doc.get('title', 'Tanpa Judul')}</h4>
                        <div style="font-size: 11px; color: #94A3B8; margin-bottom: 8px;">
                            🏢 <b>Unit:</b> {doc.get('unit_kerja', '-')} | 📁 <b>Kategori:</b> {doc.get('category', '-')} | 📅 <b>Tahun:</b> {doc.get('year', '-')}
                        </div>
                        <div style="font-size: 12px; color: #cbd5e1; background: rgba(0,0,0,0.25); padding: 10px; border-radius: 8px;">
                            {doc.get('snippet', doc.get('content', ''))}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# -------------------------------------------------------------------------
# TAB 3: VEKTOR DATABASE EXPLORER
# -------------------------------------------------------------------------
with tab_explore:
    st.markdown("### 📊 Database Vektor & Metadata Arsip (ChromaDB)")
    st.caption("Pratinjau 15 dokumen dan vektor embedding teratas.")

    if collection is not None:
        try:
            res = collection.get(limit=15, include=["metadatas", "embeddings"])
            if res and res.get("ids"):
                records = []
                for idx, d_id in enumerate(res["ids"]):
                    meta = res["metadatas"][idx] if res.get("metadatas") else {}
                    records.append({
                        "UUID / ID": meta.get("uuid", d_id),
                        "Judul Dokumen": meta.get("title", "-"),
                        "Unit Kerja": meta.get("unit_kerja", "-"),
                        "Kategori": meta.get("category", "-"),
                        "Tahun": meta.get("year", "-"),
                        "No. Dokumen": meta.get("document_number", "-"),
                    })
                st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
            else:
                st.info("Database vektor kosong.")
        except Exception as ex:
            st.error(f"Gagal membaca data vektor: {ex}")
    else:
        st.error("ChromaDB tidak terhubung.")
