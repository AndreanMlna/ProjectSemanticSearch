"""
eksperimen/view_database_gemma.py
================================
Antarmuka Streamlit Modern & Interaktif untuk Sistem SERANAH AI
(LangGraph, LangChain, dan Gemma 2 RAG) berbasis Desain Google Stitch.
"""

import os
import sys
import time
import requests
import pandas as pd
import streamlit as st
import chromadb
from dotenv import load_dotenv

# Muat konfigurasi dari file .env
load_dotenv()

# --- KONFIGURASI PATH & ENVIRONMENT VARIABLES ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION", "arsip_kampus_v2")
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", os.path.join(ROOT, "uploads"))
DEFAULT_GEMMA_API: str = os.getenv("GEMMA_API_URL", "http://localhost:8002")
API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "seranah_secret_key_2026")

# Deteksi host dan port ChromaDB otomatis
CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
if CHROMA_HOST == "chroma-server":
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
elif CHROMA_HOST in ("localhost", "127.0.0.1"):
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8001"))
else:
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="SERANAH AI - Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Impor helper penambahan/penghapusan data jika tersedia
try:
    from src.add_new_data import process_single_document, delete_document_by_id
except ImportError:
    process_single_document = None
    delete_document_by_id = None

# =========================================================================
# INJEKSI CUSTOM CSS (THEME: STITCH MODERN DARK & GLASSMORPHISM)
# =========================================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">

<style>
    /* Global Base */
    .stApp {
        background-color: #0B0F19 !important;
        color: #e4e1ed !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #F8FAFC !important;
    }

    /* Glassmorphism Panel */
    .glass-panel {
        background: rgba(22, 31, 48, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
    }

    .ai-glow-text {
        color: #c0c1ff !important;
        text-shadow: 0 0 16px rgba(192, 193, 255, 0.4) !important;
    }

    .gradient-text {
        background: linear-gradient(135deg, #c0c1ff 0%, #d0bcff 50%, #8083ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Top Badges */
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

    /* User Chat Bubble */
    .user-bubble {
        background: #1f1f27;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px 16px 2px 16px;
        padding: 14px 18px;
        color: #F8FAFC;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    /* AI Response Card */
    .ai-response-card {
        background: rgba(22, 31, 48, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #8083ff;
        border-radius: 4px 16px 16px 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }

    /* Telemetry Grid */
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

    /* Source Citation Card */
    .source-card {
        background: rgba(22, 31, 48, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px;
        transition: all 0.2s ease;
        margin-bottom: 10px;
    }
    .source-card:hover {
        transform: translateY(-2px);
        border-color: rgba(192, 193, 255, 0.3);
        box-shadow: 0 4px 16px rgba(128, 131, 255, 0.15);
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #571bc1, #8083ff) !important;
    }
    
    /* Primary buttons */
    .stButton > button {
        background: linear-gradient(135deg, #571bc1 0%, #8083ff 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 16px rgba(128, 131, 255, 0.4) !important;
    }

    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #0d0d15 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# HELPER KONEKSI CHROMADB
# =========================================================================
def connect_to_chroma():
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        try:
            return client.get_collection(name=COLLECTION_NAME)
        except Exception:
            return client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as err:
        st.sidebar.error(f"❌ ChromaDB Offline (`{CHROMA_HOST}:{CHROMA_PORT}`): {err}")
        return None


collection = connect_to_chroma()
doc_count = collection.count() if collection is not None else 0

# =========================================================================
# SIDEBAR: ADMIN PROFILE & CONFIGURATION
# =========================================================================
with st.sidebar:
    # Profile / Admin Card
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
        value=DEFAULT_GEMMA_API.rstrip("/"),
        help="Port backend eksperimen.main_api_gemma adalah 8002."
    )

    api_rag_ask = f"{api_base_url}/rag/ask"
    api_rag_status = f"{api_base_url}/rag/status"
    api_search = f"{api_base_url}/search"

    # Backend Status Card
    st.markdown(f"""
    <div class="glass-panel" style="padding: 10px; margin: 10px 0; font-family: 'JetBrains Mono', monospace; font-size: 11px; line-height: 1.6;">
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #94A3B8;">URL</span>
            <span style="color: #c0c1ff;">{api_base_url}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #94A3B8;">Model</span>
            <span style="color: #d0bcff;">Gemma 2 (2B)</span>
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
                r = requests.get(api_rag_status, timeout=5)
                if r.status_code == 200:
                    st_data = r.json().get("components", {})
                    st.success("✅ Backend RAG Online!")
                    st.caption(f"Backend: `{st_data.get('active_backend', '-')}` | Ollama: `{'Aktif' if st_data.get('ollama') else 'Nonaktif'}`")
                else:
                    st.warning(f"Status HTTP {r.status_code}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.divider()

    # Upload Archive Form
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
# HEADER UTAMA: LOGO & STATUS BADGES (STITCH TOP APP BAR)
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
            <span>🧠 Gemma 2 (CUDA)</span>
        </div>
        <div class="top-badge">
            <span style="color: #F59E0B;">⚡ 48ms Latency</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

# =========================================================================
# WORKSPACE TABS
# =========================================================================
tab_rag, tab_search, tab_explore = st.tabs([
    "🤖 Tanya Jawab Cerdas (LangGraph)",
    "🔍 Semantic Search",
    "📊 Vektor Database"
])

# Inisialisasi Riwayat Chat
if "gemma_chat_history" not in st.session_state:
    st.session_state.gemma_chat_history = []

# -------------------------------------------------------------------------
# TAB 1: TANYA JAWAB CERDAS (LANGGRAPH + GEMMA 2)
# -------------------------------------------------------------------------
with tab_rag:
    # Render Riwayat Percakapan Sebelumnya
    for item in st.session_state.gemma_chat_history:
        # 1. User Message
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
            <div class="user-bubble" style="max-width: 80%;">
                {item['question']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Workflow Execution Badge
        retry_cnt = item.get("retry_count", 0)
        rewritten_q = item.get("rewritten_query")
        
        if retry_cnt > 0 and rewritten_q:
            badge_html = f"""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; margin-left: 8px;">
                <span class="top-badge" style="color: #F59E0B; border-color: rgba(245, 158, 11, 0.3);">
                    🔄 Self-Correction (Retry #{retry_cnt}): "{rewritten_q}"
                </span>
            </div>
            """
        else:
            badge_html = """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; margin-left: 8px;">
                <span class="top-badge" style="color: #10B981; border-color: rgba(16, 185, 129, 0.3);">
                    ✅ Relevan Terverifikasi
                </span>
                <span style="font-size: 11px; color: #94A3B8; font-family: 'JetBrains Mono';">Query Analysis ➔ Vector Retrieval ➔ Reranking</span>
            </div>
            """
        st.markdown(badge_html, unsafe_allow_html=True)

        # 3. AI Response Card
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

        # 4. Source Citations
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

    # Input Form (Bottom Floating Area Style)
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
            try:
                payload = {
                    "question": q_input.strip(),
                    "top_k": int(top_k_select),
                    "engine": eng_select
                }
                headers = {"X-API-Key": API_SECRET_KEY}
                res = requests.post(api_rag_ask, json=payload, headers=headers, timeout=180)

                if res.status_code == 200:
                    res_data = res.json().get("data", {})
                    tot_latency = time.time() - t_start

                    # Simpan ke riwayat chat
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
                else:
                    st.error(f"❌ Error API (Status {res.status_code}): {res.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Gagal terhubung ke API di `{api_rag_ask}`. Pastikan server `main_api_gemma` berjalan di port 8002.")
            except Exception as ex:
                st.error(f"❌ Terjadi kesalahan: {ex}")

# -------------------------------------------------------------------------
# TAB 2: SEMANTIC SEARCH
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
            try:
                t0 = time.time()
                resp = requests.post(api_search, json={"query": search_query, "top_k": s_topk}, headers={"X-API-Key": API_SECRET_KEY}, timeout=60)
                if resp.status_code == 200:
                    results = resp.json().get("data", [])
                    elapsed = time.time() - t0
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
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Gagal mencari dokumen: {e}")

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
