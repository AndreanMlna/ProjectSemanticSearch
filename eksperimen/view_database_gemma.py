"""
eksperimen/view_database_gemma.py
================================
Antarmuka Streamlit Generatif Modern untuk SERANAH AI
Mengadopsi Desain Conversational Google Gemini / ChatGPT (Google Stitch Dark Design System)
dengan Streamlit Native Chat Messages, Floating Bottom Input, & Responsive Layout.

Fitur:
1. Google Gemini Conversational Layout (Native st.chat_message, Clean Typography, Zero Overflow).
2. Arsitektur LangGraph Corrective RAG (CRAG) + Bi-Encoder MiniLM + Cross-Encoder Reranker.
3. Dual-Mode Deployment: Streamlit Cloud Standalone In-Process & FastAPI External Backend.
4. Telemetri Real-time, Verifikasi Dokumen Sumber (PDF Citations), dan Auto-Sync ChromaDB.
"""

import os
import sys
import time
import requests
import pandas as pd
import streamlit as st
import torch
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Batasi penggunaan thread PyTorch di container cloud agar hemat RAM (<1GB)
torch.set_num_threads(1)

# Muat environment (.env)
load_dotenv()

# --- 1. KONFIGURASI PATH & ENVIRONMENT VARIABLES ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


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

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="SERANAH AI - Campus Intelligence",
    page_icon="✦",
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
    # pyrefly: ignore [missing-import]
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def get_cached_langgraph_agent():
    """Memuat instance LangGraph Agent RAG sekali ke memori cache."""
    from eksperimen.rag_agent_langgraph_gemma import get_langgraph_gemma_agent
    return get_langgraph_gemma_agent()


# =========================================================================
# 3. INJEKSI CUSTOM CSS (THEME: GOOGLE GEMINI DARK + STITCH DESIGN SYSTEM)
# =========================================================================
def inject_gemini_theme_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        /* Base Background & Typography */
        .stApp {
            background-color: #131314 !important;
            color: #e3e3e3 !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Container Centering for ChatGPT / Gemini Look */
        .main .block-container {
            max-width: 820px !important;
            padding-top: 1.5rem !important;
            padding-bottom: 5.5rem !important;
            margin: 0 auto !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: #f1f3f4 !important;
            font-weight: 600;
        }
        
        /* Sidebar Styling (Gemini Nav) */
        [data-testid="stSidebar"] {
            background-color: #1e1f20 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
        }
        
        /* Gemini Sparkle Gradient */
        .gemini-gradient {
            background: linear-gradient(135deg, #7da5ff 0%, #c0a9ff 50%, #f4a261 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }
        
        .sparkle-icon {
            display: inline-block;
            background: linear-gradient(135deg, #4285F4 0%, #9B72CB 50%, #D96570 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 22px;
            font-weight: bold;
            margin-right: 6px;
        }
        
        /* Native Streamlit Chat Message Customization */
        div[data-testid="stChatMessage"] {
            padding: 12px 16px !important;
            border-radius: 16px !important;
            margin-bottom: 16px !important;
            background-color: transparent !important;
        }
        
        /* User Chat Bubble (Right Aligned / Card) */
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
            background-color: #282a2c !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 20px 20px 4px 20px !important;
            margin-left: auto !important;
            max-width: 85% !important;
        }
        
        /* AI Assistant Avatar */
        div[data-testid="stChatMessageAvatarAssistant"] {
            background: radial-gradient(circle at 30% 30%, #4285F4, #9B72CB, #1e1f20) !important;
            color: #ffffff !important;
            border-radius: 50% !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
        }
        
        /* Hero Welcome Screen (Gemini Empty State) */
        .hero-welcome-container {
            text-align: left;
            padding: 30px 4px 20px 4px;
            max-width: 100%;
        }
        
        .hero-title {
            font-size: 36px;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 8px;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        .hero-subtitle {
            font-size: 20px;
            color: #8e918f;
            font-weight: 500;
            margin-bottom: 24px;
        }
        
        /* Source Citation Card in Expander */
        .source-card-gemini {
            background: #1e1f20;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 8px;
        }
        
        /* Telemetry & Action Footer */
        .telemetry-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 8px;
            font-size: 11.5px;
            color: #8e918f;
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Disclaimer Bottom Note */
        .gemini-disclaimer {
            text-align: center;
            font-size: 11.5px;
            color: #8e918f;
            margin-top: 14px;
            margin-bottom: 8px;
        }
        
        /* Custom Modern Button */
        .stButton > button {
            background: #282a2c !important;
            color: #e3e3e3 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            padding: 12px 16px !important;
            text-align: left !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            background: #37393b !important;
            border-color: #a8c7fa !important;
            color: #ffffff !important;
        }
        
        /* Streamlit Native Chat Input Styling */
        [data-testid="stChatInput"] {
            border-radius: 28px !important;
            background-color: #1e1f20 !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: #7da5ff !important;
            box-shadow: 0 0 10px rgba(125, 165, 255, 0.2) !important;
        }
    </style>
    """, unsafe_allow_html=True)


inject_gemini_theme_css()

# =========================================================================
# 4. INISIALISASI & SINKRONISASI DATABASE VEKTOR CHROMADB
# =========================================================================
from src.chroma_client import get_collection

collection = get_collection()
doc_count = collection.count() if collection is not None else 0

# Inisialisasi otomatis satu kali jika database kosong
if "seranah_synced" not in st.session_state:
    st.session_state["seranah_synced"] = True
    if collection is not None and doc_count == 0:
        with st.spinner("⏳ Menginisialisasi dataset arsip kampus ke database vektor..."):
            try:
                from src.sync_seranah_archives import sync_seranah_to_chromadb
                _embed_init = load_cached_embedding_model(MODEL_PATH)
                _ok, _msg, _cnt = sync_seranah_to_chromadb(_embed_init, collection)
                if _ok:
                    doc_count = _cnt
            except Exception:
                pass

# Inisialisasi Session State Riwayat Chat & Pengguna
if "gemma_chat_history" not in st.session_state:
    st.session_state.gemma_chat_history = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "active_mode" not in st.session_state:
    st.session_state.active_mode = "chat"

if "user_name" not in st.session_state:
    st.session_state.user_name = "Civitas Akademika"

if "user_role" not in st.session_state:
    st.session_state.user_role = "Pengguna Kampus"


# =========================================================================
# 5. SIDEBAR: GEMINI STYLE NAVIGATION & SYSTEM ADMIN
# =========================================================================
with st.sidebar:
    # Logo & Brand Header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px; padding: 4px 8px;">
        <span class="sparkle-icon">✦</span>
        <span style="font-size: 20px; font-weight: 700; color: #f1f3f4; letter-spacing: -0.02em;">SERANAH AI</span>
        <span style="font-size: 10px; background: rgba(168, 199, 250, 0.15); color: #a8c7fa; padding: 2px 6px; border-radius: 12px; font-family: 'JetBrains Mono';">CRAG 2.0</span>
    </div>
    """, unsafe_allow_html=True)

    # Tombol Percakapan Baru (+ New Chat)
    if st.button("➕ Percakapan Baru", use_container_width=True):
        st.session_state.gemma_chat_history = []
        st.session_state.pending_query = None
        st.rerun()

    st.markdown("<div style='margin: 12px 0;'></div>", unsafe_allow_html=True)

    # Menu Navigasi Antarmuka
    nav_option = st.radio(
        "Navigasi:",
        options=["💬 Chatbot Kampus Arsip Digital Universitas Darussalam Gontor", "🔍 Semantic Search Explorer", "📊 Database Vektor"],
        index=0 if st.session_state.active_mode == "chat" else (1 if st.session_state.active_mode == "search" else 2),
        label_visibility="collapsed"
    )

    if "Chatbot" in nav_option:
        st.session_state.active_mode = "chat"
    elif "Semantic Search" in nav_option:
        st.session_state.active_mode = "search"
    else:
        st.session_state.active_mode = "explore"

    st.divider()

    # Riwayat Topik Terbaru (Gemini Recent Chats List)
    st.markdown("<div style='font-size: 12px; font-weight: 600; color: #8e918f; margin-bottom: 8px;'>Terbaru</div>", unsafe_allow_html=True)
    if st.session_state.gemma_chat_history:
        for idx, hist in enumerate(reversed(st.session_state.gemma_chat_history[-5:])):
            q_short = hist["question"][:28] + "..." if len(hist["question"]) > 28 else hist["question"]
            st.markdown(f"""
            <div style="font-size: 12.5px; color: #c4c7c5; padding: 6px 10px; border-radius: 8px; margin-bottom: 4px; background: rgba(255,255,255,0.03); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                💬 {q_short}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("Belum ada riwayat percakapan.")

    st.divider()

    # Profil & Peran Pengguna (Dapat Disesuaikan Siapa Saja)
    with st.expander("👤 Profil & Peran Pengguna", expanded=False):
        u_name = st.text_input("Nama / Panggilan:", value=st.session_state.user_name)
        roles_list = ["Civitas Akademika", "Mahasiswa", "Dosen / Peneliti", "Tenaga Kependidikan (Tendik)", "Tamu / Publik", "System Administrator"]
        r_idx = roles_list.index(st.session_state.user_role) if st.session_state.user_role in roles_list else 0
        u_role = st.selectbox("Peran di Kampus:", options=roles_list, index=r_idx)
        if u_name != st.session_state.user_name or u_role != st.session_state.user_role:
            st.session_state.user_name = u_name
            st.session_state.user_role = u_role

    # System Status Expander
    with st.expander("⚙️ System Telemetri & Tools", expanded=False):
        is_cloud = hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets
        mode_text = "Cloud In-Process" if is_cloud else "Standalone Engine"
        doc_count_display = collection.count() if collection is not None else 1030

        st.markdown(f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94A3B8; line-height: 1.8;">
            <div>• Engine: <span style="color: #10B981;">{mode_text}</span></div>
            <div>• Vector DB: <span style="color: #a8c7fa;">ChromaDB ({doc_count_display:,} Docs)</span></div>
            <div>• AI Model: <span style="color: #c0a9ff;">Qwen 3.8 27B (Groq LPU)</span></div>
            <div>• Reranker: <span style="color: #f4a261;">MiniLM Cross-Encoder</span></div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Sync Arsip Live API", use_container_width=True):
            with st.spinner("Sinkronisasi arsip kampus..."):
                try:
                    from src.sync_seranah_archives import sync_seranah_to_chromadb
                    _embed_model = load_cached_embedding_model(MODEL_PATH)
                    _ok, _msg, _cnt = sync_seranah_to_chromadb(_embed_model, collection)
                    if _ok:
                        st.success(f"✅ Berhasil sinkronisasi {_cnt} dokumen!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Gagal: {_msg}")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Upload Arsip Form
        st.markdown("<div style='margin-top: 12px; font-weight: 600; font-size: 12px;'>➕ Upload Arsip Baru</div>", unsafe_allow_html=True)
        with st.form("sidebar_upload_form", clear_on_submit=True):
            new_title = st.text_input("Judul Dokumen", placeholder="SK Rektor No. 12/2026")
            new_unit = st.text_input("Unit Kerja", placeholder="Sekretariat Universitas")
            new_desc = st.text_area("Deskripsi / Ringkasan", placeholder="Uraian isi...")
            uploaded_file = st.file_uploader("File PDF/DOCX", type=["pdf", "docx", "doc", "txt"])
            submitted = st.form_submit_button("🚀 Index ke Vektor", use_container_width=True)

            if submitted:
                if not new_title or not uploaded_file:
                    st.warning("Judul dan File wajib diisi.")
                else:
                    with st.spinner("Indexing arsip..."):
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

    # Profile User Footer Dinamis
    user_initials = "".join([w[0].upper() for w in st.session_state.user_name.split()[:2]]) if st.session_state.user_name else "CA"
    st.markdown(f"""
    <div style="margin-top: auto; padding-top: 20px; display: flex; align-items: center; gap: 10px;">
        <div style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #4285F4, #9B72CB); display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white;">
            {user_initials}
        </div>
        <div>
            <div style="font-weight: 600; font-size: 13px; color: #f1f3f4;">{st.session_state.user_name}</div>
            <div style="font-size: 11px; color: #8e918f;">{st.session_state.user_role}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# 6. KONTEN UTAMA: MODE CHATBOT (GEMINI CONVERSATIONAL STREAM)
# =========================================================================
if st.session_state.active_mode == "chat":

    # JIKA RIWAYAT KOSONG: TAMPILKAN GEMINI WELCOME HERO & SUGGESTION CHIPS
    if not st.session_state.gemma_chat_history:
        display_name = st.session_state.user_name if st.session_state.user_name else "Civitas Akademika"
        st.markdown(f"""
        <div class="hero-welcome-container">
            <div class="hero-title">
                <span class="gemini-gradient">Halo, {display_name}</span>
            </div>
            <div class="hero-subtitle">
                Ada yang bisa saya bantu temukan di arsip kampus hari ini?
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Kartu Saran Pertanyaan (Prompt Chips)
        chip_col1, chip_col2 = st.columns(2)
        
        with chip_col1:
            if st.button("📄 Pedoman Pengabdian Masyarakat\nBagaimana integrasi penelitian dan pengabdian dalam pembelajaran?", use_container_width=True):
                st.session_state.pending_query = "Bagaimana pedoman integrasi penelitian dan pengabdian masyarakat dalam pembelajaran?"
                st.rerun()
            
            st.markdown("<div style='margin: 6px 0;'></div>", unsafe_allow_html=True)
            
            if st.button("👥 SOP Pengumuman Staf Baru\nBagaimana prosedur dan pengarahan penugasan staf pengabdian?", use_container_width=True):
                st.session_state.pending_query = "Jelaskan SOP pengumuman dan pengarahan penugasan staf pengabdian baru"
                st.rerun()

        with chip_col2:
            if st.button("✍️ Ketentuan Proofreading Skripsi\nApa urgensi dan aturan program proofreading bahasa tugas akhir?", use_container_width=True):
                st.session_state.pending_query = "Apa alasan dan ketentuan program proofreading bahasa tugas akhir mahasiswa?"
                st.rerun()
                
            st.markdown("<div style='margin: 6px 0;'></div>", unsafe_allow_html=True)
            
            if st.button("🏛️ Struktur Organisasi & Rektorat\nSiapa pimpinan dan regulasi ketetapan rektorat universitas?", use_container_width=True):
                st.session_state.pending_query = "Siapa rektor dan apa saja regulasi ketetapan pimpinan universitas?"
                st.rerun()

        st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    else:
        # JIKA ADA RIWAYAT CHAT: RENDER STREAM CONVERSATION NATIVE (GEMINI / CHATGPT)
        for item in st.session_state.gemma_chat_history:
            # 1. Pesan Pengguna
            with st.chat_message("user", avatar="👤"):
                st.markdown(item["question"])

            # 2. Respons AI Assistant
            with st.chat_message("assistant", avatar="✨"):
                # Badge jika terjadi Self-Correction pada LangGraph
                if item.get("retry_count", 0) > 0 and item.get("rewritten_query"):
                    st.caption(f"🔄 *Self-Corrected Query:* `{item.get('rewritten_query')}`")

                # Isi Jawaban Markdown Bersih (Tanpa bug tag div)
                st.markdown(item["answer"])

                # Telemetry Bar Ringkas
                st.markdown(f"""
                <div class="telemetry-row">
                    <span>⚡ Latensi: {item.get('latency', 0.0):.2f}s</span>
                    <span>•</span>
                    <span>Qwen 3.8 27B (Groq LPU)</span>
                    <span>•</span>
                    <span>ChromaDB Vector</span>
                </div>
                """, unsafe_allow_html=True)

                # Expander Sumber Dokumen Terverifikasi
                sources = item.get("sources", [])
                if sources:
                    with st.expander(f"📚 Rincian {len(sources)} Dokumen Sumber (Source Citations)", expanded=False):
                        for idx, src in enumerate(sources, start=1):
                            score_pct = int(src.get("score", 0.0) * 100)
                            st.markdown(f"""
                            <div class="source-card-gemini">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                    <span style="color: #a8c7fa; font-weight: 600; font-size: 13px;">[{idx}] {src.get('title', 'Tanpa Judul')}</span>
                                    <span style="color: #10B981; font-family: 'JetBrains Mono'; font-size: 11px;">Relevansi: {score_pct}%</span>
                                </div>
                                <div style="font-size: 11px; color: #8e918f; margin-bottom: 6px;">
                                    No. Dokumen: <b>{src.get('document_number', '-')}</b> | Unit: <b>{src.get('unit_kerja', '-')}</b> | Tahun: <b>{src.get('year', '-')}</b>
                                </div>
                                <div style="font-size: 12px; color: #c4c7c5; background: rgba(0,0,0,0.3); padding: 8px 10px; border-radius: 8px; line-height: 1.5;">
                                    {src.get('snippet', '')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # 7. INPUT CHAT STREAMLIT (PINNED DI BAGIAN BAWAH SEPERTI GEMINI/CHATGPT)
    # ---------------------------------------------------------------------
    user_prompt = st.chat_input("Tanyakan apa saja tentang regulasi, pedoman, atau SK kampus...")

    # Cek apakah ada prompt yang dipicu dari Suggestion Chip
    if not user_prompt and st.session_state.pending_query:
        user_prompt = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_prompt and user_prompt.strip():
        with st.spinner("✦ SERANAH AI sedang menganalisis dokumen arsip..."):
            t_start = time.time()
            res_data = None

            # 1. Coba hubungi FastAPI backend eksternal jika aktif
            try:
                payload = {
                    "question": user_prompt.strip(),
                    "top_k": 10,
                    "engine": "langgraph"
                }
                headers = {"X-API-Key": API_SECRET_KEY}
                res = requests.post(f"{DEFAULT_GEMMA_API}/rag/ask", json=payload, headers=headers, timeout=120)
                if res.status_code == 200:
                    res_data = res.json().get("data", {})
            except Exception:
                pass

            # 2. Fallback otomatis ke In-Process LangGraph Agent (Streamlit Cloud)
            if not res_data:
                try:
                    agent = get_cached_langgraph_agent()
                    resp_obj = agent.answer(question=user_prompt.strip(), top_k=10)
                    res_data = resp_obj.to_dict()
                except Exception as ex_in:
                    st.error(f"❌ Terjadi kesalahan saat memproses jawaban: {ex_in}")

            if res_data:
                tot_latency = time.time() - t_start
                st.session_state.gemma_chat_history.append({
                    "question": user_prompt.strip(),
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

    # Footer Disclaimer (Gemini Style)
    st.markdown("""
    <div class="gemini-disclaimer">
        SERANAH AI dapat membuat kesalahan. Selalu periksa dan verifikasi dokumen arsip resmi kampus.
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# 8. KONTEN TAB 2: SEMANTIC SEARCH EXPLORER
# =========================================================================
elif st.session_state.active_mode == "search":
    st.markdown("### 🔍 Semantic Search & Hybrid Reranking Explorer")
    st.caption("Eksplorasi penelusuran dokumen arsip menggunakan model Bi-Encoder MiniLM dan Cross-Encoder Reranker.")

    with st.container():
        s_col1, s_col2 = st.columns([4, 1])
        with s_col1:
            search_query = st.text_input("Kata Kunci Pencarian:", placeholder="Ketik kata kunci arsip (misal: beasiswa mahasiswa berprestasi)...", label_visibility="collapsed")
        with s_col2:
            s_topk = st.number_input("Jumlah Hasil:", min_value=1, max_value=20, value=5, label_visibility="collapsed")
        
        s_btn = st.button("Cari Dokumen ✨", use_container_width=True)

    if s_btn and search_query.strip():
        with st.spinner("Mencari arsip relevan..."):
            results = []
            elapsed = 0.0

            # 1. Coba melalui API
            try:
                t0 = time.time()
                resp = requests.post(
                    f"{DEFAULT_GEMMA_API}/search",
                    json={"query": search_query, "top_k": s_topk},
                    headers={"X-API-Key": API_SECRET_KEY},
                    timeout=30
                )
                if resp.status_code == 200:
                    results = resp.json().get("data", [])
                    elapsed = time.time() - t0
            except Exception:
                pass

            # 2. Fallback In-Process
            if not results and collection is not None:
                try:
                    t0 = time.time()
                    _embed_model = load_cached_embedding_model(MODEL_PATH)
                    q_vector = _embed_model.encode(search_query).tolist()
                    candidate_count = max(20, int(s_topk) * 4)

                    raw_res = collection.query(
                        query_embeddings=[q_vector],
                        n_results=candidate_count,
                        include=["metadatas", "distances", "documents"]
                    )

                    from src.reranker import get_reranker
                    reranker = get_reranker()
                    if reranker and raw_res:
                        results = reranker.rerank(query=search_query, chroma_results=raw_res, top_k=int(s_topk))
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
                    <div class="source-card-gemini">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="background: rgba(168, 199, 250, 0.2); color: #a8c7fa; font-weight: 700; font-size: 11px; padding: 2px 8px; border-radius: 4px;">Top {i}</span>
                            <span style="color: #10B981; font-family: 'JetBrains Mono'; font-size: 12px; font-weight: 600;">Skor: {doc.get('score', 0.0):.4f} ({score_pct}%)</span>
                        </div>
                        <h4 style="margin: 4px 0 8px 0; color: #f1f3f4;">{doc.get('title', 'Tanpa Judul')}</h4>
                        <div style="font-size: 11px; color: #8e918f; margin-bottom: 8px;">
                            🏢 <b>Unit:</b> {doc.get('unit_kerja', '-')} | 📁 <b>Kategori:</b> {doc.get('category', '-')} | 📅 <b>Tahun:</b> {doc.get('year', '-')}
                        </div>
                        <div style="font-size: 12px; color: #c4c7c5; background: rgba(0,0,0,0.25); padding: 10px; border-radius: 8px;">
                            {doc.get('snippet', doc.get('content', ''))}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# =========================================================================
# 9. KONTEN TAB 3: VEKTOR DATABASE EXPLORER
# =========================================================================
elif st.session_state.active_mode == "explore":
    st.markdown("### 📊 Database Vektor & Metadata Arsip (ChromaDB)")
    st.caption("Pratinjau koleksi embedding dokumen yang tersimpan dalam database vektor.")

    if collection is not None:
        try:
            res = collection.get(limit=25, include=["metadatas"])
            if res and res.get("ids"):
                records = []
                for idx, d_id in enumerate(res["ids"]):
                    meta = res["metadatas"][idx] if res.get("metadatas") else {}
                    records.append({
                        "ID": d_id[:12] + "...",
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
