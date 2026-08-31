"""
eksperimen/view_database_gemma.py
================================
Antarmuka Streamlit Generatif Modern untuk SERANAH AI
Mengadopsi Desain Conversational Google Gemini (Google Stitch Dark Design System)
dengan Visualisasi 3D Ruang Vektor Dokumen (PCA + Plotly 3D) & Animasi Mikro Interaktif.

Fitur:
1. Google Gemini Conversational Canvas (Unified View, Clean Typography, Zero Overlap).
2. 🌌 Visualisasi 3D Interaktif Ruang Vektor Dokumen (3D Embedding Space Galaxy).
3. Arsitektur LangGraph Corrective RAG (CRAG) + Bi-Encoder MiniLM + Cross-Encoder Reranker.
4. Dual-Mode Deployment: Streamlit Cloud Standalone In-Process & FastAPI External Backend.
5. Telemetri Real-time, Verifikasi Dokumen Sumber (PDF Badges), dan Auto-Sync ChromaDB.
"""

import os
import sys
import time
import requests
import numpy as np
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
    page_icon="✨",
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
# 3. INJEKSI CUSTOM CSS (ANIMASI 3D & THEME GOOGLE GEMINI DARK)
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
        
        /* Unified Container Canvas */
        .main .block-container {
            max-width: 860px !important;
            padding-top: 1.8rem !important;
            padding-bottom: 9rem !important;
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
        
        /* Sidebar Radio Navigation styled as Gemini Tab List */
        [data-testid="stSidebar"] div[role="radiogroup"] > label {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px !important;
            padding: 10px 14px !important;
            margin-bottom: 8px !important;
            transition: all 0.25s ease !important;
            cursor: pointer !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background: rgba(255, 255, 255, 0.08) !important;
            border-color: rgba(168, 199, 250, 0.25) !important;
            transform: translateX(2px);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
            background: #282a2c !important;
            border-color: rgba(168, 199, 250, 0.4) !important;
        }
        
        /* 3D Pulse Glow Keyframes */
        @keyframes pulse3DGlow {
            0% { transform: scale(1); filter: drop-shadow(0 0 6px rgba(66, 133, 244, 0.4)); }
            50% { transform: scale(1.08); filter: drop-shadow(0 0 16px rgba(155, 114, 203, 0.7)); }
            100% { transform: scale(1); filter: drop-shadow(0 0 6px rgba(66, 133, 244, 0.4)); }
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
            animation: pulse3DGlow 3.5s infinite ease-in-out;
        }
        
        /* Custom Chromatic Shader Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #131314;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #4285F4 0%, #9B72CB 50%, #D96570 100%);
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(155, 114, 203, 0.4);
        }
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #7da5ff 0%, #c0a9ff 50%, #f4a261 100%);
            box-shadow: 0 0 16px rgba(125, 165, 255, 0.7);
        }

        /* Native Streamlit Chat Message Customization */
        div[data-testid="stChatMessage"] {
            padding: 8px 12px !important;
            border-radius: 16px !important;
            margin-bottom: 20px !important;
            background-color: transparent !important;
        }
        
        /* User Chat Bubble (Compact Pill Aligned Right) */
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
            background-color: #282a2c !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 20px 20px 4px 20px !important;
            margin-left: auto !important;
            max-width: 80% !important;
            padding: 10px 18px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
        }
        
        /* Assistant Response Card with Animated Shader Left Border & Subtle Ambient Glow */
        @keyframes responseShaderGlow {
            0% {
                border-left-color: #4285F4;
                box-shadow: -4px 0 14px rgba(66, 133, 244, 0.2);
            }
            50% {
                border-left-color: #9B72CB;
                box-shadow: -4px 0 20px rgba(155, 114, 203, 0.35);
            }
            100% {
                border-left-color: #D96570;
                box-shadow: -4px 0 14px rgba(217, 101, 112, 0.2);
            }
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
            background: linear-gradient(90deg, rgba(66, 133, 244, 0.04) 0%, transparent 100%) !important;
            border-left: 2px solid #4285F4 !important;
            border-radius: 4px 16px 16px 4px !important;
            padding-left: 12px !important;
            animation: responseShaderGlow 6s infinite alternate ease-in-out !important;
            transition: all 0.3s ease !important;
        }
        
        /* AI Assistant Avatar */
        div[data-testid="stChatMessageAvatarAssistant"] {
            background: radial-gradient(circle at 30% 30%, #4285F4, #9B72CB, #1e1f20) !important;
            color: #ffffff !important;
            border-radius: 50% !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            box-shadow: 0 0 10px rgba(155, 114, 203, 0.3) !important;
        }
        
        /* Hero Welcome Screen (Gemini Empty State) */
        .hero-welcome-container {
            text-align: left;
            padding: 30px 4px 20px 4px;
            max-width: 100%;
        }
        
        .hero-title {
            font-size: 38px;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 8px;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        .hero-subtitle {
            font-size: 22px;
            color: #8e918f;
            font-weight: 500;
            margin-bottom: 28px;
        }
        
        /* Source Citation Card with Shader Glow on Hover */
        @keyframes shaderCardBorder {
            0% { border-color: rgba(66, 133, 244, 0.3); }
            50% { border-color: rgba(155, 114, 203, 0.5); }
            100% { border-color: rgba(217, 101, 112, 0.3); }
        }

        .source-card-gemini {
            background: #1e1f20;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 8px;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .source-card-gemini:hover {
            transform: translateY(-2px);
            animation: shaderCardBorder 3s infinite alternate ease-in-out !important;
            box-shadow: 0 6px 22px rgba(66, 133, 244, 0.18) !important;
        }
        
        /* Telemetry & Action Footer */
        .telemetry-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 12px;
            padding-top: 8px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 11.5px;
            color: #8e918f;
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* 3D Glassmorphism Panel */
        .glass-panel-3d {
            background: rgba(30, 31, 32, 0.7) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 16px !important;
            padding: 16px !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        }

        /* ========================================================= */
        /* AMBIENT AI SHADER MESH EFFECT                             */
        /* ========================================================= */
        @keyframes shaderMeshFlow {
            0% {
                transform: scale(1) rotate(0deg);
                opacity: 0.45;
                filter: blur(60px);
            }
            50% {
                transform: scale(1.15) translate(20px, -15px) rotate(120deg);
                opacity: 0.7;
                filter: blur(75px);
            }
            100% {
                transform: scale(1) rotate(360deg);
                opacity: 0.45;
                filter: blur(60px);
            }
        }

        .shader-ambient-container {
            position: relative;
            padding: 24px 20px;
            margin-bottom: 24px;
            border-radius: 24px;
            background: rgba(30, 31, 32, 0.45);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            overflow: hidden;
        }

        .shader-orb-blue {
            position: absolute;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: radial-gradient(circle, #4285F4 0%, #7da5ff 40%, transparent 70%);
            top: -40px;
            left: -40px;
            animation: shaderMeshFlow 14s infinite ease-in-out;
            pointer-events: none;
            z-index: 0;
        }

        .shader-orb-purple {
            position: absolute;
            width: 240px;
            height: 240px;
            border-radius: 50%;
            background: radial-gradient(circle, #9B72CB 0%, #D96570 45%, transparent 70%);
            bottom: -30px;
            right: -30px;
            animation: shaderMeshFlow 18s infinite ease-in-out reverse;
            pointer-events: none;
            z-index: 0;
        }

        .shader-hero-content {
            position: relative;
            z-index: 1;
        }
        
        /* Sidebar Recent Chat Shader Hover */
        .sidebar-recent-chip {
            font-size: 12.5px;
            color: #c4c7c5;
            padding: 8px 12px;
            border-radius: 10px;
            margin-bottom: 6px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.04);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: all 0.25s ease;
        }
        .sidebar-recent-chip:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(168, 199, 250, 0.35) !important;
            transform: translateX(3px);
            box-shadow: 0 4px 14px rgba(66, 133, 244, 0.15);
        }

        /* Disclaimer Bottom Note */
        .gemini-disclaimer {
            text-align: center;
            font-size: 11.5px;
            color: #8e918f;
            margin-top: 8px;
            margin-bottom: 12px;
        }
        
        /* Interactive Shader Shimmer Buttons (Prompt Chips) */
        @keyframes shaderShimmerBtn {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .stButton > button {
            background: #282a2c !important;
            color: #e3e3e3 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            padding: 12px 16px !important;
            text-align: left !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #2d3035 0%, #342d3d 50%, #3a2e33 100%) !important;
            background-size: 200% 200% !important;
            animation: shaderShimmerBtn 4s infinite ease !important;
            border-color: rgba(168, 199, 250, 0.45) !important;
            color: #ffffff !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(66, 133, 244, 0.18) !important;
        }
        
        /* Streamlit Native Chat Input Styling with Shader Breathing Aura */
        @keyframes inputShaderAura {
            0% {
                box-shadow: 0 0 16px rgba(66, 133, 244, 0.25), 0 4px 20px rgba(0,0,0,0.4);
                border-color: rgba(125, 165, 255, 0.5);
            }
            50% {
                box-shadow: 0 0 24px rgba(155, 114, 203, 0.4), 0 6px 25px rgba(0,0,0,0.5);
                border-color: rgba(192, 169, 255, 0.7);
            }
            100% {
                box-shadow: 0 0 16px rgba(66, 133, 244, 0.25), 0 4px 20px rgba(0,0,0,0.4);
                border-color: rgba(125, 165, 255, 0.5);
            }
        }

        [data-testid="stChatInput"] {
            border-radius: 28px !important;
            background-color: #1e1f20 !important;
            border: 1px solid rgba(255, 255, 255, 0.14) !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stChatInput"]:focus-within {
            animation: inputShaderAura 4s infinite ease-in-out !important;
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
    # Logo & Brand Header dengan Animasi Glow
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

    # Menu Navigasi Antarmuka (Termasuk 🌌 3D Vektor Galaxy)
    nav_option = st.radio(
        "Navigasi:",
        options=[
            "💬 Tanya Arsip Kampus",
            "🔍 Pencarian Dokumen",
            "🌌 Visualisasi 3D & Vektor"
        ],
        index=0 if st.session_state.active_mode == "chat" else (1 if st.session_state.active_mode == "search" else 2),
        label_visibility="collapsed"
    )

    if "Tanya Arsip" in nav_option:
        st.session_state.active_mode = "chat"
    elif "Pencarian" in nav_option:
        st.session_state.active_mode = "search"
    else:
        st.session_state.active_mode = "explore_3d"

    st.divider()

    # Riwayat Topik Terbaru (Gemini Recent Chats List)
    st.markdown("<div style='font-size: 12px; font-weight: 600; color: #8e918f; margin-bottom: 8px;'>Terbaru</div>", unsafe_allow_html=True)
    if st.session_state.gemma_chat_history:
        for idx, hist in enumerate(reversed(st.session_state.gemma_chat_history[-5:])):
            q_short = hist["question"][:28] + "..." if len(hist["question"]) > 28 else hist["question"]
            st.markdown(f"""
            <div class="sidebar-recent-chip">
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
        <div class="shader-ambient-container">
            <div class="shader-orb-blue"></div>
            <div class="shader-orb-purple"></div>
            <div class="shader-hero-content">
                <div class="hero-title">
                    <span class="gemini-gradient">Halo, {display_name}</span>
                </div>
                <div class="hero-subtitle" style="margin-bottom: 0px;">
                    Ada yang bisa saya bantu temukan di arsip kampus hari ini?
                </div>
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
            # 1. Pesan Pengguna (Kanan)
            with st.chat_message("user", avatar="👤"):
                st.markdown(item["question"])

            # 2. Respons AI Assistant (Kiri - Aliran Percakapan Utuh)
            with st.chat_message("assistant", avatar="✨"):
                # Badge jika terjadi Self-Correction pada LangGraph
                if item.get("retry_count", 0) > 0 and item.get("rewritten_query"):
                    st.caption(f"🔄 *Self-Corrected Query:* `{item.get('rewritten_query')}`")

                # Isi Jawaban Markdown Bersih
                st.markdown(item["answer"])

                # Citation Pills (Inline Dokumen Pendukung)
                sources = item.get("sources", [])
                if sources:
                    top_sources = sources[:3]
                    pills_md = " ".join([f"`📄 {s.get('title', 'Dokumen')[:30]}`" for s in top_sources])
                    st.markdown(f"<div style='margin-top: 8px;'><b>Dokumen Terkait:</b> {pills_md}</div>", unsafe_allow_html=True)

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
        # 1. Tampilkan Pertanyaan Pengguna Secara Instan di Layar
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_prompt.strip())

        # 2. Tampilkan Respons AI dengan Efek Ketikan Streaming Real-time (Gemini / ChatGPT Style)
        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("✦ SERANAH AI sedang menganalisis dokumen arsip..."):
                t_start = time.time()
                res_data = None

                # Coba hubungi FastAPI backend eksternal jika aktif
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

                # Fallback otomatis ke In-Process LangGraph Agent (Streamlit Cloud)
                if not res_data:
                    try:
                        agent = get_cached_langgraph_agent()
                        resp_obj = agent.answer(question=user_prompt.strip(), top_k=10)
                        res_data = resp_obj.to_dict()
                    except Exception as ex_in:
                        st.error(f"❌ Terjadi kesalahan saat memproses jawaban: {ex_in}")

            if res_data:
                tot_latency = time.time() - t_start
                full_answer = res_data.get("answer", "")
                sources = res_data.get("sources", [])

                # Badge jika terjadi Self-Correction pada LangGraph
                if res_data.get("retry_count", 0) > 0 and res_data.get("rewritten_query"):
                    st.caption(f"🔄 *Self-Corrected Query:* `{res_data.get('rewritten_query')}`")

                # Generator Streaming Ketikan Kata-per-Kata (Typewriter Effect)
                def stream_typing_generator(text: str):
                    words = text.split(" ")
                    for i, w in enumerate(words):
                        yield w + (" " if i < len(words) - 1 else "")
                        time.sleep(0.015)  # 15ms per kata untuk efek animasi mengetik alami

                # Ketik jawaban secara langsung di layar
                st.write_stream(stream_typing_generator(full_answer))

                # Inline Citation Pills
                if sources:
                    top_sources = sources[:3]
                    pills_md = " ".join([f"`📄 {s.get('title', 'Dokumen')[:30]}`" for s in top_sources])
                    st.markdown(f"<div style='margin-top: 8px;'><b>Dokumen Terkait:</b> {pills_md}</div>", unsafe_allow_html=True)

                # Telemetry Bar Ringkas
                st.markdown(f"""
                <div class="telemetry-row">
                    <span>⚡ Latensi: {tot_latency:.2f}s</span>
                    <span>•</span>
                    <span>Qwen 3.8 27B (Groq LPU)</span>
                    <span>•</span>
                    <span>ChromaDB Vector</span>
                </div>
                """, unsafe_allow_html=True)

                # Expander Sumber Dokumen Terverifikasi
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

                # Simpan ke riwayat sesi chat
                st.session_state.gemma_chat_history.append({
                    "question": user_prompt.strip(),
                    "answer": full_answer,
                    "latency": tot_latency,
                    "search_time": res_data.get("search_time", 0.0),
                    "rerank_time": res_data.get("rerank_time", 0.0),
                    "llm_time": res_data.get("llm_time", 0.0),
                    "retry_count": res_data.get("retry_count", 0),
                    "rewritten_query": res_data.get("rewritten_query"),
                    "sources": sources
                })

    # Footer Disclaimer (Gemini Style)
    st.markdown("""
    <div class="gemini-disclaimer">
        SERANAH AI adalah asisten kecerdasan kampus. Selalu periksa dan verifikasi dokumen arsip resmi universitas.
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
# 9. KONTEN TAB 3: 🌌 VISUALISASI 3D RUANG VEKTOR (PCA 3D + PLOTLY)
# =========================================================================
elif st.session_state.active_mode == "explore_3d":
    st.markdown("### 🌌 Visualisasi 3D Galaksi Dokumen Arsip")
    st.caption("Eksplorasi ruang semantik embedding dokumen hasil fine-tuning MiniLM yang diproyeksikan ke ruang 3D.")

    if collection is not None and doc_count > 0:
        try:
            with st.spinner("Memuat ruang vektor 3D..."):
                sample_limit = min(200, doc_count)
                raw_data = collection.get(limit=sample_limit, include=["embeddings", "metadatas", "documents"])

                if raw_data and raw_data.get("embeddings") is not None and len(raw_data["embeddings"]) > 3:
                    embeddings_array = np.array(raw_data["embeddings"])
                    metadatas = raw_data.get("metadatas", [])
                    
                    # Proyeksi Dimensi 384D -> 3D menggunakan PCA
                    from sklearn.decomposition import PCA
                    pca_3d = PCA(n_components=3, random_state=42)
                    reduced_3d = pca_3d.fit_transform(embeddings_array)

                    titles = [m.get("title", f"Dokumen #{i}")[:45] for i, m in enumerate(metadatas)]
                    units = [m.get("unit_kerja", "Universitas") for m in metadatas]
                    categories = [m.get("category", "Arsip") for m in metadatas]
                    years = [str(m.get("year", "2024")) for m in metadatas]

                    # Buat DataFrame 3D
                    df_3d = pd.DataFrame({
                        "x": reduced_3d[:, 0],
                        "y": reduced_3d[:, 1],
                        "z": reduced_3d[:, 2],
                        "Judul": titles,
                        "Unit": units,
                        "Kategori": categories,
                        "Tahun": years
                    })

                    # Render 3D Galaxy menggunakan Plotly
                    import plotly.express as px
                    fig = px.scatter_3d(
                        df_3d,
                        x="x",
                        y="y",
                        z="z",
                        color="Unit",
                        hover_name="Judul",
                        hover_data={"Unit": True, "Kategori": True, "Tahun": True, "x": False, "y": False, "z": False},
                        opacity=0.85,
                        title=f"Ruang Semantik 3D ({len(df_3d)} Arsip Kampus)"
                    )

                    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='white')))
                    fig.update_layout(
                        paper_bgcolor="#131314",
                        plot_bgcolor="#131314",
                        font=dict(color="#e3e3e3", family="Plus Jakarta Sans"),
                        margin=dict(l=0, r=0, b=0, t=40),
                        scene=dict(
                            xaxis=dict(backgroundcolor="#131314", gridcolor="rgba(255,255,255,0.06)", showbackground=False, zerolinecolor="rgba(255,255,255,0.1)"),
                            yaxis=dict(backgroundcolor="#131314", gridcolor="rgba(255,255,255,0.06)", showbackground=False, zerolinecolor="rgba(255,255,255,0.1)"),
                            zaxis=dict(backgroundcolor="#131314", gridcolor="rgba(255,255,255,0.06)", showbackground=False, zerolinecolor="rgba(255,255,255,0.1)")
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    st.info("💡 **Tips Interaksi 3D:** Klik dan geser mouse untuk memutar galaksi 360°, scroll mouse untuk zoom, atau sorot titik bintang untuk melihat rincian dokumen.")

                else:
                    st.info("Embedding belum tersedia untuk visualisasi 3D.")

        except Exception as ex_3d:
            st.warning(f"Visualisasi 3D grafis disederhanakan: {ex_3d}")

        # Tabel Dokumen Vektor
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("#### 📋 Tabel Koleksi Vektor")
        try:
            res_tab = collection.get(limit=20, include=["metadatas"])
            if res_tab and res_tab.get("ids"):
                records = []
                for idx, d_id in enumerate(res_tab["ids"]):
                    meta = res_tab["metadatas"][idx] if res_tab.get("metadatas") else {}
                    records.append({
                        "ID": d_id[:12] + "...",
                        "Judul Dokumen": meta.get("title", "-"),
                        "Unit Kerja": meta.get("unit_kerja", "-"),
                        "Kategori": meta.get("category", "-"),
                        "Tahun": meta.get("year", "-")
                    })
                st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
        except Exception:
            pass

    else:
        st.error("ChromaDB tidak terhubung atau masih kosong.")
