import streamlit as st
import chromadb
import pandas as pd
import os
import time
import sys
import requests
from dotenv import load_dotenv

# Muat konfigurasi dari file .env
load_dotenv()

# --- KONFIGURASI PATH & ENVIRONMENT VARIABLES ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION", "arsip_kampus_v2")
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", os.path.join(ROOT, "uploads"))
API_URL: str = os.getenv("API_URL", "http://localhost:8000/search")
API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "seranah_secret_key_2026")

# Deteksi host dan port otomatis (Docker Container vs Lokal Windows)
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
    page_title="Sistem Manajemen & Semantic Search Arsip",
    page_icon="📚",
    layout="wide"
)

# Impor fungsi helper penambahan/penghapusan data
try:
    from src.add_new_data import process_single_document, delete_document_by_id
except ImportError as e:
    process_single_document = None
    delete_document_by_id = None
    st.sidebar.error(f"⚠️ Modul 'add_new_data' gagal dimuat: {e}")

# =========================================================================
# SIDEBAR: PENGATURAN & TAMBAH ARSIP
# =========================================================================
with st.sidebar:
    st.header("⚙️ Konfigurasi Sistem")
    st.info(f"**Target Host:** `{CHROMA_HOST}:{CHROMA_PORT}`\n\n**Koleksi:** `{COLLECTION_NAME}`\n\n**API Search:** `{API_URL}`")
    if st.button("🔄 Segarkan Data (Refresh)", use_container_width=True):
        st.rerun()
    st.divider()

    st.header("➕ Tambah Arsip Baru")
    st.caption("Indeks dokumen PDF/Docx ke dalam database vektor ChromaDB.")

    with st.form("upload_form", clear_on_submit=True):
        new_title = st.text_input("Judul Dokumen", placeholder="Contoh: SK Rektor Pembimbing 2026")
        new_unit = st.text_input("Unit Kerja / Pengunggah", placeholder="Contoh: Sekretariat Universitas")
        new_desc = st.text_area("Deskripsi / Ringkasan Isi", placeholder="Deskripsi lengkap dokumen...")
        uploaded_file = st.file_uploader("Pilih File Lampiran", type=["pdf", "docx", "doc", "txt"])
        submitted = st.form_submit_button("🚀 Proses & Index ke Vektor", use_container_width=True)

        if submitted:
            if not new_title or not uploaded_file:
                st.warning("Mohon lengkapi Judul dan File terlebih dahulu.")
            else:
                with st.spinner("Mengekstraksi teks & membuat embedding vektor..."):
                    safe_filename = uploaded_file.name.replace(" ", "_")
                    save_path = os.path.join(UPLOAD_DIR, safe_filename)

                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    if process_single_document:
                        full_content = f"{new_desc}. Unit Kerja: {new_unit}" if new_unit else new_desc
                        success, message = process_single_document(
                            title=new_title,
                            content=full_content,
                            file_name=safe_filename,
                            file_path=save_path,
                        )
                        if success:
                            st.success(f"✅ '{new_title}' berhasil diindeks.")
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error(f"Gagal: {message}")
                    else:
                        st.error("Fungsi pemrosesan dokumen tidak aktif.")

# =========================================================================
# HALAMAN UTAMA: KONEKSI CHROMADB
# =========================================================================
st.title("📚 Sistem Manajemen Arsip & Semantic Search (SERANAH)")


def connect_to_chroma():
    """Membuka koneksi ke ChromaDB dengan fallback yang aman."""
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        try:
            return client.get_collection(name=COLLECTION_NAME)
        except Exception:
            return client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as err:
        st.error(f"❌ Gagal menghubungkan ke server ChromaDB di `{CHROMA_HOST}:{CHROMA_PORT}`: {err}")
        return None


collection = connect_to_chroma()

if collection is not None:
    try:
        doc_count = collection.count()
    except Exception as e:
        doc_count = 0
        st.warning(f"Gagal membaca jumlah dokumen: {e}")

    # Tampilan Metrik Dashboard
    m1, m2, m3 = st.columns(3)
    m1.metric("Status Database", "Aktif / Terhubung 🟢")
    m2.metric("Total Arsip Terindeks", f"{doc_count:,} Dokumen")
    m3.metric("Koleksi Target", COLLECTION_NAME)
    st.divider()

    if doc_count == 0:
        st.warning("Database saat ini masih kosong (0 dokumen). Silakan jalankan script indexing terlebih dahulu.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🔍 Semantic Search", "📊 Eksplorasi Data Lengkap", "🗑️ Hapus Data"])

    # -------------------------------------------------------------------------
    # TAB 1: SEMANTIC SEARCH
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("### 🔍 Pencarian Semantik Dokumen Arsip")
        st.markdown(
            "Cari informasi arsip menggunakan pertanyaan atau kalimat natural. "
            "Sistem akan menemukan dokumen yang relevan berdasarkan makna semantik dan bobot reranking."
        )

        with st.container(border=True):
            query = st.text_input(
                "Masukkan Kata Kunci atau Pertanyaan:",
                placeholder="Contoh: Bagaimana aturan penyusunan Renstra satker UNIDA?",
                label_visibility="collapsed",
            )
            col_btn, col_topk = st.columns([1, 1])
            with col_btn:
                btn_search = st.button("Cari Dokumen ✨", type="primary", use_container_width=True)
            with col_topk:
                top_k_val = st.slider("Jumlah Hasil (Top-K):", min_value=1, max_value=20, value=5)

        if btn_search:
            if not query.strip():
                st.warning("Kata kunci pencarian tidak boleh kosong.")
            else:
                with st.spinner("🧠 Memproses penelusuran semantik & hybrid reranking..."):
                    start_time = time.time()
                    try:
                        payload = {"query": query, "top_k": top_k_val}
                        headers = {"X-API-Key": API_SECRET_KEY}
                        response = requests.post(API_URL, json=payload, headers=headers, timeout=120)

                        if response.status_code == 200:
                            resp = response.json()
                            total_time = time.time() - start_time

                            if resp.get("status") != "success":
                                st.error("❌ Terjadi kesalahan pada Backend.")
                            else:
                                st.markdown("#### ⏱️ Metrik Inferensi")
                                st.metric("Total Waktu Request (End-to-End)", f"{total_time:.2f} detik")
                                st.divider()

                                sources = resp.get("data", [])
                                st.markdown(f"#### 📄 Referensi Arsip Ditemukan ({len(sources)} Dokumen Relevan)")

                                if not sources:
                                    st.warning("Tidak ada dokumen relevan yang cocok dengan kueri tersebut.")
                                else:
                                    for idx, source in enumerate(sources):
                                        title = source.get("title", "Tanpa Judul")
                                        score = source.get("score", 0.0)
                                        file_name = source.get("file_name", "-")
                                        doc_num = source.get("document_number", "-")
                                        year = source.get("year", "-")
                                        unit = source.get("unit_kerja", "-")
                                        category = source.get("category", "-")
                                        keywords = source.get("keywords", "-")
                                        access = source.get("access_level", "PUBLIC")

                                        content = source.get("document_asli", source.get("content_only", source.get("snippet", "")))

                                        with st.expander(f"🏆 Top {idx + 1} | {title} (Skor Relevansi: {score:.3f})"):
                                            col_info1, col_info2 = st.columns(2)
                                            with col_info1:
                                                st.markdown(f"**🏢 Unit Kerja:** `{unit}`")
                                                st.markdown(f"**📁 Kategori:** `{category}`")
                                                st.markdown(f"**🔢 No. Dokumen:** `{doc_num}`")
                                            with col_info2:
                                                st.markdown(f"**📅 Tahun:** `{year}`")
                                                st.markdown(f"**🔒 Akses:** `{access}`")
                                                st.markdown(f"**📂 Nama File:** `{file_name}`")

                                            if keywords and keywords != "-":
                                                st.markdown(f"**🏷️ Kata Kunci:** *{keywords}*")

                                            st.markdown("---")
                                            st.markdown(f"**📝 Isi / Deskripsi Dokumen:**")
                                            st.info(content)
                        else:
                            st.error(f"❌ Error dari server Backend (Status {response.status_code}): {response.text}")

                    except requests.exceptions.ConnectionError:
                        st.error(f"❌ Gagal terhubung ke API Backend di `{API_URL}`. Pastikan container 'skripsi_backend' sedang berjalan.")
                    except Exception as e:
                        st.error(f"❌ Terjadi kesalahan internal: {str(e)}")

    # -------------------------------------------------------------------------
    # TAB 2: EKSPLORASI DATA VEKTOR LENGKAP
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### 📊 Pratinjau Database Vektor & Metadata Arsip")
        st.caption("Menampilkan 15 data dokumen dan vektor embedding teratas yang tersimpan di ChromaDB.")

        try:
            results = collection.get(limit=15, include=["metadatas", "embeddings"])

            if results and results.get("ids"):
                data_list = []
                embeddings = results.get("embeddings")

                for i in range(len(results["ids"])):
                    meta = results["metadatas"][i] if results.get("metadatas") else {}
                    
                    vector_preview = "-"
                    if embeddings is not None and i < len(embeddings) and embeddings[i] is not None:
                        try:
                            vec = embeddings[i]
                            if len(vec) >= 3:
                                vector_preview = f"[{float(vec[0]):.4f}, {float(vec[1]):.4f}, {float(vec[2]):.4f}, ...]"
                            elif len(vec) > 0:
                                vector_preview = f"{[round(float(x), 4) for x in vec]}"
                        except Exception:
                            vector_preview = "[Vektor Tersimpan]"

                    data_list.append({
                        "UUID / ID": meta.get("uuid", results["ids"][i]),
                        "Judul Arsip": meta.get("title", "-"),
                        "Unit Kerja": meta.get("unit_kerja", "-"),
                        "Kategori": meta.get("category", "-"),
                        "Tahun": meta.get("year", "-"),
                        "No. Surat": meta.get("document_number", "-"),
                        "Nama File": meta.get("file_name", "-"),
                        "Kata Kunci": meta.get("keywords", "-"),
                        "Ringkasan Deskripsi": meta.get("snippet", meta.get("content", ""))[:100] + "...",
                        "Pratinjau Vektor": vector_preview,
                    })

                df = pd.DataFrame(data_list)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada data vektor yang ditemukan.")
        except Exception as e:
            st.error(f"Gagal memuat pratinjau data vektor: {e}")

    # -------------------------------------------------------------------------
    # TAB 3: HAPUS DATA
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("### 🗑️ Penghapusan Dokumen Permanen")
        st.error("Tindakan ini akan menghapus dokumen dari memori vektor ChromaDB secara permanen.")

        try:
            all_docs = collection.get(include=["metadatas"])
            if all_docs and all_docs.get("ids"):
                doc_options = {}
                for i, doc_id in enumerate(all_docs["ids"]):
                    meta = all_docs["metadatas"][i] if all_docs.get("metadatas") else {}
                    title = meta.get("title", "Tanpa Judul")
                    unit = meta.get("unit_kerja", "")
                    year = meta.get("year", "")

                    label_extra = f" [{unit} - {year}]" if unit else ""
                    label = f"{title}{label_extra} (ID: {doc_id})"
                    doc_options[label] = doc_id

                selected_label = st.selectbox("Pilih dokumen yang ingin dihapus:", options=list(doc_options.keys()))

                if st.button("⚠️ Hapus Dokumen Ini", type="primary"):
                    if delete_document_by_id:
                        target_id = doc_options[selected_label]
                        with st.spinner("Menghapus indeks vektor..."):
                            success, msg = delete_document_by_id(target_id)
                            if success:
                                st.success(f"Berhasil menghapus '{selected_label}'.")
                                time.sleep(1.5)
                                st.rerun()
                            else:
                                st.error(f"Gagal menghapus: {msg}")
                    else:
                        st.error("Fungsi penghapusan tidak tersedia (Cek modul add_new_data).")
            else:
                st.info("Tidak ada dokumen yang tersedia untuk dihapus.")
        except Exception as e:
            st.error(f"Gagal memuat daftar dokumen: {e}")
else:
    st.error("Gagal terhubung ke database vektor ChromaDB. Pastikan container 'skripsi_chroma' aktif.")