import streamlit as st
import chromadb
import pandas as pd
import os
import time
import sys
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

DB_PATH = os.path.join(ROOT, "chroma_db_storage")
COLLECTION_NAME = "arsip_kampus_v2"
UPLOAD_DIR = os.path.join(ROOT, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Konfigurasi Halaman Streamlit (Harus dipanggil pertama)
st.set_page_config(page_title="Sistem Arsip & AI RAG", page_icon="📚", layout="wide")

try:
    from src.add_new_data import process_single_document, delete_document_by_id
except ImportError as e:
    process_single_document = None
    delete_document_by_id = None
    st.sidebar.error(f"⚠️ Modul 'add_new_data' gagal dimuat: {e}")

with st.sidebar:
    # BAGIAN 1: PEMILIHAN MODEL LLM
    st.header("⚙️ Konfigurasi Mesin RAG")

    selected_agent = st.selectbox(
        "Pilih Backend LLM aktif:",
        options=["Gemma", "Llama 3", "Qwen"],
        index=0
    )

    st.success(f"Terhubung dengan API Backend untuk agen {selected_agent}.")
    st.divider()

    # BAGIAN 2: UPLOAD DOKUMEN
    st.header("➕ Tambah Arsip Baru")
    st.caption("Indeks dokumen PDF/Docx ke dalam vektor ChromaDB.")

    with st.form("upload_form", clear_on_submit=True):
        new_title = st.text_input("Judul Dokumen", placeholder="Contoh: SK Rektor 2025")
        new_desc = st.text_area("Deskripsi / Metadata", placeholder="Deskripsi singkat...")
        uploaded_file = st.file_uploader("Pilih File", type=['pdf', 'docx', 'doc', 'txt'])
        submitted = st.form_submit_button("🚀 Proses & Index", use_container_width=True)

        if submitted:
            if not new_title or not uploaded_file:
                st.warning("Mohon lengkapi Judul dan File terlebih dahulu.")
            else:
                with st.spinner("Mengekstraksi teks & membuat embedding..."):
                    safe_filename = uploaded_file.name.replace(" ", "_")
                    save_path = os.path.join(UPLOAD_DIR, safe_filename)

                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    if process_single_document:
                        success, message = process_single_document(
                            title=new_title,
                            content=new_desc,
                            file_name=safe_filename,
                            file_path=save_path
                        )
                        if success:
                            st.success(f"✅ '{new_title}' berhasil diindeks.")
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error(f"Gagal: {message}")
                    else:
                        st.error("Fungsi pemrosesan tidak aktif.")

st.title("📚 Sistem Manajemen Arsip & AI Search")

try:
    # Koneksi ke DB vektor via HTTP Client (Arsitektur Docker)
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_collection(name=COLLECTION_NAME)
    doc_count = collection.count()

    # Tampilan Metrik Dashboard
    m1, m2, m3 = st.columns(3)
    m1.metric("Status Database", "Aktif / Terhubung")
    m2.metric("Total Arsip Terindeks", f"{doc_count} Dokumen")
    m3.metric("Arsitektur", "Microservices (API)")
    st.divider()

    if doc_count == 0:
        st.warning("Database saat ini kosong. Silakan gunakan panel kiri untuk menambahkan dokumen.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["AI Semantic Search (RAG)", "Eksplorasi Data", "Hapus Data"])

    # TAB 1: AI SEMANTIC SEARCH & RAG
    with tab1:
        st.markdown("###  Pencarian Semantik Terintegrasi LLM")
        st.markdown(
            "Tanyakan informasi seputar arsip menggunakan bahasa sehari-hari. "
            "Sistem akan mencari dokumen secara semantik dan menyintesis jawaban."
        )

        # Form Pencarian Profesional
        with st.container(border=True):
            query = st.text_input(
                "Masukkan Pertanyaan Anda:",
                placeholder="Contoh: Jelaskan aturan magang PKL secara umum!",
                label_visibility="collapsed"
            )
            col_btn, _ = st.columns([1, 4])
            with col_btn:
                btn_search = st.button("Cari Jawaban ✨", type="primary", use_container_width=True)

        if btn_search:
            if not query.strip():
                st.warning("Pertanyaan tidak boleh kosong.")
            else:
                with st.spinner(f"🧠 Meminta API Backend memproses jawaban..."):
                    start_time = time.time()
                    try:

                        API_URL = os.getenv("API_URL", "http://localhost:8000/rag/ask")

                        payload = {
                            "question": query,
                            "top_k": 5
                        }

                        # Request ke Backend FastAPI
                        response = requests.post(API_URL, json=payload, timeout=120)

                        if response.status_code == 200:
                            resp = response.json()
                            total_time = time.time() - start_time

                            if resp.get("error"):
                                st.error(f"❌ Terjadi kesalahan pada Backend: {resp['error']}")
                            else:
                                # Menampilkan Jawaban Utama
                                st.markdown("#### 💡 Jawaban Sistem")
                                st.info(resp.get("answer", "Tidak ada jawaban."), icon="🤖")

                                # Menampilkan Metrik Kinerja Sederhana
                                st.markdown("#### ⏱️ Metrik Inferensi")
                                st.metric("Total Waktu Request", f"{total_time:.2f} s")

                                st.divider()

                                # Menampilkan Referensi Dokumen dari Backend
                                sources = resp.get("sources", [])
                                st.markdown(f"#### 📄 Referensi Arsip Ditemukan ({len(sources)} Dokumen)")

                                if not sources:
                                    st.warning(
                                        "Tidak ada dokumen relevan yang memenuhi standar skor untuk disajikan sebagai referensi.")
                                else:
                                    for idx, source in enumerate(sources):
                                        title = source.get('title', 'Tanpa Judul')
                                        score = source.get('score', 0.0)
                                        file_name = source.get('file_name', '-')
                                        content = source.get('full_context', '')

                                        # Expander untuk tiap sumber
                                        with st.expander(f"Top {idx + 1} | {title} (Skor Relevansi: {score:.3f})"):
                                            st.caption(
                                                f"📂 Nama File: `{file_name}` | 📏 Panjang Teks: {len(content)} karakter")
                                            st.markdown(f"> {content}")
                        else:
                            st.error(f"❌ Error dari server Backend (Status {response.status_code}): {response.text}")

                    except requests.exceptions.ConnectionError:
                        st.error(
                            "❌ Gagal terhubung ke API Backend. Pastikan container 'api-skripsi' berjalan dan URL sudah benar.")
                    except Exception as e:
                        st.error(f"❌ Terjadi kesalahan internal Streamlit: {str(e)}")

    with tab2:
        st.markdown("###  Preview Data Vektor (10 Terbaru)")
        results = collection.get(limit=10, include=["metadatas", "embeddings"])

        if results['ids']:
            data_list = []
            for i in range(len(results['ids'])):
                meta = results['metadatas'][i] if results['metadatas'] else {}
                vector = results['embeddings'][i]
                vector_preview = str(vector[:3])[:-1] + ", ...]"

                data_list.append({
                    "Vektor ID": results['ids'][i],
                    "Judul": meta.get('title', '-'),
                    "Nama File": meta.get('file_name', '-'),
                    "Cuplikan (Snippet)": meta.get('snippet', '')[:100] + "...",
                    "Pratinjau Vektor": vector_preview
                })

            df = pd.DataFrame(data_list)
            st.dataframe(df, width='stretch', hide_index=True)

    # TAB 3: HAPUS DATA
    with tab3:
        st.markdown("###  Penghapusan Dokumen Permanen")
        st.error("Tindakan ini akan menghapus dokumen dari memori vektor secara permanen dan tidak dapat dibatalkan.")

        all_docs = collection.get(include=["metadatas"])
        if all_docs['ids']:
            doc_options = {}
            for i, doc_id in enumerate(all_docs['ids']):
                meta = all_docs['metadatas'][i]
                title = meta.get('title', 'Tanpa Judul')
                label = f"{title} (ID: {doc_id})"
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
                    st.error("Fungsi penghapusan tidak tersedia (Cek impor modul).")
        else:
            st.info("Tidak ada dokumen yang tersedia untuk dihapus.")

except ValueError:
    st.error(f"Koleksi '{COLLECTION_NAME}' tidak ditemukan. Format database mungkin tidak valid.")
except Exception as e:
    st.error(f"Terjadi kesalahan sistem internal: {e}")