"""
Halaman 2 — Tanya Jawab RAG v2
Menjawab pertanyaan dengan konteks teks penuh dokumen (bukan hanya snippet)
"""

import streamlit as st
import requests
import time

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Tanya Jawab RAG", page_icon="🤖", layout="wide")

st.title("🤖 Tanya Jawab Arsip")
st.caption("Jawaban detail berdasarkan isi dokumen arsip kampus — bukan sekadar daftar judul")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan RAG")

    top_k = st.slider(
        "Dokumen konteks",
        min_value=1, max_value=5, value=3,
        help="Jumlah dokumen yang dibaca LLM. "
             "Lebih banyak = lebih akurat tapi lebih lambat."
    )

    st.divider()
    st.subheader("📊 Status Komponen")

    if st.button("🔄 Cek Status"):
        with st.spinner("Mengecek..."):
            try:
                r = requests.get(f"{API_BASE}/rag/status", timeout=5)
                s = r.json().get("components", {})
                st.write("Search API :", "✅" if s.get("search_api") else "❌")
                st.write("Ollama LLM :", "✅" if s.get("ollama") else "❌")
                st.write("Model      :", s.get("model", "-"))
                if not s.get("ollama"):
                    st.warning("Jalankan: `ollama serve`")
                if not s.get("search_api"):
                    st.warning("Jalankan: `uvicorn src.main_api:app --reload`")
            except Exception:
                st.error("❌ Tidak dapat cek status RAG")

    st.divider()
    st.info(
        "**Contoh pertanyaan:**\n\n"
        "- Apa prosedur pengajuan SK dosen tetap?\n"
        "- Jelaskan isi SOP wisuda semester ganjil\n"
        "- Apa syarat cuti akademik mahasiswa?\n"
        "- Sebutkan poin-poin SK Rektor tentang UKT"
    )

    # Tombol hapus riwayat
    st.divider()
    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Riwayat Chat ──────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Tampilkan riwayat
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])

        # Info statistik konteks
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        col_stat1.caption(f"📄 {chat.get('doc_count', 0)} dokumen dibaca")
        col_stat2.caption(f"📝 {chat.get('context_chars', 0):,} karakter konteks")
        col_stat3.caption(f"⏱ {chat.get('elapsed', 0):.1f}s")

        # Sumber dokumen
        if chat.get("sources"):
            with st.expander(f"📚 {len(chat['sources'])} dokumen sumber"):
                for src in chat["sources"]:
                    score_pct = int(src.get("score", 0) * 100)
                    ctx_chars = src.get("context_chars", 0)
                    st.markdown(
                        f"**{src['title']}** — relevansi {score_pct}% "
                        f"· {ctx_chars:,} karakter dibaca"
                    )
                    if src.get("download_url"):
                        st.markdown(f"[⬇️ Unduh dokumen]({src['download_url']})")
                    st.divider()

# ── Input Pertanyaan ──────────────────────────────────────────────
question = st.chat_input(
    "Tanyakan sesuatu tentang arsip kampus UNIDA Gontor..."
)

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Membaca dokumen dan menyusun jawaban detail..."):
            start = time.time()
            try:
                response = requests.post(
                    f"{API_BASE}/rag/ask",
                    json={"question": question, "top_k": top_k},
                    timeout=120
                )
                data = response.json()
                elapsed = time.time() - start

                answer       = data.get("answer", "")
                sources      = data.get("sources", [])
                doc_count    = data.get("search_results_count", 0)
                ctx_chars    = data.get("context_chars_total", 0)

                # Tampilkan jawaban
                st.write(answer)

                # Statistik
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Dokumen dibaca", doc_count)
                col_b.metric("Konteks", f"{ctx_chars:,} chars")
                col_c.metric("Waktu", f"{elapsed:.1f}s")

                # Sumber dokumen
                if sources:
                    with st.expander(f"📚 {len(sources)} dokumen sumber"):
                        for src in sources:
                            score_pct = int(src.get("score", 0) * 100)
                            ctx_doc   = src.get("context_chars", 0)
                            col_x, col_y = st.columns([3, 1])
                            with col_x:
                                st.markdown(f"**{src.get('title', '-')}**")
                                st.caption(
                                    f"`{src.get('file_name', '-')}` "
                                    f"· {ctx_doc:,} karakter dibaca"
                                )
                            with col_y:
                                st.metric("Relevansi", f"{score_pct}%")
                            if src.get("download_url"):
                                st.markdown(f"[⬇️ Unduh]({src['download_url']})")
                            st.divider()

                # Simpan ke riwayat
                st.session_state.chat_history.append({
                    "question":     question,
                    "answer":       answer,
                    "sources":      sources,
                    "doc_count":    doc_count,
                    "context_chars": ctx_chars,
                    "elapsed":      elapsed
                })

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Tidak dapat terhubung ke API.\n\n"
                    "Pastikan server berjalan:\n"
                    "`uvicorn src.main_api:app --reload`"
                )
            except requests.exceptions.Timeout:
                st.error(
                    "❌ LLM timeout.\n\n"
                    "Pastikan Ollama berjalan:\n"
                    "`ollama serve`\n\n"
                    "Atau gunakan model lebih kecil:\n"
                    "`ollama pull qwen2.5:3b`"
                )
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")