# 🎨 UI/UX Design Specification & Stitch Prompts
## Sistem Tanya Jawab Cerdas & Pencarian Arsip Semantik (LangGraph + Gemma 2 Edition)

Dokumen ini berisi spesifikasi arsitektur antarmuka (UI/UX), *design system*, tata letak komponen (*wireframe layout*), dan kumpulan **Prompt Siap Pakai untuk Google Stitch** guna merancang antarmuka frontend modern, interaktif, dan futuristik untuk `eksperimen/view_database_gemma.py`.

---

## 1. 📌 Ikhtisar Produk (Product Overview)

* **Nama Aplikasi:** SERANAH AI — Campus Archive Semantic Intelligence & Agentic RAG
* **Target Pengguna:** Mahasiswa, Dosen, Tenaga Kependidikan, dan Pimpinan Universitas Darussalam (UNIDA) Gontor.
* **Tujuan Utama:** 
  1. **Pencarian Semantik & Hybrid Reranking:** Menemukan dokumen arsip resmi (SK, Pedoman, Surat Edaran) secara instan berbasis makna teks (Bi-Encoder MiniLM + Cross-Encoder).
  2. **Tanya Jawab Cerdas (Agentic RAG):** Memberikan jawaban analitis yang akurat, terstruktur, dan berbasis sitasi dokumen resmi menggunakan model **Gemma 2** yang diorkestrasi oleh **LangGraph State Machine** (Self-Corrective CRAG).

---

## 2. 💎 Design System & Visual Aesthetics

### A. Palet Warna (Modern Dark & Glassmorphism Theme)
* **Background Utama:** `#0B0F19` (Deep Obsidian / Slate Navy)
* **Card & Container Surface:** `#161F30` dengan efek *Glassmorphism* (`backdrop-filter: blur(12px)`, border `1px solid rgba(255, 255, 255, 0.08)`)
* **Primary Accent (AI Glow):** `#6366F1` (Electric Indigo) $\rightarrow$ `#8B5CF6` (Cyber Violet Gradient)
* **Success / Online Status:** `#10B981` (Emerald Green)
* **Warning / Self-Correction Badge:** `#F59E0B` (Vibrant Amber)
* **Text Primary:** `#F8FAFC` (Pure White Slate)
* **Text Secondary / Metadata:** `#94A3B8` (Cool Grey)

### B. Tipografi
* **Font Family:** `Inter` atau `Plus Jakarta Sans` (Google Fonts)
* **Heading 1:** 28px, Bold, Gradient Text (`linear-gradient(135deg, #FFFFFF, #94A3B8)`)
* **Body / Chat:** 15px, Regular (Line-height: 1.6)
* **Code / Latency Telemetry:** `JetBrains Mono` / `Fira Code` (13px, Monospace)

### C. Efek & Interaksi Mikro (Micro-Animations)
* **Glow Pulses:** Indikator koneksi real-time AI & Database.
* **Graph Flow Animation:** Animasi alur node LangGraph saat berpikir (*Query Cleaning* $\rightarrow$ *Retrieval* $\rightarrow$ *Grading* $\rightarrow$ *Generation*).
* **Hover Elevation:** Kartu dokumen sumber terangkat halus saat kursor melayang (`transform: translateY(-2px)`).

---

## 3. 📐 Arsitektur Tata Letak Halaman (Layout Blueprint)

```
+---------------------------------------------------------------------------------------------------+
|  [LOGO] SERANAH AI  |  🟢 ChromaDB: 1,030 Dokumen  |  🧠 Gemma 2 (CUDA GPU)  |  ⚡ 48ms Latency   |
+---------------------+-----------------------------------------------------------------------------+
| SIDEBAR             | WORKSPACE TABS                                                              |
|                     | [ 🤖 Tanya Jawab Cerdas (LangGraph) ] [ 🔍 Semantic Search ] [ 📊 Vektor ]  |
| ⚙️ KONFIGURASI       +-----------------------------------------------------------------------------+
| • Backend: :8002    | 💬 CHAT STREAM / TIMELINE AREA                                              |
| • Model: Gemma 2    |                                                                             |
| • Engine: CRAG      | 👤 User: "Bagaimana aturan cuti akademik mahasiswa UNIDA?"                  |
|                     |                                                                             |
| 📊 STATUS BACKEND   | 🤖 Gemma AI:                                                                |
| • Ollama: 🟢 Online | "Berdasarkan Buku Pedoman Akademik UNIDA Gontor Tahun 2024 (SK No. 12/2024),|
| • Chroma: 🟢 Online | mahasiswa yang mengajukan cuti akademik wajib memenuhi syarat berikut:      |
|                     |  1. Telah menempuh minimal 2 semester aktif...                             |
| ➕ UPLOAD ARSIP     |  2. Mengajukan surat permohonan ke BAAK..."                                 |
| [ Drag & Drop File] |                                                                             |
| [ Judul / Unit ]    | 🔄 STATUS GRAPH AGENT:                                                      |
| [ Simpan Dokumen ]  | [✅ Kueri Relevan] [Skor Relevansi: 0.892] [LangGraph Iterasi: 1]            |
|                     |                                                                             |
|                     | ⏱️ TELEMETRI LATENSI:                                                       |
|                     | [ Total: 1.42s ] [ Search: 0.008s ] [ Rerank: 0.024s ] [ LLM: 1.38s ]       |
|                     |                                                                             |
|                     | 📚 DOKUMEN SITASI & SUMBER (3 Dokumen):                                     |
|                     | +-------------------------------------------------------------------------+ |
|                     | | 📄 [Top 1] SK Rektor No. 12 Tahun 2024 - Pedoman Akademik (Skor: 0.892) | |
|                     | | Unit: BAAK | Tahun: 2024 | File: SK_Pedoman_2024.pdf                    | |
|                     | | "Pasal 14: Prosedur dan Syarat Pengajuan Cuti Akademik Mahasiswa..."      | |
|                     | +-------------------------------------------------------------------------+ |
+---------------------+-----------------------------------------------------------------------------+
| INPUT PROMPT BOX: [ Tanyakan regulasi, SK, atau pedoman kampus...               ] [ Kirim 🚀 ]    |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. 🧩 Rincian Komponen UI Interaktif

### Komponen 1: Graph State Execution Badge
Menampilkan status proses penalaran agen LangGraph secara visual:
* **Badge 1 (Normal / Direct Hit):** `✅ Relevan Terverifikasi` (Hijau Neon).
* **Badge 2 (Self-Correction Active):** `🔄 Query Rewriting: "Kueri Baru..." (Iterasi #1)` (Kuning Amber Glow).

### Komponen 2: Latency Telemetry Grid (4 Cards)
Menampilkan metrik komputasi secara transparan untuk pengujian skripsi:
1. **Total Waktu End-to-End:** misal `1.24s`
2. **Pencarian Vektor GPU:** misal `0.005s`
3. **Cross-Encoder Rerank:** misal `0.018s`
4. **Gemma 2 LLM Generation:** misal `1.21s`

### Komponen 3: Source Citation Drawer / Cards
Kartu sitasi dokumen sumber yang dapat diekspansi (*accordion*):
* Tag Kategori: `SK Rektor`, `Pedoman Akademik`, `SOP`, dll.
* Skor Relevansi: *Progress bar* berwarna hijau gradien (`0.000` s/d `1.000`).
* Cuplikan Paragraf: Kotak abu-abu berlatar gelap yang menyorot kalimat kunci (*keyword highlight*).

---

## 5. 🚀 Prompt Siap Pakai untuk Google Stitch

Salin dan tempel prompt di bawah ini ke dalam **Google Stitch** untuk menghasilkan desain antarmuka lengkap:

### 📋 Prompt 1: Desain Halaman Utama (Main Dashboard & Chatbot RAG)
```text
Create a modern, high-end, futuristic web application interface for "SERANAH AI", an AI-powered University Campus Archive Management and Agentic RAG System using Gemma 2 and LangGraph.

Theme & Aesthetics:
- Sleek Dark Mode with Obsidian Navy (#0B0F19) background and deep navy glassmorphic cards (#161F30) with subtle purple/indigo glowing borders.
- Typography: Clean Google Font (Inter/Plus Jakarta Sans) with crisp hierarchy and high readability.
- Tech Stack feel: Modern AI Dashboard with real-time status chips and latency telemetry.

Layout Structure:
1. Top Navigation Bar:
   - App Logo: "SERANAH AI" with an glowing neural network archive icon.
   - Status Badges: "🟢 ChromaDB: 1,030 Docs", "🧠 Gemma 2 (CUDA)", "⚡ API Status: Healthy (Port 8002)".
   - User profile / University badge: "Universitas Darussalam Gontor".

2. Left Sidebar (Collapsible):
   - Backend Config Section: API URL input, Model selection (Gemma 2), Engine selector (LangGraph Self-Corrective CRAG vs Native RAG).
   - Component Health Check Card: Ollama status, Vector DB status, Reranker model status.
   - Quick Archive Ingestion Widget: Drag & drop PDF uploader with Title, Unit Kerja, and Description inputs.

3. Main Content Area (Multi-Tabbed Interface):
   - Tab Navigation: [🤖 AI Archive Assistant (LangGraph RAG)] [🔍 Semantic Search & Reranking] [📊 Vector Database Explorer]

   - Tab 1 Active View (AI Agentic RAG Chat):
     * Chat Message Stream with distinct User bubble and AI Agent response card.
     * AI Response Card: Clean markdown formatted answer with numbered lists, bold text, and university policy citations.
     * Workflow Step Indicator: "Self-Correction Status: ✅ Relevant on First Pass (Score: 0.892)" or "🔄 Query Rewritten by Gemma 2: 'pedoman cuti akademik mahasiswa' (Retry #1)".
     * Telemetry Grid (4 metric pills): Total Latency (1.35s), GPU Vector Search (0.007s), Cross-Encoder Rerank (0.021s), LLM Inference (1.32s).
     * Collapsible Source Document Cards: Expandable cards showing Document Title, SK Number, Year, Unit Kerja, Relevance Score Bar, and Document Snippet.

4. Bottom Floating Input Bar:
   - Elegant rounded search/prompt bar with placeholder: "Tanyakan regulasi, SK rektor, atau pedoman kampus...".
   - Top-K slider control (Default: 10).
   - Glowing "Tanya AI 🚀" primary gradient action button.
```

---

### 📋 Prompt 2: Desain Tab Semantic Search Explorer (Pencarian Cepat)
```text
Design the "Semantic Search & Hybrid Reranking" tab view for SERANAH AI Archive System.

Components to render:
1. Prominent Search Header with a search bar and a slider for "Top-K Documents (1-20)".
2. Performance Telemetry Banner: Displays "Found 10 relevant documents in 0.038s using Bi-Encoder MiniLM + Cross-Encoder Reranker".
3. Ranked Search Results List:
   - Card 1 (Top 1 Rank): Highlighted with a gold/emerald badge "🏆 Top 1 - Relevance Score 0.942". Title: "SK Rektor No. 1270 Tahun 2024 tentang Pedoman Organisasi Kemahasiswaan". Badges for Unit: "Sekretariat Universitas", Year: "2024", Category: "Arsip SK". Full preview snippet with highlighted keyword matches. Download PDF button.
   - Card 2 & 3: Similar modern glassmorphic cards with silver/bronze rank badges and relevance meters.
```

---

### 📋 Prompt 3: Desain Tab Vector Database & Knowledge Explorer
```text
Design the "Vector Database & Metadata Explorer" tab view for SERANAH AI.

Components to render:
1. Summary Metric Cards: "Total Vectors: 1,030", "Embedding Dimensions: 384 (MiniLM)", "Distance Metric: Cosine Similarity", "Active Collection: arsip_kampus_v2".
2. Interactive Data Table:
   - Columns: ID/UUID, Document Title, Unit Kerja, Category, Year, Document Number, Embedding Vector Preview ([0.0421, -0.1982, 0.0834, ...]), Actions.
   - Search/filter input for filtering table rows in real-time.
   - Action buttons: "👁️ Pratinjau Teks Lengkap" and "🗑️ Hapus dari Indeks Vektor".
```

---

### 6. 🛠️ Panduan Integrasi ke Kode Streamlit / Web App

Setelah Anda membuat mockup desain di Stitch:
1. Buka file [eksperimen/view_database_gemma.py](file:///d:/3.%20ML/projectSkripsiSemantic/eksperimen/view_database_gemma.py).
2. Anda dapat menyuntikkan (*inject*) CSS kustom modern menggunakan `st.markdown('<style>...</style>', unsafe_allow_html=True)` sesuai dengan palet warna dan class styling dari hasil Stitch.
3. Semua endpoint backend (`/rag/ask`, `/rag/status`, `/search`) pada [eksperimen/main_api_gemma.py](file:///d:/3.%20ML/projectSkripsiSemantic/eksperimen/main_api_gemma.py) sudah siap 100% menerima request dari desain frontend baru Anda.
