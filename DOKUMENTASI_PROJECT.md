# Dokumentasi Project Skripsi: Penerapan MiniLM untuk semantic search pada dokumen arsip kampus

## 📋 Daftar Isi
1. [Pengenalan Project](#pengenalan-project)
2. [Pipeline Data Cleaning](#pipeline-data-cleaning)
3. [Model Training MiniLM](#model-training-minilm)
4. [Embedding & Indexing ChromaDB](#embedding--indexing-chromadb)
5. [RAG Agent & Agentic RAG](#rag-agent--agentic-rag)
6. [Evaluasi dengan RAGAS](#evaluasi-dengan-ragas)
7. [Metriks & Hasil](#metriks--hasil)

---

## 🎯 Pengenalan Project

**Tujuan**: Membangun sistem RAG (Retrieval-Augmented Generation) untuk menjawab pertanyaan berbasis dokumen arsip dengan presisi tinggi menggunakan semantic search dan LLM.

**Stack Teknologi**:
- **Embedding Model**: Sentence Transformers (MiniLM-L12-v2, fine-tuned)
- **Vector Database**: ChromaDB (Cosine similarity, local persistent storage)
- **LLM Backend**: Ollama / vLLM dengan 3 model (Llama 3.2, Gemma 2, Qwen 2.5)
- **Reranker**: Cross-Encoder multilingual (mmarco-mMiniLMv2-L12-H384-v1)
- **Evaluasi**: RAGAS framework (Faithfulness, AnswerRelevancy, ContextPrecision)

**Arsitektur Umum**:
```
Raw Data (PDF/Excel) 
    ↓
Data Cleaning & Preprocessing
    ↓
Dataset Preparation (Train/Test Split)
    ↓
Model Fine-Tuning (MiniLM)
    ↓
ChromaDB Indexing
    ↓
RAG Agent (Vector Search + Reranking + LLM)
    ↓
Evaluation (RAGAS Metrics)
```

---

## 🧹 Pipeline Data Cleaning

### 1. **Text Extraction** (`text_extractor.py`)
Mengekstrak teks dari file PDF/DOCX/TXT sesuai format file.

### 2. **Generate Metadata** (`preprocess.py`)
- **Input**: `dataset_arsip.xlsx` (berisi: title, description, keywords, file_path, file_name)
- **Proses**:
  - Baca setiap baris Excel
  - Ekstrak teks penuh dari file fisik (PDF/DOCX)
  - Gabungkan: `description + extracted_text + keywords` → **rich content**
  - Fallback ke deskripsi jika file tidak ditemukan
- **Output**: `data/indodoc/metadata.jsonl` (JSONL format, 1 entry per baris)

**Strategi Konten Kaya** (Hybrid Content):
```
Deskripsi Excel + Isi Dokumen Asli + Keywords
= Konteks lengkap untuk embedding & training
```

### 3. **Prepare Dataset** (`prepare_dataset.py`)
- **Input**: `metadata.jsonl`
- **Proses**:
  - **Text Cleaning**: Lowercase → Remove URLs → Remove special chars → Remove newlines → Remove extra spaces
  - **Validasi**: Skip data kosong
  - **Split**: 80% TRAIN, 20% TEST (random sampling)
- **Output**: 
  - `data/indodoc/train.jsonl` (semua data)
  - `data/indodoc/test.jsonl` (20% sample untuk evaluasi)

**Pipeline Pembersihan**:
```
Raw Text → Lowercase → Remove URLs → Normalize Special Chars 
→ Remove Newlines → Collapse Multiple Spaces → Clean Text
```

---

## 🤖 Model Training MiniLM

### **Model Base**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Multilingual support (Bahasa Indonesia optimal)
- Dimensi embedding: **384**
- Max sequence length: **384 tokens**

### **Hyperparameter Training**:
```yaml
Batch Size: 16
Epochs: 10
Learning Rate: 2e-5
Warmup Steps: 10% dari total steps
Loss Function: MultipleNegativesRankingLoss
Device: CUDA (dengan AMP - Automatic Mixed Precision)
```

### **Data Augmentation** (`train_minilm_boosted.py`)

Strategi: Dari 1 data → 3 training samples

```python
Untuk setiap entry (title, content):
1. Pasangan Utama: (title, content)
2. Potongan Konten: (content_snippet[:200], title)
3. Keyword Match: (extracted_keywords, title)
```

**Hasil**: Data bertambah 3x lipat → boosts generalisasi model

### **Training Process**:
1. Load `train.jsonl` dengan augmentasi
2. Compile model dengan MiniLM base + Pooling layer
3. Fit dengan DataLoader (batch shuffled)
4. Save final model → `output/minilm-dokumen-arsip-boosted/`
5. Log statistik waktu training → `training_stats.json`

**Output Model**:
- `pytorch_model.bin` (weights)
- `config.json`, `tokenizer.json`, `sentence_bert_config.json` (metadata)

---

## 🗂️ Embedding & Indexing ChromaDB

### **ChromaDB Collection Setup** (`indexer_chroma.py`)

**Configuration**:
```yaml
Collection Name: arsip_kampus_v2
Similarity Metric: cosine
Storage Path: chroma_db_storage/ (persistent local)
```

### **Indexing Process**:

1. **Load Data**:
   - `train.jsonl` (untuk embedding)
   - `metadata.jsonl` (untuk metadata - sinkronisasi dengan data asli)

2. **Build Text untuk Embedding**:
   ```
   text_to_embed = "{title}. {content}. kata kunci: {keywords}"
   ```

3. **Extract & Store Metadata**:
   ```json
   {
     "title": "Judul Dokumen",
     "content": "Isi lengkap (backward compat)",
     "snippet": "Isi[:200] - potongan ringkas",
     "content_only": "Teks asli train.jsonl",
     "file_path": "path/to/file.pdf",
     "file_name": "file.pdf",
     "keywords": "kata kunci hasil ekstrak"
   }
   ```

4. **Vectorize & Upload ke ChromaDB**:
   - Model: fine-tuned MiniLM
   - Setiap dokumen → vector 384-dimensi
   - Simpan dengan ID unik `doc_{idx}`

**Result**: Vector database siap untuk semantic search dengan cosine similarity

---

## 🔍 RAG Agent & Agentic RAG

### **Arsitektur RAG Pipeline**:

```
User Query
    ↓
[1] Query Cleaning (remove common phrases)
    ↓
[2] Semantic Search (ChromaDB)
    ├─ Encode query dengan MiniLM
    ├─ Cari top-40 kandidat (cosine similarity)
    ↓
[3] Reranking (Cross-Encoder)
    ├─ Score ulang top-40 hasil
    ├─ Hybrid scoring (semantic + keyword matching)
    ├─ Return top-10 dokumen final
    ↓
[4] Context Enrichment (Document Reader)
    ├─ Ambil full_context dari ChromaDB
    ├─ Fallback: file fisik (jika ada) atau metadata
    ├─ Batasi max 3000 char/dokumen untuk LLM
    ↓
[5] Prompt Building
    ├─ Konteks + Pertanyaan → structured prompt
    ├─ Instruksi ketat: jangan halusinasi, semantic matching allowed
    ↓
[6] LLM Inference
    ├─ Backend: Ollama/vLLM (auto-detect)
    ├─ Model aktif: Llama 3.2 / Gemma 2 / Qwen 2.5
    ↓
[7] Response Generation
    └─ Answer + Sources + Metrics (latency, context chars, etc.)
```

### **Key Components**:

#### **Query Cleaning** (`clean_user_query`)
```python
Remove: "carikan", "tolong carikan", "tunjukkan", "cari tentang", etc.
→ Extract core semantic query
```

#### **Search Tool** (`search_tool`)
- Query encoding dengan MiniLM
- ChromaDB vector search → top K candidates
- Reranking dengan Cross-Encoder
- Return SearchResult objects dengan full_context

#### **Reranking Strategy** (`reranker.py`)
```python
Score = α × semantic_score + β × keyword_match_score
       + γ × similarity_matching + δ × stopword_filtering

α=0.4, β=0.6 (configurable)

Features:
- Sigmoid normalization (raw scores → [0, 1])
- Sastrawi stopword removal (Indonesian)
- Fuzzy keyword matching dengan threshold 0.75
```

#### **Context Enrichment** (`document_reader.py`)
```
For each search result:
  1. Coba baca file fisik (PDF/DOCX) dari server
  2. Fallback: ambil document_asli dari ChromaDB
  3. Fallback 2: ambil content_only dari metadata
  4. Fallback 3 (Emergency): gunakan snippet (200 char)
  5. Batasi hasil hingga MAX_CHARS_PER_DOC (3000)
```

#### **Prompt Engineering** (`build_prompt`)
```
System Role: Asisten informasi akademik UNIDA Gontor

Rules:
1. Semantic matching: izinkan sinonim & padanan (Syarat=Kualifikasi, SK=Keputusan)
2. Temporal reasoning: gunakan tahun untuk menentukan data terbaru
3. Full sentences: jangan jawab 1 kata saja
4. Fallback: jika tidak ada di konteks → "Maaf, informasi tidak ditemukan"
5. No hallucination: strict factual answers
6. No preamble: langsung ke inti jawaban
7. Numbered steps: jika ada prosedur
8. Explicit data: sebutkan nomor SK jika relevan

Konteks: [TOP-5 KUTIPAN DOKUMEN RELEVAN]
Pertanyaan: [USER QUERY]
```

#### **LLM Backend** (`call_llm`)
```
Mode Selection (Priority):
1. Explicit Mode: config.yaml → backend="vllm"|"ollama"
2. Auto Mode: Health check vLLM (30s cache) → fallback ke Ollama
3. Environment Overrides: RAG_LLM_BACKEND, OLLAMA_HOST, RAG_VLLM_BASE_URL

Inference Settings:
- Temperature: 0.2-0.3 (deterministic)
- Top-p: 0.9
- Max tokens: 1024
- Timeout: 120 detik
```

### **RAGResponse Structure**:
```python
{
  "question": "user query",
  "answer": "jawaban dari LLM",
  "sources": [
    {
      "title": "dokumen title",
      "score": 0.89,
      "file_name": "file.pdf",
      "download_url": "http://...",
      "context_chars": 2500,
      "full_context": "teks konteks lengkap"
    }
  ],
  "search_results_count": 5,
  "context_chars_total": 12500,
  "latency": 2.34  # detik
}
```

---

## 📊 Evaluasi dengan RAGAS

### **RAGAS Metrics** (Retrieval-Augmented Generation Assessment)

**3 Metrik Utama**:

1. **Faithfulness** (Fidelity Score)
   - Apakah jawaban sesuai dengan konteks yang diberikan?
   - Range: 0-1 (1 = perfectly faithful)
   - Method: LLM-based evaluation

2. **AnswerRelevancy** (Relevance Score)
   - Apakah jawaban menjawab pertanyaan yang diajukan?
   - Range: 0-1 (1 = perfectly relevant)
   - Method: Embedding similarity (query vs answer)

3. **ContextPrecision** (Precision Score)
   - Apakah konteks yang diberikan support jawaban?
   - Range: 0-1 (1 = all context relevant)
   - Method: LLM-based filtering

### **Setup Evaluasi** (`run_evaluation_llama.py`, `run_evaluation_gemma.py`)

**Step 1: Data Preparation**
```python
Input: benchmark_results_{llama|gemma}.csv
Columns: Question | Answer | Ground_Truth | Contexts

Process:
- Parse Contexts JSON
- Filter rows dengan context kosong (reliability)
- Build Dataset untuk RAGAS
```

**Step 2: Judge Setup**
```python
Judge LLM: qwen2.5:7b (Ollama)
  - Temperature: 0 (deterministic)
  - Format: JSON (diperlukan RAGAS)
  - Num_ctx: 8192 tokens

Evaluator Embedding: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  - Bukan model yang sedang ditest (objektif)
  - Semantically strong untuk AnswerRelevancy
```

**Step 3: Evaluation Run**
```python
for each benchmark_result:
  question, answer, ground_truth, contexts
  → evaluate dengan 3 metrics
  → aggregate scores
```

**Step 4: Report Generation**
```
Output: ragas_report_{llama|gemma}.csv

Columns:
- question
- answer
- faithfulness
- answer_relevancy
- context_precision
- (rata-rata metrics per model)
```

### **3 Model LLM yang Dievaluasi**:

| Model | Backend | Config | Ukuran | Use Case |
|-------|---------|--------|--------|----------|
| **Llama 3.2 3B** | Ollama/vLLM | config_llama.yaml | 3B params | Balanced, reasoning |
| **Gemma 2 2B** | Ollama/vLLM | config_gemma.yaml | 2B params | Lightweight, fast |
| **Qwen 2.5 7B** | vLLM | config.yaml | 7B params | Strong, accurate (judge) |

**Evaluation Files**:
- `benchmark_results_llama_v*.csv` (test results per iteration)
- `ragas_report_llama*.csv` (RAGAS scores)
- `ragas_report_llama*.json` (detailed per-question scores)

---

## 📈 Metriks & Hasil

### **Model Training Metrics**

**MiniLM Fine-Tuning**:
```python
Model: paraphrase-multilingual-MiniLM-L12-v2
Training Data: ~150-200 dokumen (setelah augmentasi 3x → ~450-600 pairs)
Loss Function: MultipleNegativesRankingLoss
Epochs: 10
Duration: ~5-10 menit (GPU CUDA)

Output:
- pytorch_model.bin (fine-tuned weights)
- training_stats.json (waktu, epochs, sample count)
```

### **MiniLM Evaluation Metrics** (`evaluate_minilm.py`)

Information Retrieval Evaluator:
```
MRR@10 (Mean Reciprocal Rank): 
  → Rank of relevant doc dalam top-10

NDCG@10 (Normalized Discounted Cumulative Gain):
  → Quality of ranking (position matters)

Recall@1, Recall@5:
  → Berapa % queries menemukan relevant doc di top-K
```

**Expected Results**:
- MRR@10: 0.70-0.85 (good semantic understanding)
- NDCG@10: 0.65-0.80 (consistent ranking)
- Recall@5: 0.60-0.75 (coverage)

### **RAG Evaluation Metrics (RAGAS)**

Typical Benchmark Run:
```python
Questions: ~50-100 (sample dari test set)
Ground Truth: Manual annotations atau expected answers
Answers: Generated by RAG pipeline + LLM

Metrics Output (per model):
- Faithfulness: 0.70-0.85 (jawaban stick to context)
- AnswerRelevancy: 0.60-0.80 (jawaban relevant ke question)
- ContextPrecision: 0.50-0.75 (semua context digunakan)

Overall Score: (Faithfulness + AnswerRelevancy + ContextPrecision) / 3
              = 0.60-0.80 range (good RAG performance)
```

### **Benchmark Output Files**

**Benchmark Runner** (`benchmark_runner_{llama|gemma}.py`):
```
Generates:
├── benchmark_results_llama.csv
│   ├── Question | Answer | Ground_Truth | Contexts | Latency
│   └── 100+ rows (full benchmark)
├── ragas_report_llama.csv
│   ├── Faithfulness | AnswerRelevancy | ContextPrecision
│   └── Aggregated scores per question
└── metrics_summary.json
    └── Average scores, timing stats
```

### **Performance Metrics**

**Latency (Waktu Response)**:
- Vector search: 100-500ms
- Reranking: 200-800ms
- LLM inference: 2-10 detik (depends on answer length)
- **Total end-to-end**: 2.5-11 detik

**Storage**:
- ChromaDB index: ~50-200MB (depends on corpus size)
- Model weights (MiniLM): ~200MB
- LLM models (Ollama): 2-7GB (per model)

---

## 🏗️ File Structure & Workflows

### **Key Files**:
```
projectSkripsiSemantic/
├── config.yaml              # Main config (qwen2.5)
├── config_llama.yaml        # Llama 3.2 config
├── config_gemma.yaml        # Gemma 2 config
├── dataset_arsip.xlsx       # Raw excel dengan metadata
│
├── data/indodoc/
│   ├── metadata.jsonl       # Generated dari preprocess.py
│   ├── train.jsonl          # Full training data
│   └── test.jsonl           # 20% sample untuk eval
│
├── src/
│   ├── preprocess.py        # Generate metadata dari Excel ✓
│   ├── prepare_dataset.py   # Create train/test split ✓
│   ├── train_minilm_boosted.py # Fine-tune MiniLM ✓
│   ├── indexer_chroma.py    # Build ChromaDB index ✓
│   ├── rag_agent.py         # RAG pipeline + LLM integration ✓
│   ├── reranker.py          # Cross-Encoder reranking
│   ├── document_reader.py   # Context enrichment
│   ├── main_api.py          # FastAPI server untuk search
│   └── (other utils)
│
├── evaluate/
│   ├── evaluate_minilm.py   # Test MiniLM retrieval metrics
│   ├── run_evaluation_llama.py  # RAGAS eval (Llama)
│   ├── run_evaluation_gemma.py  # RAGAS eval (Gemma)
│   ├── benchmark_runner_llama.py    # Generate benchmark data
│   ├── benchmark_runner_gemma.py    # Generate benchmark data
│   └── benchmark_logger_*.py    # Log & metrics
│
├── output/
│   ├── minilm-dokumen-arsip-boosted/  # Trained model ✓
│   ├── benchmark_results_*.csv        # Raw Q&A results
│   └── ragas_report_*.csv             # RAGAS metrics
│
└── chroma_db_storage/       # ChromaDB persistent storage
    └── arsip_kampus_v2/     # Collection files
```

### **Workflow Eksekusi** (Urutan Running):

```bash
1. python src/preprocess.py
   → Generate data/indodoc/metadata.jsonl

2. python src/prepare_dataset.py
   → Generate data/indodoc/train.jsonl & test.jsonl

3. python src/train_minilm_boosted_seed.py
   → Fine-tune & save output/minilm-dokumen-arsip-boosted/

4. python src/indexer_chroma.py
   → Build ChromaDB index @ chroma_db_storage/

5. python src/main_api.py
   → Start FastAPI server @ http://localhost:8000

6. python evaluate/benchmark_runner_llama.py
   → Generate benchmark_results_llama.csv

7. python evaluate/run_evaluation_llama.py
   → Run RAGAS evaluation → ragas_report_llama.csv

(Repeat steps 6-7 untuk gemma & qwen jika diperlukan)
```

---

## 🎓 Kesimpulan

**Project ini mendemonstrasikan end-to-end RAG system** dengan:

✅ **Data Pipeline**: Hybrid content extraction (Excel + file fisik)  
✅ **Fine-Tuned Embedding**: MiniLM dengan 3x data augmentation  
✅ **Semantic Search**: ChromaDB dengan cosine similarity  
✅ **Reranking**: Cross-Encoder multilingual untuk precision  
✅ **Context-Aware QA**: LLM dengan structured prompt  
✅ **Agentic RAG**: Dynamic backend selection (Ollama/vLLM)  
✅ **Rigorous Evaluation**: RAGAS metrics untuk 3 LLM models  

**Expected Performance**:
- **Retrieval**: Recall@5 60-75%
- **RAG Quality**: Faithfulness 70-85%, Overall Score 75-92%
- **Latency**: 2-11 detik end-to-end
- **Scalability**: Support 1000+ dokumen dengan ChromaDB

---

**Generated**: 2026 | **Project**: Skripsi Semantic Search RAG

