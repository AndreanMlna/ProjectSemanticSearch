# Dokumentasi: MiniLM (Bi-Encoder), Cross-Encoder (Reranker) dan RAG

Ringkasan singkat dan panduan alur kerja yang menjelaskan bagaimana MiniLM bi-encoder, MiniLM cross-encoder (sebagai reranker) dan alur RAG diimplementasikan dan terintegrasi pada proyek ini. Dokumen ini dibuat berdasarkan pembacaan kode di repository dan hanya menjelaskan — tidak mengubah kode atau logika apapun.

Checklist (apa yang dijelaskan di file ini):
- Peran masing-masing komponen (bi-encoder, cross-encoder, RAG)
- File utama yang mengimplementasikan tiap komponen
- Alur data langkah demi langkah pada saat training dan inference
- Formula hybrid scoring dan heuristik reranking yang digunakan
- Lokasi model/output penting dan tips troubleshooting singkat

CATATAN: semua referensi file merujuk pada path di repository Anda (lihat bagian "File kunci" di bawah).

------------------------------------------------------------

1) Komponen & Peran

- MiniLM (Bi-Encoder)
  - Peran: menjadi model embedding (sentence-transformers) untuk mengubah kueri dan dokumen menjadi vektor berdimensi tetap. Vektor-vektor ini diindeks ke ChromaDB untuk pencarian nearest-neighbour cepat.
  - Implementasi / training: skrip pelatihan berada di `eksperimen/train_minilm.py` (versi sederhana) dan implementasi yang lebih lengkap/augmentasi ada di `src/train_minilm_boosted.py` serta `src/train_minilm_boosted_seed.py`.
  - Output: model tersimpan ke folder di `output/` (nama folder bergantung pada konfigurasi/eksperimen).

- Cross-Encoder (Reranker)
  - Peran: melakukan re-scoring kandidat dokumen hasil pencarian vektor dengan melihat pasangan (query, document) secara bersama-sama (cross-attention). Ini biasanya meningkatkan presisi pada peringkat teratas.
  - Implementasi: `src/reranker.py` — kelas reranker memuat model `sentence_transformers.CrossEncoder` dan fungsi `rerank(...)` untuk menghasilkan skor reranker, menggabungkannya dengan skor similarity dari Chroma (hybrid), dan menerapkan beberapa heuristik fuzzy/penalty.

- RAG (Retrieval-Augmented Generation) Agent
  - Peran: Orkestrasi pipeline lengkap untuk menjawab pertanyaan pengguna dengan dukungan dokumen: (1) encode query → (2) vektor search di Chroma → (3) rerank kandidat → (4) susun konteks → (5) panggil LLM (Ollama / vLLM) untuk menghasilkan jawaban yang berbasis dokumen.
  - Implementasi: `eksperimen/rag_agent_llama.py` (fungsi `search_tool`, `build_prompt`, `call_ollama`, `call_vllm`, dan class `RAGAgent` yang mengorkestrasi semuanya).

2) File kunci dan apa yang dilakukan

- `eksperimen/train_minilm.py`
  - Inisialisasi: Transformer + Pooling → menjadi `SentenceTransformer`
  - Loss: `MultipleNegativesRankingLoss`
  - Menjalankan `model.fit(...)` dan `model.save(...)`

- `src/train_minilm_boosted.py` dan `src/train_minilm_boosted_seed.py`
  - Versi yang menambahkan augmentasi data (title+content, snippet+title, keywords+title) dan eksperimen beberapa seed.
  - Menyimpan loss history dan model hasil training ke direktori `output/` sesuai konfigurasi.

- `src/inference_minilm_boosted.py`
  - Contoh pemuatan model hasil training (`SentenceTransformer(MODEL_PATH)`), encoding query, dan query ke ChromaDB.

- `src/reranker.py`
  - Kelas reranker memuat `CrossEncoder` dan menyediakan `rerank(query, results, ...)`.
  - Mengambil candidate results (dokumen, metadata, distance) dari Chroma, membuat pasangan (query, document_text) untuk CrossEncoder, mendapatkan skor reranker (sering kali diproses dengan sigmoid), lalu menghitung hybrid score.
  - Hybird score umum: hybrid = alpha * chroma_similarity + beta * reranker_score (implementasi spesifik ada di kode). Selain itu diterapkan penalty untuk mismatch keyword/fuzzy.

- `eksperimen/rag_agent_llama.py`
  - `search_tool(...)`: membersihkan kueri, encode dengan MiniLM, panggil `chroma_collection.query(...)` untuk ambil N kandidat, lalu panggil `get_reranker().rerank(...)`.
  - `build_prompt(...)`: menyusun konteks terbaik (top-k) ke dalam prompt yang akan dikirim ke LLM. Terdapat aturan untuk panjang konteks, numbering sumber, dan bagaimana menyisipkan metadata (judul, id, jarak).
  - `call_ollama(...)` / `call_vllm(...)`: cara panggilan ke backend LLM yang tersedia (HTTP / Ollama client). Keluarkan jawaban yang diperkaya dengan sumber dan metadata.

3) Alur data (training dan inference)

- Training (MiniLM bi-encoder):
  1. Siapkan dataset pasangan positif (mis. query, relevant_document) atau augmentasi (title+content, snippet+title).
  2. Buat `Transformer` (pretrained MiniLM), `Pooling` layer dan gabungkan menjadi `SentenceTransformer`.
  3. Gunakan `MultipleNegativesRankingLoss` untuk fine-tune dengan batch sampling.
  4. Panggil `model.fit(...)` dan `model.save(output_dir)`.

- Indexing:
  1. Setelah model melatih atau saat data baru ditambahkan, dokumen di-encode ke vektor dengan MiniLM dan disimpan ke koleksi Chroma (`chroma_db_storage/` berisi sqlite dan data collection).
  2. Metadata (judul, id, source) juga disimpan sehingga hasil search dapat ditelusuri kembali ke dokumen asli.

- Inference / RAG (Detail Alur User Input sampai Output):

  PENTING: Ada dua model berbeda yang bekerja di dua tahap berbeda yang TIDAK boleh disamakan:
  - **BI-ENCODER (MiniLM)**: Hanya untuk encoding QUERY ke vektor untuk vector search.
  - **CROSS-ENCODER (reranker)**: Hanya untuk re-scoring pasangan (QUERY TEXT, DOCUMENT TEXT) bersama-sama.

  Tahap-tahap alur kerja:

  Tahap 0 - User Input:
    1. Pengguna memberikan kueri teks (misalnya: "apa saja syarat masuk UNIDA?").

  Tahap 1 - Query Cleaning & Encoding (BI-ENCODER SAJA):
    1. Kueri dibersihkan dari prefix umum (mis. "carikan", "tolong tampilkan") → hasil clean_user_query().
    2. Kueri yang sudah dibersihkan di-ENCODE menjadi VEKTOR menggunakan BI-ENCODER (MiniLM).
       Lokasi kode: `eksperimen/rag_agent_llama.py` baris 167: `query_vector = embedding_model.encode(cleaned_query).tolist()`
    3. Kueri bukan di-encode dengan Cross-Encoder — Cross-Encoder TIDAK di-gunakan pada tahap ini.

  Tahap 2 - Vector Search di ChromaDB:
    1. Vektor kueri dari Tahap 1 digunakan untuk mencari N kandidat dokumen terdekat di ChromaDB.
       Lokasi kode: `eksperimen/rag_agent_llama.py` baris 169-173:
       ```
       candidate_count = max(20, top_k * 4)
       db_results = chroma_collection.query(
           query_embeddings=[query_vector],
           n_results=candidate_count,
           include=["metadatas", "distances", "documents"]
       )
       ```
    2. Hasil: `db_results` berisi daftar N kandidat dokumen beserta metadata, jarak (distance), dan teks dokumen.
    3. Distance mengindikasikan seberapa berbeda dokumen dengan query (jarak vektor).

  Tahap 3 - Reranking (CROSS-ENCODER SAJA):
    1. Daftar N kandidat dari Tahap 2 dipass ke Cross-Encoder reranker.
       Lokasi kode: `eksperimen/rag_agent_llama.py` baris 188:
       ```
       reranker = get_reranker()
       final_ranked = reranker.rerank(query=cleaned_query, chroma_results=db_results, top_k=top_k)
       ```

    2. Di dalam reranker (lihat `src/reranker.py` baris 72-77):
       - Untuk SETIAP dokumen di daftar N kandidat, buat pasangan TEKS: [query_text, "Judul: {title}\nIsi: {doc_text}"]
       - Cross-Encoder MENERIMA PASANGAN TEKS INI (BUKAN vektor) dan menilai seberapa relevan pasangan tsb.
       - Hasil: raw scores yang di-apply sigmoid → rerank_scores (dalam range 0-1).

    3. Hybrid Scoring (kombinasi Chroma distance dengan Cross-Encoder score):
       Lokasi kode: `src/reranker.py` baris 100-102:
       ```
       chroma_similarity = 1 / (1 + float(distances[i]))      # Konversi distance ke similarity
       reranker_score = float(rerank_scores[i])               # Skor dari Cross-Encoder
       final_hybrid_score = (alpha * chroma_similarity) + (beta * reranker_score)
       ```
       - alpha = 0.6 (bobot untuk Chroma similarity)
       - beta = 0.4 (bobot untuk Cross-Encoder score)
       - Hasil: skor final yang mempertimbangkan KEDUA sumber informasi.

    4. Heuristik tambahan (fuzzy keyword matching & penalty):
       - Ekstrak keywords dari query (stopword removal).
       - Cek apakah keywords ada di dokumen (fuzzy match dengan threshold 0.75).
       - Jika overlap keyword terlalu rendah (< 25%), kurangi skor dengan penalty.
       - Filter: hanya dokumen dengan skor ≥ MIN_SCORE_THRESHOLD (0.20) yang dipertahankan.
       - Sort dan ambil top_k hasil.

  Tahap 4 - Context Enrichment:
    1. Top-K hasil reranking diperkaya dengan full context text.
       Lokasi kode: `eksperimen/rag_agent_llama.py` baris 452-457:
       ```
       enriched_docs = get_context_for_results(
           search_results=compat_results,
           query=cleaned_q,
           upload_dir=self.upload_dir
       )
       ```
    2. Fungsi ini mengambil teks dokumen yang sudah disimpan saat indexing dan mempersiapkan untuk dimuat ke prompt.

  Tahap 5 - Prompt Building:
    1. Menyusun prompt yang akan dikirim ke LLM.
    2. Format: Instruksi sistem + Konteks (top-K dokumen yang sudah diurutkan dari tahap reranking) + Pertanyaan user.
       Lokasi kode: `eksperimen/rag_agent_llama.py` baris 278-317 (fungsi `build_prompt`).
    3. Instruksi di prompt MENEKANKAN agar LLM hanya menjawab berdasarkan konteks yang diberikan dan TIDAK boleh hallucinate.

  Tahap 6 - LLM Inference:
    1. Prompt dari Tahap 5 dikirimkan ke backend LLM (Ollama atau vLLM).
       Lokasi kode: `eksperimen/rag_agent_llama.py` baris 474-476:
       ```
       t_llm_start = time.perf_counter()
       answer_text = call_llm(prompt=prompt)
       llm_time = time.perf_counter() - t_llm_start
       ```
    2. LLM membaca prompt, memahami pertanyaan, merujuk ke konteks, dan menghasilkan jawaban natural.

  Tahap 7 - Response Assembly & Output:
    1. Jawaban dari LLM dikombinasikan dengan metadata sumber (judul, file, score, url download).
    2. Dikumpulkan dalam objek RAGResponse yang berisi:
       - question: pertanyaan user
       - answer: jawaban hasil LLM
       - sources: daftar dokumen yang digunakan beserta metadata
       - search_results_count: jumlah dokumen yang dipilih
       - context_chars_total: total karakter konteks yang dikirim ke LLM
       - latency: waktu total end-to-end
       - search_time: waktu Tahap 1-2 (vector search)
       - rerank_time: waktu Tahap 3 (reranking)
       - llm_time: waktu Tahap 6 (LLM inference)

  RINGKASAN SINGKAT KAPAN ENCODER DIGUNAKAN:
  - BI-ENCODER (MiniLM): HANYA di Tahap 1 untuk mengubah QUERY ke VEKTOR untuk optimasi vector search.
  - CROSS-ENCODER (Reranker): HANYA di Tahap 3 untuk menerima PASANGAN (query_text, document_text) dan memberikan SKOR relevansi.

4) Perbedaan BI-ENCODER vs CROSS-ENCODER (PENTING!)

Kedua model ini adalah tipe "encoder" tetapi fungsi dan cara kerjanya sangat berbeda:

- BI-ENCODER (MiniLM di project ini):
  - TUJUAN: Mengubah teks (query / dokumen) menjadi VEKTOR berdimensi tetap (embedding).
  - CARA KERJA:
    1. Query di-encode terpisah → vektor q.
    2. Setiap dokumen di-encode terpisah → vektor d1, d2, d3, ...
    3. Similarity diukur dengan distance antara vektor (mis. cosine distance, L2 distance).
  - KECEPATAN: CEPAT karena encoding dilakukan sekali per teks dan vektor bisa di-cache.
  - TUGAS DI PROJECT: Encoding query di Tahap 1 untuk vector search di ChromaDB.
  - ARSITEKTUR: Transformer + Pooling → output vektor single embedding per input.

- CROSS-ENCODER (Reranker di project ini):
  - TUJUAN: Memberikan SKOR relevansi untuk pasangan teks (query, document).
  - CARA KERJA:
    1. Terima pasangan [query, document] sebagai input BERSAMAAN (bukan terpisah).
    2. Proses melalui Transformer dengan cross-attention → perhatian query ke document.
    3. Output: skor tunggal (real-valued) yang menyatakan "seberapa relevan pasangan ini".
  - KECEPATAN: LAMBAT karena harus di-process bersama untuk setiap pasangan (N × K operasi untuk N queries dan K dokumen).
  - TUGAS DI PROJECT: Re-scoring N kandidat dokumen dari ChromaDB di Tahap 3 untuk meningkatkan presisi ranking.
  - ARSITEKTUR: Transformer biasa dengan pooling output score tunggal.

VISUALISASI ALUR PROCESSING:

  BI-ENCODER (Query Encoding - Tahap 1):
  ┌─────────────────────────────────────────────┐
  │ User Query: "Apa syarat masuk UNIDA?"       │
  └─────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────┐
  │ Cleaning: "syarat masuk unida"              │
  └─────────────────────────────────────────────┘
                          ↓
  ┌──────────────────────────────────────────────────────────────┐
  │ BI-ENCODER (MiniLM): Encode string → vector                 │
  │ Input: "syarat masuk unida"                                  │
  │ Output: [0.12, -0.45, 0.78, ..., 0.33] (dimensi 384)        │
  └──────────────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────┐
  │ Vector Search di ChromaDB:                  │
  │ Cari N dokumen terdekat berdasarkan vektor  │
  │ Result: 20-40 kandidat dokumen              │
  └─────────────────────────────────────────────┘

  CROSS-ENCODER (Reranking - Tahap 3):
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Input 20-40 Kandidat dari ChromaDB                                     │
  │ Per kandidat buat pasangan:                                            │
  │ [  "syarat masuk unida",                                              │
  │    "Judul: Persyaratan Masuk Mahasiswa\nIsi: Peserta harus ... "  ]  │
  └────────────────────────────────────────────────────────────────────────┘
                          ↓
  ┌────────────────────────────────────────────────────────────────────────┐
  │ CROSS-ENCODER: Beri skor untuk SETIAP pasangan                        │
  │ Proses: [query_teks + doc_teks] → cross-attention → skor tunggal      │
  │ Output per pasangan: 2.15, -1.30, 0.98, 3.45, ... (raw scores)        │
  └────────────────────────────────────────────────────────────────────────┘
                          ↓
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Sigmoid transform + Hybrid Score:                                      │
  │ skor_final[i] = 0.6 * (1/(1+distance[i])) + 0.4 * sigmoid(score[i])  │
  │ Hasil: [0.78, 0.45, 0.92, 0.65, ...] → top-K dokumen terbaik         │
  └────────────────────────────────────────────────────────────────────────┘

4) Formula Hybrid Scoring (Tahap 3 - Reranking)

Lokasi implementasi: `src/reranker.py` baris 100-117

Konversi jarak Chroma ke similarity:
  chroma_similarity = 1 / (1 + distance)
  Logika: jarak kecil → similarity tinggi, jarak besar → similarity rendah.

Cross-Encoder scoring (raw → normalized):
  raw_scores = CrossEncoder.predict(pair_inputs)
  rerank_scores = sigmoid(raw_scores)
  Result: nilai-nilai di range [0, 1]
  Fungsi sigmoid: σ(x) = 1 / (1 + exp(-x))

Hybrid score (kombinasi kedua sumber):
  final_hybrid_score = (alpha * chroma_similarity) + (beta * reranker_score)
  alpha = 0.6 (bobot untuk vector similarity dari Chroma)
  beta = 0.4 (bobot untuk cross-encoder score)
  Interpretasi: hasil ChromaDB 60% penting, hasil Reranker 40% penting.

Heuristik keyword matching & penalty (opsional):
  1. Ekstrak keywords dari query (stopword removal dengan Sastrawi).
  2. Untuk setiap dokumen, hitung berapa keyword yang cocok (fuzzy match threshold 0.75).
  3. Jika match_ratio < 0.25 (kurang dari 25% keywords cocok):
     penalty = 0.10 * (1.0 - match_ratio)
     final_hybrid_score -= penalty
  4. Clamp ke minimum 0.0 agar skor tidak negatif.

Filtering & ranking akhir:
  1. Filter: hanya dokumen dengan skor ≥ MIN_SCORE_THRESHOLD (0.20) yang lolos.
  2. Sort: urutkan skor descending (tertinggi dulu).
  3. Ambil: top_k dokumen terbaik (default 10).
  4. Output: list dokumen terurut dengan id, title, snippet, file_name, score untuk Tahap 4.

11) Catatan terperinci tentang Cosine similarity: kapan, dimana, sebelum & sesudah

Ringkasan singkat: proyek ini menggunakan cosine sebagai metric kesamaan vektor pada beberapa titik penting: saat membuat koleksi/indeks Chroma, saat melakukan pencarian vektor (retrieval), dan juga sebagai rujukan metrik pada evaluasi/visualisasi. Namun ada beberapa transformasi nilai (distance -> similarity) dan kombinasi skor yang terjadi setelah Chroma mengembalikan hasil — penting untuk memahami alur "sebelum" (raw embeddings & distance) dan "sesudah" (similarity yang dipakai di hybrid scoring).

Di mana cosine digunakan (lokasi kode & konfigurasi):
- Index / HNSW space: saat koleksi Chroma dibuat, metadata index menyet `hnsw:space` ke "cosine". Lihat:
  - `src/indexer_chroma.py` dan `src/indexer_chroma_seed42.py` (pembuatan collection: metadata {"hnsw:space": "cosine"}).
  - Konfigurasi global: `config.yaml`, `config_llama.yaml`, `config_gemma.yaml` memiliki `similarity_metric: "cosine"`.
- Retrieval (Chroma query): `eksperimen/rag_agent_llama.py` memanggil `chroma_collection.query(...)` yang mengembalikan daftar kandidat beserta `distances`.
- Evaluasi & visualisasi: nama metrik evaluasi pada `visualize/visualize_experiment.py` dan file output evaluasi menggunakan label yang mengandung "_cosine_" (mis. _cosine_accuracy@1, _cosine_mrr@10, dll.).

Penjelasan konsep: "distance" vs "cosine similarity"
- Cosine similarity: ukuran kemiripan antara dua vektor embedding, biasanya dihitung sebagai:
  cosine(a, b) = (a · b) / (||a|| * ||b||)
  Nilai cosine semakin besar menunjukkan kemiripan semantik yang lebih tinggi.
- Distance yang dikembalikan oleh Chroma (field `distances`) adalah nilai jarak sesuai metric indeks. Dengan `hnsw:space = "cosine"`, Chroma mengukur kedekatan menurut ruang cosine — artinya nilai distance lebih kecil → lebih mirip. Konvensi persis (apakah distance = 1 - cosine atau bentuk lain) bergantung pada implementasi underlying (HNSW/FAISS/Chroma), tetapi yang pasti: lebih kecil = lebih dekat.

Transformasi yang dilakukan proyek (sebelum digunakan di hybrid scoring):
- Setelah `chroma_collection.query(...)`, proyek mengambil `distances[i]` dan menghitung:
  chroma_similarity = 1 / (1 + distance)
  - Tujuan transformasi: memetakankan jarak (0..inf, smaller better) ke skor similarity di rentang (0,1], sehingga dapat digabungkan lebih mudah dengan reranker_score (juga di [0,1]).
  - Catatan: jika Chroma mengembalikan distance = 1 - cosine, maka chroma_similarity berkorelasi langsung dengan cosine similarity setelah transformasi; jika tidak, transformasi tetap menjaga urutan relevansi (monotonitas) terhadap jarak asli.

Contoh numerik singkat:
- distance = 0.12 → chroma_similarity = 1 / (1 + 0.12) ≈ 0.8929
- distance = 0.5  → chroma_similarity = 1 / 1.5 ≈ 0.6667

Peran cosine similarity pada tiap kasus (sebelum & sesudah):
- Sebelum (input):
  - Raw teks (query & dokumen) di-encode menjadi vektor embedding oleh BI-ENCODER (MiniLM). Pada titik ini Anda memiliki vektor numeric.
  - Jika embedding sudah dinormalisasi (norm ≈ 1), maka cosine(a,b) = dot(a,b). Namun proyek tidak selalu mengandalkan normalisasi eksplisit — periksa konfigurasi pooling/model bila diperlukan.
- Saat indexing & retrieval (langsung setelah encoding):
  - Vektor dokumen di-index ke Chroma dengan metric cosine. Saat query, Chroma menghitung jarak antara vector query dan vektor dokumen sesuai metric tersebut → mengembalikan `distances` (jarak; kecil = lebih mirip).
  - Ini adalah titik utama penggunaan cosine di project (index & retrieval).
- Setelah retrieval (sesudah menerima distances):
  - Kode proyek mengubah distance menjadi chroma_similarity via 1/(1+distance) dan menggunakan nilai ini dalam hybrid scoring:
    final_hybrid_score = alpha * chroma_similarity + beta * reranker_score
  - Jadi pada tahap ini cosine tidak langsung tampil sebagai "cosine(a,b)", melainkan sebagai skor similarity yang diturunkan dari distance berbasis cosine.

Kapan cosine bukan digunakan (atau tidak relevan):
- Cross-Encoder reranker (`src/reranker.py`) TIDAK menggunakan cosine embedding langsung — reranker menerima pasangan teks (query, document) dan menghasilkan skor tekstual lewat model cross-attention. Skor ini dinormalisasi (sigmoid) dan digabungkan dengan chroma_similarity.

Rekomendasi / checklist pemeriksaan (jika ingin memastikan perilaku cosine di environment Anda):
- Pastikan `hnsw:space` diset ke "cosine" saat membuat koleksi di `src/indexer_chroma*.py`.
- Untuk debugging, hitung cosine manual dari embeddings: cosine(q,d) = dot(q,d) / (||q||*||d||). Jika embedding dinormalisasi, cukup gunakan dot(q,d).
- Ingat konversi Chroma distance → chroma_similarity (1/(1+distance)) saat membandingkan dengan cosine yang dihitung manual.

Singkatnya: proyek mengandalkan COSINE sebagai metric ruang vektor untuk indexing & retrieval. Namun pipeline scoring memakai nilai turunan (transformasi distance → chroma_similarity) yang kemudian digabung dengan skor reranker untuk menentukan peringkat akhir.

5) Lokasi model, database, dan artefak

- Chroma DB (local storage): `chroma_db_storage/chroma.sqlite3` dan subfolder collection ids.
- Output model MiniLM hasil training: `output/<nama_model_eksperimen>/` (lihat `train_minilm_boosted` dan skrip seed untuk nama folder tepatnya).
- Cross-Encoder model: biasanya dimuat dari model name/ path yang didefinisikan di `src/reranker.py`.

6) Tips debugging & troubleshooting

- Jika hasil search terlalu generik atau tidak relevan:
  - Periksa kualitas embedding: coba encode beberapa dokumen dan kueri, ukur cosine similarity manual.
  - Periksa apakah model MiniLM yang dimuat benar (path pada `MODEL_PATH` di `src/inference_minilm_boosted.py`).
  - Periksa bahwa dokumen yang di-index ke Chroma sesuai (metadata & content tidak kosong).

- Jika reranker menurunkan performa:
  - Periksa apakah CrossEncoder yang digunakan dilatih/di-finetune untuk domain serupa.
  - Cek bobot hybrid (alpha/beta) dan threshold di `src/reranker.py`.

- Jika LLM memberikan jawaban yang 'hallucinate' (mengarang sumber):
  - Periksa bahwa prompt yang dibangun `build_prompt(...)` benar menyertakan konteks yang relevan dan bahwa konteks tidak dipotong secara agresif.
  - Pastikan RAGAgent menyertakan indikasi untuk model agar menjawab hanya berdasar konteks (lihat template prompt di `eksperimen/rag_agent_llama.py`).

7) Rujukan cepat (lokasi kode penting)

- Training MiniLM (simpel):
  - D:\3. ML\projectSkripsiSemantic\eksperimen\train_minilm.py

- Training MiniLM (boosted, seed experiments):
  - D:\3. ML\projectSkripsiSemantic\src\train_minilm_boosted.py
  - D:\3. ML\projectSkripsiSemantic\src\train_minilm_boosted_seed.py

- Inference/Example pemakaian MiniLM:
  - D:\3. ML\projectSkripsiSemantic\src\inference_minilm_boosted.py

- Reranker (Cross-Encoder):
  - D:\3. ML\projectSkripsiSemantic\src\reranker.py

- RAG agent & LLM backend calls:
  - D:\3. ML\projectSkripsiSemantic\eksperimen\rag_agent_llama.py

8) Kesimpulan singkat

Proyek ini mengikuti pola RAG klasik:
- Bi-encoder (MiniLM) untuk retrieval cepat via Chroma
- Cross-encoder reranker untuk meningkatkan presisi peringkat
- Prompt-building yang menggabungkan konteks teratas lalu diproses LLM untuk jawaban berbasis dokumen

Dokumentasi ini dibuat dengan membaca dan merangkum implementasi aktual di repository. Jika Anda ingin saya memperluas bagian tertentu (mis. contoh prompt lengkap, parameter hyper yang dipakai pada eksperimen spesifik, atau contoh perintah untuk menjalankan training/inference), beri tahu bagian mana yang mau diperdalam dan saya akan tambahkan tanpa menyentuh kode Anda.

---

9) FAQ & Jawaban untuk Pertanyaan Umum

**Q: Apakah query user di-encode dengan BI-ENCODER atau CROSS-ENCODER atau KEDUANYA?**

A: HANYA BI-ENCODER. Query user di-encode SATU KALI dengan BI-ENCODER (MiniLM) menjadi vektor di Tahap 1 untuk vector search di ChromaDB. Cross-Encoder TIDAK di-gunakan untuk encoding query terpisah.

Cross-Encoder hanya menerima PASANGAN (query_text, document_text) bersama-sama di Tahap 3. Input Cross-Encoder BUKAN vektor tetapi STRING TEKS dari query dan dokumen. Oleh karena itu, Cross-Encoder berbeda fungsi:
- BI-ENCODER: encode teks individual → vektor
- CROSS-ENCODER: score pasangan (query, doc) → skor relevansi

**Q: Jadi ada dua tahap pemrosesan pada query?**

A: Tidak, hanya SATU tahap encoding query (menggunakan BI-ENCODER di Tahap 1). Query TIDAK di-encode di Tahap 3 (reranking). Di Tahap 3, Cross-Encoder menerima TEKS query (string), bukan vektor.

Apa yang ada DUA adalah:
1. Tahap 1: Query di-encode dengan BI-ENCODER ke vektor untuk vector search.
2. Tahap 3: Query (sebagai teks) + kandidat dokumen (sebagai teks) dipass ke Cross-Encoder untuk mendapatkan skor relevansi.

Tapi keduanya memproses "query" dengan cara berbeda — yang pertama output vektor, yang kedua output skor.

**Q: Apakah dokumen juga di-encode pada Tahap 3 (reranking)?**

A: Tidak. Dokumen SUDAH di-encoded dan di-index ke ChromaDB sebelum fase inference. Saat inference (Tahap 3), dokumen yang sudah tersimpan di ChromaDB hanya diambil sebagai TEKS (string), bukan di-encode ulang.

Di Tahap 3, Cross-Encoder hanya menerima teks dokumen (sudah dari ChromaDB) + teks query (original dari user) dan memberikan skor.

**Q: Kemana saja alir data query user dari input sampai output?**

A: Ikuti urutan Tahap 0-7 di bagian "Alur data" section 3:
  - Tahap 0: User input teks query
  - Tahap 1: Cleaning + BI-ENCODER encoding ke vektor
  - Tahap 2: Vector search di ChromaDB → N kandidat
  - Tahap 3: Cross-ENCODER reranking → top-K terurut
  - Tahap 4: Context enrichment → full teks dokumen
  - Tahap 5: Prompt building → instruksi + konteks + query
  - Tahap 6: LLM inference → generate answer
  - Tahap 7: Assembly + output RAGResponse

Setiap tahap dijelaskan dengan detail: what, how, where (file kode), dan output apa yang dihasilkan.

**Q: Apa input dan output dari BI-ENCODER?**

A: - Input: string teks (query atau dokumen), panjang arbitrary.
   - Output: vektor numeric dengan dimensi tetap (384 dimensi untuk MiniLM paraphrase-multilingual).
   - Used in: Tahap 0 (indexing) dan Tahap 1 (inference) untuk vector search.
   - Code: eksperimen/rag_agent_llama.py baris 167: embedding_model.encode(cleaned_query).tolist()

**Q: Apa input dan output dari CROSS-ENCODER?**

A: - Input: LIST of pasangan [query_string, document_string] (bertipe list of lists).
   - Output: array skor numeric dengan panjang = jumlah pasangan.
   - Used in: Tahap 3 (reranking) untuk re-score kandidat.
   - Code: src/reranker.py baris 72-81:
     ```
     pair_inputs = [[query, f"Judul: {title}\nIsi: {doc}"] for doc, title in ...]
     raw_scores = self.model.predict(pair_inputs)        # CrossEncoder.predict
     rerank_scores = sigmoid(raw_scores)                 # normalisasi skor
     ```

**Q: Mengapa ada dua model (BI + CROSS) daripada satu?**

A: Karena perbedaan kecepatan dan presisi:
  - BI-ENCODER: CEPAT untuk vector search (encode sekali per query, cache vektor dokumen).
  - CROSS-ENCODER: LAMBAT tapi PRESISI untuk reranking (perlu proses setiap pasangan dengan cross-attention).

Pipeline ini adalah optimasi klasik: (1) cepat narrow down ke kandidat terbaik, (2) lambat tapi teliti urutkan top candidates.

**Q: Dari mana file model BI-ENCODER dan CROSS-ENCODER dimuat?**

A: - BI-ENCODER (MiniLM): Dimuat di `src/reranker.py` atau main.py saat inisialisasi. Default model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
     Training menyimpan ke folder `output/` (nama sesuai config eksperimen).
  - CROSS-ENCODER: Dimuat di `src/reranker.py` baris 50. Default path: output/crossencoder-base-model.
     Model ini diload sebagai singleton via get_reranker() function.

**Q: Bagaimana Chroma distance dikonversi ke similarity?**

A: Formula (lihat src/reranker.py baris 100):
   chroma_similarity = 1 / (1 + distance)
   
   Logika: 
   - Jika distance = 0 (identik) → similarity = 1 / (1 + 0) = 1.0
   - Jika distance = 1 (sangat berbeda) → similarity = 1 / (1 + 1) = 0.5
   - Jika distance → ∞ (ekstrem berbeda) → similarity → 0.0

Hasilnya: similarity_score ∈ (0, 1] yang bisa langsung dikombinasi dengan sigmoid(cross_encoder_score).

**Q: Penjelasan hybrid score 0.6 chroma + 0.4 reranker kenapa ratio ini?**

A: Dari kode `src/reranker.py` baris 95-96:
   alpha = 0.6
   beta = 0.4

Interpretasi:
- 0.6 = 60% kepercayaan pada vector similarity (Chroma).
- 0.4 = 40% kepercayaan pada cross-encoder score (Reranker).

Alasan: Chroma sudah melakukan pre-filtering dengan vector similarity (sering reliable), jadi diberi bobot lebih. Reranker digunakan untuk fine-tuning ranking, jadi bobot lebih kecil.

Nilai ini bisa di-tune sesuai performa eksperimen (lihat config/benchmark di project untuk parameter lainnya).

---

10) Contoh Walkthrough Spesifik: User Query → Response

Untuk memahami alur secara KONKRET, berikut adalah contoh step-by-step dengan data dummy (sesuai implementasi project):

INPUT USER:
  user_question = "carikan apa saja syarat masuk UNIDA?"

TAHAP 0 - Penerimaan Input:
  Sistem terima: user_question = "carikan apa saja syarat masuk UNIDA?"

TAHAP 1a - Query Cleaning:
  Fungsi: clean_user_query() di eksperimen/rag_agent_llama.py baris 145-148
  Input: "carikan apa saja syarat masuk UNIDA?"
  Proses: hapus prefix "carikan "
  Output: cleaned_query = "apa saja syarat masuk UNIDA?"

TAHAP 1b - BI-ENCODER Encoding:
  Fungsi: embedding_model.encode() di eksperimen/rag_agent_llama.py baris 167
  Model: MiniLM (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
  Input string: "apa saja syarat masuk UNIDA?"
  Proses: tokenize + pass melalui Transformer + Pooling
  Output vector: [0.234, -0.156, 0.789, ..., -0.432] (dimensi 384)
  Waktu perkiraan: 5-50ms (tergantung hardware)

TAHAP 2 - Vector Search di ChromaDB:
  Fungsi: chroma_collection.query() di eksperimen/rag_agent_llama.py baris 169-173
  Input: query_embeddings=[vektor dari Tahap 1b]
         n_results=candidate_count (20-40 dokumen)
  Proses: cari N dokumen terdekat ke vektor query menggunakan cosine/L2 distance
  ChromaDB return: {
    'ids': [['doc_001', 'doc_003', 'doc_025', ...]],          # 20-40 doc IDs
    'documents': [['Syarat Masuk Lengkap...', 'Dokumen Pendaftaran...', ...]],
    'metadatas': [[{
      'title': 'Persyaratan Masuk Mahasiswa Baru',
      'snippet': 'Calon mahasiswa UNIDA harus memiliki ...',
      'file_name': 'arsip_001.pdf',
      ...
    }, {...}, ...]],
    'distances': [[0.123, 0.245, 0.189, ...]]  # distance dari vektor query
  }
  Hasil contoh TOP 3 dari 20-40:
    1. doc_001: distance=0.123, title="Persyaratan Masuk Mahasiswa Baru"
    2. doc_003: distance=0.189, title="Dokumen Pendaftaran dan Syarat Akademik"
    3. doc_025: distance=0.245, title="Prosedur Admisi UNIDA Gontor"
  Waktu perkiraan: 10-100ms (tergantung ukuran ChromaDB)

TAHAP 3 - Cross-Encoder Reranking:
  Fungsi: reranker.rerank() di src/reranker.py baris 59
  
  Sub-tahap 3a - Build Pair Inputs:
    Untuk setiap 20-40 dokumen dari Tahap 2, buat pasangan teks:
    pair_inputs = [
      ["apa saja syarat masuk UNIDA?", "Judul: Persyaratan Masuk Mahasiswa Baru\nIsi: Syarat Masuk Lengkap..."],
      ["apa saja syarat masuk UNIDA?", "Judul: Dokumen Pendaftaran dan Syarat Akademik\nIsi: Dokumen Pendaftaran..."],
      ["apa saja syarat masuk UNIDA?", "Judul: Prosedur Admisi UNIDA Gontor\nIsi: Prosedur Admisi..."],
      ... (20-40 pasangan total)
    ]
  
  Sub-tahap 3b - Cross-Encoder Predict:
    Model: CrossEncoder (dimuat dari output/crossencoder-base-model)
    Input: pair_inputs list (20-40 pasangan teks)
    Proses: encode setiap pasangan dengan cross-attention → hasilkan skor
    Output raw_scores: [2.15, -1.30, 0.98, ...] (20-40 nilai)
    
  Sub-tahap 3c - Sigmoid Normalisasi:
    raw_scores: [2.15, -1.30, 0.98, ...]
    sigmoid(x) = 1 / (1 + exp(-x))
    rerank_scores: [sigmoid(2.15), sigmoid(-1.30), sigmoid(0.98), ...]
                  ≈ [0.896, 0.214, 0.728, ...]
    
  Sub-tahap 3d - Hybrid Scoring:
    Untuk setiap dokumen i:
      distance[i] = [0.123, 0.189, 0.245, ...]
      chroma_similarity[i] = 1 / (1 + distance[i])
                           = [0.891, 0.841, 0.803, ...]  (1/(1+0.123), 1/(1+0.189), ...)
      
      reranker_score[i] = rerank_scores[i]
                        = [0.896, 0.214, 0.728, ...]
      
      final_hybrid_score[i] = 0.6 * chroma_similarity[i] + 0.4 * reranker_score[i]
      
      Contoh doc pertama:
        = 0.6 * 0.891 + 0.4 * 0.896
        = 0.535 + 0.358
        = 0.893
      
      Contoh doc kedua:
        = 0.6 * 0.841 + 0.4 * 0.214
        = 0.505 + 0.086
        = 0.591
      
      Contoh doc ketiga:
        = 0.6 * 0.803 + 0.4 * 0.728
        = 0.482 + 0.291
        = 0.773
  
  Sub-tahap 3e - Keyword Penalty & Filter:
    Ekstrak keywords dari "apa saja syarat masuk UNIDA?" (stopword removal):
      keywords = ["syarat", "masuk", "unida"]  (stopwords seperti "apa", "saja" dihapus)
    
    Untuk setiap dokumen, cek overlap keywords:
      doc1: title + text = "Persyaratan Masuk... Syarat..." 
            → cocok: "syarat" (fuzzy match), "masuk" → 2/3 = 66.7% match
            → match_ratio >= 0.25 → NO penalty
      
      doc2: title + text = "Dokumen Pendaftaran..."
            → cocok: 0/3 = 0% match
            → match_ratio < 0.25 → apply penalty
            penalty = 0.10 * (1.0 - 0.0) = 0.10
            final_hybrid_score[2] -= 0.10 → 0.591 - 0.10 = 0.491
    
    Filter threshold (MIN_SCORE_THRESHOLD = 0.20):
      doc1: 0.893 >= 0.20 ✓ LOLOS
      doc2: 0.491 >= 0.20 ✓ LOLOS
      doc3: 0.773 >= 0.20 ✓ LOLOS
  
  Sub-tahap 3f - Sort & Top-K:
    Sort by final_hybrid_score descending:
      1. doc1: 0.893
      2. doc3: 0.773
      3. doc2: 0.491
      ... (remaining docs if any)
    
    Ambil top_k=10 (atau sesuai config) → hasil top 3 dokumen terurut.
  
  Output reranking (top-K):
    [
      {"id": "doc_001", "title": "Persyaratan Masuk Mahasiswa Baru",         "score": 0.893, ...},
      {"id": "doc_025", "title": "Prosedur Admisi UNIDA Gontor",             "score": 0.773, ...},
      {"id": "doc_003", "title": "Dokumen Pendaftaran dan Syarat Akademik", "score": 0.491, ...}
    ]
  
  Waktu perkiraan: 100-500ms (tergantung jumlah dokumen dan ukuran cross-encoder)

TAHAP 4 - Context Enrichment:
  Fungsi: get_context_for_results() di src/document_reader.py
  Input: top-3 dokumen dari Tahap 3 + query
  Proses: ambil full_docs_map (teks lengkap dokumen dari ChromaDB) dan enrich dengan konteks pendukung
  Output enriched_docs: [
    {
      "id": "doc_001",
      "title": "Persyaratan Masuk Mahasiswa Baru",
      "full_context": "Berikut adalah persyaratan lengkap....\nPersyaratan Akademik: ....\nPersyaratan Administratif: ....\n...(teks lengkap)...",
      "score": 0.893,
      "file_name": "arsip_001.pdf"
    },
    ... (doc 3 + doc 25)
  ]
  Waktu perkiraan: 5-50ms (I/O pembacaan teks)

TAHAP 5 - Prompt Building:
  Fungsi: build_prompt() di eksperimen/rag_agent_llama.py baris 278
  Input: question="carikan apa saja syarat masuk UNIDA?", enriched_docs=[3 dokumen]
  Proses: buat template prompt dengan instruksi ketat + konteks + pertanyaan
  Output prompt STRING:
    """
    Anda adalah asisten informasi akademik UNIDA Gontor yang sangat teliti.
    Tugas Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan teks konteks yang disediakan.

    [ATURAN KETAT]
    1. Pencocokan Semantik: ...
    2. Penalaran Temporal: ...
    [... 8 aturan terstruktur ...]

    [KONTEKS ARSIP]
    --- KUTIPAN 1 ---
    Berikut adalah persyaratan lengkap untuk masuk UNIDA Gontor:
    Persyaratan Akademik:
    - Lulusan SMA/sederajat atau MA/Pesantren
    - Nilai rata-rata minimal 70
    Persyaratan Administratif:
    - Fotokopi ijazah
    - Sertifikat kesehatan
    [... full context doc1 ...]

    --- KUTIPAN 2 ---
    [... full context doc2 ...]

    --- KUTIPAN 3 ---
    [... full context doc3 ...]

    [PERTANYAAN PENGGUNA]
    carikan apa saja syarat masuk UNIDA?

    [JAWABAN LANGSUNG]:
    """
  
  Total prompt token count: ~2000-3000 token (tergantung panjang dokumen)
  Waktu perkiraan: <5ms (string concatenation)

TAHAP 6 - LLM Inference:
  Fungsi: call_llm() → call_ollama() atau call_vllm() 
          di eksperimen/rag_agent_llama.py baris 474-476
  Input: prompt dari Tahap 5
  Model: Ollama (llama3.2:3b) atau vLLM (meta-llama/Llama-3.2-3B-Instruct)
  Proses: 
    1. POST request ke LLM backend dengan prompt
    2. LLM membaca instruksi, memahami aturan ketat
    3. LLM mereferensi konteks (3 dokumen) dan pertanyaan
    4. LLM generate jawaban token by token
  Output string (contoh):
    """
    Syarat-syarat masuk UNIDA Gontor adalah sebagai berikut:

    Persyaratan Akademik:
    1. Lulusan SMA/sederajat atau MA/Pesantren
    2. Nilai rata-rata minimal 70

    Persyaratan Administratif:
    1. Fotokopi ijazah
    2. Sertifikat kesehatan
    3. Formulir pendaftaran yang telah diisi lengkap

    Selain itu, calon mahasiswa juga harus mengikuti proses seleksi yang mencakup tes tertulis dan wawancara.
    """
  
  Waktu perkiraan: 3-15 detik (tergantung kecepatan LLM, suhu, model)

TAHAP 7 - Response Assembly:
  Fungsi: RAGAgent.answer() return RAGResponse di eksperimen/rag_agent_llama.py baris 492-502
  Input: hasil dari Tahap 1-6
  Proses: kumpulkan jawaban + sumber + metadata + timing
  
  Output RAGResponse object:
    {
      "question": "carikan apa saja syarat masuk UNIDA?",
      "answer": "Syarat-syarat masuk UNIDA Gontor...",
      "sources": [
        {
          "title": "Persyaratan Masuk Mahasiswa Baru",
          "score": 0.893,
          "file_name": "arsip_001.pdf",
          "download_url": "http://localhost:8000/files/arsip_001.pdf",
          "context_chars": 1256
        },
        {
          "title": "Dokumen Pendaftaran dan Syarat Akademik",
          "score": 0.491,
          "file_name": "arsip_003.pdf",
          ...
        },
        ... (doc 3)
      ],
      "search_results_count": 3,
      "context_chars_total": 3512,
      "latency": 12.34,  # total end-to-end
      "search_time": 0.075,      # Tahap 1-2
      "rerank_time": 0.245,      # Tahap 3
      "llm_time": 11.82,         # Tahap 6
      "error": null
    }

RESPONSE SENT TO USER:
  💬 Jawaban:
  Syarat-syarat masuk UNIDA Gontor adalah sebagai berikut:
  [... full answer text from LLM ...]

  📊 Statistik:
  Total Waktu (End-to-End) : 12.34 detik
    - Vector Search        : 0.075 detik
    - Reranking            : 0.245 detik
    - LLM Inference        : 11.82 detik
  Dokumen dibaca           : 3
  Total konteks            : 3512 karakter

  📚 Sumber:
  1. Persyaratan Masuk Mahasiswa Baru (skor: 0.89)
  2. Dokumen Pendaftaran dan Syarat Akademik (skor: 0.49)
  3. Prosedur Admisi UNIDA Gontor (skor: 0.77)

SUMMARY OF FLOW:
  User Input → Clean Query → BI-ENC Encode → Vector Search → Get Candidates
         ↓
  Cross-ENC Rerank → Enrich Context → Build Prompt → LLM Generate → Response
  
  Waktu breakdown:
  - Vector Search (Tahap 1-2): ~75ms (cepat, cached embeddings)
  - Reranking (Tahap 3): ~245ms (lebih lambat, cross-attention)
  - LLM (Tahap 6): ~11.8 detik (paling lambat, token generation)
  Total: ~12.3 detik (mostly bottleneck ada di LLM generasi)

