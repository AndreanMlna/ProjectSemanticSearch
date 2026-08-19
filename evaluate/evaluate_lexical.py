import os
import json
import math
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURASI ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_OUTPUT_DIR = os.path.join(ROOT, "output")
DATA_FILE = os.path.join(ROOT, "data", "indodoc", "test_new.jsonl")

# Pastikan folder output ada
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


def load_test_data_for_ir(filename):
    if not os.path.exists(filename):
        print(f"[!] File {filename} tidak ditemukan.")
        return None, None, None

    queries = {}
    corpus = {}
    relevant_docs = {}

    print(f"[*] Loading data evaluasi dari {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
                qid = str(idx)
                did = str(idx)

                if "title" in data and "content" in data:
                    q_text = data["title"].strip()
                    d_text = data["content"].strip()

                    if not q_text or not d_text:
                        continue

                    queries[qid] = q_text
                    corpus[did] = d_text
                    # Asumsi: Setiap baris (kueri) hanya relevan dengan kontennya sendiri (1 to 1)
                    relevant_docs[qid] = {did}

            except json.JSONDecodeError:
                continue

    print(f"[*] Loaded {len(queries)} queries and {len(corpus)} documents.")
    return queries, corpus, relevant_docs


def compute_metrics(ranked_lists, relevant_docs):
    """
    Fungsi manual untuk menghitung metrik IR secara lengkap.
    Mencakup: Accuracy, Precision, Recall (@1, 3, 5, 10), NDCG@10, MRR@10, MAP@100.
    """
    k_values = [1, 3, 5, 10]

    # Inisialisasi dictionary metrik
    metrics = {
        "ndcg@10": 0.0,
        "mrr@10": 0.0,
        "map@100": 0.0
    }

    for k in k_values:
        metrics[f"accuracy@{k}"] = 0.0
        metrics[f"precision@{k}"] = 0.0
        metrics[f"recall@{k}"] = 0.0

    num_queries = len(ranked_lists)

    for qid, top_dids in ranked_lists.items():
        rel_docs = relevant_docs[qid]
        rank = None

        # Cari di ranking berapa dokumen yang relevan muncul
        for i, did in enumerate(top_dids):
            if did in rel_docs:
                rank = i + 1
                break

        # Hitung skor berdasarkan ranking
        if rank is not None:
            # 1. Metrik dengan K (Accuracy, Precision, Recall)
            for k in k_values:
                if rank <= k:
                    metrics[f"accuracy@{k}"] += 1.0
                    metrics[f"precision@{k}"] += (1.0 / k)
                    metrics[f"recall@{k}"] += 1.0  # Sama dengan accuracy karena hanya ada 1 relevan doc

            # 2. Metrik spesifik 10 (MRR, NDCG)
            if rank <= 10:
                metrics["mrr@10"] += (1.0 / rank)
                metrics["ndcg@10"] += (1.0 / math.log2(rank + 1))

            # 3. Metrik spesifik 100 (MAP)
            if rank <= 100:
                metrics["map@100"] += (1.0 / rank)

    # Rata-ratakan (Average) semua kueri
    for k in metrics:
        metrics[k] /= num_queries

    return metrics


def evaluate_tfidf(queries, corpus, relevant_docs):
    print("\n[*] Menjalankan Evaluasi TF-IDF (Term Frequency-Inverse Document Frequency)...")
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[did] for did in corpus_ids]

    # Inisialisasi dan latih TF-IDF pada corpus
    vectorizer = TfidfVectorizer(lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(corpus_texts)

    ranked_lists = {}

    for qid, q_text in queries.items():
        q_vec = vectorizer.transform([q_text])
        # Hitung Cosine Similarity antara Kueri dan semua Dokumen
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

        # Ambil Top 100 (Bukan lagi 10, agar MAP@100 bisa dihitung)
        top_100_idx = np.argsort(scores)[-100:][::-1]
        ranked_lists[qid] = [corpus_ids[i] for i in top_100_idx]

    return compute_metrics(ranked_lists, relevant_docs)


def evaluate_bm25(queries, corpus, relevant_docs):
    print("\n[*] Menjalankan Evaluasi BM25 (Best Matching 25)...")
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[did] for did in corpus_ids]

    # Tokenisasi corpus (mengubah jadi huruf kecil dan memisahkan per spasi)
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    ranked_lists = {}

    for qid, q_text in queries.items():
        tokenized_query = q_text.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Ambil Top 100
        top_100_idx = np.argsort(scores)[-100:][::-1]
        ranked_lists[qid] = [corpus_ids[i] for i in top_100_idx]

    return compute_metrics(ranked_lists, relevant_docs)


def run_lexical_evaluation():
    # 1. Load Data
    queries, corpus, relevant_docs = load_test_data_for_ir(DATA_FILE)
    if not queries: return

    print(f"\n{'=' * 50}")
    print(f"📊 EVALUASI MODEL PENCARIAN TRADISIONAL (LEKSIKAL)")
    print(f"{'=' * 50}")

    # Kumpulan urutan metrik agar dicetak dengan rapi sesuai contoh SentenceTransformers
    keys_to_check = [
        "accuracy@1", "accuracy@3", "accuracy@5", "accuracy@10",
        "precision@1", "precision@3", "precision@5", "precision@10",
        "recall@1", "recall@3", "recall@5", "recall@10",
        "ndcg@10", "mrr@10", "map@100"
    ]

    # ==========================================
    # --- 1. Evaluasi TF-IDF ---
    # ==========================================
    tfidf_results = evaluate_tfidf(queries, corpus, relevant_docs)

    print("\n" + "=" * 40)
    print(" HASIL EVALUASI TF-IDF")
    print("=" * 40)

    for label in keys_to_check:
        value = tfidf_results.get(label, 0.0)
        print(f"{label:<15} : {value:.4f}")

    print("=" * 40)

    output_json_tfidf = os.path.join(ROOT, "output", "evaluation_tfidf_results.json")
    with open(output_json_tfidf, "w") as f:
        # Dictionary comprehension dengan urutan yang sama
        clean_results = {k: float(tfidf_results.get(k, 0.0)) for k in keys_to_check}
        json.dump(clean_results, f, indent=4)

    print(f"[*] Laporan TF-IDF disimpan di: {output_json_tfidf}")

    # ==========================================
    # --- 2. Evaluasi BM25 ---
    # ==========================================
    bm25_results = evaluate_bm25(queries, corpus, relevant_docs)

    print("\n" + "=" * 40)
    print(" HASIL EVALUASI BM25")
    print("=" * 40)

    for label in keys_to_check:
        value = bm25_results.get(label, 0.0)
        print(f"{label:<15} : {value:.4f}")

    print("=" * 40)

    output_json_bm25 = os.path.join(ROOT, "output", "evaluation_bm25_results.json")
    with open(output_json_bm25, "w") as f:
        # Dictionary comprehension dengan urutan yang sama
        clean_results = {k: float(bm25_results.get(k, 0.0)) for k in keys_to_check}
        json.dump(clean_results, f, indent=4)

    print(f"[*] Laporan BM25 disimpan di: {output_json_bm25}")

    # ==========================================
    # --- PENUTUP ---
    # ==========================================
    print(f"\n{'=' * 50}")
    print("🎉 EVALUASI TF-IDF & BM25 SELESAI!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    run_lexical_evaluation()