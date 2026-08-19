import os
import json
import math
import torch
import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- KONFIGURASI PATH ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_OUTPUT_DIR = os.path.join(ROOT, "output")
DATA_FILE = os.path.join(ROOT, "data", "indodoc", "test_new.jsonl")


BI_ENCODER_PATH = os.path.join(ROOT, "output", "minilm-dokumen-arsip-boosted-new-seed-42")

CROSS_ENCODER_NAME = os.getenv("CE_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
MIN_SCORE_THRESHOLD = 0.20

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


# --- FUNGSI UTILITAS DARI RERANKER ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fuzzy_keyword_match(keyword: str, text_to_check: str, threshold: float = 0.75) -> bool:
    if keyword in text_to_check:
        return True

    words_in_text = text_to_check.split()
    for word in words_in_text:
        clean_word = word.strip("?,.!\"()'")
        if len(clean_word) > 3:
            if SequenceMatcher(None, keyword, clean_word).ratio() >= threshold:
                return True
    return False


def load_test_data_for_ir(filename):
    if not os.path.exists(filename):
        print(f"[!] File {filename} tidak ditemukan.")
        return None, None, None, None

    queries = {}
    corpus = {}
    titles = {}
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
                    titles[did] = q_text  # Menyimpan judul untuk format Reranker
                    relevant_docs[qid] = {did}

            except json.JSONDecodeError:
                continue

    print(f"[*] Loaded {len(queries)} queries and {len(corpus)} documents.")
    return queries, corpus, relevant_docs, titles


def compute_metrics(ranked_lists, relevant_docs):
    """
    Fungsi manual untuk menghitung metrik IR secara lengkap.
    """
    k_values = [1, 3, 5, 10]
    metrics = {"ndcg@10": 0.0, "mrr@10": 0.0, "map@100": 0.0}

    for k in k_values:
        metrics[f"accuracy@{k}"] = 0.0
        metrics[f"precision@{k}"] = 0.0
        metrics[f"recall@{k}"] = 0.0

    num_queries = len(ranked_lists)

    for qid, top_dids in ranked_lists.items():
        rel_docs = relevant_docs[qid]
        rank = None

        for i, did in enumerate(top_dids):
            if did in rel_docs:
                rank = i + 1
                break

        if rank is not None:
            for k in k_values:
                if rank <= k:
                    metrics[f"accuracy@{k}"] += 1.0
                    metrics[f"precision@{k}"] += (1.0 / k)
                    metrics[f"recall@{k}"] += 1.0

            if rank <= 10:
                metrics["mrr@10"] += (1.0 / rank)
                metrics["ndcg@10"] += (1.0 / math.log2(rank + 1))

            if rank <= 100:
                metrics["map@100"] += (1.0 / rank)

    for k in metrics:
        metrics[k] /= num_queries

    return metrics


def run_cross_encoder_evaluation():
    # 1. Load Data
    queries, corpus, relevant_docs, corpus_titles = load_test_data_for_ir(DATA_FILE)
    if not queries: return

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[did] for did in corpus_ids]

    print(f"\n{'=' * 50}")
    print(f"📊 EVALUASI HYBRID: BI-ENCODER + CROSS-ENCODER RERANKING")
    print(f"{'=' * 50}")

    # 2. Load Models & Sastrawi
    print(f"[*] Loading Bi-Encoder: {BI_ENCODER_PATH}")
    bi_encoder = SentenceTransformer(BI_ENCODER_PATH)

    print(f"[*] Loading Cross-Encoder: {CROSS_ENCODER_NAME}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

    print("[*] Inisialisasi Sastrawi Stopword Remover...")
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()

    # 3. Proses Corpus dengan Bi-Encoder (Tahap 1: Filtering)
    print("[*] Melakukan encoding seluruh corpus menggunakan Bi-Encoder...")
    corpus_embeddings = bi_encoder.encode(corpus_texts, convert_to_tensor=True)

    ranked_lists = {}
    print("[*] Memproses kueri dan melakukan Reranking dengan Hybrid Scoring (Mohon tunggu)...")

    # 4. Evaluasi per Kueri
    for qid, q_text in queries.items():
        # A. Tahap Retrieval (Bi-Encoder)
        query_embedding = bi_encoder.encode(q_text, convert_to_tensor=True)
        # Ambil Top 100 kandidat awal
        hits = cos_sim(query_embedding, corpus_embeddings)[0]
        top_100_tensor = torch.topk(hits, k=min(100, len(corpus_texts)))
        top_100_idx = top_100_tensor.indices.tolist()
        top_100_sims = top_100_tensor.values.tolist()

        # B. Siapkan Input untuk Cross-Encoder
        pair_inputs = []
        for idx in top_100_idx:
            did = corpus_ids[idx]
            title = corpus_titles[did]
            doc_text = corpus_texts[idx]
            # Format persis seperti di reranker.py
            full_context_to_score = f"Judul: {title}\nIsi: {doc_text}"
            pair_inputs.append([q_text, full_context_to_score])

        # C. Hitung Skor Cross Encoder & Sigmoid
        raw_scores = cross_encoder.predict(pair_inputs)
        rerank_scores = sigmoid(raw_scores)

        # D. Proses Ekstraksi Kata Kunci dengan Sastrawi
        cleaned_query = stopword_remover.remove(q_text.lower())
        query_keywords = list(set([
            w.strip("?,.!\"()'") for w in cleaned_query.split()
            if len(w.strip("?,.!\"()'")) > 3
        ]))

        # E. Hitung Hybrid Score dan Lexical Penalty
        reranked_list = []
        alpha = 0.6
        beta = 0.4

        for i, idx in enumerate(top_100_idx):
            did = corpus_ids[idx]
            title = corpus_titles[did]
            full_text = corpus_texts[idx]

            # Simulasi jarak ChromaDB (cosine space = 1 - cosine_sim)
            dist = 1.0 - top_100_sims[i]
            chroma_similarity = 1 / (1 + dist)

            reranker_score = float(rerank_scores[i])
            final_hybrid_score = (alpha * chroma_similarity) + (beta * reranker_score)

            # Terapkan Penalty
            if query_keywords:
                text_to_check = f"{title} {full_text}".lower()
                matched_words = sum(
                    1 for kw in query_keywords if fuzzy_keyword_match(kw, text_to_check, threshold=0.75))
                match_ratio = matched_words / len(query_keywords)

                if match_ratio < 0.25:
                    penalty = 0.10 * (1.0 - match_ratio)
                    final_hybrid_score -= penalty
                    final_hybrid_score = max(0.0, final_hybrid_score)

            # Threshold Filter
            if final_hybrid_score >= MIN_SCORE_THRESHOLD:
                reranked_list.append((idx, final_hybrid_score))

        # F. Urutkan ulang indeks berdasarkan skor hybrid akhir
        reranked_pairs = sorted(reranked_list, key=lambda x: x[1], reverse=True)
        ranked_lists[qid] = [corpus_ids[idx] for idx, score in reranked_pairs]

    # 5. Hitung Metrik Evaluasi
    results = compute_metrics(ranked_lists, relevant_docs)

    keys_to_check = [
        "accuracy@1", "accuracy@3", "accuracy@5", "accuracy@10",
        "precision@1", "precision@3", "precision@5", "precision@10",
        "recall@1", "recall@3", "recall@5", "recall@10",
        "ndcg@10", "mrr@10", "map@100"
    ]

    print("\n" + "=" * 40)
    print(" HASIL EVALUASI RERANKING (HYBRID SCORING)")
    print("=" * 40)

    for label in keys_to_check:
        value = results.get(label, 0.0)
        print(f"{label:<15} : {value:.4f}")

    print("=" * 40)

    # 6. Simpan Hasil
    output_json = os.path.join(BASE_OUTPUT_DIR, "evaluation_cross_bi_encoder_reranking_fix.json")
    with open(output_json, "w") as f:
        clean_results = {k: float(results.get(k, 0.0)) for k in keys_to_check}
        json.dump(clean_results, f, indent=4)

    print(f"[*] Laporan Hybrid Reranking disimpan di: {output_json}")


if __name__ == "__main__":
    run_cross_encoder_evaluation()