import os
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

# --- KONFIGURASI PATH ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model hasil training SERANAH
MODEL_PATH = os.path.join(ROOT, "output", "minilm-dokumen-arsip-boosted-seranah-seed-42")

# File Test Evaluasi SERANAH
DATA_FILE = os.path.join(ROOT, "data", "indodoc", "test_new_seranah.jsonl")


def load_test_data_for_ir(filename):
    """Memuat data uji kueri dan korpus dokumen dari file JSONL untuk Information Retrieval Evaluator."""
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
                    relevant_docs[qid] = {did}

            except json.JSONDecodeError:
                continue

    print(f"[*] Loaded {len(queries)} queries dan {len(corpus)} documents.")
    return queries, corpus, relevant_docs


def run_evaluation():
    print("=" * 60)
    print("[*] MEMULAI EVALUASI INFORMATION RETRIEVAL (MINILM BOOSTED SERANAH)")
    print("=" * 60)

    # 1. Load Data
    queries, corpus, relevant_docs = load_test_data_for_ir(DATA_FILE)
    if not queries:
        print("[!] Gagal memuat data evaluasi. Pastikan generate_test_data.py sudah dijalankan.")
        return

    # 2. Load Model
    print(f"[*] Loading model BOOSTED SERANAH dari: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Folder model tidak ditemukan di {MODEL_PATH}")
        print(f"[!] Silakan jalankan src/train_minilm_boosted_seranah.py terlebih dahulu!")
        return

    model = SentenceTransformer(MODEL_PATH)

    # 3. Setup Evaluator
    print("[*] Menyiapkan InformationRetrievalEvaluator...")
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="Boosted-Seranah-Evaluation",
        show_progress_bar=True,
        corpus_chunk_size=50000,
        batch_size=16,
    )

    # 4. Jalankan Evaluasi
    print("[*] Memulai evaluasi performa retrieval MiniLM Boosted...")
    results = evaluator(model)

    # 5. Tampilkan Hasil
    print("\n" + "=" * 45)
    print("  HASIL EVALUASI MINILM BOOSTED (SERANAH)")
    print("=" * 45)

    keys_to_check = {
        "MRR@10": ["Boosted-Seranah-Evaluation_cosine_mrr@10", "test_cosine_mrr@10", "mrr@10"],
        "NDCG@10": ["Boosted-Seranah-Evaluation_cosine_ndcg@10", "test_cosine_ndcg@10", "ndcg@10"],
        "Recall@1": ["Boosted-Seranah-Evaluation_cosine_recall@1", "test_cosine_recall@1", "recall@1"],
        "Recall@5": ["Boosted-Seranah-Evaluation_cosine_recall@5", "test_cosine_recall@5", "recall@5"],
    }

    for label, possible_keys in keys_to_check.items():
        value = 0.0
        for key in possible_keys:
            if key in results:
                value = results[key]
                break
        print(f"{label:<12} : {value:.4f}")

    print("=" * 45)

    # 6. Simpan Hasil Evaluasi ke File JSON
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    output_json = os.path.join(ROOT, "output", "evaluation_minilm_boosted_seranah_results.json")
    with open(output_json, "w", encoding="utf-8") as f:
        clean_results = {k: float(v) for k, v in results.items()}
        json.dump(clean_results, f, indent=4)

    print(f"[✅] Laporan Hasil Evaluasi disimpan di: {output_json}\n")


if __name__ == "__main__":
    run_evaluation()