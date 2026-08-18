import os
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

# --- KONFIGURASI ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# UBAH 1: Ganti path lokal menjadi nama model pre-trained asli di Hugging Face
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DATA_FILE = os.path.join(ROOT, "data", "indodoc", "test_new.jsonl")


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
                    relevant_docs[qid] = {did}

            except json.JSONDecodeError:
                continue

    print(f"[*] Loaded {len(queries)} queries and {len(corpus)} documents.")
    return queries, corpus, relevant_docs


def run_evaluation():
    # 1. Load Data
    queries, corpus, relevant_docs = load_test_data_for_ir(DATA_FILE)
    if not queries: return

    # 2. Load Model
    # UBAH 2: Menghapus pengecekan os.path.exists karena model akan diunduh/dimuat langsung dari cache Hugging Face
    print(f"[*] Loading model PRE-TRAINED dasar dari: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 3. Setup Evaluator
    print("[*] Menyiapkan evaluator...")
    # UBAH 3: Ganti nama evaluator agar hasil JSON-nya rapi
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="Pretrained-Evaluation",
        show_progress_bar=True,
        corpus_chunk_size=50000,
        batch_size=16
    )

    # 4. Jalankan Evaluasi
    print("[*] Memulai evaluasi MiniLM PRE-TRAINED...")
    results = evaluator(model)

    # 5. Tampilkan Hasil
    print("\n" + "=" * 40)
    print(f" HASIL EVALUASI MINILM (PRE-TRAINED)")
    print("=" * 40)

    # UBAH 4: Sesuaikan key dengan nama evaluator baru
    keys_to_check = {
        "MRR@10": ["Pretrained-Evaluation_cosine_mrr@10", "test_cosine_mrr@10", "mrr@10"],
        "NDCG@10": ["Pretrained-Evaluation_cosine_ndcg@10", "test_cosine_ndcg@10", "ndcg@10"],
        "Recall@1": ["Pretrained-Evaluation_cosine_recall@1", "test_cosine_recall@1", "recall@1"],
        "Recall@5": ["Pretrained-Evaluation_cosine_recall@5", "test_cosine_recall@5", "recall@5"],
    }

    for label, possible_keys in keys_to_check.items():
        value = 0.0
        for key in possible_keys:
            if key in results:
                value = results[key]
                break
        print(f"{label:<12} : {value:.4f}")

    print("=" * 40)

    # Simpan hasil
    output_json = os.path.join(ROOT, "output", "evaluation_MpNet_pretrained.json")

    # Pastikan folder output ada
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w") as f:
        clean_results = {k: float(v) for k, v in results.items()}
        json.dump(clean_results, f, indent=4)

    print(f"[*] Laporan Pre-Trained disimpan di: {output_json}")


if __name__ == "__main__":
    run_evaluation()