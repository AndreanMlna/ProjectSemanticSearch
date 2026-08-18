import os
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASE_OUTPUT_DIR = os.path.join(ROOT, "output")
DATA_FILE = os.path.join(ROOT, "data", "indodoc", "test.jsonl")


SEEDS = [42, 123, 456, 789, 1024]


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
    # 1. Load Data (Hanya dilakukan sekali untuk semua model)
    queries, corpus, relevant_docs = load_test_data_for_ir(DATA_FILE)
    if not queries: return

    print(f"\n[INFO] Memulai Evaluasi untuk {len(SEEDS)} Model Seed: {SEEDS}")

    for seed in SEEDS:
        print(f"\n{'=' * 50}")
        print(f"📊 EVALUASI MODEL SEED: {seed}")
        print(f"{'=' * 50}")

        model_name = f"minilm-dokumen-arsip-boosted-new-seed-{seed}"
        model_path = os.path.join(BASE_OUTPUT_DIR, model_name)

        # 2. Load Model
        if not os.path.exists(model_path):
            print(f"[!] Folder model tidak ditemukan di {model_path}. Lewati seed {seed}...")
            continue

        print(f"[*] Loading model dari: {model_path}")
        model = SentenceTransformer(model_path)

        # 3. Setup Evaluator
        print("[*] Menyiapkan evaluator...")
        evaluator_name = f"Boosted-Eval-Seed-{seed}"
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=evaluator_name,
            show_progress_bar=True,
            corpus_chunk_size=50000,
            batch_size=16
        )

        # 4. Jalankan Evaluasi
        print(f"[*] Memulai evaluasi MiniLM BOOSTED (Seed {seed})...")
        results = evaluator(model)

        # 5. Tampilkan Hasil
        print("\n" + "-" * 40)
        print(f" HASIL EVALUASI MINILM (SEED {seed})")
        print("-" * 40)

        keys_to_check = {
            "MRR@10": [f"{evaluator_name}_cosine_mrr@10", "cosine_mrr@10", "mrr@10"],
            "NDCG@10": [f"{evaluator_name}_cosine_ndcg@10", "cosine_ndcg@10", "ndcg@10"],
            "Recall@1": [f"{evaluator_name}_cosine_recall@1", "cosine_recall@1", "recall@1"],
            "Recall@5": [f"{evaluator_name}_cosine_recall@5", "cosine_recall@5", "recall@5"],
        }

        for label, possible_keys in keys_to_check.items():
            value = 0.0
            for key in possible_keys:
                if key in results:
                    value = results[key]
                    break
            print(f"{label:<12} : {value:.4f}")

        print("-" * 40)

        # Simpan hasil
        output_json_name = f"evaluation_minilm_boosted_seed_{seed}.json"
        output_json = os.path.join(BASE_OUTPUT_DIR, output_json_name)

        with open(output_json, "w") as f:
            # Menyimpan hasil dengan menambahkan informasi seed di dalamnya
            clean_results = {k: float(v) for k, v in results.items()}
            clean_results["seed_dipakai"] = seed
            json.dump(clean_results, f, indent=4)

        print(f"[*] Laporan Seed {seed} disimpan di: {output_json}")

    print(f"\n{'=' * 50}")
    print(f"🎉 SEMUA {len(SEEDS)} EVALUASI SELESAI!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    run_evaluation()