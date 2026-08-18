import os
import json
import random


random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_MINILM_FILE = os.path.join(ROOT, "data", "indodoc", "test_new.jsonl")
TEST_CE_OUTPUT = os.path.join(ROOT, "data", "indodoc", "test_cross_encoder.jsonl")


def generate_ce_test_data():
    if not os.path.exists(TEST_MINILM_FILE):
        print(f"[!] File {TEST_MINILM_FILE} tidak ditemukan.")
        return

    with open(TEST_MINILM_FILE, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    ce_test_pairs = []

    for i, item in enumerate(test_data):
        query = item["title"]
        doc_positive = item["content"]

        # 1. Pasangan Positif (Label 1.0)
        ce_test_pairs.append({
            "query": query,
            "document": f"Isi: {doc_positive}",
            "label": 1.0
        })


        negatives = random.sample([d for j, d in enumerate(test_data) if j != i], 2)
        for neg in negatives:
            ce_test_pairs.append({
                "query": query,
                "document": f"Isi: {neg['content']}",
                "label": 0.0
            })


    random.shuffle(ce_test_pairs)

    with open(TEST_CE_OUTPUT, "w", encoding="utf-8") as f:
        for pair in ce_test_pairs:
            json.dump(pair, f, ensure_ascii=False)
            f.write("\n")

    print(f"[*] Berhasil membuat {len(ce_test_pairs)} pasangan data uji evaluasi Cross-Encoder.")
    print(f"[*] Rasio: {len(test_data)} Positif, {len(test_data) * 2} Negatif.")
    print(f"[-] Tersimpan di: {TEST_CE_OUTPUT}")


if __name__ == "__main__":
    generate_ce_test_data()