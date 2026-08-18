import os
import json
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers.cross_encoder import CrossEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CE_FILE = os.path.join(ROOT, "data", "indodoc", "test_cross_encoder.jsonl")

OUTPUT_DIR = os.path.join(ROOT, "output")

SEEDS = [42, 123, 456, 789, 1024]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def run_evaluation():
    if not os.path.exists(TEST_CE_FILE):
        print(f"[!] Data uji tidak ditemukan di {TEST_CE_FILE}. Jalankan generate_test_crossencoder.py dulu.")
        return

    queries_docs = []
    true_labels = []

    print(f"[*] Membaca data uji dari {TEST_CE_FILE} (Hanya dilakukan sekali)...")
    with open(TEST_CE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            queries_docs.append([data["query"], data["document"]])
            true_labels.append(data["label"])

    # Pastikan folder output ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[INFO] Memulai Evaluasi untuk {len(SEEDS)} model: {SEEDS}")

    # Looping untuk mengevaluasi setiap model berdasarkan SEED
    for seed in SEEDS:
        model_path = os.path.join(ROOT, "output", f"crossencoder-arsip-finetuned-seed-{seed}")

        if not os.path.exists(model_path):
            print(f"\n[!] Model tidak ditemukan di {model_path}. Melewati evaluasi Seed {seed}...")
            continue

        print(f"\n[*] Memuat model Reranker (Seed {seed}) dari:\n    {model_path}...")
        model = CrossEncoder(model_path)

        print(f"[*] Mengevaluasi {len(queries_docs)} pasangan dokumen secara prediktif (Seed {seed})...")

        # Model Cross-Encoder memprediksi skor
        raw_scores = model.predict(queries_docs, show_progress_bar=True)

        # Normalisasi skor menjadi probabilitas 0 hingga 1
        probabilities = sigmoid(raw_scores)

        # Jika probabilitas kemiripan >= 50% (0.5), dianggap relevan (1.0), jika tidak maka tidak relevan (0.0)
        predicted_labels = [1.0 if p >= 0.5 else 0.0 for p in probabilities]

        # Menampilkan Laporan di Terminal
        print("\n" + "=" * 55)
        print(f"📊 HASIL EVALUASI CROSS-ENCODER (SEED {seed})")
        print("=" * 55)

        acc = accuracy_score(true_labels, predicted_labels)
        print(f"Akurasi Total (Accuracy)  : {acc * 100:.2f}%\n")

        print("Detail Classification Report:")
        report_str = classification_report(true_labels, predicted_labels,
                                           target_names=["Negatif (Salah)", "Positif (Benar)"])
        print(report_str)

        # Ambil dictionary metrik untuk JSON
        report_dict = classification_report(true_labels, predicted_labels,
                                            target_names=["Negatif (Salah)", "Positif (Benar)"], output_dict=True)

        evaluation_results = {
            "model_path": model_path,
            "total_test_samples": len(queries_docs),
            "threshold": 0.5,
            "accuracy": acc,
            "classification_report": report_dict
        }

        # Simpan di folder /output/ dengan penamaan spesifik seed
        output_file_json = os.path.join(OUTPUT_DIR, f"test_evaluation_report_seed{seed}.json")
        output_file_txt = os.path.join(OUTPUT_DIR, f"test_evaluation_report_seed{seed}.txt")

        # 1. Simpan ke format JSON
        with open(output_file_json, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=4, ensure_ascii=False)

        # 2. Simpan ke format TXT (Bisa langsung di-copy paste ke laporan skripsi)
        with open(output_file_txt, "w", encoding="utf-8") as f:
            f.write("=" * 55 + "\n")
            f.write(f"📊 HASIL EVALUASI CROSS-ENCODER (SEED {seed})\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Akurasi Total (Accuracy)  : {acc * 100:.2f}%\n\n")
            f.write("Detail Classification Report:\n")
            f.write(report_str)

        print(f"[+] Laporan evaluasi Seed {seed} berhasil disimpan secara otomatis di:")
        print(f"    - {output_file_json}")
        print(f"    - {output_file_txt}")

    print("\n" + "=" * 55)
    print("✅ SEMUA EVALUASI SELESAI")
    print("=" * 55)


if __name__ == "__main__":
    run_evaluation()