import csv
import os
import time
import json  # Tambahkan ini untuk menangani format list ke CSV


def log_evaluation(model_name: str, question: str, answer: str, ground_truth: str, latency: float, sources_count: int,
                   similarity_score: float, contexts: list = None):  # Tambahkan parameter contexts (opsional)

    # Dapatkan direktori tempat script ini berada (src/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Naik satu level untuk mendapatkan direktori root
    root_dir = os.path.dirname(current_dir)
    # Tentukan path folder output di root
    folder_path = os.path.join(root_dir, "output")

    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, "benchmark_results_gemma_v2.csv")

    file_exists = os.path.isfile(file_path)

    # Gunakan list kosong jika contexts tidak diberikan (untuk kompatibilitas)
    if contexts is None:
        contexts = []

    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Menambahkan kolom "Contexts" di akhir
            writer.writerow(
                ["Timestamp", "Model", "Latency_Seconds", "Sources_Count", "Auto_Similarity_Score", "Question",
                 "Answer", "Ground_Truth", "Manual_Score", "Contexts"])

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            round(latency, 2),
            sources_count,
            round(similarity_score, 4),
            question,
            answer,
            ground_truth,
            "",
            json.dumps(contexts)
        ])