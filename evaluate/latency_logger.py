import csv
import os
import time
import json


def log_latency_per_query(model_name: str, question: str, search_time: float, rerank_time: float,
                          llm_inference_time: float, total_time: float):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    folder_path = os.path.join(root_dir, "output")

    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, "latency_results_gemma.csv")
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Model", "Question",
                "Vector_Search_Sec", "Reranking_Sec", "LLM_Inference_Sec", "Total_Latency_Sec"
            ])

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            question,
            round(search_time, 4),
            round(rerank_time, 4),
            round(llm_inference_time, 4),
            round(total_time, 4)
        ])


def save_latency_summary(model_name: str, summary_data: dict):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    folder_path = os.path.join(root_dir, "output")

    os.makedirs(folder_path, exist_ok=True)
    # Nama file JSON disesuaikan dengan nama model
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    file_path = os.path.join(folder_path, f"latency_summary_{safe_model_name}.json")

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4)