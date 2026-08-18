import pandas as pd
import json
import os
import sys
import types
import warnings
import torch
from datasets import Dataset
from dotenv import dotenv_values


current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
env_path = os.path.join(project_root_dir, ".env")

env_vars = dotenv_values(env_path)

if "HF_TOKEN" in env_vars:
    os.environ["HF_TOKEN"] = env_vars["HF_TOKEN"]
    print("✅ HF_TOKEN berhasil dimuat dari file .env!")
else:
    print("❌ PERINGATAN: HF_TOKEN tidak ditemukan di dalam file .env!")

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Quick fix untuk Ragas 0.4.3 VertexAI bug
dummy_module = types.ModuleType("langchain_community.chat_models.vertexai")


class FakeChatVertexAI:
    pass


dummy_module.ChatVertexAI = FakeChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = dummy_module

from ragas.evaluation import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def run_ragas_evaluation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(current_dir)
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    csv_path = os.path.join(OUTPUT_DIR, "benchmark_results_gemma_v2.csv")
    report_path = os.path.join(OUTPUT_DIR, "ragas_report_gemma_v2.csv")


    EVALUATOR_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} tidak ditemukan.")
        return

    df = pd.read_csv(csv_path)
    contexts_list = [json.loads(c) if isinstance(c, str) else [] for c in df['Contexts']]

    # Filter Context yang kosong
    mask = [len(c) > 0 for c in contexts_list]
    data = {
        "question": [q for q, m in zip(df["Question"].tolist(), mask) if m],
        "answer": [a for a, m in zip(df["Answer"].tolist(), mask) if m],
        "ground_truth": [gt for gt, m in zip(df["Ground_Truth"].tolist(), mask) if m],
        "contexts": [c for c, m in zip(contexts_list, mask) if m]
    }
    dataset = Dataset.from_dict(data)

    total_original = len(df)
    total_filtered = len(dataset)
    print(
        f"[*] Data siap: {total_filtered} baris dievaluasi ({total_original - total_filtered} baris diabaikan karena Context kosong).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🛠️ Setup Model Evaluator (Menggunakan perangkat: {device.upper()})...")

    # Setup Judge LLM dengan paksaan format JSON (Diperlukan oleh RAGAS)
    llm_judge = ChatOllama(
        model="qwen2.5:7b",
        temperature=0,
        format="json",
        num_ctx=8192
    )
    ragas_llm = LangchainLLMWrapper(llm_judge)

    # PERBAIKAN 2: Load model juri embedding langsung tanpa try-except fallback
    # all-mpnet-base-v2 sangat direkomendasikan karena akurasinya tinggi dalam menghitung AnswerRelevancy
    print(f"🛠️ Memuat Embedding Juri: {EVALUATOR_EMBEDDING_MODEL} ...")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EVALUATOR_EMBEDDING_MODEL,
        model_kwargs={'device': device}
    )
    ragas_emb = LangchainEmbeddingsWrapper(hf_embeddings)

    # PERBAIKAN 3: Tetap pada 3 Metrik sesuai permintaan
    metrics_list = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision()
    ]

    # Konfigurasi RunConfig
    # max_workers=1 memastikan kestabilan VRAM, timeout=600 mencegah gagal koneksi ke Ollama
    run_config = RunConfig(timeout=600, max_workers=1, max_retries=3)

    print("📊 Menjalankan Evaluasi RAGAS (Mode antrean tunggal)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics_list,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_config,
        raise_exceptions=False
    )

    print("\n=== HASIL EVALUASI RAGAS ===")
    print(results)

    result_df = results.to_pandas()
    result_df.to_csv(report_path, index=False)
    print(f"💾 Hasil disimpan di '{report_path}'")


if __name__ == "__main__":
    run_ragas_evaluation()